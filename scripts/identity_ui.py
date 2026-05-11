"""Wizards et helpers UI pour la gestion des identites + ecriture du rapport.

Ce module est un compagnon de scripts/test_app.py et heberge la nouvelle
surface d'usage : enregistrement multi-poses (20 prises), suppression d'une
identite et generation du rapport markdown d'evaluation. La charte
graphique (Theme) est partagee avec le tableau de bord -- elle est
importee depuis test_app pour garantir une coherence visuelle stricte.

Toutes les operations longues (encodage, ecriture CSV, leave-one-out)
sont dispatchees au thread worker via cmd_queue. La fenetre Tk principale
n'est jamais bloquee : les modales d'attente affichent un ttk.Progressbar
indetermine pendant le calcul.
"""
from __future__ import annotations

import re
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# La racine du projet est deja dans sys.path (positionnee par test_app au
# moment ou ce module est importe).
from src import config, dataset, evaluation, features, landmarks
from src import snake as snk

if TYPE_CHECKING:
    from test_app import TestApp


# ---------------------------------------------------------------------------
# Charte graphique partagee (test_app importe Theme depuis ce module)
# ---------------------------------------------------------------------------

class Theme:
    BG_BASE      = "#141414"   # cadre exterieur
    SURFACE_1    = "#1c1c1c"   # corps des panneaux (= door_sim BG)
    SURFACE_2    = "#242424"   # panneaux interieurs / lignes alternees
    SURFACE_3    = "#2e2e2e"   # survol / etat actif
    BORDER_SOFT  = "#3a3a3a"
    BORDER_HARD  = "#5a5a5a"

    TEXT_HI      = "#e8e8e8"
    TEXT_MID     = "#9a9a9a"
    TEXT_LOW     = "#6a6a6a"

    INFO         = "#5b9bd5"   # accent froid (groupe distances)
    INFO_SOFT    = "#3a6a9a"
    WARM         = "#c89858"   # accent chaud (groupe forme)
    WARM_SOFT    = "#7a5a30"

    SUCCESS      = "#33ff66"   # = door_sim LIGHT_GREEN
    SUCCESS_DIM  = "#1f7a3a"
    DANGER       = "#ff3333"   # = door_sim LIGHT_RED
    DANGER_DIM   = "#7a1f1f"
    NEUTRAL      = "#d0a040"   # ambre "en attente"

    @staticmethod
    def heading(size=14, weight="bold"):
        return ("Segoe UI Semibold", size, weight) if weight == "bold" \
               else ("Segoe UI", size)

    @staticmethod
    def body(size=10):
        return ("Segoe UI", size)

    @staticmethod
    def label(size=9):
        return ("Segoe UI", size)

    @staticmethod
    def mono(size=10):
        return ("Cascadia Mono", size)


def _bgr_to_imagetk(bgr, target_w, target_h, fit="contain"):
    """Convertit une image BGR OpenCV en ImageTk redimensionnee.
    fit='contain' garde l'aspect (lettre-boxe), fit='stretch' force la taille."""
    if bgr is None:
        return None, (target_w, target_h, 0, 0)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else \
          cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    pil = Image.fromarray(rgb)
    if fit == "contain":
        ratio = min(target_w / pil.width, target_h / pil.height)
        new_w = max(1, int(pil.width * ratio))
        new_h = max(1, int(pil.height * ratio))
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
        ox = (target_w - new_w) // 2
        oy = (target_h - new_h) // 2
        return ImageTk.PhotoImage(pil), (new_w, new_h, ox, oy)
    pil = pil.resize((target_w, target_h), Image.NEAREST)
    return ImageTk.PhotoImage(pil), (target_w, target_h, 0, 0)


# ---------------------------------------------------------------------------
# Commandes worker / evenements
# ---------------------------------------------------------------------------

CMD_ENROL_BATCH     = "enrol_batch"
CMD_DELETE_PERSON   = "delete_person"

EV_ENROL_BATCH_DONE = "enrol_batch_done"
EV_ENROL_BATCH_FAIL = "enrol_batch_fail"
EV_DELETE_DONE      = "delete_done"
EV_DELETE_FAIL      = "delete_fail"


# ---------------------------------------------------------------------------
# Plan des 20 prises et instructions par etape
# ---------------------------------------------------------------------------

POSE_GROUPS = [
    ("NEUTRE",      5),
    ("EXPRESSIONS", 5),
    ("ROTATIONS",   5),
    ("ECLAIRAGE",   5),
]
TOTAL_CAPTURES = sum(n for _, n in POSE_GROUPS)


POSE_INSTRUCTIONS = [
    "Regardez l'objectif, expression neutre, bouche fermee.",
    "Conservez la position. Stabilisez la respiration.",
    "Reculez tres legerement (10 cm) en gardant le regard fixe.",
    "Avancez tres legerement (10 cm) en gardant le regard fixe.",
    "Inclinez tres faiblement la tete vers le haut (~5 degres).",
    "Souriez naturellement, levres fermees.",
    "Souriez largement, dents visibles.",
    "Plissez les yeux comme face a une lumiere vive.",
    "Affichez une expression de surprise (sourcils releves).",
    "Pincez legerement les levres.",
    "Tournez la tete d'environ 15 degres a droite.",
    "Tournez la tete d'environ 15 degres a gauche.",
    "Inclinez la tete vers l'epaule droite (~10 degres).",
    "Inclinez la tete vers l'epaule gauche (~10 degres).",
    "Levez le menton (~10 degres) en gardant l'objectif des yeux.",
    "Eclairage frontal, ambiance neutre.",
    "Eclairage lateral droit (lampe a droite).",
    "Eclairage lateral gauche (lampe a gauche).",
    "Faible luminosite (eteignez une source).",
    "Forte luminosite (rapprochez une source).",
]


# ---------------------------------------------------------------------------
# Validation du nom
# ---------------------------------------------------------------------------

_INVALID_NAME_CHARS = re.compile(r"[;\\/]")


def validate_name(raw: str) -> Optional[str]:
    """Retourne un message d'erreur ou None si le nom est valide."""
    if raw != raw.strip():
        return "Le nom ne doit pas commencer ni finir par une espace."
    if not raw:
        return "Le nom ne peut pas etre vide."
    if "\n" in raw or "\r" in raw:
        return "Retours-chariot interdits."
    if _INVALID_NAME_CHARS.search(raw):
        return "Caracteres interdits : ; / \\."
    return None


# ---------------------------------------------------------------------------
# Style ttk additionnel (progressbar, scrollbar de la liste)
# ---------------------------------------------------------------------------

def install_extra_styles(style: ttk.Style) -> None:
    style.configure("VPO.Horizontal.TProgressbar",
                    background=Theme.INFO,
                    troughcolor=Theme.SURFACE_3,
                    bordercolor=Theme.BORDER_SOFT,
                    lightcolor=Theme.INFO,
                    darkcolor=Theme.INFO_SOFT,
                    thickness=8)
    style.configure("VPO.Vertical.TScrollbar",
                    background=Theme.SURFACE_2,
                    troughcolor=Theme.SURFACE_1,
                    bordercolor=Theme.BORDER_SOFT,
                    arrowcolor=Theme.TEXT_MID,
                    lightcolor=Theme.SURFACE_2,
                    darkcolor=Theme.SURFACE_2)
    style.map("VPO.Vertical.TScrollbar",
              background=[("active", Theme.SURFACE_3)])
    style.configure("VPOGhost.TButton",
                    background=Theme.SURFACE_1,
                    foreground=Theme.TEXT_MID,
                    bordercolor=Theme.BORDER_SOFT,
                    lightcolor=Theme.SURFACE_1,
                    darkcolor=Theme.SURFACE_1,
                    padding=(12, 6),
                    font=Theme.body(10))
    style.map("VPOGhost.TButton",
              background=[("active", Theme.SURFACE_2)],
              foreground=[("active", Theme.TEXT_HI)])


# ---------------------------------------------------------------------------
# LoadingPopup : modale bloquante avec barre indeterminee
# ---------------------------------------------------------------------------

class LoadingPopup(tk.Toplevel):
    """Modale non-fermable affichant un titre, un message et un Progressbar."""

    def __init__(self, parent, title: str, message: str):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=Theme.SURFACE_1)
        self.resizable(False, False)
        self.transient(parent)
        try:
            self.grab_set()
        except tk.TclError:
            pass
        self.protocol("WM_DELETE_WINDOW", lambda: None)

        install_extra_styles(ttk.Style(self))

        outer = tk.Frame(self, bg=Theme.SURFACE_1, highlightthickness=1,
                         highlightbackground=Theme.BORDER_HARD)
        outer.pack(fill="both", expand=True)

        # Bandeau de titre discret
        head = tk.Frame(outer, bg=Theme.SURFACE_1)
        head.pack(side="top", fill="x", padx=22, pady=(16, 0))
        tk.Label(head, text=title.upper(),
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_MID,
                 font=Theme.label(9)).pack(side="left")

        body = tk.Frame(outer, bg=Theme.SURFACE_1)
        body.pack(side="top", fill="both", expand=True, padx=22, pady=(8, 18))

        self._msg_var = tk.StringVar(value=message)
        tk.Label(body, textvariable=self._msg_var,
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_HI,
                 font=Theme.body(11), anchor="w",
                 justify="left", wraplength=420).pack(side="top", anchor="w")

        self._pb = ttk.Progressbar(body, mode="indeterminate", length=460,
                                   style="VPO.Horizontal.TProgressbar")
        self._pb.pack(side="top", fill="x", pady=(14, 4))
        self._pb.start(11)

        # Centrage relatif au parent
        self.update_idletasks()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        try:
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            x = px + (pw - w) // 2
            y = py + (ph - h) // 2
            self.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        except tk.TclError:
            pass

    def set_message(self, msg: str):
        if self.winfo_exists():
            self._msg_var.set(msg)

    def close(self):
        if not self.winfo_exists():
            return
        try:
            self._pb.stop()
        except Exception:
            pass
        try:
            self.grab_release()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# EnrolWizard : assistant de prise (20 captures)
# ---------------------------------------------------------------------------

class EnrolWizard(tk.Toplevel):
    """Modal guidant l'utilisateur a travers 20 prises (5 x 4 groupes)."""

    PREVIEW_W = 480
    PREVIEW_H = 360

    def __init__(self, app: "TestApp"):
        super().__init__(app.root)
        self.app = app
        self.title("Ajouter une identite")
        self.configure(bg=Theme.SURFACE_1)
        self.resizable(False, False)
        # Modale -- mais sans grab_set pour preserver l'usage du curseur
        # de seuil de la fenetre principale (cf. cahier des charges).
        self.transient(app.root)
        self.protocol("WM_DELETE_WINDOW", self._on_close_request)

        install_extra_styles(ttk.Style(self))

        # Etat
        self._name: Optional[str] = None
        self._mode_extend: bool = False              # "ajouter a une identite existante"
        self._dest_dir: Optional[Path] = None
        self._pre_existing_folder: bool = True       # le dossier existait avant l'assistant
        self._taken_files: list[Path] = []
        self._count: int = 0                         # nombre d'images deja prises (0..20)
        self._index_offset: int = 0                  # decalage de numerotation des fichiers
        self._encoding_popup: Optional[LoadingPopup] = None
        self._photo_ref = None
        self._poll_after_id: Optional[str] = None
        self._closed = False

        # Conteneur principal
        outer = tk.Frame(self, bg=Theme.SURFACE_1, highlightthickness=1,
                         highlightbackground=Theme.BORDER_HARD)
        outer.pack(fill="both", expand=True)

        self._phase_a = tk.Frame(outer, bg=Theme.SURFACE_1)
        self._phase_b = tk.Frame(outer, bg=Theme.SURFACE_1)

        self._build_phase_a()
        self._build_phase_b()

        self._phase_a.pack(fill="both", expand=True)

        # Centre la fenetre par rapport a la principale
        self.update_idletasks()
        try:
            px = app.root.winfo_rootx()
            py = app.root.winfo_rooty()
            pw = app.root.winfo_width()
            ph = app.root.winfo_height()
            w = self.winfo_reqwidth()
            h = self.winfo_reqheight()
            x = px + (pw - w) // 2
            y = py + (ph - h) // 2
            self.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        except tk.TclError:
            pass

        # Raccourcis
        self.bind("<Escape>", lambda e: self._on_close_request())
        self.bind("<space>", self._on_space)
        # Focus initial dans le champ nom
        self.after(50, lambda: self._name_entry.focus_set())

    # ---------------- Phase A : saisie du nom -------------------------
    def _build_phase_a(self):
        body = tk.Frame(self._phase_a, bg=Theme.SURFACE_1)
        body.pack(fill="both", expand=True, padx=28, pady=(22, 18))

        tk.Label(body, text="ENREGISTRER UNE IDENTITE",
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_MID,
                 font=Theme.label(9)).pack(side="top", anchor="w")
        tk.Label(body, text="Nouvelle identite",
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_HI,
                 font=Theme.heading(15)).pack(side="top", anchor="w",
                                              pady=(4, 2))
        tk.Label(body,
                 text=("Vingt prises seront capturees pour couvrir "
                       "neutre / expressions / rotations / eclairage."),
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_MID,
                 font=Theme.body(10),
                 wraplength=520, justify="left").pack(side="top",
                                                      anchor="w",
                                                      pady=(0, 14))

        # Saisie du nom
        tk.Label(body, text="NOM",
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_MID,
                 font=Theme.label(9)).pack(side="top", anchor="w")

        entry_wrap = tk.Frame(body, bg=Theme.SURFACE_2,
                              highlightbackground=Theme.BORDER_SOFT,
                              highlightthickness=1)
        entry_wrap.pack(side="top", fill="x", pady=(4, 4))
        self._name_entry = tk.Entry(entry_wrap,
                                    bg=Theme.SURFACE_2, fg=Theme.TEXT_HI,
                                    insertbackground=Theme.TEXT_HI,
                                    relief="flat",
                                    font=Theme.body(11))
        self._name_entry.pack(side="left", fill="x", expand=True,
                              padx=10, ipady=8)
        self._name_entry.bind("<KeyRelease>", lambda e: self._validate_phase_a())

        self._error_var = tk.StringVar(value="")
        tk.Label(body, textvariable=self._error_var,
                 bg=Theme.SURFACE_1, fg=Theme.DANGER,
                 font=Theme.label(9), anchor="w",
                 justify="left").pack(side="top", anchor="w", pady=(2, 8))

        # Mode : nouvelle identite vs ajout a existante
        self._mode_var = tk.StringVar(value="new")
        mode = tk.Frame(body, bg=Theme.SURFACE_1)
        mode.pack(side="top", fill="x", pady=(2, 2))

        self._mode_new = tk.Radiobutton(
            mode, text="Nouvelle identite",
            variable=self._mode_var, value="new",
            bg=Theme.SURFACE_1, fg=Theme.TEXT_HI,
            activebackground=Theme.SURFACE_1,
            activeforeground=Theme.TEXT_HI,
            selectcolor=Theme.SURFACE_2,
            highlightthickness=0,
            font=Theme.body(10),
            command=self._validate_phase_a)
        self._mode_new.pack(side="top", anchor="w")

        self._mode_extend_btn = tk.Radiobutton(
            mode, text="Ajouter d'autres prises a une identite existante",
            variable=self._mode_var, value="extend",
            bg=Theme.SURFACE_1, fg=Theme.TEXT_HI,
            activebackground=Theme.SURFACE_1,
            activeforeground=Theme.TEXT_HI,
            selectcolor=Theme.SURFACE_2,
            highlightthickness=0,
            font=Theme.body(10),
            state="disabled",
            command=self._validate_phase_a)
        self._mode_extend_btn.pack(side="top", anchor="w", pady=(2, 0))

        self._collision_lbl = tk.Label(body, text="",
                                       bg=Theme.SURFACE_1,
                                       fg=Theme.NEUTRAL,
                                       font=Theme.label(9),
                                       anchor="w", justify="left")
        self._collision_lbl.pack(side="top", anchor="w", pady=(8, 0))

        # Boutons
        btns = tk.Frame(body, bg=Theme.SURFACE_1)
        btns.pack(side="bottom", fill="x", pady=(20, 0))
        self._start_btn = ttk.Button(btns, text="Commencer la session",
                                     style="VPOPrimary.TButton",
                                     command=self._on_start_session,
                                     state="disabled")
        self._start_btn.pack(side="right")
        ttk.Button(btns, text="Annuler",
                   style="VPOGhost.TButton",
                   command=self._on_close_request).pack(side="right",
                                                        padx=(0, 8))

        self.bind("<Return>", lambda e: self._on_start_session())

    def _validate_phase_a(self):
        raw = self._name_entry.get()
        err = validate_name(raw)
        existing = self._existing_persons()
        if err is not None:
            self._error_var.set(err)
            self._collision_lbl.configure(text="")
            self._mode_extend_btn.configure(state="disabled")
            self._mode_var.set("new")
            self._start_btn.configure(state="disabled")
            return
        self._error_var.set("")
        if raw in existing:
            self._collision_lbl.configure(
                text=f"\"{raw}\" existe deja ({existing[raw]} vecteurs). "
                     "Selectionnez \"Ajouter d'autres prises\" pour completer.",
                fg=Theme.NEUTRAL)
            self._mode_extend_btn.configure(state="normal")
            valid = self._mode_var.get() == "extend"
        else:
            self._collision_lbl.configure(text="")
            self._mode_extend_btn.configure(state="disabled")
            self._mode_var.set("new")
            valid = True
        self._start_btn.configure(state="normal" if valid else "disabled")

    def _existing_persons(self) -> dict[str, int]:
        with self.app.dataset_lock:
            counts: dict[str, int] = {}
            for n in self.app.names:
                counts[n] = counts.get(n, 0) + 1
        return counts

    def _on_start_session(self):
        if str(self._start_btn["state"]) == "disabled":
            return
        name = self._name_entry.get().strip()
        if validate_name(name) is not None:
            return
        self._name = name
        self._mode_extend = (self._mode_var.get() == "extend")

        # Prepare le dossier captures/<name>
        dest = config.CAPTURES_DIR / name
        self._pre_existing_folder = dest.exists()
        dest.mkdir(parents=True, exist_ok=True)
        self._dest_dir = dest
        self._index_offset = len(list(dest.glob("*.jpg")))

        # Bascule sur la phase B
        self._phase_a.pack_forget()
        self._phase_b.pack(fill="both", expand=True)
        self._refresh_step_panel()
        self._start_preview_loop()

    # ---------------- Phase B : capture -------------------------------
    def _build_phase_b(self):
        head = tk.Frame(self._phase_b, bg=Theme.SURFACE_1)
        head.pack(side="top", fill="x", padx=24, pady=(20, 6))

        self._title_var = tk.StringVar(value="ENREGISTRER UNE IDENTITE")
        tk.Label(head, textvariable=self._title_var,
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_MID,
                 font=Theme.label(9)).pack(side="left")

        self._global_var = tk.StringVar(value="0 / 20")
        tk.Label(head, textvariable=self._global_var,
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_HI,
                 font=Theme.mono(11)).pack(side="right")

        # Strip de progression segmentee
        prog = tk.Frame(self._phase_b, bg=Theme.SURFACE_1)
        prog.pack(side="top", fill="x", padx=24, pady=(2, 14))
        self._progress_canvas = tk.Canvas(prog, height=44,
                                          bg=Theme.SURFACE_1,
                                          highlightthickness=0)
        self._progress_canvas.pack(side="top", fill="x", expand=True)

        # Corps : preview a gauche, instructions a droite
        body = tk.Frame(self._phase_b, bg=Theme.SURFACE_1)
        body.pack(side="top", fill="both", expand=True, padx=24, pady=(0, 8))

        # Preview
        prev = tk.Frame(body, bg=Theme.SURFACE_1)
        prev.pack(side="left", fill="y")

        self._preview_canvas = tk.Canvas(prev,
                                         width=self.PREVIEW_W,
                                         height=self.PREVIEW_H,
                                         bg=Theme.SURFACE_2,
                                         highlightthickness=1,
                                         highlightbackground=Theme.BORDER_SOFT)
        self._preview_canvas.pack(side="top")
        self._preview_img_id = self._preview_canvas.create_image(
            self.PREVIEW_W // 2, self.PREVIEW_H // 2, anchor="center")
        self._preview_placeholder_id = self._preview_canvas.create_text(
            self.PREVIEW_W // 2, self.PREVIEW_H // 2,
            text="Initialisation de la webcam...",
            fill=Theme.TEXT_LOW, font=Theme.body(11))

        # Banniere d'etat sous l'apercu
        st = tk.Frame(prev, bg=Theme.SURFACE_1)
        st.pack(side="top", fill="x", pady=(8, 0))
        self._face_var = tk.StringVar(value="visage : --")
        self._face_lbl = tk.Label(st, textvariable=self._face_var,
                                  bg=Theme.SURFACE_1, fg=Theme.TEXT_MID,
                                  font=Theme.mono(10))
        self._face_lbl.pack(side="left")

        # Instructions a droite
        info = tk.Frame(body, bg=Theme.SURFACE_1)
        info.pack(side="left", fill="both", expand=True, padx=(20, 0))

        self._pose_label_var = tk.StringVar(value="NEUTRE")
        tk.Label(info, textvariable=self._pose_label_var,
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_HI,
                 font=Theme.heading(20)).pack(side="top", anchor="w")

        self._instruction_var = tk.StringVar(value="")
        tk.Label(info, textvariable=self._instruction_var,
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_HI,
                 font=Theme.body(11),
                 wraplength=320, justify="left").pack(side="top", anchor="w",
                                                      pady=(8, 16))

        # Compteurs
        self._step_var = tk.StringVar(value="0 / 5 -- neutre")
        tk.Label(info, textvariable=self._step_var,
                 bg=Theme.SURFACE_1, fg=Theme.INFO,
                 font=Theme.mono(11)).pack(side="top", anchor="w")

        # Hint
        tk.Label(info,
                 text="Appuyez sur Espace ou Capturer une fois la pose tenue.",
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_LOW,
                 font=Theme.label(9), wraplength=320,
                 justify="left").pack(side="top", anchor="w", pady=(16, 0))

        # Boutons
        btns = tk.Frame(self._phase_b, bg=Theme.SURFACE_1)
        btns.pack(side="bottom", fill="x", padx=24, pady=(6, 18))
        self._cancel_btn = ttk.Button(btns, text="Annuler la session",
                                      style="VPODanger.TButton",
                                      command=self._on_cancel_session)
        self._cancel_btn.pack(side="left")

        right = tk.Frame(btns, bg=Theme.SURFACE_1)
        right.pack(side="right")
        self._undo_btn = ttk.Button(right, text="Reprendre la derniere prise",
                                    style="VPOGhost.TButton",
                                    command=self._on_undo,
                                    state="disabled")
        self._undo_btn.pack(side="left", padx=(0, 8))
        self._capture_btn = ttk.Button(right, text="Capturer  (Espace)",
                                       style="VPOPrimary.TButton",
                                       command=self._on_capture,
                                       state="disabled")
        self._capture_btn.pack(side="left")

    # ---------------- Compteurs / strip de progression ----------------
    def _current_group_idx(self) -> tuple[int, int, int]:
        """Renvoie (group_index, position dans groupe, taille du groupe)."""
        cum = 0
        for gi, (_, n) in enumerate(POSE_GROUPS):
            if self._count < cum + n:
                return gi, self._count - cum, n
            cum += n
        # Tous termines -> point sur le dernier
        gi = len(POSE_GROUPS) - 1
        return gi, POSE_GROUPS[gi][1], POSE_GROUPS[gi][1]

    def _refresh_step_panel(self):
        gi, pos, n = self._current_group_idx()
        group_name = POSE_GROUPS[gi][0]
        self._title_var.set("ENREGISTRER UNE IDENTITE  /  "
                            f"{self._name}".upper())
        self._pose_label_var.set(group_name)
        if self._count < TOTAL_CAPTURES:
            self._instruction_var.set(POSE_INSTRUCTIONS[self._count])
        else:
            self._instruction_var.set("Toutes les prises sont effectuees.")
        self._step_var.set(f"{pos} / {n} -- {group_name.lower()}")
        self._global_var.set(f"{self._count} / {TOTAL_CAPTURES}")
        self._undo_btn.configure(
            state=("normal" if self._taken_files else "disabled"))
        self._draw_progress()

    def _draw_progress(self):
        c = self._progress_canvas
        c.delete("all")
        c.update_idletasks()
        W = max(int(c.winfo_width()), 1)
        H = int(c["height"])
        n_groups = len(POSE_GROUPS)
        gw = W / n_groups

        cum = 0
        for gi, (label, n) in enumerate(POSE_GROUPS):
            gx0 = gi * gw
            gx1 = (gi + 1) * gw

            # Etat du groupe
            done = self._count >= cum + n
            active = (cum <= self._count < cum + n)
            label_color = (Theme.INFO if active else
                           Theme.TEXT_HI if done else Theme.TEXT_MID)
            label_font = (("Segoe UI Semibold", 9) if (active or done)
                          else Theme.label(9))
            c.create_text((gx0 + gx1) / 2, 12,
                          text=label, fill=label_color, font=label_font)

            # Pastilles
            dot_y = H - 14
            inset = 18
            if n > 1:
                spacing = (gx1 - gx0 - 2 * inset) / (n - 1)
            else:
                spacing = 0
            for di in range(n):
                cx = gx0 + inset + di * spacing
                global_idx = cum + di
                if global_idx < self._count:
                    fill = Theme.SUCCESS
                    outline = Theme.SUCCESS_DIM
                elif global_idx == self._count:
                    fill = Theme.WARM
                    outline = Theme.WARM_SOFT
                else:
                    fill = Theme.SURFACE_2
                    outline = Theme.BORDER_SOFT
                c.create_oval(cx - 5, dot_y - 5, cx + 5, dot_y + 5,
                              fill=fill, outline=outline, width=2)

            # Separateur
            if gi < n_groups - 1:
                c.create_line(gx1, 4, gx1, H - 4,
                              fill=Theme.BORDER_SOFT, dash=(2, 4))

            cum += n

    # ---------------- Boucle d'apercu (lit app.last_frame) ------------
    def _start_preview_loop(self):
        self._tick_preview()

    def _tick_preview(self):
        if self._closed or not self.winfo_exists():
            return
        fr = self.app.last_frame
        if fr is None:
            self._face_var.set("visage : --")
            self._face_lbl.configure(fg=Theme.TEXT_MID)
        else:
            photo, _ = _bgr_to_imagetk(fr.bgr,
                                       self.PREVIEW_W,
                                       self.PREVIEW_H,
                                       fit="contain")
            self._photo_ref = photo
            self._preview_canvas.itemconfigure(self._preview_img_id,
                                               image=photo)
            self._preview_canvas.delete(self._preview_placeholder_id)
            if fr.aligned is not None:
                self._face_var.set("visage : oui")
                self._face_lbl.configure(fg=Theme.SUCCESS)
            else:
                self._face_var.set("Visage non detecte")
                self._face_lbl.configure(fg=Theme.DANGER)

        # Bouton de capture : actif si visage aligne disponible et il
        # reste des prises a faire et aucun popup d'encodage en cours.
        can_capture = (
            fr is not None and fr.aligned is not None
            and self._count < TOTAL_CAPTURES
            and self._encoding_popup is None
        )
        self._capture_btn.configure(
            state="normal" if can_capture else "disabled")

        self._poll_after_id = self.after(33, self._tick_preview)

    def _on_space(self, _e=None):
        if self._phase_b.winfo_ismapped():
            self._on_capture()

    def _on_capture(self):
        if self._count >= TOTAL_CAPTURES:
            return
        fr = self.app.last_frame
        if fr is None or fr.aligned is None:
            return
        # Ecriture immediate (sans encodage)
        idx = self._index_offset + len(self._taken_files)
        path = self._dest_dir / f"img_{idx:03d}.jpg"
        if not cv2.imwrite(str(path), fr.aligned):
            self.app.show_toast("Echec de l'ecriture du JPEG.", level="error")
            return
        self._taken_files.append(path)
        self._count += 1

        # Feedback visuel : flash WARM sur le bouton (re-enable apres)
        self._capture_btn.configure(state="disabled")
        self.after(120, lambda: self._capture_btn.configure(state="normal")
                   if self.winfo_exists() else None)

        if self._count >= TOTAL_CAPTURES:
            self._begin_encoding()
        else:
            self._refresh_step_panel()

    def _on_undo(self):
        if not self._taken_files:
            return
        last = self._taken_files.pop()
        try:
            last.unlink(missing_ok=True)
        except Exception:
            pass
        self._count = max(0, self._count - 1)
        self._refresh_step_panel()

    def _on_cancel_session(self):
        # Supprime les fichiers crees par l'assistant et le dossier si on
        # l'avait cree nous-memes.
        for p in self._taken_files:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        self._taken_files = []
        if (self._dest_dir is not None
                and not self._pre_existing_folder
                and self._dest_dir.exists()):
            try:
                shutil.rmtree(self._dest_dir)
            except Exception:
                pass
        self._close_self()
        self.app.show_toast("Session annulee.", level="info")

    def _on_close_request(self):
        # Si phase A : juste fermer
        if not self._phase_b.winfo_ismapped():
            self._close_self()
            return
        # Si l'encodage est en cours, on ignore
        if self._encoding_popup is not None:
            return
        if not self._taken_files:
            self._close_self()
            return
        # Demande de confirmation
        ConfirmDialog(self,
                      title="Abandonner la session ?",
                      message=(f"Vous allez supprimer "
                               f"{len(self._taken_files)} image(s) "
                               "deja capturee(s)."),
                      danger_label="Abandonner",
                      cancel_label="Continuer la session",
                      on_confirm=self._on_cancel_session)

    # ---------------- Encodage des 20 vecteurs ------------------------
    def _begin_encoding(self):
        self._capture_btn.configure(state="disabled")
        self._undo_btn.configure(state="disabled")
        self._cancel_btn.configure(state="disabled")
        self._refresh_step_panel()

        self._encoding_popup = LoadingPopup(
            self, "Encodage", "Encodage des vecteurs en cours...")
        self.app.cmd_queue.put(
            (CMD_ENROL_BATCH,
             {"name": self._name,
              "paths": [str(p) for p in self._taken_files]})
        )

    def on_batch_enrolled(self, payload: dict):
        """Appele par le poll Tk principal lors de l'evenement worker."""
        if self._encoding_popup is not None:
            self._encoding_popup.close()
            self._encoding_popup = None

        name = payload.get("name", self._name or "?")
        kept = int(payload.get("kept", 0))
        rejected = int(payload.get("rejected", 0))

        if kept == 0:
            ConfirmDialog(
                self,
                title="Encodage echoue",
                message=("Aucun vecteur n'a pu etre encode "
                         f"({rejected} images rejetees). "
                         "Verifiez que le visage est visible "
                         "puis recommencez."),
                danger_label="Fermer",
                cancel_label=None,
                on_confirm=self._close_self)
            return

        # Encodage termine : ferme l'assistant et rend le focus a la
        # fenetre principale.
        if rejected == 0:
            self.app.show_toast(
                f"Identite {name} enregistree -- {kept} vecteurs encodes",
                level="ok")
        else:
            self.app.show_toast(
                f"{kept} / {TOTAL_CAPTURES} vecteurs encodes -- "
                f"{rejected} image(s) rejetee(s)",
                level="info")
        self._close_self()
        # Re-focus sur Identifier pour permettre I/Enter
        try:
            self.app.focus_identify()
        except Exception:
            pass

    def on_batch_failed(self, payload: dict):
        if self._encoding_popup is not None:
            self._encoding_popup.close()
            self._encoding_popup = None
        msg = payload.get("error", "Erreur inconnue durant l'encodage.")
        ConfirmDialog(self,
                      title="Echec de l'encodage",
                      message=msg,
                      danger_label="Fermer",
                      cancel_label=None,
                      on_confirm=self._close_self)

    # ---------------- Cleanup ----------------------------------------
    def _close_self(self):
        if self._closed:
            return
        self._closed = True
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except Exception:
                pass
        # Notifie l'app pour qu'elle libere la reference
        try:
            self.app.on_wizard_closed(self)
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# IdentityManager : modale liste + suppression
# ---------------------------------------------------------------------------

class IdentityManager(tk.Toplevel):
    WIN_W = 640
    WIN_H = 540
    ROW_H = 56

    def __init__(self, app: "TestApp"):
        super().__init__(app.root)
        self.app = app
        self.title("Gerer les identites")
        self.configure(bg=Theme.SURFACE_1)
        self.geometry(f"{self.WIN_W}x{self.WIN_H}")
        self.minsize(560, 420)
        self.transient(app.root)
        try:
            self.grab_set()
        except tk.TclError:
            pass
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        install_extra_styles(ttk.Style(self))

        self._delete_popup: Optional[LoadingPopup] = None
        self._closed = False
        self._pending_name: Optional[str] = None
        self._pending_files_deleted: bool = False
        self._row_widgets: list[dict] = []

        outer = tk.Frame(self, bg=Theme.SURFACE_1, highlightthickness=1,
                         highlightbackground=Theme.BORDER_HARD)
        outer.pack(fill="both", expand=True)

        head = tk.Frame(outer, bg=Theme.SURFACE_1)
        head.pack(side="top", fill="x", padx=24, pady=(20, 4))
        tk.Label(head, text="GERER LES IDENTITES",
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_MID,
                 font=Theme.label(9)).pack(side="left")
        self._summary_var = tk.StringVar(value="")
        tk.Label(head, textvariable=self._summary_var,
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_MID,
                 font=Theme.mono(10)).pack(side="right")

        tk.Label(outer,
                 text="Liste des personnes enregistrees dans dataset.csv",
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_HI,
                 font=Theme.body(11)).pack(side="top", anchor="w",
                                            padx=24, pady=(0, 12))

        # Cadre liste avec scrollbar
        list_wrap = tk.Frame(outer, bg=Theme.SURFACE_1)
        list_wrap.pack(side="top", fill="both", expand=True,
                       padx=24, pady=(0, 8))

        self._list_canvas = tk.Canvas(list_wrap, bg=Theme.SURFACE_1,
                                      highlightthickness=1,
                                      highlightbackground=Theme.BORDER_SOFT,
                                      bd=0)
        self._list_canvas.pack(side="left", fill="both", expand=True)

        self._scroll = ttk.Scrollbar(list_wrap, orient="vertical",
                                     command=self._list_canvas.yview,
                                     style="VPO.Vertical.TScrollbar")
        self._scroll.pack(side="right", fill="y")
        self._list_canvas.configure(yscrollcommand=self._scroll.set)

        self._list_inner = tk.Frame(self._list_canvas, bg=Theme.SURFACE_1)
        self._list_window = self._list_canvas.create_window(
            (0, 0), window=self._list_inner, anchor="nw")
        self._list_canvas.bind(
            "<Configure>",
            lambda e: self._list_canvas.itemconfigure(
                self._list_window, width=e.width))
        self._list_inner.bind(
            "<Configure>",
            lambda e: self._list_canvas.configure(
                scrollregion=self._list_canvas.bbox("all")))
        # Defilement molette
        self._list_canvas.bind_all(
            "<MouseWheel>", self._on_mousewheel, add="+")

        # Pied : bouton fermer
        foot = tk.Frame(outer, bg=Theme.SURFACE_1)
        foot.pack(side="bottom", fill="x", padx=24, pady=(8, 18))
        ttk.Button(foot, text="Fermer", style="VPO.TButton",
                   command=self._on_close).pack(side="right")

        self.bind("<Escape>", lambda e: self._on_close())
        self.refresh()

    def refresh(self):
        # Compte vecteurs / images
        with self.app.dataset_lock:
            counts: dict[str, int] = {}
            for n in self.app.names:
                counts[n] = counts.get(n, 0) + 1
        total_vecs = sum(counts.values())
        total_persons = len(counts)
        self._summary_var.set(
            f"{total_persons} personnes  |  {total_vecs} vecteurs en base")

        for w in self._list_inner.winfo_children():
            w.destroy()
        self._row_widgets.clear()

        if not counts:
            empty = tk.Frame(self._list_inner, bg=Theme.SURFACE_1)
            empty.pack(fill="x", pady=24)
            tk.Label(empty,
                     text="Aucune identite enregistree.",
                     bg=Theme.SURFACE_1, fg=Theme.TEXT_LOW,
                     font=Theme.body(11)).pack(anchor="center")
            return

        for i, name in enumerate(sorted(counts.keys(), key=str.casefold)):
            row = self._build_row(self._list_inner, name, counts[name], i)
            row.pack(side="top", fill="x", padx=1, pady=(0, 1))
            self._row_widgets.append({"frame": row, "name": name})

    def _build_row(self, parent, name: str, n_vecs: int, i: int) -> tk.Frame:
        bg_default = Theme.SURFACE_1 if i % 2 == 0 else Theme.SURFACE_1
        bg_hover = Theme.SURFACE_2

        row = tk.Frame(parent, bg=bg_default, height=self.ROW_H)
        row.pack_propagate(False)

        # Bande de gauche : couleur de selection (par defaut neutre)
        accent = tk.Frame(row, width=4, bg=Theme.SURFACE_1)
        accent.pack(side="left", fill="y")

        inner = tk.Frame(row, bg=bg_default)
        inner.pack(side="left", fill="both", expand=True, padx=(12, 12))

        head = tk.Frame(inner, bg=bg_default)
        head.pack(side="top", fill="x", pady=(8, 0))
        name_lbl = tk.Label(head, text=name,
                            bg=bg_default, fg=Theme.TEXT_HI,
                            font=Theme.body(12), anchor="w")
        name_lbl.pack(side="left")

        # Compte d'images sur disque
        on_disk = 0
        try:
            on_disk = len(list((config.CAPTURES_DIR / name).glob("*.jpg")))
        except Exception:
            on_disk = 0

        meta_lbl = tk.Label(
            inner,
            text=f"{n_vecs} vecteur{'s' if n_vecs > 1 else ''}  |  "
                 f"{on_disk} image{'s' if on_disk > 1 else ''} sur disque",
            bg=bg_default, fg=Theme.TEXT_MID,
            font=Theme.mono(10), anchor="w")
        meta_lbl.pack(side="top", anchor="w", pady=(2, 0))

        # Bouton Supprimer
        actions = tk.Frame(row, bg=bg_default)
        actions.pack(side="right", fill="y", padx=(0, 14))
        del_btn = ttk.Button(actions, text="Supprimer",
                             style="VPODanger.TButton",
                             command=lambda n=name, c=n_vecs, d=on_disk:
                                 self._on_delete(n, c, d))
        del_btn.pack(side="right", pady=14)

        # Survol
        def enter(_e=None, frames=(row, inner, head, actions),
                  labels=(name_lbl, meta_lbl), acc=accent):
            for f in frames:
                f.configure(bg=bg_hover)
            for l in labels:
                l.configure(bg=bg_hover)
            acc.configure(bg=Theme.INFO_SOFT)

        def leave(_e=None, frames=(row, inner, head, actions),
                  labels=(name_lbl, meta_lbl), acc=accent):
            for f in frames:
                f.configure(bg=bg_default)
            for l in labels:
                l.configure(bg=bg_default)
            acc.configure(bg=Theme.SURFACE_1)

        for w in (row, inner, head, actions, name_lbl, meta_lbl):
            w.bind("<Enter>", enter)
            w.bind("<Leave>", leave)

        return row

    def _on_mousewheel(self, e):
        try:
            self._list_canvas.yview_scroll(int(-e.delta / 120), "units")
        except Exception:
            pass

    # ---------------- Actions ----------------------------------------
    def _on_delete(self, name: str, n_vecs: int, n_imgs: int):
        DeleteConfirmDialog(self, name=name, n_vecs=n_vecs,
                            n_imgs=n_imgs,
                            on_confirm=lambda also_delete:
                                self._submit_delete(name, also_delete))

    def _submit_delete(self, name: str, also_delete: bool):
        self._pending_name = name
        self._pending_files_deleted = also_delete
        self._delete_popup = LoadingPopup(
            self, "Suppression",
            f"Mise a jour du dataset (\"{name}\") en cours...")
        self.app.cmd_queue.put(
            (CMD_DELETE_PERSON,
             {"name": name, "delete_files": also_delete})
        )

    def on_delete_done(self, payload: dict):
        if self._delete_popup is not None:
            self._delete_popup.close()
            self._delete_popup = None
        name = payload.get("name", self._pending_name) or "?"
        self.app.show_toast(
            f"Identite \"{name}\" supprimee.",
            level="ok")
        self._on_close()

    def on_delete_failed(self, payload: dict):
        if self._delete_popup is not None:
            self._delete_popup.close()
            self._delete_popup = None
        msg = payload.get("error", "Erreur inconnue durant la suppression.")
        self.app.show_toast(f"Suppression echouee : {msg}", level="error")
        self.refresh()

    def _on_close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._list_canvas.unbind_all("<MouseWheel>")
        except Exception:
            pass
        try:
            self.app.on_manager_closed(self)
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dialogues de confirmation
# ---------------------------------------------------------------------------

class ConfirmDialog(tk.Toplevel):
    """Dialogue oui/non sur fond sombre."""

    def __init__(self, parent, *, title: str, message: str,
                 danger_label: str, on_confirm,
                 cancel_label: Optional[str] = "Annuler"):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=Theme.SURFACE_1)
        self.resizable(False, False)
        self.transient(parent)
        try:
            self.grab_set()
        except tk.TclError:
            pass

        outer = tk.Frame(self, bg=Theme.SURFACE_1, highlightthickness=1,
                         highlightbackground=Theme.BORDER_HARD)
        outer.pack(fill="both", expand=True)
        body = tk.Frame(outer, bg=Theme.SURFACE_1)
        body.pack(fill="both", expand=True, padx=22, pady=20)

        tk.Label(body, text=title.upper(),
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_MID,
                 font=Theme.label(9)).pack(side="top", anchor="w")
        tk.Label(body, text=message,
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_HI,
                 font=Theme.body(11),
                 wraplength=420, justify="left").pack(side="top",
                                                      anchor="w", pady=(8, 16))

        btns = tk.Frame(body, bg=Theme.SURFACE_1)
        btns.pack(side="bottom", fill="x")

        def confirm():
            try:
                self.destroy()
            finally:
                on_confirm()

        if cancel_label is not None:
            ttk.Button(btns, text=cancel_label,
                       style="VPOGhost.TButton",
                       command=self.destroy).pack(side="right",
                                                  padx=(0, 8))
        ttk.Button(btns, text=danger_label,
                   style="VPODanger.TButton",
                   command=confirm).pack(side="right")

        # Centrage
        self.update_idletasks()
        try:
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            w = self.winfo_reqwidth()
            h = self.winfo_reqheight()
            self.geometry(
                f"+{max(px + (pw - w) // 2, 0)}+{max(py + (ph - h) // 2, 0)}")
        except tk.TclError:
            pass

        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<Return>", lambda e: confirm())


class DeleteConfirmDialog(tk.Toplevel):
    """Dialogue de confirmation specifique a la suppression d'identite."""

    def __init__(self, parent, *, name: str, n_vecs: int,
                 n_imgs: int, on_confirm):
        super().__init__(parent)
        self.title("Confirmer la suppression")
        self.configure(bg=Theme.SURFACE_1)
        self.resizable(False, False)
        self.transient(parent)
        try:
            self.grab_set()
        except tk.TclError:
            pass

        outer = tk.Frame(self, bg=Theme.SURFACE_1, highlightthickness=1,
                         highlightbackground=Theme.BORDER_HARD)
        outer.pack(fill="both", expand=True)
        body = tk.Frame(outer, bg=Theme.SURFACE_1)
        body.pack(fill="both", expand=True, padx=24, pady=22)

        tk.Label(body, text="CONFIRMER LA SUPPRESSION",
                 bg=Theme.SURFACE_1, fg=Theme.DANGER,
                 font=Theme.label(9)).pack(side="top", anchor="w")
        tk.Label(body, text=name,
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_HI,
                 font=Theme.heading(16)).pack(side="top", anchor="w",
                                              pady=(2, 12))

        details = tk.Frame(body, bg=Theme.SURFACE_2,
                           highlightbackground=Theme.BORDER_SOFT,
                           highlightthickness=1)
        details.pack(side="top", fill="x", pady=(0, 14))
        tk.Label(details,
                 text=f"  {n_vecs} vecteur{'s' if n_vecs > 1 else ''} "
                      "dans dataset.csv",
                 bg=Theme.SURFACE_2, fg=Theme.TEXT_HI,
                 font=Theme.body(10),
                 anchor="w").pack(side="top", fill="x",
                                  padx=10, pady=(8, 2))
        tk.Label(details,
                 text=f"  {n_imgs} image{'s' if n_imgs > 1 else ''} "
                      f"dans captures/{name}/",
                 bg=Theme.SURFACE_2, fg=Theme.TEXT_HI,
                 font=Theme.body(10),
                 anchor="w").pack(side="top", fill="x",
                                  padx=10, pady=(0, 8))

        # Checkbox suppression disque
        self._del_files = tk.BooleanVar(value=True)
        cb = tk.Checkbutton(body,
                            text="Supprimer aussi les images sur le disque",
                            variable=self._del_files,
                            bg=Theme.SURFACE_1, fg=Theme.TEXT_HI,
                            selectcolor=Theme.SURFACE_2,
                            activebackground=Theme.SURFACE_1,
                            activeforeground=Theme.TEXT_HI,
                            highlightthickness=0,
                            font=Theme.body(10))
        cb.pack(side="top", anchor="w", pady=(0, 8))

        tk.Label(body,
                 text="Cette action est irreversible.",
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_MID,
                 font=Theme.label(9)).pack(side="top", anchor="w",
                                            pady=(0, 14))

        btns = tk.Frame(body, bg=Theme.SURFACE_1)
        btns.pack(side="bottom", fill="x")

        def go():
            also = self._del_files.get()
            try:
                self.destroy()
            finally:
                on_confirm(also)

        ttk.Button(btns, text="Annuler",
                   style="VPOGhost.TButton",
                   command=self.destroy).pack(side="right", padx=(0, 8))
        ttk.Button(btns, text="Confirmer la suppression",
                   style="VPODanger.TButton",
                   command=go).pack(side="right")

        # Centrage
        self.update_idletasks()
        try:
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            w = self.winfo_reqwidth()
            h = self.winfo_reqheight()
            self.geometry(
                f"+{max(px + (pw - w) // 2, 0)}+{max(py + (ph - h) // 2, 0)}")
        except tk.TclError:
            pass

        self.bind("<Escape>", lambda e: self.destroy())


