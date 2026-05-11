"""Banc de test Tkinter pour le pipeline d'identification faciale.

Application en fenetre unique, organisee en mosaique "bento" : webcam
temps reel + panneaux d'inspection des etapes intermediaires (visage
aligne, contour actif + points caracteristiques, vecteur 30D, top 3
candidats, banniere de decision). Complement de scripts/main.py : main.py
reste la demo de la porte ; cette application sert a la mise au point et
a la soutenance orale (montrer pourquoi le systeme decide ce qu'il decide).

Architecture :
- Un thread worker capture la webcam, lance Haar + alignement sur chaque
  frame, et execute le snake + le vecteur 30D + l'identification
  uniquement sur demande (touche I). Il pousse les resultats dans des
  files. Aucune operation Tk hors du thread principal.
- Le thread Tk principal lit les files toutes les ~33 ms et met a jour
  les widgets.
- Le curseur de seuil recalcule la decision et la confiance a partir du
  dernier resultat memorise, sans rejouer le pipeline.
"""
from __future__ import annotations

import queue
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

from PIL import Image, ImageTk

from src import config, dataset, detection, evaluation, features, identify, landmarks
from src import snake as snk

# Theme + helpers UI partages avec les modales (wizard, gestionnaire,
# popup d'attente). Ce sous-module est l'unique source de verite pour
# la charte graphique, ce qui evite tout cycle a l'import quand
# test_app est lance directement (auquel cas Python le charge en tant que
# __main__, et un import "from test_app" depuis identity_ui echouerait).
from identity_ui import (
    Theme,
    _bgr_to_imagetk,
    EnrolWizard,
    IdentityManager,
    LoadingPopup,
    install_extra_styles,
    CMD_ENROL_BATCH,
    CMD_DELETE_PERSON,
    EV_ENROL_BATCH_DONE,
    EV_ENROL_BATCH_FAIL,
    EV_DELETE_DONE,
    EV_DELETE_FAIL,
)


# ---------------------------------------------------------------------------
# Donnees inter-threads
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    bgr: np.ndarray
    bbox: Optional[tuple]               # (x, y, w, h) en coords frame
    eyes: Optional[tuple]               # ((xl, yl), (xr, yr)) en coords frame
    aligned: Optional[np.ndarray]       # 128x128 gris egalise
    fps: float


@dataclass
class IdentifyResult:
    aligned: np.ndarray
    snake_points: np.ndarray            # (N, 2) en coords 128x128
    landmarks: dict                     # 8 points cles
    vector: np.ndarray                  # 30D
    candidates: list                    # [(name, dist), ...]
    threshold_at_run: float
    elapsed_ms: float
    auto_threshold: bool                # True si seuil issu d'un sweep


# Commandes UI -> worker
# CMD_ENROL_BATCH / CMD_DELETE_PERSON sont importes depuis identity_ui.
CMD_IDENTIFY = "identify"


# ---------------------------------------------------------------------------
# Pipeline allege (vit dans le worker, sans toucher a src/*)
# ---------------------------------------------------------------------------

def _align_with_eyes(gray):
    """Reproduit src.alignment.align_face mais expose aussi la bbox et les yeux."""
    bbox = detection.detect_face(gray)
    if bbox is None:
        return None, None, None
    x, y, w, h = bbox
    roi = gray[y:y + h, x:x + w]
    eye_pair = detection.detect_eyes(roi)
    eyes_full = None
    aligned_roi = roi
    if eye_pair is not None:
        (xg, yg), (xd, yd) = eye_pair
        eyes_full = ((xg + x, yg + y), (xd + x, yd + y))
        angle = float(np.degrees(np.arctan2(yd - yg, xd - xg)))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        aligned_roi = cv2.warpAffine(roi, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
    face = cv2.resize(aligned_roi, (config.IMG_SIZE, config.IMG_SIZE),
                      interpolation=cv2.INTER_AREA)
    aligned = cv2.equalizeHist(face)
    return bbox, eyes_full, aligned


# ---------------------------------------------------------------------------
# Helpers UI
# ---------------------------------------------------------------------------

def _make_panel(parent, title):
    """Cree un panneau (carte) avec en-tete petit-cap et zone de contenu."""
    outer = tk.Frame(parent, bg=Theme.SURFACE_1,
                     highlightbackground=Theme.BORDER_SOFT,
                     highlightthickness=1, bd=0)
    head = tk.Frame(outer, bg=Theme.SURFACE_1)
    head.pack(side="top", fill="x", padx=14, pady=(10, 6))
    tk.Label(head, text=title.upper(), bg=Theme.SURFACE_1,
             fg=Theme.TEXT_MID, font=Theme.label(9)).pack(side="left")
    tk.Frame(outer, height=1, bg=Theme.BORDER_SOFT).pack(side="top",
                                                        fill="x", padx=14)
    body = tk.Frame(outer, bg=Theme.SURFACE_1)
    body.pack(side="top", fill="both", expand=True, padx=14, pady=(8, 12))
    return outer, body


# ---------------------------------------------------------------------------
# Application principale
# ---------------------------------------------------------------------------

class TestApp:
    WIN_W = 1200
    WIN_H = 780

    # Tailles fixees des canevas internes
    WEBCAM_CV_W   = 540
    WEBCAM_CV_H   = 280
    FACE_CV_SIZE  = 256
    BAR_CV_W      = 540
    BAR_CV_H      = 234
    DOOR_ICON_SIZE = 60

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VPO -- Banc de test du modele")
        self.root.geometry(f"{self.WIN_W}x{self.WIN_H}")
        self.root.minsize(1024, 680)
        self.root.configure(bg=Theme.BG_BASE)

        # Style ttk
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self._init_ttk_style()

        # Etat partage
        self.names, self.vectors = dataset.load()
        self.threshold_var = tk.DoubleVar(value=config.DEFAULT_THRESHOLD)
        # Coche "Auto" : si vraie, chaque identification recalcule le
        # seuil optimal par balayage (LOO) avant de decider. Sinon le
        # systeme respecte la valeur courante du curseur.
        self.auto_threshold_var = tk.BooleanVar(value=False)
        self.last_identify: Optional[IdentifyResult] = None
        self.last_frame: Optional[FrameResult] = None
        self.fps_smoothed = 0.0

        self.frame_queue: "queue.Queue[FrameResult]" = queue.Queue(maxsize=2)
        self.event_queue: "queue.Queue[tuple]" = queue.Queue()
        self.cmd_queue:   "queue.Queue[tuple]" = queue.Queue()
        self.stop_event = threading.Event()

        # Verrou pour names/vectors (worker lit, UI ecrit apres enrol)
        self.dataset_lock = threading.Lock()

        # References d'images Tk (sinon GC)
        self._img_refs = {}

        # Modales actives (wizard d'enregistrement, gestionnaire)
        self._enrol_wizard: Optional[EnrolWizard] = None
        self._identity_manager: Optional[IdentityManager] = None

        # Construction UI
        self._build_ui()

        # Lancement du worker
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

        # Polling du thread principal
        self.root.after(33, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
        self._bind_keys()

    # ------------------------------------------------------------------ TTK
    def _init_ttk_style(self):
        s = self.style
        s.configure("VPO.TButton",
                    background=Theme.SURFACE_2,
                    foreground=Theme.TEXT_HI,
                    bordercolor=Theme.BORDER_HARD,
                    lightcolor=Theme.SURFACE_2,
                    darkcolor=Theme.SURFACE_2,
                    focusthickness=2,
                    focuscolor=Theme.INFO,
                    padding=(14, 8),
                    font=Theme.body(10))
        s.map("VPO.TButton",
              background=[("active", Theme.SURFACE_3),
                          ("pressed", Theme.BORDER_SOFT)],
              foreground=[("disabled", Theme.TEXT_LOW)])

        s.configure("VPOPrimary.TButton",
                    background=Theme.INFO_SOFT,
                    foreground=Theme.TEXT_HI,
                    bordercolor=Theme.INFO,
                    lightcolor=Theme.INFO_SOFT,
                    darkcolor=Theme.INFO_SOFT,
                    focusthickness=2,
                    focuscolor=Theme.INFO,
                    padding=(14, 8),
                    font=Theme.body(10))
        s.map("VPOPrimary.TButton",
              background=[("active", Theme.INFO),
                          ("pressed", Theme.INFO_SOFT)])

        s.configure("VPODanger.TButton",
                    background=Theme.SURFACE_2,
                    foreground=Theme.DANGER,
                    bordercolor=Theme.DANGER_DIM,
                    lightcolor=Theme.SURFACE_2,
                    darkcolor=Theme.SURFACE_2,
                    padding=(14, 8),
                    font=Theme.body(10))
        s.map("VPODanger.TButton",
              background=[("active", Theme.DANGER_DIM),
                          ("pressed", Theme.DANGER_DIM)],
              foreground=[("active", Theme.TEXT_HI)])

        s.configure("VPO.Horizontal.TScale",
                    background=Theme.SURFACE_1,
                    troughcolor=Theme.SURFACE_3,
                    bordercolor=Theme.BORDER_SOFT,
                    lightcolor=Theme.INFO,
                    darkcolor=Theme.INFO_SOFT)

        s.configure("VPO.TCheckbutton",
                    background=Theme.SURFACE_1,
                    foreground=Theme.TEXT_HI,
                    focuscolor=Theme.INFO,
                    font=Theme.label(9))
        s.map("VPO.TCheckbutton",
              background=[("active", Theme.SURFACE_1)],
              foreground=[("disabled", Theme.TEXT_LOW)])

        # Styles additionnels (LoadingPopup, scrollbar du gestionnaire,
        # boutons fantomes des dialogues).
        install_extra_styles(s)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        outer = tk.Frame(self.root, bg=Theme.BG_BASE)
        outer.pack(fill="both", expand=True, padx=14, pady=14)

        # En-tete
        header = tk.Frame(outer, bg=Theme.BG_BASE)
        header.pack(side="top", fill="x")

        title_frame = tk.Frame(header, bg=Theme.BG_BASE)
        title_frame.pack(side="left")
        tk.Label(title_frame, text="IDENTIFICATION VPO",
                 bg=Theme.BG_BASE, fg=Theme.TEXT_HI,
                 font=Theme.heading(15)).pack(side="top", anchor="w")
        tk.Label(title_frame, text="Banc de test du pipeline geometrique",
                 bg=Theme.BG_BASE, fg=Theme.TEXT_MID,
                 font=Theme.label(9)).pack(side="top", anchor="w")

        self.dataset_label = tk.Label(
            header,
            text=self._dataset_text(),
            bg=Theme.BG_BASE, fg=Theme.TEXT_MID,
            font=Theme.mono(10))
        self.dataset_label.pack(side="right", anchor="e")

        # Barre de controles : construite AVANT la zone bento pour que
        # son espace en bas du conteneur soit reserve par pack avant
        # la zone bento (qui a expand=True). Sans cela, sur une fenetre
        # maximisee le bento absorbait toute la hauteur et la barre
        # se retrouvait en dehors de la zone visible.
        self._build_controls(outer)

        # Zone bento : 3 colonnes x 2 lignes
        bento = tk.Frame(outer, bg=Theme.BG_BASE)
        bento.pack(side="top", fill="both", expand=True, pady=(14, 12))

        for c, w in enumerate((4, 2, 2)):
            bento.grid_columnconfigure(c, weight=w, uniform="bento_col")
        for r in (0, 1):
            bento.grid_rowconfigure(r, weight=1, uniform="bento_row")

        # --- Ligne 1 -----------------------------------------------------
        self._build_webcam_panel(bento)
        self._build_aligned_panel(bento)
        self._build_overlay_panel(bento)
        # --- Ligne 2 -----------------------------------------------------
        self._build_chart_panel(bento)
        self._build_top3_panel(bento)
        self._build_decision_panel(bento)

    def _dataset_text(self):
        with self.dataset_lock:
            n = len(self.vectors)
            persons = len(set(self.names))
        return f"{persons} personnes  |  {n} vecteurs en base"

    # ---- Panneaux ----------------------------------------------------
    def _build_webcam_panel(self, parent):
        outer, body = _make_panel(parent, "Webcam")
        outer.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))

        self.webcam_canvas = tk.Canvas(body,
                                       width=self.WEBCAM_CV_W,
                                       height=self.WEBCAM_CV_H,
                                       bg=Theme.SURFACE_2,
                                       highlightthickness=0)
        self.webcam_canvas.pack(side="top")
        self._webcam_img_id = self.webcam_canvas.create_image(
            self.WEBCAM_CV_W // 2, self.WEBCAM_CV_H // 2, anchor="center")
        self.webcam_canvas.create_text(
            self.WEBCAM_CV_W // 2, self.WEBCAM_CV_H // 2,
            text="Initialisation de la webcam...",
            fill=Theme.TEXT_LOW, font=Theme.body(11),
            tags=("webcam_placeholder",))

        caption = tk.Frame(body, bg=Theme.SURFACE_1)
        caption.pack(side="top", fill="x", pady=(8, 0))
        self.webcam_fps_lbl = tk.Label(caption, text="FPS: --",
                                       bg=Theme.SURFACE_1, fg=Theme.TEXT_MID,
                                       font=Theme.mono(10))
        self.webcam_fps_lbl.pack(side="left")
        tk.Frame(caption, bg=Theme.SURFACE_1).pack(side="left", expand=True,
                                                  fill="x")
        self.webcam_state_lbl = tk.Label(caption, text="visage: --",
                                         bg=Theme.SURFACE_1, fg=Theme.TEXT_MID,
                                         font=Theme.mono(10))
        self.webcam_state_lbl.pack(side="right")

    def _build_aligned_panel(self, parent):
        outer, body = _make_panel(parent, "Visage aligne (128 -> 256)")
        outer.grid(row=0, column=1, sticky="nsew", padx=8, pady=(0, 8))

        wrap = tk.Frame(body, bg=Theme.SURFACE_1)
        wrap.pack(side="top", expand=True)
        self.aligned_canvas = tk.Canvas(wrap,
                                        width=self.FACE_CV_SIZE,
                                        height=self.FACE_CV_SIZE,
                                        bg=Theme.SURFACE_2,
                                        highlightthickness=1,
                                        highlightbackground=Theme.BORDER_SOFT)
        self.aligned_canvas.pack()
        self._aligned_img_id = self.aligned_canvas.create_image(
            self.FACE_CV_SIZE // 2, self.FACE_CV_SIZE // 2, anchor="center")
        self.aligned_canvas.create_text(
            self.FACE_CV_SIZE // 2, self.FACE_CV_SIZE // 2,
            text="aucun visage",
            fill=Theme.TEXT_LOW, font=Theme.body(10),
            tags=("aligned_placeholder",))

        cap = tk.Frame(body, bg=Theme.SURFACE_1)
        cap.pack(side="top", fill="x", pady=(8, 0))
        tk.Label(cap, text="Haar -> rotation yeux -> 128x128 + egalisation",
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_LOW,
                 font=Theme.label(9)).pack(side="left")

    def _build_overlay_panel(self, parent):
        outer, body = _make_panel(parent, "Contour + points caracteristiques")
        outer.grid(row=0, column=2, sticky="nsew", padx=(8, 0), pady=(0, 8))

        wrap = tk.Frame(body, bg=Theme.SURFACE_1)
        wrap.pack(side="top", expand=True)
        self.overlay_canvas = tk.Canvas(wrap,
                                        width=self.FACE_CV_SIZE,
                                        height=self.FACE_CV_SIZE,
                                        bg=Theme.SURFACE_2,
                                        highlightthickness=1,
                                        highlightbackground=Theme.BORDER_SOFT)
        self.overlay_canvas.pack()
        self._overlay_img_id = self.overlay_canvas.create_image(
            self.FACE_CV_SIZE // 2, self.FACE_CV_SIZE // 2, anchor="center")
        self.overlay_canvas.create_text(
            self.FACE_CV_SIZE // 2, self.FACE_CV_SIZE // 2,
            text="appuyez sur Identifier",
            fill=Theme.TEXT_LOW, font=Theme.body(10),
            tags=("overlay_placeholder",))

        cap = tk.Frame(body, bg=Theme.SURFACE_1)
        cap.pack(side="top", fill="x", pady=(8, 0))
        tk.Label(cap,
                 text=f"snake : {config.SNAKE_ITERATIONS} iter, "
                      f"a={config.SNAKE_ALPHA}, b={config.SNAKE_BETA}, "
                      f"k={config.SNAKE_KAPPA}",
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_LOW,
                 font=Theme.mono(9)).pack(side="left")

    def _build_chart_panel(self, parent):
        outer, body = _make_panel(parent, "Vecteur caracteristique 30D")
        outer.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(8, 0))

        self.chart_canvas = tk.Canvas(body,
                                      width=self.BAR_CV_W,
                                      height=self.BAR_CV_H,
                                      bg=Theme.SURFACE_2,
                                      highlightthickness=0)
        self.chart_canvas.pack(side="top")

        cap = tk.Frame(body, bg=Theme.SURFACE_1)
        cap.pack(side="top", fill="x", pady=(8, 0))
        # Petite legende avec puces de couleur
        leg_l = tk.Frame(cap, bg=Theme.SURFACE_1)
        leg_l.pack(side="left")
        tk.Frame(leg_l, width=10, height=10, bg=Theme.INFO).pack(side="left",
                                                                 padx=(0, 6))
        tk.Label(leg_l, text="14 distances inter-points (norm. diagonale)",
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_MID,
                 font=Theme.label(9)).pack(side="left")

        leg_r = tk.Frame(cap, bg=Theme.SURFACE_1)
        leg_r.pack(side="right")
        tk.Label(leg_r, text="16 rayons de forme (snake / rayon moyen)",
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_MID,
                 font=Theme.label(9)).pack(side="right", padx=(6, 0))
        tk.Frame(leg_r, width=10, height=10, bg=Theme.WARM).pack(side="right")

        self._draw_chart(None)

    def _build_top3_panel(self, parent):
        outer, body = _make_panel(parent, "Top 3 candidats")
        outer.grid(row=1, column=1, sticky="nsew", padx=8, pady=(8, 0))

        self.top3_rows = []
        for i in range(3):
            row = tk.Frame(body, bg=Theme.SURFACE_1)
            row.pack(side="top", fill="x", pady=(0 if i == 0 else 6, 0))
            # Bande accent gauche
            accent = tk.Frame(row, width=4, bg=Theme.SURFACE_1)
            accent.pack(side="left", fill="y")
            inner = tk.Frame(row, bg=Theme.SURFACE_1)
            inner.pack(side="left", fill="x", expand=True, padx=(8, 0))

            head = tk.Frame(inner, bg=Theme.SURFACE_1)
            head.pack(side="top", fill="x")
            rank = tk.Label(head, text=f"{i + 1}.",
                            bg=Theme.SURFACE_1, fg=Theme.TEXT_LOW,
                            font=Theme.mono(11))
            rank.pack(side="left")
            name = tk.Label(head, text="--",
                            bg=Theme.SURFACE_1, fg=Theme.TEXT_HI,
                            font=Theme.body(11), anchor="w")
            name.pack(side="left", padx=(8, 0), fill="x", expand=True)
            dist = tk.Label(head, text="d=--",
                            bg=Theme.SURFACE_1, fg=Theme.TEXT_MID,
                            font=Theme.mono(10))
            dist.pack(side="right")

            bar_wrap = tk.Frame(inner, bg=Theme.SURFACE_1)
            bar_wrap.pack(side="top", fill="x", pady=(4, 0))
            bar_canvas = tk.Canvas(bar_wrap, height=8, bg=Theme.SURFACE_2,
                                   highlightthickness=0)
            bar_canvas.pack(side="left", fill="x", expand=True)
            conf = tk.Label(inner, text="confiance: --",
                            bg=Theme.SURFACE_1, fg=Theme.TEXT_LOW,
                            font=Theme.mono(9))
            conf.pack(side="top", anchor="e", pady=(2, 0))

            self.top3_rows.append({
                "row": row, "accent": accent, "name": name,
                "dist": dist, "bar": bar_canvas, "conf": conf,
            })

        # Etat vide
        self._render_top3([], 0.0, accepted=False)

    def _build_decision_panel(self, parent):
        outer, body = _make_panel(parent, "Decision")
        outer.grid(row=1, column=2, sticky="nsew", padx=(8, 0), pady=(8, 0))

        wrap = tk.Frame(body, bg=Theme.SURFACE_1)
        wrap.pack(side="top", fill="both", expand=True)

        # Banniere
        self.banner_lbl = tk.Label(wrap, text="EN ATTENTE",
                                   bg=Theme.SURFACE_1,
                                   fg=Theme.NEUTRAL,
                                   font=Theme.heading(18))
        self.banner_lbl.pack(side="top", pady=(6, 0))

        self.banner_sub_lbl = tk.Label(wrap, text="Lancez une identification.",
                                       bg=Theme.SURFACE_1,
                                       fg=Theme.TEXT_MID,
                                       font=Theme.body(10))
        self.banner_sub_lbl.pack(side="top", pady=(2, 8))

        # Bloc valeurs
        values = tk.Frame(wrap, bg=Theme.SURFACE_2,
                          highlightbackground=Theme.BORDER_SOFT,
                          highlightthickness=1)
        values.pack(side="top", fill="x", padx=8, pady=(0, 8))

        def _kv(label, key):
            r = tk.Frame(values, bg=Theme.SURFACE_2)
            r.pack(side="top", fill="x", padx=10, pady=2)
            tk.Label(r, text=label, bg=Theme.SURFACE_2,
                     fg=Theme.TEXT_LOW, font=Theme.label(9))\
                .pack(side="left")
            v = tk.Label(r, text="--", bg=Theme.SURFACE_2,
                         fg=Theme.TEXT_HI, font=Theme.mono(10))
            v.pack(side="right")
            return v

        self.dist_val_lbl   = _kv("distance", "dist")
        self.thr_val_lbl    = _kv("seuil",    "thr")
        self.conf_val_lbl   = _kv("confiance", "conf")

        # Icone porte
        door_wrap = tk.Frame(wrap, bg=Theme.SURFACE_1)
        door_wrap.pack(side="top", pady=(4, 6))
        self.door_canvas = tk.Canvas(door_wrap,
                                     width=110, height=110,
                                     bg=Theme.SURFACE_1,
                                     highlightthickness=0)
        self.door_canvas.pack()
        self._draw_door("wait")

    # ---- Barre de controles -----------------------------------------
    def _build_controls(self, parent):
        bar = tk.Frame(parent, bg=Theme.SURFACE_1,
                       highlightbackground=Theme.BORDER_SOFT,
                       highlightthickness=1)
        # Ancrage en bas du conteneur exterieur : la barre conserve sa
        # taille naturelle et n'est jamais rognee, meme quand la zone
        # bento (expand=True) absorbe l'espace vertical d'une fenetre
        # maximisee.
        bar.pack(side="bottom", fill="x", pady=(8, 0))

        inner = tk.Frame(bar, bg=Theme.SURFACE_1)
        inner.pack(side="top", fill="x", padx=14, pady=12)

        # Boutons gauche
        left = tk.Frame(inner, bg=Theme.SURFACE_1)
        left.pack(side="left")
        self.identify_btn = ttk.Button(left, text="Identifier  (I)",
                                       style="VPOPrimary.TButton",
                                       command=self._on_identify)
        self.identify_btn.pack(side="left", padx=(0, 8))
        ttk.Button(left, text="Capturer  (E)", style="VPO.TButton",
                   command=self._on_open_enrol_wizard).pack(side="left",
                                                            padx=(0, 8))
        ttk.Button(left, text="Gerer les identites", style="VPO.TButton",
                   command=self._on_open_manager).pack(side="left")

        # Quitter (droite)
        ttk.Button(inner, text="Quitter  (Q)", style="VPODanger.TButton",
                   command=self._on_quit).pack(side="right")

        # Curseur seuil au centre
        thr = tk.Frame(inner, bg=Theme.SURFACE_1)
        thr.pack(side="left", expand=True, fill="x", padx=24)

        head = tk.Frame(thr, bg=Theme.SURFACE_1)
        head.pack(side="top", fill="x")
        tk.Label(head, text="SEUIL DE DECISION",
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_MID,
                 font=Theme.label(9)).pack(side="left")
        self.thr_readout = tk.Label(head,
                                    text=f"{self.threshold_var.get():.3f}",
                                    bg=Theme.SURFACE_1, fg=Theme.TEXT_HI,
                                    font=Theme.mono(10))
        self.thr_readout.pack(side="right")
        self.auto_thr_chk = ttk.Checkbutton(
            head, text="Auto (sweep LOO)",
            variable=self.auto_threshold_var,
            style="VPO.TCheckbutton",
            command=self._on_auto_toggle)
        self.auto_thr_chk.pack(side="right", padx=(0, 12))

        scale = ttk.Scale(thr, from_=0.05, to=0.50,
                          orient="horizontal",
                          variable=self.threshold_var,
                          style="VPO.Horizontal.TScale",
                          command=self._on_threshold_change)
        scale.pack(side="top", fill="x", pady=(4, 0))

        # Repere visuel min/max
        rng = tk.Frame(thr, bg=Theme.SURFACE_1)
        rng.pack(side="top", fill="x")
        tk.Label(rng, text="0.05 strict",
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_LOW,
                 font=Theme.label(8)).pack(side="left")
        tk.Label(rng, text="0.50 permissif",
                 bg=Theme.SURFACE_1, fg=Theme.TEXT_LOW,
                 font=Theme.label(8)).pack(side="right")

    # ---- Liaisons clavier --------------------------------------------
    def _bind_keys(self):
        self.root.bind("<KeyPress-i>", lambda e: self._on_identify())
        self.root.bind("<KeyPress-I>", lambda e: self._on_identify())
        self.root.bind("<KeyPress-e>", lambda e: self._on_open_enrol_wizard())
        self.root.bind("<KeyPress-E>", lambda e: self._on_open_enrol_wizard())
        self.root.bind("<KeyPress-q>", lambda e: self._on_quit())
        self.root.bind("<KeyPress-Q>", lambda e: self._on_quit())
        # Entree relance l'identification quand le focus est sur la fenetre
        # principale (utilise apres la fermeture de l'assistant pour
        # permettre "I" ou Entree)
        self.root.bind("<Return>", self._on_root_return)

    def _on_root_return(self, event):
        # Ne consomme la touche que si le focus n'est pas dans un Entry
        try:
            w = event.widget
            if isinstance(w, tk.Entry) or isinstance(w, ttk.Entry):
                return
            if str(w.winfo_toplevel()) != str(self.root):
                # Une modale a le focus -- laisser passer
                return
        except Exception:
            pass
        self._on_identify()

    # ---- Worker ------------------------------------------------------
    def _worker_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.event_queue.put(("error", "Webcam indisponible."))
            return

        last_t = time.time()
        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                bbox, eyes, aligned = _align_with_eyes(gray)

                now = time.time()
                dt = now - last_t
                last_t = now
                fps = 1.0 / max(dt, 1e-3)

                fr = FrameResult(bgr=frame, bbox=bbox, eyes=eyes,
                                 aligned=aligned, fps=fps)
                # Remplace la frame en attente plutot que d'attendre
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self.frame_queue.put_nowait(fr)
                except queue.Full:
                    pass

                # Drain commandes
                while True:
                    try:
                        action, payload = self.cmd_queue.get_nowait()
                    except queue.Empty:
                        break
                    if action == CMD_IDENTIFY:
                        self._do_identify(aligned, payload)
                    elif action == CMD_ENROL_BATCH:
                        self._do_enrol_batch(payload)
                    elif action == CMD_DELETE_PERSON:
                        self._do_delete_person(payload)
        finally:
            cap.release()

    def _do_identify(self, aligned, payload):
        if aligned is None:
            self.event_queue.put(("toast", "Aucun visage detecte."))
            return
        payload = payload or {}
        auto_mode = bool(payload.get("auto", False))
        manual_thr = float(payload.get("manual_threshold",
                                       config.DEFAULT_THRESHOLD))
        # Snapshot du dataset (sous verrou) -- evite toute race avec
        # un add/delete concurrent.
        with self.dataset_lock:
            names_snapshot = list(self.names)
            vectors_snapshot = (self.vectors.copy()
                                if len(self.vectors) else self.vectors)
        # Mode Auto : balayage de seuil (LOO) pour caler le seuil
        # optimal sur l'etat courant. Si la base est trop petite (LOO
        # degenere) ou si Auto est desactive, on respecte le seuil du
        # curseur.
        used_thr = manual_thr
        used_auto = False
        if auto_mode and (len(vectors_snapshot) >= 4
                          and len(set(names_snapshot)) >= 2):
            try:
                sweep = evaluation.threshold_sweep(
                    names_snapshot, vectors_snapshot,
                    np.linspace(0.05, 0.50, 19))
                best = max(sweep, key=lambda r: r["accuracy"])
                used_thr = float(best["threshold"])
                used_auto = True
            except Exception:
                pass
        # Pipeline d'extraction
        t0 = time.time()
        contour = snk.fit_snake(aligned)
        lm = landmarks.detect_landmarks(aligned, snake=contour)
        vec = features.build_feature_vector(lm, snake=contour)
        cands = identify.search(vec, names_snapshot, vectors_snapshot)
        elapsed_ms = (time.time() - t0) * 1000.0
        res = IdentifyResult(aligned=aligned, snake_points=contour,
                             landmarks=lm, vector=vec,
                             candidates=cands,
                             threshold_at_run=used_thr,
                             elapsed_ms=elapsed_ms,
                             auto_threshold=used_auto)
        self.event_queue.put(("identify", res))

    def _do_enrol_batch(self, payload):
        """Encode toutes les images capturees par l'assistant pour <name>.

        Lit chaque JPEG aligne 128x128, lance snake -> landmarks ->
        features. Accumule dans une liste temporaire puis met a jour
        dataset.csv et l'etat memoire sous un unique critical-section.
        """
        name = payload["name"]
        paths = [Path(p) for p in payload["paths"]]
        kept_vectors = []
        rejected = 0
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None or img.shape != (config.IMG_SIZE, config.IMG_SIZE):
                rejected += 1
                continue
            try:
                contour = snk.fit_snake(img)
                lm = landmarks.detect_landmarks(img, snake=contour)
                vec = features.build_feature_vector(lm, snake=contour)
            except Exception:
                rejected += 1
                continue
            kept_vectors.append(vec)

        if not kept_vectors:
            self.event_queue.put((EV_ENROL_BATCH_DONE,
                                  {"name": name, "kept": 0,
                                   "rejected": rejected}))
            return

        try:
            with self.dataset_lock:
                for v in kept_vectors:
                    dataset.append(name, v)
                stack = np.asarray(kept_vectors, dtype=np.float32)
                self.names.extend([name] * len(kept_vectors))
                self.vectors = (np.vstack([self.vectors, stack])
                                if len(self.vectors) else stack)
        except Exception as exc:
            self.event_queue.put(
                (EV_ENROL_BATCH_FAIL,
                 {"error": f"{type(exc).__name__}: {exc}"}))
            return

        self.event_queue.put(
            (EV_ENROL_BATCH_DONE,
             {"name": name, "kept": len(kept_vectors),
              "rejected": rejected}))

    def _do_delete_person(self, payload):
        """Supprime toutes les lignes <name> de dataset.csv et,
        optionnellement, le dossier captures/<name>/."""
        name = payload["name"]
        delete_files = bool(payload.get("delete_files", True))
        try:
            with self.dataset_lock:
                mask = np.array([n != name for n in self.names], dtype=bool)
                new_names = [n for n, keep in zip(self.names, mask) if keep]
                if len(self.vectors) > 0:
                    new_vectors = self.vectors[mask]
                else:
                    new_vectors = self.vectors
                dataset.write_all(new_names, new_vectors)
                self.names = new_names
                self.vectors = (new_vectors
                                if isinstance(new_vectors, np.ndarray)
                                else np.asarray(new_vectors,
                                                dtype=np.float32))
        except Exception as exc:
            self.event_queue.put(
                (EV_DELETE_FAIL,
                 {"error": f"{type(exc).__name__}: {exc}"}))
            return

        if delete_files:
            try:
                shutil.rmtree(config.CAPTURES_DIR / name, ignore_errors=True)
            except Exception:
                pass

        self.event_queue.put((EV_DELETE_DONE, {"name": name}))

    # ---- Polling Tk --------------------------------------------------
    def _poll(self):
        # Frames live (consomme la plus recente)
        latest = None
        try:
            while True:
                latest = self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            self._render_frame(latest)

        # Evenements (un par tour pour eviter les bouchons)
        try:
            while True:
                kind, payload = self.event_queue.get_nowait()
                if kind == "identify":
                    self._on_identify_result(payload)
                elif kind == "toast":
                    self._show_toast(payload)
                elif kind == "error":
                    self._show_toast(payload, level="error")
                elif kind == EV_ENROL_BATCH_DONE:
                    self._on_enrol_batch_done(payload)
                elif kind == EV_ENROL_BATCH_FAIL:
                    self._on_enrol_batch_fail(payload)
                elif kind == EV_DELETE_DONE:
                    self._on_delete_done(payload)
                elif kind == EV_DELETE_FAIL:
                    self._on_delete_fail(payload)
        except queue.Empty:
            pass

        if not self.stop_event.is_set():
            self.root.after(33, self._poll)

    # ---- Rendu webcam + visage aligne (live) -------------------------
    def _render_frame(self, fr: FrameResult):
        self.last_frame = fr
        cv = self.webcam_canvas

        # FPS lisse (EMA)
        self.fps_smoothed = (0.85 * self.fps_smoothed + 0.15 * fr.fps
                             if self.fps_smoothed > 0 else fr.fps)
        self.webcam_fps_lbl.config(text=f"FPS: {self.fps_smoothed:5.1f}")

        # Image webcam
        photo, geom = _bgr_to_imagetk(fr.bgr, self.WEBCAM_CV_W,
                                       self.WEBCAM_CV_H, fit="contain")
        self._img_refs["webcam"] = photo
        cv.itemconfigure(self._webcam_img_id, image=photo)
        cv.delete("webcam_overlay")
        cv.delete("webcam_placeholder")

        # Bbox + yeux superposes (canvas items en coords canvas)
        if fr.bbox is not None:
            new_w, new_h, ox, oy = geom
            sx = new_w / fr.bgr.shape[1]
            sy = new_h / fr.bgr.shape[0]
            x, y, w, h = fr.bbox
            x1 = int(ox + x * sx); y1 = int(oy + y * sy)
            x2 = int(ox + (x + w) * sx); y2 = int(oy + (y + h) * sy)
            cv.create_rectangle(x1, y1, x2, y2,
                                outline=Theme.SUCCESS, width=2,
                                tags=("webcam_overlay",))
            if fr.eyes is not None:
                for (ex, ey) in fr.eyes:
                    cx = int(ox + ex * sx); cy = int(oy + ey * sy)
                    cv.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                                   outline=Theme.SUCCESS, width=2,
                                   tags=("webcam_overlay",))
            self.webcam_state_lbl.config(text="visage: oui",
                                         fg=Theme.SUCCESS)
        else:
            self.webcam_state_lbl.config(text="visage: non",
                                         fg=Theme.DANGER)

        # Visage aligne (gros pixels NEAREST)
        if fr.aligned is not None:
            photo_a, _ = _bgr_to_imagetk(fr.aligned,
                                         self.FACE_CV_SIZE,
                                         self.FACE_CV_SIZE,
                                         fit="stretch")
            self._img_refs["aligned"] = photo_a
            self.aligned_canvas.itemconfigure(self._aligned_img_id,
                                              image=photo_a)
            self.aligned_canvas.delete("aligned_placeholder")

    # ---- Resultat d'identification -----------------------------------
    def _on_identify_result(self, res: IdentifyResult):
        self.last_identify = res
        # Si Auto etait coche, le worker a calcule un seuil optimal par
        # balayage LOO -- on le pousse dans le curseur pour que la
        # decision affichee soit faite a ce seuil. En mode manuel, le
        # curseur reste maitre : on n'y touche pas.
        if res.auto_threshold:
            self.threshold_var.set(res.threshold_at_run)
            self.thr_readout.config(text=f"{res.threshold_at_run:.3f}")
        # Snake + landmarks overlay (sur image figee)
        photo, _ = _bgr_to_imagetk(res.aligned,
                                   self.FACE_CV_SIZE, self.FACE_CV_SIZE,
                                   fit="stretch")
        self._img_refs["overlay"] = photo
        cv = self.overlay_canvas
        cv.itemconfigure(self._overlay_img_id, image=photo)
        cv.delete("overlay_placeholder")
        cv.delete("overlay_anno")

        scale = self.FACE_CV_SIZE / config.IMG_SIZE
        # Snake (polyline ferme)
        pts = res.snake_points * scale
        flat = []
        for x, y in pts:
            flat.extend([float(x), float(y)])
        flat.extend([float(pts[0, 0]), float(pts[0, 1])])
        cv.create_line(*flat, fill=Theme.WARM, width=2, smooth=True,
                       tags=("overlay_anno",))
        # 8 landmarks
        labels_color = {
            "eye_l": Theme.INFO, "eye_r": Theme.INFO,
            "eye_l_out": Theme.INFO_SOFT, "eye_r_out": Theme.INFO_SOFT,
            "nose": Theme.SUCCESS,
            "mouth_l": Theme.WARM, "mouth_r": Theme.WARM,
            "chin": Theme.NEUTRAL,
        }
        for k, color in labels_color.items():
            x, y = res.landmarks[k]
            cx, cy = x * scale, y * scale
            cv.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                           fill=color, outline=Theme.SURFACE_1, width=1,
                           tags=("overlay_anno",))

        # Bar chart
        self._draw_chart(res.vector)

        # Recompute decision banner + top-3 a partir du seuil courant
        self._refresh_decision()

    def _refresh_decision(self):
        thr = float(self.threshold_var.get())
        self.thr_readout.config(text=f"{thr:.3f}")
        res = self.last_identify
        if res is None:
            self._render_top3([], thr, accepted=False)
            self._render_banner(state="wait")
            return
        cands = res.candidates
        if not cands:
            self._render_top3([], thr, accepted=False)
            self._render_banner(state="empty")
            return
        best_name, best_dist = cands[0]
        accepted = best_dist <= thr
        conf = identify.confidence(best_dist, thr) if accepted else 0.0
        self._render_top3(cands, thr, accepted=accepted)
        self._render_banner(state="ok" if accepted else "no",
                            name=best_name, dist=best_dist,
                            conf=conf, thr=thr,
                            elapsed_ms=res.elapsed_ms)

    # ---- Banniere de decision ----------------------------------------
    def _render_banner(self, state, name="", dist=None, conf=None,
                        thr=None, elapsed_ms=None):
        if state == "ok":
            self.banner_lbl.config(text="ACCES AUTORISE", fg=Theme.SUCCESS)
            self.banner_sub_lbl.config(text=f"Identifie : {name}",
                                       fg=Theme.TEXT_HI)
            self._draw_door("open")
        elif state == "no":
            self.banner_lbl.config(text="ACCES REFUSE", fg=Theme.DANGER)
            self.banner_sub_lbl.config(text="Identite inconnue",
                                       fg=Theme.TEXT_MID)
            self._draw_door("closed")
        elif state == "empty":
            self.banner_lbl.config(text="BASE VIDE", fg=Theme.NEUTRAL)
            self.banner_sub_lbl.config(text="Aucun vecteur dans dataset.csv.",
                                       fg=Theme.TEXT_MID)
            self._draw_door("closed")
        else:  # wait
            self.banner_lbl.config(text="EN ATTENTE", fg=Theme.NEUTRAL)
            self.banner_sub_lbl.config(text="Lancez une identification.",
                                       fg=Theme.TEXT_MID)
            self._draw_door("wait")

        if dist is not None:
            self.dist_val_lbl.config(text=f"{dist:.3f}")
        else:
            self.dist_val_lbl.config(text="--")
        if thr is not None:
            self.thr_val_lbl.config(text=f"{thr:.3f}")
        else:
            self.thr_val_lbl.config(text=f"{float(self.threshold_var.get()):.3f}")
        if conf is not None:
            self.conf_val_lbl.config(text=f"{conf * 100:.0f} %")
        else:
            self.conf_val_lbl.config(text="--")

    def _draw_door(self, state):
        c = self.door_canvas
        c.delete("all")
        w = int(c["width"]); h = int(c["height"])
        # Cadre
        c.create_rectangle(15, 6, w - 15, h - 6,
                            outline=Theme.BORDER_HARD, width=2,
                            fill=Theme.SURFACE_2)
        if state == "open":
            # Battant entrouvert + voyant vert
            c.create_polygon(20, 12, w // 2, 22, w // 2, h - 18, 20, h - 12,
                              fill="#0f0f0f", outline=Theme.BORDER_HARD)
            c.create_polygon(w // 2, 22, w - 20, 12, w - 20, h - 12,
                              w // 2, h - 18,
                              fill="#5a4a30", outline=Theme.BORDER_HARD)
            c.create_oval(w - 22, 4, w - 6, 20,
                           fill=Theme.SUCCESS, outline=Theme.SUCCESS_DIM,
                           width=2)
        elif state == "closed":
            c.create_rectangle(20, 12, w - 20, h - 12,
                                fill="#5a4a30", outline=Theme.BORDER_HARD,
                                width=2)
            c.create_oval(w - 30, h // 2 - 4, w - 24, h // 2 + 4,
                           fill="#cccccc", outline=Theme.BORDER_HARD)
            c.create_oval(w - 22, 4, w - 6, 20,
                           fill=Theme.DANGER, outline=Theme.DANGER_DIM,
                           width=2)
        else:  # wait
            c.create_rectangle(20, 12, w - 20, h - 12,
                                fill="#3a3328", outline=Theme.BORDER_HARD,
                                width=2)
            c.create_oval(w - 22, 4, w - 6, 20,
                           fill=Theme.NEUTRAL, outline=Theme.WARM_SOFT,
                           width=2)

    # ---- Top 3 -------------------------------------------------------
    def _render_top3(self, cands, thr, accepted):
        for i, row in enumerate(self.top3_rows):
            if i < len(cands):
                name, dist = cands[i]
                row["name"].config(text=name, fg=Theme.TEXT_HI)
                row["dist"].config(text=f"d = {dist:.3f}",
                                   fg=Theme.TEXT_MID)
                # Confiance: 1 - d/thr, clampe
                rel = max(0.0, min(1.0, 1.0 - dist / max(thr, 1e-6)))
                row["conf"].config(text=f"confiance: {rel * 100:5.1f} %")
                # Bar render
                bar = row["bar"]
                bar.delete("all")
                bar.update_idletasks()
                W = max(int(bar.winfo_width()), 1)
                H = int(bar["height"])
                bar.create_rectangle(0, 0, W, H,
                                      fill=Theme.SURFACE_2, outline="")
                # Le gagnant accepte est vert, sinon ambre
                fill = (Theme.SUCCESS if (i == 0 and accepted)
                        else (Theme.WARM if i == 0 else Theme.INFO_SOFT))
                bar.create_rectangle(0, 0, int(W * rel), H,
                                      fill=fill, outline="")
                # Bande gauche d'emphase pour le gagnant
                if i == 0:
                    row["accent"].config(
                        bg=Theme.SUCCESS if accepted else Theme.DANGER_DIM)
                    row["row"].config(bg=Theme.SURFACE_2)
                    for child in row["row"].winfo_children():
                        if child is row["accent"]:
                            continue
                        child.config(bg=Theme.SURFACE_2)
                        for sub in child.winfo_children():
                            if isinstance(sub, (tk.Label, tk.Frame)):
                                sub.config(bg=Theme.SURFACE_2)
                                for s2 in sub.winfo_children():
                                    if isinstance(s2, tk.Label):
                                        s2.config(bg=Theme.SURFACE_2)
                else:
                    row["accent"].config(bg=Theme.SURFACE_1)
                    row["row"].config(bg=Theme.SURFACE_1)
                    for child in row["row"].winfo_children():
                        if child is row["accent"]:
                            continue
                        child.config(bg=Theme.SURFACE_1)
                        for sub in child.winfo_children():
                            if isinstance(sub, (tk.Label, tk.Frame)):
                                sub.config(bg=Theme.SURFACE_1)
                                for s2 in sub.winfo_children():
                                    if isinstance(s2, tk.Label):
                                        s2.config(bg=Theme.SURFACE_1)
            else:
                row["name"].config(text="--", fg=Theme.TEXT_LOW)
                row["dist"].config(text="d = --", fg=Theme.TEXT_LOW)
                row["conf"].config(text="confiance: --")
                row["bar"].delete("all")
                row["accent"].config(bg=Theme.SURFACE_1)

    # ---- Bar chart 30D -----------------------------------------------
    def _draw_chart(self, vec):
        c = self.chart_canvas
        c.delete("all")
        W = self.BAR_CV_W
        H = self.BAR_CV_H
        margin_l, margin_r = 36, 14
        margin_t, margin_b = 14, 28
        plot_w = W - margin_l - margin_r
        plot_h = H - margin_t - margin_b
        x0 = margin_l
        y0 = H - margin_b

        # Lignes de grille horizontales
        for frac in (0.0, 0.5, 1.0):
            y = y0 - frac * plot_h
            c.create_line(x0, y, x0 + plot_w, y,
                          fill=Theme.BORDER_SOFT, dash=(2, 4))
            c.create_text(x0 - 6, y, anchor="e",
                          text=f"{frac:.1f}",
                          fill=Theme.TEXT_LOW, font=Theme.mono(8))

        # Echelles par groupe (bornes choisies en lien avec features.py)
        DIST_MAX = 0.70   # diagonale 128sqrt(2) -> distances rares > 0.7
        SHAPE_MAX = 1.50  # rayons normalises ~1, on borne a 1.5

        slot = plot_w / 30.0
        bar_w = max(2, int(slot * 0.78))

        if vec is None:
            empty_text = "Aucun vecteur calcule. Appuyez sur Identifier."
            c.create_text(W // 2, H // 2 - 8, text=empty_text,
                          fill=Theme.TEXT_LOW, font=Theme.body(10))

        for i in range(30):
            cx = x0 + (i + 0.5) * slot
            if i < 14:
                color = Theme.INFO
                color_dim = Theme.INFO_SOFT
                vmax = DIST_MAX
            else:
                color = Theme.WARM
                color_dim = Theme.WARM_SOFT
                vmax = SHAPE_MAX
            # Fond colonne
            c.create_rectangle(cx - bar_w / 2, y0 - plot_h,
                               cx + bar_w / 2, y0,
                               outline="", fill=Theme.SURFACE_1)
            if vec is not None:
                v = float(vec[i])
                rel = min(1.0, max(0.0, v / vmax))
                top = y0 - rel * plot_h
                c.create_rectangle(cx - bar_w / 2, top,
                                   cx + bar_w / 2, y0,
                                   outline=color_dim, fill=color)

        # Axe horizontal
        c.create_line(x0, y0, x0 + plot_w, y0, fill=Theme.BORDER_HARD)

        # Separateur entre groupes
        sep_x = x0 + 14 * slot
        c.create_line(sep_x, y0 - plot_h, sep_x, y0,
                      fill=Theme.BORDER_HARD, dash=(3, 3))

        # Labels groupes
        c.create_text(x0 + 7 * slot, H - 10,
                      text="DISTANCES (1..14)",
                      fill=Theme.TEXT_MID, font=Theme.label(9))
        c.create_text(x0 + (14 + 8) * slot, H - 10,
                      text="FORME (15..30)",
                      fill=Theme.TEXT_MID, font=Theme.label(9))

        # Indices repere
        for idx in (1, 14, 15, 30):
            cx = x0 + (idx - 0.5) * slot
            c.create_text(cx, y0 + 2, anchor="n",
                          text=str(idx),
                          fill=Theme.TEXT_LOW, font=Theme.mono(8))

    # ---- Actions UI --------------------------------------------------
    def _on_identify(self):
        # Lit la coche Auto sur le thread Tk (jamais cote worker) et
        # transmet la decision au worker via la commande.
        auto = bool(self.auto_threshold_var.get())
        manual_thr = float(self.threshold_var.get())
        self.cmd_queue.put(
            (CMD_IDENTIFY, {"auto": auto, "manual_threshold": manual_thr}))

    def _on_auto_toggle(self):
        if self.auto_threshold_var.get():
            self._show_toast(
                "Auto active : le seuil sera recalcule a chaque identification.",
                level="info")
        else:
            self._show_toast(
                "Auto desactive : seuil pilote par le curseur.",
                level="info")

    def _on_open_enrol_wizard(self):
        """Ouvre le wizard d'enregistrement multi-poses."""
        if (self._enrol_wizard is not None
                and self._enrol_wizard.winfo_exists()):
            self._enrol_wizard.lift()
            return
        self._enrol_wizard = EnrolWizard(self)

    def _on_open_manager(self):
        """Ouvre le gestionnaire des identites."""
        if (self._identity_manager is not None
                and self._identity_manager.winfo_exists()):
            self._identity_manager.lift()
            return
        self._identity_manager = IdentityManager(self)

    # ---- Callbacks pour les evenements du worker ---------------------
    def _on_enrol_batch_done(self, payload):
        self.dataset_label.config(text=self._dataset_text())
        if (self._enrol_wizard is not None
                and self._enrol_wizard.winfo_exists()):
            self._enrol_wizard.on_batch_enrolled(payload)
        else:
            kept = payload.get("kept", 0)
            rej = payload.get("rejected", 0)
            self._show_toast(
                f"Encodage termine : {kept} vecteurs ({rej} rejetes).",
                level="ok" if rej == 0 else "info")

    def _on_enrol_batch_fail(self, payload):
        if (self._enrol_wizard is not None
                and self._enrol_wizard.winfo_exists()):
            self._enrol_wizard.on_batch_failed(payload)
        else:
            self._show_toast(
                f"Encodage echoue : {payload.get('error', '?')}",
                level="error")

    def _on_delete_done(self, payload):
        name = payload.get("name")
        self.dataset_label.config(text=self._dataset_text())
        # Reset banner si la decision en cours referencait l'identite
        if (self.last_identify is not None
                and any(cn == name
                        for cn, _ in self.last_identify.candidates)):
            self.last_identify = None
            self._render_top3([], float(self.threshold_var.get()),
                              accepted=False)
            self._render_banner(state="wait")
            self._draw_chart(None)
            self.overlay_canvas.delete("overlay_anno")
        if (self._identity_manager is not None
                and self._identity_manager.winfo_exists()):
            self._identity_manager.on_delete_done(payload)

    def _on_delete_fail(self, payload):
        if (self._identity_manager is not None
                and self._identity_manager.winfo_exists()):
            self._identity_manager.on_delete_failed(payload)
        else:
            self._show_toast(
                f"Suppression echouee : {payload.get('error', '?')}",
                level="error")

    # ---- Helpers exposes aux modales ---------------------------------
    def show_toast(self, text, level="info"):
        """Alias public pour les modales filles."""
        self._show_toast(text, level=level)

    def focus_identify(self):
        """Donne le focus au bouton Identifier (utilise apres la
        fermeture du wizard pour permettre Enter ou Espace)."""
        try:
            self.identify_btn.focus_set()
        except Exception:
            pass

    def on_wizard_closed(self, wizard):
        if self._enrol_wizard is wizard:
            self._enrol_wizard = None

    def on_manager_closed(self, manager):
        if self._identity_manager is manager:
            self._identity_manager = None

    def _on_threshold_change(self, _value):
        self._refresh_decision()

    def _on_quit(self):
        self.stop_event.set()
        try:
            self.worker.join(timeout=1.0)
        except RuntimeError:
            pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # ---- Toast (en superposition discrete) ---------------------------
    def _show_toast(self, text, level="info"):
        color = {"ok": Theme.SUCCESS, "error": Theme.DANGER,
                 "info": Theme.TEXT_HI}.get(level, Theme.TEXT_HI)
        if not hasattr(self, "_toast_lbl") or not self._toast_lbl.winfo_exists():
            self._toast_lbl = tk.Label(self.root, text="",
                                       bg=Theme.SURFACE_2, fg=Theme.TEXT_HI,
                                       font=Theme.body(10),
                                       padx=14, pady=8,
                                       highlightbackground=Theme.BORDER_HARD,
                                       highlightthickness=1)
        self._toast_lbl.config(text=text, fg=color)
        self._toast_lbl.place(relx=0.5, rely=0.96, anchor="s")
        self.root.after(3500, lambda l=self._toast_lbl: l.place_forget()
                        if l.winfo_exists() else None)

    # ------------------------------------------------------------------
    def run(self):
        self.root.mainloop()


def main():
    # Emet une trace lisible dans la console pour le debug etudiant
    print("[VPO] Banc de test du modele -- demarrage")
    print(f"[VPO] dataset.csv : {config.DATASET_CSV}")
    app = TestApp()
    with app.dataset_lock:
        n = len(app.vectors)
        persons = len(set(app.names))
    print(f"[VPO] base chargee : {persons} personnes / {n} vecteurs")
    print(f"[VPO] seuil par defaut : {config.DEFAULT_THRESHOLD}")
    app.run()
    print("[VPO] Au revoir.")


if __name__ == "__main__":
    main()
