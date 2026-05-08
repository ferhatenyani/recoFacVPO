"""Fenetre Tkinter: simulation d'une porte (vue en perspective).

Etats: fermee + voyant rouge | ouverte + voyant vert.
Communique via une queue: ('open', name, conf) | ('denied', _, dist) |
                          ('wait', _, _)        | ('quit', _, _).
"""
import queue
import tkinter as tk


class DoorWindow:
    BG          = "#1c1c1c"
    BG_FLASH    = "#3a0a0a"
    DOOR_FRAME  = "#3d3d3d"
    DOOR_BODY   = "#8a7a5c"
    DOOR_EDGE   = "#5a4a30"
    LIGHT_RED   = "#ff3333"
    LIGHT_GREEN = "#33ff66"

    def __init__(self, event_queue):
        self.q = event_queue
        self.root = tk.Tk()
        self.root.title("Porte securisee")
        self.root.geometry("440x540")
        self.root.configure(bg=self.BG)

        self.status = tk.Label(self.root, text="En attente...",
                               font=("Helvetica", 14, "bold"),
                               fg="#cccccc", bg=self.BG)
        self.status.pack(pady=(12, 2))

        self.detail = tk.Label(self.root, text="Lancez l'identification.",
                               font=("Helvetica", 10),
                               fg="#888888", bg=self.BG)
        self.detail.pack()

        self.canvas = tk.Canvas(self.root, width=420, height=400,
                                bg="#2a2a2a", highlightthickness=0)
        self.canvas.pack(pady=12)

        self._draw_closed()
        self.root.after(80, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def run(self):
        self.root.mainloop()

    # --- rendu ---------------------------------------------------------
    def _draw_frame(self):
        c = self.canvas
        c.delete("all")
        c.create_polygon(80, 60, 340, 60, 360, 380, 60, 380,
                         fill=self.DOOR_FRAME, outline="#666666", width=2)

    def _draw_closed(self):
        self._draw_frame()
        c = self.canvas
        c.create_polygon(95, 75, 325, 75, 340, 365, 80, 365,
                         fill=self.DOOR_BODY, outline=self.DOOR_EDGE, width=2)
        c.create_oval(305, 215, 320, 230, fill="#cccccc", outline="#888888")
        self._light(self.LIGHT_RED, "#660000")

    def _draw_open(self):
        self._draw_frame()
        c = self.canvas
        c.create_polygon(95, 75, 200, 92, 200, 358, 80, 365,
                         fill="#101010", outline="#333333", width=2)
        c.create_polygon(200, 92, 325, 75, 340, 365, 200, 358,
                         fill=self.DOOR_BODY, outline=self.DOOR_EDGE, width=2)
        c.create_oval(310, 215, 325, 230, fill="#cccccc", outline="#888888")
        self._light(self.LIGHT_GREEN, "#005522")

    def _light(self, fill, outline):
        self.canvas.create_oval(380, 28, 405, 53, fill=fill, outline=outline, width=2)

    # --- evenements ----------------------------------------------------
    def _poll(self):
        try:
            while True:
                self._handle(self.q.get_nowait())
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def _handle(self, event):
        action, label, value = event
        if action == "open":
            self._draw_open()
            self.status.config(text=f"ACCES AUTORISE — {label}", fg=self.LIGHT_GREEN)
            self.detail.config(text=f"Confiance: {value * 100:.1f}%")
            self.root.after(2500, self._reset)
        elif action == "denied":
            self._draw_closed()
            self.status.config(text="ACCES REFUSE — Inconnu", fg=self.LIGHT_RED)
            self.detail.config(text=f"Distance min.: {value:.3f}")
            self._flash()
        elif action == "wait":
            self._reset()
        elif action == "quit":
            self.root.quit()

    def _reset(self):
        self._draw_closed()
        self.status.config(text="En attente...", fg="#cccccc")
        self.detail.config(text="Lancez l'identification.")

    def _flash(self, n=4):
        if n <= 0:
            self.root.configure(bg=self.BG)
            self.status.configure(bg=self.BG)
            self.detail.configure(bg=self.BG)
            return
        bg = self.BG_FLASH if n % 2 else self.BG
        self.root.configure(bg=bg)
        self.status.configure(bg=bg)
        self.detail.configure(bg=bg)
        self.root.after(150, lambda: self._flash(n - 1))

    def _on_close(self):
        self.root.quit()
