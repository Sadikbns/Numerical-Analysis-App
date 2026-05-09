import tkinter as tk
from tkinter import ttk, font


class MainScreen(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Outil d'Analyse Numérique")
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self._main_w = min(800, sw - 80)
        self._main_h = min(700, sh - 80)
        self.geometry(f"{self._main_w}x{self._main_h}")
        self.resizable(False, False)
        self.configure(bg="#f5f5f5")

        # Container that fills the whole window; child frames are swapped inside it
        self._container = tk.Frame(self)
        self._container.pack(fill="both", expand=True)

        self._current_screen = None
        self._show_main()

    def _swap(self, new_frame):
        """Destroy the current screen frame and show new_frame."""
        if self._current_screen is not None:
            self._current_screen.destroy()
        self._current_screen = new_frame
        new_frame.pack(fill="both", expand=True)

    def _show_main(self):
        frame = tk.Frame(self._container, bg="#f5f5f5")

        # ── Title bar ──────────────────────────────────────────────
        title_frame = tk.Frame(frame, bg="#2c3e50", height=60)
        title_frame.pack(fill="x")
        tk.Label(
            title_frame,
            text="Outil d'Analyse Numérique",
            bg="#2c3e50",
            fg="white",
            font=("Helvetica", 20, "bold"),
        ).pack(pady=15)

        # ── App description ────────────────────────────────────────
        desc_frame = tk.LabelFrame(
            frame,
            text="À propos",
            bg="#f5f5f5",
            fg="#2c3e50",
            font=("Helvetica", 11, "bold"),
            padx=15,
            pady=10,
        )
        desc_frame.pack(fill="x", padx=30, pady=(25, 10))

        desc_text = (
            "Cet outil regroupe trois modules d'analyse numérique :\n"
            "  • Axe 1 — Résolution d'équations non-linéaires (Dichotomie, Point Fixe, Newton)\n"
            "  • Axe 2 — Résolution de systèmes linéaires Ax = b (méthodes directes et itératives)\n"
            "  • Axe 3 — Interpolation et approximation (Lagrange, Newton, Moindres Carrés, Tchebychev, Descente de Gradient)"
        )
        tk.Label(
            desc_frame,
            text=desc_text,
            bg="#f5f5f5",
            fg="#444",
            font=("Helvetica", 11),
            justify="left",
            wraplength=500,
        ).pack(anchor="w")

        # ── Axe selector ──────────────────────────────────────────
        sel_frame = tk.LabelFrame(
            frame,
            text="Choisir un module",
            bg="#f5f5f5",
            fg="#2c3e50",
            font=("Helvetica", 11, "bold"),
            padx=15,
            pady=15,
        )
        sel_frame.pack(fill="x", padx=30, pady=10)

        axes = [
            ("Axe 1 — Équations Non-Linéaires",        "#27ae60", "axe1"),
            ("Axe 2 — Systèmes Linéaires",             "#2980b9", "axe2"),
            ("Axe 3 — Interpolation / Approximation",  "#8e44ad", "axe3"),
        ]

        for label, color, key in axes:
            btn = tk.Button(
                sel_frame,
                text=label,
                bg=color,
                fg="white",
                font=("Helvetica", 13, "bold"),
                width=36,
                height=2,
                relief="flat",
                cursor="hand2",
                command=lambda k=key: self._open_module(k),
            )
            btn.pack(pady=6)

        # ── Footer ────────────────────────────────────────────────
        tk.Label(
            frame,
            text="Sélectionnez un module pour commencer",
            bg="#f5f5f5",
            fg="#999",
            font=("Helvetica", 9, "italic"),
        ).pack(pady=5)

        self._swap(frame)

    def _open_module(self, key):
        """Import the axe screen module and show it as a frame inside this window."""
        import importlib.util, os

        module_map = {
            "axe1": "axe1_screen.py",
            "axe2": "axe2_screen.py",
            "axe3": "axe3_screen.py",
        }
        filename = module_map[key]
        path = os.path.join(os.path.dirname(__file__), filename)

        spec = importlib.util.spec_from_file_location(key, path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class_name = {"axe1": "Axe1Screen", "axe2": "Axe2Screen", "axe3": "Axe3Screen"}[key]
        screen_class = getattr(mod, class_name)

        # Resize window to fit the axe screen
        sizes = {"axe1": (1020, 720), "axe2": (1150, 820), "axe3": (1200, 860)}
        w, h = sizes[key]
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(w, sw - 80), min(h, sh - 80)
        self.geometry(f"{w}x{h}")
        self.resizable(True, True)

        frame = screen_class(self._container, back_callback=self._go_back)
        self._swap(frame)

    def _go_back(self):
        """Called by axe screens' back button — return to main menu."""
        self.geometry(f"{self._main_w}x{self._main_h}")
        self.resizable(False, False)
        self._show_main()


if __name__ == "__main__":
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    app = MainScreen()
    app.mainloop()