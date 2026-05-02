import tkinter as tk
from tkinter import ttk, font


class MainScreen(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Numerical Analysis Tool")
        self.geometry("600x500")
        self.resizable(False, False)
        self.configure(bg="#f5f5f5")
        self._build()

    def _build(self):
        # ── Title bar ──────────────────────────────────────────────
        title_frame = tk.Frame(self, bg="#2c3e50", height=60)
        title_frame.pack(fill="x")
        tk.Label(
            title_frame,
            text="Numerical Analysis Tool",
            bg="#2c3e50",
            fg="white",
            font=("Helvetica", 20, "bold"),
        ).pack(pady=15)

        # ── App description ────────────────────────────────────────
        desc_frame = tk.LabelFrame(
            self,
            text="About",
            bg="#f5f5f5",
            fg="#2c3e50",
            font=("Helvetica", 11, "bold"),
            padx=15,
            pady=10,
        )
        desc_frame.pack(fill="x", padx=30, pady=(25, 10))

        desc_text = (
            "This application provides a comprehensive suite of numerical analysis tools.\n"
            "Choose one of the modules below to explore function analysis (Axe 1),\n"
            "linear system solvers (Axe 2), or additional tools (Axe 3)."
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
            self,
            text="Choose a Module",
            bg="#f5f5f5",
            fg="#2c3e50",
            font=("Helvetica", 11, "bold"),
            padx=15,
            pady=15,
        )
        sel_frame.pack(fill="x", padx=30, pady=10)

        axes = [
            ("Axe 1 — Function Analysis", "#27ae60", "axe1_screen.py"),
            ("Axe 2 — Linear Systems", "#2980b9", "axe2_screen.py"),
            ("Axe 3 — Interpolation / Approximation", "#8e44ad", "axe3_screen.py"),
        ]

        for label, color, module in axes:
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
                command=lambda m=module: self._open_module(m),
            )
            btn.pack(pady=6)

        # ── Footer ────────────────────────────────────────────────
        tk.Label(
            self,
            text="Select a module to begin",
            bg="#f5f5f5",
            fg="#999",
            font=("Helvetica", 9, "italic"),
        ).pack(pady=5)

    def _open_module(self, module_file):
        import subprocess, sys, os

        path = os.path.join(os.path.dirname(__file__), module_file)
        subprocess.Popen([sys.executable, path])


if __name__ == "__main__":
    app = MainScreen()
    app.mainloop()
