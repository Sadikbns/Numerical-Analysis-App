import tkinter as tk
from tkinter import ttk


class Axe1Screen(tk.Tk):
    """
    Axe 1 — Function Analysis
    Layout:
      Left panel  : user inputs + action menus
      Right panel : results area (graph placeholder + iterations table)
    """

    def __init__(self):
        super().__init__()
        self.title("Axe 1 — Function Analysis")
        self.geometry("950x680")
        self.configure(bg="#f0f4f8")
        self._build()

    # ──────────────────────────────────────────────────────────────
    def _build(self):
        self._header()
        content = tk.Frame(self, bg="#f0f4f8")
        content.pack(fill="both", expand=True, padx=15, pady=10)
        content.columnconfigure(0, weight=2)
        content.columnconfigure(1, weight=3)
        content.rowconfigure(0, weight=1)

        self._left_panel(content)
        self._right_panel(content)

    # ── Header ────────────────────────────────────────────────────
    def _header(self):
        bar = tk.Frame(self, bg="#27ae60", height=55)
        bar.pack(fill="x")
        tk.Button(
            bar, text="← Main", bg="#1e8449", fg="white",
            font=("Helvetica", 10), relief="flat", cursor="hand2",
            command=self._back,
        ).pack(side="left", padx=10, pady=12)
        tk.Label(
            bar, text="Axe 1 — Function Analysis",
            bg="#27ae60", fg="white", font=("Helvetica", 16, "bold"),
        ).pack(side="left", padx=10)

    # ── Left panel ────────────────────────────────────────────────
    def _left_panel(self, parent):
        frame = tk.Frame(parent, bg="#f0f4f8")
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # User inputs
        inp = tk.LabelFrame(frame, text="User Inputs", bg="#f0f4f8",
                            fg="#27ae60", font=("Helvetica", 11, "bold"),
                            padx=10, pady=8)
        inp.pack(fill="x", pady=(0, 10))

        tk.Label(inp, text="Function formula f(x):", bg="#f0f4f8",
                 font=("Helvetica", 10)).pack(anchor="w")
        self.func_entry = tk.Entry(inp, font=("Courier", 12), width=28,
                                   relief="solid", bd=1)
        self.func_entry.insert(0, "x**3 - x - 2")
        self.func_entry.pack(fill="x", pady=(2, 6))

        tk.Label(inp, text="Interval [a, b]:", bg="#f0f4f8",
                 font=("Helvetica", 10)).pack(anchor="w")
        iv = tk.Frame(inp, bg="#f0f4f8")
        iv.pack(fill="x", pady=(2, 6))
        self.a_entry = tk.Entry(iv, font=("Courier", 12), width=8,
                                relief="solid", bd=1)
        self.a_entry.insert(0, "1")
        self.a_entry.pack(side="left")
        tk.Label(iv, text="  to  ", bg="#f0f4f8").pack(side="left")
        self.b_entry = tk.Entry(iv, font=("Courier", 12), width=8,
                                relief="solid", bd=1)
        self.b_entry.insert(0, "2")
        self.b_entry.pack(side="left")

        tk.Label(inp, text="Tolerance (ε):", bg="#f0f4f8",
                 font=("Helvetica", 10)).pack(anchor="w")
        self.tol_entry = tk.Entry(inp, font=("Courier", 12), width=14,
                                  relief="solid", bd=1)
        self.tol_entry.insert(0, "1e-6")
        self.tol_entry.pack(anchor="w", pady=(2, 4))

        # Actions on functions
        act = tk.LabelFrame(frame, text="Actions on Functions",
                            bg="#f0f4f8", fg="#27ae60",
                            font=("Helvetica", 11, "bold"), padx=10, pady=8)
        act.pack(fill="x", pady=(0, 10))

        for label in ["Calc Derivatives", "Verify Continuity",
                      "Stability", "Contractante"]:
            tk.Button(act, text=label, bg="#ecf9f1", fg="#1e8449",
                      font=("Helvetica", 10), relief="solid", bd=1,
                      width=24, cursor="hand2").pack(pady=3, anchor="w")

        # Algorithm menu
        alg = tk.LabelFrame(frame, text="Available Algorithms",
                            bg="#f0f4f8", fg="#27ae60",
                            font=("Helvetica", 11, "bold"), padx=10, pady=8)
        alg.pack(fill="x", pady=(0, 10))

        self.algo_var = tk.StringVar(value="Dichotomie")
        for algo in ["Dichotomie", "Point Fixe", "Newton"]:
            tk.Radiobutton(alg, text=algo, variable=self.algo_var,
                           value=algo, bg="#f0f4f8",
                           font=("Helvetica", 10)).pack(anchor="w")

        tk.Button(frame, text="▶  Run Algorithm", bg="#27ae60", fg="white",
                  font=("Helvetica", 12, "bold"), relief="flat",
                  cursor="hand2", height=2).pack(fill="x", pady=4)

    # ── Right panel ───────────────────────────────────────────────
    def _right_panel(self, parent):
        frame = tk.Frame(parent, bg="#f0f4f8")
        frame.grid(row=0, column=1, sticky="nsew")
        frame.rowconfigure(0, weight=2)
        frame.rowconfigure(1, weight=3)

        # Graph placeholder
        graph_frame = tk.LabelFrame(frame, text="Graph",
                                    bg="#f0f4f8", fg="#2c3e50",
                                    font=("Helvetica", 11, "bold"))
        graph_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        canvas = tk.Canvas(graph_frame, bg="#ffffff", relief="solid", bd=1)
        canvas.pack(fill="both", expand=True, padx=8, pady=8)
        canvas.create_text(
            200, 90,
            text="[ Graph will appear here ]",
            fill="#bbb", font=("Helvetica", 13, "italic"),
        )

        # Iterations table
        tbl_frame = tk.LabelFrame(frame, text="Iteration Table",
                                  bg="#f0f4f8", fg="#2c3e50",
                                  font=("Helvetica", 11, "bold"))
        tbl_frame.grid(row=1, column=0, sticky="nsew")

        cols = ("n", "aₙ", "bₙ", "mₙ", "f(mₙ)", "ε")
        tree = ttk.Treeview(tbl_frame, columns=cols, show="headings",
                            height=8)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=80, anchor="center")
        tree.pack(fill="both", expand=True, padx=8, pady=4)

        # Download buttons
        dl = tk.Frame(tbl_frame, bg="#f0f4f8")
        dl.pack(pady=6)
        tk.Button(dl, text="⬇ Download Graph", bg="#2980b9", fg="white",
                  font=("Helvetica", 10), relief="flat",
                  cursor="hand2").pack(side="left", padx=6)
        tk.Button(dl, text="⬇ Download Table", bg="#8e44ad", fg="white",
                  font=("Helvetica", 10), relief="flat",
                  cursor="hand2").pack(side="left", padx=6)

    def _back(self):
        import subprocess, sys, os
        path = os.path.join(os.path.dirname(__file__), "main_screen.py")
        subprocess.Popen([sys.executable, path])
        self.destroy()


if __name__ == "__main__":
    Axe1Screen().mainloop()
