import tkinter as tk
from tkinter import ttk


class Axe2Screen(tk.Tk):
    """
    Axe 2 — Linear Systems
    Layout:
      Left panel  : user inputs (matrix up to 4x4) + algorithm menus
      Right panel : results area (graph placeholder + iterations table)
    """

    def __init__(self):
        super().__init__()
        self.title("Axe 2 — Linear Systems")
        self.geometry("1050x750")
        self.configure(bg="#f0f4f8")
        self._matrix_entries = []
        self._b_entries = []
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
        bar = tk.Frame(self, bg="#2980b9", height=55)
        bar.pack(fill="x")
        tk.Button(
            bar, text="← Main", bg="#1a6499", fg="white",
            font=("Helvetica", 10), relief="flat", cursor="hand2",
            command=self._back,
        ).pack(side="left", padx=10, pady=12)
        tk.Label(
            bar, text="Axe 2 — Linear Systems",
            bg="#2980b9", fg="white", font=("Helvetica", 16, "bold"),
        ).pack(side="left", padx=10)

    # ── Left panel ────────────────────────────────────────────────
    def _left_panel(self, parent):
        frame = tk.Frame(parent, bg="#f0f4f8")
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # Matrix size selector
        size_lf = tk.LabelFrame(frame, text="User Inputs",
                                bg="#f0f4f8", fg="#2980b9",
                                font=("Helvetica", 11, "bold"),
                                padx=10, pady=8)
        size_lf.pack(fill="x", pady=(0, 8))

        tk.Label(size_lf, text="System size (n × n), max 4:",
                 bg="#f0f4f8", font=("Helvetica", 10)).pack(anchor="w")
        size_row = tk.Frame(size_lf, bg="#f0f4f8")
        size_row.pack(anchor="w", pady=(2, 6))
        self.size_var = tk.IntVar(value=3)
        for n in [2, 3, 4]:
            tk.Radiobutton(size_row, text=f"{n}×{n}", variable=self.size_var,
                           value=n, bg="#f0f4f8",
                           command=self._rebuild_matrix,
                           font=("Helvetica", 10)).pack(side="left", padx=4)

        tk.Label(size_lf, text="Linear system  Ax = b :",
                 bg="#f0f4f8", font=("Helvetica", 10, "italic")).pack(anchor="w")

        # Matrix input area
        self.matrix_frame = tk.Frame(size_lf, bg="#f0f4f8")
        self.matrix_frame.pack(anchor="w", pady=4)
        self._rebuild_matrix()

        # Actions on matrices
        act_lf = tk.LabelFrame(frame, text="Actions on Matrices",
                               bg="#f0f4f8", fg="#2980b9",
                               font=("Helvetica", 11, "bold"),
                               padx=10, pady=8)
        act_lf.pack(fill="x", pady=(0, 8))

        for label in ["Normes Induites", "Matrice Triangulaire Sup / Inf", "..."]:
            tk.Button(act_lf, text=label, bg="#eaf4fb", fg="#1a6499",
                      font=("Helvetica", 10), relief="solid", bd=1,
                      width=30, cursor="hand2").pack(pady=3, anchor="w")

        # Direct algorithms
        dir_lf = tk.LabelFrame(frame, text="Direct Algorithms",
                               bg="#f0f4f8", fg="#2980b9",
                               font=("Helvetica", 11, "bold"),
                               padx=10, pady=8)
        dir_lf.pack(fill="x", pady=(0, 8))

        self.algo_var = tk.StringVar(value="Gauss Elimination")
        direct_algos = [
            ("Gauss Elimination", "  (pivot partiel / total)"),
            ("Décomposition LU", ""),
            ("Cholesky", ""),
        ]
        for algo, note in direct_algos:
            row = tk.Frame(dir_lf, bg="#f0f4f8")
            row.pack(anchor="w", pady=1)
            tk.Radiobutton(row, text=algo, variable=self.algo_var,
                           value=algo, bg="#f0f4f8",
                           font=("Helvetica", 10)).pack(side="left")
            if note:
                tk.Label(row, text=note, bg="#f0f4f8", fg="#e74c3c",
                         font=("Helvetica", 9, "italic")).pack(side="left")

        # Indirect algorithms
        ind_lf = tk.LabelFrame(frame, text="Indirect Algorithms",
                               bg="#f0f4f8", fg="#2980b9",
                               font=("Helvetica", 11, "bold"),
                               padx=10, pady=8)
        ind_lf.pack(fill="x", pady=(0, 8))

        indirect_algos = ["Jacobi", "Gauss-Seidel", "Relaxation"]
        for algo in indirect_algos:
            tk.Radiobutton(ind_lf, text=algo, variable=self.algo_var,
                           value=algo, bg="#f0f4f8",
                           font=("Helvetica", 10)).pack(anchor="w")

        tk.Label(ind_lf, text="  Requires: Matrice DDS",
                 bg="#f0f4f8", fg="#8e44ad",
                 font=("Helvetica", 9, "italic")).pack(anchor="w")

        tk.Button(frame, text="▶  Run Algorithm", bg="#2980b9", fg="white",
                  font=("Helvetica", 12, "bold"), relief="flat",
                  cursor="hand2", height=2).pack(fill="x", pady=4)

    # ── Matrix entry grid ─────────────────────────────────────────
    def _rebuild_matrix(self):
        for w in self.matrix_frame.winfo_children():
            w.destroy()
        self._matrix_entries.clear()
        self._b_entries.clear()

        n = self.size_var.get()

        # Column headers
        tk.Label(self.matrix_frame, text="A", bg="#f0f4f8",
                 font=("Helvetica", 9, "bold"), width=4).grid(
            row=0, column=0, columnspan=n)
        tk.Label(self.matrix_frame, text="b", bg="#f0f4f8",
                 font=("Helvetica", 9, "bold")).grid(
            row=0, column=n + 1, padx=(6, 0))

        for i in range(n):
            row_entries = []
            for j in range(n):
                e = tk.Entry(self.matrix_frame, width=5,
                             font=("Courier", 10), relief="solid", bd=1,
                             justify="center")
                e.insert(0, "0")
                e.grid(row=i + 1, column=j, padx=2, pady=2)
                row_entries.append(e)
            self._matrix_entries.append(row_entries)

            tk.Label(self.matrix_frame, text="|", bg="#f0f4f8").grid(
                row=i + 1, column=n, padx=4)

            b = tk.Entry(self.matrix_frame, width=5,
                         font=("Courier", 10), relief="solid", bd=1,
                         justify="center")
            b.insert(0, "0")
            b.grid(row=i + 1, column=n + 1, padx=2, pady=2)
            self._b_entries.append(b)

    # ── Right panel ───────────────────────────────────────────────
    def _right_panel(self, parent):
        frame = tk.Frame(parent, bg="#f0f4f8")
        frame.grid(row=0, column=1, sticky="nsew")
        frame.rowconfigure(0, weight=2)
        frame.rowconfigure(1, weight=3)
        frame.columnconfigure(0, weight=1)

        # Graph placeholder
        graph_frame = tk.LabelFrame(frame, text="Graph / Convergence Plot",
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

        cols = ("k", "x₁", "x₂", "x₃", "x₄", "‖r‖")
        tree = ttk.Treeview(tbl_frame, columns=cols, show="headings",
                            height=10)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=90, anchor="center")
        tree.pack(fill="both", expand=True, padx=8, pady=4)

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
    Axe2Screen().mainloop()
