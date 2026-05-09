import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
import sys
import os
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "modules", "chap 2 and 3"))

from linear_systems_resolution import (
    gaussian_elimination_partial_pivot,
    gaussian_elimination_total_pivot,
    solve_lu,
    solve_cholesky,
    solve_iteratif,
    gauss_seidel,
    is_strictly_diagonally_dominant,
    is_symmetric_positive_definite,
    induced_matrix_norm,
    spectral_radius,
)


class Axe2Screen(tk.Frame):
    """
    Axe 2 — Linear Systems (fully functional)
    Layout mirrors Axe 3:
      Left panel  : user inputs (matrix up to 4×4) + algorithm selector
      Right panel : result text box + matplotlib figure (plot + table inside figure)
    """

    def __init__(self, parent, back_callback=None):
        super().__init__(parent, bg="#f0f4f8")
        self._back_callback = back_callback
        self._matrix_entries = []
        self._b_entries = []
        self._current_fig = None
        self._current_table_data = None
        self.omega_entry = None
        self._build()
        self._update_extra()

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
            bar, text="← Retour", bg="#1a6499", fg="white",
            font=("Helvetica", 10), relief="flat", cursor="hand2",
            command=self._back,
        ).pack(side="left", padx=10, pady=12)
        tk.Label(
            bar, text="Axe 2 — Systèmes Linéaires",
            bg="#2980b9", fg="white", font=("Helvetica", 16, "bold"),
        ).pack(side="left", padx=10)

    # ── Left panel ────────────────────────────────────────────────
    def _left_panel(self, parent):
        frame = tk.Frame(parent, bg="#f0f4f8")
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # ── Matrix size + input ──
        size_lf = tk.LabelFrame(frame, text="Entrées Utilisateur", bg="#f0f4f8",
                                fg="#2980b9", font=("Helvetica", 11, "bold"),
                                padx=10, pady=8)
        size_lf.pack(fill="x", pady=(0, 8))

        tk.Label(size_lf, text="Taille du système (n × n), max 4 :",
                 bg="#f0f4f8", font=("Helvetica", 10)).pack(anchor="w")
        size_row = tk.Frame(size_lf, bg="#f0f4f8")
        size_row.pack(anchor="w", pady=(2, 6))
        self.size_var = tk.IntVar(value=3)
        for n in [2, 3, 4]:
            tk.Radiobutton(size_row, text=f"{n}×{n}", variable=self.size_var,
                           value=n, bg="#f0f4f8", command=self._rebuild_matrix,
                           font=("Helvetica", 10)).pack(side="left", padx=4)

        tk.Label(size_lf, text="Système linéaire  Ax = b :",
                 bg="#f0f4f8", font=("Helvetica", 10, "italic")).pack(anchor="w")

        self.matrix_frame = tk.Frame(size_lf, bg="#f0f4f8")
        self.matrix_frame.pack(anchor="w", pady=4)
        self._rebuild_matrix()

        # ── Iterative parameters ──
        iter_lf = tk.LabelFrame(frame, text="Paramètres Itératifs",
                                bg="#f0f4f8", fg="#2980b9",
                                font=("Helvetica", 11, "bold"), padx=10, pady=6)
        iter_lf.pack(fill="x", pady=(0, 8))

        row_tol = tk.Frame(iter_lf, bg="#f0f4f8")
        row_tol.pack(fill="x", pady=2)
        tk.Label(row_tol, text="Tolérance (ε) :", bg="#f0f4f8",
                 font=("Helvetica", 10)).pack(side="left")
        self.tol_entry = tk.Entry(row_tol, width=10, font=("Courier", 10),
                                  relief="solid", bd=1)
        self.tol_entry.insert(0, "1e-6")
        self.tol_entry.pack(side="left", padx=6)

        row_iter = tk.Frame(iter_lf, bg="#f0f4f8")
        row_iter.pack(fill="x", pady=2)
        tk.Label(row_iter, text="Itérations max :", bg="#f0f4f8",
                 font=("Helvetica", 10)).pack(side="left")
        self.maxiter_entry = tk.Entry(row_iter, width=6, font=("Courier", 10),
                                      relief="solid", bd=1)
        self.maxiter_entry.insert(0, "100")
        self.maxiter_entry.pack(side="left", padx=6)

        # ── Extra param frame (ω appears dynamically for Relaxation) ──
        self.extra_frame = tk.Frame(frame, bg="#f0f4f8")
        self.extra_frame.pack(fill="x", pady=(0, 4))

        # ── Matrix operations ──
        act_lf = tk.LabelFrame(frame, text="Opérations Matricielles",
                               bg="#f0f4f8", fg="#2980b9",
                               font=("Helvetica", 11, "bold"), padx=10, pady=8)
        act_lf.pack(fill="x", pady=(0, 8))

        ops = [
            ("Normes induites  (‖A‖₁  ‖A‖₂  ‖A‖∞)", self._show_norms),
            ("Déterminant & Rang",                     self._show_det_rank),
            ("Vérifier SDD / DPS",                     self._show_convergence_check),
            ("Rayon spectral  ρ(A)",                   self._show_spectral_radius),
        ]
        for label, cmd in ops:
            tk.Button(act_lf, text=label, bg="#eaf4fb", fg="#1a6499",
                      font=("Helvetica", 10), relief="solid", bd=1,
                      width=34, cursor="hand2", anchor="w",
                      command=cmd).pack(pady=3, anchor="w")

        # ── Algorithm selector ──
        alg_lf = tk.LabelFrame(frame, text="Algorithme", bg="#f0f4f8",
                               fg="#2980b9", font=("Helvetica", 11, "bold"),
                               padx=10, pady=8)
        alg_lf.pack(fill="x", pady=(0, 8))

        self.algo_var = tk.StringVar(value="Gauss (partial pivot)")

        direct_lf = tk.LabelFrame(alg_lf, text="Direct", bg="#f0f4f8",
                                  fg="#27ae60", font=("Helvetica", 9, "bold"))
        direct_lf.pack(fill="x", pady=(0, 4))
        for algo in ["Gauss (partial pivot)", "Gauss (total pivot)",
                     "LU Decomposition", "Cholesky"]:
            tk.Radiobutton(direct_lf, text=algo, variable=self.algo_var,
                           value=algo, bg="#f0f4f8",
                           command=self._update_extra,
                           font=("Helvetica", 10)).pack(anchor="w")

        indirect_lf = tk.LabelFrame(alg_lf, text="Itératif  (nécessite une matrice SDD)",
                                    bg="#f0f4f8", fg="#e67e22",
                                    font=("Helvetica", 9, "bold"))
        indirect_lf.pack(fill="x")
        for algo in ["Jacobi", "Gauss-Seidel", "Relaxation"]:
            tk.Radiobutton(indirect_lf, text=algo, variable=self.algo_var,
                           value=algo, bg="#f0f4f8",
                           command=self._update_extra,
                           font=("Helvetica", 10)).pack(anchor="w")

        tk.Button(frame, text="▶  Lancer l'algorithme", bg="#2980b9", fg="white",
                  font=("Helvetica", 12, "bold"), relief="flat",
                  cursor="hand2", height=2,
                  command=self._run_algorithm).pack(fill="x", pady=6)

    # ── Matrix entry grid ─────────────────────────────────────────
    def _rebuild_matrix(self):
        for w in self.matrix_frame.winfo_children():
            w.destroy()
        self._matrix_entries.clear()
        self._b_entries.clear()

        n = self.size_var.get()
        tk.Label(self.matrix_frame, text="A", bg="#f0f4f8",
                 font=("Helvetica", 9, "bold")).grid(
            row=0, column=0, columnspan=n)
        tk.Label(self.matrix_frame, text="b", bg="#f0f4f8",
                 font=("Helvetica", 9, "bold")).grid(
            row=0, column=n + 1, padx=(6, 0))

        for i in range(n):
            row_entries = []
            for j in range(n):
                e = tk.Entry(self.matrix_frame, width=6, font=("Courier", 10),
                             relief="solid", bd=1, justify="center")
                e.insert(0, "0")
                e.grid(row=i + 1, column=j, padx=2, pady=2)
                row_entries.append(e)
            self._matrix_entries.append(row_entries)
            tk.Label(self.matrix_frame, text="|", bg="#f0f4f8").grid(
                row=i + 1, column=n, padx=4)
            b = tk.Entry(self.matrix_frame, width=6, font=("Courier", 10),
                         relief="solid", bd=1, justify="center")
            b.insert(0, "0")
            b.grid(row=i + 1, column=n + 1, padx=2, pady=2)
            self._b_entries.append(b)

    def _update_extra(self):
        for w in self.extra_frame.winfo_children():
            w.destroy()
        self.omega_entry = None
        if self.algo_var.get() == "Relaxation":
            row = tk.Frame(self.extra_frame, bg="#f0f4f8")
            row.pack(anchor="w", pady=2)
            tk.Label(row, text="Facteur de relaxation ω (0 < ω < 2) :", bg="#f0f4f8",
                     font=("Helvetica", 10)).pack(side="left")
            self.omega_entry = tk.Entry(row, width=6, font=("Courier", 10),
                                        relief="solid", bd=1)
            self.omega_entry.insert(0, "1.25")
            self.omega_entry.pack(side="left", padx=6)

    # ── Right panel — mirrors axe3 structure exactly ──────────────
    def _right_panel(self, parent):
        frame = tk.Frame(parent, bg="#f0f4f8")
        frame.grid(row=0, column=1, sticky="nsew")

        self.result_frame = tk.LabelFrame(frame, text="Résultats", bg="#f0f4f8",
                                          fg="#2c3e50",
                                          font=("Helvetica", 11, "bold"))
        self.result_frame.pack(fill="both", expand=True, padx=8, pady=8)

        # Result text box
        res_lf = tk.LabelFrame(self.result_frame, text="Résultat",
                               bg="#f0f4f8", fg="#2980b9")
        res_lf.pack(fill="x", padx=8, pady=6)
        self.result_text = tk.Text(res_lf, height=4, font=("Courier", 10),
                                   bg="#eaf4fb", state="disabled")
        self.result_text.pack(fill="x", padx=8, pady=6)

        # Download buttons
        dl = tk.Frame(self.result_frame, bg="#f0f4f8")
        dl.pack(pady=(0, 4))
        tk.Button(dl, text="⬇ Télécharger le graphe", bg="#2980b9", fg="white",
                  font=("Helvetica", 10), relief="flat", cursor="hand2",
                  command=self._download_graph).pack(side="left", padx=6)
        tk.Button(dl, text="⬇ Télécharger le tableau", bg="#8e44ad", fg="white",
                  font=("Helvetica", 10), relief="flat", cursor="hand2",
                  command=self._download_table).pack(side="left", padx=6)

    # ── Helpers ───────────────────────────────────────────────────
    def _read_matrix(self):
        n = self.size_var.get()
        A = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(float(self._matrix_entries[i][j].get().strip()))
            A.append(row)
        b = [float(self._b_entries[i].get().strip()) for i in range(n)]
        return np.array(A, dtype=float), np.array(b, dtype=float)

    def _set_result(self, text):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)
        self.result_text.config(state="disabled")

    def _clear_plot_canvas(self):
        """Destroy only matplotlib canvas widgets — identified by the
        '_is_plot_canvas' attribute we set in _embed_figure.
        This preserves the result text box and download buttons."""
        for widget in list(self.result_frame.winfo_children()):
            if getattr(widget, "_is_plot_canvas", False):
                widget.destroy()
        self._current_fig = None
        self._current_table_data = None

    # ── Matrix operation callbacks ────────────────────────────────
    def _show_norms(self):
        try:
            A, _ = self._read_matrix()
            n1 = induced_matrix_norm(A, 1)
            n2 = induced_matrix_norm(A, 2)
            ni = induced_matrix_norm(A, np.inf)
            self._set_result(
                f"‖A‖₁  (max somme col) = {n1:.6f}\n"
                f"‖A‖₂  (spectrale)     = {n2:.6f}\n"
                f"‖A‖∞  (max somme lig) = {ni:.6f}"
            )
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def _show_det_rank(self):
        try:
            A, _ = self._read_matrix()
            det  = np.linalg.det(A)
            rank = np.linalg.matrix_rank(A)
            self._set_result(f"det(A)  = {det:.6f}\nrank(A) = {rank}")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def _show_convergence_check(self):
        try:
            A, _ = self._read_matrix()
            dds = is_strictly_diagonally_dominant(A)
            spd = is_symmetric_positive_definite(A)
            verdict = ("✔ SDD satisfaite — convergence des méthodes itératives garantie."
                       if dds else
                       "✘ SDD non satisfaite — les méthodes itératives peuvent diverger.")
            self._set_result(
                f"Strictement Diagonalement Dominante (SDD) : {dds}\n"
                f"Définie Positive Symétrique          (DPS) : {spd}\n"
                f"{verdict}"
            )
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def _show_spectral_radius(self):
        try:
            A, _ = self._read_matrix()
            rho = spectral_radius(A)
            verdict = ("✔ ρ < 1 — la méthode itérative converge."
                       if rho < 1 else
                       "✘ ρ ≥ 1 — la méthode itérative peut ne pas converger.")
            self._set_result(f"Rayon spectral ρ(A) = {rho:.6f}\n{verdict}")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    # ── Run algorithm ─────────────────────────────────────────────
    def _run_algorithm(self):
        try:
            A, b = self._read_matrix()
            algo     = self.algo_var.get()
            tol      = float(self.tol_entry.get())
            max_iter = int(self.maxiter_entry.get())
            n        = len(b)

            self._clear_plot_canvas()

            result_str   = ""
            history      = []
            is_iterative = algo in ("Jacobi", "Gauss-Seidel", "Relaxation")

            # ── Precondition checks ─────────────────────────────
            if algo == "Cholesky":
                if not is_symmetric_positive_definite(A):
                    messagebox.showerror(
                        "Matrice invalide",
                        "Cholesky nécessite une matrice Définie Positive Symétrique (DPS).\n"
                        "Utilisez 'Vérifier SDD / DPS' pour valider votre matrice."
                    )
                    return

            if algo in ("Jacobi", "Gauss-Seidel", "Relaxation"):
                if not is_strictly_diagonally_dominant(A):
                    # Warn but still allow — convergence not guaranteed
                    proceed = messagebox.askyesno(
                        "Avertissement",
                        "La matrice n'est pas Strictement Diagonalement Dominante (SDD).\n"
                        "La méthode itérative peut ne pas converger.\n\n"
                        "Continuer quand même ?"
                    )
                    if not proceed:
                        return

            if algo == "Relaxation":
                omega = float(self.omega_entry.get()) if self.omega_entry else 1.25
                if not (0 < omega < 2):
                    messagebox.showerror("Facteur ω invalide",
                                         "Le facteur de relaxation ω doit satisfaire 0 < ω < 2.")
                    return

            # ── Direct methods ──────────────────────────────────
            if algo == "Gauss (partial pivot)":
                x, U, _ = gaussian_elimination_partial_pivot(A, b)
                result_str = self._format_solution(x)
                self._plot_direct(A, b, x, U, "Gauss — Partial Pivot")

            elif algo == "Gauss (total pivot)":
                x, U, _, perm = gaussian_elimination_total_pivot(A, b)
                result_str = self._format_solution(x)
                self._plot_direct(A, b, x, U, "Gauss — Total Pivot")

            elif algo == "LU Decomposition":
                x, P, L, U = solve_lu(A, b)
                result_str = (self._format_solution(x) +
                              f"\n\nL =\n{np.array2string(L, precision=4)}"
                              f"\n\nU =\n{np.array2string(U, precision=4)}")
                self._plot_direct(A, b, x, U, "LU Decomposition")

            elif algo == "Cholesky":
                x, L = solve_cholesky(A, b)
                result_str = (self._format_solution(x) +
                              f"\n\nL =\n{np.array2string(L, precision=4)}")
                self._plot_direct(A, b, x, L @ L.T, "Cholesky")

            # ── Iterative methods ───────────────────────────────
            elif algo == "Jacobi":
                x0 = np.zeros(n)
                _, rho_B, hist = solve_iteratif(A, b, x0, tol,
                                                methode="jacobi",
                                                max_iter=max_iter)
                history    = [(row[0], row[1], row[2]) for row in hist]
                x          = hist[-1][1] if hist else x0
                result_str = self._format_solution(x) + f"\nρ(B) = {rho_B:.6f}"

            elif algo == "Gauss-Seidel":
                x0 = np.zeros(n)
                x, hist = gauss_seidel(A, b, x0=x0, tol=tol, max_iter=max_iter)
                history    = [(row[0], row[1], row[2]) for row in hist]
                result_str = self._format_solution(x)

            elif algo == "Relaxation":
                x0    = np.zeros(n)
                x, history = self._relaxation(A, b, x0, omega, tol, max_iter)
                result_str = self._format_solution(x) + f"\nω = {omega}"

            self._set_result(result_str)

            if is_iterative and history:
                self._plot_iterative(history, algo)

        except Exception as e:
            messagebox.showerror("Erreur", f"Échec de l'exécution :\n{str(e)}")

    # ── Relaxation (SOR) ──────────────────────────────────────────
    def _relaxation(self, A, b, x0, omega, tol, max_iter):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        x = x0.astype(float).copy()
        history = []
        for k in range(max_iter):
            x_old = x.copy()
            for i in range(n):
                # Use updated x[:i] (Gauss-Seidel style) + old x[i+1:]
                s    = (b[i]
                        - np.dot(A[i, :i],   x[:i])
                        - np.dot(A[i, i+1:], x_old[i+1:]))
                x[i] = (1 - omega) * x_old[i] + omega * s / A[i, i]
            err = np.linalg.norm(x - x_old, ord=np.inf)
            history.append((k + 1, x.copy(), err))
            if err < tol:
                break
        return x, history

    # ── Plotting — direct methods ─────────────────────────────────
    def _plot_direct(self, A, b, x, U, title):
        n         = len(b)
        residuals = np.abs(A @ x - b)

        table_rows = [[f"x{i+1}", f"{x[i]:.8f}", f"{residuals[i]:.2e}"]
                      for i in range(n)]
        self._current_table_data = (["Variable", "Valeur", "|rᵢ|"], table_rows)

        fig = Figure(figsize=(10, 8), dpi=100)
        gs  = fig.add_gridspec(3, 2, height_ratios=[3, 3, 2],
                               hspace=0.55, wspace=0.4)

        # Diagramme en barres des résidus
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar([f"eq{i+1}" for i in range(n)], residuals,
                color="#2980b9", edgecolor="white")
        ax1.set_title("Résidu |Ax − b| par équation")
        ax1.set_ylabel("|rᵢ|")
        ax1.set_yscale("symlog", linthresh=1e-14)
        ax1.grid(axis="y", alpha=0.4)

        # Diagramme en barres de la solution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar([f"x{i+1}" for i in range(n)], x,
                color="#27ae60", edgecolor="white")
        ax2.set_title("Vecteur solution x")
        ax2.grid(axis="y", alpha=0.4)

        # Carte de chaleur triangulaire supérieure
        ax3 = fig.add_subplot(gs[1, 0])
        im  = ax3.imshow(np.abs(U), cmap="Blues", aspect="auto")
        fig.colorbar(im, ax=ax3)
        ax3.set_title("Triangulaire supérieure |U|")

        # Nuage de valeurs propres
        ax4  = fig.add_subplot(gs[1, 1])
        cond = np.linalg.cond(A)
        eigs = np.linalg.eigvals(A)
        ax4.scatter(eigs.real, eigs.imag, color="#8e44ad", s=80, zorder=5)
        ax4.axhline(0, color="gray", lw=0.8)
        ax4.axvline(0, color="gray", lw=0.8)
        ax4.set_title(f"Valeurs propres  (κ = {cond:.2e})")
        ax4.set_xlabel("Re")
        ax4.set_ylabel("Im")
        ax4.grid(True, alpha=0.3)

        # Tableau de la solution dans la figure
        ax_tbl = fig.add_subplot(gs[2, :])
        ax_tbl.axis("off")
        tbl = ax_tbl.table(
            cellText=table_rows,
            colLabels=["Variable", "Valeur", "|rᵢ|"],
            cellLoc="center", loc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        tbl.scale(1.2, 2.0)

        fig.suptitle(title, fontsize=13, fontweight="bold")
        self._embed_figure(fig)

    # ── Plotting — iterative methods ──────────────────────────────
    def _plot_iterative(self, history, title):
        ks   = [h[0] for h in history]
        errs = [h[2] for h in history]

        n_vars     = len(history[0][1])
        cols       = ["k"] + [f"x{i+1}" for i in range(n_vars)] + ["‖err‖∞"]
        table_rows = []
        for k, xk, err in history:
            table_rows.append(
                [str(k)] + [f"{v:.6f}" for v in xk] + [f"{err:.2e}"]
            )
        self._current_table_data = (cols, table_rows)

        fig = Figure(figsize=(10, 8), dpi=100)
        gs  = fig.add_gridspec(3, 1, height_ratios=[4, 0.4, 2.5])

        # Convergence curve
        ax = fig.add_subplot(gs[0])
        ax.semilogy(ks, errs, "b-o", markersize=4, label="‖xₖ₊₁ − xₖ‖∞")
        ax.set_xlabel("Itération k")
        ax.set_ylabel("Erreur (échelle log)")
        ax.set_title(f"{title} — Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Iteration table inside figure — max 15 rows for readability
        ax_tbl = fig.add_subplot(gs[2])
        ax_tbl.axis("off")
        display_rows = table_rows[:15]
        tbl = ax_tbl.table(
            cellText=display_rows,
            colLabels=cols,
            cellLoc="center", loc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        tbl.scale(1.2, 2.0)

        fig.suptitle(title, fontsize=13, fontweight="bold")
        self._embed_figure(fig)

    # ── Embed figure ──────────────────────────────────────────────
    def _embed_figure(self, fig):
        self._current_fig = fig
        canvas = FigureCanvasTkAgg(fig, self.result_frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        # Tag it so _clear_plot_canvas can find and destroy it reliably
        widget._is_plot_canvas = True
        widget.pack(fill="both", expand=True)

    # ── Format solution ───────────────────────────────────────────
    def _format_solution(self, x):
        lines = [f"x{i+1} = {v:.8f}" for i, v in enumerate(x)]
        return "Solution x :\n" + "\n".join(lines)

    # ── Downloads ─────────────────────────────────────────────────
    def _download_graph(self):
        if self._current_fig is None:
            messagebox.showinfo("Info", "Aucun graphe à sauvegarder. Lancez d'abord un algorithme.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("Image PNG", "*.png"), ("PDF", "*.pdf")],
            title="Sauvegarder le graphe",
        )
        if path:
            self._current_fig.savefig(path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Sauvegardé", f"Graphe sauvegardé :\n{path}")

    def _download_table(self):
        if self._current_table_data is None:
            messagebox.showinfo("Info", "Aucun tableau à sauvegarder. Lancez d'abord un algorithme.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("Fichier CSV", "*.csv")],
            title="Sauvegarder le tableau",
        )
        if path:
            cols, rows = self._current_table_data
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                writer.writerows(rows)
            messagebox.showinfo("Sauvegardé", f"Tableau sauvegardé :\n{path}")

    # ── Back ──────────────────────────────────────────────────────
    def _back(self):
        if self._back_callback:
            self._back_callback()
        else:
            import subprocess
            path = os.path.join(os.path.dirname(__file__), "main_screen.py")
            subprocess.Popen([sys.executable, path])
            self.winfo_toplevel().destroy()


if __name__ == "__main__":
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    Axe2Screen().mainloop()