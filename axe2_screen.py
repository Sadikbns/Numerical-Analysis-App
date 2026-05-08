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
    solve_lu,  # pivoted LU (P, L, U)
    solve_cholesky,
    solve_iteratif,
    gauss_seidel,
    is_strictly_diagonally_dominant,
    is_symmetric_positive_definite,
    induced_matrix_norm,
    spectral_radius,
)


# ──────────────────────────────────────────────────────────────────────────────
# Extra helpers (course-aligned LU "sans pivot" + substitution + iteration B)
# ──────────────────────────────────────────────────────────────────────────────
def _forward_substitution(L, b):
    L = np.array(L, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    y = np.zeros(n, dtype=float)
    for i in range(n):
        if abs(L[i, i]) < 1e-15:
            raise ValueError("Singular lower-triangular matrix (zero diagonal).")
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y


def _back_substitution(U, y):
    U = np.array(U, dtype=float)
    y = np.array(y, dtype=float)
    n = len(y)
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-15:
            raise ValueError("Singular upper-triangular matrix (zero diagonal).")
        x[i] = (y[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]
    return x


def lu_no_pivot_course(A):
    """
    Course-aligned LU decomposition WITHOUT pivoting:
      A = L U
    with:
      L unit lower triangular (1s on diagonal)
      U upper triangular

    This is essentially Gauss elimination without row/column exchanges,
    storing the multipliers m_ik in L.

    Conditions:
      - all pivots U[k,k] must be non-zero (and not too small).
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square.")

    U = A.copy()
    L = np.eye(n, dtype=float)

    for k in range(n - 1):
        pivot = U[k, k]
        if abs(pivot) < 1e-15:
            raise ValueError(
                "LU (no pivot) failed: zero (or near-zero) pivot encountered.\n"
                "Use pivoted LU or Gauss with pivoting."
            )

        for i in range(k + 1, n):
            m = U[i, k] / pivot
            L[i, k] = m
            U[i, k:] = U[i, k:] - m * U[k, k:]

    return L, U


def jacobi_iteration_matrix(A):
    A = np.array(A, dtype=float)
    D = np.diag(np.diag(A))
    if np.any(np.abs(np.diag(D)) < 1e-15):
        raise ValueError("Jacobi: zero diagonal entry encountered (cannot invert D).")
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    B = np.linalg.inv(D) @ (L + U)
    f = np.linalg.inv(D) @ np.array([0.0])  # placeholder; f depends on b, not needed for rho
    return B


def gauss_seidel_iteration_matrix(A):
    A = np.array(A, dtype=float)
    D = np.diag(np.diag(A))
    if np.any(np.abs(np.diag(D)) < 1e-15):
        raise ValueError("Gauss-Seidel: zero diagonal entry encountered.")
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    # B_GS = (D - L)^(-1) U
    B = np.linalg.inv(D - L) @ U
    return B


def spectral_radius_matrix(M):
    eigs = np.linalg.eigvals(M)
    return float(np.max(np.abs(eigs))) if eigs.size else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
class Axe2Screen(tk.Tk):
    """
    Axe 2 — Résolution des Systèmes Linéaires
    =========================================

    Direct methods (course):
      - Gauss (partial pivot)
      - Gauss (total pivot)
      - LU (no pivot — course)
      - LU (pivoted)
      - Cholesky (SPD only)

    Iterative methods (course):
      - Jacobi
      - Gauss-Seidel
      - Relaxation (SOR)

    Matrix operations:
      - Induced norms
      - Determinant & Rank
      - DDS / SPD checks
      - Spectral radius ρ(A) (property of A, NOT a convergence criterion)
      - Spectral radii ρ(B_J), ρ(B_GS) (true iterative convergence criterion)
    """

    def __init__(self):
        super().__init__()
        self.title("Axe 2 — Résolution des Systèmes Linéaires")
        self.geometry("1150x860")
        self.configure(bg="#f0f4f8")

        self._matrix_entries = []
        self._b_entries = []
        self._current_fig = None
        self._current_table_data = None
        self.omega_entry = None

        self._build()
        self._update_extra()

    # ══════════════════════════════════════════════════════════════
    # BUILD
    # ══════════════════════════════════════════════════════════════
    def _build(self):
        self._header()
        content = tk.Frame(self, bg="#f0f4f8")
        content.pack(fill="both", expand=True, padx=15, pady=10)
        content.columnconfigure(0, weight=2)
        content.columnconfigure(1, weight=3)
        content.rowconfigure(0, weight=1)
        self._left_panel(content)
        self._right_panel(content)

    def _header(self):
        bar = tk.Frame(self, bg="#2980b9", height=55)
        bar.pack(fill="x")
        tk.Button(
            bar,
            text="← Main",
            bg="#1a6499",
            fg="white",
            font=("Helvetica", 10),
            relief="flat",
            cursor="hand2",
            command=self._back,
        ).pack(side="left", padx=10, pady=12)
        tk.Label(
            bar,
            text="Axe 2 — Résolution des Systèmes Linéaires",
            bg="#2980b9",
            fg="white",
            font=("Helvetica", 16, "bold"),
        ).pack(side="left", padx=10)

    # ══════════════════════════════════════════════════════════════
    # LEFT PANEL
    # ══════════════════════════════════════════════════════════════
    def _left_panel(self, parent):
        frame = tk.Frame(parent, bg="#f0f4f8")
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # ── Matrix size + Ax=b input
        size_lf = tk.LabelFrame(
            frame,
            text="User Inputs — Ax = b",
            bg="#f0f4f8",
            fg="#2980b9",
            font=("Helvetica", 11, "bold"),
            padx=10,
            pady=8,
        )
        size_lf.pack(fill="x", pady=(0, 8))

        tk.Label(
            size_lf,
            text="System size n × n  (max 4 × 4):",
            bg="#f0f4f8",
            font=("Helvetica", 10),
        ).pack(anchor="w")

        size_row = tk.Frame(size_lf, bg="#f0f4f8")
        size_row.pack(anchor="w", pady=(2, 6))
        self.size_var = tk.IntVar(value=3)
        for n in [2, 3, 4]:
            tk.Radiobutton(
                size_row,
                text=f"{n}×{n}",
                variable=self.size_var,
                value=n,
                bg="#f0f4f8",
                command=self._rebuild_matrix,
                font=("Helvetica", 10),
            ).pack(side="left", padx=4)

        tk.Label(
            size_lf,
            text="Enter matrix A and vector b:",
            bg="#f0f4f8",
            font=("Helvetica", 10, "italic"),
        ).pack(anchor="w")

        self.matrix_frame = tk.Frame(size_lf, bg="#f0f4f8")
        self.matrix_frame.pack(anchor="w", pady=4)
        self._rebuild_matrix()

        # ── Iterative parameters
        iter_lf = tk.LabelFrame(
            frame,
            text="Iterative Method Parameters",
            bg="#f0f4f8",
            fg="#2980b9",
            font=("Helvetica", 11, "bold"),
            padx=10,
            pady=6,
        )
        iter_lf.pack(fill="x", pady=(0, 8))

        row_tol = tk.Frame(iter_lf, bg="#f0f4f8")
        row_tol.pack(fill="x", pady=2)
        tk.Label(row_tol, text="Tolerance ε:", bg="#f0f4f8", font=("Helvetica", 10)).pack(
            side="left"
        )
        self.tol_entry = tk.Entry(row_tol, width=10, font=("Courier", 10), relief="solid", bd=1)
        self.tol_entry.insert(0, "1e-6")
        self.tol_entry.pack(side="left", padx=6)

        row_iter = tk.Frame(iter_lf, bg="#f0f4f8")
        row_iter.pack(fill="x", pady=2)
        tk.Label(row_iter, text="Max iterations:", bg="#f0f4f8", font=("Helvetica", 10)).pack(
            side="left"
        )
        self.maxiter_entry = tk.Entry(
            row_iter, width=6, font=("Courier", 10), relief="solid", bd=1
        )
        self.maxiter_entry.insert(0, "100")
        self.maxiter_entry.pack(side="left", padx=6)

        # Extra (ω) frame
        self.extra_frame = tk.Frame(frame, bg="#f0f4f8")
        self.extra_frame.pack(fill="x", pady=(0, 4))

        # ── Matrix operations
        act_lf = tk.LabelFrame(
            frame,
            text="Matrix Operations",
            bg="#f0f4f8",
            fg="#2980b9",
            font=("Helvetica", 11, "bold"),
            padx=10,
            pady=8,
        )
        act_lf.pack(fill="x", pady=(0, 8))

        ops = [
            ("Induced Norms  ‖A‖₁  ‖A‖₂  ‖A‖∞", self._show_norms),
            ("Determinant  &  Rank", self._show_det_rank),
            ("Check DDS / SPD", self._show_convergence_check),
            ("Spectral Radius  ρ(A)  (property)", self._show_spectral_radius_A),
            ("Iterative Convergence  ρ(B)  (Jacobi/GS)", self._show_spectral_radius_B),
        ]
        for label, cmd in ops:
            tk.Button(
                act_lf,
                text=label,
                bg="#eaf4fb",
                fg="#1a6499",
                font=("Helvetica", 10),
                relief="solid",
                bd=1,
                width=36,
                cursor="hand2",
                anchor="w",
                command=cmd,
            ).pack(pady=3, anchor="w")

        # ── Algorithm selector
        alg_lf = tk.LabelFrame(
            frame,
            text="Algorithm Selection",
            bg="#f0f4f8",
            fg="#2980b9",
            font=("Helvetica", 11, "bold"),
            padx=10,
            pady=8,
        )
        alg_lf.pack(fill="x", pady=(0, 8))

        self.algo_var = tk.StringVar(value="Gauss (partial pivot)")

        direct_lf = tk.LabelFrame(
            alg_lf,
            text="Direct Methods",
            bg="#f0f4f8",
            fg="#27ae60",
            font=("Helvetica", 9, "bold"),
        )
        direct_lf.pack(fill="x", pady=(0, 4))

        for algo in [
            "Gauss (partial pivot)",
            "Gauss (total pivot)",
            "LU (no pivot — course)",
            "LU Decomposition (pivoted)",
            "Cholesky",
        ]:
            tk.Radiobutton(
                direct_lf,
                text=algo,
                variable=self.algo_var,
                value=algo,
                bg="#f0f4f8",
                command=self._update_extra,
                font=("Helvetica", 10),
            ).pack(anchor="w")

        indirect_lf = tk.LabelFrame(
            alg_lf,
            text="Iterative Methods  (DDS is sufficient)",
            bg="#f0f4f8",
            fg="#e67e22",
            font=("Helvetica", 9, "bold"),
        )
        indirect_lf.pack(fill="x")

        for algo in ["Jacobi", "Gauss-Seidel", "Relaxation"]:
            tk.Radiobutton(
                indirect_lf,
                text=algo,
                variable=self.algo_var,
                value=algo,
                bg="#f0f4f8",
                command=self._update_extra,
                font=("Helvetica", 10),
            ).pack(anchor="w")

        tk.Button(
            frame,
            text="▶  Run Algorithm",
            bg="#2980b9",
            fg="white",
            font=("Helvetica", 12, "bold"),
            relief="flat",
            cursor="hand2",
            height=2,
            command=self._run_algorithm,
        ).pack(fill="x", pady=6)

    # ── Matrix entry grid
    def _rebuild_matrix(self):
        for w in self.matrix_frame.winfo_children():
            w.destroy()
        self._matrix_entries.clear()
        self._b_entries.clear()

        n = self.size_var.get()

        tk.Label(self.matrix_frame, text="A", bg="#f0f4f8", font=("Helvetica", 9, "bold")).grid(
            row=0, column=0, columnspan=n
        )
        tk.Label(self.matrix_frame, text="b", bg="#f0f4f8", font=("Helvetica", 9, "bold")).grid(
            row=0, column=n + 1, padx=(6, 0)
        )

        for i in range(n):
            row_entries = []
            for j in range(n):
                e = tk.Entry(
                    self.matrix_frame,
                    width=6,
                    font=("Courier", 10),
                    relief="solid",
                    bd=1,
                    justify="center",
                )
                e.insert(0, "0")
                e.grid(row=i + 1, column=j, padx=2, pady=2)
                row_entries.append(e)
            self._matrix_entries.append(row_entries)

            tk.Label(self.matrix_frame, text="|", bg="#f0f4f8").grid(row=i + 1, column=n, padx=4)

            b = tk.Entry(
                self.matrix_frame,
                width=6,
                font=("Courier", 10),
                relief="solid",
                bd=1,
                justify="center",
            )
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
            tk.Label(
                row,
                text="Relaxation factor ω  (0 < ω < 2):",
                bg="#f0f4f8",
                font=("Helvetica", 10),
            ).pack(side="left")
            self.omega_entry = tk.Entry(row, width=6, font=("Courier", 10), relief="solid", bd=1)
            self.omega_entry.insert(0, "1.25")
            self.omega_entry.pack(side="left", padx=6)

    # ══════════════════════════════════════════════════════════════
    # RIGHT PANEL
    # ══════════════════════════════════════════════════════════════
    def _right_panel(self, parent):
        frame = tk.Frame(parent, bg="#f0f4f8")
        frame.grid(row=0, column=1, sticky="nsew")

        self.result_frame = tk.LabelFrame(
            frame, text="Results", bg="#f0f4f8", fg="#2c3e50", font=("Helvetica", 11, "bold")
        )
        self.result_frame.pack(fill="both", expand=True, padx=8, pady=8)

        res_lf = tk.LabelFrame(self.result_frame, text="Result", bg="#f0f4f8", fg="#2980b9")
        res_lf.pack(fill="x", padx=8, pady=6)
        self.result_text = tk.Text(
            res_lf, height=6, font=("Courier", 10), bg="#eaf4fb", state="disabled"
        )
        self.result_text.pack(fill="x", padx=8, pady=6)

        dl = tk.Frame(self.result_frame, bg="#f0f4f8")
        dl.pack(pady=(0, 4))
        tk.Button(
            dl,
            text="⬇ Download Graph",
            bg="#2980b9",
            fg="white",
            font=("Helvetica", 10),
            relief="flat",
            cursor="hand2",
            command=self._download_graph,
        ).pack(side="left", padx=6)
        tk.Button(
            dl,
            text="⬇ Download Table",
            bg="#8e44ad",
            fg="white",
            font=("Helvetica", 10),
            relief="flat",
            cursor="hand2",
            command=self._download_table,
        ).pack(side="left", padx=6)

    # ══════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════
    def _read_matrix(self):
        n = self.size_var.get()
        try:
            A = []
            for i in range(n):
                row = []
                for j in range(n):
                    row.append(float(self._matrix_entries[i][j].get().strip()))
                A.append(row)
            b = [float(self._b_entries[i].get().strip()) for i in range(n)]
        except Exception:
            raise ValueError("Invalid input: please fill A and b with numeric values.")
        return np.array(A, dtype=float), np.array(b, dtype=float)

    def _set_result(self, text):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)
        self.result_text.config(state="disabled")

    def _clear_plot_canvas(self):
        for widget in list(self.result_frame.winfo_children()):
            if getattr(widget, "_is_plot_canvas", False):
                widget.destroy()
        self._current_fig = None
        self._current_table_data = None

    def _format_solution(self, x):
        lines = [f"  x{i+1} = {v:.8f}" for i, v in enumerate(x)]
        return "Solution x*:\n" + "\n".join(lines)

    def _recommend_algorithm(self, A):
        try:
            spd = is_symmetric_positive_definite(A)
            dds = is_strictly_diagonally_dominant(A)
            cond = np.linalg.cond(A)

            if spd:
                return (
                    "Recommendation:\n"
                    "  → Cholesky (matrix is SPD).\n"
                    "  → Efficient and stable."
                )
            if dds:
                return (
                    "Recommendation:\n"
                    "  → Jacobi / Gauss-Seidel (DDS is satisfied).\n"
                    "  → Iterative convergence is guaranteed."
                )
            if cond < 1e4:
                return (
                    f"Recommendation:\n"
                    f"  → LU Decomposition (κ(A) = {cond:.2e}).\n"
                    "  → Efficient for repeated solves with same A."
                )
            return (
                f"Recommendation:\n"
                f"  → Gauss (partial pivot) (κ(A) = {cond:.2e}).\n"
                "  → Pivoting improves numerical stability."
            )
        except Exception:
            return ""

    # ══════════════════════════════════════════════════════════════
    # MATRIX OPS
    # ══════════════════════════════════════════════════════════════
    def _show_norms(self):
        try:
            A, _ = self._read_matrix()
            n1 = induced_matrix_norm(A, 1)
            n2 = induced_matrix_norm(A, 2)
            ni = induced_matrix_norm(A, np.inf)
            self._set_result(
                "Induced Matrix Norms:\n"
                f"  ‖A‖₁  = {n1:.6f}\n"
                f"  ‖A‖₂  = {n2:.6f}\n"
                f"  ‖A‖∞  = {ni:.6f}"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _show_det_rank(self):
        try:
            A, _ = self._read_matrix()
            det = np.linalg.det(A)
            rank = np.linalg.matrix_rank(A)
            unique = "✔ Unique solution likely (det ≠ 0)." if abs(det) > 1e-12 else "✘ det ≈ 0 (no unique solution)."
            self._set_result(f"det(A)  = {det:.6f}\nrank(A) = {rank}\n{unique}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _show_convergence_check(self):
        try:
            A, _ = self._read_matrix()
            dds = is_strictly_diagonally_dominant(A)
            spd = is_symmetric_positive_definite(A)
            self._set_result(
                "Convergence / Applicability Checks:\n"
                f"  DDS (Strictly Diagonally Dominant): {dds}\n"
                f"  SPD (Symmetric Positive Definite):  {spd}\n\n"
                "Notes:\n"
                "  • DDS is a sufficient condition for Jacobi/GS convergence.\n"
                "  • SPD is required for Cholesky."
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _show_spectral_radius_A(self):
        try:
            A, _ = self._read_matrix()
            rho = spectral_radius(A)
            self._set_result(
                "Spectral Radius (property of A):\n"
                f"  ρ(A) = {rho:.6f}\n\n"
                "Important:\n"
                "  ρ(A) is NOT the convergence criterion for Jacobi/GS.\n"
                "  For iterative methods, you must check ρ(B) < 1."
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _show_spectral_radius_B(self):
        try:
            A, _ = self._read_matrix()
            BJ = jacobi_iteration_matrix(A)
            BGS = gauss_seidel_iteration_matrix(A)
            rhoJ = spectral_radius_matrix(BJ)
            rhoGS = spectral_radius_matrix(BGS)
            verdictJ = "✔ converges (ρ(BJ) < 1)" if rhoJ < 1 else "✘ may diverge (ρ(BJ) ≥ 1)"
            verdictGS = "✔ converges (ρ(BGS) < 1)" if rhoGS < 1 else "✘ may diverge (ρ(BGS) ≥ 1)"
            self._set_result(
                "Iterative Convergence Criterion (course):\n"
                "  Convergence ⇔ ρ(B) < 1\n\n"
                f"Jacobi:\n  ρ(BJ)  = {rhoJ:.6f}  → {verdictJ}\n\n"
                f"Gauss-Seidel:\n  ρ(BGS) = {rhoGS:.6f}  → {verdictGS}"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ══════════════════════════════════════════════════════════════
    # RUN
    # ══════════════════════════════════════════════════════════════
    def _run_algorithm(self):
        try:
            A, b = self._read_matrix()
            algo = self.algo_var.get()
            tol = float(self.tol_entry.get())
            max_iter = int(self.maxiter_entry.get())
            n = len(b)

            self._clear_plot_canvas()

            history = []
            is_iterative = algo in ("Jacobi", "Gauss-Seidel", "Relaxation")

            # Guards
            if algo == "Cholesky" and not is_symmetric_positive_definite(A):
                messagebox.showerror(
                    "Invalid Matrix",
                    "Cholesky requires an SPD matrix.\nUse 'Check DDS / SPD' first.",
                )
                return

            if is_iterative and not is_strictly_diagonally_dominant(A):
                proceed = messagebox.askyesno(
                    "Warning — DDS not satisfied",
                    "Matrix is NOT strictly diagonally dominant.\n"
                    "Iterative method may diverge.\n\nProceed anyway?",
                )
                if not proceed:
                    return

            omega = 1.25
            if algo == "Relaxation":
                omega = float(self.omega_entry.get()) if self.omega_entry else 1.25
                if not (0 < omega < 2):
                    messagebox.showerror("Invalid ω", "Relaxation factor must satisfy 0 < ω < 2.")
                    return

            result_str = ""

            # ── Direct methods
            if algo == "Gauss (partial pivot)":
                x, U, _ = gaussian_elimination_partial_pivot(A, b)
                result_str = self._format_solution(x)
                self._plot_direct(A, b, x, U, "Gauss — Partial Pivot")

            elif algo == "Gauss (total pivot)":
                x, U, _, perm = gaussian_elimination_total_pivot(A, b)
                _ = perm  # not displayed, but computation is correct
                result_str = self._format_solution(x)
                self._plot_direct(A, b, x, U, "Gauss — Total Pivot")

            elif algo == "LU (no pivot — course)":
                L, U = lu_no_pivot_course(A)
                y = _forward_substitution(L, b)
                x = _back_substitution(U, y)
                result_str = (
                    self._format_solution(x)
                    + "\n\nCourse LU (no pivot):  A = L U"
                    + f"\n\nL =\n{np.array2string(L, precision=4)}"
                    + f"\n\nU =\n{np.array2string(U, precision=4)}"
                )
                self._plot_direct(A, b, x, U, "LU (Course — no pivot)")

            elif algo == "LU Decomposition (pivoted)":
                x, P, L, U = solve_lu(A, b)
                # Convention in many implementations: P @ A = L @ U
                result_str = (
                    self._format_solution(x)
                    + "\n\nPivoted LU:  P·A = L·U"
                    + f"\n\nP =\n{np.array2string(P, precision=4)}"
                    + f"\n\nL =\n{np.array2string(L, precision=4)}"
                    + f"\n\nU =\n{np.array2string(U, precision=4)}"
                )
                self._plot_direct(A, b, x, U, "LU Decomposition (pivoted)")

            elif algo == "Cholesky":
                x, L = solve_cholesky(A, b)
                result_str = (
                    self._format_solution(x)
                    + "\n\nCholesky:  A = L·Lᵀ"
                    + f"\n\nL =\n{np.array2string(L, precision=4)}"
                )
                self._plot_direct(A, b, x, L @ L.T, "Cholesky")

            # ── Iterative methods
            elif algo == "Jacobi":
                # Will throw if diagonal has zeros (good)
                x0 = np.zeros(n)
                _, rho_B, hist = solve_iteratif(A, b, x0, tol, methode="jacobi", max_iter=max_iter)
                # hist: [k, xk, err]
                history = []
                for k, xk, err in hist:
                    res = float(np.linalg.norm(A @ xk - b, ord=np.inf))
                    history.append((k, xk, float(err), res))
                x = hist[-1][1] if hist else x0
                res_final = float(np.linalg.norm(A @ x - b, ord=np.inf))
                result_str = (
                    self._format_solution(x)
                    + f"\n\nρ(B_Jacobi) = {rho_B:.6f}"
                    + f"\n‖Ax-b‖∞     = {res_final:.2e}"
                )

            elif algo == "Gauss-Seidel":
                x0 = np.zeros(n)
                x, hist = gauss_seidel(A, b, x0=x0, tol=tol, max_iter=max_iter)
                history = []
                for k, xk, err in hist:
                    res = float(np.linalg.norm(A @ xk - b, ord=np.inf))
                    history.append((k, xk, float(err), res))
                res_final = float(np.linalg.norm(A @ x - b, ord=np.inf))
                result_str = self._format_solution(x) + f"\n\n‖Ax-b‖∞ = {res_final:.2e}"

            elif algo == "Relaxation":
                x0 = np.zeros(n)
                x, hist = self._relaxation(A, b, x0, omega, tol, max_iter)
                history = []
                for k, xk, err in hist:
                    res = float(np.linalg.norm(A @ xk - b, ord=np.inf))
                    history.append((k, xk, float(err), res))
                res_final = float(np.linalg.norm(A @ x - b, ord=np.inf))
                result_str = self._format_solution(x) + f"\n\nω = {omega}\n‖Ax-b‖∞ = {res_final:.2e}"

            # Recommendation (assignment requirement)
            rec = self._recommend_algorithm(A)
            if rec:
                result_str += "\n\n─────────────────────────\n" + rec

            self._set_result(result_str)

            if is_iterative and history:
                self._plot_iterative(history, algo)

        except Exception as e:
            messagebox.showerror("Error", f"Execution failed:\n{str(e)}")

    # ── Relaxation (SOR)
    def _relaxation(self, A, b, x0, omega, tol, max_iter):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)

        x = np.array(x0, dtype=float).copy()
        history = []

        for k in range(max_iter):
            x_old = x.copy()

            for i in range(n):
                if abs(A[i, i]) < 1e-15:
                    raise ValueError("SOR: zero pivot encountered on diagonal (a_ii = 0).")

                s = b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1 :], x_old[i + 1 :])
                x[i] = (1 - omega) * x_old[i] + omega * s / A[i, i]

            err = float(np.linalg.norm(x - x_old, ord=np.inf))
            history.append((k + 1, x.copy(), err))
            if err < tol:
                break

        return x, history

    # ══════════════════════════════════════════════════════════════
    # PLOTTING
    # ══════════════════════════════════════════════════════════════
    def _plot_direct(self, A, b, x, U, title):
        n = len(b)
        residuals = np.abs(A @ x - b)

        table_rows = [[f"x{i+1}", f"{x[i]:.8f}", f"{residuals[i]:.2e}"] for i in range(n)]
        self._current_table_data = (["Variable", "Value", "|rᵢ|"], table_rows)

        fig = Figure(figsize=(10, 8), dpi=100)
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 3, 2], hspace=0.55, wspace=0.4)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar([f"eq{i+1}" for i in range(n)], residuals, color="#2980b9", edgecolor="white")
        ax1.set_title("Residual  |Ax − b|  per equation")
        ax1.set_ylabel("|rᵢ|")
        ax1.set_yscale("symlog", linthresh=1e-14)
        ax1.grid(axis="y", alpha=0.4)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar([f"x{i+1}" for i in range(n)], x, color="#27ae60", edgecolor="white")
        ax2.set_title("Solution vector x")
        ax2.grid(axis="y", alpha=0.4)

        ax3 = fig.add_subplot(gs[1, 0])
        im = ax3.imshow(np.abs(U), cmap="Blues", aspect="auto")
        fig.colorbar(im, ax=ax3)
        ax3.set_title("|U| (upper triangular factor)")

        ax4 = fig.add_subplot(gs[1, 1])
        cond = np.linalg.cond(A)
        eigs = np.linalg.eigvals(A)
        ax4.scatter(eigs.real, eigs.imag, color="#8e44ad", s=80, zorder=5)
        ax4.axhline(0, color="gray", lw=0.8)
        ax4.axvline(0, color="gray", lw=0.8)
        ax4.set_title(f"Eigenvalues   κ(A) = {cond:.2e}")
        ax4.set_xlabel("Re")
        ax4.set_ylabel("Im")
        ax4.grid(True, alpha=0.3)

        ax_tbl = fig.add_subplot(gs[2, :])
        ax_tbl.axis("off")
        tbl = ax_tbl.table(
            cellText=table_rows, colLabels=["Variable", "Value", "|rᵢ|"], cellLoc="center", loc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        tbl.scale(1.2, 2.0)

        fig.suptitle(title, fontsize=13, fontweight="bold")
        self._embed_figure(fig)

    def _plot_iterative(self, history, title):
        """
        history entries here are: (k, xk, err_update, res_inf)
        """
        ks = [h[0] for h in history]
        errs = [h[2] for h in history]
        ress = [h[3] for h in history]

        n_vars = len(history[0][1])
        cols = ["k"] + [f"x{i+1}" for i in range(n_vars)] + ["‖xk+1-xk‖∞", "‖Ax-b‖∞"]

        table_rows = []
        for k, xk, err, res in history:
            table_rows.append([str(k)] + [f"{v:.6f}" for v in xk] + [f"{err:.2e}", f"{res:.2e}"])
        self._current_table_data = (cols, table_rows)

        fig = Figure(figsize=(10, 8), dpi=100)
        gs = fig.add_gridspec(3, 1, height_ratios=[4, 0.4, 2.5])

        ax = fig.add_subplot(gs[0])
        ax.semilogy(ks, errs, "b-o", markersize=4, label="‖xk+1 − xk‖∞")
        ax.semilogy(ks, ress, "r--s", markersize=4, label="‖Axk − b‖∞")
        ax.set_xlabel("Iteration k")
        ax.set_ylabel("Value (log scale)")
        ax.set_title(f"{title} — Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax_tbl = fig.add_subplot(gs[2])
        ax_tbl.axis("off")
        display_rows = table_rows[:15]
        tbl = ax_tbl.table(cellText=display_rows, colLabels=cols, cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.0)
        tbl.scale(1.15, 1.9)

        fig.suptitle(title, fontsize=13, fontweight="bold")
        self._embed_figure(fig)

    def _embed_figure(self, fig):
        self._current_fig = fig
        canvas = FigureCanvasTkAgg(fig, self.result_frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget._is_plot_canvas = True
        widget.pack(fill="both", expand=True)

    # ══════════════════════════════════════════════════════════════
    # DOWNLOADS
    # ══════════════════════════════════════════════════════════════
    def _download_graph(self):
        if self._current_fig is None:
            messagebox.showinfo("Info", "Run an algorithm first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF", "*.pdf")],
            title="Save Graph",
        )
        if path:
            self._current_fig.savefig(path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Graph saved:\n{path}")

    def _download_table(self):
        if self._current_table_data is None:
            messagebox.showinfo("Info", "Run an algorithm first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV file", "*.csv")],
            title="Save Table",
        )
        if path:
            cols, rows = self._current_table_data
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                writer.writerows(rows)
            messagebox.showinfo("Saved", f"Table saved:\n{path}")

    # ══════════════════════════════════════════════════════════════
    # NAVIGATION
    # ══════════════════════════════════════════════════════════════
    def _back(self):
        import subprocess

        path = os.path.join(os.path.dirname(__file__), "main_screen.py")
        subprocess.Popen([sys.executable, path])
        self.destroy()


if __name__ == "__main__":
    Axe2Screen().mainloop()
