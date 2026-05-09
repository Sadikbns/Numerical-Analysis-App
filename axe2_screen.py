import os
import sys
import csv
import tkinter as tk
from tkinter import filedialog, messagebox

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

# ──────────────────────────────────────────────────────────────────────────────
# Step-by-step Gauss elimination (course-aligned, records each A^(k))
# ──────────────────────────────────────────────────────────────────────────────
def gauss_partial_steps(A, b):
    """
    Gauss elimination with partial pivoting.
    Returns x, list of step dicts, and final U.
    Each step dict has: {'pivot_row': i, 'pivot_col': k, 'A': matrix, 'b': vector}
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    M = np.hstack([A.copy(), b.reshape(-1, 1)])
    steps = []

    for k in range(n - 1):
        # Partial pivot: find max in column k below row k
        pivot_row = k + int(np.argmax(np.abs(M[k:, k])))
        if abs(M[pivot_row, k]) < 1e-15:
            raise ValueError(f"Matrice singulière ou presque singulière (pivot nul à l'étape {k+1}).")
        if pivot_row != k:
            M[[k, pivot_row]] = M[[pivot_row, k]]

        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k:] -= factor * M[k, k:]

        steps.append({
            'step': k + 1,
            'pivot': float(M[k, k]),
            'A': M[:, :n].copy(),
            'b': M[:, n].copy(),
        })

    U = M[:, :n]
    b_mod = M[:, n]
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b_mod[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x, steps, U


def gauss_total_steps(A, b):
    """
    Gauss elimination with total pivoting.
    Returns x, list of step dicts, final U, and column permutation.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    M = np.hstack([A.copy(), b.reshape(-1, 1)])
    perm = list(range(n))
    steps = []

    for k in range(n - 1):
        # Total pivot: find max in submatrix M[k:, k:n]
        sub = np.abs(M[k:, k:n])
        i_rel, j_rel = np.unravel_index(np.argmax(sub), sub.shape)
        i_max, j_max = k + i_rel, k + j_rel

        if abs(M[i_max, j_max]) < 1e-15:
            raise ValueError(f"Matrice singulière (pivot nul à l'étape {k+1}).")

        if i_max != k:
            M[[k, i_max]] = M[[i_max, k]]
        if j_max != k:
            M[:, [k, j_max]] = M[:, [j_max, k]]
            perm[k], perm[j_max] = perm[j_max], perm[k]

        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k:] -= factor * M[k, k:]

        steps.append({
            'step': k + 1,
            'pivot': float(M[k, k]),
            'A': M[:, :n].copy(),
            'b': M[:, n].copy(),
        })

    U = M[:, :n]
    b_mod = M[:, n]
    x_perm = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_perm[i] = (b_mod[i] - np.dot(U[i, i+1:], x_perm[i+1:])) / U[i, i]

    x = np.zeros(n)
    for i, p in enumerate(perm):
        x[p] = x_perm[i]

    return x, steps, U, perm


def lu_steps(A):
    """
    LU decomposition recording each elementary step (multipliers).
    Returns L, U, and list of step dicts with intermediate A^(k).
    Course-aligned: A = L U, L unit lower triangular.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    steps = []

    for k in range(n - 1):
        if abs(U[k, k]) < 1e-15:
            raise ValueError(
                f"LU sans pivot: pivot nul à l'étape {k+1}.\n"
                "Utilisez Gauss avec pivotage."
            )
        multipliers = {}
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            multipliers[i] = m
            U[i, k:] -= m * U[k, k:]

        steps.append({
            'step': k + 1,
            'pivot': float(U[k, k]),
            'multipliers': multipliers,
            'U_current': U.copy(),
            'L_current': L.copy(),
        })

    return L, U, steps


def format_matrix_str(M, name="", precision=4):
    """Format a numpy matrix as a readable string."""
    M = np.array(M, dtype=float)
    rows = []
    n, m = M.shape
    for i in range(n):
        row_str = "  [ " + "  ".join(f"{M[i,j]:+.{precision}f}" for j in range(m)) + " ]"
        rows.append(row_str)
    header = f"{name} =\n" if name else ""
    return header + "\n".join(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Design tokens
# ──────────────────────────────────────────────────────────────────────────────
COLORS = {
    "bg":          "#f0f4f8",
    "surface":     "#ffffff",
    "border":      "#d0d7e3",
    "primary":     "#1a6499",
    "primary_mid": "#2980b9",
    "primary_btn": "#1a6499",
    "primary_lt":  "#eaf4fb",
    "hdr_bg":      "#f5fbff",
    "text":        "#2c3e50",
    "text_label":  "#34495e",
    "text_muted":  "#7f8c8d",
    "purple":      "#8e44ad",
    "orange":      "#e67e22",
    "green":       "#27ae60",
    "red":         "#e74c3c",
}


def _text(parent, txt, size=10, bold=False, color=None, bg=None, **kw):
    opts = {
        "text": txt,
        "font": ("Helvetica", size, "bold" if bold else "normal"),
        "bg": bg or parent.cget("bg"),
        "fg": color or COLORS["text"],
    }
    opts.update(kw)
    return tk.Label(parent, **opts)


def _field(parent, default="", width=10, mono=True):
    e = tk.Entry(
        parent,
        width=width,
        font=("Courier" if mono else "Helvetica", 10),
        bg=COLORS["surface"],
        fg=COLORS["text"],
        relief="solid",
        bd=1,
        highlightthickness=1,
        highlightcolor=COLORS["primary_mid"],
        highlightbackground=COLORS["border"],
        insertbackground=COLORS["primary_mid"],
        justify="center",
    )
    e.insert(0, default)
    return e


def _button(parent, label, bg=None, fg="white", bold=False, size=10,
            pad_x=10, pad_y=6, cmd=None):
    return tk.Button(
        parent,
        text=label,
        bg=bg or COLORS["primary_mid"],
        fg=fg,
        font=("Helvetica", size, "bold" if bold else "normal"),
        relief="flat",
        cursor="hand2",
        padx=pad_x,
        pady=pad_y,
        activebackground=COLORS["primary"],
        activeforeground="white",
        command=cmd,
    )


def _line(parent):
    return tk.Frame(parent, bg=COLORS["border"], height=1)


def _panel(parent, title, dot=None, compact=False):
    card = tk.Frame(parent, bg=COLORS["border"])
    inner = tk.Frame(card, bg=COLORS["surface"])
    inner.pack(fill="both", expand=True, padx=1, pady=1)

    hdr = tk.Frame(inner, bg=COLORS["hdr_bg"], height=30)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)

    if dot:
        tk.Frame(hdr, bg=dot, width=4).pack(side="left", fill="y")
    _text(hdr, title, size=10, bold=True, bg=COLORS["hdr_bg"]).pack(
        side="left", padx=10, pady=6
    )

    _line(inner).pack(fill="x")

    body = tk.Frame(
        inner,
        bg=COLORS["surface"],
        padx=8 if compact else 12,
        pady=6 if compact else 10,
    )
    body.pack(fill="x")
    return card, body, inner


# ──────────────────────────────────────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────────────────────────────────────
class Axe2Screen(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Axe 2 — Résolution des Systèmes Linéaires")

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1200, sw - 80), min(860, sh - 80)
        self.geometry(f"{w}x{h}")
        self.minsize(1020, 680)
        self.configure(bg=COLORS["bg"])

        self._matrix_entries = []
        self._b_entries = []
        self.omega_entry = None
        self._current_fig = None
        self._current_table_data = None

        self._build()

    # ──────────────────────────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────────────────────────
    def _build(self):
        self._build_header()
        content = tk.Frame(self, bg=COLORS["bg"])
        content.pack(fill="both", expand=True, padx=12, pady=10)
        content.columnconfigure(0, minsize=320, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)
        self._build_left(content)
        self._build_right(content)

    def _build_header(self):
        bar = tk.Frame(self, bg=COLORS["primary"], height=52)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        _button(bar, "  Retour", bg=COLORS["primary_btn"],
                pad_x=14, pad_y=8, cmd=self._back).pack(side="left", padx=12, pady=8)
        _text(bar, "Axe 2 — Résolution des Systèmes Linéaires",
              size=14, bold=True, color="white", bg=COLORS["primary"]).pack(side="left", padx=8)

        self._pill_var = tk.StringVar(value="prêt")
        self._pill = tk.Label(bar, textvariable=self._pill_var,
                              bg=COLORS["primary_mid"], fg="white",
                              font=("Helvetica", 9), padx=12, pady=4)
        self._pill.pack(side="right", padx=14)

    def _set_pill(self, text, kind="idle"):
        pal = {"idle": COLORS["primary_mid"], "ok": COLORS["green"],
               "warn": COLORS["orange"], "error": COLORS["red"],
               "running": COLORS["primary_mid"]}
        self._pill.config(bg=pal.get(kind, COLORS["primary_mid"]))
        self._pill_var.set(text)

    # ──────────────────────────────────────────────────────────────
    # Left panel (scrollable)
    # ──────────────────────────────────────────────────────────────
    def _build_left(self, parent):
        container = tk.Frame(parent, bg=COLORS["bg"])
        container.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        lc = tk.Canvas(container, bg=COLORS["bg"], highlightthickness=0)
        vsb = tk.Scrollbar(container, orient="vertical", command=lc.yview)
        lc.configure(yscrollcommand=vsb.set)
        lc.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        outer = tk.Frame(lc, bg=COLORS["bg"])
        win = lc.create_window((0, 0), window=outer, anchor="nw")
        outer.bind("<Configure>", lambda e: lc.configure(scrollregion=lc.bbox("all")))
        lc.bind("<Configure>", lambda e: lc.itemconfig(win, width=e.width))
        lc.bind("<MouseWheel>", lambda e: lc.yview_scroll(int(-1*(e.delta/120)), "units"))

        self._build_left_content(outer)

    def _build_left_content(self, parent):
        # ── Inputs
        card, body, _ = _panel(parent, "  Entrées (Ax = b)", dot=COLORS["primary_mid"], compact=True)
        card.pack(fill="x", pady=(0, 6))

        _text(body, "Taille n×n (max 4) :", size=9, color=COLORS["text_label"]).pack(anchor="w")
        size_row = tk.Frame(body, bg=COLORS["surface"])
        size_row.pack(anchor="w", pady=(2, 6))
        self.size_var = tk.IntVar(value=3)
        for n in (2, 3, 4):
            tk.Radiobutton(size_row, text=f"{n}×{n}", variable=self.size_var, value=n,
                           bg=COLORS["surface"], fg=COLORS["text"],
                           selectcolor=COLORS["primary_mid"],
                           font=("Helvetica", 10),
                           command=self._rebuild_matrix).pack(side="left", padx=6)

        self.matrix_frame = tk.Frame(body, bg=COLORS["surface"])
        self.matrix_frame.pack(fill="x", pady=(2, 0))
        self._rebuild_matrix()

        # ── Iterative params
        card2, body2, _ = _panel(parent, "  Paramètres itératifs", dot=COLORS["orange"], compact=True)
        card2.pack(fill="x", pady=(0, 6))

        row = tk.Frame(body2, bg=COLORS["surface"])
        row.pack(fill="x", pady=2)
        _text(row, "Tolérance ε :", size=9, color=COLORS["text_label"], bg=COLORS["surface"]).pack(side="left")
        self.tol_entry = _field(row, "1e-6", width=10)
        self.tol_entry.pack(side="right")

        row = tk.Frame(body2, bg=COLORS["surface"])
        row.pack(fill="x", pady=2)
        _text(row, "Max itérations :", size=9, color=COLORS["text_label"], bg=COLORS["surface"]).pack(side="left")
        self.maxiter_entry = _field(row, "100", width=10)
        self.maxiter_entry.pack(side="right")

        self.extra_frame = tk.Frame(body2, bg=COLORS["surface"])
        self.extra_frame.pack(fill="x", pady=(4, 0))

        # ── Matrix operations
        card3, body3, _ = _panel(parent, "  Opérations sur A", dot=COLORS["green"], compact=True)
        card3.pack(fill="x", pady=(0, 6))

        for label, cmd in [
            ("Normes induites  (‖A‖₁, ‖A‖₂, ‖A‖∞)", self._show_norms),
            ("Déterminant det(A)", self._show_det),
            ("DDS / SPD", self._show_dds_spd),
            ("Rayon spectral ρ(A)", self._show_spectral_A),
        ]:
            _button(body3, label, bg=COLORS["primary_lt"], fg=COLORS["primary"],
                    pad_x=10, pad_y=5, size=10, cmd=cmd).pack(fill="x", pady=2)

        # ── Algorithms
        card4, body4, _ = _panel(parent, "  Algorithmes", dot=COLORS["primary_mid"], compact=True)
        card4.pack(fill="x", pady=(0, 6))

        self.algo_var = tk.StringVar(value="Gauss (pivot partiel)")

        _text(body4, "Méthodes directes :", size=9, bold=True,
              color=COLORS["text_label"], bg=COLORS["surface"]).pack(anchor="w", pady=(0, 3))
        for algo in ("Gauss (pivot partiel)", "Gauss (pivot total)",
                     "LU Decomposition", "Cholesky"):
            tk.Radiobutton(body4, text=algo, variable=self.algo_var, value=algo,
                           bg=COLORS["surface"], fg=COLORS["text"],
                           selectcolor=COLORS["primary_mid"],
                           font=("Helvetica", 10),
                           command=self._update_extra).pack(anchor="w", pady=1)

        _line(body4).pack(fill="x", pady=6)

        _text(body4, "Méthodes itératives :", size=9, bold=True,
              color=COLORS["text_label"], bg=COLORS["surface"]).pack(anchor="w", pady=(0, 3))
        for algo in ("Jacobi", "Gauss-Seidel", "Relaxation"):
            tk.Radiobutton(body4, text=algo, variable=self.algo_var, value=algo,
                           bg=COLORS["surface"], fg=COLORS["text"],
                           selectcolor=COLORS["primary_mid"],
                           font=("Helvetica", 10),
                           command=self._update_extra).pack(anchor="w", pady=1)

        run_frame = tk.Frame(parent, bg=COLORS["bg"])
        run_frame.pack(fill="x", pady=(0, 6))
        _button(run_frame, "  ▶  Lancer", bg=COLORS["primary_mid"], bold=True,
                size=12, pad_y=8, cmd=self._run_algorithm).pack(fill="x")

        self._update_extra()

    def _rebuild_matrix(self):
        for w in self.matrix_frame.winfo_children():
            w.destroy()
        self._matrix_entries.clear()
        self._b_entries.clear()

        n = self.size_var.get()
        tk.Label(self.matrix_frame, text="A", bg=COLORS["surface"],
                 fg=COLORS["text_muted"], font=("Helvetica", 9, "bold")).grid(
            row=0, column=0, columnspan=n, sticky="ew")
        tk.Label(self.matrix_frame, text="b", bg=COLORS["surface"],
                 fg=COLORS["text_muted"], font=("Helvetica", 9, "bold")).grid(
            row=0, column=n+1, padx=(6, 0), sticky="ew")

        for i in range(n):
            row_entries = []
            for j in range(n):
                e = _field(self.matrix_frame, "0", width=6)
                e.grid(row=i+1, column=j, padx=2, pady=2)
                row_entries.append(e)
            self._matrix_entries.append(row_entries)
            tk.Label(self.matrix_frame, text="|", bg=COLORS["surface"],
                     fg=COLORS["text_muted"]).grid(row=i+1, column=n, padx=4)
            b = _field(self.matrix_frame, "0", width=6)
            b.grid(row=i+1, column=n+1, padx=2, pady=2)
            self._b_entries.append(b)

    def _update_extra(self):
        for w in self.extra_frame.winfo_children():
            w.destroy()
        self.omega_entry = None
        if self.algo_var.get() == "Relaxation":
            row = tk.Frame(self.extra_frame, bg=COLORS["surface"])
            row.pack(fill="x", pady=2)
            _text(row, "ω (0 < ω < 2) :", size=9,
                  color=COLORS["text_label"], bg=COLORS["surface"]).pack(side="left")
            self.omega_entry = _field(row, "1.25", width=10)
            self.omega_entry.pack(side="right")

    # ──────────────────────────────────────────────────────────────
    # Right panel
    # ──────────────────────────────────────────────────────────────
    def _build_right(self, parent):
        outer = tk.Frame(parent, bg=COLORS["bg"])
        outer.grid(row=0, column=1, sticky="nsew")
        outer.rowconfigure(0, weight=3)
        outer.rowconfigure(1, weight=2)
        outer.columnconfigure(0, weight=1)

        # Plot card
        plot_card = tk.Frame(outer, bg=COLORS["border"])
        plot_card.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        inner_p = tk.Frame(plot_card, bg=COLORS["surface"])
        inner_p.pack(fill="both", expand=True, padx=1, pady=1)
        inner_p.rowconfigure(2, weight=1)
        inner_p.columnconfigure(0, weight=1)

        phdr = tk.Frame(inner_p, bg=COLORS["hdr_bg"], height=30)
        phdr.grid(row=0, column=0, sticky="ew")
        phdr.grid_propagate(False)
        _text(phdr, "Visualisation", size=10, bold=True, bg=COLORS["hdr_bg"]).pack(
            side="left", padx=10, pady=6)

        dl = tk.Frame(phdr, bg=COLORS["hdr_bg"])
        dl.pack(side="right", padx=8)
        _button(dl, "  PNG/PDF", bg=COLORS["primary_mid"], pad_x=10, pad_y=3,
                size=9, cmd=self._download_graph).pack(side="right", padx=3)
        _button(dl, "  CSV", bg=COLORS["purple"], pad_x=10, pad_y=3,
                size=9, cmd=self._download_table).pack(side="right", padx=3)

        _line(inner_p).grid(row=1, column=0, sticky="ew")

        self.plot_frame = tk.Frame(inner_p, bg=COLORS["surface"])
        self.plot_frame.grid(row=2, column=0, sticky="nsew")

        # Result text card
        res_card = tk.Frame(outer, bg=COLORS["border"])
        res_card.grid(row=1, column=0, sticky="nsew")
        inner_r = tk.Frame(res_card, bg=COLORS["surface"])
        inner_r.pack(fill="both", expand=True, padx=1, pady=1)
        inner_r.rowconfigure(2, weight=1)
        inner_r.columnconfigure(0, weight=1)

        rhdr = tk.Frame(inner_r, bg=COLORS["hdr_bg"], height=30)
        rhdr.grid(row=0, column=0, sticky="ew")
        rhdr.grid_propagate(False)
        _text(rhdr, "Résultat & Étapes", size=10, bold=True, bg=COLORS["hdr_bg"]).pack(
            side="left", padx=10, pady=6)

        _line(inner_r).grid(row=1, column=0, sticky="ew")

        self.result_text = tk.Text(inner_r, height=10, font=("Courier", 9),
                                   bg="#f8fbff", fg=COLORS["text"],
                                   relief="flat", padx=10, pady=8, wrap="word")
        self.result_text.grid(row=2, column=0, sticky="nsew")
        sb = tk.Scrollbar(inner_r, command=self.result_text.yview)
        sb.grid(row=2, column=1, sticky="ns")
        self.result_text.configure(yscrollcommand=sb.set)

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────
    def _read_matrix(self):
        n = self.size_var.get()
        try:
            A = [[float(self._matrix_entries[i][j].get().strip()) for j in range(n)]
                 for i in range(n)]
            b = [float(self._b_entries[i].get().strip()) for i in range(n)]
        except Exception:
            raise ValueError("Entrées invalides : A et b doivent être numériques.")
        return np.array(A, dtype=float), np.array(b, dtype=float)

    def _set_result(self, text):
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)

    def _clear_plot(self):
        for w in self.plot_frame.winfo_children():
            w.destroy()
        self._current_fig = None
        self._current_table_data = None

    def _embed_figure(self, fig):
        self._current_fig = fig
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ──────────────────────────────────────────────────────────────
    # Matrix operations
    # ──────────────────────────────────────────────────────────────
    def _show_norms(self):
        try:
            A, _ = self._read_matrix()
            self._set_result(
                "Normes induites :\n"
                f"  ‖A‖₁  = {induced_matrix_norm(A,1):.6f}\n"
                f"  ‖A‖₂  = {induced_matrix_norm(A,2):.6f}\n"
                f"  ‖A‖∞  = {induced_matrix_norm(A,np.inf):.6f}"
            )
            self._set_pill("ok", "ok")
        except Exception as e:
            messagebox.showerror("Erreur", str(e)); self._set_pill("erreur", "error")

    def _show_det(self):
        try:
            A, _ = self._read_matrix()
            det = float(np.linalg.det(A))
            msg = f"det(A) = {det:.6f}\n"
            msg += "✔ det(A) ≠ 0 → solution unique" if abs(det) > 1e-12 else "✘ det(A) ≈ 0 → pas de solution unique"
            self._set_result(msg)
            self._set_pill("ok", "ok")
        except Exception as e:
            messagebox.showerror("Erreur", str(e)); self._set_pill("erreur", "error")

    def _show_dds_spd(self):
        try:
            A, _ = self._read_matrix()
            dds = is_strictly_diagonally_dominant(A)
            spd = is_symmetric_positive_definite(A)
            self._set_result(
                "Conditions :\n"
                f"  DDS (dominance diagonale stricte) : {dds}\n"
                f"  SPD (symétrique définie positive)  : {spd}\n\n"
                "Notes :\n"
                "  • SPD requis pour Cholesky.\n"
                "  • DDS est une condition suffisante pour la convergence\n"
                "    de Jacobi et Gauss-Seidel."
            )
            self._set_pill("ok", "ok")
        except Exception as e:
            messagebox.showerror("Erreur", str(e)); self._set_pill("erreur", "error")

    def _show_spectral_A(self):
        try:
            A, _ = self._read_matrix()
            rho = spectral_radius(A)
            self._set_result(f"Rayon spectral :\n  ρ(A) = {rho:.6f}")
            self._set_pill("ok", "ok")
        except Exception as e:
            messagebox.showerror("Erreur", str(e)); self._set_pill("erreur", "error")

    # ──────────────────────────────────────────────────────────────
    # Run algorithm
    # ──────────────────────────────────────────────────────────────
    def _run_algorithm(self):
        self._set_pill("calcul...", "running")
        self.update()
        try:
            A, b   = self._read_matrix()
            algo   = self.algo_var.get()
            tol    = float(self.tol_entry.get())
            max_it = int(self.maxiter_entry.get())
            n      = len(b)

            self._clear_plot()

            is_iter = algo in ("Jacobi", "Gauss-Seidel", "Relaxation")

            if is_iter and not is_strictly_diagonally_dominant(A):
                if not messagebox.askyesno("Avertissement",
                    "La matrice n'est pas DDS.\nLa convergence n'est pas garantie.\n\nContinuer ?"):
                    self._set_pill("annulé", "warn"); return

            omega = 1.25
            if algo == "Relaxation":
                omega = float(self.omega_entry.get()) if self.omega_entry else 1.25
                if not (0 < omega < 2):
                    messagebox.showerror("Erreur", "ω doit vérifier 0 < ω < 2.")
                    self._set_pill("erreur", "error"); return

            # ── Gauss pivot partiel ────────────────────────────
            if algo == "Gauss (pivot partiel)":
                x, steps, U = gauss_partial_steps(A, b)
                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))

                out = "═══ Gauss — Pivot Partiel ═══\n\n"
                out += "Matrice initiale [A|b] :\n"
                out += format_matrix_str(np.hstack([A, b.reshape(-1,1)]), precision=4) + "\n"

                for s in steps:
                    out += f"\n── Étape {s['step']} (pivot = {s['pivot']:+.4f}) ──\n"
                    out += format_matrix_str(
                        np.hstack([s['A'], s['b'].reshape(-1,1)]), precision=4) + "\n"

                out += f"\nMatrice triangulaire U finale :\n"
                out += format_matrix_str(U, precision=4) + "\n"
                out += "\nSolution x* :\n"
                out += "\n".join(f"  x{i+1} = {x[i]:.8f}" for i in range(n))
                out += f"\n\n‖Ax - b‖∞ = {res_inf:.2e}"

                self._set_result(out)
                self._plot_gauss_steps(steps, U, "Gauss — Pivot Partiel")
                self._current_table_data = (
                    ["i", "j", "U_ij"],
                    [[i, j, float(U[i,j])] for i in range(n) for j in range(n)]
                )
                self._set_pill("ok", "ok")

            # ── Gauss pivot total ──────────────────────────────
            elif algo == "Gauss (pivot total)":
                x, steps, U, perm = gauss_total_steps(A, b)
                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))

                out = "═══ Gauss — Pivot Total ═══\n\n"
                out += "Matrice initiale [A|b] :\n"
                out += format_matrix_str(np.hstack([A, b.reshape(-1,1)]), precision=4) + "\n"

                for s in steps:
                    out += f"\n── Étape {s['step']} (pivot = {s['pivot']:+.4f}) ──\n"
                    out += format_matrix_str(
                        np.hstack([s['A'], s['b'].reshape(-1,1)]), precision=4) + "\n"

                out += f"\nPermutation des colonnes : {perm}\n"
                out += f"\nMatrice triangulaire U finale :\n"
                out += format_matrix_str(U, precision=4) + "\n"
                out += "\nSolution x* :\n"
                out += "\n".join(f"  x{i+1} = {x[i]:.8f}" for i in range(n))
                out += f"\n\n‖Ax - b‖∞ = {res_inf:.2e}"

                self._set_result(out)
                self._plot_gauss_steps(steps, U, "Gauss — Pivot Total")
                self._current_table_data = (
                    ["i", "j", "U_ij"],
                    [[i, j, float(U[i,j])] for i in range(n) for j in range(n)]
                )
                self._set_pill("ok", "ok")

            # ── LU Decomposition ───────────────────────────────
            elif algo == "LU Decomposition":
                try:
                    L, U, steps = lu_steps(A)
                except ValueError as ve:
                    messagebox.showerror("Erreur LU", str(ve))
                    self._set_pill("erreur", "error"); return

                # Solve via forward/back substitution
                y = np.zeros(n)
                for i in range(n):
                    y[i] = (b[i] - np.dot(L[i,:i], y[:i])) / L[i,i]
                x = np.zeros(n)
                for i in range(n-1, -1, -1):
                    x[i] = (y[i] - np.dot(U[i,i+1:], x[i+1:])) / U[i,i]

                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))

                out = "═══ Décomposition LU (sans pivotage) ═══\n"
                out += "Convention : A = L · U\n\n"

                for s in steps:
                    out += f"── Étape {s['step']} (pivot = {s['pivot']:+.4f}) ──\n"
                    mults = ", ".join(
                        f"m{i+1}{s['step']} = {v:+.4f}"
                        for i, v in s['multipliers'].items()
                    )
                    out += f"  Multiplicateurs : {mults}\n"
                    out += "  U courant :\n"
                    out += format_matrix_str(s['U_current'], precision=4) + "\n\n"

                out += "Matrice L (triangulaire inférieure, 1 sur la diagonale) :\n"
                out += format_matrix_str(L, precision=4) + "\n\n"
                out += "Matrice U (triangulaire supérieure) :\n"
                out += format_matrix_str(U, precision=4) + "\n\n"
                out += "Résolution LY = b  →  Y :\n"
                out += "  " + "  ".join(f"{v:+.6f}" for v in y) + "\n\n"
                out += "Résolution UX = Y  →  X* :\n"
                out += "\n".join(f"  x{i+1} = {x[i]:.8f}" for i in range(n))
                out += f"\n\n‖Ax - b‖∞ = {res_inf:.2e}"

                self._set_result(out)
                self._plot_lu_steps(steps, L, U, "LU Décomposition")
                cols = ["mat", "i", "j", "val"]
                rows = (
                    [["L", i, j, float(L[i,j])] for i in range(n) for j in range(n)] +
                    [["U", i, j, float(U[i,j])] for i in range(n) for j in range(n)]
                )
                self._current_table_data = (cols, rows)
                self._set_pill("ok", "ok")

            # ── Cholesky ───────────────────────────────────────
            elif algo == "Cholesky":
                if not is_symmetric_positive_definite(A):
                    messagebox.showerror("Matrice invalide",
                        "Cholesky requiert une matrice SPD.")
                    self._set_pill("erreur", "error"); return

                x, L = solve_cholesky(A, b)
                R = L.T   # A = R^T · R  (R upper triangular)
                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))

                # Show how R is built element by element (course formula)
                out = "═══ Cholesky ═══\n"
                out += "Convention : A = Rᵀ · R  (R triangulaire supérieure)\n\n"
                out += "Calcul de R :\n"
                for i in range(n):
                    for j in range(i, n):
                        if i == j:
                            s = A[i,i] - sum(R[k,i]**2 for k in range(i))
                            out += f"  R[{i+1},{j+1}] = sqrt({A[i,i]:.4f} - Σ R[k,{i+1}]²) = {R[i,j]:.6f}\n"
                        else:
                            s = (A[i,j] - sum(R[k,i]*R[k,j] for k in range(i))) / R[i,i]
                            out += f"  R[{i+1},{j+1}] = ({A[i,j]:.4f} - Σ ...) / R[{i+1},{i+1}] = {R[i,j]:.6f}\n"
                out += "\nMatrice R (triangulaire supérieure) :\n"
                out += format_matrix_str(R, precision=4) + "\n\n"
                out += "Solution x* :\n"
                out += "\n".join(f"  x{i+1} = {x[i]:.8f}" for i in range(n))
                out += f"\n\n‖Ax - b‖∞ = {res_inf:.2e}"

                self._set_result(out)
                self._plot_matrix_heatmap(R, "Cholesky — Matrice R", cmap="Greens")
                self._current_table_data = (
                    ["i", "j", "R_ij"],
                    [[i, j, float(R[i,j])] for i in range(n) for j in range(n)]
                )
                self._set_pill("ok", "ok")

            # ── Iterative methods ──────────────────────────────
            elif algo == "Jacobi":
                x0 = np.zeros(n)
                _, rho_B, hist = solve_iteratif(A, b, x0, tol, methode="jacobi", max_iter=max_it)
                history = [(k, xk, float(err), float(np.linalg.norm(A@xk-b, ord=np.inf)))
                           for k, xk, err in hist]
                x = hist[-1][1] if hist else x0
                res_inf = float(np.linalg.norm(A@x-b, ord=np.inf))
                self._set_result(
                    "Jacobi\n\n"
                    + "\n".join(f"  x{i+1} = {x[i]:.8f}" for i in range(n))
                    + f"\n\nρ(B_J) = {float(rho_B):.6f}\n‖Ax-b‖∞ = {res_inf:.2e}"
                )
                self._plot_iterative(history, "Jacobi")
                self._set_pill("ok", "ok")

            elif algo == "Gauss-Seidel":
                x0 = np.zeros(n)
                x, hist = gauss_seidel(A, b, x0=x0, tol=tol, max_iter=max_it)
                history = [(k, xk, float(err), float(np.linalg.norm(A@xk-b, ord=np.inf)))
                           for k, xk, err in hist]
                res_inf = float(np.linalg.norm(A@x-b, ord=np.inf))
                self._set_result(
                    "Gauss-Seidel\n\n"
                    + "\n".join(f"  x{i+1} = {x[i]:.8f}" for i in range(n))
                    + f"\n\n‖Ax-b‖∞ = {res_inf:.2e}"
                )
                self._plot_iterative(history, "Gauss-Seidel")
                self._set_pill("ok", "ok")

            elif algo == "Relaxation":
                x0 = np.zeros(n)
                x, hist = self._relaxation(A, b, x0, omega, tol, max_it)
                history = [(k, xk, float(err), float(np.linalg.norm(A@xk-b, ord=np.inf)))
                           for k, xk, err in hist]
                res_inf = float(np.linalg.norm(A@x-b, ord=np.inf))
                self._set_result(
                    f"SOR (Relaxation)  ω = {omega}\n\n"
                    + "\n".join(f"  x{i+1} = {x[i]:.8f}" for i in range(n))
                    + f"\n\n‖Ax-b‖∞ = {res_inf:.2e}"
                )
                self._plot_iterative(history, "SOR")
                self._set_pill("ok", "ok")

        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            self._set_pill("erreur", "error")

    # ──────────────────────────────────────────────────────────────
    # SOR implementation
    # ──────────────────────────────────────────────────────────────
    def _relaxation(self, A, b, x0, omega, tol, max_iter):
        A = np.array(A, dtype=float); b = np.array(b, dtype=float)
        n = len(b); x = np.array(x0, dtype=float).copy()
        history = []
        for k in range(max_iter):
            x_old = x.copy()
            for i in range(n):
                if abs(A[i,i]) < 1e-15:
                    raise ValueError("SOR: pivot nul sur la diagonale.")
                s = b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,i+1:], x_old[i+1:])
                x[i] = (1 - omega) * x_old[i] + omega * s / A[i,i]
            err = float(np.linalg.norm(x - x_old, ord=np.inf))
            history.append((k+1, x.copy(), err))
            if err < tol: break
        return x, history

    # ──────────────────────────────────────────────────────────────
    # Plotting
    # ──────────────────────────────────────────────────────────────
    def _plot_gauss_steps(self, steps, U, title):
        """
        Show the evolution of the matrix through Gauss steps.
        One heatmap per step + final U.
        """
        self._clear_plot()
        n_steps = len(steps)
        cols = min(n_steps + 1, 4)
        rows = (n_steps + 1 + cols - 1) // cols

        fig = Figure(figsize=(max(9, cols * 3), rows * 3 + 0.5), dpi=100)
        fig.suptitle(title, fontsize=11, fontweight="bold")

        all_mats = [s['A'] for s in steps] + [U]
        all_titles = [f"A⁽{s['step']}⁾  (pivot={s['pivot']:+.3f})" for s in steps] + ["U finale"]

        vmax = max(np.abs(m).max() for m in all_mats) or 1.0

        for idx, (mat, ttl) in enumerate(zip(all_mats, all_titles)):
            ax = fig.add_subplot(rows, cols, idx + 1)
            im = ax.imshow(np.abs(mat), cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
            ax.set_title(ttl, fontsize=8)
            ax.set_xlabel("j", fontsize=7)
            ax.set_ylabel("i", fontsize=7)
            ax.tick_params(labelsize=7)

        fig.tight_layout()
        self._embed_figure(fig)

    def _plot_lu_steps(self, steps, L, U, title):
        """
        Show L being filled step by step alongside U progression.
        """
        self._clear_plot()
        n_steps = len(steps)
        fig = Figure(figsize=(10, max(4, 3 * ((n_steps + 2 + 1) // 2))), dpi=100)
        fig.suptitle(title, fontsize=11, fontweight="bold")

        total = n_steps * 2 + 2
        cols = 4
        rows = (total + cols - 1) // cols

        idx = 1
        for s in steps:
            ax = fig.add_subplot(rows, cols, idx)
            im = ax.imshow(np.abs(s['L_current']), cmap="Greens", aspect="auto")
            ax.set_title(f"L étape {s['step']}", fontsize=8)
            ax.tick_params(labelsize=7)
            idx += 1

            ax = fig.add_subplot(rows, cols, idx)
            im = ax.imshow(np.abs(s['U_current']), cmap="Blues", aspect="auto")
            ax.set_title(f"U étape {s['step']}", fontsize=8)
            ax.tick_params(labelsize=7)
            idx += 1

        # Final L and U
        ax = fig.add_subplot(rows, cols, idx)
        ax.imshow(np.abs(L), cmap="Greens", aspect="auto")
        ax.set_title("L finale", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

        ax = fig.add_subplot(rows, cols, idx + 1)
        ax.imshow(np.abs(U), cmap="Blues", aspect="auto")
        ax.set_title("U finale", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

        fig.tight_layout()
        self._embed_figure(fig)

    def _plot_matrix_heatmap(self, M, title, cmap="Blues"):
        self._clear_plot()
        fig = Figure(figsize=(8.8, 4.8), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(np.abs(np.array(M, dtype=float)), cmap=cmap, aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("j"); ax.set_ylabel("i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        self._embed_figure(fig)

    def _plot_iterative(self, history, title):
        self._clear_plot()
        ks   = [h[0] for h in history]
        errs = [h[2] for h in history]
        ress = [h[3] for h in history]

        n_vars = len(history[0][1])
        cols = ["k"] + [f"x{i+1}" for i in range(n_vars)] + ["‖xk+1-xk‖∞", "‖Ax-b‖∞"]
        rows = []
        for k, xk, err, res in history:
            rows.append([k] + [float(v) for v in xk] + [err, res])
        self._current_table_data = (cols, rows)

        fig = Figure(figsize=(9.0, 5.2), dpi=100)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.35)

        ax = fig.add_subplot(gs[0])
        ax.semilogy(ks, errs, "b-o", markersize=3, label="‖xk+1-xk‖∞")
        ax.semilogy(ks, ress, "r--s", markersize=3, label="‖Axk-b‖∞")
        ax.set_title(f"{title} — Convergence", fontsize=11, fontweight="bold")
        ax.set_xlabel("Itération k"); ax.grid(True, alpha=0.25); ax.legend(fontsize=9)

        ax_tbl = fig.add_subplot(gs[1])
        ax_tbl.axis("off")
        display = []
        for r in rows[:12]:
            display.append(
                [str(r[0])] + [f"{v:.6f}" for v in r[1:1+n_vars]] +
                [f"{r[-2]:.2e}", f"{r[-1]:.2e}"]
            )
        tbl = ax_tbl.table(cellText=display, colLabels=cols,
                           cellLoc="center", loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.05, 1.6)
        fig.tight_layout()
        self._embed_figure(fig)

    # ──────────────────────────────────────────────────────────────
    # Downloads
    # ──────────────────────────────────────────────────────────────
    def _download_graph(self):
        if self._current_fig is None:
            messagebox.showinfo("Info", "Lancez un algorithme d'abord."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")],
            title="Sauvegarder la figure")
        if path:
            self._current_fig.savefig(path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("OK", f"Figure sauvegardée :\n{path}")

    def _download_table(self):
        if self._current_table_data is None:
            messagebox.showinfo("Info", "Aucun tableau à exporter."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            title="Exporter le tableau")
        if not path: return
        cols, rows = self._current_table_data
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(cols)
            for r in rows: w.writerow(r)
        messagebox.showinfo("OK", f"Tableau exporté :\n{path}")

    # ──────────────────────────────────────────────────────────────
    # Navigation
    # ──────────────────────────────────────────────────────────────
    def _back(self):
        import subprocess
        path = os.path.join(os.path.dirname(__file__), "main_screen.py")
        subprocess.Popen([sys.executable, path])
        self.destroy()


if __name__ == "__main__":
    Axe2Screen().mainloop()
