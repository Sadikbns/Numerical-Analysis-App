import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
import sys
import os
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

_HERE = os.path.dirname(os.path.abspath(__file__))

# Ensure we can import chap2/chap3 modules even if folder name contains spaces
MODULES_DIR = os.path.join(_HERE, "modules", "chap 2 and 3")
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

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

<<<<<<< HEAD

class Axe2Screen(tk.Frame):
    """
    Axe 2 — Linear Systems (fully functional)
    Layout mirrors Axe 3:
      Left panel  : user inputs (matrix up to 4×4) + algorithm selector
      Right panel : result text box + matplotlib figure (plot + table inside figure)
    """
=======
# ──────────────────────────────────────────────────────────────────────────────
# Step-by-step helpers (for visualization + pedagogical output)
# ──────────────────────────────────────────────────────────────────────────────
def gauss_partial_steps(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    M = np.hstack([A.copy(), b.reshape(-1, 1)])
    steps = []

    for k in range(n - 1):
        pivot_row = k + int(np.argmax(np.abs(M[k:, k])))
        if abs(M[pivot_row, k]) < 1e-15:
            raise ValueError(f"Matrice singulière (pivot nul à l'étape {k+1}).")
        if pivot_row != k:
            M[[k, pivot_row]] = M[[pivot_row, k]]
        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k:] -= factor * M[k, k:]
        steps.append({"step": k + 1, "pivot": float(M[k, k]), "A": M[:, :n].copy(), "b": M[:, n].copy()})

    U = M[:, :n].copy()
    b_mod = M[:, n].copy()
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b_mod[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]
    return x, steps, U


def gauss_total_steps(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    M = np.hstack([A.copy(), b.reshape(-1, 1)])
    perm = list(range(n))
    steps = []

    for k in range(n - 1):
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
        steps.append({"step": k + 1, "pivot": float(M[k, k]), "A": M[:, :n].copy(), "b": M[:, n].copy()})

    U = M[:, :n].copy()
    b_mod = M[:, n].copy()
    x_perm = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_perm[i] = (b_mod[i] - np.dot(U[i, i + 1 :], x_perm[i + 1 :])) / U[i, i]
    x = np.zeros(n)
    for i, p in enumerate(perm):
        x[p] = x_perm[i]
    return x, steps, U, perm


def lu_steps(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    steps = []

    for k in range(n - 1):
        if abs(U[k, k]) < 1e-15:
            raise ValueError(
                f"LU sans pivot: pivot nul à l'étape {k+1}.\n" "Utilisez Gauss avec pivotage."
            )
        multipliers = {}
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            multipliers[i] = m
            U[i, k:] -= m * U[k, k:]
        steps.append(
            {"step": k + 1, "pivot": float(U[k, k]), "multipliers": multipliers, "U_current": U.copy(), "L_current": L.copy()}
        )
    return L, U, steps


def format_matrix_str(M, name="", precision=4):
    M = np.array(M, dtype=float)
    rows_out = []
    for i in range(M.shape[0]):
        row_str = "  [ " + "  ".join(f"{M[i, j]:+.{precision}f}" for j in range(M.shape[1])) + " ]"
        rows_out.append(row_str)
    header = f"{name} =\n" if name else ""
    return header + "\n".join(rows_out)


# ──────────────────────────────────────────────────────────────────────────────
# Design tokens — BLUE theme for Axe 2 (consistent with Axe 1 green & Axe 3 purple)
# ──────────────────────────────────────────────────────────────────────────────
COLORS = {
    "bg": "#f0f4f8",
    "surface": "#ffffff",
    "border": "#d0d7e3",
    "primary": "#1a6499",
    "primary_mid": "#2980b9",
    "primary_btn": "#1a6499",
    "primary_lt": "#eaf4fb",
    "hdr_bg": "#f5fbff",
    "text": "#2c3e50",
    "text_label": "#34495e",
    "text_muted": "#7f8c8d",
    "purple": "#8e44ad",
    "orange": "#e67e22",
    "green": "#27ae60",
    "red": "#e74c3c",
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


def _button(parent, label, bg=None, fg="white", bold=False, size=10, pad_x=10, pad_y=6, cmd=None):
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
    _text(hdr, title, size=10, bold=True, bg=COLORS["hdr_bg"]).pack(side="left", padx=10, pady=6)
    _line(inner).pack(fill="x")
    body = tk.Frame(inner, bg=COLORS["surface"], padx=8 if compact else 12, pady=6 if compact else 10)
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
>>>>>>> e01eae2 (Axe2 - changed matrices shown in cholesky)

    def __init__(self, parent, back_callback=None):
        super().__init__(parent, bg="#f0f4f8")
        self._back_callback = back_callback
        self._matrix_entries = []
        self._b_entries = []
        self._current_fig = None
        self._current_table_data = None
<<<<<<< HEAD
        self.omega_entry = None
        self._build()
=======

        self._last_A_orig = None

        self._viz_canvas = None
        self.plot_frame = None
        self._viz_win = None

        self._iter_area = None
        self.table_tree = None
        self.answers_text = None

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
        _button(bar, "  Retour", bg=COLORS["primary_btn"], pad_x=14, pad_y=8, cmd=self._back).pack(
            side="left", padx=12, pady=8
        )
        _text(
            bar,
            "Axe 2 — Résolution des Systèmes Linéaires",
            size=14,
            bold=True,
            color="white",
            bg=COLORS["primary"],
        ).pack(side="left", padx=8)
        self._pill_var = tk.StringVar(value="prêt")
        self._pill = tk.Label(
            bar,
            textvariable=self._pill_var,
            bg=COLORS["primary_mid"],
            fg="white",
            font=("Helvetica", 9),
            padx=12,
            pady=4,
        )
        self._pill.pack(side="right", padx=14)

    def _set_pill(self, text, kind="idle"):
        pal = {
            "idle": COLORS["primary_mid"],
            "ok": COLORS["green"],
            "warn": COLORS["orange"],
            "error": COLORS["red"],
            "running": COLORS["primary_mid"],
        }
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
        lc.bind("<MouseWheel>", lambda e: lc.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        self._build_left_content(outer)

    def _build_left_content(self, parent):
        card, body, _ = _panel(parent, "  Entrées (Ax = b)", dot=COLORS["primary_mid"], compact=True)
        card.pack(fill="x", pady=(0, 6))

        _text(body, "Taille n×n (max 4) :", size=9, color=COLORS["text_label"]).pack(anchor="w")
        size_row = tk.Frame(body, bg=COLORS["surface"])
        size_row.pack(anchor="w", pady=(2, 6))
        self.size_var = tk.IntVar(value=3)
        for n in (2, 3, 4):
            tk.Radiobutton(
                size_row,
                text=f"{n}×{n}",
                variable=self.size_var,
                value=n,
                bg=COLORS["surface"],
                fg=COLORS["text"],
                selectcolor=COLORS["primary_mid"],
                font=("Helvetica", 10),
                command=self._rebuild_matrix,
            ).pack(side="left", padx=6)

        self.matrix_frame = tk.Frame(body, bg=COLORS["surface"])
        self.matrix_frame.pack(fill="x", pady=(2, 0))
        self._rebuild_matrix()

        card2, body2, _ = _panel(parent, "  Paramètres itératifs", dot=COLORS["orange"], compact=True)
        card2.pack(fill="x", pady=(0, 6))

        for lbl, attr, default in [("Tolérance ε :", "tol_entry", "1e-6"), ("Max itérations :", "maxiter_entry", "100")]:
            row = tk.Frame(body2, bg=COLORS["surface"])
            row.pack(fill="x", pady=2)
            _text(row, lbl, size=9, color=COLORS["text_label"], bg=COLORS["surface"]).pack(side="left")
            e = _field(row, default, width=10)
            e.pack(side="right")
            setattr(self, attr, e)

        self.extra_frame = tk.Frame(body2, bg=COLORS["surface"])
        self.extra_frame.pack(fill="x", pady=(4, 0))

        card3, body3, _ = _panel(parent, "  Opérations sur A", dot=COLORS["green"], compact=True)
        card3.pack(fill="x", pady=(0, 6))

        for label, cmd in [
            ("Normes induites  (‖A‖₁, ‖A‖₂, ‖A‖∞)", self._show_norms),
            ("Déterminant det(A)", self._show_det),
            ("DDS / SPD", self._show_dds_spd),
            ("Rayon spectral ρ(A)", self._show_spectral_A),
        ]:
            _button(
                body3,
                label,
                bg=COLORS["primary_lt"],
                fg=COLORS["primary"],
                pad_x=10,
                pad_y=5,
                size=10,
                cmd=cmd,
            ).pack(fill="x", pady=2)

        card4, body4, _ = _panel(parent, "  Algorithmes", dot=COLORS["primary_mid"], compact=True)
        card4.pack(fill="x", pady=(0, 6))

        self.algo_var = tk.StringVar(value="Gauss (pivot partiel)")

        _text(body4, "Méthodes directes :", size=9, bold=True, color=COLORS["text_label"], bg=COLORS["surface"]).pack(
            anchor="w", pady=(0, 3)
        )
        for algo in ("Gauss (pivot partiel)", "Gauss (pivot total)", "LU Decomposition", "Cholesky"):
            tk.Radiobutton(
                body4,
                text=algo,
                variable=self.algo_var,
                value=algo,
                bg=COLORS["surface"],
                fg=COLORS["text"],
                selectcolor=COLORS["primary_mid"],
                font=("Helvetica", 10),
                command=self._update_extra,
            ).pack(anchor="w", pady=1)

        _line(body4).pack(fill="x", pady=6)

        _text(body4, "Méthodes itératives :", size=9, bold=True, color=COLORS["text_label"], bg=COLORS["surface"]).pack(
            anchor="w", pady=(0, 3)
        )
        for algo in ("Jacobi", "Gauss-Seidel", "Relaxation"):
            tk.Radiobutton(
                body4,
                text=algo,
                variable=self.algo_var,
                value=algo,
                bg=COLORS["surface"],
                fg=COLORS["text"],
                selectcolor=COLORS["primary_mid"],
                font=("Helvetica", 10),
                command=self._update_extra,
            ).pack(anchor="w", pady=1)

        run_frame = tk.Frame(parent, bg=COLORS["bg"])
        run_frame.pack(fill="x", pady=(0, 6))
        _button(
            run_frame,
            "  ▶  Lancer",
            bg=COLORS["primary_mid"],
            bold=True,
            size=12,
            pad_y=8,
            cmd=self._run_algorithm,
        ).pack(fill="x")

>>>>>>> e01eae2 (Axe2 - changed matrices shown in cholesky)
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

<<<<<<< HEAD
    # ── Right panel — mirrors axe3 structure exactly ──────────────
    def _right_panel(self, parent):
        frame = tk.Frame(parent, bg="#f0f4f8")
        frame.grid(row=0, column=1, sticky="nsew")

        self.result_frame = tk.LabelFrame(frame, text="Résultats", bg="#f0f4f8",
                                          fg="#2c3e50",
                                          font=("Helvetica", 11, "bold"))
        self.result_frame.pack(fill="both", expand=True, padx=8, pady=8)
=======
    # ──────────────────────────────────────────────────────────────
    # Right panel
    # ──────────────────────────────────────────────────────────────
    def _build_right(self, parent):
        outer = tk.Frame(parent, bg=COLORS["bg"])
        outer.grid(row=0, column=1, sticky="nsew")
        outer.rowconfigure(0, weight=3)
        outer.rowconfigure(1, weight=2)
        outer.columnconfigure(0, weight=1)

        viz_card = tk.Frame(outer, bg=COLORS["border"])
        viz_card.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        inner_v = tk.Frame(viz_card, bg=COLORS["surface"])
        inner_v.pack(fill="both", expand=True, padx=1, pady=1)
        inner_v.rowconfigure(2, weight=1)
        inner_v.columnconfigure(0, weight=1)
>>>>>>> e01eae2 (Axe2 - changed matrices shown in cholesky)

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

<<<<<<< HEAD
    # ── Helpers ───────────────────────────────────────────────────
=======
        _line(inner_v).grid(row=1, column=0, sticky="ew")

        self._viz_canvas = tk.Canvas(inner_v, bg=COLORS["surface"], highlightthickness=0)
        viz_vsb = tk.Scrollbar(inner_v, orient="vertical", command=self._viz_canvas.yview)
        self._viz_canvas.configure(yscrollcommand=viz_vsb.set)
        self._viz_canvas.grid(row=2, column=0, sticky="nsew")
        viz_vsb.grid(row=2, column=1, sticky="ns")

        self.plot_frame = tk.Frame(self._viz_canvas, bg=COLORS["surface"])
        self._viz_win = self._viz_canvas.create_window((0, 0), window=self.plot_frame, anchor="nw")
        self.plot_frame.bind("<Configure>", self._on_plot_frame_configure)
        self._viz_canvas.bind("<Configure>", self._on_viz_canvas_configure)
        self._viz_canvas.bind("<MouseWheel>", lambda e: self._viz_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        res_card = tk.Frame(outer, bg=COLORS["border"])
        res_card.grid(row=1, column=0, sticky="nsew")
        inner_r = tk.Frame(res_card, bg=COLORS["surface"])
        inner_r.pack(fill="both", expand=True, padx=1, pady=1)
        inner_r.columnconfigure(0, weight=1)

        rhdr = tk.Frame(inner_r, bg=COLORS["hdr_bg"], height=30)
        rhdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        rhdr.grid_propagate(False)
        _text(rhdr, "Résultats / Réponses", size=10, bold=True, bg=COLORS["hdr_bg"]).pack(side="left", padx=10, pady=6)

        _line(inner_r).grid(row=1, column=0, columnspan=2, sticky="ew")

        self._iter_area = tk.Frame(inner_r, bg=COLORS["surface"])
        self._iter_area.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=(10, 6))
        self._iter_area.columnconfigure(0, weight=1)
        self._iter_area.rowconfigure(0, weight=1)

        self.table_tree = ttk.Treeview(self._iter_area, show="headings", height=7)
        self.table_tree.grid(row=0, column=0, sticky="nsew")

        ysb = ttk.Scrollbar(self._iter_area, orient="vertical", command=self.table_tree.yview)
        ysb.grid(row=0, column=1, sticky="ns")
        xsb = ttk.Scrollbar(self._iter_area, orient="horizontal", command=self.table_tree.xview)
        xsb.grid(row=1, column=0, sticky="ew")

        self.table_tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Treeview", font=("Helvetica", 9), rowheight=22)
        style.configure("Treeview.Heading", font=("Helvetica", 9, "bold"))

        _line(inner_r).grid(row=3, column=0, columnspan=2, sticky="ew")

        inner_r.rowconfigure(4, weight=1)
        self.answers_text = tk.Text(
            inner_r,
            height=8,
            font=("Courier", 9),
            bg="#f8fbff",
            fg=COLORS["text"],
            relief="flat",
            padx=10,
            pady=8,
            wrap="word",
        )
        self.answers_text.grid(row=4, column=0, sticky="nsew", padx=(10, 0), pady=(6, 10))
        sb2 = tk.Scrollbar(inner_r, command=self.answers_text.yview)
        sb2.grid(row=4, column=1, sticky="ns", pady=(6, 10))
        self.answers_text.configure(yscrollcommand=sb2.set)

        self._hide_iter_area()

    def _on_plot_frame_configure(self, event):
        self._viz_canvas.configure(scrollregion=self._viz_canvas.bbox("all"))

    def _on_viz_canvas_configure(self, event):
        self._viz_canvas.itemconfig(self._viz_win, width=event.width)

    def _hide_iter_area(self):
        if self._iter_area is not None:
            self._iter_area.grid_remove()

    def _show_iter_area(self):
        if self._iter_area is not None:
            self._iter_area.grid()

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────
>>>>>>> e01eae2 (Axe2 - changed matrices shown in cholesky)
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
<<<<<<< HEAD
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)
        self.result_text.config(state="disabled")
=======
        self.answers_text.delete("1.0", tk.END)
        self.answers_text.insert("1.0", text)
>>>>>>> e01eae2 (Axe2 - changed matrices shown in cholesky)

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

<<<<<<< HEAD
            # ── Direct methods ──────────────────────────────────
            if algo == "Gauss (partial pivot)":
                x, U, _ = gaussian_elimination_partial_pivot(A, b)
                result_str = self._format_solution(x)
                self._plot_direct(A, b, x, U, "Gauss — Partial Pivot")
=======
            # ── Direct methods ─────────────────────────────────────
            if algo == "Gauss (pivot partiel)":
                x, steps, U = gauss_partial_steps(A, b)
                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))
>>>>>>> e01eae2 (Axe2 - changed matrices shown in cholesky)

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
<<<<<<< HEAD
                result_str = (self._format_solution(x) +
                              f"\n\nL =\n{np.array2string(L, precision=4)}")
                self._plot_direct(A, b, x, L @ L.T, "Cholesky")
=======
                # R = Lᵀ is upper triangular; its diagonal holds the √d_i values
                R = L.T

                # Extract D^(1/2) and the unit upper-triangular factor L̃ᵀ
                # so that R = D^(1/2) · L̃ᵀ  =>  A = L̃ · D · L̃ᵀ
                d_sqrt = np.diag(np.diag(R))               # diagonal matrix of √d_i
                L_tilde_T = np.linalg.inv(d_sqrt) @ R      # unit upper-triangular

                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))

                out = "═══ Cholesky — vue LDLᵀ ═══\n"
                out += "A = Rᵀ·R  avec  R = D^(½)·L̃ᵀ\n\n"
                out += "\n".join(f"  x{i+1} = {x[i]:.8f}" for i in range(n))
                out += f"\n\n‖Ax - b‖∞ = {res_inf:.2e}\n\n"
                out += "Entrées diagonales de D^(½)  (= racines des pivots) :\n"
                out += "  " + "  ".join(f"√d{i+1} = {d_sqrt[i, i]:.6f}" for i in range(n)) + "\n"
                out += "\nMatrice R = D^(½)·L̃ᵀ  (triangulaire supérieure) :\n"
                out += format_matrix_str(R, precision=4)
                self._set_result(out)

                self._plot_cholesky_ldlt(R, d_sqrt, L_tilde_T,
                                         title="Cholesky — Décomposition LDLᵀ")
                self._set_pill("ok", "ok")
>>>>>>> e01eae2 (Axe2 - changed matrices shown in cholesky)

            # ── Iterative methods ──────────────────────────────────
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

            else:
                messagebox.showerror("Erreur", f"Algorithme non supporté : {algo}")
                self._set_pill("erreur", "error")

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
<<<<<<< HEAD
        tbl.set_fontsize(9.5)
        tbl.scale(1.2, 2.0)
=======
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.6)

        for (row_idx, col_idx), cell in tbl.get_celld().items():
            if row_idx == 0:
                cell.set_facecolor("#d6eaf8")
                cell.set_text_props(fontweight="bold")
            elif highlight_row is not None and row_idx - 1 == highlight_row:
                cell.set_facecolor("#aed6f1")
            elif highlight_col is not None and col_idx == highlight_col:
                cell.set_facecolor("#d5f5e3")

    def _plot_steps_as_tables(self, steps, U, b_orig, title):
        self._clear_plot()
        n = U.shape[0]

        all_mats, all_bvecs, all_titles, all_hr, all_hc = [], [], [], [], []
        all_mats.append(self._last_A_orig.copy())
        all_bvecs.append(b_orig.copy())
        all_titles.append("A⁽0⁾ (initial)")
        all_hr.append(None)
        all_hc.append(None)

        for s in steps:
            all_mats.append(s["A"])
            all_bvecs.append(s["b"])
            all_titles.append(f"A⁽{s['step']}⁾  pivot={s['pivot']:+.3f}")
            all_hr.append(s["step"] - 1)
            all_hc.append(s["step"] - 1)

        all_mats.append(U)
        all_bvecs.append(None)
        all_titles.append("U (finale)")
        all_hr.append(None)
        all_hc.append(None)

        n_total = len(all_mats)
        n_cols = min(n_total, 3)
        n_rows = (n_total + n_cols - 1) // n_cols

        row_h = 0.6 + n * 0.45
        fig = Figure(figsize=(n_cols * 3.8, n_rows * row_h + 0.6), dpi=100)
        fig.suptitle(title, fontsize=10, fontweight="bold", y=0.98)
        gs = fig.add_gridspec(n_rows, n_cols)
        fig.subplots_adjust(top=0.90, hspace=0.55, wspace=0.25)

        for idx, (mat, bv, ttl, hr, hc) in enumerate(zip(all_mats, all_bvecs, all_titles, all_hr, all_hc)):
            r, c = divmod(idx, n_cols)
            self._add_matrix_table(fig, gs[r, c], mat, bv, ttl, n, highlight_row=hr, highlight_col=hc)

        self._embed_figure(fig)

    def _plot_lu_steps_as_tables(self, steps, L, U, title):
        self._clear_plot()
        n = L.shape[0]

        n_steps = len(steps)
        n_pairs = n_steps + 1
        n_cols = 4
        n_rows = (n_pairs * 2 + n_cols - 1) // n_cols

        row_h = 0.6 + n * 0.45
        fig = Figure(figsize=(n_cols * 3.4, n_rows * row_h + 0.6), dpi=100)
        fig.suptitle(title, fontsize=10, fontweight="bold", y=0.98)
        gs = fig.add_gridspec(n_rows, n_cols)
        fig.subplots_adjust(top=0.90, hspace=0.55, wspace=0.25)

        col_labels_sq = [f"col{j+1}" for j in range(n)]

        def add_sq_table(gs_slot, M, ttl, header_color):
            ax = fig.add_subplot(gs_slot)
            ax.axis("off")
            ax.set_title(ttl, fontsize=8, fontweight="bold", pad=4)
            cell_text = [[f"{M[i, j]:+.3f}" for j in range(n)] for i in range(n)]
            tbl = ax.table(cellText=cell_text, colLabels=col_labels_sq, cellLoc="center", loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1.0, 1.6)
            for (ri, _ci), cell in tbl.get_celld().items():
                if ri == 0:
                    cell.set_facecolor(header_color)
                    cell.set_text_props(fontweight="bold")

        slot = 0
        for s in steps:
            r, c = divmod(slot, n_cols)
            add_sq_table(gs[r, c], s["L_current"], f"L étape {s['step']}", "#d5f5e3")
            slot += 1
            r, c = divmod(slot, n_cols)
            add_sq_table(gs[r, c], s["U_current"], f"U étape {s['step']}", "#d6eaf8")
            slot += 1

        r, c = divmod(slot, n_cols)
        add_sq_table(gs[r, c], L, "L (finale)", "#a9dfbf")
        slot += 1
        r, c = divmod(slot, n_cols)
        add_sq_table(gs[r, c], U, "U (finale)", "#aed6f1")

        self._embed_figure(fig)

    def _plot_cholesky_ldlt(self, R, d_sqrt, L_tilde_T, title):
        """
        Display four matrices side by side:
          L̃ᵀ  |  D^(½)  |  R = D^(½)·L̃ᵀ  |  RᵀR ≈ A
        This makes the LDLᵀ structure fully visible without changing the algorithm.
        """
        self._clear_plot()
        n = R.shape[0]

        fig = Figure(figsize=(14, 3.5 + n * 0.55), dpi=100)
        fig.suptitle(title, fontsize=10, fontweight="bold", y=0.98)
        gs = fig.add_gridspec(1, 4)
        fig.subplots_adjust(top=0.86, wspace=0.4)

        col_labels = [f"col{j+1}" for j in range(n)]

        def add_tbl(gs_slot, M, ttl, color):
            ax = fig.add_subplot(gs_slot)
            ax.axis("off")
            ax.set_title(ttl, fontsize=9, fontweight="bold", pad=4)
            cell_text = [[f"{M[i, j]:+.4f}" for j in range(n)] for i in range(n)]
            tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                           cellLoc="center", loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1.0, 1.8)
            for (ri, _ci), cell in tbl.get_celld().items():
                if ri == 0:
                    cell.set_facecolor(color)
                    cell.set_text_props(fontweight="bold")

        add_tbl(gs[0], L_tilde_T,  "L̃ᵀ  (unit upper)",   "#fde8d8")
        add_tbl(gs[1], d_sqrt,      "D^(½)  (diagonal)",   "#fef9e7")
        add_tbl(gs[2], R,           "R = D^(½)·L̃ᵀ",       "#a9dfbf")
        add_tbl(gs[3], R.T @ R,     "RᵀR  (≈ A)",          "#d6eaf8")
>>>>>>> e01eae2 (Axe2 - changed matrices shown in cholesky)

        fig.suptitle(title, fontsize=13, fontweight="bold")
        self._embed_figure(fig)

    # ── Plotting — iterative methods ──────────────────────────────
    def _plot_iterative(self, history, title):
<<<<<<< HEAD
        ks   = [h[0] for h in history]
=======
        self._clear_plot()

        if not history:
            fig = Figure(figsize=(8.0, 3.0), dpi=100)
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.set_title(f"{title} — Convergence", fontsize=11, fontweight="bold", pad=10)
            ax.text(0.5, 0.5, "Aucune itération enregistrée.", ha="center", va="center")
            self._embed_figure(fig)
            self._current_table_data = (["k"], [])
            return

        ks = [h[0] for h in history]
>>>>>>> e01eae2 (Axe2 - changed matrices shown in cholesky)
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
<<<<<<< HEAD
        if self._back_callback:
            self._back_callback()
        else:
            import subprocess
            path = os.path.join(os.path.dirname(__file__), "main_screen.py")
            subprocess.Popen([sys.executable, path])
            self.winfo_toplevel().destroy()
=======
        import subprocess
        path = os.path.join(os.path.dirname(__file__), "main_screen.py")
        subprocess.Popen([sys.executable, path])
        self.destroy()
>>>>>>> e01eae2 (Axe2 - changed matrices shown in cholesky)


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