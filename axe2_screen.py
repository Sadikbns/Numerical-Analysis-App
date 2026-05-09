import os
import sys
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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
# Step-by-step Gauss elimination
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
        steps.append(
            {
                "step": k + 1,
                "pivot": float(M[k, k]),
                "A": M[:, :n].copy(),
                "b": M[:, n].copy(),
            }
        )

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
        steps.append(
            {
                "step": k + 1,
                "pivot": float(M[k, k]),
                "A": M[:, :n].copy(),
                "b": M[:, n].copy(),
            }
        )

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
                f"LU sans pivot: pivot nul à l'étape {k+1}.\n"
                "Utilisez Gauss avec pivotage."
            )
        multipliers = {}
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            multipliers[i] = m
            U[i, k:] -= m * U[k, k:]
        steps.append(
            {
                "step": k + 1,
                "pivot": float(U[k, k]),
                "multipliers": multipliers,
                "U_current": U.copy(),
                "L_current": L.copy(),
            }
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
# Design tokens
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

        self._matrix_entries = []
        self._b_entries = []
        self.omega_entry = None
        self._current_fig = None
        self._current_table_data = None

        self._last_A_orig = None

        self._viz_canvas = None
        self.plot_frame = None
        self._viz_win = None

        # Results bottom: we keep BOTH (table + answers text) always.
        # - Iter table is optional (shown only for iterative methods)
        # - Answers text ALWAYS visible at bottom (below table)
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
            bar, textvariable=self._pill_var, bg=COLORS["primary_mid"], fg="white", font=("Helvetica", 9), padx=12, pady=4
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
            _button(body3, label, bg=COLORS["primary_lt"], fg=COLORS["primary"], pad_x=10, pad_y=5, size=10, cmd=cmd).pack(
                fill="x", pady=2
            )

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
            run_frame, "  ▶  Lancer", bg=COLORS["primary_mid"], bold=True, size=12, pad_y=8, cmd=self._run_algorithm
        ).pack(fill="x")

        self._update_extra()

    def _rebuild_matrix(self):
        for w in self.matrix_frame.winfo_children():
            w.destroy()
        self._matrix_entries.clear()
        self._b_entries.clear()

        n = self.size_var.get()
        tk.Label(self.matrix_frame, text="A", bg=COLORS["surface"], fg=COLORS["text_muted"], font=("Helvetica", 9, "bold")).grid(
            row=0, column=0, columnspan=n, sticky="ew"
        )
        tk.Label(self.matrix_frame, text="b", bg=COLORS["surface"], fg=COLORS["text_muted"], font=("Helvetica", 9, "bold")).grid(
            row=0, column=n + 1, padx=(6, 0), sticky="ew"
        )

        for i in range(n):
            row_entries = []
            for j in range(n):
                e = _field(self.matrix_frame, "0", width=6)
                e.grid(row=i + 1, column=j, padx=2, pady=2)
                row_entries.append(e)
            self._matrix_entries.append(row_entries)
            tk.Label(self.matrix_frame, text="|", bg=COLORS["surface"], fg=COLORS["text_muted"]).grid(row=i + 1, column=n, padx=4)
            b = _field(self.matrix_frame, "0", width=6)
            b.grid(row=i + 1, column=n + 1, padx=2, pady=2)
            self._b_entries.append(b)

    def _update_extra(self):
        for w in self.extra_frame.winfo_children():
            w.destroy()
        self.omega_entry = None
        if self.algo_var.get() == "Relaxation":
            row = tk.Frame(self.extra_frame, bg=COLORS["surface"])
            row.pack(fill="x", pady=2)
            _text(row, "ω (0 < ω < 2) :", size=9, color=COLORS["text_label"], bg=COLORS["surface"]).pack(side="left")
            self.omega_entry = _field(row, "1.25", width=10)
            self.omega_entry.pack(side="right")

    # ──────────────────────────────────────────────────────────────
    # Right panel: Visualisation (top) + Results card (bottom)
    # Bottom card contains:
    #   - Iterations table (optional, shown only for iterative)
    #   - Answers text ALWAYS visible, BELOW the table
    # ──────────────────────────────────────────────────────────────
    def _build_right(self, parent):
        outer = tk.Frame(parent, bg=COLORS["bg"])
        outer.grid(row=0, column=1, sticky="nsew")
        outer.rowconfigure(0, weight=3)
        outer.rowconfigure(1, weight=2)
        outer.columnconfigure(0, weight=1)

        # ── Visualization card (top) ──────────────────────────────
        viz_card = tk.Frame(outer, bg=COLORS["border"])
        viz_card.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        inner_v = tk.Frame(viz_card, bg=COLORS["surface"])
        inner_v.pack(fill="both", expand=True, padx=1, pady=1)
        inner_v.rowconfigure(2, weight=1)
        inner_v.columnconfigure(0, weight=1)

        vhdr = tk.Frame(inner_v, bg=COLORS["hdr_bg"], height=30)
        vhdr.grid(row=0, column=0, sticky="ew")
        vhdr.grid_propagate(False)
        _text(vhdr, "Visualisation", size=10, bold=True, bg=COLORS["hdr_bg"]).pack(side="left", padx=10, pady=6)

        dl = tk.Frame(vhdr, bg=COLORS["hdr_bg"])
        dl.pack(side="right", padx=8)
        _button(dl, "  PNG/PDF", bg=COLORS["primary_mid"], pad_x=10, pad_y=3, size=9, cmd=self._download_graph).pack(
            side="right", padx=3
        )
        _button(dl, "  CSV", bg=COLORS["purple"], pad_x=10, pad_y=3, size=9, cmd=self._download_table).pack(
            side="right", padx=3
        )

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

        # ── Results card (bottom) ─────────────────────────────────
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

        # Iterations table area (optional)
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

        # Answers text ALWAYS visible (below the table)
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

        # start hidden (direct methods)
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
    def _read_matrix(self):
        n = self.size_var.get()
        try:
            A = [[float(self._matrix_entries[i][j].get().strip()) for j in range(n)] for i in range(n)]
            b = [float(self._b_entries[i].get().strip()) for i in range(n)]
        except Exception:
            raise ValueError("Entrées invalides : A et b doivent être numériques.")
        return np.array(A, dtype=float), np.array(b, dtype=float)

    def _set_result(self, text):
        # write to the ALWAYS-visible answers box (bottom)
        self.answers_text.delete("1.0", tk.END)
        self.answers_text.insert("1.0", text)

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

    def _set_iter_table(self, columns, rows):
        for item in self.table_tree.get_children():
            self.table_tree.delete(item)
        self.table_tree["columns"] = columns
        for c in columns:
            self.table_tree.heading(c, text=c)
            self.table_tree.column(c, anchor="center", width=110, stretch=True)
        for r in rows:
            self.table_tree.insert("", "end", values=list(r))

    # ──────────────────────────────────────────────────────────────
    # Matrix operations
    # ──────────────────────────────────────────────────────────────
    def _show_norms(self):
        try:
            A, _ = self._read_matrix()
            self._set_result(
                "Normes induites :\n"
                f"  ‖A‖₁  = {induced_matrix_norm(A, 1):.6f}\n"
                f"  ‖A‖₂  = {induced_matrix_norm(A, 2):.6f}\n"
                f"  ‖A‖∞  = {induced_matrix_norm(A, np.inf):.6f}"
            )
            self._set_pill("ok", "ok")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            self._set_pill("erreur", "error")

    def _show_det(self):
        try:
            A, _ = self._read_matrix()
            det = float(np.linalg.det(A))
            msg = f"det(A) = {det:.6f}\n"
            msg += "✔ det(A) ≠ 0 → solution unique" if abs(det) > 1e-12 else "✘ det(A) ≈ 0 → pas de solution unique"
            self._set_result(msg)
            self._set_pill("ok", "ok")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            self._set_pill("erreur", "error")

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
            messagebox.showerror("Erreur", str(e))
            self._set_pill("erreur", "error")

    def _show_spectral_A(self):
        try:
            A, _ = self._read_matrix()
            rho = spectral_radius(A)
            self._set_result(f"Rayon spectral :\n  ρ(A) = {rho:.6f}")
            self._set_pill("ok", "ok")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            self._set_pill("erreur", "error")

    # ──────────────────────────────────────────────────────────────
    # Run algorithm
    # ──────────────────────────────────────────────────────────────
    def _run_algorithm(self):
        self._set_pill("calcul...", "running")
        self.update()
        try:
            A, b = self._read_matrix()
            self._last_A_orig = A.copy()

            algo = self.algo_var.get()
            tol = float(self.tol_entry.get())
            max_it = int(self.maxiter_entry.get())
            n = len(b)

            self._clear_plot()
            self._current_table_data = None

            is_iter = algo in ("Jacobi", "Gauss-Seidel", "Relaxation")
            if is_iter:
                self._show_iter_area()
            else:
                self._hide_iter_area()

            if is_iter and not is_strictly_diagonally_dominant(A):
                if not messagebox.askyesno(
                    "Avertissement",
                    "La matrice n'est pas DDS.\nLa convergence n'est pas garantie.\n\nContinuer ?",
                ):
                    self._set_pill("annulé", "warn")
                    return

            omega = 1.25
            if algo == "Relaxation":
                omega = float(self.omega_entry.get()) if self.omega_entry else 1.25
                if not (0 < omega < 2):
                    messagebox.showerror("Erreur", "ω doit vérifier 0 < ω < 2.")
                    self._set_pill("erreur", "error")
                    return

            # ── Direct methods ──────────────────────────────────
            if algo == "Gauss (pivot partiel)":
                x, steps, U = gauss_partial_steps(A, b)
                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))

                out = "═══ Gauss — Pivot Partiel ═══\n\n"
                out += "\n".join(f"  x{i+1} = {x[i]:.8f}" for i in range(n))
                out += f"\n\n‖Ax - b‖∞ = {res_inf:.2e}\n\n"
                out += "Pivots utilisés :\n"
                for s in steps:
                    out += f"  Étape {s['step']} : pivot = {s['pivot']:+.6f}\n"
                out += "\nMatrice U finale :\n"
                out += format_matrix_str(U, precision=4)
                self._set_result(out)

                self._plot_steps_as_tables(steps, U, b, title="Gauss — Pivot Partiel : évolution de [A|b]")
                self._set_pill("ok", "ok")

            elif algo == "Gauss (pivot total)":
                x, steps, U, perm = gauss_total_steps(A, b)
                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))

                out = "═══ Gauss — Pivot Total ═══\n\n"
                out += "\n".join(f"  x{i+1} = {x[i]:.8f}" for i in range(n))
                out += f"\n\n‖Ax - b‖∞ = {res_inf:.2e}\n\n"
                out += f"Permutation des colonnes : {perm}\n\n"
                out += "Pivots utilisés :\n"
                for s in steps:
                    out += f"  Étape {s['step']} : pivot = {s['pivot']:+.6f}\n"
                out += "\nMatrice U finale :\n"
                out += format_matrix_str(U, precision=4)
                self._set_result(out)

                self._plot_steps_as_tables(steps, U, b, title="Gauss — Pivot Total : évolution de [A|b]")
                self._set_pill("ok", "ok")

            elif algo == "LU Decomposition":
                L, U, steps = lu_steps(A)

                y = np.zeros(n)
                for i in range(n):
                    y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
                x = np.zeros(n)
                for i in range(n - 1, -1, -1):
                    x[i] = (y[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]
                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))

                out = "═══ Décomposition LU ═══\nA = L · U\n\n"
                out += "\n".join(f"  x{i+1} = {x[i]:.8f}" for i in range(n))
                out += f"\n\n‖Ax - b‖∞ = {res_inf:.2e}\n\n"
                out += "Multiplicateurs :\n"
                for s in steps:
                    mults = ", ".join(f"m{i+1}{s['step']} = {v:+.4f}" for i, v in s["multipliers"].items())
                    out += f"  Étape {s['step']} : {mults}\n"
                out += "\nMatrice L :\n" + format_matrix_str(L, precision=4)
                out += "\n\nMatrice U :\n" + format_matrix_str(U, precision=4)
                self._set_result(out)

                self._plot_lu_steps_as_tables(steps, L, U, title="LU Décomposition : construction de L et U")
                self._set_pill("ok", "ok")

            elif algo == "Cholesky":
                if not is_symmetric_positive_definite(A):
                    messagebox.showerror("Matrice invalide", "Cholesky requiert une matrice SPD.")
                    self._set_pill("erreur", "error")
                    return

                x, L = solve_cholesky(A, b)
                R = L.T
                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))

                out = "═══ Cholesky ═══\nA = Rᵀ · R\n\n"
                out += "\n".join(f"  x{i+1} = {x[i]:.8f}" for i in range(n))
                out += f"\n\n‖Ax - b‖∞ = {res_inf:.2e}\n\n"
                out += "Matrice R (triangulaire supérieure) :\n"
                out += format_matrix_str(R, precision=4)
                self._set_result(out)

                self._plot_cholesky_table(R, title="Cholesky — Matrice R (triangulaire supérieure)")
                self._set_pill("ok", "ok")

            # ── Iterative methods ───────────────────────────────
            elif algo == "Jacobi":
                x0 = np.zeros(n)
                _, rho_B, hist = solve_iteratif(A, b, x0, tol, methode="jacobi", max_iter=max_it)
                history = [(k, xk, float(err), float(np.linalg.norm(A @ xk - b, ord=np.inf))) for k, xk, err in hist]
                x = history[-1][1] if history else x0
                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))

                out = "═══ Jacobi ═══\n\n"
                out += "\n".join(f"  x{i+1} = {x[i]:.8f}" for i in range(n))
                out += f"\n\nρ(B_J) = {float(rho_B):.6f}"
                out += f"\n‖Ax - b‖∞ = {res_inf:.2e}"
                self._set_result(out)

                self._plot_iterative(history, "Jacobi")
                self._set_iter_table(*self._current_table_data)
                self._set_pill("ok", "ok")

            elif algo == "Gauss-Seidel":
                x0 = np.zeros(n)
                x, hist = gauss_seidel(A, b, x0=x0, tol=tol, max_iter=max_it)
                history = [(k, xk, float(err), float(np.linalg.norm(A @ xk - b, ord=np.inf))) for k, xk, err in hist]
                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))

                out = "═══ Gauss-Seidel ═══\n\n"
                out += "\n".join(f"  x{i+1} = {x[i]:.8f}" for i in range(n))
                out += f"\n\n‖Ax - b‖∞ = {res_inf:.2e}"
                self._set_result(out)

                self._plot_iterative(history, "Gauss-Seidel")
                self._set_iter_table(*self._current_table_data)
                self._set_pill("ok", "ok")

            elif algo == "Relaxation":
                x0 = np.zeros(n)
                x, hist = self._relaxation(A, b, x0, omega, tol, max_it)
                history = [(k, xk, float(err), float(np.linalg.norm(A @ xk - b, ord=np.inf))) for k, xk, err in hist]
                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))

                out = f"═══ SOR (Relaxation) — ω = {omega} ═══\n\n"
                out += "\n".join(f"  x{i+1} = {x[i]:.8f}" for i in range(n))
                out += f"\n\n‖Ax - b‖∞ = {res_inf:.2e}"
                self._set_result(out)

                self._plot_iterative(history, f"SOR  (ω = {omega})")
                self._set_iter_table(*self._current_table_data)
                self._set_pill("ok", "ok")

        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            self._set_pill("erreur", "error")

    # ──────────────────────────────────────────────────────────────
    # SOR
    # ──────────────────────────────────────────────────────────────
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
                    raise ValueError("SOR: pivot nul sur la diagonale.")
                s = b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1 :], x_old[i + 1 :])
                x[i] = (1 - omega) * x_old[i] + omega * s / A[i, i]
            err = float(np.linalg.norm(x - x_old, ord=np.inf))
            history.append((k + 1, x.copy(), err))
            if err < tol:
                break
        return x, history

    # ──────────────────────────────────────────────────────────────
    # Visualization helpers
    # ──────────────────────────────────────────────────────────────
    def _matrix_to_cell_text(self, M, b_vec=None, precision=3):
        M = np.array(M, dtype=float)
        n, m = M.shape
        rows = []
        for i in range(n):
            row = [f"{M[i, j]:+.{precision}f}" for j in range(m)]
            if b_vec is not None:
                row.append(f"{b_vec[i]:+.{precision}f}")
            rows.append(row)
        return rows

    def _make_col_labels(self, n, include_b=True):
        labels = [f"a{chr(0x2081 + j)}" for j in range(n)]
        if include_b:
            labels.append("b")
        return labels

    def _add_matrix_table(self, fig, gs_slot, M, b_vec, title, n, highlight_row=None, highlight_col=None):
        ax = fig.add_subplot(gs_slot)
        ax.axis("off")
        ax.set_title(title, fontsize=8, fontweight="bold", pad=4)

        cell_text = self._matrix_to_cell_text(M, b_vec, precision=3)
        col_labels = self._make_col_labels(n, include_b=(b_vec is not None))

        tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
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

    def _plot_cholesky_table(self, R, title):
        self._clear_plot()
        n = R.shape[0]

        fig = Figure(figsize=(10, 3.5 + n * 0.5), dpi=100)
        fig.suptitle(title, fontsize=10, fontweight="bold", y=0.98)
        gs = fig.add_gridspec(1, 3)
        fig.subplots_adjust(top=0.86, wspace=0.35)

        col_labels = [f"col{j+1}" for j in range(n)]

        def add_tbl(gs_slot, M, ttl, color):
            ax = fig.add_subplot(gs_slot)
            ax.axis("off")
            ax.set_title(ttl, fontsize=9, fontweight="bold", pad=4)
            cell_text = [[f"{M[i, j]:+.4f}" for j in range(n)] for i in range(n)]
            tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1.0, 1.8)
            for (ri, _ci), cell in tbl.get_celld().items():
                if ri == 0:
                    cell.set_facecolor(color)
                    cell.set_text_props(fontweight="bold")

        add_tbl(gs[0], R, "R", "#a9dfbf")
        add_tbl(gs[1], R.T, "Rᵀ", "#d5f5e3")
        add_tbl(gs[2], R.T @ R, "Rᵀ·R  (≈ A)", "#fef9e7")

        self._embed_figure(fig)

    def _plot_iterative(self, history, title):
        self._clear_plot()

        ks = [h[0] for h in history]
        errs = [h[2] for h in history]
        ress = [h[3] for h in history]

        n_vars = len(history[0][1])
        cols = ["k"] + [f"x{i+1}" for i in range(n_vars)] + ["‖xk+1-xk‖∞", "‖Ax-b‖∞"]
        rows = []
        for k, xk, err, res in history:
            rows.append([k] + [float(v) for v in xk] + [err, res])
        self._current_table_data = (cols, rows)

        fig = Figure(figsize=(9.0, 4.2), dpi=100)
        ax = fig.add_subplot(111)
        ax.semilogy(ks, errs, "b-o", markersize=3, label="‖xk+1 - xk‖∞")
        ax.semilogy(ks, ress, "r--s", markersize=3, label="‖Axk - b‖∞")
        ax.set_title(f"{title} — Convergence", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Itération k")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        fig.tight_layout()
        self._embed_figure(fig)

    # ──────────────────────────────────────────────────────────────
    # Downloads
    # ──────────────────────────────────────────────────────────────
    def _download_graph(self):
        if self._current_fig is None:
            messagebox.showinfo("Info", "Lancez un algorithme d'abord.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")],
            title="Sauvegarder la figure",
        )
        if path:
            self._current_fig.savefig(path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("OK", f"Figure sauvegardée :\n{path}")

    def _download_table(self):
        if self._current_table_data is None:
            messagebox.showinfo("Info", "Aucun tableau à exporter.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], title="Exporter le tableau")
        if not path:
            return
        cols, rows = self._current_table_data
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(cols)
            for r in rows:
                w.writerow(r)
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