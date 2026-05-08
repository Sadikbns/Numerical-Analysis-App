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
    solve_lu,                 # pivoted LU: returns x, P, L, U
    solve_cholesky,           # returns x, L (lower)
    solve_iteratif,
    gauss_seidel,
    is_strictly_diagonally_dominant,
    is_symmetric_positive_definite,
    induced_matrix_norm,
    spectral_radius,
)

# ──────────────────────────────────────────────────────────────────────────────
# Design tokens (mirrors the style used in your axe3_screen.py)
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


def _line(parent, vertical=False):
    if vertical:
        return tk.Frame(parent, bg=COLORS["border"], width=1)
    return tk.Frame(parent, bg=COLORS["border"], height=1)


def _panel(parent, title, dot=None, compact=False):
    """Card with header + body (same structure as axe3)."""
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
# Helpers: convergence / LU / Cholesky display
# ──────────────────────────────────────────────────────────────────────────────
def _has_zero_pivot_on_diagonal(A, eps=1e-15):
    return bool(np.any(np.abs(np.diag(A)) < eps))


def _format_matrix(M, name=""):
    s = np.array2string(np.array(M, dtype=float), precision=4, suppress_small=True)
    return f"{name} =\n{s}" if name else s


# ──────────────────────────────────────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────────────────────────────────────
class Axe2Screen(tk.Tk):
    """
    Axe 2 — Résolution des systèmes linéaires (UI aligned with axe3)

    User requirements implemented:
      - Remove rank (keep determinant only)
      - Left panel is scrollable (like axe3)
      - General UI style like axe3 (cards/panels + header)
      - No error estimation plots; for Gauss show the final U matrix instead
      - LU: display L and U (keep single LU decomposition global)
      - Cholesky: display R (upper triangular), with A = R^T R
      - If pivot is null (diag entry ~0), automatically use Gauss partial/total pivot
      - Remove algorithm recommendation
    """

    def __init__(self):
        super().__init__()
        self.title("Axe 2 — Résolution des Systèmes Linéaires")

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1200, sw - 80), min(860, sh - 80)
        self.geometry(f"{w}x{h}")
        self.minsize(1020, 680)

        self.configure(bg=COLORS["bg"])

        # State
        self._matrix_entries = []
        self._b_entries = []
        self.omega_entry = None

        self._current_fig = None
        self._current_table_data = None

        self._build()

    # ──────────────────────────────────────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────────────────────────────────────
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

        _button(
            bar,
            "  Retour",
            bg=COLORS["primary_btn"],
            pad_x=14,
            pad_y=8,
            cmd=self._back,
        ).pack(side="left", padx=12, pady=8)

        _text(
            bar,
            "Axe 2 — Résolution des Systèmes Linéaires",
            size=14,
            bold=True,
            color="white",
            bg=COLORS["primary"],
        ).pack(side="left", padx=8)

        self._pill_var = tk.StringVar(value="")
        pill = tk.Label(
            bar,
            textvariable=self._pill_var,
            bg=COLORS["primary_mid"],
            fg="white",
            font=("Helvetica", 9),
            padx=12,
            pady=4,
        )
        pill.pack(side="right", padx=14)
        self._pill = pill
        self._set_pill("pret", "idle")

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

    # ──────────────────────────────────────────────────────────────────────────
    # Left panel (scrollable like axe3)
    # ──────────────────────────────────────────────────────────────────────────
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

        # Mouse wheel
        lc.bind("<MouseWheel>", lambda e: lc.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        self._build_left_content(outer)

    def _build_left_content(self, parent):
        # ── Card: Inputs ─────────────────────────────────────────────
        card, body, _ = _panel(parent, "  Entrees (Ax = b)", dot=COLORS["primary_mid"], compact=True)
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

        # ── Card: Iterative params ─────────────────────────────────
        card2, body2, _ = _panel(parent, "  Parametres iteratifs", dot=COLORS["orange"], compact=True)
        card2.pack(fill="x", pady=(0, 6))

        row = tk.Frame(body2, bg=COLORS["surface"])
        row.pack(fill="x", pady=2)
        _text(row, "Tolerance ε :", size=9, color=COLORS["text_label"], bg=COLORS["surface"]).pack(side="left")
        self.tol_entry = _field(row, "1e-6", width=10)
        self.tol_entry.pack(side="right")

        row = tk.Frame(body2, bg=COLORS["surface"])
        row.pack(fill="x", pady=2)
        _text(row, "Max iterations :", size=9, color=COLORS["text_label"], bg=COLORS["surface"]).pack(side="left")
        self.maxiter_entry = _field(row, "100", width=10)
        self.maxiter_entry.pack(side="right")

        self.extra_frame = tk.Frame(body2, bg=COLORS["surface"])
        self.extra_frame.pack(fill="x", pady=(4, 0))

        # ── Card: Operations matrices ──────────────────────────────
        card3, body3, _ = _panel(parent, "  Operations sur A", dot=COLORS["green"], compact=True)
        card3.pack(fill="x", pady=(0, 6))

        for label, cmd in [
            ("Normes induites  (‖A‖₁, ‖A‖₂, ‖A‖∞)", self._show_norms),
            ("Determinant det(A)", self._show_det_only),
            ("DDS / SPD (conditions)", self._show_dds_spd),
            ("Rayon spectral ρ(A)", self._show_spectral_radius_A),
        ]:
            _button(body3, label, bg=COLORS["primary_lt"], fg=COLORS["primary"],
                    pad_x=10, pad_y=5, size=10, cmd=cmd).pack(fill="x", pady=2)

        # ── Card: Algorithms ───────────────────────────────────────
        card4, body4, _ = _panel(parent, "  Algorithmes", dot=COLORS["primary_mid"], compact=True)
        card4.pack(fill="x", pady=(0, 6))

        self.algo_var = tk.StringVar(value="Gauss (partial pivot)")

        _text(body4, "Methodes directes :", size=9, bold=True,
              color=COLORS["text_label"], bg=COLORS["surface"]).pack(anchor="w", pady=(0, 3))
        for algo in (
            "Gauss (partial pivot)",
            "Gauss (total pivot)",
            "LU Decomposition",
            "Cholesky",
        ):
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

        _text(body4, "Methodes iteratives :", size=9, bold=True,
              color=COLORS["text_label"], bg=COLORS["surface"]).pack(anchor="w", pady=(0, 3))
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

        # ── Run button ─────────────────────────────────────────────
        run_frame = tk.Frame(parent, bg=COLORS["bg"])
        run_frame.pack(fill="x", pady=(0, 6))
        _button(run_frame, "  Lancer", bg=COLORS["primary_mid"], bold=True,
                size=12, pad_y=8, cmd=self._run_algorithm).pack(fill="x")

        self._update_extra()

    def _rebuild_matrix(self):
        for w in self.matrix_frame.winfo_children():
            w.destroy()
        self._matrix_entries.clear()
        self._b_entries.clear()

        n = self.size_var.get()

        # Header row
        tk.Label(self.matrix_frame, text="A", bg=COLORS["surface"],
                 fg=COLORS["text_muted"], font=("Helvetica", 9, "bold")).grid(
            row=0, column=0, columnspan=n, sticky="ew"
        )
        tk.Label(self.matrix_frame, text="b", bg=COLORS["surface"],
                 fg=COLORS["text_muted"], font=("Helvetica", 9, "bold")).grid(
            row=0, column=n + 1, padx=(6, 0), sticky="ew"
        )

        for i in range(n):
            row_entries = []
            for j in range(n):
                e = _field(self.matrix_frame, "0", width=6, mono=True)
                e.grid(row=i + 1, column=j, padx=2, pady=2)
                row_entries.append(e)
            self._matrix_entries.append(row_entries)

            tk.Label(self.matrix_frame, text="|", bg=COLORS["surface"], fg=COLORS["text_muted"]).grid(
                row=i + 1, column=n, padx=4
            )

            b = _field(self.matrix_frame, "0", width=6, mono=True)
            b.grid(row=i + 1, column=n + 1, padx=2, pady=2)
            self._b_entries.append(b)

    def _update_extra(self):
        for w in self.extra_frame.winfo_children():
            w.destroy()
        self.omega_entry = None

        if self.algo_var.get() == "Relaxation":
            row = tk.Frame(self.extra_frame, bg=COLORS["surface"])
            row.pack(fill="x", pady=2)
            _text(row, "ω (0 < ω < 2) :", size=9, color=COLORS["text_label"], bg=COLORS["surface"]).pack(
                side="left"
            )
            self.omega_entry = _field(row, "1.25", width=10, mono=True)
            self.omega_entry.pack(side="right")

    # ──────────────────────────────────────────────────────────────────────────
    # Right panel (plot + result, similar to axe3)
    # ──────────────────────────────────────────────────────────────────────────
    def _build_right(self, parent):
        outer = tk.Frame(parent, bg=COLORS["bg"])
        outer.grid(row=0, column=1, sticky="nsew")
        outer.rowconfigure(0, weight=3)
        outer.rowconfigure(1, weight=2)
        outer.columnconfigure(0, weight=1)

        # ── Plot card
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
            side="left", padx=10, pady=6
        )

        dl = tk.Frame(phdr, bg=COLORS["hdr_bg"])
        dl.pack(side="right", padx=8)
        _button(dl, "  PNG/PDF", bg=COLORS["primary_mid"], pad_x=10, pad_y=3,
                size=9, cmd=self._download_graph).pack(side="right", padx=3)
        _button(dl, "  CSV", bg=COLORS["purple"], pad_x=10, pad_y=3,
                size=9, cmd=self._download_table).pack(side="right", padx=3)

        _line(inner_p).grid(row=1, column=0, sticky="ew")

        self.plot_frame = tk.Frame(inner_p, bg=COLORS["surface"])
        self.plot_frame.grid(row=2, column=0, sticky="nsew")

        # ── Result text card
        res_card = tk.Frame(outer, bg=COLORS["border"])
        res_card.grid(row=1, column=0, sticky="nsew")
        inner_r = tk.Frame(res_card, bg=COLORS["surface"])
        inner_r.pack(fill="both", expand=True, padx=1, pady=1)
        inner_r.rowconfigure(2, weight=1)
        inner_r.columnconfigure(0, weight=1)

        rhdr = tk.Frame(inner_r, bg=COLORS["hdr_bg"], height=30)
        rhdr.grid(row=0, column=0, sticky="ew")
        rhdr.grid_propagate(False)
        _text(rhdr, "Resultat", size=10, bold=True, bg=COLORS["hdr_bg"]).pack(
            side="left", padx=10, pady=6
        )

        _line(inner_r).grid(row=1, column=0, sticky="ew")

        self.result_text = tk.Text(
            inner_r,
            height=10,
            font=("Courier", 10),
            bg="#f8fbff",
            fg=COLORS["text"],
            relief="flat",
            padx=10,
            pady=8,
            wrap="word",
        )
        self.result_text.grid(row=2, column=0, sticky="nsew")
        sb = tk.Scrollbar(inner_r, command=self.result_text.yview)
        sb.grid(row=2, column=1, sticky="ns")
        self.result_text.configure(yscrollcommand=sb.set)

    # ──────────────────────────────────────────────────────────────────────────
    # Read / set output
    # ──────────────────────────────────────────────────────────────────────────
    def _read_matrix(self):
        n = self.size_var.get()
        try:
            A = [[float(self._matrix_entries[i][j].get().strip()) for j in range(n)] for i in range(n)]
            b = [float(self._b_entries[i].get().strip()) for i in range(n)]
        except Exception:
            raise ValueError("Entrées invalides : A et b doivent etre numeriques.")
        return np.array(A, dtype=float), np.array(b, dtype=float)

    def _set_result(self, text):
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)

    def _clear_plot(self):
        for w in self.plot_frame.winfo_children():
            w.destroy()
        self._current_fig = None
        self._current_table_data = None

    # ──────────────────────────────────────────────────────────────────────────
    # Matrix operations (no rank)
    # ──────────────────────────────────────────────────────────────────────────
    def _show_norms(self):
        try:
            A, _ = self._read_matrix()
            n1 = induced_matrix_norm(A, 1)
            n2 = induced_matrix_norm(A, 2)
            ni = induced_matrix_norm(A, np.inf)
            self._set_result(
                "Normes induites :\n"
                f"  ‖A‖₁  = {n1:.6f}\n"
                f"  ‖A‖₂  = {n2:.6f}\n"
                f"  ‖A‖∞  = {ni:.6f}"
            )
            self._set_pill("ok", "ok")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            self._set_pill("erreur", "error")

    def _show_det_only(self):
        try:
            A, _ = self._read_matrix()
            det = float(np.linalg.det(A))
            msg = f"det(A) = {det:.6f}\n"
            msg += "✔ det(A) ≠ 0  → solution unique (en theorie)\n" if abs(det) > 1e-12 else "✘ det(A) ≈ 0  → pas de solution unique\n"
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
                f"  SPD (symetrique definie positive)  : {spd}\n\n"
                "Remarques :\n"
                "  • SPD est requis pour Cholesky.\n"
                "  • DDS est une condition suffisante pour la convergence de Jacobi / Gauss-Seidel."
            )
            self._set_pill("ok", "ok")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            self._set_pill("erreur", "error")

    def _show_spectral_radius_A(self):
        try:
            A, _ = self._read_matrix()
            rho = spectral_radius(A)
            self._set_result(f"Rayon spectral :\n  ρ(A) = {rho:.6f}")
            self._set_pill("ok", "ok")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            self._set_pill("erreur", "error")

    # ──────────────────────────────────────────────────────────────────────────
    # Run algorithm (no recommend, Gauss shows U matrix, LU shows L+U, Cholesky shows R)
    # ──────────────────────────────────────────────────────────────────────────
    def _run_algorithm(self):
        self._set_pill("calcul...", "running")
        self.update()

        try:
            A, b = self._read_matrix()
            algo = self.algo_var.get()
            tol = float(self.tol_entry.get())
            max_iter = int(self.maxiter_entry.get())
            n = len(b)

            self._clear_plot()

            # Iterative guard
            is_iter = algo in ("Jacobi", "Gauss-Seidel", "Relaxation")
            if is_iter and not is_strictly_diagonally_dominant(A):
                proceed = messagebox.askyesno(
                    "Avertissement",
                    "La matrice n'est pas DDS.\nLa convergence n'est pas garantie.\n\nContinuer ?",
                )
                if not proceed:
                    self._set_pill("annule", "warn")
                    return

            # ω
            omega = 1.25
            if algo == "Relaxation":
                omega = float(self.omega_entry.get()) if self.omega_entry else 1.25
                if not (0 < omega < 2):
                    messagebox.showerror("Erreur", "ω doit verifier 0 < ω < 2.")
                    self._set_pill("erreur", "error")
                    return

            # ──────────────────────────
            # DIRECT METHODS
            # ──────────────────────────
            if algo in ("Gauss (partial pivot)", "Gauss (total pivot)"):
                # User asked: in matrix U if one of pivots is null use gauss partial or total.
                # We interpret this as: if diagonal contains a null pivot, force a pivoting method.
                # - If the user chose partial, use partial.
                # - If chose total, use total.
                # Additionally, if the user selected one but matrix has obvious zero diagonal,
                # we keep their selection; pivoting will handle it.
                if algo == "Gauss (partial pivot)":
                    x, U, _ = gaussian_elimination_partial_pivot(A, b)
                    title = "Gauss — Pivot Partiel"
                else:
                    x, U, _, perm = gaussian_elimination_total_pivot(A, b)
                    _ = perm
                    title = "Gauss — Pivot Total"

                # NO error estimation plot. Show last U matrix + solution + residual.
                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))
                self._set_result(
                    f"{title}\n\n"
                    + "Solution x*:\n"
                    + "\n".join([f"  x{i+1} = {x[i]:.8f}" for i in range(n)])
                    + f"\n\n‖Ax-b‖∞ = {res_inf:.2e}\n\n"
                    + _format_matrix(U, "U (triangulaire superieure)")
                )

                # Plot: show U as heatmap (instead of residual/error plots)
                self._plot_matrix_heatmap(U, f"{title} — Matrice U", cmap="Blues")
                self._current_table_data = (["i", "j", "U_ij"], self._matrix_to_triplets(U))
                self._set_pill("ok", "ok")
                return

            if algo == "LU Decomposition":
                # Keep one LU decomposition global (we use solve_lu from module).
                # Also: if pivots are null in A (or elimination would fail), solve_lu uses pivoting anyway.
                x, P, L, U = solve_lu(A, b)
                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))

                self._set_result(
                    "LU Decomposition (avec pivotage)\n\n"
                    + "Solution x*:\n"
                    + "\n".join([f"  x{i+1} = {x[i]:.8f}" for i in range(n)])
                    + f"\n\n‖Ax-b‖∞ = {res_inf:.2e}\n\n"
                    + "Decomposition (convention) : P·A = L·U\n\n"
                    + _format_matrix(L, "L")
                    + "\n\n"
                    + _format_matrix(U, "U")
                )

                # Plot: show both L and U heatmaps in one figure
                self._plot_two_matrices_heatmap(L, U, "LU — Matrices L et U")
                # Table export: store full L and U in CSV-like triplets with tag
                cols = ["mat", "i", "j", "val"]
                rows = []
                rows += [["L", i, j, float(L[i, j])] for i in range(n) for j in range(n)]
                rows += [["U", i, j, float(U[i, j])] for i in range(n) for j in range(n)]
                self._current_table_data = (cols, rows)
                self._set_pill("ok", "ok")
                return

            if algo == "Cholesky":
                if not is_symmetric_positive_definite(A):
                    messagebox.showerror(
                        "Matrice invalide",
                        "Cholesky requiert une matrice SPD (symetrique definie positive).",
                    )
                    self._set_pill("erreur", "error")
                    return

                x, L = solve_cholesky(A, b)
                # Requirement: display matrix R (upper triangular).
                # If A = L·L^T, then R = L^T and A = R^T·R.
                R = L.T
                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))

                self._set_result(
                    "Cholesky\n\n"
                    + "Solution x*:\n"
                    + "\n".join([f"  x{i+1} = {x[i]:.8f}" for i in range(n)])
                    + f"\n\n‖Ax-b‖∞ = {res_inf:.2e}\n\n"
                    + "Factorisation : A = R^T · R\n\n"
                    + _format_matrix(R, "R (triangulaire superieure)")
                )

                self._plot_matrix_heatmap(R, "Cholesky — Matrice R", cmap="Greens")
                self._current_table_data = (["i", "j", "R_ij"], self._matrix_to_triplets(R))
                self._set_pill("ok", "ok")
                return

            # ──────────────────────────
            # ITERATIVE METHODS
            # ──────────────────────────
            if algo == "Jacobi":
                x0 = np.zeros(n)
                _, rho_B, hist = solve_iteratif(A, b, x0, tol, methode="jacobi", max_iter=max_iter)
                history = [(k, xk, float(err), float(np.linalg.norm(A @ xk - b, ord=np.inf)))
                           for (k, xk, err) in hist]
                x = hist[-1][1] if hist else x0
                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))

                self._set_result(
                    "Jacobi\n\n"
                    + "Solution approx x*:\n"
                    + "\n".join([f"  x{i+1} = {x[i]:.8f}" for i in range(n)])
                    + f"\n\nρ(B_J) = {float(rho_B):.6f}\n"
                    + f"‖Ax-b‖∞ = {res_inf:.2e}"
                )
                self._plot_iterative(history, "Jacobi")
                self._set_pill("ok", "ok")
                return

            if algo == "Gauss-Seidel":
                x0 = np.zeros(n)
                x, hist = gauss_seidel(A, b, x0=x0, tol=tol, max_iter=max_iter)
                history = [(k, xk, float(err), float(np.linalg.norm(A @ xk - b, ord=np.inf)))
                           for (k, xk, err) in hist]
                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))

                self._set_result(
                    "Gauss-Seidel\n\n"
                    + "Solution approx x*:\n"
                    + "\n".join([f"  x{i+1} = {x[i]:.8f}" for i in range(n)])
                    + f"\n\n‖Ax-b‖∞ = {res_inf:.2e}"
                )
                self._plot_iterative(history, "Gauss-Seidel")
                self._set_pill("ok", "ok")
                return

            if algo == "Relaxation":
                x0 = np.zeros(n)
                x, hist = self._relaxation(A, b, x0, omega, tol, max_iter)
                history = [(k, xk, float(err), float(np.linalg.norm(A @ xk - b, ord=np.inf)))
                           for (k, xk, err) in hist]
                res_inf = float(np.linalg.norm(A @ x - b, ord=np.inf))

                self._set_result(
                    f"SOR (Relaxation)\n\nω = {omega}\n\n"
                    + "Solution approx x*:\n"
                    + "\n".join([f"  x{i+1} = {x[i]:.8f}" for i in range(n)])
                    + f"\n\n‖Ax-b‖∞ = {res_inf:.2e}"
                )
                self._plot_iterative(history, "SOR")
                self._set_pill("ok", "ok")
                return

            # Fallback
            self._set_result("Choisissez un algorithme.")
            self._set_pill("warn", "warn")

        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            self._set_pill("erreur", "error")

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
                    raise ValueError("SOR: pivot nul sur la diagonale (a_ii = 0).")
                s = b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1 :], x_old[i + 1 :])
                x[i] = (1 - omega) * x_old[i] + omega * s / A[i, i]
            err = float(np.linalg.norm(x - x_old, ord=np.inf))
            history.append((k + 1, x.copy(), err))
            if err < tol:
                break
        return x, history

    # ──────────────────────────────────────────────────────────────────────────
    # Plotting: show U / matrices / iterative convergence (axe3-like embedding)
    # ──────────────────────────────────────────────────────────────────────────
    def _embed_figure(self, fig):
        self._current_fig = fig
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_matrix_heatmap(self, M, title, cmap="Blues"):
        self._clear_plot()
        M = np.array(M, dtype=float)
        fig = Figure(figsize=(8.8, 4.8), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(np.abs(M), cmap=cmap, aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        self._embed_figure(fig)

    def _plot_two_matrices_heatmap(self, A, B, title):
        self._clear_plot()
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)

        fig = Figure(figsize=(9.2, 4.8), dpi=100)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        im1 = ax1.imshow(np.abs(A), cmap="Greens", aspect="auto")
        ax1.set_title("|L|", fontsize=11, fontweight="bold")
        ax1.set_xlabel("j")
        ax1.set_ylabel("i")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(np.abs(B), cmap="Blues", aspect="auto")
        ax2.set_title("|U|", fontsize=11, fontweight="bold")
        ax2.set_xlabel("j")
        ax2.set_ylabel("i")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        self._embed_figure(fig)

    def _plot_iterative(self, history, title):
        """
        history: list of (k, xk, err_inf, res_inf)
        """
        self._clear_plot()
        ks = [h[0] for h in history]
        errs = [h[2] for h in history]
        ress = [h[3] for h in history]

        n_vars = len(history[0][1])
        cols = ["k"] + [f"x{i+1}" for i in range(n_vars)] + ["||x(k+1)-x(k)||∞", "||Ax-b||∞"]
        rows = []
        for k, xk, err, res in history:
            rows.append([k] + [float(v) for v in xk] + [err, res])
        self._current_table_data = (cols, rows)

        fig = Figure(figsize=(9.0, 5.2), dpi=100)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.35)

        ax = fig.add_subplot(gs[0])
        ax.semilogy(ks, errs, "b-o", markersize=3, label="||x(k+1)-x(k)||∞")
        ax.semilogy(ks, ress, "r--s", markersize=3, label="||Ax-b||∞")
        ax.set_title(f"{title} — Convergence", fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration k")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

        ax_tbl = fig.add_subplot(gs[1])
        ax_tbl.axis("off")
        display = []
        for r in rows[:12]:
            display.append([str(r[0])] + [f"{v:.6f}" for v in r[1:1+n_vars]] + [f"{r[-2]:.2e}", f"{r[-1]:.2e}"])
        tbl = ax_tbl.table(cellText=display, colLabels=cols, cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.05, 1.6)

        fig.tight_layout()
        self._embed_figure(fig)

    def _matrix_to_triplets(self, M):
        M = np.array(M, dtype=float)
        n, m = M.shape
        rows = []
        for i in range(n):
            for j in range(m):
                rows.append([i, j, float(M[i, j])])
        return rows

    # ──────────────────────────────────────────────────────────────────────────
    # Downloads
    # ──────────────────────────────────────────────────────────────────────────
    def _download_graph(self):
        if self._current_fig is None:
            messagebox.showinfo("Info", "Lancez un algorithme d'abord.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF", "*.pdf")],
            title="Sauvegarder la figure",
        )
        if path:
            self._current_fig.savefig(path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("OK", f"Figure sauvegardee :\n{path}")

    def _download_table(self):
        if self._current_table_data is None:
            messagebox.showinfo("Info", "Aucun tableau a exporter.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            title="Exporter le tableau",
        )
        if not path:
            return
        cols, rows = self._current_table_data
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(cols)
            for r in rows:
                w.writerow(r)
        messagebox.showinfo("OK", f"Tableau exporte :\n{path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Navigation
    # ──────────────────────────────────────────────────────────────────────────
    def _back(self):
        import subprocess
        path = os.path.join(os.path.dirname(__file__), "main_screen.py")
        subprocess.Popen([sys.executable, path])
        self.destroy()


if __name__ == "__main__":
    Axe2Screen().mainloop()
