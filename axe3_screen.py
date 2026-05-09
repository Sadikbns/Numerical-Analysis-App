import os
import sys
import math
import tkinter as tk
from tkinter import filedialog

import numpy as np
import sympy as sp
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── Module import ──────────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "modules", "chap 4 and 5"))
try:
    from interpolation_approximation import (
        lagrange_interpolation,
        newton_differences,
        newton_polynomial,
        least_squares_polynomial,
        chebyshev_approximation,
        gradient_descent_sympy,
        all_discrete_norms,
        all_continuous_norms,
        approximation_error_discrete,
        approximation_error_continuous,
    )
except ImportError as e:
    print("Warning: Could not import interpolation module:", e)

# ── Design tokens (mirrors axe1_widgets.COLORS) ───────────────────────────────
COLORS = {
    "bg":          "#f0f4f8",
    "surface":     "#ffffff",
    "border":      "#d0d7e3",
    "primary":     "#6c3483",
    "primary_mid": "#8e44ad",
    "primary_btn": "#7d3c98",
    "primary_lt":  "#f4eefa",
    "hdr_bg":      "#f8f5fc",
    "text":        "#2c3e50",
    "text_label":  "#34495e",
    "text_muted":  "#7f8c8d",
    "success_fg":  "#1e8449",
    "warn_bg":     "#fffbeb",
    "warn_fg":     "#856404",
    "err_bg":      "#fdf2f2",
    "err_fg":      "#c0392b",
    "blue":        "#2980b9",
    "purple":      "#8e44ad",
    "orange":      "#e67e22",
    "red":         "#e74c3c",
    "green":       "#27ae60",
}

SAFE = {
    "np": np, "math": math,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "abs": np.abs, "pi": math.pi, "e": math.e,
}

METHODS   = ("Lagrange", "Newton", "Moindres Carrés", "Chebyshev", "Descente de Gradient")
EXAMPLES  = ("sin(x)", "cos(x)", "exp(-x**2)", "1/(1+25*x**2)")
METHOD_TAG = {
    "Lagrange":         "Interpolation",
    "Newton":           "Differences Div.",
    "Moindres Carrés":    "Approximation",
    "Chebyshev":        "Minimax",
    "Descente de Gradient": "Optimisation",
}
STATS_DEF = (
    ("mse",   "MSE / err"),
    ("deg",   "degre / iter"),
    ("nodes", "noeuds"),
)


# ── Tiny UI helpers (same pattern as axe1_widgets) ────────────────────────────
def _text(parent, txt, size=10, bold=False, color=None, bg=None):
    kw = {"text": txt,
          "font": ("Helvetica", size, "bold" if bold else "normal"),
          "bg": bg or parent.cget("bg")}
    if color:
        kw["fg"] = color
    return tk.Label(parent, **kw)


def _field(parent, default="", width=28, mono=False):
    e = tk.Entry(parent, width=width,
                 font=("Courier" if mono else "Helvetica", 10),
                 bg=COLORS["surface"], fg=COLORS["text"],
                 relief="solid", bd=1,
                 highlightthickness=1,
                 highlightcolor=COLORS["primary_mid"],
                 highlightbackground=COLORS["border"],
                 insertbackground=COLORS["primary_mid"])
    e.insert(0, default)
    return e


def _button(parent, label, bg=None, fg="white", bold=False,
            size=10, pad_x=10, pad_y=5, cmd=None):
    return tk.Button(parent, text=label,
                     bg=bg or COLORS["primary_mid"], fg=fg,
                     font=("Helvetica", size, "bold" if bold else "normal"),
                     relief="flat", cursor="hand2",
                     padx=pad_x, pady=pad_y,
                     activebackground=COLORS["primary"],
                     activeforeground="white",
                     command=cmd)


def _line(parent, vertical=False):
    if vertical:
        return tk.Frame(parent, bg=COLORS["border"], width=1)
    return tk.Frame(parent, bg=COLORS["border"], height=1)


def _panel(parent, title, dot=None, compact=False):
    """Card with a coloured-dot header — mirrors axe1's _panel."""
    card  = tk.Frame(parent, bg=COLORS["border"])
    inner = tk.Frame(card, bg=COLORS["surface"])
    inner.pack(fill="both", expand=True, padx=1, pady=1)

    hdr = tk.Frame(inner, bg=COLORS["hdr_bg"], height=28)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)

    if dot:
        tk.Frame(hdr, bg=dot, width=4).pack(side="left", fill="y")
    _text(hdr, title, size=10, bold=True,
          bg=COLORS["hdr_bg"], color=COLORS["text"]).pack(side="left", padx=8, pady=5)

    _line(inner).pack(fill="x")

    body = tk.Frame(inner, bg=COLORS["surface"],
                    padx=8 if compact else 12,
                    pady=6 if compact else 10)
    body.pack(fill="x")

    return card, body, inner


def _action_row(parent, icon, label, badge, cmd, pady=2):
    row = tk.Frame(parent, bg=COLORS["surface"])
    row.pack(fill="x", pady=pady)

    ic = tk.Frame(row, bg=COLORS["primary_mid"], width=22, height=22)
    ic.pack(side="left", padx=(0, 6))
    ic.pack_propagate(False)
    tk.Label(ic, text=icon, bg=COLORS["primary_mid"], fg="white",
             font=("Helvetica", 8, "bold")).pack(expand=True)

    _button(row, label, bg=COLORS["primary_lt"], fg=COLORS["primary"],
            pad_x=8, pad_y=4, cmd=cmd).pack(side="left", fill="x", expand=True)

    if badge:
        tk.Label(row, text=badge, bg="#fff3cd", fg="#856404",
                 font=("Helvetica", 8), padx=5, pady=1).pack(side="right")


def show_dialog(root, title, msg, kind="info"):
    pal = {
        "info":    (COLORS["primary_lt"], COLORS["primary_mid"]),
        "success": ("#e8f5e9",            COLORS["green"]),
        "warn":    (COLORS["warn_bg"],    COLORS["warn_fg"]),
        "error":   (COLORS["err_bg"],     COLORS["err_fg"]),
    }
    bg, fg = pal.get(kind, pal["info"])
    d = tk.Toplevel(root)
    d.title(title)
    d.resizable(False, False)
    d.configure(bg=bg)
    d.grab_set()
    tk.Label(d, text=title, bg=bg, fg=fg,
             font=("Helvetica", 11, "bold")).pack(padx=20, pady=(14, 4))
    tk.Label(d, text=msg, bg=bg, fg=COLORS["text"],
             font=("Helvetica", 10), justify="left",
             wraplength=340).pack(padx=20, pady=(0, 10))
    _button(d, "OK", bg=fg, cmd=d.destroy, pad_x=20, pad_y=5).pack(pady=(0, 14))
    d.update_idletasks()
    x = root.winfo_x() + (root.winfo_width()  - d.winfo_width())  // 2
    y = root.winfo_y() + (root.winfo_height() - d.winfo_height()) // 2
    d.geometry(f"+{x}+{y}")


# ── Main screen ───────────────────────────────────────────────────────────────
class Axe3Screen(tk.Frame):
    """Axe 3 — Interpolation / Approximation"""

    def __init__(self, parent, back_callback=None):
        super().__init__(parent, bg=COLORS["bg"])
        self._back_callback = back_callback

        # State
        self._last_y_approx      = None
        self._last_real_func     = None
        self._last_interval      = (-1.0, 1.0)
        self._graph_data_export  = None

        self._build()

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build(self):
        self._build_header()
        content = tk.Frame(self, bg=COLORS["bg"])
        content.pack(fill="both", expand=True, padx=12, pady=10)
        content.columnconfigure(0, minsize=285, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)
        self._build_left(content)
        self._build_right(content)

    def _build_header(self):
        bar = tk.Frame(self, bg=COLORS["primary"], height=52)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        _button(bar, "← Retour", bg=COLORS["primary_btn"],
                cmd=self._back, pad_x=14, pad_y=8).pack(side="left", padx=12, pady=8)

        _text(bar, "Axe 3 — Interpolation / Approximation",
              size=14, bold=True, color="white",
              bg=COLORS["primary"]).pack(side="left", padx=8)

        self._pill_var = tk.StringVar(value="")
        self._pill = tk.Label(bar, textvariable=self._pill_var,
                              bg=COLORS["primary_mid"], fg="white",
                              font=("Helvetica", 9), padx=12, pady=4)
        self._pill.pack(side="right", padx=14)

    # ── Left panel (scrollable) ───────────────────────────────────────────────
    def _build_left(self, parent):
        container = tk.Frame(parent, bg=COLORS["bg"])
        container.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        lc  = tk.Canvas(container, bg=COLORS["bg"], highlightthickness=0)
        vsb = tk.Scrollbar(container, orient="vertical", command=lc.yview)
        lc.configure(yscrollcommand=vsb.set)
        lc.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        outer = tk.Frame(lc, bg=COLORS["bg"])
        win   = lc.create_window((0, 0), window=outer, anchor="nw")
        outer.bind("<Configure>", lambda e: lc.configure(scrollregion=lc.bbox("all")))
        lc.bind("<Configure>",    lambda e: lc.itemconfig(win, width=e.width))
        lc.bind("<MouseWheel>",
                lambda e: lc.yview_scroll(int(-1*(e.delta/120)), "units"))

        self._build_left_content(outer)

    def _build_left_content(self, parent):

        # ── Card 1: data points ───────────────────────────────────────────────
        card, body, _ = _panel(parent, "  Points de donnees",
                               dot=COLORS["primary_mid"], compact=True)
        card.pack(fill="x", pady=(0, 5))

        _text(body, "Points (x, y)", color=COLORS["text_label"]).pack(anchor="w")
        self.points_frame = tk.Frame(body, bg=COLORS["surface"])
        self.points_frame.pack(fill="x", pady=(4, 0))
        self._build_points_input()

        # ── Card 2: function + method ─────────────────────────────────────────
        card2, body2, _ = _panel(parent, "  Methode",
                                 dot=COLORS["primary_mid"], compact=True)
        card2.pack(fill="x", pady=(0, 5))

        _text(body2, "Formule f(x) :", color=COLORS["text_label"]).pack(anchor="w")
        self.func_entry = _field(body2, "cos(x)", mono=True)
        self.func_entry.pack(fill="x", pady=(2, 4))

        _text(body2, "Exemples :", size=9,
              color=COLORS["text_muted"]).pack(anchor="w")
        ex_row = tk.Frame(body2, bg=COLORS["surface"])
        ex_row.pack(fill="x", pady=(2, 6))
        for ex in EXAMPLES:
            tk.Button(ex_row, text=ex, bg=COLORS["hdr_bg"],
                      fg=COLORS["text_muted"], font=("Courier", 8),
                      relief="solid", bd=1, cursor="hand2",
                      padx=4, pady=2,
                      command=lambda v=ex: self._paste_func(v)
                      ).pack(side="left", padx=2)

        _line(body2).pack(fill="x", pady=4)

        self.method_var = tk.StringVar(value="Lagrange")
        for m in METHODS:
            row = tk.Frame(body2, bg=COLORS["surface"])
            row.pack(fill="x", pady=2)
            tk.Radiobutton(row, text=m, variable=self.method_var, value=m,
                           bg=COLORS["surface"], fg=COLORS["text"],
                           font=("Helvetica", 11),
                           activebackground=COLORS["surface"],
                           selectcolor=COLORS["primary_mid"],
                           command=self._update_inputs).pack(side="left")
            tk.Label(row, text=METHOD_TAG[m],
                     bg=COLORS["primary_lt"], fg=COLORS["success_fg"],
                     font=("Helvetica", 8), padx=6, pady=1).pack(side="right")

        self.extra_frame = tk.Frame(body2, bg=COLORS["surface"])
        self.extra_frame.pack(fill="x", pady=(4, 0))
        self._update_inputs()

        # ── Card 3: norms & error ─────────────────────────────────────────────
        card3, body3, _ = _panel(parent, "  Normes & Erreur",
                                 dot=COLORS["blue"], compact=True)
        card3.pack(fill="x", pady=(0, 5))

        for icon, label, badge, cmd in [
            ("L1", "Normes Discretes (L1, L2, Linf)",  None, self._compute_discrete_norms),
            ("~",  "Normes Continues (L1, L2, Linf)",  None, self._compute_continuous_norms),
            ("D",  "Analyse d'Erreur (Discrete)",       None, self._error_discrete),
            ("C",  "Analyse d'Erreur (Continue)",       None, self._error_continuous),
        ]:
            _action_row(body3, icon, label, badge, cmd, pady=2)

        # ── Run button ────────────────────────────────────────────────────────
        run_frame = tk.Frame(parent, bg=COLORS["bg"])
        run_frame.pack(fill="x", pady=(0, 3))
        self._run_btn = _button(run_frame, "  Lancer la methode",
                                bg=COLORS["primary_mid"], bold=True,
                                size=12, pad_y=7,
                                cmd=self._run_method)
        self._run_btn.pack(fill="x")

        # ── Inline result / info boxes ────────────────────────────────────────
        result_area = tk.Frame(parent, bg=COLORS["bg"])
        result_area.pack(fill="x")

        self._result_outer = tk.Frame(result_area, bg=COLORS["primary_lt"],
                                      highlightthickness=1,
                                      highlightbackground="#a8d5b5")
        self._result_lbl = tk.Label(self._result_outer, text="",
                                    bg=COLORS["primary_lt"],
                                    fg=COLORS["success_fg"],
                                    font=("Helvetica", 10, "bold"),
                                    justify="left", wraplength=260, anchor="w")
        self._result_lbl.pack(padx=10, pady=6, anchor="w")

        self._info_outer = tk.Frame(result_area, bg=COLORS["warn_bg"],
                                    highlightthickness=1,
                                    highlightbackground="#f0c040")
        self._info_lbl = tk.Label(self._info_outer, text="",
                                  bg=COLORS["warn_bg"],
                                  fg=COLORS["warn_fg"],
                                  font=("Helvetica", 9),
                                  justify="left", wraplength=260, anchor="w")
        self._info_lbl.pack(padx=10, pady=6, anchor="w")

    def _build_points_input(self):
        for w in self.points_frame.winfo_children():
            w.destroy()
        self.point_entries = []
        for col, lbl in enumerate(["x", "y"]):
            tk.Label(self.points_frame, text=lbl, width=9,
                     bg=COLORS["hdr_bg"], fg=COLORS["text_muted"],
                     font=("Helvetica", 9, "bold"),
                     relief="solid", bd=1).grid(row=0, column=col, sticky="ew")
        defaults_x = [0, 1, 2, 3, 4, 5, 6, 7]
        defaults_y = [0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6]
        for i in range(8):
            for col, val in enumerate([defaults_x[i], defaults_y[i]]):
                e = tk.Entry(self.points_frame, width=9,
                             font=("Courier", 10), justify="center",
                             relief="solid", bd=1, bg=COLORS["surface"])
                e.insert(0, str(val))
                e.grid(row=i+1, column=col, pady=1, padx=1)
                if col == 0:
                    xe = e
                else:
                    self.point_entries.append((xe, e))

    def _update_inputs(self):
        for w in self.extra_frame.winfo_children():
            w.destroy()
        method = self.method_var.get()

        if method == "Moindres Carrés":
            _line(self.extra_frame).pack(fill="x", pady=(0, 6))
            _text(self.extra_frame, "Degre du polynome :",
                  color=COLORS["text_label"]).pack(anchor="w")
            self.degree_var = tk.IntVar(value=2)
            f = tk.Frame(self.extra_frame, bg=COLORS["surface"])
            f.pack(anchor="w", pady=2)
            for d in [1, 2, 3, 4]:
                tk.Radiobutton(f, text=str(d), variable=self.degree_var, value=d,
                               bg=COLORS["surface"], fg=COLORS["text"],
                               selectcolor=COLORS["primary_mid"],
                               font=("Helvetica", 10)).pack(side="left", padx=10)

        elif method == "Chebyshev":
            _line(self.extra_frame).pack(fill="x", pady=(0, 6))
            _text(self.extra_frame, "Intervalle [a, b] :",
                  color=COLORS["text_label"]).pack(anchor="w")
            iv = tk.Frame(self.extra_frame, bg=COLORS["surface"])
            iv.pack(anchor="w", pady=(2, 4))
            self.a_entry = _field(iv, "-1", width=6, mono=True)
            self.a_entry.pack(side="left")
            _text(iv, "  a  ", color=COLORS["text_muted"],
                  bg=COLORS["surface"]).pack(side="left")
            self.b_entry = _field(iv, "1", width=6, mono=True)
            self.b_entry.pack(side="left")

            _text(self.extra_frame, "Degre :",
                  color=COLORS["text_label"]).pack(anchor="w")
            self.cheb_degree = tk.IntVar(value=5)
            f = tk.Frame(self.extra_frame, bg=COLORS["surface"])
            f.pack(anchor="w", pady=2)
            for d in [3, 5, 7, 9]:
                tk.Radiobutton(f, text=str(d), variable=self.cheb_degree, value=d,
                               bg=COLORS["surface"], fg=COLORS["text"],
                               selectcolor=COLORS["primary_mid"],
                               font=("Helvetica", 10)).pack(side="left", padx=10)

        elif method == "Descente de Gradient":
            _line(self.extra_frame).pack(fill="x", pady=(0, 6))

            # Highlighted box — mirrors axe1's x0_outer style
            gd_box = tk.Frame(self.extra_frame, bg="#fffbeb",
                              highlightthickness=1,
                              highlightbackground="#f0c040")
            gd_box.pack(fill="x", pady=(0, 6))
            _text(gd_box, "Fonction f(vars) :", size=9,
                  color=COLORS["warn_fg"], bg="#fffbeb").pack(anchor="w",
                                                               padx=8, pady=(4, 0))
            self.gd_func_entry = _field(gd_box, "(x-1)**2 + 2*(y+2)**2", mono=True)
            self.gd_func_entry.config(bg="#fffdef")
            self.gd_func_entry.pack(fill="x", padx=8, pady=(2, 6))

            for lbl, attr, default in [
                ("Variables :",      "gd_vars_entry",     "x,y"),
                ("x0 initial :",     "gd_x0_entry",       "0,0"),
                ("Learning rate :",  "gd_lr_entry",       "0.01"),
                ("Max iterations :", "gd_max_iter_entry", "1000"),
                ("Tolerance :",      "gd_tol_entry",      "1e-6"),
            ]:
                row = tk.Frame(self.extra_frame, bg=COLORS["surface"])
                row.pack(fill="x", pady=2)
                _text(row, lbl, size=9,
                      color=COLORS["text_label"]).pack(side="left")
                e = _field(row, default, width=14, mono=True)
                e.pack(side="right")
                setattr(self, attr, e)

    # ── Right panel ───────────────────────────────────────────────────────────
    def _build_right(self, parent):
        outer = tk.Frame(parent, bg=COLORS["bg"])
        outer.grid(row=0, column=1, sticky="nsew")
        outer.rowconfigure(0, weight=3)
        outer.rowconfigure(1, weight=2)
        outer.columnconfigure(0, weight=1)

        # ── Plot card ─────────────────────────────────────────────────────────
        plot_card = tk.Frame(outer, bg=COLORS["border"])
        plot_card.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        inner_p = tk.Frame(plot_card, bg=COLORS["surface"])
        inner_p.pack(fill="both", expand=True, padx=1, pady=1)
        inner_p.rowconfigure(2, weight=1)
        inner_p.columnconfigure(0, weight=1)

        phdr = tk.Frame(inner_p, bg=COLORS["hdr_bg"], height=30)
        phdr.grid(row=0, column=0, sticky="ew")
        phdr.grid_propagate(False)
        _text(phdr, "Graphe de l'approximation", size=10, bold=True,
              bg=COLORS["hdr_bg"]).pack(side="left", padx=10, pady=6)

        dl = tk.Frame(phdr, bg=COLORS["hdr_bg"])
        dl.pack(side="right", padx=8)
        _button(dl, "  CSV", bg=COLORS["purple"], pad_x=10, pad_y=3,
                size=9, cmd=self._download_table).pack(side="right", padx=3)

        _line(inner_p).grid(row=1, column=0, sticky="ew")

        self.plot_frame = tk.Frame(inner_p, bg=COLORS["surface"])
        self.plot_frame.grid(row=2, column=0, sticky="nsew")

        _line(inner_p).grid(row=3, column=0, sticky="ew")

        # Stats bar ────────────────────────────────────────────────────────────
        stats_bar = tk.Frame(inner_p, bg=COLORS["hdr_bg"])
        stats_bar.grid(row=4, column=0, sticky="ew")
        stats_bar.columnconfigure((0, 1, 2), weight=1)
        self._stat_vars = {}
        for i, (key, lbl) in enumerate(STATS_DEF):
            f = tk.Frame(stats_bar, bg=COLORS["hdr_bg"])
            f.grid(row=0, column=i, padx=8, pady=6, sticky="ew")
            if i > 0:
                _line(stats_bar, vertical=True).grid(
                    row=0, column=i, sticky="ns")
            v = tk.StringVar(value="--")
            self._stat_vars[key] = v
            tk.Label(f, textvariable=v, bg=COLORS["hdr_bg"],
                     fg=COLORS["primary_mid"],
                     font=("Helvetica", 13, "bold")).pack()
            tk.Label(f, text=lbl, bg=COLORS["hdr_bg"],
                     fg=COLORS["text_muted"],
                     font=("Helvetica", 8)).pack()

        # ── Polynomial / result text card ─────────────────────────────────────
        tbl_card = tk.Frame(outer, bg=COLORS["border"])
        tbl_card.grid(row=1, column=0, sticky="nsew")
        inner_t = tk.Frame(tbl_card, bg=COLORS["surface"])
        inner_t.pack(fill="both", expand=True, padx=1, pady=1)
        inner_t.rowconfigure(2, weight=1)
        inner_t.columnconfigure(0, weight=1)

        thdr = tk.Frame(inner_t, bg=COLORS["hdr_bg"], height=30)
        thdr.grid(row=0, column=0, sticky="ew")
        thdr.grid_propagate(False)
        _text(thdr, "Polynome et Resultat", size=10, bold=True,
              bg=COLORS["hdr_bg"]).pack(side="left", padx=10, pady=6)
        self._conv_lbl = tk.Label(thdr, text="", bg=COLORS["hdr_bg"],
                                  font=("Helvetica", 8), padx=5)
        self._conv_lbl.pack(side="left")

        _line(inner_t).grid(row=1, column=0, sticky="ew")

        self.poly_text = tk.Text(inner_t, height=7,
                                 font=("Courier", 10),
                                 bg="#f8f5ff", fg=COLORS["text"],
                                 relief="flat", padx=10, pady=8,
                                 wrap="word")
        self.poly_text.grid(row=2, column=0, sticky="nsew")

        sb = tk.Scrollbar(inner_t, command=self.poly_text.yview)
        sb.grid(row=2, column=1, sticky="ns")
        self.poly_text.configure(yscrollcommand=sb.set)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _paste_func(self, text):
        self.func_entry.delete(0, "end")
        self.func_entry.insert(0, text)

    def _set_pill(self, text, kind="idle"):
        colors = {
            "idle":    COLORS["primary_mid"],
            "ok":      COLORS["green"],
            "warn":    COLORS["orange"],
            "error":   COLORS["red"],
            "running": COLORS["blue"],
        }
        self._pill.config(bg=colors.get(kind, COLORS["primary_mid"]))
        self._pill_var.set(text)

    def _show_result(self, text, kind="ok"):
        cfg = {
            "ok":   (COLORS["primary_lt"], COLORS["success_fg"], "#a8d5b5"),
            "warn": (COLORS["warn_bg"],    COLORS["warn_fg"],    "#f0c040"),
            "err":  (COLORS["err_bg"],     COLORS["err_fg"],     "#f5b7b7"),
        }
        bg, fg, border = cfg.get(kind, cfg["ok"])
        self._result_outer.config(bg=bg, highlightbackground=border)
        self._result_lbl.config(text=text, bg=bg, fg=fg)
        self._result_outer.pack(fill="x", pady=(4, 0))
        self._info_outer.pack_forget()

    def _show_info(self, text):
        self._info_lbl.config(text=text)
        self._info_outer.pack(fill="x", pady=(4, 0))
        self._result_outer.pack_forget()

    def _set_stats(self, mse_s, deg_s, nodes_s):
        self._stat_vars["mse"].set(mse_s)
        self._stat_vars["deg"].set(str(deg_s))
        self._stat_vars["nodes"].set(str(nodes_s))

    def _set_conv_badge(self, ok, label="calcule"):
        if ok:
            self._conv_lbl.config(text=label, bg="#e8f5e9",
                                  fg=COLORS["success_fg"])
        else:
            self._conv_lbl.config(text="non converge",
                                  bg=COLORS["err_bg"], fg=COLORS["err_fg"])

    def _show_error(self, title, text):
        show_dialog(self, title, text, "error")
        self._set_pill("erreur", "error")

    # ── Run ───────────────────────────────────────────────────────────────────
    def _run_method(self):
        self._result_outer.pack_forget()
        self._info_outer.pack_forget()
        self._set_pill("calcul...", "running")
        self.update()

        try:
            method = self.method_var.get()
            x_data = [float(xe.get()) for xe, ye in self.point_entries
                      if xe.get().strip() and ye.get().strip()]
            y_data = [float(ye.get()) for xe, ye in self.point_entries
                      if xe.get().strip() and ye.get().strip()]
            x_sym  = sp.symbols('x')

            result_text = ""
            y_approx    = None
            real_func   = None
            mse_s = deg_s = nodes_s = "--"

            if method in ("Lagrange", "Newton", "Moindres Carrés") and len(x_data) < 2:
                show_dialog(self, "Entree invalide",
                            "Au moins 2 points requis.", "warn")
                self._set_pill("erreur", "error")
                return

            if method == "Lagrange":
                poly, _     = lagrange_interpolation(x_data, y_data)
                result_text = str(poly)
                y_approx    = sp.lambdify(x_sym, poly, "numpy")
                nodes_s     = len(x_data)
                deg_s       = len(x_data) - 1

            elif method == "Newton":
                diff_table  = newton_differences(x_data, y_data)
                poly        = newton_polynomial(x_data, diff_table)
                result_text = str(poly)
                y_approx    = sp.lambdify(x_sym, poly, "numpy")
                nodes_s     = len(x_data)
                deg_s       = len(x_data) - 1

            elif method == "Moindres Carrés":
                degree      = self.degree_var.get()
                func_str    = self.func_entry.get()
                func_sym    = sp.sympify(func_str)
                real_func   = sp.lambdify(x_sym, func_sym, "numpy")
                res         = least_squares_polynomial(x_data, y_data, degree)
                result_text = str(res["poly_sympy"])
                y_approx    = res["poly_numpy"]
                mse_s       = f"{res['mse']:.3e}"
                deg_s       = degree
                nodes_s     = len(x_data)

            elif method == "Chebyshev":
                a         = float(self.a_entry.get())
                b         = float(self.b_entry.get())
                deg       = self.cheb_degree.get()
                func_str  = self.func_entry.get()
                func_sym  = sp.sympify(func_str)
                real_func = sp.lambdify(x_sym, func_sym, "numpy")
                res       = chebyshev_approximation(real_func, deg, a, b)
                result_text = str(res["poly_sympy"])
                y_approx  = res["evaluate"]
                deg_s     = deg
                nodes_s   = deg + 1
                x_data    = list(np.linspace(a, b, deg + 1))
                y_data    = [float(real_func(xi)) for xi in x_data]
                self._last_interval = (a, b)

            elif method == "Descente de Gradient":
                func_str = self.gd_func_entry.get()
                vars_str = self.gd_vars_entry.get()
                x0_str   = self.gd_x0_entry.get()
                lr       = float(self.gd_lr_entry.get())
                max_iter = int(self.gd_max_iter_entry.get())
                tol      = float(self.gd_tol_entry.get())

                var_names = [s.strip() for s in vars_str.split(',') if s.strip()]
                if not var_names:
                    show_dialog(self, "Erreur",
                                "Entrez au moins une variable.", "error")
                    self._set_pill("erreur", "error")
                    return

                vars_syms = sp.symbols(' '.join(var_names))
                if len(var_names) == 1:
                    vars_syms = (vars_syms,)

                local_dict = {n: v for n, v in zip(var_names, vars_syms)}
                func_sympy = sp.sympify(func_str, locals=local_dict)

                x0 = [float(s.strip()) for s in x0_str.split(',') if s.strip()]
                if len(x0) != len(var_names):
                    show_dialog(self, "Erreur",
                                "x0 doit avoir autant d'elements que les variables.",
                                "error")
                    self._set_pill("erreur", "error")
                    return

                x_opt, history = gradient_descent_sympy(
                    func_sympy, vars_syms, x0,
                    lr=lr, max_iter=max_iter, tol=tol)

                func_eval = sp.lambdify(vars_syms, func_sympy, modules="numpy")
                obj_vals  = [float(func_eval(*np.atleast_1d(xk)))
                             for xk, _ in history]

                vals        = [float(v) for v in np.atleast_1d(x_opt)]
                result_text = (f"x* = [{', '.join(f'{v:.8g}' for v in vals)}]\n"
                               f"f(x*) = {obj_vals[-1]:.6g}")
                mse_s   = f"{obj_vals[-1]:.3e}"
                deg_s   = len(history)
                nodes_s = len(var_names)
                y_approx = None

            # ── Update UI ─────────────────────────────────────────────────────
            self.poly_text.delete("1.0", tk.END)
            prefix = "" if method == "Descente de Gradient" else "P(x) = "
            self.poly_text.insert("1.0", prefix + result_text)

            self._last_y_approx  = y_approx
            self._last_real_func = real_func
            if x_data:
                self._last_interval = (min(x_data), max(x_data))

            self._set_stats(mse_s, deg_s, nodes_s)

            if method == "Descente de Gradient":
                self._plot_gradient_descent(history, obj_vals)
                converged = history[-1][1] <= tol
                self._set_conv_badge(converged,
                                     "converge" if converged else "non converge")
                self._show_result(result_text, "ok" if converged else "warn")
            else:
                self._plot_approx(x_data, y_data, y_approx, method, real_func)
                self._set_conv_badge(True, "calcule")
                self._show_result(
                    f"P(x) calcule  ({nodes_s} noeuds, degre {deg_s})", "ok")

            self._set_pill("calcule", "ok")

        except Exception as e:
            self._show_error("Erreur d'execution", str(e))

    # ── Plots ─────────────────────────────────────────────────────────────────
    def _plot_approx(self, x_data, y_data, y_approx, method, real_func=None):
        for w in self.plot_frame.winfo_children():
            w.destroy()

        fig = Figure(figsize=(9, 4), dpi=100)
        fig.patch.set_facecolor(COLORS["surface"])
        ax = fig.add_subplot(111)
        ax.set_facecolor("#fafafa")
        for spine in ax.spines.values():
            spine.set_color(COLORS["border"])
        ax.tick_params(colors=COLORS["text_muted"], labelsize=8)

        x_plot = np.linspace(min(x_data) - 0.5, max(x_data) + 0.5, 500)

        if real_func is not None:
            y_real = np.array([float(real_func(xi)) for xi in x_plot])
            ax.plot(x_plot, y_real, color=COLORS["blue"],
                    linewidth=1.5, linestyle="--", alpha=0.7,
                    label="f(x) reelle", zorder=2)

        if y_approx is not None:
            y_plot = np.array([float(y_approx(xi)) for xi in x_plot])
            ax.plot(x_plot, y_plot, color=COLORS["primary_mid"],
                    linewidth=2.2, label=f"{method} P(x)", zorder=3)

        ax.scatter(x_data, y_data, color=COLORS["red"],
                   s=55, zorder=5, label="Points", edgecolors="white", linewidths=0.6)

        ax.set_title(f"Approximation — {method}",
                     color=COLORS["text"], fontsize=10, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.2, color=COLORS["border"])
        fig.tight_layout(pad=1.2)

        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._graph_data_export = (x_data, y_data, y_approx, real_func)

    def _plot_gradient_descent(self, history, obj_vals):
        for w in self.plot_frame.winfo_children():
            w.destroy()

        fig = Figure(figsize=(9, 4), dpi=100)
        fig.patch.set_facecolor(COLORS["surface"])
        ax1 = fig.add_subplot(111)
        ax1.set_facecolor("#fafafa")
        for spine in ax1.spines.values():
            spine.set_color(COLORS["border"])

        iters      = np.arange(len(history))
        grad_norms = [gn for _, gn in history]

        ax1.semilogy(iters, obj_vals, color=COLORS["primary_mid"],
                     marker="o", markersize=3, linewidth=1.8, label="f(x)")
        ax1.set_xlabel("Iteration", color=COLORS["text_muted"], fontsize=9)
        ax1.set_ylabel("Valeur objectif", color=COLORS["primary_mid"], fontsize=9)
        ax1.tick_params(colors=COLORS["text_muted"], labelsize=8)
        ax1.grid(True, alpha=0.2)

        ax2 = ax1.twinx()
        ax2.semilogy(iters, grad_norms, color=COLORS["orange"],
                     linestyle="--", marker="s", markersize=3,
                     linewidth=1.5, label="||gf||")
        ax2.set_ylabel("Norme du gradient", color=COLORS["orange"], fontsize=9)
        ax2.tick_params(colors=COLORS["text_muted"], labelsize=8)

        ax1.set_title("Convergence — Descente de Gradient",
                      color=COLORS["text"], fontsize=10, fontweight="bold")
        lines1, lab1 = ax1.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=9, framealpha=0.9)
        fig.tight_layout(pad=1.2)

        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── Norms & Error buttons ─────────────────────────────────────────────────
    def _compute_discrete_norms(self):
        try:
            y_values = [float(ye.get()) for _, ye in self.point_entries
                        if ye.get().strip()]
            if not y_values:
                show_dialog(self, "Erreur", "Entrez des valeurs Y.", "warn")
                return
            r   = all_discrete_norms(y_values)
            msg = (f"L1  = {r['L1']:.6g}\n"
                   f"L2  = {r['L2']:.6g}\n"
                   f"Linf = {r['Linf']:.6g}")
            self._show_info(f"Normes discretes :\n{msg}")
            show_dialog(self, "Normes Discretes", msg, "info")
        except Exception as e:
            self._show_error("Erreur", str(e))

    def _compute_continuous_norms(self):
        try:
            func_str = self.func_entry.get().strip() or "cos(x)"
            f_sym    = sp.sympify(func_str)
            f_np     = sp.lambdify(sp.symbols('x'), f_sym, "numpy")
            a, b     = self._last_interval
            r        = all_continuous_norms(f_np, a, b)
            msg = (f"Sur [{a:.4g}, {b:.4g}] :\n"
                   f"L1   = {r['L1']:.6g}\n"
                   f"L2   = {r['L2']:.6g}\n"
                   f"Linf = {r['Linf']:.6g}")
            self._show_info(msg)
            show_dialog(self, "Normes Continues", msg, "info")
        except Exception as e:
            self._show_error("Erreur", str(e))

    def _error_discrete(self):
        try:
            x_vals = [float(xe.get()) for xe, ye in self.point_entries
                      if xe.get().strip() and ye.get().strip()]
            y_data = [float(ye.get()) for xe, ye in self.point_entries
                      if xe.get().strip() and ye.get().strip()]
            if len(y_data) < 2:
                show_dialog(self, "Erreur", "Au moins 2 points requis.", "warn")
                return
            if self._last_y_approx is not None:
                y_approx = [float(self._last_y_approx(xi)) for xi in x_vals]
            else:
                from scipy.interpolate import interp1d
                fi = interp1d(range(len(y_data)), y_data, kind='linear')
                y_approx = [float(fi(i)) for i in range(len(y_data))]
            r   = approximation_error_discrete(y_data, y_approx)
            msg = (f"L1   = {r['L1']:.6g}\n"
                   f"L2   = {r['L2']:.6g}\n"
                   f"Linf = {r['Linf']:.6g}  "
                   f"(index {r['Linf_index']},  val={r['Linf_value']:.6g})")
            self._show_info(f"Erreur discrete :\n{msg}")
            show_dialog(self, "Erreur Discrete", msg, "info")
        except Exception as e:
            self._show_error("Erreur", str(e))

    def _error_continuous(self):
        try:
            if self._last_y_approx is None:
                show_dialog(self, "Erreur",
                            "Lancez d'abord une methode pour obtenir P(x).",
                            "warn")
                return
            if self._last_real_func is None:
                show_dialog(self, "Erreur",
                            "L'erreur continue necessite une fonction reelle.\n"
                            "Utilisez Moindres Carrés ou Chebyshev.",
                            "warn")
                return
            a, b = self._last_interval
            r    = approximation_error_continuous(
                       self._last_real_func, self._last_y_approx, a, b)
            msg = (f"Intervalle : [{a:.4g}, {b:.4g}]\n"
                   f"L1   = {r['L1']:.6g}\n"
                   f"L2   = {r['L2']:.6g}\n"
                   f"Linf = {r['Linf']:.6g}  (x = {r['Linf_xmax']:.6g})")
            self._show_info(f"Erreur continue :\n{msg}")
            show_dialog(self, "Erreur Continue", msg, "info")
        except Exception as e:
            self._show_error("Erreur", str(e))

    # ── Export ────────────────────────────────────────────────────────────────
    def _download_table(self):
        if self._graph_data_export is None:
            show_dialog(self, "Info", "Lancez d'abord une methode.", "info")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Texte", "*.txt")],
            title="Exporter les donnees")
        if not path:
            return
        try:
            x_data, y_data, y_approx, real_func = self._graph_data_export
            with open(path, "w", encoding="utf-8") as fp:
                header = "x;y_data;y_approx"
                if real_func is not None:
                    header += ";y_real"
                fp.write(header + "\n")
                for xi, yi in zip(x_data, y_data):
                    row = f"{xi:.10g};{yi:.10g}"
                    if y_approx is not None:
                        try:
                            row += f";{float(y_approx(xi)):.10g}"
                        except Exception:
                            row += ";NaN"
                    if real_func is not None:
                        try:
                            row += f";{float(real_func(xi)):.10g}"
                        except Exception:
                            row += ";NaN"
                    fp.write(row + "\n")
            show_dialog(self, "Succes",
                        f"Fichier sauvegarde :\n{path}", "success")
        except Exception as ex:
            self._show_error("Erreur", str(ex))

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
    Axe3Screen().mainloop()