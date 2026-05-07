import importlib.util
import math
import os
import tkinter as tk
from tkinter import filedialog

import numpy as np

from ui.axe1_widgets import (
    COLORS,
    GraphCanvas,
    TableCanvas,
    _button,
    _field,
    _line,
    _panel,
    _text,
    show_dialog,
)

_mod_path = os.path.join(os.path.dirname(__file__),
                          "modules", "chap 1", "nonlinear_resolution.py")
_spec = importlib.util.spec_from_file_location("nonlinear_resolution", _mod_path)
_nl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nl)

_analysis_path = os.path.join(os.path.dirname(__file__),
                              "modules", "chap 1", "function_analysis.py")
_analysis_spec = importlib.util.spec_from_file_location("function_analysis", _analysis_path)
_analysis = importlib.util.module_from_spec(_analysis_spec)
_analysis_spec.loader.exec_module(_analysis)

COLS = {
    "Dichotomie": ("n", "an", "bn", "mn", "f(mn)", "eps"),
    "Point Fixe": ("n", "xn", "xn+1", "|xn+1 - xn|"),
    "Newton":     ("n", "xn", "xn+1", "|xn+1 - xn|"),
}

ALGO_TAG = {
    "Dichotomie": "Bissection",
    "Point Fixe": "Iteration",
    "Newton":     "Tangente",
}

ALGORITHMS = ("Dichotomie", "Point Fixe", "Newton")
EXAMPLES = ("sin(x)", "cos(x)", "exp(x)", "sqrt(x)")
GRAPH_LEGEND = (
    ("f(x)", "primary_mid"),
    ("a", "red"),
    ("b", "blue"),
    ("racine", "orange"),
)
STATS = (
    ("racine", "racine"),
    ("iter", "iterations"),
    ("err", "erreur finale"),
)
SAFE_FUNCTIONS = {
    "np": np,
    "math": math,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "pi": math.pi,
    "e": math.e,
}

class Axe1Screen(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Axe 1 — Function Analysis")
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        w = min(1020, sw - 80)
        h = min(720, sh - 80)
        self.geometry(f"{w}x{h}")
        self.minsize(min(880, w), min(600, h))
        self.configure(bg=COLORS["bg"])

        self._last_table_data = []
        self._last_table_cols = []
        self._graph_data = None

        self._build()

    def _build(self):
        self._build_header()
        content = tk.Frame(self, bg=COLORS["bg"])
        content.pack(fill="both", expand=True, padx=12, pady=10)
        content.columnconfigure(0, minsize=268, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)
        self._build_left(content)
        self._build_right(content)

    def _build_header(self):
        bar = tk.Frame(self, bg=COLORS["primary"], height=52)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        back = _button(bar, "  Retour", bg=COLORS["primary_btn"],
                    cmd=self._back, pad_x=14, pad_y=8)
        back.pack(side="left", padx=12, pady=8)

        _text(bar, "Axe 1 — Equations non-lineaires",
             size=14, bold=True, color="white",
             bg=COLORS["primary"]).pack(side="left", padx=8)

        self._pill_var = tk.StringVar(value="")
        self._pill = tk.Label(bar, textvariable=self._pill_var,
                               bg=COLORS["primary_mid"], fg="white",
                               font=("Helvetica", 9), padx=12, pady=4)
        self._pill.pack(side="right", padx=14)

    def _build_left(self, parent):
        container = tk.Frame(parent, bg=COLORS["bg"])
        container.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        _lc = tk.Canvas(container, bg=COLORS["bg"], highlightthickness=0)
        _vsb = tk.Scrollbar(container, orient="vertical", command=_lc.yview)
        _lc.configure(yscrollcommand=_vsb.set)
        _lc.grid(row=0, column=0, sticky="nsew")
        _vsb.grid(row=0, column=1, sticky="ns")

        outer = tk.Frame(_lc, bg=COLORS["bg"])
        _win = _lc.create_window((0, 0), window=outer, anchor="nw")

        outer.bind("<Configure>", lambda e: _lc.configure(scrollregion=_lc.bbox("all")))
        _lc.bind("<Configure>", lambda e: _lc.itemconfig(_win, width=e.width))
        _lc.bind("<MouseWheel>",
                 lambda e: _lc.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        card, body, _ = _panel(outer, "  Entrees", dot=COLORS["primary_mid"], compact=True)
        card.pack(fill="x", pady=(0, 5))

        _text(body, "Formule f(x)", color=COLORS["text_label"]).pack(anchor="w")
        self.func_entry = _field(body, "x**3 - x - 2", mono=True)
        self.func_entry.pack(fill="x", pady=(2, 4))

        _text(body, "Exemples :", size=9,
             color=COLORS["text_muted"]).pack(anchor="w")
        sc_row = tk.Frame(body, bg=COLORS["surface"])
        sc_row.pack(fill="x", pady=(2, 4))
        for ex in EXAMPLES:
            b = tk.Button(sc_row, text=ex, bg=COLORS["hdr_bg"],
                          fg=COLORS["text_muted"], font=("Courier", 8),
                          relief="solid", bd=1, cursor="hand2",
                          padx=4, pady=2,
                          command=lambda v=ex: self._paste_func(v))
            b.pack(side="left", padx=2)

        _line(body).pack(fill="x", pady=4)

        _text(body, "Intervalle [a, b]", color=COLORS["text_label"]).pack(anchor="w")
        iv = tk.Frame(body, bg=COLORS["surface"])
        iv.pack(fill="x", pady=(2, 4))
        self.a_entry = _field(iv, "1", width=7, mono=True)
        self.a_entry.pack(side="left")
        _text(iv, "  →  ", color=COLORS["text_muted"],
             bg=COLORS["surface"]).pack(side="left")
        self.b_entry = _field(iv, "2", width=7, mono=True)
        self.b_entry.pack(side="left")

        _text(body, "Tolerance (eps)", color=COLORS["text_label"]).pack(anchor="w")
        self.tol_entry = _field(body, "1e-6", width=14, mono=True)
        self.tol_entry.pack(anchor="w", pady=(2, 0))

        self.x0_outer = tk.Frame(body, bg="#fffbeb",
                                  highlightthickness=1,
                                  highlightbackground="#f0c040")
        _text(self.x0_outer, "Point initial x0 :",
             size=9, color=COLORS["warn_fg"],
             bg="#fffbeb").pack(anchor="w", padx=8, pady=(4, 0))
        self.x0_entry = _field(self.x0_outer, "1.5", width=12, mono=True)
        self.x0_entry.config(bg="#fffdef")
        self.x0_entry.pack(anchor="w", padx=8, pady=(2, 6))

        card2, body2, _ = _panel(outer, "  Algorithme", dot=COLORS["primary_mid"], compact=True)
        card2.pack(fill="x", pady=(0, 5))

        self.algo_var = tk.StringVar(value="Dichotomie")
        for algo in ALGORITHMS:
            row = tk.Frame(body2, bg=COLORS["surface"])
            row.pack(fill="x", pady=2)
            rb = tk.Radiobutton(row, text=algo, variable=self.algo_var,
                                value=algo, bg=COLORS["surface"], fg=COLORS["text"],
                                font=("Helvetica", 11),
                                activebackground=COLORS["surface"],
                                selectcolor=COLORS["primary_mid"],
                                command=self._on_algo_change)
            rb.pack(side="left")
            tag_lbl = tk.Label(row, text=ALGO_TAG[algo],
                               bg=COLORS["primary_lt"], fg=COLORS["success_fg"],
                               font=("Helvetica", 8), padx=6, pady=1)
            tag_lbl.pack(side="right")

        card3, body3, _ = _panel(outer, "  Analyser f(x)", dot=COLORS["blue"], compact=True)
        card3.pack(fill="x", pady=(0, 5))

        for icon, label, badge, cmd in [
            ("f'",  "Derivees f'(x), f''(x)", None,  self._calc_derivative),
            ("~",   "Continuite sur [a, b]",   None,  self._verify_continuity),
            ("g",   "Stabilite de g(x)",        "k",   self._check_stability),
            ("|k|", "Contractante sur [a, b]",  "k<1", self._check_contractante),
        ]:
            self._action_row(body3, icon, label, badge, cmd, pady=1)

        run_frame = tk.Frame(outer, bg=COLORS["bg"])
        run_frame.pack(fill="x", pady=(0, 3))
        self._run_btn = _button(run_frame, "  Lancer l'algorithme",
                              bg=COLORS["primary_mid"], bold=True,
                              size=12, pad_y=7,
                              cmd=self._run_algorithm)
        self._run_btn.pack(fill="x")

        # Result/info boxes are outside the scrollable area so they never add scroll height
        result_area = tk.Frame(container, bg=COLORS["bg"])
        result_area.grid(row=1, column=0, columnspan=2, sticky="ew")

        self._result_outer = tk.Frame(result_area, bg=COLORS["primary_lt"],
                                       highlightthickness=1,
                                       highlightbackground="#a8d5b5")
        self._result_lbl = tk.Label(self._result_outer, text="",
                                     bg=COLORS["primary_lt"],
                                     fg=COLORS["success_fg"],
                                     font=("Helvetica", 10, "bold"),
                                     justify="left", wraplength=250,
                                     anchor="w")
        self._result_lbl.pack(padx=10, pady=6, anchor="w")

        self._info_outer = tk.Frame(result_area, bg=COLORS["warn_bg"],
                                     highlightthickness=1,
                                     highlightbackground="#f0c040")
        self._info_lbl = tk.Label(self._info_outer, text="",
                                   bg=COLORS["warn_bg"],
                                   fg=COLORS["warn_fg"],
                                   font=("Helvetica", 9),
                                   justify="left", wraplength=250,
                                   anchor="w")
        self._info_lbl.pack(padx=10, pady=6, anchor="w")

    def _action_row(self, parent, icon, label, badge, cmd, pady=2):
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

    def _build_right(self, parent):
        outer = tk.Frame(parent, bg=COLORS["bg"])
        outer.grid(row=0, column=1, sticky="nsew")
        outer.rowconfigure(0, weight=2)
        outer.rowconfigure(1, weight=3)
        outer.columnconfigure(0, weight=1)

        graph_card = tk.Frame(outer, bg=COLORS["border"])
        graph_card.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        inner_g = tk.Frame(graph_card, bg=COLORS["surface"])
        inner_g.pack(fill="both", expand=True, padx=1, pady=1)
        inner_g.rowconfigure(2, weight=1)
        inner_g.columnconfigure(0, weight=1)

        ghdr = tk.Frame(inner_g, bg=COLORS["hdr_bg"], height=30)
        ghdr.grid(row=0, column=0, sticky="ew")
        ghdr.grid_propagate(False)
        _text(ghdr, "Graphe de f(x)", size=10, bold=True,
             bg=COLORS["hdr_bg"]).pack(side="left", padx=10, pady=6)
        lgd = tk.Frame(ghdr, bg=COLORS["hdr_bg"])
        lgd.pack(side="right", padx=10)
        for txt, color_key in GRAPH_LEGEND:
            col = COLORS[color_key]
            tk.Label(lgd, text=txt, bg=col, fg="white",
                     font=("Helvetica", 8), padx=7, pady=2
                     ).pack(side="left", padx=2)

        _line(inner_g).grid(row=1, column=0, sticky="ew")

        self._graph = GraphCanvas(inner_g)
        self._graph.grid(row=2, column=0, sticky="nsew", padx=8, pady=6)

        _line(inner_g).grid(row=3, column=0, sticky="ew")
        stats_bar = tk.Frame(inner_g, bg=COLORS["hdr_bg"])
        stats_bar.grid(row=4, column=0, sticky="ew")
        stats_bar.columnconfigure((0, 1, 2), weight=1)

        self._stat_vars = {}
        for i, (key, lbl) in enumerate(STATS):
            f = tk.Frame(stats_bar, bg=COLORS["hdr_bg"])
            f.grid(row=0, column=i, padx=8, pady=6, sticky="ew")
            if i > 0:
                _line(stats_bar, vertical=True).grid(
                    row=0, column=i, sticky="ns", padx=(0, 0))
            v = tk.StringVar(value="—")
            self._stat_vars[key] = v
            tk.Label(f, textvariable=v, bg=COLORS["hdr_bg"],
                     fg=COLORS["primary_mid"],
                     font=("Helvetica", 13, "bold")).pack()
            tk.Label(f, text=lbl, bg=COLORS["hdr_bg"],
                     fg=COLORS["text_muted"],
                     font=("Helvetica", 8)).pack()

        tbl_card = tk.Frame(outer, bg=COLORS["border"])
        tbl_card.grid(row=1, column=0, sticky="nsew")
        inner_t = tk.Frame(tbl_card, bg=COLORS["surface"])
        inner_t.pack(fill="both", expand=True, padx=1, pady=1)
        inner_t.rowconfigure(2, weight=1)
        inner_t.columnconfigure(0, weight=1)

        thdr = tk.Frame(inner_t, bg=COLORS["hdr_bg"], height=30)
        thdr.grid(row=0, column=0, sticky="ew")
        thdr.grid_propagate(False)
        _text(thdr, "Tableau des iterations", size=10, bold=True,
             bg=COLORS["hdr_bg"]).pack(side="left", padx=10, pady=6)
        self._conv_lbl = tk.Label(thdr, text="",
                                   bg=COLORS["hdr_bg"],
                                   font=("Helvetica", 8), padx=5)
        self._conv_lbl.pack(side="left")

        dl = tk.Frame(thdr, bg=COLORS["hdr_bg"])
        dl.pack(side="right", padx=8)
        for txt, col, cmd in [
            ("CSV",   COLORS["purple"], self._download_table),
            ("Graph", COLORS["blue"],   self._download_graph),
        ]:
            _button(dl, f"  {txt}", bg=col, pad_x=10, pad_y=3,
                 size=9, cmd=cmd).pack(side="right", padx=3)

        _line(inner_t).grid(row=1, column=0, sticky="ew")

        self._table = TableCanvas(inner_t)
        self._table.grid(row=2, column=0, sticky="nsew", padx=0, pady=0)

    def _paste_func(self, text):
        self.func_entry.delete(0, "end")
        self.func_entry.insert(0, text)

    def _on_algo_change(self):
        algo = self.algo_var.get()
        if algo in ("Point Fixe", "Newton"):
            self.x0_outer.pack(fill="x", pady=(6, 0))
        else:
            self.x0_outer.pack_forget()

    def _set_pill(self, text, kind="idle"):
        colors = {
            "idle":    COLORS["primary_mid"],
            "ok":      "#27ae60",
            "warn":    "#e67e22",
            "error":   "#e74c3c",
            "running": "#2980b9",
        }
        self._pill.config(text=text, bg=colors.get(kind, COLORS["primary_mid"]))
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

    def _set_stats(self, root_s, iter_s, err_s):
        self._stat_vars["racine"].set(root_s)
        self._stat_vars["iter"].set(str(iter_s))
        self._stat_vars["err"].set(err_s)

    def _set_conv_badge(self, converged):
        if converged:
            self._conv_lbl.config(text="converge",
                                   bg="#e8f5e9", fg=COLORS["success_fg"])
        else:
            self._conv_lbl.config(text="non converge",
                                   bg=COLORS["err_bg"], fg=COLORS["err_fg"])

    def _show_error(self, title, text):
        show_dialog(self, title, text, "error")
        self._set_pill("erreur", "error")

    def _show_dialog(self, title, text, kind="info"):
        show_dialog(self, title, text, kind)

    def _read_x0(self):
        try:
            return float(self.x0_entry.get())
        except ValueError:
            raise ValueError("x0 doit etre un nombre.")

    def _finish_algorithm(self, f, a, b, root, err, rows, cols, converged,
                          note=None):
        self._fill_table(cols, rows, converged)
        self._set_stats(f"{root:.7g}", len(rows), f"{err:.2e}")

        msg = f"Racine  {root:.8g}   ({len(rows)} iterations)"
        if note:
            msg += f"\n{note}"
        self._show_result(msg, "ok" if converged else "warn")
        self._plot(f, a, b, root)
        self._set_pill("converge" if converged else "non converge",
                       "ok" if converged else "warn")

    def _parse_function(self):
        expr = self.func_entry.get().strip()
        try:
            code = compile(expr, "<fonction>", "eval")
            env = dict(SAFE_FUNCTIONS)
            env["x"] = 1.0
            eval(code, {"__builtins__": {}}, env)
        except Exception as ex:
            raise ValueError(f"Expression invalide : {ex}")

        def f(x):
            env = dict(SAFE_FUNCTIONS)
            env["x"] = x
            return eval(code, {"__builtins__": {}}, env)
        return f

    def _parse_inputs(self):
        try:
            a = float(self.a_entry.get())
            b = float(self.b_entry.get())
            tol = float(self.tol_entry.get())
        except ValueError:
            raise ValueError("a, b et eps doivent etre des nombres valides.")
        if a >= b:
            raise ValueError("Il faut a < b.")
        if tol <= 0:
            raise ValueError("La tolerance doit etre > 0.")
        return a, b, tol

    def _plot(self, f, a, b, root=None):
        marge = abs(b - a) * 0.35
        xs = np.linspace(a - marge, b + marge, 400)
        try:
            ys = np.array([f(xi) for xi in xs], dtype=float)
        except Exception:
            return
        self._graph_data = (xs, ys, a, b, root)
        self._graph.plot(xs, ys, a, b, root)

    def _fill_table(self, cols, rows, converged=True):
        self._last_table_cols = list(cols)
        self._last_table_data = rows
        self._table.set_data(cols, rows)
        self._set_conv_badge(converged)

    def _run_algorithm(self):
        self._result_outer.pack_forget()
        self._info_outer.pack_forget()
        self._set_pill("calcul...", "running")
        self.update()

        try:
            f = self._parse_function()
            a, b, tol = self._parse_inputs()
            algo = self.algo_var.get()
        except ValueError as e:
            self._show_error("Erreur", str(e))
            return

        try:
            if algo == "Dichotomie":
                self._run_bisection(f, a, b, tol)
            elif algo == "Point Fixe":
                self._run_fixed_point(f, a, b, tol)
            elif algo == "Newton":
                self._run_newton(f, a, b, tol)
        except ValueError as e:
            self._show_error("Erreur", str(e))

    def _run_bisection(self, f, a, b, tol):
        fa = f(a)
        fb = f(b)
        if fa * fb >= 0:
            self._show_error(
                "Dichotomie impossible",
                f"f(a) x f(b) doit etre < 0.\n"
                f"f({a}) = {fa:.4f},  f({b}) = {fb:.4f}\n"
                "Choisissez un intervalle contenant une racine.")
            return

        rows = _nl.solve_bisection_detail(a, b, eps=tol, max_it=100, func=f)

        root = rows[-1][3]
        err = rows[-1][5]
        self._finish_algorithm(
            f, a, b, root, err, rows, COLS["Dichotomie"], err < tol)

    def _run_fixed_point(self, f, a, b, tol):
        x0 = self._read_x0()
        g = lambda x: x - f(x)
        history = _nl.executer_point_fixe(g, x0, epsilon=tol, max_iter=50)

        if not history:
            self._show_dialog("Info", "Aucune iteration produite.", "warn")
            self._set_pill("erreur", "error")
            return

        rows = [(r[0], r[1], r[2], r[3]) for r in history]
        root = history[-1][2]
        err = history[-1][3]
        converged = err < tol
        note = None if converged else "Non converge — verifiez la contractante."
        self._finish_algorithm(
            f, a, b, root, err, rows, COLS["Point Fixe"], converged, note)

    def _run_newton(self, f, a, b, tol):
        x0 = self._read_x0()
        df = _analysis.numeric_derivative(f)
        history = _nl.solve_newton(x0, eps=tol, max_it=50, func=f, dfunc=df)

        if not history:
            self._show_dialog(
                "Newton",
                "Derivee nulle en x0 — choisissez un autre point.",
                "warn")
            self._set_pill("erreur", "error")
            return

        rows = [(r[0], r[1], r[2], r[3]) for r in history]
        root = history[-1][2]
        err = history[-1][3]
        converged = err < tol
        note = None if converged else "Non converge — essayez un autre x0."
        self._finish_algorithm(
            f, a, b, root, err, rows, COLS["Newton"], converged, note)

    def _calc_derivative(self):
        try:
            f = self._parse_function()
            a, b, _ = self._parse_inputs()
        except ValueError as e:
            self._show_error("Erreur", str(e))
            return

        expr = self.func_entry.get().strip()
        self._show_info(_analysis.analyze_derivatives(f, expr, a, b))

    def _verify_continuity(self):
        try:
            f = self._parse_function()
            a, b, _ = self._parse_inputs()
        except ValueError as e:
            self._show_error("Erreur", str(e))
            return

        expr = self.func_entry.get().strip()
        self._show_info(_analysis.verify_continuity(f, expr, a, b))

    def _check_stability(self):
        try:
            f = self._parse_function()
            a, b, _ = self._parse_inputs()
        except ValueError as e:
            self._show_error("Erreur", str(e))
            return

        self._show_info(_analysis.check_stability(f, a, b))

    def _check_contractante(self):
        try:
            f = self._parse_function()
            a, b, _ = self._parse_inputs()
        except ValueError as e:
            self._show_error("Erreur", str(e))
            return

        self._show_info(_analysis.check_contractante(f, a, b))

    def _download_graph(self):
        if self._graph_data is None:
            self._show_dialog("Info", "Lancez d'abord un algorithme.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".svg",
            filetypes=[("SVG", "*.svg"), ("Texte", "*.txt")],
            title="Exporter le graphe")
        if not path:
            return
        xs, ys, a, b, root = self._graph_data
        W, H = 600, 300
        m = {"top": 20, "right": 20, "bottom": 40, "left": 50}
        px_l = m["left"]
        px_r = W - m["right"]
        px_t = m["top"]
        px_b = H - m["bottom"]
        pw = px_r - px_l
        ph = px_b - px_t
        valid = np.isfinite(ys)
        if not np.any(valid):
            self._show_dialog("Graphe vide", "Aucune donnee a exporter.", "warn")
            return
        x_min, x_max = float(xs[0]), float(xs[-1])
        y_min = float(np.min(ys[valid]))
        y_max = float(np.max(ys[valid]))
        pad = max((y_max - y_min) * 0.1, 1e-9)
        y_min -= pad
        y_max += pad

        def tp(xv, yv):
            px = px_l + (xv - x_min) / (x_max - x_min) * pw
            py = px_b - (yv - y_min) / (y_max - y_min) * ph
            return px, py

        pts = []
        for xv, yv in zip(xs, ys):
            if math.isfinite(yv):
                px, py = tp(xv, yv)
                pts.append(f"{px:.2f},{py:.2f}")
        pts = " ".join(pts)
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
                 f'<rect width="{W}" height="{H}" fill="#f0f4f8"/>',
                 f'<rect x="{px_l}" y="{px_t}" width="{pw}" height="{ph}" fill="white" stroke="#d0d7e3"/>',
                 f'<polyline points="{pts}" fill="none" stroke="{COLORS["primary_mid"]}" stroke-width="2.5"/>']
        for xv, col, lbl in [(a, COLORS["red"], f"a={a:g}"),
                               (b, COLORS["blue"], f"b={b:g}")]:
            px, _ = tp(xv, y_min)
            lines.append(f'<line x1="{px:.1f}" y1="{px_t}" x2="{px:.1f}" y2="{px_b}" '
                         f'stroke="{col}" stroke-dasharray="4,3" stroke-width="1.5"/>')
        if root is not None and x_min <= root <= x_max:
            prx, _ = tp(root, y_min)
            lines.append(f'<line x1="{prx:.1f}" y1="{px_t}" x2="{prx:.1f}" y2="{px_b}" '
                         f'stroke="{COLORS["orange"]}" stroke-width="2" stroke-dasharray="6,3"/>')
            lines.append(f'<circle cx="{prx:.1f}" cy="{px_b:.1f}" r="5" '
                         f'fill="{COLORS["orange"]}" stroke="white" stroke-width="1"/>')
        lines.append("</svg>")
        try:
            with open(path, "w", encoding="utf-8") as fp:
                fp.write("\n".join(lines))
            self._show_dialog("Succes", f"Graphe SVG sauvegarde :\n{path}", "success")
        except Exception as ex:
            self._show_error("Erreur", str(ex))

    def _download_table(self):
        if not self._last_table_data:
            self._show_dialog("Info", "Lancez d'abord un algorithme.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Texte", "*.txt")],
            title="Enregistrer le tableau")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fp:
                fp.write(";".join(self._last_table_cols) + "\n")
                for row in self._last_table_data:
                    fp.write(";".join(
                        f"{v:.10g}" if isinstance(v, (float, np.floating))
                        else str(v) for v in row) + "\n")
            self._show_dialog("Succes", f"Tableau sauvegarde :\n{path}", "success")
        except Exception as ex:
            self._show_error("Erreur", str(ex))

    def _back(self):
        import subprocess, sys
        path = os.path.join(os.path.dirname(__file__), "main_screen.py")
        subprocess.Popen([sys.executable, path])
        self.destroy()


if __name__ == "__main__":
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    Axe1Screen().mainloop()
