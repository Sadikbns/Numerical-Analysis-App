"""
Axe 1 — Function Analysis
Pure Python / Tkinter — aucune dependance externe (pas de matplotlib, pas de ttk).
Graphe dessine sur tk.Canvas. Tableau fait main avec Canvas + scrollbar.
"""
import tkinter as tk
import importlib.util
import os
import math
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "bg":          "#f0f4f8",
    "surface":     "#ffffff",
    "border":      "#d0d7e3",
    "hdr_bg":      "#f7f9fc",
    "primary":     "#1a5c38",
    "primary_mid": "#27ae60",
    "primary_lt":  "#e8f5e9",
    "primary_btn": "#155e38",
    "text":        "#2c3e50",
    "text_muted":  "#6c7a89",
    "text_label":  "#555555",
    "success_bg":  "#e8f5e9",
    "success_fg":  "#1a5c38",
    "warn_bg":     "#fff8e1",
    "warn_fg":     "#7d4a00",
    "err_bg":      "#fdecea",
    "err_fg":      "#7b1e1e",
    "blue":        "#2980b9",
    "purple":      "#8e44ad",
    "orange":      "#f39c12",
    "red":         "#e74c3c",
    "row_even":    "#f7f9fc",
    "row_odd":     "#ffffff",
    "row_last":    "#dff0d8",
    "row_last_fg": "#1a5c38",
}

# ── Module nonlinear ──────────────────────────────────────────────────────────
_mod_path = os.path.join(os.path.dirname(__file__),
                          "modules", "chap 1", "nonlinear_resolution.py")
_spec = importlib.util.spec_from_file_location("nonlinear_resolution", _mod_path)
_nl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nl)

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

ROW_H = 22   # hauteur d'une ligne du tableau
COL_W = 100  # largeur colonne par defaut


# ── Helpers bas niveau ────────────────────────────────────────────────────────
def _sep(parent, vertical=False, **kw):
    if vertical:
        return tk.Frame(parent, bg=C["border"], width=1, **kw)
    return tk.Frame(parent, bg=C["border"], height=1, **kw)


def _lbl(parent, text, size=10, bold=False, color=None, bg=None, **kw):
    font = ("Helvetica", size, "bold" if bold else "normal")
    return tk.Label(parent, text=text,
                    bg=bg or C["surface"],
                    fg=color or C["text"],
                    font=font, **kw)


def _entry(parent, default, width=28, mono=False):
    e = tk.Entry(parent,
                 font=("Courier" if mono else "Helvetica", 11),
                 relief="solid", bd=1,
                 bg="#f8f9fa", fg=C["text"],
                 insertbackground=C["text"],
                 highlightthickness=0,
                 width=width)
    e.insert(0, default)
    return e


def _btn(parent, text, bg, fg="white", cmd=None, pad_x=10, pad_y=4,
         size=10, bold=False, **kw):
    font = ("Helvetica", size, "bold" if bold else "normal")
    b = tk.Button(parent, text=text, bg=bg, fg=fg,
                  font=font, relief="flat", cursor="hand2",
                  padx=pad_x, pady=pad_y,
                  activebackground=bg, activeforeground=fg,
                  command=cmd or (lambda: None), **kw)
    # hover leger
    darker = _darken(bg)
    b.bind("<Enter>", lambda e, d=darker: b.config(bg=d))
    b.bind("<Leave>", lambda e: b.config(bg=bg))
    return b


def _darken(hex_color):
    try:
        h = hex_color.lstrip("#")
        r, g, bv = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = max(0, int(r * 0.82))
        g = max(0, int(g * 0.82))
        bv = max(0, int(bv * 0.82))
        return f"#{r:02x}{g:02x}{bv:02x}"
    except Exception:
        return hex_color


# ── Card (frame avec header colore) ──────────────────────────────────────────
def _card(parent, title, dot=None):
    outer = tk.Frame(parent, bg=C["border"], bd=0)
    inner = tk.Frame(outer, bg=C["surface"])
    inner.pack(fill="both", expand=True, padx=1, pady=1)

    hdr = tk.Frame(inner, bg=C["hdr_bg"])
    hdr.pack(fill="x")
    if dot:
        tk.Frame(hdr, bg=dot, width=7, height=7).pack(side="left",
                                                        padx=(10, 4), pady=11)
    _lbl(hdr, title, size=10, bold=True,
         color=C["primary_mid"], bg=C["hdr_bg"]).pack(side="left", pady=7)
    _sep(inner).pack(fill="x")

    body = tk.Frame(inner, bg=C["surface"], padx=10, pady=8)
    body.pack(fill="both", expand=True)
    return outer, body, inner


# ── GraphCanvas — dessin vectoriel pur tkinter ────────────────────────────────
class GraphCanvas(tk.Canvas):
    """Canvas qui trace f(x) avec axes, grille et racine."""

    MARGIN = {"top": 14, "right": 14, "bottom": 30, "left": 42}

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C["surface"],
                         highlightthickness=0,
                         relief="flat", **kw)
        self._data = None
        self.bind("<Configure>", lambda e: self._redraw())
        self._draw_empty()

    def plot(self, xs, ys, a, b, root=None):
        self._data = (xs, ys, a, b, root)
        self._redraw()

    def clear(self):
        self._data = None
        self.delete("all")
        self._draw_empty()

    def _draw_empty(self):
        self.delete("all")
        self.update_idletasks()
        w = self.winfo_width() or 400
        h = self.winfo_height() or 180
        # grille pointillee
        for xi in range(0, w, 40):
            self.create_line(xi, 0, xi, h, fill="#ececec", dash=(2, 4))
        for yi in range(0, h, 30):
            self.create_line(0, yi, w, yi, fill="#ececec", dash=(2, 4))
        self.create_text(w // 2, h // 2,
                         text="Lancez un algorithme pour voir le graphe",
                         fill="#c0c8d4",
                         font=("Helvetica", 10, "italic"))

    def _redraw(self):
        if self._data is None:
            self._draw_empty()
            return
        self.delete("all")
        self.update_idletasks()
        W = self.winfo_width()
        H = self.winfo_height()
        if W < 10 or H < 10:
            return

        xs, ys, a, b, root = self._data
        m = self.MARGIN

        px_left   = m["left"]
        px_right  = W - m["right"]
        px_top    = m["top"]
        px_bottom = H - m["bottom"]
        pw = px_right - px_left
        ph = px_bottom - px_top

        # Plage Y avec marge
        valid = np.isfinite(ys)
        if not np.any(valid):
            self._draw_empty()
            return
        y_min = float(np.min(ys[valid]))
        y_max = float(np.max(ys[valid]))
        pad_y = max((y_max - y_min) * 0.12, 1e-9)
        y_min -= pad_y; y_max += pad_y

        x_min = float(xs[0])
        x_max = float(xs[-1])
        if x_max == x_min:
            x_max = x_min + 1

        def to_px(xv, yv):
            px = px_left + (xv - x_min) / (x_max - x_min) * pw
            py = px_bottom - (yv - y_min) / (y_max - y_min) * ph
            return px, py

        # ─ Grille ─
        def nice_step(span, n=5):
            raw = span / n
            mag = 10 ** math.floor(math.log10(abs(raw) + 1e-12))
            for s in [1, 2, 2.5, 5, 10]:
                if raw <= s * mag:
                    return s * mag
            return 10 * mag

        for xv in self._grid_vals(x_min, x_max, nice_step(x_max - x_min)):
            px, _ = to_px(xv, y_min)
            self.create_line(px, px_top, px, px_bottom,
                             fill="#ebebeb", dash=(2, 3))
            self.create_text(px, px_bottom + 10, text=f"{xv:.2g}",
                             fill=C["text_muted"], font=("Helvetica", 7))

        for yv in self._grid_vals(y_min, y_max, nice_step(y_max - y_min)):
            _, py = to_px(x_min, yv)
            self.create_line(px_left, py, px_right, py,
                             fill="#ebebeb", dash=(2, 3))
            self.create_text(px_left - 4, py, text=f"{yv:.2g}",
                             fill=C["text_muted"], font=("Helvetica", 7),
                             anchor="e")

        # ─ Cadre ─
        self.create_rectangle(px_left, px_top, px_right, px_bottom,
                               outline=C["border"], width=1)

        # ─ Axe X (y=0) ─
        if y_min < 0 < y_max:
            _, py0 = to_px(x_min, 0)
            self.create_line(px_left, py0, px_right, py0,
                             fill="#aaaaaa", width=1)

        # ─ Lignes a et b ─
        for xv, col, lbl in [(a, C["red"], f"a={a:g}"),
                               (b, C["blue"], f"b={b:g}")]:
            if x_min <= xv <= x_max:
                px, _ = to_px(xv, y_min)
                self.create_line(px, px_top, px, px_bottom,
                                 fill=col, width=1, dash=(4, 3))
                self.create_text(px + 3, px_top + 4, text=lbl,
                                 fill=col, font=("Helvetica", 8), anchor="w")

        # ─ Courbe f(x) — segments ─
        pts = []
        for xv, yv in zip(xs, ys):
            if math.isfinite(yv):
                px_, py_ = to_px(xv, yv)
                pts.append((px_, py_))
        if len(pts) >= 2:
            flat = []
            for px_, py_ in pts:
                flat += [px_, py_]
            self.create_line(flat, fill=C["primary_mid"],
                             width=2.5, smooth=True, joinstyle="round")

        # ─ Racine ─
        if root is not None and x_min <= root <= x_max:
            prx, _ = to_px(root, y_min)
            self.create_line(prx, px_top, prx, px_bottom,
                             fill=C["orange"], width=2, dash=(6, 3))
            self.create_oval(prx - 5, px_bottom - 5,
                              prx + 5, px_bottom + 5,
                              fill=C["orange"], outline="white", width=1)
            self.create_text(prx + 6, px_top + 16,
                             text=f"x={root:.5g}",
                             fill=C["orange"], font=("Helvetica", 8, "bold"),
                             anchor="w")

    @staticmethod
    def _grid_vals(vmin, vmax, step):
        if step <= 0:
            return []
        start = math.ceil(vmin / step) * step
        vals = []
        v = start
        while v <= vmax + 1e-9:
            vals.append(round(v, 10))
            v += step
        return vals


# ── TableCanvas — tableau scrollable sans ttk ─────────────────────────────────
class TableCanvas(tk.Frame):
    """Tableau fait sur Canvas avec scrollbar verticale, sans ttk.Treeview."""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C["surface"], **kw)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._cols = []
        self._rows_data = []

        # Header fixe
        self._hdr_canvas = tk.Canvas(self, bg=C["hdr_bg"],
                                      highlightthickness=0, height=ROW_H + 4)
        self._hdr_canvas.grid(row=0, column=0, sticky="ew")

        # Corps scrollable
        self._body_canvas = tk.Canvas(self, bg=C["surface"],
                                       highlightthickness=0)
        self._body_canvas.grid(row=1, column=0, sticky="nsew")

        self._vsb = tk.Scrollbar(self, orient="vertical",
                                  command=self._body_canvas.yview)
        self._vsb.grid(row=1, column=1, sticky="ns")
        self._body_canvas.configure(yscrollcommand=self._vsb.set)

        self._body_canvas.bind("<Configure>", self._on_resize)
        self._body_canvas.bind("<MouseWheel>",
                                lambda e: self._body_canvas.yview_scroll(
                                    int(-1 * (e.delta / 120)), "units"))

    def set_data(self, cols, rows):
        self._cols = cols
        self._rows_data = rows
        self._draw()

    def _on_resize(self, event=None):
        self._draw()

    def _col_width(self):
        self.update_idletasks()
        W = self._body_canvas.winfo_width() or 500
        vsb_w = self._vsb.winfo_width() or 16
        total = W - vsb_w - 2
        return max(60, total // max(1, len(self._cols)))

    def _draw(self):
        if not self._cols:
            return
        cw = self._col_width()
        W = cw * len(self._cols)

        # ─ Header ─
        self._hdr_canvas.delete("all")
        self._hdr_canvas.config(width=W)
        for ci, col in enumerate(self._cols):
            x0 = ci * cw
            self._hdr_canvas.create_rectangle(
                x0, 0, x0 + cw, ROW_H + 4,
                fill=C["hdr_bg"], outline=C["border"])
            self._hdr_canvas.create_text(
                x0 + cw // 2, (ROW_H + 4) // 2,
                text=col, fill=C["text"],
                font=("Helvetica", 9, "bold"))

        # ─ Corps ─
        self._body_canvas.delete("all")
        total_h = ROW_H * len(self._rows_data)
        self._body_canvas.config(scrollregion=(0, 0, W, total_h))

        for ri, row in enumerate(self._rows_data):
            y0 = ri * ROW_H
            is_last = (ri == len(self._rows_data) - 1)
            bg = (C["row_last"] if is_last
                  else (C["row_even"] if ri % 2 == 0 else C["row_odd"]))
            fg = C["row_last_fg"] if is_last else C["text"]

            for ci, val in enumerate(row):
                x0 = ci * cw
                self._body_canvas.create_rectangle(
                    x0, y0, x0 + cw, y0 + ROW_H,
                    fill=bg, outline=C["border"], width=0.5)
                txt = f"{val:.8g}" if isinstance(val, (float, np.floating)) \
                      else str(val)
                self._body_canvas.create_text(
                    x0 + cw // 2, y0 + ROW_H // 2,
                    text=txt, fill=fg,
                    font=("Courier", 9, "bold" if is_last else "normal"))


# ── Ecran principal ───────────────────────────────────────────────────────────
class Axe1Screen(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Axe 1 — Function Analysis")
        self.geometry("1020x720")
        self.minsize(880, 600)
        self.configure(bg=C["bg"])

        self._last_table_data = []
        self._last_table_cols = []
        self._graph_data = None   # (xs, ys, a, b, root)

        self._build()

    # ── Layout ───────────────────────────────────────────────────────────────
    def _build(self):
        self._build_header()
        content = tk.Frame(self, bg=C["bg"])
        content.pack(fill="both", expand=True, padx=12, pady=10)
        content.columnconfigure(0, minsize=268, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)
        self._build_left(content)
        self._build_right(content)

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        bar = tk.Frame(self, bg=C["primary"], height=52)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        back = _btn(bar, "  Retour", bg=C["primary_btn"],
                    cmd=self._back, pad_x=14, pad_y=8)
        back.pack(side="left", padx=12, pady=8)

        _lbl(bar, "Axe 1 — Equations non-lineaires",
             size=14, bold=True, color="white",
             bg=C["primary"]).pack(side="left", padx=8)

        self._pill_var = tk.StringVar(value="")
        self._pill = tk.Label(bar, textvariable=self._pill_var,
                               bg=C["primary_mid"], fg="white",
                               font=("Helvetica", 9), padx=12, pady=4)
        self._pill.pack(side="right", padx=14)

    # ── Left panel ───────────────────────────────────────────────────────────
    def _build_left(self, parent):
        outer = tk.Frame(parent, bg=C["bg"])
        outer.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # ─ Card Inputs ─
        card, body, _ = _card(outer, "  Entrees", dot=C["primary_mid"])
        card.pack(fill="x", pady=(0, 8))

        _lbl(body, "Formule f(x)", color=C["text_label"]).pack(anchor="w")
        self.func_entry = _entry(body, "x**3 - x - 2", mono=True)
        self.func_entry.pack(fill="x", pady=(2, 6))

        # Raccourcis fonctions
        shortcuts = tk.Frame(body, bg=C["surface"])
        shortcuts.pack(fill="x", pady=(0, 8))
        _lbl(body, "Exemples :", size=9,
             color=C["text_muted"]).pack(anchor="w")
        sc_row = tk.Frame(body, bg=C["surface"])
        sc_row.pack(fill="x", pady=(2, 8))
        for ex in ["sin(x)", "cos(x)", "exp(x)", "sqrt(x)"]:
            b = tk.Button(sc_row, text=ex, bg=C["hdr_bg"],
                          fg=C["text_muted"], font=("Courier", 8),
                          relief="solid", bd=1, cursor="hand2",
                          padx=4, pady=2,
                          command=lambda v=ex: self._paste_func(v))
            b.pack(side="left", padx=2)

        _sep(body).pack(fill="x", pady=6)

        _lbl(body, "Intervalle [a, b]", color=C["text_label"]).pack(anchor="w")
        iv = tk.Frame(body, bg=C["surface"])
        iv.pack(fill="x", pady=(2, 8))
        self.a_entry = _entry(iv, "1", width=7, mono=True)
        self.a_entry.pack(side="left")
        _lbl(iv, "  →  ", color=C["text_muted"],
             bg=C["surface"]).pack(side="left")
        self.b_entry = _entry(iv, "2", width=7, mono=True)
        self.b_entry.pack(side="left")

        _lbl(body, "Tolerance (eps)", color=C["text_label"]).pack(anchor="w")
        self.tol_entry = _entry(body, "1e-6", width=14, mono=True)
        self.tol_entry.pack(anchor="w", pady=(2, 0))

        # x0 — cache jusqu'a besoin
        self.x0_outer = tk.Frame(body, bg="#fffbeb",
                                  highlightthickness=1,
                                  highlightbackground="#f0c040")
        _lbl(self.x0_outer, "Point initial x0 :",
             size=9, color=C["warn_fg"],
             bg="#fffbeb").pack(anchor="w", padx=8, pady=(4, 0))
        self.x0_entry = _entry(self.x0_outer, "1.5", width=12, mono=True)
        self.x0_entry.config(bg="#fffdef")
        self.x0_entry.pack(anchor="w", padx=8, pady=(2, 6))

        # ─ Card Algorithme ─
        card2, body2, _ = _card(outer, "  Algorithme", dot=C["primary_mid"])
        card2.pack(fill="x", pady=(0, 8))

        self.algo_var = tk.StringVar(value="Dichotomie")
        for algo in ["Dichotomie", "Point Fixe", "Newton"]:
            row = tk.Frame(body2, bg=C["surface"])
            row.pack(fill="x", pady=3)
            rb = tk.Radiobutton(row, text=algo, variable=self.algo_var,
                                value=algo, bg=C["surface"], fg=C["text"],
                                font=("Helvetica", 11),
                                activebackground=C["surface"],
                                selectcolor=C["primary_mid"],
                                command=self._on_algo_change)
            rb.pack(side="left")
            tag_lbl = tk.Label(row, text=ALGO_TAG[algo],
                               bg=C["primary_lt"], fg=C["success_fg"],
                               font=("Helvetica", 8), padx=6, pady=1)
            tag_lbl.pack(side="right")

        # ─ Card Actions ─
        card3, body3, _ = _card(outer, "  Analyser f(x)", dot=C["blue"])
        card3.pack(fill="x", pady=(0, 8))

        for icon, label, badge, cmd in [
            ("f'",  "Derivees f'(x), f''(x)", None,  self._calc_derivative),
            ("~",   "Continuite sur [a, b]",   None,  self._verify_continuity),
            ("g",   "Stabilite de g(x)",        "k",   self._check_stability),
            ("|k|", "Contractante sur [a, b]",  "k<1", self._check_contractante),
        ]:
            self._action_row(body3, icon, label, badge, cmd)

        # ─ Bouton Run ─
        run_frame = tk.Frame(outer, bg=C["bg"])
        run_frame.pack(fill="x", pady=(0, 6))
        self._run_btn = _btn(run_frame, "  Lancer l'algorithme",
                              bg=C["primary_mid"], bold=True,
                              size=12, pad_y=10,
                              cmd=self._run_algorithm)
        self._run_btn.pack(fill="x")

        # ─ Barre de resultat/info ─
        self._result_outer = tk.Frame(outer, bg=C["primary_lt"],
                                       highlightthickness=1,
                                       highlightbackground="#a8d5b5")
        self._result_lbl = tk.Label(self._result_outer, text="",
                                     bg=C["primary_lt"],
                                     fg=C["success_fg"],
                                     font=("Helvetica", 10, "bold"),
                                     justify="left", wraplength=250,
                                     anchor="w")
        self._result_lbl.pack(padx=10, pady=7, anchor="w")

        self._info_outer = tk.Frame(outer, bg=C["warn_bg"],
                                     highlightthickness=1,
                                     highlightbackground="#f0c040")
        self._info_lbl = tk.Label(self._info_outer, text="",
                                   bg=C["warn_bg"],
                                   fg=C["warn_fg"],
                                   font=("Helvetica", 9),
                                   justify="left", wraplength=250,
                                   anchor="w")
        self._info_lbl.pack(padx=10, pady=7, anchor="w")

    def _action_row(self, parent, icon, label, badge, cmd):
        row = tk.Frame(parent, bg=C["surface"])
        row.pack(fill="x", pady=2)

        ic = tk.Frame(row, bg=C["primary_mid"], width=22, height=22)
        ic.pack(side="left", padx=(0, 6))
        ic.pack_propagate(False)
        tk.Label(ic, text=icon, bg=C["primary_mid"], fg="white",
                 font=("Helvetica", 8, "bold")).pack(expand=True)

        _btn(row, label, bg=C["primary_lt"], fg=C["primary"],
             pad_x=8, pad_y=4, cmd=cmd).pack(side="left", fill="x", expand=True)

        if badge:
            tk.Label(row, text=badge, bg="#fff3cd", fg="#856404",
                     font=("Helvetica", 8), padx=5, pady=1).pack(side="right")

    # ── Right panel ───────────────────────────────────────────────────────────
    def _build_right(self, parent):
        outer = tk.Frame(parent, bg=C["bg"])
        outer.grid(row=0, column=1, sticky="nsew")
        outer.rowconfigure(0, weight=2)
        outer.rowconfigure(1, weight=3)
        outer.columnconfigure(0, weight=1)

        # ─ Carte Graphe ─
        graph_card = tk.Frame(outer, bg=C["border"])
        graph_card.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        inner_g = tk.Frame(graph_card, bg=C["surface"])
        inner_g.pack(fill="both", expand=True, padx=1, pady=1)
        inner_g.rowconfigure(2, weight=1)
        inner_g.columnconfigure(0, weight=1)

        # Header graphe
        ghdr = tk.Frame(inner_g, bg=C["hdr_bg"], height=30)
        ghdr.grid(row=0, column=0, sticky="ew")
        ghdr.grid_propagate(False)
        _lbl(ghdr, "Graphe de f(x)", size=10, bold=True,
             bg=C["hdr_bg"]).pack(side="left", padx=10, pady=6)
        lgd = tk.Frame(ghdr, bg=C["hdr_bg"])
        lgd.pack(side="right", padx=10)
        for txt, col in [("f(x)", C["primary_mid"]),
                          ("a", C["red"]),
                          ("b", C["blue"]),
                          ("racine", C["orange"])]:
            tk.Label(lgd, text=txt, bg=col, fg="white",
                     font=("Helvetica", 8), padx=7, pady=2
                     ).pack(side="left", padx=2)

        _sep(inner_g).grid(row=1, column=0, sticky="ew")

        self._graph = GraphCanvas(inner_g)
        self._graph.grid(row=2, column=0, sticky="nsew", padx=8, pady=6)

        # Barre stats
        _sep(inner_g).grid(row=3, column=0, sticky="ew")
        stats_bar = tk.Frame(inner_g, bg=C["hdr_bg"])
        stats_bar.grid(row=4, column=0, sticky="ew")
        stats_bar.columnconfigure((0, 1, 2), weight=1)

        self._stat_vars = {}
        for i, (key, lbl) in enumerate([("racine", "racine"),
                                          ("iter",   "iterations"),
                                          ("err",    "erreur finale")]):
            f = tk.Frame(stats_bar, bg=C["hdr_bg"])
            f.grid(row=0, column=i, padx=8, pady=6, sticky="ew")
            if i > 0:
                _sep(stats_bar, vertical=True).grid(
                    row=0, column=i, sticky="ns", padx=(0, 0))
            v = tk.StringVar(value="—")
            self._stat_vars[key] = v
            tk.Label(f, textvariable=v, bg=C["hdr_bg"],
                     fg=C["primary_mid"],
                     font=("Helvetica", 13, "bold")).pack()
            tk.Label(f, text=lbl, bg=C["hdr_bg"],
                     fg=C["text_muted"],
                     font=("Helvetica", 8)).pack()

        # ─ Carte Tableau ─
        tbl_card = tk.Frame(outer, bg=C["border"])
        tbl_card.grid(row=1, column=0, sticky="nsew")
        inner_t = tk.Frame(tbl_card, bg=C["surface"])
        inner_t.pack(fill="both", expand=True, padx=1, pady=1)
        inner_t.rowconfigure(2, weight=1)
        inner_t.columnconfigure(0, weight=1)

        # Header tableau
        thdr = tk.Frame(inner_t, bg=C["hdr_bg"], height=30)
        thdr.grid(row=0, column=0, sticky="ew")
        thdr.grid_propagate(False)
        _lbl(thdr, "Tableau des iterations", size=10, bold=True,
             bg=C["hdr_bg"]).pack(side="left", padx=10, pady=6)
        self._conv_lbl = tk.Label(thdr, text="",
                                   bg=C["hdr_bg"],
                                   font=("Helvetica", 8), padx=5)
        self._conv_lbl.pack(side="left")

        dl = tk.Frame(thdr, bg=C["hdr_bg"])
        dl.pack(side="right", padx=8)
        for txt, col, cmd in [
            ("CSV",   C["purple"], self._download_table),
            ("Graph", C["blue"],   self._download_graph),
        ]:
            _btn(dl, f"  {txt}", bg=col, pad_x=10, pad_y=3,
                 size=9, cmd=cmd).pack(side="right", padx=3)

        _sep(inner_t).grid(row=1, column=0, sticky="ew")

        self._table = TableCanvas(inner_t)
        self._table.grid(row=2, column=0, sticky="nsew", padx=0, pady=0)

    # ── Helpers UI ────────────────────────────────────────────────────────────
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
            "idle":    C["primary_mid"],
            "ok":      "#27ae60",
            "warn":    "#e67e22",
            "error":   "#e74c3c",
            "running": "#2980b9",
        }
        self._pill.config(text=text, bg=colors.get(kind, C["primary_mid"]))
        self._pill_var.set(text)

    def _show_result(self, text, kind="ok"):
        cfg = {
            "ok":   (C["primary_lt"], C["success_fg"], "#a8d5b5"),
            "warn": (C["warn_bg"],    C["warn_fg"],    "#f0c040"),
            "err":  (C["err_bg"],     C["err_fg"],     "#f5b7b7"),
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
                                   bg="#e8f5e9", fg=C["success_fg"])
        else:
            self._conv_lbl.config(text="non converge",
                                   bg=C["err_bg"], fg=C["err_fg"])

    # ── Parse ─────────────────────────────────────────────────────────────────
    def _parse_function(self):
        expr = self.func_entry.get().strip()
        safe = {
            "x": 0, "np": np, "math": math,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "log": np.log, "log10": np.log10,
            "sqrt": np.sqrt, "abs": np.abs,
            "pi": math.pi, "e": math.e,
        }
        try:
            t = dict(safe); t["x"] = 1.0
            eval(compile(expr, "<string>", "eval"), {"__builtins__": {}}, t)
        except Exception as ex:
            raise ValueError(f"Expression invalide : {ex}")

        def f(x):
            env = dict(safe); env["x"] = x
            return eval(compile(expr, "<string>", "eval"),
                        {"__builtins__": {}}, env)
        return f

    def _parse_inputs(self):
        try:
            a   = float(self.a_entry.get())
            b   = float(self.b_entry.get())
            tol = float(self.tol_entry.get())
        except ValueError:
            raise ValueError("a, b et eps doivent etre des nombres valides.")
        if a >= b:
            raise ValueError("Il faut a < b.")
        if tol <= 0:
            raise ValueError("La tolerance doit etre > 0.")
        return a, b, tol

    # ── Graphe ────────────────────────────────────────────────────────────────
    def _plot(self, f, a, b, root=None):
        marge = abs(b - a) * 0.35
        xs = np.linspace(a - marge, b + marge, 400)
        try:
            ys = np.array([f(xi) for xi in xs], dtype=float)
        except Exception:
            return
        self._graph_data = (xs, ys, a, b, root)
        self._graph.plot(xs, ys, a, b, root)

    # ── Tableau ───────────────────────────────────────────────────────────────
    def _fill_table(self, cols, rows, converged=True):
        self._last_table_cols = list(cols)
        self._last_table_data = rows
        self._table.set_data(cols, rows)
        self._set_conv_badge(converged)

    # ── Run ───────────────────────────────────────────────────────────────────
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
            from tkinter import messagebox
            messagebox.showerror("Erreur", str(e))
            self._set_pill("erreur", "error")
            return

        root = None; converged = True

        if algo == "Dichotomie":
            fa, fb = f(a), f(b)
            if fa * fb >= 0:
                from tkinter import messagebox
                messagebox.showerror(
                    "Dichotomie impossible",
                    f"f(a) x f(b) doit etre < 0.\n"
                    f"f({a}) = {fa:.4f},  f({b}) = {fb:.4f}\n"
                    "Choisissez un intervalle contenant une racine.")
                self._set_pill("erreur", "error")
                return
            rows = []; ai, bi = a, b
            for n in range(1, 101):
                m = (ai + bi) / 2; fm = f(m); eps = (bi - ai) / 2
                rows.append((n, ai, bi, m, fm, eps))
                if eps < tol: break
                if f(ai) * fm < 0: bi = m
                else: ai = m
            root = rows[-1][3]
            err = rows[-1][5]
            converged = err < tol
            self._fill_table(COLS["Dichotomie"], rows, converged)
            self._set_stats(f"{root:.7g}", len(rows), f"{err:.2e}")
            self._show_result(
                f"Racine  {root:.8g}   ({len(rows)} iterations)",
                "ok" if converged else "warn")

        elif algo == "Point Fixe":
            try:
                x0 = float(self.x0_entry.get())
            except ValueError:
                from tkinter import messagebox
                messagebox.showerror("Erreur", "x0 doit etre un nombre.")
                self._set_pill("erreur", "error"); return
            g = lambda x: x - f(x)
            data = _nl.executer_point_fixe(g, x0, epsilon=tol, max_iter=50)
            if not data:
                from tkinter import messagebox
                messagebox.showinfo("Info", "Aucune iteration produite.")
                self._set_pill("erreur", "error"); return
            rows = [(r[0], r[1], r[2], r[3]) for r in data]
            root = data[-1][2]; last_err = data[-1][3]
            converged = last_err < tol
            self._fill_table(COLS["Point Fixe"], rows, converged)
            self._set_stats(f"{root:.7g}", len(rows), f"{last_err:.2e}")
            msg = f"Racine  {root:.8g}   ({len(rows)} iterations)"
            if not converged:
                msg += "\nNon converge — verifiez la contractante."
            self._show_result(msg, "ok" if converged else "warn")

        elif algo == "Newton":
            try:
                x0 = float(self.x0_entry.get())
            except ValueError:
                from tkinter import messagebox
                messagebox.showerror("Erreur", "x0 doit etre un nombre.")
                self._set_pill("erreur", "error"); return
            h = 1e-7
            df = lambda x: (f(x + h) - f(x - h)) / (2 * h)
            history = _nl.solve_newton(x0, eps=tol, max_it=50,
                                        func=f, dfunc=df)
            if not history:
                from tkinter import messagebox
                messagebox.showinfo("Newton",
                    "Derivee nulle en x0 — choisissez un autre point.")
                self._set_pill("erreur", "error"); return
            rows = [(r[0], r[1], r[2], r[3]) for r in history]
            root = history[-1][2]; last_err = history[-1][3]
            converged = last_err < tol
            self._fill_table(COLS["Newton"], rows, converged)
            self._set_stats(f"{root:.7g}", len(rows), f"{last_err:.2e}")
            msg = f"Racine  {root:.8g}   ({len(rows)} iterations)"
            if not converged:
                msg += "\nNon converge — essayez un autre x0."
            self._show_result(msg, "ok" if converged else "warn")

        self._plot(f, a, b, root)
        self._set_pill("converge" if converged else "non converge",
                        "ok" if converged else "warn")

    # ── Actions ──────────────────────────────────────────────────────────────
    def _calc_derivative(self):
        try:
            f = self._parse_function()
        except ValueError as e:
            from tkinter import messagebox
            messagebox.showerror("Erreur", str(e)); return
        try:
            import sympy as sp
            x = sp.Symbol("x")
            expr = self.func_entry.get().strip()
            df  = sp.diff(sp.sympify(expr), x)
            ddf = sp.diff(df, x)
            self._show_info(
                f"f'(x)  = {sp.simplify(df)}\n"
                f"f''(x) = {sp.simplify(ddf)}")
        except Exception:
            h = 1e-5
            try: av = float(self.a_entry.get())
            except: av = 0.0
            dfn = (f(av + h) - f(av - h)) / (2 * h)
            self._show_info(f"f'({av}) approx {dfn:.6g}  (numerique)")

    def _verify_continuity(self):
        try:
            f = self._parse_function(); a, b, _ = self._parse_inputs()
        except ValueError as e:
            from tkinter import messagebox
            messagebox.showerror("Erreur", str(e)); return
        xs = np.linspace(a, b, 500)
        try:
            ys = np.array([f(xi) for xi in xs])
            jumps = np.abs(np.diff(ys))
            thresh = 10 * np.std(ys) if np.std(ys) > 0 else 1e3
            disc = np.any(jumps > thresh)
            self._show_info(
                "Discontinuite detectee sur [a, b]." if disc
                else "f semble continue sur [a, b].")
        except Exception:
            self._show_info("Impossible d'evaluer f sur l'intervalle.")

    def _check_stability(self):
        try:
            f = self._parse_function(); a, b, _ = self._parse_inputs()
        except ValueError as e:
            from tkinter import messagebox
            messagebox.showerror("Erreur", str(e)); return
        g = lambda x: x - f(x)
        xs = np.linspace(a, b, 200)
        try:
            gs = np.array([g(xi) for xi in xs])
            stable = np.all((gs >= a) & (gs <= b))
            self._show_info(
                f"Stabilite g(x) = x - f(x) sur [a, b] : "
                f"{'OUI' if stable else 'NON'}\n"
                f"g(x) doit rester dans [{a}, {b}].")
        except Exception:
            self._show_info("Impossible de verifier la stabilite.")

    def _check_contractante(self):
        try:
            f = self._parse_function(); a, b, _ = self._parse_inputs()
        except ValueError as e:
            from tkinter import messagebox
            messagebox.showerror("Erreur", str(e)); return
        g = lambda x: x - f(x)
        h = 1e-7
        dg = lambda x: (g(x + h) - g(x - h)) / (2 * h)
        xs = np.linspace(a, b, 200)
        try:
            k = float(np.max(np.abs([dg(xi) for xi in xs])))
            ok = k < 1
            self._show_info(
                f"Contractante : {'OUI' if ok else 'NON'}\n"
                f"k = max|g'(x)| = {k:.4f}  (doit etre < 1)")
        except Exception:
            self._show_info("Impossible de verifier la contractante.")

    # ── Downloads ────────────────────────────────────────────────────────────
    def _download_graph(self):
        if self._graph_data is None:
            from tkinter import messagebox
            messagebox.showinfo("Info", "Lancez d'abord un algorithme."); return
        from tkinter import filedialog, messagebox
        path = filedialog.asksaveasfilename(
            defaultextension=".svg",
            filetypes=[("SVG", "*.svg"), ("Texte", "*.txt")],
            title="Exporter le graphe")
        if not path: return
        xs, ys, a, b, root = self._graph_data
        W, H = 600, 300
        m = {"top": 20, "right": 20, "bottom": 40, "left": 50}
        px_l = m["left"]; px_r = W - m["right"]
        px_t = m["top"]; px_b = H - m["bottom"]
        pw = px_r - px_l; ph = px_b - px_t
        valid = np.isfinite(ys)
        if not np.any(valid):
            messagebox.showwarning("Graphe vide", "Aucune donnee a exporter."); return
        x_min, x_max = float(xs[0]), float(xs[-1])
        y_min = float(np.min(ys[valid])); y_max = float(np.max(ys[valid]))
        pad = max((y_max - y_min) * 0.1, 1e-9)
        y_min -= pad; y_max += pad
        def tp(xv, yv):
            px = px_l + (xv - x_min) / (x_max - x_min) * pw
            py = px_b - (yv - y_min) / (y_max - y_min) * ph
            return px, py
        pts = " ".join(f"{tp(xv,yv)[0]:.2f},{tp(xv,yv)[1]:.2f}"
                       for xv, yv in zip(xs, ys) if math.isfinite(yv))
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
                 f'<rect width="{W}" height="{H}" fill="#f0f4f8"/>',
                 f'<rect x="{px_l}" y="{px_t}" width="{pw}" height="{ph}" fill="white" stroke="#d0d7e3"/>',
                 f'<polyline points="{pts}" fill="none" stroke="{C["primary_mid"]}" stroke-width="2.5"/>']
        for xv, col, lbl in [(a, C["red"], f"a={a:g}"),
                               (b, C["blue"], f"b={b:g}")]:
            px, _ = tp(xv, y_min)
            lines.append(f'<line x1="{px:.1f}" y1="{px_t}" x2="{px:.1f}" y2="{px_b}" '
                         f'stroke="{col}" stroke-dasharray="4,3" stroke-width="1.5"/>')
        if root is not None and x_min <= root <= x_max:
            prx, _ = tp(root, y_min)
            lines.append(f'<line x1="{prx:.1f}" y1="{px_t}" x2="{prx:.1f}" y2="{px_b}" '
                         f'stroke="{C["orange"]}" stroke-width="2" stroke-dasharray="6,3"/>')
            lines.append(f'<circle cx="{prx:.1f}" cy="{px_b:.1f}" r="5" '
                         f'fill="{C["orange"]}" stroke="white" stroke-width="1"/>')
        lines.append("</svg>")
        try:
            with open(path, "w", encoding="utf-8") as fp:
                fp.write("\n".join(lines))
            messagebox.showinfo("Succes", f"Graphe SVG sauvegarde :\n{path}")
        except Exception as ex:
            messagebox.showerror("Erreur", str(ex))

    def _download_table(self):
        if not self._last_table_data:
            from tkinter import messagebox
            messagebox.showinfo("Info", "Lancez d'abord un algorithme."); return
        from tkinter import filedialog, messagebox
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Texte", "*.txt")],
            title="Enregistrer le tableau")
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as fp:
                fp.write(";".join(self._last_table_cols) + "\n")
                for row in self._last_table_data:
                    fp.write(";".join(
                        f"{v:.10g}" if isinstance(v, (float, np.floating))
                        else str(v) for v in row) + "\n")
            messagebox.showinfo("Succes", f"Tableau sauvegarde :\n{path}")
        except Exception as ex:
            messagebox.showerror("Erreur", str(ex))

    def _back(self):
        import subprocess, sys
        path = os.path.join(os.path.dirname(__file__), "main_screen.py")
        subprocess.Popen([sys.executable, path])
        self.destroy()


if __name__ == "__main__":
    Axe1Screen().mainloop()