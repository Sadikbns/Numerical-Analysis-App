import math
import tkinter as tk

import numpy as np


def enable_dpi_awareness():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


COLORS = {
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

ROW_H = 22


def _line(parent, vertical=False, **kw):
    if vertical:
        return tk.Frame(parent, bg=COLORS["border"], width=1, **kw)
    return tk.Frame(parent, bg=COLORS["border"], height=1, **kw)


def _text(parent, text, size=10, bold=False, color=None, bg=None, **kw):
    font = ("Helvetica", size, "bold" if bold else "normal")
    return tk.Label(parent, text=text,
                    bg=bg or COLORS["surface"],
                    fg=color or COLORS["text"],
                    font=font, **kw)


def _field(parent, default, width=28, mono=False):
    e = tk.Entry(parent,
                 font=("Courier" if mono else "Helvetica", 11),
                 relief="solid", bd=1,
                 bg="#f8f9fa", fg=COLORS["text"],
                 insertbackground=COLORS["text"],
                 highlightthickness=0,
                 width=width)
    e.insert(0, default)
    return e


def _button(parent, text, bg, fg="white", cmd=None, pad_x=10, pad_y=4,
            size=10, bold=False, **kw):
    font = ("Helvetica", size, "bold" if bold else "normal")
    b = tk.Button(parent, text=text, bg=bg, fg=fg,
                  font=font, relief="flat", cursor="hand2",
                  padx=pad_x, pady=pad_y,
                  activebackground=bg, activeforeground=fg,
                  command=cmd or (lambda: None), **kw)
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


def _panel(parent, title, dot=None):
    outer = tk.Frame(parent, bg=COLORS["border"], bd=0)
    inner = tk.Frame(outer, bg=COLORS["surface"])
    inner.pack(fill="both", expand=True, padx=1, pady=1)

    hdr = tk.Frame(inner, bg=COLORS["hdr_bg"])
    hdr.pack(fill="x")
    if dot:
        tk.Frame(hdr, bg=dot, width=7, height=7).pack(side="left",
                                                        padx=(10, 4), pady=11)
    _text(hdr, title, size=10, bold=True,
          color=COLORS["primary_mid"], bg=COLORS["hdr_bg"]).pack(side="left", pady=7)
    _line(inner).pack(fill="x")

    body = tk.Frame(inner, bg=COLORS["surface"], padx=10, pady=8)
    body.pack(fill="both", expand=True)
    return outer, body, inner


def show_dialog(parent, title, message, kind="info"):
    styles = {
        "info": (COLORS["primary_lt"], COLORS["success_fg"], COLORS["primary_mid"], "i"),
        "success": (COLORS["primary_lt"], COLORS["success_fg"], COLORS["primary_mid"], "OK"),
        "warn": (COLORS["warn_bg"], COLORS["warn_fg"], COLORS["orange"], "!"),
        "error": (COLORS["err_bg"], COLORS["err_fg"], COLORS["red"], "!"),
    }
    bg, fg, accent, icon = styles.get(kind, styles["info"])

    win = tk.Toplevel(parent)
    win.title(title)
    win.configure(bg=COLORS["surface"])
    win.resizable(False, False)
    win.transient(parent)
    win.grab_set()

    outer = tk.Frame(win, bg=COLORS["border"])
    outer.pack(fill="both", expand=True, padx=1, pady=1)

    header = tk.Frame(outer, bg=accent)
    header.pack(fill="x")
    tk.Label(header, text=title, bg=accent, fg="white",
             font=("Helvetica", 11, "bold"), padx=12, pady=8).pack(side="left")

    body = tk.Frame(outer, bg=COLORS["surface"], padx=16, pady=14)
    body.pack(fill="both", expand=True)

    badge = tk.Frame(body, bg=bg, width=34, height=34,
                     highlightthickness=1, highlightbackground=accent)
    badge.grid(row=0, column=0, sticky="n", padx=(0, 12))
    badge.grid_propagate(False)
    tk.Label(badge, text=icon, bg=bg, fg=fg,
             font=("Helvetica", 11, "bold")).place(relx=0.5, rely=0.5, anchor="center")

    tk.Label(body, text=message, bg=COLORS["surface"], fg=COLORS["text"],
             font=("Helvetica", 10), justify="left",
             wraplength=360).grid(row=0, column=1, sticky="w")

    footer = tk.Frame(outer, bg=COLORS["hdr_bg"], padx=12, pady=10)
    footer.pack(fill="x")
    _button(footer, "OK", bg=accent, cmd=win.destroy,
            pad_x=18, pad_y=5, bold=True).pack(side="right")

    win.update_idletasks()
    parent.update_idletasks()
    x = parent.winfo_rootx() + (parent.winfo_width() - win.winfo_width()) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - win.winfo_height()) // 2
    win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
    win.wait_window()


class GraphCanvas(tk.Canvas):
    MARGIN = {"top": 14, "right": 14, "bottom": 30, "left": 42}

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=COLORS["surface"],
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
        width = self.winfo_width()
        height = self.winfo_height()
        if width < 10 or height < 10:
            return

        xs, ys, a, b, root = self._data
        m = self.MARGIN

        px_left = m["left"]
        px_right = width - m["right"]
        px_top = m["top"]
        px_bottom = height - m["bottom"]
        graph_w = px_right - px_left
        graph_h = px_bottom - px_top

        valid = np.isfinite(ys)
        if not np.any(valid):
            self._draw_empty()
            return
        y_min = float(np.min(ys[valid]))
        y_max = float(np.max(ys[valid]))
        pad_y = max((y_max - y_min) * 0.12, 1e-9)
        y_min -= pad_y
        y_max += pad_y

        x_min = float(xs[0])
        x_max = float(xs[-1])
        if x_max == x_min:
            x_max = x_min + 1

        def to_px(xv, yv):
            px = px_left + (xv - x_min) / (x_max - x_min) * graph_w
            py = px_bottom - (yv - y_min) / (y_max - y_min) * graph_h
            return px, py

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
                             fill=COLORS["text_muted"], font=("Helvetica", 7))

        for yv in self._grid_vals(y_min, y_max, nice_step(y_max - y_min)):
            _, py = to_px(x_min, yv)
            self.create_line(px_left, py, px_right, py,
                             fill="#ebebeb", dash=(2, 3))
            self.create_text(px_left - 4, py, text=f"{yv:.2g}",
                             fill=COLORS["text_muted"], font=("Helvetica", 7),
                             anchor="e")

        self.create_rectangle(px_left, px_top, px_right, px_bottom,
                               outline=COLORS["border"], width=1)

        if y_min < 0 < y_max:
            _, py0 = to_px(x_min, 0)
            self.create_line(px_left, py0, px_right, py0,
                             fill="#aaaaaa", width=1)

        for xv, col, lbl in [(a, COLORS["red"], f"a={a:g}"),
                               (b, COLORS["blue"], f"b={b:g}")]:
            if x_min <= xv <= x_max:
                px, _ = to_px(xv, y_min)
                self.create_line(px, px_top, px, px_bottom,
                                 fill=col, width=1, dash=(4, 3))
                self.create_text(px + 3, px_top + 4, text=lbl,
                                 fill=col, font=("Helvetica", 8), anchor="w")

        pts = []
        for xv, yv in zip(xs, ys):
            if math.isfinite(yv):
                px, py = to_px(xv, yv)
                pts.append((px, py))
        if len(pts) >= 2:
            flat = []
            for px, py in pts:
                flat.extend((px, py))
            self.create_line(flat, fill=COLORS["primary_mid"],
                             width=2.5, smooth=True, joinstyle="round")

        if root is not None and x_min <= root <= x_max:
            prx, _ = to_px(root, y_min)
            self.create_line(prx, px_top, prx, px_bottom,
                             fill=COLORS["orange"], width=2, dash=(6, 3))
            self.create_oval(prx - 5, px_bottom - 5,
                              prx + 5, px_bottom + 5,
                              fill=COLORS["orange"], outline="white", width=1)
            self.create_text(prx + 6, px_top + 16,
                             text=f"x={root:.5g}",
                             fill=COLORS["orange"], font=("Helvetica", 8, "bold"),
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


class TableCanvas(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=COLORS["surface"], **kw)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._cols = []
        self._rows_data = []

        self._hdr_canvas = tk.Canvas(self, bg=COLORS["hdr_bg"],
                                      highlightthickness=0, height=ROW_H + 4)
        self._hdr_canvas.grid(row=0, column=0, sticky="ew")

        self._body_canvas = tk.Canvas(self, bg=COLORS["surface"],
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
        width = self._body_canvas.winfo_width() or 500
        vsb_w = self._vsb.winfo_width() or 16
        total = width - vsb_w - 2
        return max(60, total // max(1, len(self._cols)))

    def _draw(self):
        if not self._cols:
            return
        cw = self._col_width()
        width = cw * len(self._cols)

        self._hdr_canvas.delete("all")
        self._hdr_canvas.config(width=width)
        for ci, col in enumerate(self._cols):
            x0 = ci * cw
            self._hdr_canvas.create_rectangle(
                x0, 0, x0 + cw, ROW_H + 4,
                fill=COLORS["hdr_bg"], outline=COLORS["border"])
            self._hdr_canvas.create_text(
                x0 + cw // 2, (ROW_H + 4) // 2,
                text=col, fill=COLORS["text"],
                font=("Helvetica", 9, "bold"))

        self._body_canvas.delete("all")
        total_h = ROW_H * len(self._rows_data)
        self._body_canvas.config(scrollregion=(0, 0, width, total_h))

        for ri, row in enumerate(self._rows_data):
            y0 = ri * ROW_H
            is_last = ri == len(self._rows_data) - 1
            bg = (COLORS["row_last"] if is_last
                  else (COLORS["row_even"] if ri % 2 == 0 else COLORS["row_odd"]))
            fg = COLORS["row_last_fg"] if is_last else COLORS["text"]

            for ci, val in enumerate(row):
                x0 = ci * cw
                self._body_canvas.create_rectangle(
                    x0, y0, x0 + cw, y0 + ROW_H,
                    fill=bg, outline=COLORS["border"], width=0.5)
                txt = f"{val:.8g}" if isinstance(val, (float, np.floating)) else str(val)
                self._body_canvas.create_text(
                    x0 + cw // 2, y0 + ROW_H // 2,
                    text=txt, fill=fg,
                    font=("Courier", 9, "bold" if is_last else "normal"))
