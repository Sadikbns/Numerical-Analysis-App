import numpy as np


def numeric_derivative(f, h=1e-7):
    return lambda x: (f(x + h) - f(x - h)) / (2 * h)


def analyze_derivatives(f, expr, a, b):
    df_expr = None
    ddf_expr = None

    try:
        import sympy as sp
        x = sp.Symbol("x")
        df = sp.diff(sp.sympify(expr), x)
        ddf = sp.diff(df, x)
        df_expr = sp.simplify(df)
        ddf_expr = sp.simplify(ddf)
    except Exception:
        pass

    df = numeric_derivative(f)
    ddf = numeric_derivative(df)
    xs = np.linspace(a, b, 300)

    try:
        d1 = np.array([df(xi) for xi in xs], dtype=float)
        d2 = np.array([ddf(xi) for xi in xs], dtype=float)
        valid = np.isfinite(d1) & np.isfinite(d2)
        if not np.any(valid):
            return "Impossible d'analyser les derivees sur [a, b]."

        d1 = d1[valid]
        d2 = d2[valid]
        meme_signe = np.all(d1 * d2 >= 0)
        sens = ("suite xn decroissante et majoree"
                if meme_signe else
                "suite xn croissante et minoree")

        lines = []
        if df_expr is not None and ddf_expr is not None:
            lines.append(f"f'(x)  = {df_expr}")
            lines.append(f"f''(x) = {ddf_expr}")
        else:
            mid = (a + b) / 2
            lines.append(f"f'({mid:.4g}) approx {df(mid):.6g}")
            lines.append(f"f''({mid:.4g}) approx {ddf(mid):.6g}")

        lines.append(
            "f' et f'' ont le meme signe sur [a, b]."
            if meme_signe else
            "f' et f'' ne gardent pas le meme signe sur [a, b].")
        lines.append(sens)
        return "\n".join(lines)
    except Exception:
        return "Impossible de calculer f' et f'' sur [a, b]."


def verify_continuity(f, expr, a, b):
    try:
        import sympy as sp
        x = sp.Symbol("x")
        sym_expr = sp.sympify(expr)
        domain = sp.calculus.util.continuous_domain(sym_expr, x, sp.Interval(a, b))
        if domain == sp.Interval(a, b):
            return f"f est continue sur [{a}, {b}]."
    except Exception:
        pass

    xs = np.linspace(a, b, 800)
    try:
        ys = np.array([f(xi) for xi in xs], dtype=float)
        if not np.all(np.isfinite(ys)):
            return "f n'est pas continue sur tout [a, b]."

        jumps = np.abs(np.diff(ys))
        variation = np.nanmax(ys) - np.nanmin(ys)
        thresh = max(1e-6, variation * 0.2)
        disc = np.any(jumps > thresh)
        if disc:
            return f"Discontinuite probable sur [{a}, {b}]."
        return f"f semble continue sur [{a}, {b}]."
    except Exception:
        return "Impossible d'evaluer f sur l'intervalle."


def check_stability(f, a, b):
    g = lambda x: x - f(x)
    xs = np.linspace(a, b, 800)

    try:
        gs = np.array([g(xi) for xi in xs], dtype=float)
        if not np.all(np.isfinite(gs)):
            return "g(x) n'est pas definie sur tout [a, b]."

        g_min = float(np.min(gs))
        g_max = float(np.max(gs))
        stable = g_min >= a and g_max <= b
        return (
            f"Stabilite g(x) = x - f(x) : {'OUI' if stable else 'NON'}\n"
            f"g([a, b]) approx [{g_min:.6g}, {g_max:.6g}]\n"
            f"Il faut g(x) dans [{a}, {b}] pour tout x dans [a, b].")
    except Exception:
        return "Impossible de verifier la stabilite."


def check_contractante(f, a, b):
    g = lambda x: x - f(x)
    dg = numeric_derivative(g)
    xs = np.linspace(a, b, 800)

    try:
        vals = np.array([dg(xi) for xi in xs], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return "Impossible de calculer g'(x) sur [a, b]."

        k = float(np.max(np.abs(vals)))
        ok = k < 1
        return (
            f"Contractante : {'OUI' if ok else 'NON'}\n"
            f"k = max|g'(x)| approx {k:.6g}\n"
            f"Condition : max(abs(g'(x))) < 1 sur [{a}, {b}].")
    except Exception:
        return "Impossible de verifier la contractante."


__all__ = [
    "numeric_derivative",
    "analyze_derivatives",
    "verify_continuity",
    "check_stability",
    "check_contractante",
]
