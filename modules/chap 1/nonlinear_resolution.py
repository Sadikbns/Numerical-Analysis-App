import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return x**3 - 7 * x**2 + 14 * x - 6

def f2(x):
    return np.exp(x) * np.sin(x) - 2


def f3(x):
    return 2 * x + 3 * np.cos(x) - np.exp(x)


def f4(x):
    return x - 2 ** (-x)


def f5(x):
    return x**2 - 4 * x + 4 - np.log(x)


def get_exo1_functions():
    return [
        (f1, "f(x) = x^3 - 7x^2 + 14x - 6", (-1, 6)),
        (f2, "f(x) = exp(x) * sin(x) - 2", (-2, 4)),
        (f3, "f(x) = 2x + 3*cos(x) - exp(x)", (-2, 4)),
        (f4, "f(x) = x - 2^(-x)", (-1, 2)),
        (f5, "f(x) = x^2 - 4x + 4 - ln(x)", (0.05, 5)),
    ]
def eq1_left(x):
    return x**2 - 1


def eq1_right(x):
    return np.log(x)


def eq2_left(x):
    return 2 * np.exp(-x)


def eq2_right(x):
    return 1 / (x + 2) + 1 / (x + 1)


def eq3_left(x):
    return np.cos(x)


def eq3_right(x):
    return x**2 + x


def eq4_left(x):
    return np.exp(x) - 2


def eq4_right(x):
    return np.cos(np.exp(x) - 2)


def eq5_left(x):
    return x


def eq5_right(x):
    return np.sign(6 * x - 4) * (np.abs(6 * x - 4) ** (1 / 5))


def get_exo2_equations():
    return [
        (eq1_left, eq1_right, (0, 2.5), "function 1"),
        (eq2_left, eq2_right, (-0.9, 4), "function 2"),
        (eq3_left, eq3_right, (-2, 2), "function 3"),
        (eq4_left, eq4_right, (-2, 2), "function 4"),
        (eq5_left, eq5_right, (-3, 3), "function 5"),
    ]

def exo3_function(x):
    return 2 * x * np.cos(2 * x) - (x + 1) ** 2


def bisection(a, b, precision, func=exo3_function):
    if func(a) * func(b) >= 0:
        return None

    while (b - a) / 2 > precision:
        c = (a + b) / 2
        if func(c) == 0:
            return c
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c

    return (a + b) / 2

def exo4_f(x):
    return x**2 - np.log(1 + x)


def phi(x):
    return np.sqrt(np.log(1 + x))


def d_phi(x):
    return 1 / (2 * (1 + x) * np.sqrt(np.log(1 + x)))


def psi(x):
    return np.exp(x**2) - 1


def d_psi(x):
    return 2 * x * np.exp(x**2)


def verifier_conditions(g, dg, a, b):
    x_test = np.linspace(a, b, 100)
    g_vals = g(x_test)
    dg_vals = np.abs(dg(x_test))

    stable = np.all((g_vals >= a) & (g_vals <= b))
    k_max = np.max(dg_vals)
    contractive = k_max < 1

    return {
        "stable": bool(stable),
        "contractive": bool(contractive),
        "k_max": float(k_max),
        "valid": bool(stable and contractive),
    }


def executer_point_fixe(g, x0, epsilon=1e-5, max_iter=15):
    data = []
    x_curr = x0

    for i in range(max_iter):
        x_next = g(x_curr)
        err = abs(x_next - x_curr)
        data.append([i + 1, x_curr, x_next, err])

        if err < epsilon:
            break

        x_curr = x_next

        if err > 1e5:
            break

    return data


def afficher_table(title, columns, cells):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_axis_off()

    table = ax.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title(title, fontweight="bold", pad=20)
    plt.show()


# exo5.py

def exo5_f(x):
    return np.exp(x) + (x**2 / 2) + x - 1


def exo5_df(x):
    return np.exp(x) + x + 1


def exo5_ddf(x):
    return np.exp(x) + 1


def solve_newton(x0, eps=1e-5, max_it=10, func=exo5_f, dfunc=exo5_df):
    history = []
    x_n = x0

    for i in range(max_it):
        fx = func(x_n)
        dfx = dfunc(x_n)
        if abs(dfx) < 1e-12:
            break

        x_next = x_n - fx / dfx
        err = abs(x_next - x_n)
        history.append([i + 1, x_n, x_next, err])

        if err < eps:
            break

        x_n = x_next

    return history


def solve_bisection(a, b, eps=1e-5, max_it=15, func=exo5_f):
    history = []
    if func(a) * func(b) >= 0:
        return history

    for i in range(max_it):
        c = (a + b) / 2
        err = (b - a) / 2
        history.append([i + 1, c, err])

        if err < eps:
            break

        if func(a) * func(c) < 0:
            b = c
        else:
            a = c

    return history


__all__ = [
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "get_exo1_functions",
    "eq1_left",
    "eq1_right",
    "eq2_left",
    "eq2_right",
    "eq3_left",
    "eq3_right",
    "eq4_left",
    "eq4_right",
    "eq5_left",
    "eq5_right",
    "get_exo2_equations",
    "exo3_function",
    "bisection",
    "exo4_f",
    "phi",
    "d_phi",
    "psi",
    "d_psi",
    "verifier_conditions",
    "executer_point_fixe",
    "afficher_table",
    "exo5_f",
    "exo5_df",
    "exo5_ddf",
    "solve_newton",
    "solve_bisection",
]
