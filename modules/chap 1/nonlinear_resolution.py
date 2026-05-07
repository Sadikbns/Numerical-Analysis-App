import numpy as np


def exo5_f(x):
    return np.exp(x) + (x**2 / 2) + x - 1


def exo5_df(x):
    return np.exp(x) + x + 1


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


def solve_bisection_detail(a, b, eps=1e-5, max_it=100, func=exo5_f):
    history = []
    fa = func(a)
    fb = func(b)

    if fa * fb >= 0:
        return history

    left = a
    right = b
    f_left = fa

    for i in range(max_it):
        mid = (left + right) / 2
        f_mid = func(mid)
        err = (right - left) / 2
        history.append([i + 1, left, right, mid, f_mid, err])

        if err < eps:
            break

        if f_left * f_mid < 0:
            right = mid
        else:
            left = mid
            f_left = f_mid

    return history


__all__ = [
    "exo5_f",
    "exo5_df",
    "executer_point_fixe",
    "solve_newton",
    "solve_bisection_detail",
]
