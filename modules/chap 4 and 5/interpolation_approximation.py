import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def lagrange_interpolation(x_nodes, y_nodes):
    x = sp.symbols("x")
    n = len(x_nodes)
    poly_final = 0
    li_basis = []

    for i in range(n):
        li = 1
        for j in range(n):
            if i != j:
                li *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])

        li_basis.append(sp.simplify(li))
        poly_final += y_nodes[i] * li

    return sp.simplify(poly_final), li_basis


def newton_differences(x_nodes, y_nodes):
    n = len(x_nodes)
    matrix = np.zeros((n, n))
    matrix[:, 0] = y_nodes

    for j in range(1, n):
        for i in range(n - j):
            matrix[i, j] = (
                matrix[i + 1, j - 1] - matrix[i, j - 1]
            ) / (x_nodes[i + j] - x_nodes[i])

    return matrix


def newton_polynomial(x_nodes, matrix):
    x = sp.symbols("x")
    n = len(x_nodes)
    poly = matrix[0, 0]
    product = 1

    for i in range(1, n):
        product *= x - x_nodes[i - 1]
        poly += matrix[0, i] * product

    return sp.simplify(poly)


def least_squares_polynomial(x_data, y_data, degree):
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    if x_data.shape != y_data.shape:
        raise ValueError("x_data and y_data must have the same length")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if degree >= len(x_data):
        raise ValueError("degree must be smaller than number of data points")

    coeffs = np.polyfit(x_data, y_data, degree)
    poly_np = np.poly1d(coeffs)

    x = sp.symbols("x")
    poly_sp = sum(float(c) * x ** (degree - i) for i, c in enumerate(coeffs))

    y_pred = poly_np(x_data)
    residuals = y_data - y_pred
    mse = float(np.mean(residuals**2))

    return {
        "coefficients": coeffs,
        "poly_numpy": poly_np,
        "poly_sympy": sp.simplify(poly_sp),
        "residuals": residuals,
        "mse": mse,
    }


def chebyshev_approximation(func, degree, a, b):
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if not (a < b):
        raise ValueError("interval must satisfy a < b")

    x = sp.symbols("x")
    t = sp.symbols("t")
    x_from_t = (a + b) / 2 + (b - a) * t / 2
    f_t = lambda tt: func((a + b) / 2 + (b - a) * tt / 2)

    roots = np.cos((2 * np.arange(1, degree + 2) - 1) * np.pi / (2 * (degree + 1)))
    values = f_t(roots)
    cheb_coeffs = np.polynomial.chebyshev.chebfit(roots, values, degree)
    poly_t = np.polynomial.chebyshev.Chebyshev(cheb_coeffs, domain=[-1, 1])
    poly_t_power = np.polynomial.Polynomial(poly_t.convert(kind=np.polynomial.Polynomial).coef)
    power_coeffs = poly_t_power.coef
    t_expr = (2 * x - (a + b)) / (b - a)
    poly_x = sum(float(c) * t_expr**i for i, c in enumerate(power_coeffs))
    poly_x = sp.expand(poly_x)

    eval_fn = lambda xx: poly_t((2 * np.asarray(xx) - (a + b)) / (b - a))

    return {
        "chebyshev_coefficients": cheb_coeffs,
        "poly_t": poly_t,
        "poly_sympy": sp.simplify(poly_x),
        "evaluate": eval_fn,
        "interval": (a, b),
    }


def gradient_descent_numpy(grad_fn, x0, lr=0.01, max_iter=1000, tol=1e-6, verbose=False):
    x = np.asarray(x0, dtype=float)
    history = []
    for k in range(1, max_iter + 1):
        g = np.asarray(grad_fn(x), dtype=float)
        gn = np.linalg.norm(g)
        history.append((x.copy(), gn))
        
        if verbose and (k % max(1, max_iter // 10) == 0 or gn <= tol):
            print(f"iter={k}, ||∇f|| = {gn:.3e}")
        
        if gn <= tol:
            break
        
        x = x - lr * g
    
    return x, history


def gradient_descent_sympy(func_sympy, vars, x0, lr=0.01, max_iter=1000, tol=1e-6, verbose=False):
    
    vars = tuple(vars)
    grad_exprs = [sp.diff(func_sympy, v) for v in vars]
    grad_fn = sp.lambdify(vars, grad_exprs, modules="numpy")
    def _grad_fn_numeric(x_arr):
        return np.asarray(grad_fn(*tuple(x_arr)), dtype=float)

    x_opt, history = gradient_descent_numpy(_grad_fn_numeric, x0, lr=lr, max_iter=max_iter, tol=tol, verbose=verbose)
    return x_opt, history


def run_tp4_visualizations():
    x = sp.symbols("x")
    x_pts = [0, 1, 5, 8]
    y_pts = [0, 3, 2, 2]
    p_lagrange, _ = lagrange_interpolation(x_pts, y_pts)
    p_func = sp.lambdify(x, p_lagrange, "numpy")

    plt.figure("Ex 3.1: Point Cloud")
    x_range = np.linspace(0, 8, 100)
    plt.plot(x_range, p_func(x_range), label="Interpolating polynomial")
    plt.scatter(x_pts, y_pts, color="red", label="Given points")
    plt.title("Point cloud interpolation")
    plt.legend()
    plt.grid(True)

    functions = [
        {"name": "cos(x)", "f": lambda t: np.cos(t), "lim": [-4 * np.pi, 4 * np.pi]},
        {
            "name": "exp(-1/(1+x^2))",
            "f": lambda t: np.exp(-1 / (1 + t**2)),
            "lim": [-4, 4],
        },
        {
            "name": "Runge: 1/(1+25x^2)",
            "f": lambda t: 1 / (1 + 25 * t**2),
            "lim": [-1, 1],
        },
    ]

    for f_info in functions:
        plt.figure(f"Ex 3.2: {f_info['name']}")
        x_fine = np.linspace(f_info["lim"][0], f_info["lim"][1], 500)
        plt.plot(x_fine, f_info["f"](x_fine), "k--", label="Real function", alpha=0.3)

        for n in [4, 8, 10]:
            x_n = np.linspace(f_info["lim"][0], f_info["lim"][1], n + 1)
            y_n = f_info["f"](x_n)
            p, _ = lagrange_interpolation(x_n, y_n)
            p_eval = sp.lambdify(x, p, "numpy")
            plt.plot(x_fine, p_eval(x_fine), label=f"n={n}")

        plt.title(f"Analysis of {f_info['name']}")
        plt.legend()
        plt.grid(True)

    plt.show()


__all__ = [
    "lagrange_interpolation",
    "newton_differences",
    "newton_polynomial",
    "least_squares_polynomial",
    "chebyshev_approximation",
    "gradient_descent_numpy",
    "gradient_descent_sympy",
    "run_tp4_visualizations",
]
