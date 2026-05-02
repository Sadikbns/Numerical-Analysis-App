import tkinter as tk
from tkinter import messagebox
import sys
import os
import numpy as np
import sympy as sp
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import your algorithms
sys.path.append(os.path.join(os.path.dirname(__file__), "modules", "chap 4 and 5"))
try:
    from interpolation_approximation import (
        lagrange_interpolation,
        newton_differences,
        newton_polynomial,
        least_squares_polynomial,
        chebyshev_approximation,
        gradient_descent_sympy,
    )
except ImportError as e:
    print("Warning: Could not import interpolation module:", e)


class Axe3Screen(tk.Tk):
    """
    Axe 3 — Interpolation / Approximation with matplotlib table
    """

    def __init__(self):
        super().__init__()
        self.title("Axe 3 — Interpolation / Approximation")
        self.geometry("1180x860")
        self.configure(bg="#f0f4f8")
        self._build()

    def _build(self):
        self._header()
        content = tk.Frame(self, bg="#f0f4f8")
        content.pack(fill="both", expand=True, padx=15, pady=10)
        content.columnconfigure(0, weight=2)
        content.columnconfigure(1, weight=3)
        content.rowconfigure(0, weight=1)

        self._left_panel(content)
        self._right_panel(content)

    def _header(self):
        bar = tk.Frame(self, bg="#8e44ad", height=55)
        bar.pack(fill="x")
        tk.Button(bar, text="← Main", bg="#6c3483", fg="white",
                  font=("Helvetica", 10), relief="flat", cursor="hand2",
                  command=self._back).pack(side="left", padx=10, pady=12)
        tk.Label(bar, text="Axe 3 — Interpolation / Approximation",
                 bg="#8e44ad", fg="white", font=("Helvetica", 16, "bold")).pack(side="left", padx=10)

    # ── Left Panel (unchanged) ───────────────────────────────────
    def _left_panel(self, parent):
        frame = tk.Frame(parent, bg="#f0f4f8")
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        inp = tk.LabelFrame(frame, text="User Inputs", bg="#f0f4f8", fg="#8e44ad",
                            font=("Helvetica", 11, "bold"), padx=10, pady=8)
        inp.pack(fill="x", pady=(0, 10))

        tk.Label(inp, text="Method:", bg="#f0f4f8", font=("Helvetica", 10, "bold")).pack(anchor="w")
        self.method_var = tk.StringVar(value="Lagrange")

        for m in ["Lagrange", "Newton", "Least Squares", "Chebyshev", "Gradient Descent"]:
            tk.Radiobutton(inp, text=m, variable=self.method_var, value=m,
                           bg="#f0f4f8", command=self._update_inputs).pack(anchor="w")

        tk.Label(inp, text="Data Points (x, y):", bg="#f0f4f8", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(8, 2))
        self.points_frame = tk.Frame(inp, bg="#f0f4f8")
        self.points_frame.pack(fill="x")
        self._build_points_input()

        self.extra_frame = tk.Frame(inp, bg="#f0f4f8")
        self.extra_frame.pack(fill="x", pady=8)
        self._update_inputs()

        tk.Button(frame, text="▶  Run Method", bg="#8e44ad", fg="white",
                  font=("Helvetica", 12, "bold"), relief="flat", height=2,
                  command=self._run_method).pack(fill="x", pady=12)

    def _build_points_input(self):
        for w in self.points_frame.winfo_children():
            w.destroy()
        self.point_entries = []
        tk.Label(self.points_frame, text="x", width=10, bg="#f0f4f8").grid(row=0, column=0)
        tk.Label(self.points_frame, text="y", width=10, bg="#f0f4f8").grid(row=0, column=1)
        for i in range(8):
            xe = tk.Entry(self.points_frame, width=10, font=("Courier", 10), justify="center")
            xe.insert(0, str(i))
            xe.grid(row=i+1, column=0, pady=1)
            ye = tk.Entry(self.points_frame, width=10, font=("Courier", 10), justify="center")
            ye.insert(0, str(round(i*0.8, 2)))
            ye.grid(row=i+1, column=1, pady=1)
            self.point_entries.append((xe, ye))

    def _update_inputs(self):
        for w in self.extra_frame.winfo_children():
            w.destroy()
        method = self.method_var.get()

        if method == "Least Squares":
            tk.Label(self.extra_frame, text="Degree:", bg="#f0f4f8").pack(anchor="w")
            self.degree_var = tk.IntVar(value=2)
            f = tk.Frame(self.extra_frame, bg="#f0f4f8")
            f.pack(anchor="w")
            for d in [1, 2, 3, 4]:
                tk.Radiobutton(f, text=d, variable=self.degree_var, value=d, bg="#f0f4f8").pack(side="left", padx=10)

        elif method == "Chebyshev":
            tk.Label(self.extra_frame, text="Interval [a, b]", bg="#f0f4f8").pack(anchor="w")
            iv = tk.Frame(self.extra_frame, bg="#f0f4f8")
            iv.pack(anchor="w", pady=4)
            self.a_entry = tk.Entry(iv, width=6); self.a_entry.insert(0, "-1"); self.a_entry.pack(side="left")
            tk.Label(iv, text="  to  ", bg="#f0f4f8").pack(side="left")
            self.b_entry = tk.Entry(iv, width=6); self.b_entry.insert(0, "1"); self.b_entry.pack(side="left")

            tk.Label(self.extra_frame, text="Degree:", bg="#f0f4f8").pack(anchor="w")
            self.cheb_degree = tk.IntVar(value=5)
            for d in [3,5,7,9]:
                tk.Radiobutton(self.extra_frame, text=d, variable=self.cheb_degree, value=d, bg="#f0f4f8").pack(anchor="w")

        elif method == "Gradient Descent":
            # (your existing GD inputs - kept unchanged)
            tk.Label(self.extra_frame, text="Function f(vars):", bg="#f0f4f8", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(8, 2))
            self.gd_func_entry = tk.Entry(self.extra_frame, width=40)
            self.gd_func_entry.insert(0, "(x-1)**2 + 2*(y+2)**2")
            self.gd_func_entry.pack(anchor="w", pady=2)

            tk.Label(self.extra_frame, text="Variables (comma-separated):", bg="#f0f4f8").pack(anchor="w", pady=(8, 2))
            self.gd_vars_entry = tk.Entry(self.extra_frame, width=20)
            self.gd_vars_entry.insert(0, "x,y")
            self.gd_vars_entry.pack(anchor="w", pady=2)

            tk.Label(self.extra_frame, text="Initial estimation x0 (comma-separated):", bg="#f0f4f8").pack(anchor="w", pady=(8, 2))
            self.gd_x0_entry = tk.Entry(self.extra_frame, width=20)
            self.gd_x0_entry.insert(0, "0,0")
            self.gd_x0_entry.pack(anchor="w", pady=2)

            f2 = tk.Frame(self.extra_frame, bg="#f0f4f8")
            f2.pack(anchor="w", pady=4)
            tk.Label(f2, text="lr:", bg="#f0f4f8").pack(side="left")
            self.gd_lr_entry = tk.Entry(f2, width=6); self.gd_lr_entry.insert(0, "0.1"); self.gd_lr_entry.pack(side="left", padx=4)
            tk.Label(f2, text="max_iter:", bg="#f0f4f8").pack(side="left")
            self.gd_max_iter_entry = tk.Entry(f2, width=6); self.gd_max_iter_entry.insert(0, "200"); self.gd_max_iter_entry.pack(side="left", padx=4)
            tk.Label(f2, text="tol:", bg="#f0f4f8").pack(side="left")
            self.gd_tol_entry = tk.Entry(f2, width=8); self.gd_tol_entry.insert(0, "1e-6"); self.gd_tol_entry.pack(side="left", padx=4)

    # ── Right Panel with Graph + Matplotlib Table ───────────────
    def _right_panel(self, parent):
        frame = tk.Frame(parent, bg="#f0f4f8")
        frame.grid(row=0, column=1, sticky="nsew")

        self.result_frame = tk.LabelFrame(frame, text="Results", bg="#f0f4f8", fg="#2c3e50",
                                          font=("Helvetica", 11, "bold"))
        self.result_frame.pack(fill="both", expand=True, padx=8, pady=8)

        # Polynomial / Result
        poly_lf = tk.LabelFrame(self.result_frame, text="Result", bg="#f0f4f8", fg="#8e44ad")
        poly_lf.pack(fill="x", padx=8, pady=6)
        self.poly_text = tk.Text(poly_lf, height=4, font=("Courier", 10), bg="#f8f1ff")
        self.poly_text.pack(fill="x", padx=8, pady=6)


    def _run_method(self):
        try:
            # Get points
            x_data = []
            y_data = []
            for xe, ye in self.point_entries:
                if xe.get().strip() and ye.get().strip():
                    x_data.append(float(xe.get()))
                    y_data.append(float(ye.get()))

            method = self.method_var.get()
            x = sp.symbols('x')
            result_text = ""
            y_approx = None
            real_func = None
            history = []
            obj_vals = []

            if method in ["Lagrange", "Newton", "Least Squares", "Chebyshev"] and len(x_data) < 2:
                messagebox.showerror("Error", "Enter at least 2 points")
                return

            if method == "Lagrange":
                poly, _ = lagrange_interpolation(x_data, y_data)
                result_text = str(poly)
                y_approx = sp.lambdify(x, poly, "numpy")

            elif method == "Newton":
                diff_table = newton_differences(x_data, y_data)
                poly = newton_polynomial(x_data, diff_table)
                result_text = str(poly)
                y_approx = sp.lambdify(x, poly, "numpy")

            elif method == "Least Squares":
                degree = self.degree_var.get()
                res = least_squares_polynomial(x_data, y_data, degree)
                poly = res["poly_sympy"]
                result_text = str(poly)
                y_approx = res["poly_numpy"]

            elif method == "Chebyshev":
                a = float(self.a_entry.get())
                b = float(self.b_entry.get())
                deg = self.cheb_degree.get()
                def f(t):
                    return np.cos(np.asarray(t, dtype=float))
                res = chebyshev_approximation(f, deg, a, b)
                poly = res["poly_sympy"]
                result_text = str(poly)
                y_approx = res["evaluate"]
                real_func = f

            elif method == "Gradient Descent":
                func_str = self.gd_func_entry.get()
                vars_str = self.gd_vars_entry.get()
                x0_str = self.gd_x0_entry.get()
                lr = float(self.gd_lr_entry.get())
                max_iter = int(self.gd_max_iter_entry.get())
                tol = float(self.gd_tol_entry.get())

                var_names = [s.strip() for s in vars_str.split(',') if s.strip()]
                if len(var_names) == 0:
                    messagebox.showerror("Error", "Enter at least one variable name")
                    return
                vars_syms = sp.symbols(' '.join(var_names))
                if len(var_names) == 1:
                    vars_syms = (vars_syms,)

                local_dict = {n: v for n, v in zip(var_names, vars_syms)}
                try:
                    func_sympy = sp.sympify(func_str, locals=local_dict)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to parse function: {e}")
                    return

                try:
                    x0 = [float(s.strip()) for s in x0_str.split(',') if s.strip()]
                except Exception:
                    messagebox.showerror("Error", "Invalid initial estimation format")
                    return

                if len(x0) != len(var_names):
                    messagebox.showerror("Error", "Initial estimation must match number of variables")
                    return

                x_opt, history = gradient_descent_sympy(
                    func_sympy, vars_syms, x0, lr=lr, max_iter=max_iter, tol=tol, verbose=False
                )
                
                func_eval = sp.lambdify(vars_syms, func_sympy, modules="numpy")
                obj_vals = []
                for xk, _ in history:
                    try:
                        fk = float(func_eval(*np.atleast_1d(xk)))
                        obj_vals.append(fk)
                    except Exception:
                        obj_vals.append(np.nan)

                vals = [float(v) for v in np.atleast_1d(x_opt)]
                result_text = f"x* = [{', '.join(f'{v:.8g}' for v in vals)}]"

                y_approx = None
                real_func = None

            else:
                messagebox.showinfo("Info", "Method not fully implemented yet.")
                return

            # Update result text
            self.poly_text.delete("1.0", tk.END)
            if method == "Gradient Descent":
                self.poly_text.insert("1.0", result_text)
            else:
                self.poly_text.insert("1.0", f"P(x) = {result_text}")

            # Plot
            if method == "Gradient Descent":
                self._plot_gradient_descent(history, obj_vals)
            else:
                self._create_plot_with_table(x_data, y_data, y_approx, method, real_func)

        except Exception as e:
            messagebox.showerror("Error", f"Execution failed:\n{str(e)}")


    def _create_plot_with_table(self, x_data, y_data, y_approx, method, real_func=None):
        """Combined Graph + Table using plt.table - FIXED"""
        # Clear only plot canvases, keep poly_text and LabelFrames
        for widget in list(self.result_frame.winfo_children()):
            if isinstance(widget, tk.Canvas) or "FigureCanvas" in str(type(widget)):
                widget.destroy()

        fig = Figure(figsize=(10, 8), dpi=105)
        gs = fig.add_gridspec(3, 1, height_ratios=[4, 0.4, 2.5])

        # Plot
        ax = fig.add_subplot(gs[0])
        ax.scatter(x_data, y_data, color='red', s=70, label='Data Points', zorder=5)

        x_plot = np.linspace(min(x_data)-1, max(x_data)+1, 400)
        y_plot = np.array([y_approx(xi) for xi in x_plot]) if callable(y_approx) else y_approx(x_plot)

        ax.plot(x_plot, y_plot, 'b-', linewidth=2.5, label=f'{method} Approximation')
        ax.set_title(f"{method} Approximation")
        ax.legend()
        ax.grid(True)

        # Table
        ax_table = fig.add_subplot(gs[2])
        ax_table.axis('off')

        x_eval = np.linspace(min(x_data), max(x_data), 8)
        table_data = []
        for i, xi in enumerate(x_eval):
            yi_real = float(np.interp(xi, x_data, y_data)) if real_func is None else float(real_func(xi))
            yi_approx = float(y_approx(xi))
            err = abs(yi_real - yi_approx)
            table_data.append([f"{i}", f"{xi:.4f}", f"{yi_real:.4f}", f"{yi_approx:.4f}", f"{err:.2e}"])

        table = ax_table.table(cellText=table_data,
                               colLabels=["i", "x", "y_real", "y_approx", "Error"],
                               cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1.3, 2.0)

        canvas = FigureCanvasTkAgg(fig, self.result_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


    def _plot_gradient_descent(self, history, obj_vals):
        """Gradient Descent Plot"""
        for widget in list(self.result_frame.winfo_children()):
            if isinstance(widget, tk.Canvas) or "FigureCanvas" in str(type(widget)):
                widget.destroy()

        fig = Figure(figsize=(9, 7), dpi=100)
        ax1 = fig.add_subplot(111)
        iters = np.arange(len(history))
        grad_norms = [gn for _, gn in history]

        ax1.semilogy(iters, obj_vals, 'b-o', label='f(x)')
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Objective Value", color='b')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.semilogy(iters, grad_norms, 'r--s', label='||∇f||')
        ax2.set_ylabel("Gradient Norm", color='r')

        ax1.set_title("Gradient Descent Convergence")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.result_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _back(self):
        import subprocess
        path = os.path.join(os.path.dirname(__file__), "main_screen.py")
        subprocess.Popen([sys.executable, path])
        self.destroy()


if __name__ == "__main__":
    Axe3Screen().mainloop()