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
        all_discrete_norms,
        all_continuous_norms,
        approximation_error_discrete,
    )
except ImportError as e:
    print("Warning: Could not import interpolation module:", e)


class Axe3Screen(tk.Tk):
    """
    Axe 3 — Interpolation / Approximation (Styled + Scrollable Left Panel)
    """

    def __init__(self):
        super().__init__()
        self.title("Axe 3 — Interpolation / Approximation")
        self.geometry("1280x920")
        self.minsize(1100, 700)
        self.configure(bg="#f0f4f8")
        self._last_y_approx = None
        self._last_real_func = None
        self._last_interval = (-1, 1)
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

    # ── Left Panel with Scrollbar ─────────────────────────────────
    def _left_panel(self, parent):
        outer = tk.Frame(parent, bg="#f0f4f8")
        outer.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        canvas = tk.Canvas(outer, bg="#f0f4f8", highlightthickness=0)
        scrollbar = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg="#f0f4f8")

        self.scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        self._build_left_content(self.scrollable_frame)

    def _build_left_content(self, parent):
        # User Inputs
        inp = tk.LabelFrame(parent, text="User Inputs", bg="#f0f4f8", fg="#8e44ad",
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

        tk.Button(parent, text="▶  Run Method", bg="#8e44ad", fg="white",
                  font=("Helvetica", 12, "bold"), relief="flat", height=2,
                  command=self._run_method).pack(fill="x", pady=12)

        # Norms & Error Analysis
        norm_card = tk.LabelFrame(parent, text="Norms & Error Analysis", bg="#f0f4f8", 
                                  fg="#8e44ad", font=("Helvetica", 11, "bold"), padx=10, pady=8)
        norm_card.pack(fill="x", pady=8)

        for text, cmd in [
            ("Discrete Norms (L1, L2, L∞)", self._compute_discrete_norms),
            ("Continuous Norms", self._compute_continuous_norms),
            ("Error Analysis (Discrete)", self._error_discrete),
            ("Error Analysis (Continuous)", self._error_continuous),
        ]:
            tk.Button(norm_card, text=text, bg="#f4eefa", fg="#6c3483",
                      font=("Helvetica", 10), relief="solid", bd=1,
                      width=35, cursor="hand2", command=cmd).pack(pady=4, anchor="w")

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
            tk.Label(self.extra_frame, text="Function f(x):", bg="#f0f4f8", font=("Helvetica", 10, "bold")).pack(anchor="w")
            self.ls_func_entry = tk.Entry(self.extra_frame, width=35, font=("Courier", 11))
            self.ls_func_entry.insert(0, "cos(x)")
            self.ls_func_entry.pack(anchor="w", pady=4)

            tk.Label(self.extra_frame, text="Polynomial Degree:", bg="#f0f4f8").pack(anchor="w")
            self.degree_var = tk.IntVar(value=2)
            f = tk.Frame(self.extra_frame, bg="#f0f4f8")
            f.pack(anchor="w")
            for d in [1, 2, 3, 4]:
                tk.Radiobutton(f, text=d, variable=self.degree_var, value=d, bg="#f0f4f8").pack(side="left", padx=12)

        elif method == "Chebyshev":
            tk.Label(self.extra_frame, text="Interval [a, b]:", bg="#f0f4f8").pack(anchor="w")
            iv = tk.Frame(self.extra_frame, bg="#f0f4f8")
            iv.pack(anchor="w", pady=4)
            self.a_entry = tk.Entry(iv, width=6); self.a_entry.insert(0, "-1"); self.a_entry.pack(side="left")
            tk.Label(iv, text=" to ", bg="#f0f4f8").pack(side="left")
            self.b_entry = tk.Entry(iv, width=6); self.b_entry.insert(0, "1"); self.b_entry.pack(side="left")

            tk.Label(self.extra_frame, text="Degree:", bg="#f0f4f8").pack(anchor="w")
            self.cheb_degree = tk.IntVar(value=5)
            for d in [3,5,7,9]:
                tk.Radiobutton(self.extra_frame, text=d, variable=self.cheb_degree, value=d, bg="#f0f4f8").pack(anchor="w")

        elif method == "Gradient Descent":
            tk.Label(self.extra_frame, text="Function f(vars):", bg="#f0f4f8", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(8, 2))
            self.gd_func_entry = tk.Entry(self.extra_frame, width=40)
            self.gd_func_entry.insert(0, "(x-1)**2 + 2*(y+2)**2")
            self.gd_func_entry.pack(anchor="w", pady=2)

            tk.Label(self.extra_frame, text="Variables:", bg="#f0f4f8").pack(anchor="w", pady=(8, 2))
            self.gd_vars_entry = tk.Entry(self.extra_frame, width=20)
            self.gd_vars_entry.insert(0, "x,y")
            self.gd_vars_entry.pack(anchor="w", pady=2)

            tk.Label(self.extra_frame, text="x0 (comma-separated):", bg="#f0f4f8").pack(anchor="w", pady=(8, 2))
            self.gd_x0_entry = tk.Entry(self.extra_frame, width=20)
            self.gd_x0_entry.insert(0, "0,0")
            self.gd_x0_entry.pack(anchor="w", pady=2)

            tk.Label(self.extra_frame, text="Learning Rate (lr):", bg="#f0f4f8").pack(anchor="w", pady=(8, 2))
            self.gd_lr_entry = tk.Entry(self.extra_frame, width=12)
            self.gd_lr_entry.insert(0, "0.01")
            self.gd_lr_entry.pack(anchor="w", pady=2)

            tk.Label(self.extra_frame, text="Max Iterations:", bg="#f0f4f8").pack(anchor="w", pady=(4, 2))
            self.gd_max_iter_entry = tk.Entry(self.extra_frame, width=12)
            self.gd_max_iter_entry.insert(0, "1000")
            self.gd_max_iter_entry.pack(anchor="w", pady=2)

            tk.Label(self.extra_frame, text="Tolerance:", bg="#f0f4f8").pack(anchor="w", pady=(4, 2))
            self.gd_tol_entry = tk.Entry(self.extra_frame, width=12)
            self.gd_tol_entry.insert(0, "1e-6")
            self.gd_tol_entry.pack(anchor="w", pady=2)

    # ── Right Panel ───────────────────────────────────────────────
    def _right_panel(self, parent):
        frame = tk.Frame(parent, bg="#f0f4f8")
        frame.grid(row=0, column=1, sticky="nsew")

        self.result_frame = tk.LabelFrame(frame, text="Results", bg="#f0f4f8", fg="#2c3e50",
                                          font=("Helvetica", 11, "bold"))
        self.result_frame.pack(fill="both", expand=True, padx=8, pady=8)

        poly_lf = tk.LabelFrame(self.result_frame, text="Result", bg="#f0f4f8", fg="#8e44ad")
        poly_lf.pack(fill="x", padx=8, pady=6)
        self.poly_text = tk.Text(poly_lf, height=4, font=("Courier", 10), bg="#f8f1ff")
        self.poly_text.pack(fill="x", padx=8, pady=6)

    # ── Run Method (clean) ───────────────────────────────────────
    def _run_method(self):
        try:
            method = self.method_var.get()
            x_data = [float(xe.get()) for xe, ye in self.point_entries if xe.get().strip() and ye.get().strip()]
            y_data = [float(ye.get()) for xe, ye in self.point_entries if xe.get().strip() and ye.get().strip()]

            x_sym = sp.symbols('x')
            result_text = ""
            y_approx = None
            real_func = None
            history = []
            obj_vals = []

            if method in ["Lagrange", "Newton", "Least Squares"] and len(x_data) < 2:
                messagebox.showerror("Error", "Please enter at least 2 points.")
                return

            if method == "Lagrange":
                poly, _ = lagrange_interpolation(x_data, y_data)
                result_text = str(poly)
                y_approx = sp.lambdify(x_sym, poly, "numpy")

            elif method == "Newton":
                diff_table = newton_differences(x_data, y_data)
                poly = newton_polynomial(x_data, diff_table)
                result_text = str(poly)
                y_approx = sp.lambdify(x_sym, poly, "numpy")

            elif method == "Least Squares":
                degree = self.degree_var.get()
                func_str = self.ls_func_entry.get()
                func_sym = sp.sympify(func_str)
                real_func = sp.lambdify(x_sym, func_sym, "numpy")

                res = least_squares_polynomial(x_data, y_data, degree)
                result_text = str(res["poly_sympy"])
                y_approx = res["poly_numpy"]

            elif method == "Chebyshev":
                a = float(self.a_entry.get())
                b = float(self.b_entry.get())
                deg = self.cheb_degree.get()
                def f(t): return np.cos(np.asarray(t, dtype=float))
                res = chebyshev_approximation(f, deg, a, b)
                result_text = str(res["poly_sympy"])
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
                if not var_names:
                    messagebox.showerror("Error", "Enter at least one variable name")
                    return

                vars_syms = sp.symbols(' '.join(var_names))
                if len(var_names) == 1:
                    vars_syms = (vars_syms,)

                local_dict = {n: v for n, v in zip(var_names, vars_syms)}
                func_sympy = sp.sympify(func_str, locals=local_dict)

                x0 = [float(s.strip()) for s in x0_str.split(',') if s.strip()]
                if len(x0) != len(var_names):
                    messagebox.showerror("Error", "Initial estimation must match number of variables")
                    return

                x_opt, history = gradient_descent_sympy(
                    func_sympy, vars_syms, x0, lr=lr, max_iter=max_iter, tol=tol, verbose=False
                )
                
                func_eval = sp.lambdify(vars_syms, func_sympy, modules="numpy")
                obj_vals = [float(func_eval(*np.atleast_1d(xk))) for xk, _ in history]

                vals = [float(v) for v in np.atleast_1d(x_opt)]
                result_text = f"x* = [{', '.join(f'{v:.8g}' for v in vals)}]"

                y_approx = None
                real_func = None

            self.poly_text.delete("1.0", tk.END)
            if method == "Gradient Descent":
                self.poly_text.insert("1.0", result_text)
            else:
                self.poly_text.insert("1.0", f"P(x) = {result_text}")

            # Cache for Norms & Error Analysis buttons
            self._last_y_approx = y_approx
            self._last_real_func = real_func
            if x_data:
                self._last_interval = (min(x_data), max(x_data))

            if method == "Gradient Descent":
                self._plot_gradient_descent(history, obj_vals)
            else:
                self._create_plot_with_table(x_data, y_data, y_approx, method, real_func)

        except Exception as e:
            messagebox.showerror("Error", f"Execution failed:\n{str(e)}")

    # ── Plot Methods (unchanged) ─────────────────────────────────
    def _create_plot_with_table(self, x_data, y_data, y_approx, method, real_func=None):
        for widget in list(self.result_frame.winfo_children()):
            if isinstance(widget, tk.Canvas) or "FigureCanvas" in str(type(widget)):
                widget.destroy()

        fig = Figure(figsize=(10, 8), dpi=105)
        gs = fig.add_gridspec(3, 1, height_ratios=[4, 0.4, 2.5])

        ax = fig.add_subplot(gs[0])
        ax.scatter(x_data, y_data, color='red', s=70, label='Data Points', zorder=5)

        x_plot = np.linspace(min(x_data)-1, max(x_data)+1, 400)
        y_plot = np.array([y_approx(xi) for xi in x_plot]) if callable(y_approx) else y_approx(x_plot)

        ax.plot(x_plot, y_plot, 'b-', linewidth=2.5, label=f'{method} Approximation')
        ax.set_title(f"{method} Approximation")
        ax.legend()
        ax.grid(True)

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

    # ── Norms & Error Buttons ─────────────────────────────────────
    def _compute_discrete_norms(self):
        try:
            y_values = [float(ye.get()) for xe, ye in self.point_entries if ye.get().strip()]
            if not y_values:
                messagebox.showerror("Error", "Enter some Y values in data points")
                return
            result = all_discrete_norms(y_values)
            self.poly_text.delete("1.0", tk.END)
            self.poly_text.insert("1.0", f"Discrete Norms:\n{result}")
            messagebox.showinfo("Discrete Norms", str(result))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _compute_continuous_norms(self):
        try:
            # Use function from Least Squares if available, otherwise default
            func_str = getattr(self, 'ls_func_entry', None)
            f_str = func_str.get() if func_str and func_str.get() else "cos(x)"
            
            f_sym = sp.sympify(f_str)
            f_np = sp.lambdify(sp.symbols('x'), f_sym, "numpy")
            
            result = all_continuous_norms(f_np, -2, 2)
            
            self.poly_text.delete("1.0", tk.END)
            self.poly_text.insert("1.0", f"Continuous Norms on [-2, 2]:\n{result}")
            messagebox.showinfo("Continuous Norms", str(result))
        except Exception as e:
            messagebox.showerror("Error", str(e))
    def _error_discrete(self):
        try:
            x_vals = [float(xe.get()) for xe, ye in self.point_entries if xe.get().strip() and ye.get().strip()]
            y_data = [float(ye.get()) for xe, ye in self.point_entries if xe.get().strip() and ye.get().strip()]
            if len(y_data) < 2:
                messagebox.showerror("Error", "Need at least 2 points")
                return

            # Use the last-run polynomial approximation if available, else linear interpolation
            if hasattr(self, '_last_y_approx') and self._last_y_approx is not None:
                y_approx = [float(self._last_y_approx(xi)) for xi in x_vals]
            else:
                from scipy.interpolate import interp1d
                f_interp = interp1d(range(len(y_data)), y_data, kind='linear')
                y_approx = [float(f_interp(i)) for i in range(len(y_data))]

            result = approximation_error_discrete(y_data, y_approx)
            msg = (
                f"L1  = {result['L1']:.6g}\n"
                f"L2  = {result['L2']:.6g}\n"
                f"L∞  = {result['Linf']:.6g}  (at index {result['Linf_index']}, "
                f"error = {result['Linf_value']:.6g})"
            )
            self.poly_text.delete("1.0", tk.END)
            self.poly_text.insert("1.0", f"Error Analysis (Discrete):\n{msg}")
            messagebox.showinfo("Error Analysis (Discrete)", msg)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _error_continuous(self):
        try:
            # Require a last-run approximation polynomial
            if not hasattr(self, '_last_y_approx') or self._last_y_approx is None:
                messagebox.showerror("Error", "Run a method first to obtain an approximation polynomial.")
                return
            if not hasattr(self, '_last_real_func') or self._last_real_func is None:
                messagebox.showerror("Error", "Continuous error requires a known real function.\n"
                                              "Run 'Least Squares' or 'Chebyshev' (which use a real function).")
                return

            from interpolation_approximation import approximation_error_continuous
            a = self._last_interval[0]
            b = self._last_interval[1]
            result = approximation_error_continuous(self._last_real_func, self._last_y_approx, a, b)
            msg = (
                f"Interval: [{a:.4g}, {b:.4g}]\n"
                f"L1  = {result['L1']:.6g}\n"
                f"L2  = {result['L2']:.6g}\n"
                f"L∞  = {result['Linf']:.6g}  (at x = {result['Linf_xmax']:.6g})"
            )
            self.poly_text.delete("1.0", tk.END)
            self.poly_text.insert("1.0", f"Error Analysis (Continuous):\n{msg}")
            messagebox.showinfo("Error Analysis (Continuous)", msg)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _back(self):
        import subprocess
        path = os.path.join(os.path.dirname(__file__), "main_screen.py")
        subprocess.Popen([sys.executable, path])
        self.destroy()


if __name__ == "__main__":
    Axe3Screen().mainloop()