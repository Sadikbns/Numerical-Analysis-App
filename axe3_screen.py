import tkinter as tk
from tkinter import ttk, messagebox
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
    )
except ImportError as e:
    print("Warning: Could not import interpolation module:", e)


class Axe3Screen(tk.Tk):
    """
    Axe 3 — Interpolation / Approximation (Fully Integrated)
    """

    def __init__(self):
        super().__init__()
        self.title("Axe 3 — Interpolation / Approximation")
        self.geometry("1150x820")
        self.configure(bg="#f0f4f8")

        self.canvas_widget = None  # for matplotlib
        self._build()

    # ──────────────────────────────────────────────────────────────
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

    # ── Left Panel ────────────────────────────────────────────────
    def _left_panel(self, parent):
        frame = tk.Frame(parent, bg="#f0f4f8")
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # User Inputs
        inp = tk.LabelFrame(frame, text="User Inputs", bg="#f0f4f8", fg="#8e44ad",
                            font=("Helvetica", 11, "bold"), padx=10, pady=8)
        inp.pack(fill="x", pady=(0, 10))

        tk.Label(inp, text="Method:", bg="#f0f4f8", font=("Helvetica", 10, "bold")).pack(anchor="w")
        self.method_var = tk.StringVar(value="Lagrange")

        for m in ["Lagrange", "Newton", "Least Squares", "Chebyshev"]:
            tk.Radiobutton(inp, text=m, variable=self.method_var, value=m,
                           bg="#f0f4f8", command=self._update_inputs).pack(anchor="w")

        # Points
        tk.Label(inp, text="Data Points (x, y):", bg="#f0f4f8", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(8, 2))
        self.points_frame = tk.Frame(inp, bg="#f0f4f8")
        self.points_frame.pack(fill="x")
        self._build_points_input()

        # Extra params
        self.extra_frame = tk.Frame(inp, bg="#f0f4f8")
        self.extra_frame.pack(fill="x", pady=8)
        self._update_inputs()

        # Run Button
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

    # ── Right Panel ───────────────────────────────────────────────
    def _right_panel(self, parent):
        frame = tk.Frame(parent, bg="#f0f4f8")
        frame.grid(row=0, column=1, sticky="nsew")

        # Graph
        self.graph_frame = tk.LabelFrame(frame, text="Visualization", bg="#f0f4f8", fg="#2c3e50",
                                         font=("Helvetica", 11, "bold"))
        self.graph_frame.pack(fill="both", expand=True, pady=(0, 8), padx=5)

        # Polynomial
        poly_lf = tk.LabelFrame(frame, text="Resulting Polynomial", bg="#f0f4f8", fg="#8e44ad")
        poly_lf.pack(fill="x", padx=8, pady=6)
        self.poly_text = tk.Text(poly_lf, height=5, font=("Courier", 10), bg="#f8f1ff", wrap="word")
        self.poly_text.pack(fill="x", padx=8, pady=6)

        # Table
        tbl_lf = tk.LabelFrame(frame, text="Evaluation Table", bg="#f0f4f8", fg="#2c3e50")
        tbl_lf.pack(fill="both", expand=True, padx=8, pady=6)
        cols = ("i", "x", "y_real", "y_approx", "error")
        self.tree = ttk.Treeview(tbl_lf, columns=cols, show="headings", height=8)
        for c, w in zip(cols, [40, 80, 100, 100, 100]):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=6, pady=4)

    # ── Main Execution ───────────────────────────────────────────
    def _run_method(self):
        try:
            # Get points
            x_data = []
            y_data = []
            for xe, ye in self.point_entries:
                if xe.get().strip() and ye.get().strip():
                    x_data.append(float(xe.get()))
                    y_data.append(float(ye.get()))

            if len(x_data) < 2:
                messagebox.showerror("Error", "Enter at least 2 points")
                return

            method = self.method_var.get()
            x = sp.symbols('x')
            result_text = ""
            y_approx = None

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
                # Use first function as example or make a function input later
                def f(t): return np.cos(t) if abs(t) < 10 else 0
                res = chebyshev_approximation(f, deg, a, b)
                poly = res["poly_sympy"]
                result_text = str(poly)
                y_approx = res["evaluate"]

            else:
                messagebox.showinfo("Info", "Method not fully implemented yet.")
                return

            # Update polynomial display
            self.poly_text.delete("1.0", tk.END)
            self.poly_text.insert("1.0", f"P(x) = {result_text}")

            # Update table
            for item in self.tree.get_children():
                self.tree.delete(item)

            x_eval = np.linspace(min(x_data)-0.5, max(x_data)+0.5, 10)
            for i, xi in enumerate(x_eval):
                yi_real = np.interp(xi, x_data, y_data) if method != "Chebyshev" else 0
                yi_approx = y_approx(xi) if callable(y_approx) else float(y_approx(xi))
                err = abs(yi_real - yi_approx)
                self.tree.insert("", "end", values=(i, f"{xi:.4f}", f"{yi_real:.4f}", f"{yi_approx:.4f}", f"{err:.2e}"))

            # Plot
            self._plot_results(x_data, y_data, y_approx, method)

        except Exception as e:
            messagebox.showerror("Error", f"Execution failed:\n{str(e)}")

    def _plot_results(self, x_data, y_data, y_approx, method):
        # Clear previous plot
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(7, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(x_data, y_data, color='red', s=60, label='Data Points', zorder=5)

        x_plot = np.linspace(min(x_data)-1, max(x_data)+1, 300)
        if callable(y_approx):
            y_plot = y_approx(x_plot)
        else:
            y_plot = y_approx(x_plot) if hasattr(y_approx, '__call__') else [0]*len(x_plot)

        ax.plot(x_plot, y_plot, 'b-', label=f'{self.method_var.get()} Approximation')
        ax.set_title(f"{method} Interpolation / Approximation")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _back(self):
        import subprocess
        path = os.path.join(os.path.dirname(__file__), "main_screen.py")
        subprocess.Popen([sys.executable, path])
        self.destroy()


if __name__ == "__main__":
    Axe3Screen().mainloop()