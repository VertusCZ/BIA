import math
import tkinter as tk
from tkinter import ttk

from animations import animate_tsp, animate_algorithm, animate_de, animate_pso, animate_soma
def main():
    root = tk.Tk()
    root.title("Optimization Algorithms Visualization")
    root.geometry("1300x760")

    menu_container = tk.Frame(root)
    menu_container.pack(side="left", fill="y")

    canvas = tk.Canvas(menu_container, width=240)
    scrollbar = ttk.Scrollbar(menu_container, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="y", expand=True)
    scrollbar.pack(side="right", fill="y")

    plot_frame = tk.Frame(root)
    plot_frame.pack(side="right", fill="both", expand=True)

    # TSP section
    ttk.Label(scrollable_frame, text="TSP - Genetic Algorithm",
              font=("Arial", 11, "bold"), foreground="blue").pack(pady=(8, 2))

    ttk.Button(scrollable_frame, text="TSP (20 měst)",
               command=lambda: animate_tsp(plot_frame, n_cities=20, generations=200, seed=42)).pack(padx=5, pady=2,
                                                                                                     fill="x")
    ttk.Button(scrollable_frame, text="TSP (30 měst)",
               command=lambda: animate_tsp(plot_frame, n_cities=30, generations=250, seed=42)).pack(padx=5, pady=2,
                                                                                                     fill="x")
    ttk.Button(scrollable_frame, text="TSP (40 měst)",
               command=lambda: animate_tsp(plot_frame, n_cities=40, generations=300, seed=42)).pack(padx=5, pady=2,
                                                                                                     fill="x")

    ttk.Separator(scrollable_frame, orient='horizontal').pack(fill='x', pady=8)

    functions = [
        ("Sphere", "sphere", -5, 5),
        ("Ackley", "ackley", -5, 5),
        ("Rastrigin", "rastrigin", -5.12, 5.12),
        ("Rosenbrock", "rosenbrock", -3, 3),
        ("Schwefel", "schwefel", -500, 500),
        ("Levy", "levy", -10, 10),
        ("Michalewicz", "michalewicz", 0, math.pi),
        ("Zakharov", "zakharov", -5, 5)
    ]

    for label, fname, lb, ub in functions:
        ttk.Label(scrollable_frame, text=label, font=("Arial", 10, "bold")).pack(pady=(8, 2))
        ttk.Button(scrollable_frame, text=f"Blind Search {label}",
                   command=lambda f=fname, lo=lb, up=ub: animate_algorithm(plot_frame, f, lo, up, algo="blind", iterations=300, seed=42)).pack(padx=5, pady=2, fill="x")
        ttk.Button(scrollable_frame, text=f"Hill Climbing {label}",
                   command=lambda f=fname, lo=lb, up=ub: animate_algorithm(plot_frame, f, lo, up, algo="hill", iterations=300, seed=42)).pack(padx=5, pady=2, fill="x")
        ttk.Button(scrollable_frame, text=f"Simulated Annealing {label}",
                   command=lambda f=fname, lo=lb, up=ub: animate_algorithm(plot_frame, f, lo, up, algo="sa",
                                                                            iterations=300, seed=42)).pack(padx=5,
                                                                                                           pady=2,
                                                                                                           fill="x")
        ttk.Button(scrollable_frame, text=f"Diff. Evolution {label}",
                   command=lambda f=fname, lo=lb, up=ub: animate_de(plot_frame, f, lo, up, 
                                                                     iterations=200, NP=50, F=0.5, CR=0.9, seed=42)).pack(padx=5, pady=2, fill="x")
        ttk.Button(scrollable_frame, text=f"PSO {label}",
                   command=lambda f=fname, lo=lb, up=ub: animate_pso(plot_frame, f, lo, up, 
                                                                      iterations=200, pop_size=30, w=0.7, c1=1.5, c2=1.5, seed=42)).pack(padx=5, pady=2, fill="x")
        ttk.Button(scrollable_frame, text=f"SOMA {label}",
                   command=lambda f=fname, lo=lb, up=ub: animate_soma(plot_frame, f, lo, up,
                                                                       iterations=200, pop_size=30, path_length=3.0, step=0.11, prt=0.1, strategy='all_to_one', seed=42)).pack(padx=5, pady=2, fill="x")

    root.mainloop()
