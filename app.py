import tkinter as tk
from tkinter import ttk

# Import animation routines
from animations import (
    animate_tsp as animate_tsp_ga,
    animate_algorithm,
    animate_de,
    animate_pso,
    animate_soma,
)
from aco_animation import animate_aco


def build_sidebar(parent, plot_frame):
    sidebar = ttk.Frame(parent)
    sidebar.pack(side="left", fill="y", padx=8, pady=8)

    # Title
    ttk.Label(sidebar, text="Optimalizační algoritmy", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 6))

    # TSP section
    ttk.Label(sidebar, text="TSP - Genetic Algorithm", font=("Arial", 11, "bold"), foreground="blue").pack(anchor="w", pady=(8, 2))
    ttk.Button(sidebar, text="TSP (20 měst)", command=lambda: animate_tsp_ga(plot_frame, n_cities=20, generations=200, seed=42)).pack(fill="x", pady=2)
    ttk.Button(sidebar, text="TSP (30 měst)", command=lambda: animate_tsp_ga(plot_frame, n_cities=30, generations=250, seed=42)).pack(fill="x", pady=2)
    ttk.Button(sidebar, text="TSP (40 měst)", command=lambda: animate_tsp_ga(plot_frame, n_cities=40, generations=300, seed=42)).pack(fill="x", pady=2)

    ttk.Separator(sidebar, orient='horizontal').pack(fill='x', pady=8)

    # ACO section
    ttk.Label(sidebar, text="TSP - Ant Colony Optimization", font=("Arial", 11, "bold"), foreground="green").pack(anchor="w", pady=(8, 2))
    ttk.Button(sidebar, text="ACO (20 měst)", command=lambda: animate_aco(plot_frame, n_cities=20, iterations=100, seed=42)).pack(fill="x", pady=2)
    ttk.Button(sidebar, text="ACO (30 měst)", command=lambda: animate_aco(plot_frame, n_cities=30, iterations=150, seed=42)).pack(fill="x", pady=2)
    ttk.Button(sidebar, text="ACO (40 měst)", command=lambda: animate_aco(plot_frame, n_cities=40, iterations=200, seed=42)).pack(fill="x", pady=2)

    ttk.Separator(sidebar, orient='horizontal').pack(fill='x', pady=8)

    # Benchmark functions section
    ttk.Label(sidebar, text="Benchmark Functions", font=("Arial", 11, "bold"), foreground="purple").pack(anchor="w", pady=(8, 2))

    # Buttons for a couple of functions with different algorithms
    def add_func_group(name, key, lb, ub):
        frm = ttk.LabelFrame(sidebar, text=name)
        frm.pack(fill="x", padx=0, pady=4)
        ttk.Button(frm, text="Blind Search", command=lambda: animate_algorithm(plot_frame, key, lb, ub, algo="blind", iterations=150, seed=42)).pack(fill="x", padx=4, pady=2)
        ttk.Button(frm, text="Hill Climbing", command=lambda: animate_algorithm(plot_frame, key, lb, ub, algo="hill", iterations=200, seed=42)).pack(fill="x", padx=4, pady=2)
        ttk.Button(frm, text="Simulated Annealing", command=lambda: animate_algorithm(plot_frame, key, lb, ub, algo="simann", iterations=200, seed=42)).pack(fill="x", padx=4, pady=2)
        ttk.Button(frm, text="Differential Evolution", command=lambda: animate_de(plot_frame, key, lb, ub, iterations=200, NP=40, seed=42)).pack(fill="x", padx=4, pady=2)
        ttk.Button(frm, text="Particle Swarm", command=lambda: animate_pso(plot_frame, key, lb, ub, iterations=200, pop_size=30, seed=42)).pack(fill="x", padx=4, pady=2)
        ttk.Button(frm, text="SOMA", command=lambda: animate_soma(plot_frame, key, lb, ub, iterations=200, pop_size=30, seed=42)).pack(fill="x", padx=4, pady=2)

    add_func_group("Sphere", "sphere", -5, 5)
    add_func_group("Ackley", "ackley", -5, 5)

    return sidebar


def build_plot_area(parent):
    frame = ttk.Frame(parent)
    frame.pack(side="right", fill="both", expand=True)
    # A placeholder label until an animation is loaded
    lbl = ttk.Label(frame, text="Vyberte algoritmus vlevo…", anchor="center")
    lbl.pack(fill="both", expand=True, padx=20, pady=20)
    return frame


def main():
    root = tk.Tk()
    root.title("BIA CV1 - Vizualizace")
    root.geometry("1200x700")

    container = ttk.Frame(root)
    container.pack(fill="both", expand=True)

    plot_frame = build_plot_area(container)
    build_sidebar(container, plot_frame)

    root.mainloop()


if __name__ == "__main__":
    main()
