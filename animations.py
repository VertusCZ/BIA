import math
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed by Matplotlib)
import numpy as np

from functions_algos import Function, blind_search, hill_climbing, simulated_annealing, differential_evolution
from tsp_ga import TSP, genetic_algorithm_tsp


# --- Helper to embed Matplotlib in Tkinter ---
def show_in_tk(fig, frame, anim=None, update_fn=None, total_frames=None):
    for w in frame.winfo_children():
        w.destroy()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    widget = canvas.get_tk_widget()
    widget.pack(fill="both", expand=True)
    frame._canvas = canvas
    frame._fig = fig
    frame._anim = anim
    frame._anim_update = update_fn
    frame._anim_total = total_frames
    return canvas


# --- TSP animation ---
def animate_tsp(root_frame, n_cities=30, population_size=20, generations=200, seed=42):
    tsp = TSP(n_cities=n_cities, seed=seed)
    history = genetic_algorithm_tsp(tsp, population_size=population_size,
                                    generations=generations, seed=seed)

    fig = plt.Figure(figsize=(12, 6))
    ax_map = fig.add_subplot(121)
    ax_fitness = fig.add_subplot(122)

    ax_map.set_xlim(-5, 105)
    ax_map.set_ylim(-5, 105)
    ax_map.set_title('TSP - Aktuální nejlepší trasa')
    ax_map.set_xlabel('X')
    ax_map.set_ylabel('Y')
    ax_map.grid(True, alpha=0.3)

    cities_x = tsp.cities[:, 0]
    cities_y = tsp.cities[:, 1]
    ax_map.scatter(cities_x, cities_y, c='red', s=100, zorder=5, marker='o')
    for i in range(n_cities):
        ax_map.text(cities_x[i] + 1, cities_y[i] + 1, str(i), fontsize=8)

    ax_fitness.set_title('Vývoj fitness')
    ax_fitness.set_xlabel('Generace')
    ax_fitness.set_ylabel('Délka trasy')
    ax_fitness.grid(True, alpha=0.3)

    route_line, = ax_map.plot([], [], 'b-', linewidth=2, alpha=0.7)
    best_fitness_line, = ax_fitness.plot([], [], 'g-', linewidth=2, label='Nejlepší')
    avg_fitness_line, = ax_fitness.plot([], [], 'b--', linewidth=1, alpha=0.7, label='Průměr')
    ax_fitness.legend()

    frames = len(history)
    state = {'idx': 0, 'playing': True, 'interval': 200}

    def init():
        route_line.set_data([], [])
        best_fitness_line.set_data([], [])
        avg_fitness_line.set_data([], [])
        return route_line, best_fitness_line, avg_fitness_line

    def update(frame):
        i = int(frame)
        state['idx'] = i

        current = history[i]
        best_route = current['best']

        route_x = [tsp.cities[city_idx, 0] for city_idx in best_route]
        route_y = [tsp.cities[city_idx, 1] for city_idx in best_route]
        route_x.append(route_x[0])
        route_y.append(route_y[0])
        route_line.set_data(route_x, route_y)

        generations_data = [h['generation'] for h in history[:i + 1]]
        best_fits = [h['best_fitness'] for h in history[:i + 1]]
        avg_fits = [np.mean(h['fitness']) for h in history[:i + 1]]

        best_fitness_line.set_data(generations_data, best_fits)
        avg_fitness_line.set_data(generations_data, avg_fits)

        ax_fitness.set_xlim(0, max(10, max(generations_data)))
        ax_fitness.set_ylim(min(best_fits) * 0.9, max(avg_fits) * 1.1)

        return route_line, best_fitness_line, avg_fitness_line

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                                  interval=state['interval'], blit=False)
    ani.event_source.start()

    canvas = show_in_tk(fig, root_frame, anim=ani, update_fn=update, total_frames=frames)

    # Info panel nad ovládacími prvky
    info_frame = tk.Frame(root_frame, relief=tk.RIDGE, borderwidth=2, bg='white')
    info_frame.pack(side="bottom", fill="x", padx=5, pady=(5, 0))
    info_label = tk.Label(info_frame, text='Generace: 0 | Nejlepší délka: - | Průměrná délka: -',
                          font=('Arial', 10, 'bold'), bg='white', fg='black', padx=10, pady=8)
    info_label.pack(side='left')

    ctrl = tk.Frame(root_frame)
    ctrl.pack(side="bottom", fill="x")
    slider = ttk.Scale(ctrl, from_=0, to=frames - 1, orient='horizontal', length=400)
    slider.set(0)
    slider.pack(side='left', padx=6, pady=4)
    frame_label = ttk.Label(ctrl, text=f"1/{frames}")
    frame_label.pack(side='left', padx=6)

    def slider_changed(val):
        i = int(float(val))
        ani.event_source.stop()
        update(i)
        canvas.draw()
        frame_label.config(text=f"{i + 1}/{frames}")
        state['playing'] = False
        # Aktualizace info labelu
        current = history[i]
        info_label.config(text=f'Generace: {current["generation"]} | '
                               f'Nejlepší délka: {current["best_fitness"]:.2f} | '
                               f'Průměrná délka: {np.mean(current["fitness"]):.2f}')

    slider.config(command=slider_changed)

    def on_play_pause():
        if state['playing']:
            ani.event_source.stop()
            state['playing'] = False
            play_btn.config(text='Play')
        else:
            ani.event_source.start()
            state['playing'] = True
            play_btn.config(text='Pause')

    play_btn = ttk.Button(ctrl, text='Pause', command=on_play_pause)
    play_btn.pack(side='left', padx=4)

    def step_forward():
        i = min(frames - 1, state['idx'] + 1)
        slider.set(i)
        slider_changed(i)

    def step_back():
        i = max(0, state['idx'] - 1)
        slider.set(i)
        slider_changed(i)

    ttk.Button(ctrl, text='◀', command=step_back).pack(side='left', padx=2)
    ttk.Button(ctrl, text='▶', command=step_forward).pack(side='left', padx=2)
    ttk.Button(ctrl, text='Restart', command=lambda: (slider.set(0), slider_changed(0))).pack(side='left', padx=6)

    ttk.Label(ctrl, text='Rychlost (ms):').pack(side='left', padx=(12, 2))
    speed_var = tk.IntVar(value=state['interval'])

    def speed_changed():
        ival = max(10, speed_var.get())
        state['interval'] = ival
        ani.event_source.interval = ival

    speed_spin = ttk.Spinbox(ctrl, from_=10, to=2000, increment=10, textvariable=speed_var, width=6,
                             command=speed_changed)
    speed_spin.pack(side='left')

    def on_timer(event):
        i = int(state['idx'])
        slider.set(i)
        frame_label.config(text=f"{i + 1}/{frames}")
        # Aktualizace info labelu
        if i < len(history):
            current = history[i]
            info_label.config(text=f'Generace: {current["generation"]} | '
                                   f'Nejlepší délka: {current["best_fitness"]:.2f} | '
                                   f'Průměrná délka: {np.mean(current["fitness"]):.2f}')
        root_frame.after(50, lambda: on_timer(None))

    root_frame.after(50, lambda: on_timer(None))

    return ani


# --- Benchmark algorithms animation ---

def animate_algorithm(root_frame, func_name, lb, ub, algo="blind",
                      iterations=200, seed=42,
                      grid_res=60, max_trail_points=80):
    func = Function(func_name)
    if algo == "blind":
        _, history = blind_search(func, iterations=iterations, lb=lb, ub=ub, seed=seed)
    elif algo == "hill":
        history = hill_climbing(func, lb=lb, ub=ub, iterations=iterations, sigma=0.4, k_neighbors=10, seed=seed)['history']
    elif algo == "sa":
        history = simulated_annealing(func, lb=lb, ub=ub, iterations=iterations, seed=seed)['history']
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    xs = np.linspace(lb, ub, grid_res)
    ys = np.linspace(lb, ub, grid_res)
    X, Y = np.meshgrid(xs, ys)
    vec_eval = np.vectorize(lambda a, b: float(func.eval([a, b])))
    Z = vec_eval(X, Y)

    fig = plt.Figure(figsize=(9, 5))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    ax3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8, linewidth=0, antialiased=False)
    mesh = ax2d.pcolormesh(X, Y, Z, shading='auto')
    fig.colorbar(mesh, ax=ax2d, shrink=0.6)

    ax3d.set_title(f"{func.name.capitalize()} (3D)")
    ax2d.set_title(f"{func.name.capitalize()} (2D kontura)")
    ax2d.set_xlim(lb, ub)
    ax2d.set_ylim(lb, ub)

    positions = np.array([h[0] for h in history])
    values = np.array([h[1] for h in history])
    frames = positions.shape[0]

    trail3d = ax3d.scatter([], [], [], s=18, c='red', alpha=0.25)
    trail2d = ax2d.scatter([], [], s=18, c='red', alpha=0.25)
    scatter3d = ax3d.scatter([], [], [], s=60, c='red')
    scatter2d = ax2d.scatter([], [], s=60, c='red')
    best3d = ax3d.scatter([], [], [], s=140, marker='*', color='gold')
    best2d = ax2d.scatter([], [], s=140, marker='*', color='gold')

    ax3d.view_init(elev=30, azim=-60)

    state = {'idx': 0, 'playing': True, 'interval': 100}

    def init():
        trail3d._offsets3d = ([], [], [])
        trail2d.set_offsets([])
        scatter3d._offsets3d = ([], [], [])
        scatter2d.set_offsets([])
        best3d._offsets3d = ([], [], [])
        best2d.set_offsets([])
        return trail3d, trail2d, scatter3d, scatter2d, best3d, best2d

    def decimate_points(pts, vals, max_points=max_trail_points):
        n = pts.shape[0]
        if n <= max_points:
            return pts, vals
        stride = max(1, n // max_points)
        idx = np.arange(0, n, stride)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        return pts[idx], vals[idx]

    def update(frame):
        i = int(frame)
        state['idx'] = i

        pts = positions[:i + 1]
        zs = values[:i + 1]
        pts_disp, zs_disp = decimate_points(pts, zs, max_points=max_trail_points)
        trail3d._offsets3d = (pts_disp[:, 0], pts_disp[:, 1], zs_disp)
        trail2d.set_offsets(pts_disp[:, 0:2])

        cx, cy = positions[i]
        cz = values[i]
        scatter3d._offsets3d = ([cx], [cy], [cz])
        scatter2d.set_offsets([[cx, cy]])

        best_idx = np.argmin(values[:i + 1])
        bx, by = positions[best_idx]
        bz = values[best_idx]
        best3d._offsets3d = ([bx], [by], [bz])
        best2d.set_offsets([[bx, by]])

        return trail3d, trail2d, scatter3d, scatter2d, best3d, best2d

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                                  interval=state['interval'], blit=False)
    ani.event_source.start()

    canvas = show_in_tk(fig, root_frame, anim=ani, update_fn=update, total_frames=frames)

    ctrl = tk.Frame(root_frame)
    ctrl.pack(side="bottom", fill="x")
    slider = ttk.Scale(ctrl, from_=0, to=frames - 1, orient='horizontal', length=400)
    slider.set(0)
    slider.pack(side='left', padx=6, pady=4)
    frame_label = ttk.Label(ctrl, text=f"1/{frames}")
    frame_label.pack(side='left', padx=6)

    def slider_changed(val):
        i = int(float(val))
        ani.event_source.stop()
        update(i)
        canvas.draw()
        frame_label.config(text=f"{i + 1}/{frames}")
        state['playing'] = False

    slider.config(command=slider_changed)

    def on_play_pause():
        if state['playing']:
            ani.event_source.stop()
            state['playing'] = False
            play_btn.config(text='Play')
        else:
            ani.event_source.start()
            state['playing'] = True
            play_btn.config(text='Pause')

    play_btn = ttk.Button(ctrl, text='Pause', command=on_play_pause)
    play_btn.pack(side='left', padx=4)

    def step_forward():
        i = min(frames - 1, state['idx'] + 1)
        slider.set(i)
        slider_changed(i)

    def step_back():
        i = max(0, state['idx'] - 1)
        slider.set(i)
        slider_changed(i)

    ttk.Button(ctrl, text='◀', command=step_back).pack(side='left', padx=2)
    ttk.Button(ctrl, text='▶', command=step_forward).pack(side='left', padx=2)
    ttk.Button(ctrl, text='Restart', command=lambda: (slider.set(0), slider_changed(0))).pack(side='left', padx=6)

    ttk.Label(ctrl, text='Rychlost (ms):').pack(side='left', padx=(12, 2))
    speed_var = tk.IntVar(value=state['interval'])

    def speed_changed():
        ival = max(10, speed_var.get())
        state['interval'] = ival
        ani.event_source.interval = ival

    speed_spin = ttk.Spinbox(ctrl, from_=10, to=2000, increment=10, textvariable=speed_var, width=6,
                             command=speed_changed)
    speed_spin.pack(side='left')

    view_frame = tk.Frame(root_frame)
    view_frame.pack(side='bottom', fill='x')

    def set_view(elev, azim):
        ax3d.view_init(elev=elev, azim=azim)
        canvas.draw()

    for text, e, a in [('Front', 20, -60), ('Back', 20, 120), ('Top', 90, -90), ('Side', 20, 0)]:
        ttk.Button(view_frame, text=text, command=lambda ee=e, aa=a: set_view(ee, aa)).pack(side='left', padx=4)

    def on_timer(event):
        i = int(state['idx'])
        slider.set(i)
        frame_label.config(text=f"{i + 1}/{frames}")
        root_frame.after(50, lambda: on_timer(None))

    root_frame.after(50, lambda: on_timer(None))

    return ani


# --- Differential Evolution animation ---

def animate_de(root_frame, func_name, lb, ub,
               iterations=200, NP=50, F=0.5, CR=0.9, seed=42,
               grid_res=60):
    """
    Animace Differential Evolution algoritmu
    
    Parametry:
    - root_frame: Tkinter frame pro zobrazení
    - func_name: název funkce (sphere, rastrigin, atd.)
    - lb, ub: dolní a horní hranice
    - iterations: počet generací
    - NP: velikost populace
    - F: mutační konstanta
    - CR: crossover konstanta
    - seed: random seed
    - grid_res: rozlišení gridu pro vykreslení funkce
    """
    func = Function(func_name)
    result = differential_evolution(func, dimension=2, lb=lb, ub=ub, 
                                   NP=NP, F=F, CR=CR, iterations=iterations, seed=seed)
    history = result['history']
    
    # Vytvoření gridu pro vykreslení funkce
    xs = np.linspace(lb, ub, grid_res)
    ys = np.linspace(lb, ub, grid_res)
    X, Y = np.meshgrid(xs, ys)
    vec_eval = np.vectorize(lambda a, b: float(func.eval([a, b])))
    Z = vec_eval(X, Y)
    
    fig = plt.Figure(figsize=(9, 5))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)
    
    # 3D surface a 2D kontura (stejně jako u hill climbing)
    ax3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8, linewidth=0, antialiased=False)
    mesh = ax2d.pcolormesh(X, Y, Z, shading='auto')
    fig.colorbar(mesh, ax=ax2d, shrink=0.6)
    
    ax3d.set_title(f"{func.name.capitalize()} - DE (3D)")
    ax2d.set_title(f"{func.name.capitalize()} - DE (2D kontura)")
    ax2d.set_xlim(lb, ub)
    ax2d.set_ylim(lb, ub)
    ax3d.view_init(elev=30, azim=-60)
    
    # Scatter pro populaci (aktuální generace)
    pop_scatter3d = ax3d.scatter([], [], [], s=30, c='red', alpha=0.6)
    pop_scatter2d = ax2d.scatter([], [], s=30, c='red', alpha=0.6)
    
    # Nejlepší řešení
    best_scatter3d = ax3d.scatter([], [], [], s=140, marker='*', color='gold')
    best_scatter2d = ax2d.scatter([], [], s=140, marker='*', color='gold')
    
    frames = len(history)
    state = {'idx': 0, 'playing': True, 'interval': 150}
    
    def init():
        pop_scatter3d._offsets3d = ([], [], [])
        pop_scatter2d.set_offsets([])
        best_scatter3d._offsets3d = ([], [], [])
        best_scatter2d.set_offsets([])
        return pop_scatter3d, pop_scatter2d, best_scatter3d, best_scatter2d
    
    def update(frame):
        i = int(frame)
        state['idx'] = i
        
        positions, fitness, best_pos, best_fit = history[i]
        
        # Aktualizace populace v 3D a 2D
        pop_scatter3d._offsets3d = (positions[:, 0], positions[:, 1], fitness)
        pop_scatter2d.set_offsets(positions[:, 0:2])
        
        # Nejlepší řešení
        best_scatter3d._offsets3d = ([best_pos[0]], [best_pos[1]], [best_fit])
        best_scatter2d.set_offsets([[best_pos[0], best_pos[1]]])
        
        return pop_scatter3d, pop_scatter2d, best_scatter3d, best_scatter2d
    
    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                                  interval=state['interval'], blit=False)
    ani.event_source.start()
    
    canvas = show_in_tk(fig, root_frame, anim=ani, update_fn=update, total_frames=frames)
    
    # Info panel
    info_frame = tk.Frame(root_frame, relief=tk.RIDGE, borderwidth=2, bg='white')
    info_frame.pack(side="bottom", fill="x", padx=5, pady=(5, 0))
    info_label = tk.Label(info_frame, 
                         text=f'Generace: 0 | Nejlepší fitness: - | Průměrná fitness: - | NP={NP}, F={F}, CR={CR}',
                         font=('Arial', 10, 'bold'), bg='white', fg='black', padx=10, pady=8)
    info_label.pack(side='left')
    
    # Ovládací prvky
    ctrl = tk.Frame(root_frame)
    ctrl.pack(side="bottom", fill="x")
    slider = ttk.Scale(ctrl, from_=0, to=frames - 1, orient='horizontal', length=400)
    slider.set(0)
    slider.pack(side='left', padx=6, pady=4)
    frame_label = ttk.Label(ctrl, text=f"1/{frames}")
    frame_label.pack(side='left', padx=6)
    
    def slider_changed(val):
        i = int(float(val))
        ani.event_source.stop()
        update(i)
        canvas.draw()
        frame_label.config(text=f"{i + 1}/{frames}")
        state['playing'] = False
        # Aktualizace info labelu
        _, fitness, _, best_fit = history[i]
        info_label.config(text=f'Generace: {i} | Nejlepší fitness: {best_fit:.6f} | '
                              f'Průměrná fitness: {np.mean(fitness):.6f} | NP={NP}, F={F}, CR={CR}')
    
    slider.config(command=slider_changed)
    
    def on_play_pause():
        if state['playing']:
            ani.event_source.stop()
            state['playing'] = False
            play_btn.config(text='Play')
        else:
            ani.event_source.start()
            state['playing'] = True
            play_btn.config(text='Pause')
    
    play_btn = ttk.Button(ctrl, text='Pause', command=on_play_pause)
    play_btn.pack(side='left', padx=4)
    
    def step_forward():
        i = min(frames - 1, state['idx'] + 1)
        slider.set(i)
        slider_changed(i)
    
    def step_back():
        i = max(0, state['idx'] - 1)
        slider.set(i)
        slider_changed(i)
    
    ttk.Button(ctrl, text='◀', command=step_back).pack(side='left', padx=2)
    ttk.Button(ctrl, text='▶', command=step_forward).pack(side='left', padx=2)
    ttk.Button(ctrl, text='Restart', command=lambda: (slider.set(0), slider_changed(0))).pack(side='left', padx=6)
    
    ttk.Label(ctrl, text='Rychlost (ms):').pack(side='left', padx=(12, 2))
    speed_var = tk.IntVar(value=state['interval'])
    
    def speed_changed():
        ival = max(10, speed_var.get())
        state['interval'] = ival
        ani.event_source.interval = ival
    
    speed_spin = ttk.Spinbox(ctrl, from_=10, to=2000, increment=10, textvariable=speed_var, width=6,
                             command=speed_changed)
    speed_spin.pack(side='left')
    
    # Tlačítka pro změnu pohledu
    view_frame = tk.Frame(root_frame)
    view_frame.pack(side='bottom', fill='x')
    
    def set_view(elev, azim):
        ax3d.view_init(elev=elev, azim=azim)
        canvas.draw()
    
    for text, e, a in [('Front', 20, -60), ('Back', 20, 120), ('Top', 90, -90), ('Side', 20, 0)]:
        ttk.Button(view_frame, text=text, command=lambda ee=e, aa=a: set_view(ee, aa)).pack(side='left', padx=4)
    
    def on_timer_de(event):
        i = int(state['idx'])
        slider.set(i)
        frame_label.config(text=f"{i + 1}/{frames}")
        # Aktualizace info labelu
        if i < len(history):
            _, fitness, _, best_fit = history[i]
            info_label.config(text=f'Generace: {i} | Nejlepší fitness: {best_fit:.6f} | '
                                  f'Průměrná fitness: {np.mean(fitness):.6f} | NP={NP}, F={F}, CR={CR}')
        root_frame.after(50, lambda: on_timer_de(None))
    
    root_frame.after(50, lambda: on_timer_de(None))
    
    return ani
