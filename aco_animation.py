import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from ant_colony import AntColony


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

def animate_aco(root_frame, n_cities=30, n_ants=20, iterations=100, alpha=1, beta=5, rho=0.5, Q=100, seed=42):
    aco = AntColony(n_cities=n_cities, n_ants=n_ants, iterations=iterations, alpha=alpha, beta=beta, rho=rho, Q=Q, seed=seed)
    history = aco.run()

    fig = plt.Figure(figsize=(12, 6))
    ax_map = fig.add_subplot(121)
    ax_fitness = fig.add_subplot(122)

    ax_map.set_xlim(-5, 105)
    ax_map.set_ylim(-5, 105)
    ax_map.set_title('ACO - Aktuální nejlepší trasa')
    ax_map.set_xlabel('X')
    ax_map.set_ylabel('Y')
    ax_map.grid(True, alpha=0.3)

    cities_x = aco.cities[:, 0]
    cities_y = aco.cities[:, 1]
    ax_map.scatter(cities_x, cities_y, c='red', s=100, zorder=5, marker='o')
    for i in range(n_cities):
        ax_map.text(cities_x[i] + 1, cities_y[i] + 1, str(i), fontsize=8)

    ax_fitness.set_title('Vývoj délky trasy')
    ax_fitness.set_xlabel('Iterace')
    ax_fitness.set_ylabel('Délka trasy')
    ax_fitness.grid(True, alpha=0.3)

    route_line, = ax_map.plot([], [], 'b-', linewidth=2, alpha=0.7)
    best_fitness_line, = ax_fitness.plot([], [], 'g-', linewidth=2, label='Nejlepší')
    avg_fitness_line, = ax_fitness.plot([], [], 'b--', linewidth=1, alpha=0.7, label='Průměr')
    ax_fitness.legend()

    frames = len(history)
    state = {'idx': 0, 'playing': True, 'interval': 100}

    def init():
        route_line.set_data([], [])
        best_fitness_line.set_data([], [])
        avg_fitness_line.set_data([], [])
        return route_line, best_fitness_line, avg_fitness_line

    def update(frame):
        i = int(frame)
        state['idx'] = i

        current = history[i]
        best_route = current['best_route']

        if best_route:
            route_x = [aco.cities[city_idx, 0] for city_idx in best_route]
            route_y = [aco.cities[city_idx, 1] for city_idx in best_route]
            route_x.append(route_x[0])
            route_y.append(route_y[0])
            route_line.set_data(route_x, route_y)

        iterations_data = [h['iteration'] for h in history[:i + 1]]
        best_fits = [h['best_length'] for h in history[:i + 1]]
        avg_fits = [h['avg_length'] for h in history[:i + 1]]

        best_fitness_line.set_data(iterations_data, best_fits)
        avg_fitness_line.set_data(iterations_data, avg_fits)

        if best_fits:
            ax_fitness.set_xlim(0, max(10, max(iterations_data)))
            ax_fitness.set_ylim(min(best_fits) * 0.9, max(avg_fits) * 1.1)

        return route_line, best_fitness_line, avg_fitness_line

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                                  interval=state['interval'], blit=False)
    ani.event_source.start()

    canvas = show_in_tk(fig, root_frame, anim=ani, update_fn=update, total_frames=frames)

    info_frame = tk.Frame(root_frame, relief=tk.RIDGE, borderwidth=2, bg='white')
    info_frame.pack(side="bottom", fill="x", padx=5, pady=(5, 0))
    info_label = tk.Label(info_frame, text='Iterace: 0 | Nejlepší délka: - | Průměrná délka: -',
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
        current = history[i]
        info_label.config(text=f'Iterace: {current["iteration"]} | '
                               f'Nejlepší délka: {current["best_length"]:.2f} | '
                               f'Průměrná délka: {current["avg_length"]:.2f}')

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
        if i < len(history):
            current = history[i]
            info_label.config(text=f'Iterace: {current["iteration"]} | '
                                   f'Nejlepší délka: {current["best_length"]:.2f} | '
                                   f'Průměrná délka: {current["avg_length"]:.2f}')
        root_frame.after(50, lambda: on_timer(None))

    root_frame.after(50, lambda: on_timer(None))

    return ani
