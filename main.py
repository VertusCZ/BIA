import tkinter as tk
from tkinter import ttk
import numpy as np
import math
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# --------------------------
# Definice benchmark funkcí
# --------------------------
class Function:
    def __init__(self, name='sphere'):
        self.name = name.lower()

    # Jednoduchá kvadratická funkce (koule)
    def sphere(self, x): return np.sum(np.asarray(x)**2)

    # Ackleyho funkce (vícečetné lokální minima)
    def ackley(self, x, a=20, b=0.2, c=2*math.pi):
        x = np.asarray(x); d = x.size
        return (-a * math.exp(-b * math.sqrt(np.sum(x**2) / d))
                - math.exp(np.sum(np.cos(c*x)) / d)
                + a + math.e)

    # Rastriginova funkce (silně multimodální)
    def rastrigin(self, x, A=10):
        x = np.asarray(x); d = x.size
        return A*d + np.sum(x**2 - A*np.cos(2*math.pi*x))

    # Rosenbrockova funkce (banánové údolí)
    def rosenbrock(self, x):
        x = np.asarray(x)
        return np.sum(100*(x[1:]-x[:-1]**2)**2 + (x[:-1]-1)**2)

    def griewank(self, x):
        x = np.asarray(x)
        sum_term = np.sum(x**2)/4000
        prod_term = np.prod(np.cos(x/np.sqrt(np.arange(1, x.size+1))))
        return sum_term - prod_term + 1

    def schwefel(self, x):
        x = np.asarray(x); d = x.size
        return 418.9829*d - np.sum(x*np.sin(np.sqrt(np.abs(x))))

    def levy(self, x):
        x = np.asarray(x)
        w = 1 + (x-1)/4
        term1 = math.sin(math.pi*w[0])**2
        term3 = (w[-1]-1)**2*(1+math.sin(2*math.pi*w[-1])**2)
        term2 = np.sum((w[:-1]-1)**2 * (1+10*np.sin(math.pi*w[:-1]+1)**2))
        return term1+term2+term3

    def michalewicz(self, x, m=10):
        x = np.asarray(x)
        i = np.arange(1, x.size+1)
        return -np.sum(np.sin(x)*(np.sin(i*x**2/math.pi))**(2*m))

    def zakharov(self, x):
        x = np.asarray(x); i = np.arange(1, x.size+1)
        sum1 = np.sum(x**2)
        sum2 = np.sum(0.5*i*x)
        return sum1 + sum2**2 + sum2**4

    def eval(self, x): return getattr(self, self.name)(x)


# --------------------------
# Implementace algoritmů
# --------------------------
# Náhodné vyhledávání (Blind Search)
def blind_search(func: Function, iterations=200, lb=-5, ub=5, seed=0):
    if seed: np.random.seed(seed)
    best = np.random.uniform(lb, ub, size=2)
    best_f = float(func.eval(best))
    history = [(best.copy(), best_f)]
    for _ in range(iterations):
        cand = np.random.uniform(lb, ub, size=2)
        fval = float(func.eval(cand))
        if fval < best_f:
            best, best_f = cand, fval
        history.append((cand.copy(), fval))
    return best, history

# Hill Climbing (lokální prohledávání sousedů)
def hill_climbing(func: Function,
                  dimension=2,
                  lb=-5.0, ub=5.0,
                  iterations=300,
                  sigma=0.3,
                  k_neighbors=5,
                  seed=None):
    if seed is not None: np.random.seed(seed)
    current = np.random.uniform(lb, ub, size=dimension)
    current_f = float(func.eval(current))
    best = current.copy(); best_f = current_f
    history = [(current.copy(), current_f)]
    for _ in range(iterations):
        neighbors = []
        for _ in range(max(1, int(k_neighbors))):
            cand = current + np.random.normal(0, sigma, size=dimension)
            cand = np.clip(cand, lb, ub)
            fval = float(func.eval(cand))
            neighbors.append((cand, fval))
        neighbors.sort(key=lambda t: t[1])
        cand, fval = neighbors[0]
        if fval < current_f:
            current, current_f = cand, fval
        if current_f < best_f:
            best, best_f = current.copy(), current_f
        history.append((current.copy(), current_f))
    return {'best': best, 'best_f': best_f, 'history': history}


# --------------------------
# Pomocné funkce pro Tkinter
# --------------------------

def show_in_tk(fig, frame, anim=None, update_fn=None, total_frames=None):
    # Vyčistí starý obsah a vloží graf do Tkinter okna
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


# --------------------------
# Animace algoritmů
# --------------------------

def animate_algorithm(root_frame, func_name, lb, ub, algo="blind", iterations=200, seed=42):
    # Získání historie zvoleného algoritmu
    func = Function(func_name)
    if algo=="blind":
        _, history = blind_search(func, iterations=iterations, lb=lb, ub=ub, seed=seed)
    else:
        history = hill_climbing(func, lb=lb, ub=ub, iterations=iterations, sigma=0.4, k_neighbors=10, seed=seed)['history']

    # Vytvoření mřížky pro vykreslení povrchu
    grid_res = 140
    xs = np.linspace(lb, ub, grid_res); ys = np.linspace(lb, ub, grid_res)
    X,Y = np.meshgrid(xs, ys)
    vec_eval = np.vectorize(lambda a,b: float(func.eval([a,b])))
    Z = vec_eval(X, Y)

    # Vytvoření figure: levý subplot 3D povrch, pravý subplot kontury
    fig = plt.Figure(figsize=(9,5))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    ax3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7, linewidth=0, antialiased=True)
    cset = ax2d.contourf(X, Y, Z, levels=60, cmap="viridis")
    fig.colorbar(cset, ax=ax2d, shrink=0.6)

    ax3d.set_title(f"{func.name.capitalize()} (3D)")
    ax2d.set_title(f"{func.name.capitalize()} (2D kontura)")
    ax2d.set_xlim(lb, ub); ax2d.set_ylim(lb, ub)

    # Připravení historie pozic a hodnot
    positions = np.array([h[0] for h in history])
    values = np.array([h[1] for h in history])
    frames = positions.shape[0]

    # Scatter body: průhledná stopa, aktuální bod, nejlepší bod
    trail3d = ax3d.scatter([], [], [], s=20, c='red', alpha=0.3)
    trail2d = ax2d.scatter([], [], s=20, c='red', alpha=0.3)
    scatter3d = ax3d.scatter([], [], [], s=60, c='red')
    scatter2d = ax2d.scatter([], [], s=60, c='red')
    best3d = ax3d.scatter([], [], [], s=120, marker='*', color='gold')
    best2d = ax2d.scatter([], [], s=120, marker='*', color='gold')

    ax3d.view_init(elev=30, azim=-60)

    state = {'idx':0, 'playing':True, 'interval':100}

    # Inicializace prázdné animace
    def init():
        trail3d._offsets3d = ([], [], [])
        trail2d.set_offsets([])
        scatter3d._offsets3d = ([], [], [])
        scatter2d.set_offsets([])
        best3d._offsets3d = ([], [], [])
        best2d.set_offsets([])
        return trail3d, trail2d, scatter3d, scatter2d, best3d, best2d

    # Aktualizace pro každý frame animace
    def update(frame):
        i = int(frame)
        state['idx'] = i

        # Stopová trajektorie do aktuálního bodu
        pts = positions[:i+1]
        zs = values[:i+1]
        trail3d._offsets3d = (pts[:,0], pts[:,1], zs)
        trail2d.set_offsets(pts[:,0:2])

        # Aktuální bod
        cx, cy = positions[i]
        cz = values[i]
        scatter3d._offsets3d = ([cx], [cy], [cz])
        scatter2d.set_offsets([[cx, cy]])

        # Nejlepší nalezený bod
        best_idx = np.argmin(values[:i+1])
        bx, by = positions[best_idx]
        bz = values[best_idx]
        best3d._offsets3d = ([bx], [by], [bz])
        best2d.set_offsets([[bx, by]])

        return trail3d, trail2d, scatter3d, scatter2d, best3d, best2d

    # Spuštění animace
    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                                  interval=state['interval'], blit=False)
    ani.event_source.start()

    # Vložení figure do tkinter frame
    canvas = show_in_tk(fig, root_frame, anim=ani, update_fn=update, total_frames=frames)

    # --------------------------
    # Ovládací panel pod grafikou
    # --------------------------
    ctrl = tk.Frame(root_frame); ctrl.pack(side="bottom", fill="x")

    # Posuvník pro přeskakování snímků
    slider = ttk.Scale(ctrl, from_=0, to=frames-1, orient='horizontal', length=400)
    slider.set(0)
    slider.pack(side='left', padx=6, pady=4)

    frame_label = ttk.Label(ctrl, text=f"1/{frames}")
    frame_label.pack(side='left', padx=6)

    def slider_changed(val):
        i = int(float(val))
        ani.event_source.stop()
        update(i)
        canvas.draw()
        frame_label.config(text=f"{i+1}/{frames}")
        state['playing'] = False
    slider.config(command=slider_changed)

    # Tlačítka Play/Pause a krokování
    def on_play_pause():
        if state['playing']:
            ani.event_source.stop(); state['playing']=False; play_btn.config(text='Play')
        else:
            ani.event_source.start(); state['playing']=True; play_btn.config(text='Pause')
    play_btn = ttk.Button(ctrl, text='Pause', command=on_play_pause)
    play_btn.pack(side='left', padx=4)

    def step_forward():
        i = min(frames-1, state['idx']+1)
        slider.set(i); slider_changed(i)
    def step_back():
        i = max(0, state['idx']-1)
        slider.set(i); slider_changed(i)
    ttk.Button(ctrl, text='◀', command=step_back).pack(side='left', padx=2)
    ttk.Button(ctrl, text='▶', command=step_forward).pack(side='left', padx=2)

    def restart():
        slider.set(0); slider_changed(0)
    ttk.Button(ctrl, text='Restart', command=restart).pack(side='left', padx=6)

    # Nastavení rychlosti animace
    ttk.Label(ctrl, text='Rychlost (ms):').pack(side='left', padx=(12,2))
    speed_var = tk.IntVar(value=state['interval'])
    def speed_changed():
        ival = max(10, speed_var.get())
        state['interval'] = ival
        ani.event_source.interval = ival
    speed_spin = ttk.Spinbox(ctrl, from_=10, to=2000, increment=10, textvariable=speed_var, width=6, command=speed_changed)
    speed_spin.pack(side='left')

    # Ovládání pohledu (3D kamera)
    view_frame = tk.Frame(root_frame); view_frame.pack(side='bottom', fill='x')
    def set_view(elev, azim):
        ax3d.view_init(elev=elev, azim=azim); canvas.draw()
    for text, e,a in [('Front',20,-60), ('Back',20,120), ('Top',90,-90), ('Side',20,0)]:
        ttk.Button(view_frame, text=text, command=lambda ee=e,aa=a: set_view(ee,aa)).pack(side='left', padx=4)

    # Synchronizace slideru při běhu animace
    def on_timer(event):
        i = int(state['idx'])
        slider.set(i)
        frame_label.config(text=f"{i+1}/{frames}")
        root_frame.after(50, lambda: on_timer(None))
    root_frame.after(50, lambda: on_timer(None))

    return ani


# --------------------------
# Vykreslení pouze povrchu (bez animace)
# --------------------------
def plot_function(func: Function, lb=-5, ub=5, res=200):
    xs = np.linspace(lb, ub, res); ys = np.linspace(lb, ub, res)
    X, Y = np.meshgrid(xs, ys)
    vec_eval = np.vectorize(lambda a,b: float(func.eval([a,b])))
    Z = vec_eval(X, Y)
    fig = plt.Figure(figsize=(6,5)); ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)
    ax.set_title(func.name.capitalize())
    return fig


# --------------------------
# Hlavní GUI aplikace
# --------------------------
def main():
    root = tk.Tk()
    root.title("Function Visualization App — české komentáře")
    root.geometry("1300x760")

    # Levý panel: menu funkcí
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

    # Pravý panel: vykreslovací oblast
    plot_frame = tk.Frame(root)
    plot_frame.pack(side="right", fill="both", expand=True)

    functions = [
        ("Sphere","sphere",-5,5),
        ("Ackley","ackley",-5,5),
        ("Rastrigin","rastrigin",-5.12,5.12),
        ("Rosenbrock","rosenbrock",-3,3),
        ("Schwefel","schwefel",-500,500),
        ("Levy","levy",-10,10),
        ("Michalewicz","michalewicz",0,math.pi),
        ("Zakharov","zakharov",-5,5)
    ]

    for label,fname,lb,ub in functions:
        ttk.Label(scrollable_frame, text=label, font=("Arial",10,"bold")).pack(pady=(8,2))
        ttk.Button(scrollable_frame, text=f"Plot {label}",
                   command=lambda f=fname,lo=lb,up=ub: show_in_tk(plot_function(Function(f), lb=lo, ub=up), plot_frame)).pack(padx=5, pady=2, fill="x")
        ttk.Button(scrollable_frame, text=f"Blind Search {label}",
                   command=lambda f=fname,lo=lb,up=ub: animate_algorithm(plot_frame, f, lo, up, algo="blind", iterations=300, seed=42)).pack(padx=5, pady=2, fill="x")
        ttk.Button(scrollable_frame, text=f"Hill Climbing {label}",
                   command=lambda f=fname,lo=lb,up=ub: animate_algorithm(plot_frame, f, lo, up, algo="hill", iterations=300, seed=42)).pack(padx=5, pady=2, fill="x")

    root.mainloop()


if __name__=="__main__":
    main()
