import math
import numpy as np
from copy import deepcopy


class Function:
    """
    Benchmark objective function dispatcher.
    """
    def __init__(self, name='sphere'):
        self.name = name.lower()

    def sphere(self, x):
        return np.sum(np.asarray(x) ** 2)

    def ackley(self, x, a=20, b=0.2, c=2 * math.pi):
        x = np.asarray(x)
        d = x.size
        return (-a * math.exp(-b * math.sqrt(np.sum(x ** 2) / d))
                - math.exp(np.sum(np.cos(c * x)) / d)
                + a + math.e)

    def rastrigin(self, x, A=10):
        x = np.asarray(x)
        d = x.size
        return A * d + np.sum(x ** 2 - A * np.cos(2 * math.pi * x))

    def rosenbrock(self, x):
        x = np.asarray(x)
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)

    def griewank(self, x):
        x = np.asarray(x)
        sum_term = np.sum(x ** 2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, x.size + 1))))
        return sum_term - prod_term + 1

    def schwefel(self, x):
        x = np.asarray(x)
        d = x.size
        return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def levy(self, x):
        x = np.asarray(x)
        w = 1 + (x - 1) / 4
        term1 = math.sin(math.pi * w[0]) ** 2
        term3 = (w[-1] - 1) ** 2 * (1 + math.sin(2 * math.pi * w[-1]) ** 2)
        term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(math.pi * w[:-1] + 1) ** 2))
        return term1 + term2 + term3

    def michalewicz(self, x, m=10):
        x = np.asarray(x)
        i = np.arange(1, x.size + 1)
        return -np.sum(np.sin(x) * (np.sin(i * x ** 2 / math.pi)) ** (2 * m))

    def zakharov(self, x):
        x = np.asarray(x)
        i = np.arange(1, x.size + 1)
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(0.5 * i * x)
        return sum1 + sum2 ** 2 + sum2 ** 4

    def eval(self, x):
        return getattr(self, self.name)(x)


# --- Algorithms ---

def blind_search(func: Function, iterations=200, lb=-5, ub=5, seed=0):
    if seed:
        np.random.seed(seed)
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


def hill_climbing(func: Function,
                  dimension=2,
                  lb=-5.0, ub=5.0,
                  iterations=300,
                  sigma=0.3,
                  k_neighbors=5,
                  seed=None):
    if seed is not None:
        np.random.seed(seed)
    current = np.random.uniform(lb, ub, size=dimension)
    current_f = float(func.eval(current))
    best = current.copy()
    best_f = current_f
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


def simulated_annealing(func: Function,
                        dimension=2,
                        lb=-5.0, ub=5.0,
                        iterations=300,
                        T0=100, Tmin=0.5, alpha=0.95,
                        sigma=0.3,
                        seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Inicializace - vytvoření náhodného počátečního bodu.
    current = np.random.uniform(lb, ub, size=dimension)
    current_f = float(func.eval(current))
    best = current.copy()
    best_f = current_f
    history = [(current.copy(), current_f)]

    # Nastavení počáteční "teploty", která ovlivňuje pravděpodobnost přijetí horšího řešení.
    T = T0
    for _ in range(iterations):
        # Krok 2: Vytvoření nového kandidátského řešení v okolí aktuálního.
        cand = current + np.random.normal(0, sigma, size=dimension)
        cand = np.clip(cand, lb, ub)
        fval = float(func.eval(cand))

        # Pokud je nové řešení lepší, přijme se vždy.
        if fval < current_f:
            current, current_f = cand, fval
        else:
            # Pokud je horší, přijme se s určitou pravděpodobností, která klesá s "teplotou".
            # To umožňuje algoritmu uniknout z lokálních minim.
            r = np.random.rand()
            if r < np.exp(-(fval - current_f) / T):
                current, current_f = cand, fval

        # Aktualizace nejlepšího nalezeného řešení.
        if current_f < best_f:
            best, best_f = current.copy(), current_f

        history.append((current.copy(), current_f))

        # snižování teploty.
        T = max(T * alpha, Tmin)

    return {'best': best, 'best_f': best_f, 'history': history}


# --- Differential Evolution ---

class Solution:
    """Třída pro reprezentaci jedince (řešení)"""
    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.params = np.random.uniform(lower_bound, upper_bound, dimension)
        self.f = float('inf')  # fitness value
    
    def evaluate(self, fitness_function):
        """Vyhodnocení fitness funkce"""
        self.f = fitness_function(self.params)
        return self.f


def differential_evolution(func: Function,
                          dimension=2,
                          lb=-5.0, ub=5.0,
                          NP=50, F=0.5, CR=0.9,
                          iterations=300,
                          seed=None):
    """
    Differential Evolution algoritmus
    
    Parametry:
    - func: Function objekt k optimalizaci
    - dimension: dimenze problému
    - lb: dolní hranice proměnných
    - ub: horní hranice proměnných
    - NP: velikost populace
    - F: mutační konstanta (0 < F <= 2)
    - CR: crossover konstanta (0 <= CR <= 1)
    - iterations: maximální počet generací
    - seed: random seed
    """
    if seed is not None:
        np.random.seed(seed)
    
    def check_boundaries(params):
        """Kontrola a oprava hranic"""
        return np.clip(params, lb, ub)
    
    # Generování počáteční populace
    pop = [Solution(dimension, lb, ub) for _ in range(NP)]
    
    # Vyhodnocení počáteční populace
    for sol in pop:
        sol.evaluate(lambda x: func.eval(x))
    
    best_solution = None
    history = []
    
    g = 0
    while g < iterations:
        # Uložení aktuálního stavu pro animaci
        current_positions = np.array([sol.params for sol in pop])
        current_fitness = np.array([sol.f for sol in pop])
        best_idx = np.argmin(current_fitness)
        
        if best_solution is None or current_fitness[best_idx] < best_solution.f:
            best_solution = deepcopy(pop[best_idx])
        
        # Pro 2D vizualizaci ukládáme pouze první 2 dimenze
        if dimension == 2:
            history.append((current_positions.copy(), current_fitness.copy(), 
                          best_solution.params.copy(), best_solution.f))
        else:
            # Pro vyšší dimenze ukládáme jen nejlepší řešení
            history.append((current_positions[:, :2].copy(), current_fitness.copy(),
                          best_solution.params[:2].copy(), best_solution.f))
        
        new_pop = deepcopy(pop)
        
        for i, x in enumerate(pop):  # x je target vector
            # Výběr tří náhodných indexů odlišných od i
            indices = [idx for idx in range(NP) if idx != i]
            r1, r2, r3 = np.random.choice(indices, 3, replace=False)
            
            # MUTACE: v = (x_r1 - x_r2) * F + x_r3
            v = (pop[r1].params - pop[r2].params) * F + pop[r3].params
            v = check_boundaries(v)  # KONTROLA HRANIC
            
            # KŘÍŽENÍ (Crossover)
            u = np.zeros(dimension)  # trial vector
            j_rnd = np.random.randint(0, dimension)
            
            for j in range(dimension):
                if np.random.uniform() < CR or j == j_rnd:
                    u[j] = v[j]  # alespoň 1 parametr z mutačního vektoru
                else:
                    u[j] = x.params[j]
            
            # Vyhodnocení trial vektoru
            f_u = func.eval(u)
            
            # SELEKCE: porovnání s target vektorem
            if f_u <= x.f:  # lepší nebo stejné řešení
                new_x = Solution(dimension, lb, ub)
                new_x.params = u
                new_x.f = f_u
                new_pop[i] = new_x
        
        pop = new_pop
        g += 1
    
    # Finální záznam
    current_positions = np.array([sol.params for sol in pop])
    current_fitness = np.array([sol.f for sol in pop])
    best_idx = np.argmin(current_fitness)
    if current_fitness[best_idx] < best_solution.f:
        best_solution = deepcopy(pop[best_idx])
    
    return {'best': best_solution.params, 'best_f': best_solution.f, 'history': history}
