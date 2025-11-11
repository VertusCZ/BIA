import numpy as np


class FireflyAlgorithm:
    """
    Implementace Firefly Algorithm (Algoritmus světlušek)
    """
    def __init__(self, n_fireflies=25, max_iterations=150, alpha=0.5,
                 beta0=1.0, gamma=1.0, bounds=None, seed=None):
        """
        Parametry:
        - n_fireflies: počet světlušek (populace)
        - max_iterations: maximální počet iterací
        - alpha: parametr náhodnosti (0.2-0.5)
        - beta0: přitažlivost při r=0 (0.5-2.0)
        - gamma: koeficient absorpce světla (0.01-100)
        - bounds: hranice prohledávaného prostoru [(min, max), ...]
        """
        self.n_fireflies = int(n_fireflies)
        self.max_iterations = int(max_iterations)
        self.alpha_initial = float(alpha)
        self.beta0 = float(beta0)
        self.gamma = float(gamma)
        self.bounds = bounds  # list of (min, max) for each dim
        self.seed = None if seed is None else int(seed)

        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []
        self.population_history = []

        if self.seed is not None:
            np.random.seed(self.seed)

    def _initialize_population(self, dim):
        lb = np.array([b[0] for b in self.bounds], dtype=float)
        ub = np.array([b[1] for b in self.bounds], dtype=float)
        pop = lb + (ub - lb) * np.random.rand(self.n_fireflies, dim)
        return pop

    @staticmethod
    def _distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _attractiveness(self, r):
        # β(r) = β₀ · e^(-γr²)
        return self.beta0 * np.exp(-self.gamma * (r ** 2))

    def _move(self, xi, xj, alpha):
        # Pohyb světlušky i směrem ke světlušce j + náhodná složka
        r = self._distance(xi, xj)
        beta = self._attractiveness(r)
        random_component = alpha * (np.random.rand(xi.size) - 0.5)
        new_position = xi + beta * (xj - xi) + random_component
        # Ošetření hranic
        for d in range(new_position.size):
            new_position[d] = np.clip(new_position[d], self.bounds[d][0], self.bounds[d][1])
        return new_position

    def optimize(self, fitness_function, dim=2, save_history=False):
        """
        Hlavní optimalizační smyčka. Vrací dict s:
        - history: list of tuples (positions, fitness, best_pos, best_fit)
        - best_pos, best_fit
        """
        # Inicializace populace
        population = self._initialize_population(dim)
        fitness = np.array([fitness_function(ind) for ind in population], dtype=float)

        # Uložení nejlepšího řešení
        best_idx = int(np.argmin(fitness))
        self.best_solution = population[best_idx].copy()
        self.best_fitness = float(fitness[best_idx])

        frames = []
        self.history = []
        if save_history:
            self.population_history = []
        else:
            self.population_history = []  # keep empty unless requested

        for iteration in range(self.max_iterations):
            # Adaptivní snižování alpha
            alpha = self.alpha_initial * (0.97 ** iteration)

            # Seřazení podle fitness (vzestupně)
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]

            # Pohyb světlušek
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if fitness[j] < fitness[i]:
                        population[i] = self._move(population[i], population[j], alpha)
                        fitness[i] = float(fitness_function(population[i]))
                        if fitness[i] < self.best_fitness:
                            self.best_fitness = fitness[i]
                            self.best_solution = population[i].copy()

            # Každých 20 iterací restart nejhorších 5 světlušek (mimo první iteraci)
            if iteration % 20 == 0 and iteration > 0 and self.n_fireflies >= 5:
                worst_indices = np.argsort(fitness)[-5:]
                for idx in worst_indices:
                    for d in range(dim):
                        population[idx, d] = np.random.uniform(self.bounds[d][0], self.bounds[d][1])
                    fitness[idx] = float(fitness_function(population[idx]))

            # Uložení historie
            self.history.append(self.best_fitness)
            if save_history:
                self.population_history.append(population.copy())

            # Snímek pro animaci
            frames.append((population.copy(), fitness.copy(), self.best_solution.copy(), float(self.best_fitness)))

            # Průběžný výpis každých 10 iterací
            if (iteration + 1) % 10 == 0:
                avg_fit = float(np.mean(fitness)) if fitness.size else float('inf')
                print(f"Iterace {iteration + 1}/{self.max_iterations}, Nejlepší fitness: {self.best_fitness:.6f}, Průměrná fitness: {avg_fit:.6f}")

        return {
            'history': frames,
            'best_pos': self.best_solution.copy(),
            'best_fit': float(self.best_fitness),
        }
