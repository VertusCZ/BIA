import numpy as np


class TSP:
    def __init__(self, n_cities=30, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_cities = n_cities
        # Random city positions in [0,100]x[0,100]
        self.cities = np.random.rand(n_cities, 2) * 100

    def distance(self, city1_idx, city2_idx):
        c1 = self.cities[city1_idx]
        c2 = self.cities[city2_idx]
        return np.sqrt(np.sum((c1 - c2) ** 2))

    def route_length(self, route):
        length = 0
        for i in range(len(route)):
            length += self.distance(route[i], route[(i + 1) % len(route)])
        return length


def genetic_algorithm_tsp(tsp: 'TSP',
                          population_size=20,
                          generations=200,
                          mutation_prob=0.5,
                          seed=None):
    if seed is not None:
        np.random.seed(seed)

    n_cities = tsp.n_cities

    # Initialize population
    population = []
    for _ in range(population_size):
        individual = np.random.permutation(n_cities)
        fitness = tsp.route_length(individual)
        population.append((individual.copy(), fitness))

    # History for visualization
    history = []
    best_individual = min(population, key=lambda x: x[1])
    history.append({
        'generation': 0,
        'population': [ind.copy() for ind, _ in population],
        'fitness': [fit for _, fit in population],
        'best': best_individual[0].copy(),
        'best_fitness': best_individual[1]
    })

    # Evolution
    for gen in range(generations):
        new_population = []

        for j in range(population_size):
            parent_A, fitness_A = population[j]

            parent_B_idx = np.random.choice([i for i in range(population_size) if i != j])
            parent_B, _ = population[parent_B_idx]

            crossover_point = np.random.randint(1, n_cities - 1)
            offspring = np.zeros(n_cities, dtype=int)
            offspring[:crossover_point] = parent_A[:crossover_point]

            remaining = [city for city in parent_B if city not in offspring[:crossover_point]]
            offspring[crossover_point:] = remaining

            if np.random.rand() < mutation_prob:
                idx1, idx2 = np.random.choice(n_cities, size=2, replace=False)
                offspring[idx1], offspring[idx2] = offspring[idx2], offspring[idx1]

            offspring_fitness = tsp.route_length(offspring)

            if offspring_fitness <= fitness_A:
                new_population.append((offspring.copy(), offspring_fitness))
            else:
                new_population.append((parent_A.copy(), fitness_A))

        population = new_population

        best_individual = min(population, key=lambda x: x[1])
        history.append({
            'generation': gen + 1,
            'population': [ind.copy() for ind, _ in population],
            'fitness': [fit for _, fit in population],
            'best': best_individual[0].copy(),
            'best_fitness': best_individual[1]
        })

    return history
