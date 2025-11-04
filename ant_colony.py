import numpy as np


class AntColony:
    def __init__(self, n_cities=30, n_ants=20, iterations=100, alpha=1.0, beta=5.0, rho=0.5, Q=100.0, seed=None):
        self.n_cities = int(n_cities)
        self.n_ants = int(n_ants)
        self.iterations = int(iterations)
        self.alpha = float(alpha)  # importance of pheromone
        self.beta = float(beta)    # importance of heuristic (1/distance)
        self.rho = float(rho)      # evaporation rate
        self.Q = float(Q)          # pheromone deposit constant
        if seed is not None:
            np.random.seed(int(seed))

        # Random cities in 0..100 x 0..100
        self.cities = np.random.rand(self.n_cities, 2) * 100.0
        # Distance matrix
        self.distances = self._compute_distance_matrix(self.cities)
        # Initial pheromone matrix (small positive value)
        self.pheromones = np.ones((self.n_cities, self.n_cities), dtype=float)
        np.fill_diagonal(self.pheromones, 0.0)

        # Heuristic information (1/d), avoid div by zero
        with np.errstate(divide='ignore'):
            self.eta = 1.0 / np.where(self.distances > 0, self.distances, np.inf)
        np.fill_diagonal(self.eta, 0.0)

    @staticmethod
    def _compute_distance_matrix(cities):
        n = cities.shape[0]
        diff = cities.reshape(n, 1, 2) - cities.reshape(1, n, 2)
        return np.sqrt((diff ** 2).sum(axis=2))

    def route_length(self, route):
        """Compute total length of a route (closed tour)."""
        r = np.asarray(route, dtype=int)
        return float(sum(self.distances[r[i], r[(i + 1) % len(r)]] for i in range(len(r))))

    def construct_route(self):
        """Build a route for one ant using probabilistic transition rule."""
        n = self.n_cities
        start = np.random.randint(n)
        route = [start]
        unvisited = set(range(n))
        unvisited.remove(start)

        current = start
        while unvisited:
            candidates = np.array(list(unvisited), dtype=int)
            tau = self.pheromones[current, candidates] ** self.alpha
            eta = self.eta[current, candidates] ** self.beta
            probs_unnorm = tau * eta
            if not np.isfinite(probs_unnorm).any() or probs_unnorm.sum() == 0:
                # fallback to uniform choice
                next_city = int(np.random.choice(candidates))
            else:
                p = probs_unnorm / probs_unnorm.sum()
                next_city = int(np.random.choice(candidates, p=p))
            route.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        return route

    def update_pheromones(self, routes, lengths):
        n = self.n_cities
        # Evaporation
        self.pheromones *= (1.0 - self.rho)
        # Deposit
        for route, L in zip(routes, lengths):
            if L <= 0:
                continue
            deposit = self.Q / L
            for i in range(n):
                a = route[i]
                b = route[(i + 1) % n]
                self.pheromones[a, b] += deposit
                self.pheromones[b, a] += deposit
        # Avoid pheromone going to zero exactly
        self.pheromones = np.clip(self.pheromones, 1e-12, None)
        np.fill_diagonal(self.pheromones, 0.0)

    def run(self):
        """Execute ACO and return history for animation."""
        best_length = np.inf
        best_route = None
        history = []

        for iteration in range(self.iterations):
            all_routes = []
            all_lengths = []

            for _ in range(self.n_ants):
                route = self.construct_route()
                length = self.route_length(route)
                all_routes.append(route)
                all_lengths.append(length)
                if length < best_length:
                    best_length = float(length)
                    best_route = list(route)

            self.update_pheromones(all_routes, all_lengths)

            avg_len = float(np.mean(all_lengths)) if all_lengths else float('inf')
            history.append({
                'iteration': iteration + 1,
                'best_route': best_route,
                'best_length': best_length,
                'avg_length': avg_len,
            })

        return history
