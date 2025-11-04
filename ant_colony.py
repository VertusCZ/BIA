import numpy as np


class AntColony:
    """
    Jednoduchá implementace kolonie mravenců (ACO) pro TSP.
    Důležité parametry:
    - n_cities: počet měst (uzlů) v instanci TSP
    - n_ants: počet mravenců spuštěných v každé iteraci
    - iterations: počet iterací algoritmu
    - alpha: důležitost feromonové stopy (váha matice feromonů)
    - beta: důležitost heuristiky (zpravidla 1/vzdálenost)
    - rho: míra evaporace (odpařování) feromonů v rozsahu 0..1
    - Q: konstanta množství ukládaného feromonu (škáluje vklad 1/L)
    - seed: volitelný seed pro reprodukovatelnost
    """
    def __init__(self, n_cities=30, n_ants=20, iterations=100, alpha=1.0, beta=5.0, rho=0.5, Q=100.0, seed=None):
        # Přetypování a uložení hyperparametrů
        self.n_cities = int(n_cities)
        self.n_ants = int(n_ants)
        self.iterations = int(iterations)
        self.alpha = float(alpha)  # důležitost feromonů v přechodovém pravidle
        self.beta = float(beta)    # důležitost heuristiky (1/vzdálenost) v přechodovém pravidle
        self.rho = float(rho)      # míra odpařování feromonů
        self.Q = float(Q)          # konstanta pro ukládání feromonů
        if seed is not None:
            np.random.seed(int(seed))  # nastavení náhodného seedu pro reprodukovatelnost

        # Náhodná pozice měst v oblasti 0..100 x 0..100
        self.cities = np.random.rand(self.n_cities, 2) * 100.0
        # Matice vzdáleností mezi všemi páry měst
        self.distances = self._compute_distance_matrix(self.cities)
        # Počáteční matice feromonů (kladné hodnoty, nuly na diagonále)
        self.pheromones = np.ones((self.n_cities, self.n_cities), dtype=float)
        np.fill_diagonal(self.pheromones, 0.0)

        # Heuristická informace (eta = 1/d), pozor na dělení nulou
        with np.errstate(divide='ignore'):
            self.eta = 1.0 / np.where(self.distances > 0, self.distances, np.inf)
        np.fill_diagonal(self.eta, 0.0)

    @staticmethod
    def _compute_distance_matrix(cities):
        # Výpočet Eukleidovských vzdáleností pro všechny dvojice měst
        n = cities.shape[0]
        diff = cities.reshape(n, 1, 2) - cities.reshape(1, n, 2)
        return np.sqrt((diff ** 2).sum(axis=2))

    def route_length(self, route):
        """Spočítá celkovou délku okruhu (uzavřená trasa přes všechna města)."""
        r = np.asarray(route, dtype=int)
        return float(sum(self.distances[r[i], r[(i + 1) % len(r)]] for i in range(len(r))))

    def construct_route(self):
        """
        Postaví trasu pro jednoho mravence pomocí pravděpodobnostního přechodového pravidla:
        p(i->j) ~ (tau_ij^alpha) * (eta_ij^beta), kde
        - tau jsou feromony, eta je heuristika (1/d)
        """
        n = self.n_cities
        start = np.random.randint(n)  # náhodný startovní uzel
        route = [start]
        unvisited = set(range(n))
        unvisited.remove(start)

        current = start
        while unvisited:
            candidates = np.array(list(unvisited), dtype=int)  # dosud nenavštívené uzly
            # Feromony a heuristika na hranách z aktuálního uzlu do kandidátů
            tau = self.pheromones[current, candidates] ** self.alpha
            eta = self.eta[current, candidates] ** self.beta
            probs_unnorm = tau * eta  # ne-normalizované pravděpodobnosti
            if not np.isfinite(probs_unnorm).any() or probs_unnorm.sum() == 0:
                # Nouzový režim: pokud se něco pokazí (NaN/Inf/součet 0), zvol uniformně
                next_city = int(np.random.choice(candidates))
            else:
                p = probs_unnorm / probs_unnorm.sum()  # normalizace na pravděpodobnosti
                next_city = int(np.random.choice(candidates, p=p))
            route.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        return route

    def update_pheromones(self, routes, lengths):
        # Aktualizace feromonové matice: nejdříve odpaření, pak vklad podle kvality tras
        n = self.n_cities
        # Odpařování (snižuje feromony a pomáhá vyhnout se stagnaci)
        self.pheromones *= (1.0 - self.rho)
        # Ukládání feromonů podél hran navštívených v každé trase, úměrně 1/délce
        for route, L in zip(routes, lengths):
            if L <= 0:
                continue
            deposit = self.Q / L
            for i in range(n):
                a = route[i]
                b = route[(i + 1) % n]
                # Ukládáme na obě orientace, protože graf je neorientovaný
                self.pheromones[a, b] += deposit
                self.pheromones[b, a] += deposit
        # Zabránit přesnému vynulování (numerická stabilita) a nulám na diagonále
        self.pheromones = np.clip(self.pheromones, 1e-12, None)
        np.fill_diagonal(self.pheromones, 0.0)

    def run(self):
        """Spusť ACO a vrať historii pro animaci (best/avg délky po iteracích)."""
        best_length = np.inf
        best_route = None
        history = []

        for iteration in range(self.iterations):
            all_routes = []
            all_lengths = []

            # Každý mravenec nezávisle zkonstruuje trasu a změří její délku
            for _ in range(self.n_ants):
                route = self.construct_route()
                length = self.route_length(route)
                all_routes.append(route)
                all_lengths.append(length)
                # Sledování aktuálně nejlepší trasy
                if length < best_length:
                    best_length = float(length)
                    best_route = list(route)

            # Globální aktualizace feromonů po celé populaci mravenců
            self.update_pheromones(all_routes, all_lengths)

            # Statistiky pro vizualizaci/analýzu
            avg_len = float(np.mean(all_lengths)) if all_lengths else float('inf')
            history.append({
                'iteration': iteration + 1,
                'best_route': best_route,
                'best_length': best_length,
                'avg_length': avg_len,
            })

        return history
