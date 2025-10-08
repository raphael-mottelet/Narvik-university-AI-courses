from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class GAParams:
    pop_size: int = 80
    gens: int = 150
    elite: int = 2
    tournament: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.15
    blx_alpha: float = 0.2
    mut_sigma: float = 0.1
    seed: int = 42
    log_every: int = 10


class GA:
    def __init__(self, fitness_fn, bounds, params: GAParams, init_pop: np.ndarray | None = None):
        self.fitness_fn = fitness_fn
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = len(bounds)
        self.p = params
        self.rng = np.random.default_rng(self.p.seed)
        if init_pop is None:
            self.pop = self._random_pop(self.p.pop_size)
        else:
            self.pop = init_pop.astype(float)
        self.fit = self._evaluate(self.pop)

    def _random_pop(self, n: int) -> np.ndarray:
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        u = self.rng.random((n, self.dim))
        return lo + u * (hi - lo)

    def _clip_pop(self, X: np.ndarray) -> np.ndarray:
        return np.minimum(self.bounds[:, 1], np.maximum(self.bounds[:, 0], X))

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.fitness_fn(ind) for ind in X], dtype=float)

    def _tournament(self, k: int) -> int:
        idx = self.rng.integers(0, len(self.pop), size=k)
        best = idx[np.argmax(self.fit[idx])]
        return int(best)

    def _crossover_blx(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        d = hi - lo
        lo_e = lo - self.p.blx_alpha * d
        hi_e = hi + self.p.blx_alpha * d
        u = self.rng.random(self.dim)
        child = lo_e + u * (hi_e - lo_e)
        return child

    def _mutate(self, x: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.dim) < self.p.mutation_rate
        if not mask.any():
            return x
        span = self.bounds[:, 1] - self.bounds[:, 0]
        noise = self.rng.normal(0.0, self.p.mut_sigma * span, size=self.dim)
        x_new = x.copy()
        x_new[mask] += noise[mask]
        return x_new

    def run(self):
        history = []
        elite_n = max(0, min(self.p.elite, len(self.pop)))
        for g in range(1, self.p.gens + 1):
            elite_idx = np.argsort(-self.fit)[:elite_n]
            next_pop = self.pop[elite_idx].copy() if elite_n > 0 else np.empty((0, self.dim))

            while next_pop.shape[0] < self.p.pop_size:
                p1 = self.pop[self._tournament(self.p.tournament)]
                p2 = self.pop[self._tournament(self.p.tournament)]
                child = p1.copy()
                if self.rng.random() < self.p.crossover_rate:
                    child = self._crossover_blx(p1, p2)
                child = self._mutate(child)
                next_pop = np.vstack([next_pop, child[None, :]])

            self.pop = self._clip_pop(next_pop[:self.p.pop_size])
            self.fit = self._evaluate(self.pop)

            if (g % self.p.log_every) == 0 or g == 1 or g == self.p.gens:
                best = float(self.fit.max())
                history.append((g, best))
                print(f"[GA] gen {g:04d}  best {best:.6f}")

        best_idx = int(np.argmax(self.fit))
        return self.pop[best_idx], float(self.fit[best_idx]), history
