from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class GAParams:
    pop_size: int = 200
    gens: int = 300
    elite: int = 2
    tournament: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.08
    blx_alpha: float = 0.2
    mut_sigma: float = 0.1
    seed: int = 42
    log_every: int = 10
    binary: bool = True
    kill_copy_age: int = 20
    kill_penalty: float = 1e12


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
        self.pop = self._clip_pop(self.pop)
        self.fit = self._evaluate(self.pop)
        self._champ = None
        self._champ_age = 0

    def _random_pop(self, n: int) -> np.ndarray:
        if self.p.binary:
            return (self.rng.random((n, self.dim)) < 0.2).astype(float)
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        u = self.rng.random((n, self.dim))
        return lo + u * (hi - lo)

    def _clip_pop(self, X: np.ndarray) -> np.ndarray:
        if self.p.binary:
            return (X >= 0.5).astype(float)
        return np.minimum(self.bounds[:, 1], np.maximum(self.bounds[:, 0], X))

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.fitness_fn(ind) for ind in X], dtype=float)

    def _tournament(self, k: int) -> int:
        idx = self.rng.integers(0, len(self.pop), size=k)
        best = idx[np.argmax(self.fit[idx])]
        return int(best)

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.p.binary:
            mask = self.rng.random(self.dim) < 0.5
            child = np.where(mask, a, b)
            return child
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        d = hi - lo
        lo_e = lo - self.p.blx_alpha * d
        hi_e = hi + self.p.blx_alpha * d
        u = self.rng.random(self.dim)
        child = lo_e + u * (hi_e - lo_e)
        return child

    def _mutate(self, x: np.ndarray) -> np.ndarray:
        if self.p.binary:
            mask = self.rng.random(self.dim) < self.p.mutation_rate
            x_new = x.copy()
            x_new[mask] = 1.0 - x_new[mask]
            return x_new
        mask = self.rng.random(self.dim) < self.p.mutation_rate
        if not mask.any():
            return x
        span = self.bounds[:, 1] - self.bounds[:, 0]
        noise = self.rng.normal(0.0, self.p.mut_sigma * span, size=self.dim)
        x_new = x.copy()
        x_new[mask] += noise[mask]
        return x_new

    def _apply_diversity_kill(self):
        if not self.p.binary:
            return
        champ_idx = int(np.argmax(self.fit))
        champ_bits = (self.pop[champ_idx] >= 0.5).astype(float)
        if self._champ is None:
            self._champ = champ_bits
            self._champ_age = 0
            return
        if np.array_equal(champ_bits, self._champ):
            self._champ_age += 1
        else:
            self._champ = champ_bits
            self._champ_age = 0
        if self._champ_age >= self.p.kill_copy_age:
            mask = np.all((self.pop >= 0.5) == (self._champ >= 0.5), axis=1)
            self.fit[mask] = -self.p.kill_penalty

    def run(self):
        history = []
        elite_n = max(0, min(self.p.elite, len(self.pop)))
        for g in range(1, self.p.gens + 1):
            self._apply_diversity_kill()
            elite_idx = np.argsort(-self.fit)[:elite_n]
            next_pop = self.pop[elite_idx].copy() if elite_n > 0 else np.empty((0, self.dim))
            while next_pop.shape[0] < self.p.pop_size:
                p1 = self.pop[self._tournament(self.p.tournament)]
                p2 = self.pop[self._tournament(self.p.tournament)]
                child = p1.copy()
                if self.rng.random() < self.p.crossover_rate:
                    child = self._crossover(p1, p2)
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
