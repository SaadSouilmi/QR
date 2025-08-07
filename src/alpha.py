from abc import ABC, abstractmethod

import numpy as np

from .distributions import ExpDistribution, UniformDistribution
from .orderbook import LimitOrderBook


class Alpha(ABC):
    @abstractmethod
    def initialise(self, lob: LimitOrderBook) -> None:
        raise NotImplementedError

    @abstractmethod
    def sample_jump(self) -> float:
        raise NotImplementedError


class ImbalanceAlpha(Alpha):
    def __init__(self) -> None:
        self.imbalance = None
        self.eps = None

    def update_imbalance(self, lob: LimitOrderBook) -> None:        
        self.imbalance = lob.imbalance

    def compute_value(self) -> None:
        self.value = self.imbalance

    def initialise(self, lob: LimitOrderBook) -> None:
        self.update_imbalance(lob)
        self.compute_value()

    def sample_jump(self) -> float:
        return float("inf")


class ImbalanceWithJumps(Alpha):
    def __init__(self, beta_imb: float, beta_eps: float, lamda_dt: float, lamda_eps: float, eps_min: float, eps_max: float, rng: np.random.Generator) -> None:
        """alpha_t = beta * imbalance_t + (1 - beta) * eps_t"""
        self.beta_imb = beta_imb
        self.beta_eps = beta_eps
        self.lamda_dt = lamda_dt
        self.lamda_eps = lamda_eps
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_jump_dist = UniformDistribution(eps_min, eps_max, rng)
        self.eps_deltat_dist = ExpDistribution(lamda_dt, rng)
        self.rng = rng
        self.imbalance = None
        self.eps = None

    def compute_value(self) -> None:
        self.value = self.beta_imb * self.imbalance + self.beta_eps * self.eps

    def update_imbalance(self, lob: LimitOrderBook) -> None:
        self.imbalance = lob.imbalance

    def update_eps(self) -> None:
        sign = 1 if self.rng.random() < 0.5 else -1
        self.eps = sign * self.rng.exponential(scale=self.lamda_eps, size=1).item()

    def initialise(self, lob: LimitOrderBook) -> None:
        self.update_imbalance(lob)
        self.update_eps()
        self.compute_value()

    def sample_jump(self) -> float:
        jump_time = self.eps_deltat_dist.sample().item()
        return jump_time
