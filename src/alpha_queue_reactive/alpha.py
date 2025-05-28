from abc import ABC, abstractmethod

import numpy as np

from .orderbook import LimitOrderBook
from .distributions import ExpDistribution
from .race import Race


class Alpha(ABC):
    @abstractmethod
    def initialise(self, lob: LimitOrderBook) -> None:
        raise NotImplemented

    @abstractmethod
    def sample_jump(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplemented

    @abstractmethod
    def sample_value(self, lob: LimitOrderBook) -> None:
        raise NotImplemented


class ImbalanceAlpha(Alpha):
    def __init__(self, eps: int = int(1e3)) -> None:
        self.eps = eps

    def initialise(self, lob: LimitOrderBook) -> None:
        self.value = lob.imbalance

    def sample_value(self, lob: LimitOrderBook) -> None:
        self.value = lob.imbalance

    def sample_jump(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns jump_time, jump_value"""
        return float("inf"), None


class ImbalanceWithJumps(Alpha):
    def __init__(self, lamda: float, race_model: Race, rng: np.random.Generator) -> None:
        self.jump_dist = ExpDistribution(lamda, rng)
        self.rng = rng
        self.race_model = race_model

    def initialise(self, lob: LimitOrderBook) -> None:
        self.value = lob.imbalance
    
    def sample_value(self, lob: LimitOrderBook) -> None:
        self.value = lob.imbalance
    
    def sample_jump(self) -> tuple[np.ndarray, np.ndarray]:
        jump_time = self.jump_dist.sample().item()
        jump_value = 2 * (self.rng.uniform() < 0.5) - 1
        return jump_time, jump_value
