from abc import ABC, abstractmethod

import numpy as np

from .orderbook import LimitOrderBook


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

    def sample_jump(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns jump_time, jump_value"""
        return float("inf"), None

    def sample_value(self, lob: LimitOrderBook) -> None:
        self.value = lob.imbalance
