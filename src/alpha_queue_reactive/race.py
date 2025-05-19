from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np

from .orderbook import LimitOrderBook, Order, Side, Cancel, Trade, TradeAll


class Race(ABC):
    @abstractmethod
    def probability(self, lob: LimitOrderBook, alpha: float) -> float:
        raise NotImplemented

    @abstractmethod
    def sample_race(self, lob: LimitOrderBook, alpha: float) -> list[Order]:
        raise NotImplementedError


class NoRace(Race):
    def __init__(self) -> None:
        pass

    def probability(self, lob: LimitOrderBook, alpha: float) -> float:
        return -1

    def sample_race(self, lob: LimitOrderBook, alpha: float) -> list[Order]:
        raise NotImplemented


class SimpleRace(Race):
    def __init__(self, theta_N: float, theta_p: float, alpha_threshold: float, max_spread: int, rng: np.random.Generator) -> None:
        self.theta_N = theta_N
        self.theta_p = theta_p
        self.alpha_threshold = alpha_threshold
        self.max_spread = max_spread
        self.rng = rng

    def probability(self, lob: LimitOrderBook, alpha: float) -> float:
        if np.abs(alpha) > self.alpha_threshold and lob.spread <= self.max_spread:
            return self.theta_p
        return 0

    def sample_race(self, lob: LimitOrderBook, alpha: float) -> list[Order]:
        n = self.rng.geometric(p=self.theta_N)
        side = Side(np.where(alpha < 0, -1, 1))
        price = lob.best_bid_price if side is Side.B else lob.best_ask_price
        order_types = self.rng.choice(a=[Cancel, Trade, TradeAll], size=n, p= [0.5, 0.3, 0.2])
        orders = [
            order_type(side=side, price=price, size=1, race=True, spread=lob.spread, imbalance=lob.imbalance, alpha=alpha, ask=deepcopy(lob.ask), bid=deepcopy(lob.bid)) for order_type in order_types
        ]
        return orders
