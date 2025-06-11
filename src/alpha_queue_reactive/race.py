from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from .orderbook import LimitOrderBook, Order, Side, Add, Cancel, Trade, TradeAll
from .distributions import ExpDistribution, DiscreteDistribution


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

    def sample_deltat(self) -> int:
        return float("inf")
    
    def probability(self, lob: LimitOrderBook, alpha: float) -> float:
        return -1

    def sample_race(self, lob: LimitOrderBook, alpha: float) -> list[Order]:
        raise NotImplemented

class SimpleRace(Race):

    def __init__(
        self,
        race_id: int,
        theta_N: float,
        theta_p: float,
        alpha_threshold: float,
        max_spread: int,
        order_volumes_distrib: dict[str, DiscreteDistribution],
        event_weights: list[float],
        rng: np.random.Generator,
    ) -> None:
        self.race_id = race_id
        self.theta_N = theta_N
        self.theta_p = theta_p
        self.alpha_threshold = alpha_threshold
        self.max_spread = max_spread
        self.order_volumes_distrib = order_volumes_distrib
        self.event_weights = event_weights
        self.rng = rng

    def probability(self, lob: LimitOrderBook, alpha: float) -> float:
        if np.abs(alpha) > self.alpha_threshold and lob.spread <= self.max_spread:
            return self.theta_p
        return 0

    def sample_race(self, lob: LimitOrderBook, alpha: float) -> list[Order]:
        n = self.rng.geometric(p=self.theta_N)
        side = Side(np.where(alpha < 0, -1, 1))
        price = lob.best_bid_price if side is Side.B else lob.best_ask_price
        order_types = self.rng.choice(
            a=[Cancel, Trade], size=n, p=self.event_weights
        )
        orders = [
            order_type(
                side=side,
                price=price,
                size=self.order_volumes_distrib[order_type.action].sample().item(),
                race=self.race_id,
                spread=lob.spread,
                imbalance=lob.imbalance,
                alpha=alpha,
                ask=deepcopy(lob.ask),
                bid=deepcopy(lob.bid),
            )
            for order_type in order_types
        ]
        return orders

class ExternalEvent(ABC):
    @abstractmethod
    def sample_deltat(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def probability(self, lob: LimitOrderBook) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def sample_order(self, lob: LimitOrderBook, alpha: float) -> Order | Iterable[Order]:
        raise NotImplementedError

class NoEvent(ExternalEvent):
    def sample_deltat(self) -> int:
        return float("inf")
    
    def probability(self, lob: LimitOrderBook) -> int:
        return -1

    def sample_order(self, lob: LimitOrderBook, alpha: float) -> Order | Iterable[Order]:
        raise NotImplemented

class ExpEvent(ExternalEvent):

    def __init__(
        self,
        race_id: int,
        lamda: float,
        theta: float,
        order_volumes_distrib: dict[str, DiscreteDistribution],
        rng: np.random.Generator,
    ):
        self.race_id = race_id
        self.deltat_distrib = ExpDistribution(lamda, rng)
        self.theta = theta
        self.order_volumes_distrib = order_volumes_distrib
        self.rng = rng

    def sample_deltat(self) -> int:
        return self.deltat_distrib.sample().item()

    def probability(self, lob: LimitOrderBook) -> int:
        return self.theta * (lob.spread == 1)

    def sample_order(self, lob: LimitOrderBook, alpha: float) -> Order | Iterable[Order]:
        side = Side.A if self.rng.uniform() < 0.5 else Side.B
        price = lob.best_bid_price if side is Side.B else lob.best_ask_price
        type = self.rng.choice([Add, Cancel, Trade])
        size = self.order_volumes_distrib[type.action].sample().item()
        state = dict(
            spread=lob.spread,
            imbalance=lob.imbalance,
            alpha=alpha,
            ask=deepcopy(lob.ask),
            bid=deepcopy(lob.bid),
            race=self.race_id,
        )
        return [type(side=side, price=price, size=size, **state)]
