from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Iterable

import numpy as np

from .alpha import Alpha
from .distributions import (
    DiscreteDistribution,
    ExpDistribution,
    GammaDistribution,
)
from .orderbook import (
    Add,
    Cancel,
    LimitOrderBook,
    Order,
    Side,
    Trade,
    TradeAdd,
)


class Race(ABC):
    @abstractmethod
    def probability(self, lob: LimitOrderBook, alpha: float) -> float:
        raise NotImplementedError

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
        raise NotImplementedError


class SimpleRace(Race):

    def __init__(
        self,
        race_id: int,
        p_min: float,
        p_max: float,
        theta_min: float,
        theta_max: float,
        alpha_lower: float,
        alpha_upper: float,
        max_spread: int,
        order_volumes_distrib: dict[str, DiscreteDistribution],
        event_weights: list[float],
        full_alpha: bool,
        rng: np.random.Generator,
    ) -> None:
        self.race_id = race_id
        self.p_min = p_min
        self.p_max = p_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.alpha_lower = alpha_lower
        self.alpha_upper = alpha_upper
        self.max_spread = max_spread
        self.order_volumes_distrib = order_volumes_distrib
        self.event_weights = event_weights
        self.full_alpha = full_alpha  # whether to use alpha.value or alpha.eps
        self.rng = rng

    def probability(self, lob: LimitOrderBook, alpha: Alpha) -> float:
        race_cond = (
            lob.spread <= self.max_spread
            and np.abs(alpha.value if self.full_alpha else alpha.eps)
            >= self.alpha_lower
        )
        p = race_cond * (
            self.p_min
            + (self.p_max - self.p_min)
            * (
                np.abs(alpha.value if self.full_alpha else alpha.eps)
                - self.alpha_lower
            )
            / (self.alpha_upper - self.alpha_lower)
        )

        return min(p, self.p_max)

    def sample_race(self, lob: LimitOrderBook, alpha: Alpha) -> list[Order]:
        race_cond = (
            np.abs(alpha.value if self.full_alpha else alpha.eps)
            >= self.alpha_lower
        )
        theta = self.theta_max + race_cond * (
            self.theta_min - self.theta_max
        ) * (
            np.abs(alpha.value if self.full_alpha else alpha.eps)
            - self.alpha_lower
        ) / (
            self.alpha_upper - self.alpha_lower
        )
        theta = max(theta, self.theta_min)
        n = self.rng.geometric(p=theta)
        if self.full_alpha:
            side = Side.B if alpha.value < 0 else Side.A
        else:
            side = Side.B if alpha.eps < 0 else Side.A
        price = lob.best_bid_price if side is Side.B else lob.best_ask_price
        order_types = self.rng.choice(
            a=[TradeAdd, Cancel], size=n, p=self.event_weights
        )
        orders = [
            order_type(
                side=side,
                price=price,
                size=self.order_volumes_distrib[
                    {"Trd_Add": "Trd", "Can": "Can"}[order_type.action]
                ]
                .sample()
                .item(),
                race=self.race_id,
                spread=lob.spread,
                imbalance=lob.imbalance,
                alpha=alpha,
                ask_sent=dict(lob.ask),
                bid_sent=dict(lob.bid),
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
    def sample_orders(
        self, lob: LimitOrderBook, alpha: float
    ) -> Order | Iterable[Order]:
        raise NotImplementedError


class NoEvent(ExternalEvent):
    def sample_deltat(self) -> int:
        return float("inf")

    def probability(self, lob: LimitOrderBook) -> int:
        return -1

    def sample_orders(
        self, lob: LimitOrderBook, alpha: float
    ) -> Order | Iterable[Order]:
        raise NotImplemented


class ExpEvent(ExternalEvent):

    def __init__(
        self,
        race_id: int,
        lamda: float,
        theta_N: float,
        theta_p: float,
        imb_corr: float,
        order_volumes_distrib: dict[str, DiscreteDistribution],
        event_weights: list[float],
        rng: np.random.Generator,
    ):
        self.race_id = race_id
        self.deltat_distrib = ExpDistribution(lamda, rng)
        self.theta_N = theta_N
        self.theta_p = theta_p
        self.imb_corr = imb_corr
        self.order_volumes_distrib = order_volumes_distrib
        self.event_weights = event_weights
        self.rng = rng

    def sample_deltat(self) -> int:
        return self.deltat_distrib.sample().item()

    def probability(self, lob: LimitOrderBook) -> int:
        return self.theta_p * (lob.spread == 1)

    def sample_orders(
        self, lob: LimitOrderBook, alpha: float
    ) -> Order | Iterable[Order]:
        n = self.rng.geometric(p=self.theta_N)
        side = (
            Side.A
            if self.rng.uniform() < 0.5 * (1 + self.imb_corr * lob.imbalance)
            else Side.B
        )
        price = lob.best_bid_price if side is Side.B else lob.best_ask_price
        order_types = self.rng.choice(
            a=[TradeAdd, Cancel], size=n, p=self.event_weights
        )
        orders = [
            order_type(
                side=side,
                price=price,
                size=self.order_volumes_distrib[
                    {"Trd_Add": "Trd", "Can": "Can"}[order_type.action]
                ]
                .sample()
                .item(),
                race=self.race_id,
                spread=lob.spread,
                imbalance=lob.imbalance,
                alpha=alpha,
                ask_sent=dict(lob.ask),
                bid_sent=dict(lob.bid),
            )
            for order_type in order_types
        ]
        return orders


class GammaEvent(ExternalEvent):

    def __init__(
        self,
        race_id: int,
        shape: float,
        scale: float,
        theta: float,
        order_volumes_distrib: dict[str, DiscreteDistribution],
        rng: np.random.Generator,
    ):
        self.race_id = race_id
        self.deltat_distrib = GammaDistribution(shape, scale, rng)
        self.theta = theta
        self.order_volumes_distrib = order_volumes_distrib
        self.rng = rng

    def sample_deltat(self) -> int:
        return self.deltat_distrib.sample().item()

    def probability(self, lob: LimitOrderBook) -> int:
        return self.theta * (lob.spread == 1)

    def sample_orders(
        self, lob: LimitOrderBook, alpha: float
    ) -> Order | Iterable[Order]:
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
