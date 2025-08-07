from abc import ABC, abstractmethod

import numpy as np
import polars as pl

from .alpha import Alpha
from .distributions import DiscreteDistribution, ExpDistribution
from .orderbook import (
    Add,
    Cancel,
    Create_Ask,
    Create_Bid,
    LimitOrderBook,
    Order,
    Side,
    Trade,
    TradeAll,
)


class QRModel(ABC):
    @abstractmethod
    def sample_deltat(self, lob: LimitOrderBook) -> float:
        raise NotImplementedError

    @abstractmethod
    def sample_order(self, lob: LimitOrderBook, alpha: float) -> Order:
        raise NotImplementedError

class ImbalanceQR(QRModel):
    def __init__(
        self,
        deltat_distrib: dict[tuple[int, int], ExpDistribution],
        probabilities: pl.DataFrame,
        event_distrib: dict[tuple[int, int], DiscreteDistribution],
        order_volumes_distrib: dict[tuple[str, int], dict[tuple[int, int], DiscreteDistribution]],
        imbalance_bins: np.ndarray,
        rng: np.random.Generator = np.random.default_rng(1337),
    ) -> None:
        self.deltat_distrib = deltat_distrib
        self.probabilities = probabilities
        self.event_distrib = event_distrib
        self.imbalance_bins = imbalance_bins
        self.order_volumes_distrib = order_volumes_distrib
        self.rng = rng

    def map_imbalance_bin(self, imbalance: float) -> float:
        return np.digitize(imbalance, self.imbalance_bins).item()

    def sample_deltat(self, lob: LimitOrderBook) -> int:
        state = (self.map_imbalance_bin(lob.imbalance), min(lob.spread, 2))
        return self.deltat_distrib[state].sample().item() 
    
    def _adjust_probabilities(
        self, eps: float, exclude_cancels: bool = True
    ) -> pl.DataFrame:
        """Multiply Trd probabilities by (1+eps)/(1-eps) for Ask/Bid"""
        events = ["Trd"]
        if not exclude_cancels:
            events.append("Can")
        eps = np.clip(eps, -0.25, 0.25).item()
        is_ask = pl.col("event").is_in(events) & pl.col("event_side").eq("A")
        is_bid = pl.col("event").is_in(events) & pl.col("event_side").eq("B")
        probabilities = self.probabilities.with_columns(
            pl.when(is_ask)
            .then(pl.col("probability").mul(1 + eps))
            .when(is_bid)
            .then(pl.col("probability").mul(1 - eps))
            .otherwise(pl.col("probability"))
            .alias("probability")
        )
        probabilities = probabilities.with_columns(
            pl.col("probability").truediv(
                pl.col("probability").sum().over(["imbalance_left", "spread"])
            )
        )
        return probabilities.filter(pl.col("spread").eq(1))
    
    def adjust_event_distrib(
        self, eps: float, exclude_cancels: bool = True
    ) -> None:
        probabilities = self._adjust_probabilities(eps, exclude_cancels)
        grouped = probabilities.group_by(["imbalance_bins", "spread"]).agg(
            [
                pl.col("event").alias("events"),
                pl.col("event_side").alias("event_sides"),
                pl.col("event_queue_nbr").alias("event_queue_nbrs"),
                pl.col("probability").alias("probability"),
            ]
        )
        for row in grouped.to_dicts():
            state = (row["imbalance_bins"], row["spread"])
            self.event_distrib[state].values = list(
                zip(row["events"], row["event_sides"], row["event_queue_nbrs"])
            )
            self.event_distrib[state].probabilities = row["probability"] 

    def sample_order(self, lob: LimitOrderBook, alpha: Alpha) -> Order:
        if lob.spread > 1:
            state = (self.map_imbalance_bin(lob.imbalance), 2)
            order_type, order_side, order_queue = (
                self.event_distrib[state].sample().flatten()
            )
            if lob.spread % 2 == 0:
                available_queues = np.arange(-(lob.spread // 2) + 1, lob.spread // 2)
            else:
                available_queues = np.hstack(
                    (
                        np.arange(-(lob.spread // 2), 0),
                        np.arange(1, lob.spread // 2 + 1),
                    )
                )
            order_queue = self.rng.choice(available_queues)
            order_size = self.order_volumes_distrib[(order_type.item(), Side[order_side].value)][state].sample().item()
        else:
            state = (self.map_imbalance_bin(lob.imbalance), lob.spread)
            order_type, order_side, order_queue = self.event_distrib[state].sample().flatten()
            order_size = self.order_volumes_distrib[(order_type.item(), int(order_queue))][state].sample().item()

        order = (order_type, order_side, order_queue, order_size)
        return self._create_order(order, lob, alpha.value)

    def _create_order(
        self, order: tuple[str, str, int], lob: LimitOrderBook, alpha: float
    ) -> Order:
        state = dict(
            spread=int(lob.spread),
            imbalance=float(lob.imbalance),
            alpha=alpha,
            ask_sent=dict(lob.ask),
            bid_sent=dict(lob.bid),
        )
        order_type, order_side, order_queue, order_size = order
        order_queue = order_queue.astype(int).item()
        match order_side:
            case "B":
                best_bid_queue = -((lob.spread + 1) // 2)
                if order_queue > 0 and lob.spread % 2 == 1:
                    price = lob.best_bid_price - (best_bid_queue - order_queue) - 1
                else:
                    price = lob.best_bid_price - (best_bid_queue - order_queue)
            case "A":
                best_ask_queue = (lob.spread + 1) // 2
                if order_queue < 0 and lob.spread % 2 == 1:
                    price = lob.best_ask_price + (order_queue - best_ask_queue) + 1
                else:
                    price = lob.best_ask_price + (order_queue - best_ask_queue)

        match order_type:
            case "Add":
                return Add(side=Side[order_side], price=price, size=order_size, **state)
            case "Can":
                return Cancel(side=Side[order_side], price=price, size=order_size, **state)
            case "Trd":
                return Trade(side=Side[order_side], price=price, size=order_size, **state)
            case "Trd_All":
                return TradeAll(side=Side[order_side], price=price, size=order_size, **state)
            case "Create_Ask":
                return Create_Ask(side=Side[order_side], price=price, size=order_size, **state)
            case "Create_Bid":
                return Create_Bid(side=Side[order_side], price=price, size=order_size, **state)
