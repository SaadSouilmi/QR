from copy import deepcopy
from abc import ABC, abstractmethod

import polars as pl
import numpy as np

from .distributions import DiscreteDistribution, ExpDistribution
from .orderbook import (
    LimitOrderBook,
    Side,
    Order,
    Add,
    Cancel,
    Trade,
    TradeAll,
    Create_Ask,
    Create_Bid,
)


class QRModel(ABC):
    @abstractmethod
    def sample_deltat(self, lob: LimitOrderBook) -> float:
        raise NotImplementedError

    @abstractmethod
    def sample_order(self, lob: LimitOrderBook) -> Order:
        raise NotImplementedError


class FixedSizeQR(QRModel):
    def __init__(
        self,
        intensities: dict[tuple[int, int], ExpDistribution],
        probabilities: dict[tuple[int, int], DiscreteDistribution],
        imbalance_bins: np.ndarray,
        rng: np.random.Generator = np.random.default_rng(1337),
    ) -> None:
        self.intensities = intensities
        self.probabilities = probabilities
        self.imbalance_bins = imbalance_bins
        self.rng = rng

    def map_imbalance_bin(self, imbalance: float) -> float:
        return np.digitize(imbalance, self.imbalance_bins).item()

    def sample_deltat(self, lob: LimitOrderBook) -> int:
        state = (self.map_imbalance_bin(lob.imbalance), min(lob.spread, 2))
        return self.intensities[state].sample().item()

    def sample_order(self, lob: LimitOrderBook, alpha: float) -> Order:
        if lob.spread > 1:
            state = (self.map_imbalance_bin(lob.imbalance), 2)
            order_type, order_side, order_queue = (
                self.probabilities[state].sample().flatten()
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
            order = (order_type, order_side, order_queue)
        else:
            state = (self.map_imbalance_bin(lob.imbalance), lob.spread)
            order = self.probabilities[state].sample().flatten()

        return self.create_order(order, lob, alpha)

    def create_order(
        self, order: tuple[str, str, int], lob: LimitOrderBook, alpha: float
    ) -> Order:
        state = dict(
            spread=lob.spread,
            imbalance=lob.imbalance,
            alpha=alpha,
            ask=deepcopy(lob.ask),
            bid=deepcopy(lob.bid),
        )
        order_type, order_side, order_queue = order
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
                return Add(side=Side[order_side], price=price, size=1, **state)
            case "Can":
                return Cancel(side=Side[order_side], price=price, size=1, **state)
            case "Trd":
                return Trade(side=Side[order_side], price=price, size=1, **state)
            case "Trd_All":
                return TradeAll(side=Side[order_side], price=price, size=1, **state)
            case "Create_Ask":
                return Create_Ask(side=Side[order_side], price=price, size=1, **state)
            case "Create_Bid":
                return Create_Bid(side=Side[order_side], price=price, size=1, **state)


def init_fixed_size_qr(
    intensities: pl.DataFrame,
    probabilities: pl.DataFrame,
    rng: np.random.Generator,
) -> FixedSizeQR:
    """Since we consider large tick assets, we assume the spread tightens immediatly if
    it is greater than 1."""

    imbalance_bins = np.sort(probabilities["imbalance_left"].unique().to_numpy())
    intensities = intensities.with_columns(
        imbalance_bins=pl.col("imbalance_left").map_elements(
            lambda x: np.digitize(x, imbalance_bins), return_dtype=int
        )
    )
    intensities_dict = {
        (row["imbalance_bins"], row["spread"]): ExpDistribution(
            lamda=row["lambda"], rng=rng
        )
        for row in intensities.to_dicts()
    }

    probabilities = probabilities.with_columns(
        imbalance_bins=pl.col("imbalance_left").map_elements(
            lambda x: np.digitize(x, imbalance_bins), return_dtype=int
        )
    )
    grouped = probabilities.group_by(["imbalance_bins", "spread"]).agg(
        [
            pl.col("event").alias("events"),
            pl.col("event_side").alias("event_sides"),
            pl.col("event_queue_nbr").alias("event_queue_nbrs"),
            pl.col("probability").alias("probability"),
        ]
    )
    probabilities_dict = {
        (row["imbalance_bins"], row["spread"]): DiscreteDistribution(
            values=list(
                zip(row["events"], row["event_sides"], row["event_queue_nbrs"])
            ),
            probabilities=row["probability"],
            rng=rng,
        )
        for row in grouped.to_dicts()
    }

    return FixedSizeQR(intensities_dict, probabilities_dict, imbalance_bins, rng=rng)
