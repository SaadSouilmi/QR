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

class ImbalanceQR(QRModel):
    def __init__(
        self,
        deltat_distrib: dict[tuple[int, int], ExpDistribution],
        event_distrib: dict[tuple[int, int], DiscreteDistribution],
        order_volumes_distrib: dict[tuple[str, int], dict[tuple[int, int], DiscreteDistribution]],
        imbalance_bins: np.ndarray,
        offset: int,
        rng: np.random.Generator = np.random.default_rng(1337),
    ) -> None:
        self.deltat_distrib = deltat_distrib
        self.event_distrib = event_distrib
        self.imbalance_bins = imbalance_bins
        self.order_volumes_distrib = order_volumes_distrib
        self.offset = offset
        self.rng = rng

    def map_imbalance_bin(self, imbalance: float) -> float:
        return np.digitize(imbalance, self.imbalance_bins).item()

    def sample_deltat(self, lob: LimitOrderBook) -> int:
        state = (self.map_imbalance_bin(lob.imbalance), min(lob.spread, 2))
        return self.deltat_distrib[state].sample().item() + self.offset

    def sample_order(self, lob: LimitOrderBook, alpha: float) -> Order:
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

