import datetime as dt

import polars as pl

from .orderbook import LimitOrderBook, Order
from .alpha import Alpha


class Buffer:
    def __init__(self) -> None:
        self.bid_volumes = []
        self.ask_volumes = []
        self.bid_prices = []
        self.ask_prices = []
        self.imbalance = []
        self.spread = []
        self.orders = []
        self.alpha = []

    def record(self, lob: LimitOrderBook, alpha: Alpha, order: Order) -> None:
        self.bid_prices.append(list(lob.bid.keys()))
        self.bid_volumes.append(list(lob.bid.values()))
        self.ask_prices.append(list(lob.ask.keys()))
        self.ask_volumes.append(list(lob.ask.values()))
        self.imbalance.append(lob.imbalance)
        self.spread.append(lob.spread)
        self.orders.append(order)
        self.alpha.append(alpha.value)

    def to_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                (
                    order.ts,
                    order.xt,
                    order.dt,
                    order.imbalance,
                    order.spread,
                    order.alpha,
                    order.side.value,
                    order.price,
                    order.size,
                    order.action,
                    order.partial,
                    order.rejected,
                    order.race,
                    order.trader_id,
                    *list(order.bid.keys()),
                    *list(order.bid.values()),
                    *list(order.ask.values()),
                    *list(order.ask.keys()),
                    *bid_prices,
                    *bid_volumes,
                    *ask_volumes,
                    *ask_prices,
                )
                for order, bid_prices, bid_volumes, ask_volumes, ask_prices in zip(
                    self.orders,
                    self.bid_prices,
                    self.bid_volumes,
                    self.ask_volumes,
                    self.ask_prices,
                )
            ],
            schema=[
                "ts",
                "xt",
                "dt",
                "imbalance",
                "spread",
                "alpha",
                "side",
                "price",
                "size",
                "event",
                "partial",
                "rejected",
                "race",
                "trader_id",
                *[f"P_{i}_event" for i in range(-4, 0)],
                *[f"Q_{i}_event" for i in range(-4, 0)],
                *[f"Q_{i}_event" for i in range(1, 5)],
                *[f"P_{i}_event" for i in range(1, 5)],
                *[f"P_{i}_recv" for i in range(-4, 0)],
                *[f"Q_{i}_recv" for i in range(-4, 0)],
                *[f"Q_{i}_recv" for i in range(1, 5)],
                *[f"P_{i}_recv" for i in range(1, 5)],
            ],
            orient="row",
        )
