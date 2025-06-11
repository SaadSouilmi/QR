import os 
import datetime as dt
from typing import Any
import gc 

import polars as pl

from .orderbook import LimitOrderBook, Order
from .alpha import Alpha
from .trader import Trader


class Buffer:
    schema = {
        "sequence": pl.Int64,
        "ts": pl.Int64,
        "xt": pl.Int64,
        "dt": pl.Int64,
        "imbalance": pl.Float64,
        "spread": pl.Int64,
        "alpha": pl.Float64,
        "side": pl.Int8,
        "price": pl.Int64,
        "size": pl.Int64,
        "event": pl.String,
        "partial": pl.Boolean,
        "rejected": pl.Boolean,
        "race": pl.Int8,
        "trader_id": pl.Int8,
        "trader_pos": pl.Int64,
        **{
            key: pl.Int64
            for key in [
                *[f"P_{i}_event" for i in range(-4, 0)],
                *[f"Q_{i}_event" for i in range(-4, 0)],
                *[f"Q_{i}_event" for i in range(1, 5)],
                *[f"P_{i}_event" for i in range(1, 5)],
                *[f"P_{i}_recv" for i in range(-4, 0)],
                *[f"Q_{i}_recv" for i in range(-4, 0)],
                *[f"Q_{i}_recv" for i in range(1, 5)],
                *[f"P_{i}_recv" for i in range(1, 5)],
            ]
        },
    }

    def __init__(self, max_size: int | None=None, checkpoint_dir: str | None=None) -> None:
        self.max_size = max_size if checkpoint_dir is not None else None
        self.save_checkpoints = self.max_size is not None
        self.checkpoint_dir = checkpoint_dir
        if self.save_checkpoints:
            os.makedirs(checkpoint_dir, exist_ok=True)
        self.columns: dict[str, list[str, Any]] = {col: [] for col in self.schema.keys()}
        self.checkpoint_nbr = 0

    def clear_buffer(self) -> None:
        for col in self.columns.values():
            col.clear()
        self.current_size = 0
        gc.collect()

    def write_buffer(self, path: str) -> None:
        self.to_df().write_parquet(path)

    def save_checkpoint(self) -> None:
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{self.checkpoint_nbr}.parquet")
        self.write_buffer(path)
        self.checkpoint_nbr += 1
        print(f"Checkpoint saved to {path}")

    def record(self, lob: LimitOrderBook, alpha: Alpha, order: Order, trader: Trader, sequence: int) -> None:
        if self.save_checkpoints and len(self.columns["ts"]) >= self.max_size:
            self.save_checkpoint()
            self.clear_buffer()

        self.columns["sequence"].append(sequence)
        self.columns["ts"].append(order.ts)
        self.columns["xt"].append(order.xt)
        self.columns["dt"].append(order.dt)
        self.columns["imbalance"].append(lob.imbalance)
        self.columns["spread"].append(lob.spread)
        self.columns["alpha"].append(alpha.value)
        self.columns["side"].append(order.side.value)
        self.columns["price"].append(order.price)
        self.columns["size"].append(order.size)
        self.columns["event"].append(order.action)
        self.columns["partial"].append(order.partial)
        self.columns["rejected"].append(order.rejected)
        self.columns["race"].append(order.race)
        self.columns["trader_id"].append(order.trader_id)
        self.columns["trader_pos"].append(trader.curr_pos)
        for i in range(-4, 0):
            p_event, q_event = order.bid.peekitem(4+i)
            p_recv, q_recv = lob.bid.peekitem(4+i)
            self.columns[f"P_{i}_event"].append(p_event)
            self.columns[f"Q_{i}_event"].append(q_event)
            self.columns[f"P_{i}_recv"].append(p_recv)
            self.columns[f"Q_{i}_recv"].append(q_recv)
        for i in range(1, 5):
            p_event, q_event = order.ask.peekitem(i-1)
            p_recv, q_recv = lob.ask.peekitem(i-1)
            self.columns[f"P_{i}_event"].append(p_event)
            self.columns[f"Q_{i}_event"].append(q_event)
            self.columns[f"P_{i}_recv"].append(p_recv)
            self.columns[f"Q_{i}_recv"].append(q_recv)

    def to_df(self) -> pl.DataFrame:
        return pl.DataFrame(self.columns, schema=self.schema)
