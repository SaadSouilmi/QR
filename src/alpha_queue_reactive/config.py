"""I use this module to parse config files for slurm jobs. Example:
{
    "ticker": "AAL",
    "alpha": "imbalance",
    "matching_engine": {
        "l1": 1847,
        "l2": 1830,
        "delta": 30000,
        "gamma": 0.2
    },
    "race_model": {
        "type": "no_race",
        "params": {}
    },
    "trader": {
        "trader_id": 1,
        "max_spread": 1,
        "max_volume": 1,
        "alpha_threshold": 0.7,
        "probability_": 0.25
    }
}
"""


import json

import numpy as np
import polars as pl

from ..qr_params import QRParams
from .distributions import DiscreteDistribution, ConstantDistribution
from .race import Race, NoRace, SimpleRace
from .trader import Trader
from .matching_engine import MatchingEngine
from .alpha import Alpha, ImbalanceAlpha


def read_config(path: str) -> dict:
    with open(path, "r") as file:
        config = json.load(file)
    return config


def write_config(config: dict, path: str) -> None:
    with open(path, "w") as file:
        json.dump(config, file, indent=4)


def gamma_distribution(
    loader: QRParams, ticker: str, quantile: float, rng: np.random.Generator
) -> DiscreteDistribution:
    df = loader.load_raw_files(ticker=ticker)
    df = df.filter(
        pl.col("event_queue_nbr")
        .ge(pl.col("best_bid_nbr").sub(1))
        .and_(pl.col("event_queue_nbr").le(pl.col("best_ask_nbr").add(1)))
    )
    deltat = df.select(
        pl.col("ts_event")
        .diff()
        .over("date")
        .drop_nulls()
        .alias("deltat")
        .cast(pl.Int64)
    )
    deltat = deltat.filter(pl.col("deltat").gt(0))
    deltat = deltat.filter(
        pl.col("deltat").le(pl.col("deltat").quantile(quantile))
    ).collect()
    gamma = (
        pl.DataFrame(
            {
                "deltat": pl.arange(
                    deltat.select(pl.col("deltat").min()).item(),
                    deltat.select(pl.col("deltat").max().add(1)).item(),
                    eager=True,
                )
            }
        )
        .join(
            deltat.select(pl.col("deltat").value_counts()).unnest("deltat"),
            on="deltat",
            how="left",
        )
        .fill_null(0)
        .with_columns(
            pl.col("count").truediv(pl.col("count").sum()).alias("probability")
        )
    )
    return DiscreteDistribution(
        gamma.select("deltat").to_numpy().flatten(),
        gamma.select("probability").to_numpy().flatten(),
        rng=rng,
    )


def init_matching_engine(
    loader: QRParams, ticker: str, params: dict, rng: np.random.Generator
) -> MatchingEngine:
    return MatchingEngine(
        l1=params["l1"],
        l2=params["l2"],
        delta=ConstantDistribution(value=params["delta"]),
        gamma=gamma_distribution(loader, ticker, params["gamma"], rng),
    )


def init_alpha(type: str) -> Alpha:
    if type == "imbalance":
        return ImbalanceAlpha()


def init_race(params: dict, rng: np.random.Generator) -> Race:
    match params["type"]:
        case "no_race":
            return NoRace()
        case "simple_race":
            return SimpleRace(**params["params"], rng=rng)


def init_trader(params: dict) -> Trader:
    return Trader(**params)


def parse_config(
    config: dict, loader: QRParams, model_rng: np.random.Generator
) -> dict:
    return {
        "alpha": init_alpha(config["alpha"]),
        "matching_engine": init_matching_engine(
            loader, config["ticker"], config["matching_engine"], model_rng
        ),
        "race_model": init_race(config["race_model"], model_rng),
        "trader": init_trader(config["trader"]),
    }
