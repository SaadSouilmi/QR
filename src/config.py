import json

import numpy as np
import polars as pl
from sortedcontainers import SortedDict

from ...utils import pl_select
from ..qr_params import QRParams
from .alpha import Alpha, ImbalanceAlpha, ImbalanceWithJumps
from .distributions import (
    ConstantDistribution,
    DiscreteDistribution,
    ExpDistribution,
)
from .matching_engine import MatchingEngine
from .orderbook import LimitOrderBook
from .queue_reactive import ImbalanceQR
from .race import (
    ExpEvent,
    ExternalEvent,
    GammaEvent,
    NoEvent,
    NoRace,
    Race,
    SimpleRace,
)
from .trader import Trader
from .utils import compute_inv_distributions


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
    # deltat = deltat.with_columns(pl.col("deltat").floordiv(1000)) # convert to micros 
    deltat = deltat.filter(pl.col("deltat").gt(0))
    deltat = deltat.filter(
        pl.col("deltat").le(pl.col("deltat").quantile(quantile)) & pl.col("deltat").ge(4000)
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

def init_order_volumes_distrib(
    loader: QRParams, ticker: str, rng: np.random.Generator
) -> dict[str, DiscreteDistribution]:
    median_event_size = loader.compute_mes(ticker).collect()
    event_size = pl_select(
        condlist=[pl.col("event_queue_nbr").eq(i) for i in range(-2, 3) if not i == 0],
        choicelist=[
            pl.col("event_size")
            .truediv(
                median_event_size.select(
                    pl.col("event_size").filter(pl.col("event_queue_nbr").eq(i))
                ).item()
            )
            .ceil()
            .cast(int)
            for i in range(-2, 3)
            if not i == 0
        ],
    )
    df = (
        loader.load_raw_files(ticker="AAL")
        .filter(
            pl.col("event_queue_nbr").ge(pl.col("best_bid_nbr").sub(1))
            & pl.col("event_queue_nbr").le(pl.col("best_ask_nbr").add(1))
        )
        .with_columns(pl.col("ts_event").diff().cast(pl.Int64).alias("ts_delta"))
        .filter(pl.col("ts_delta").gt(0))
    )
    df = df.with_columns(pl.col("event").replace({"Trd_All": "Trd"}))
    grouped = (
        df.filter(pl.col("spread").eq(1) & pl.col("event_queue_nbr").abs().le(1))
        .group_by("event")
        .agg(event_size.value_counts())
        .collect()
    )
    order_volumes_distrib = dict()

    for event in ["Add", "Can", "Trd"]:
        v_counts = (
            grouped.select(pl.col("event_size").filter(pl.col("event").eq(event)))
            .item()
            .to_frame("event_size")
            .unnest("event_size")
            .sort("event_size")
            .filter(pl.col("event_size").le(25))
            .with_columns(pl.col("count").truediv(pl.col("count").sum()))
        )
        order_volumes_distrib[event] = DiscreteDistribution(
            values=v_counts["event_size"].to_numpy().flatten(),
            probabilities=v_counts["count"].to_numpy().flatten(),
            rng=rng,
        )

    return order_volumes_distrib

def init_conditional_order_volumes_distrib(
    loader: QRParams, ticker: str, bin_edges: np.ndarray, rng: np.random.Generator
) -> dict[tuple[str, int], dict[tuple[int, int], DiscreteDistribution]]:
    median_event_size = loader.compute_mes(ticker).collect()
    df = (
        loader.load_raw_files(ticker)
        .filter(
            pl.col("event_queue_nbr").ge(pl.col("best_bid_nbr").sub(1))
            & pl.col("event_queue_nbr").le(pl.col("best_ask_nbr").add(1))
        )
        .with_columns(pl.col("ts_event").diff().cast(pl.Int64).alias("ts_delta"))
        .filter(pl.col("ts_delta").gt(0))
    )
    df = df.filter(pl.col("spread").le(2))
    imbalance_left: pl.Expr = pl_select(
        condlist=[
            pl.col("imbalance").gt(left) & pl.col("imbalance").le(right)
            for left, right in zip(bin_edges[:-1], bin_edges[1:])
        ],
        choicelist=[pl.lit(left) for left in bin_edges[:-1]],
    ).alias("imbalance_left")
    df = df.with_columns(imbalance_left)
    df = df.with_columns(pl.col("event").replace({"Trd_All": "Trd"}))
    event_size = pl_select(
        condlist=[pl.col("event_queue_nbr").eq(i) for i in range(-2, 3) if not i == 0],
        choicelist=[
            pl.col("event_size")
            .truediv(
                median_event_size.select(
                    pl.col("event_size").filter(pl.col("event_queue_nbr").eq(i))
                ).item()
            )
            .ceil()
            .cast(int)
            for i in range(-2, 3)
            if not i == 0
        ],
    )

    order_volumes_distrib = dict()

    grouped = (
        df.group_by(["imbalance_left", "event", "event_queue_nbr", "spread"])
        .agg(event_size.value_counts())
        .collect()
    )

    for event in ["Add", "Can", "Trd"]:
        subdf = grouped.filter(pl.col("event").eq(event) & pl.col("spread").eq(1))
        for event_queue_nbr in (
            subdf.select(pl.col("event_queue_nbr").unique()).to_numpy().flatten()
        ):
            order_volumes_distrib[(event, event_queue_nbr)] = dict()
            for imbalance_bin, imbalance_left in enumerate(bin_edges[:-1]):
                volumes = (
                    subdf.select(
                        pl.col("event_size").filter(
                            pl.col("event_queue_nbr").eq(event_queue_nbr)
                            & pl.col("imbalance_left").eq(imbalance_left)
                        )
                    )
                    .item()
                    .to_frame("event_size")
                    .unnest("event_size")
                    .sort("event_size")
                )
                volumes = volumes.filter(pl.col("event_size").le(25)).with_columns(
                    pl.col("count").truediv(pl.col("count").sum())
                )
                order_volumes_distrib[(event, event_queue_nbr)][
                    (imbalance_bin + 1, 1)
                ] = DiscreteDistribution(
                    values=volumes["event_size"].to_numpy(),
                    probabilities=volumes["count"].to_numpy(),
                    rng=rng,
                )

    df = df.with_columns(
        pl.col("event").is_in(["Create_Ask", "Create_Bid"]).alias("add_in_spread")
    )
    price_group = (
        pl.when("add_in_spread")
        .then(pl.lit(1))
        .otherwise(
            pl.col("price")
            .ne(pl.col("price").shift().over("date"))
            .cast(int)
            .fill_null(1)
        )
        .cum_sum()
        .alias("price_group")
    )

    grouped = (
        df.group_by(price_group)
        .agg(
            pl.col("event_side").first(),
            pl.col("add_in_spread").first(),
            pl.col("Q_-1").last(),
            pl.col("Q_1").last(),
            pl.col("event").last(),
            pl.col("event_size").last(),
            pl.col("event").first().alias("first_event"),
            pl.len(),
        )
        .filter(pl.col("add_in_spread"))
    )

    grouped = grouped.with_columns(
        pl.col("Q_-1")
        .add(
            pl.when(pl.col("event_side").eq("B") & pl.col("event").eq("Add"))
            .then("event_size")
            .otherwise(pl.lit(0))
        )
        .truediv(
            median_event_size.select(
                pl.col("event_size").filter(pl.col("event_queue_nbr").eq(-1))
            ).item()
        )
        .ceil()
        .cast(int),
        pl.col("Q_1")
        .add(
            pl.when(pl.col("event_side").eq("A") & pl.col("event").eq("Add"))
            .then("event_size")
            .otherwise(pl.lit(0))
        )
        .truediv(
            median_event_size.select(
                pl.col("event_size").filter(pl.col("event_queue_nbr").eq(1))
            ).item()
        )
        .ceil()
        .cast(int),
    )
    grouped = grouped.with_columns(
        pl.when(pl.col("event_side").eq("B"))
        .then("Q_-1")
        .otherwise("Q_1")
        .alias("size")
    ).collect()

    for event, side in zip(["Create_Ask", "Create_Bid"], [1, -1]):
        order_volumes_distrib[(event, side)] = dict()
        subdf = grouped.filter(pl.col("first_event").eq(event))
        volumes = (
            subdf.select(pl.col("size").value_counts()).unnest("size").sort("size")
        )
        volumes = volumes.filter(pl.col("size").le(25)).with_columns(
            pl.col("count").truediv(pl.col("count").sum())
        )
        distribution = DiscreteDistribution(
            values=volumes["size"].to_numpy(),
            probabilities=volumes["count"].to_numpy(),
            rng=rng,
        )
        for imbalance_bin, imbalance_left in enumerate(bin_edges[:-1]):
            order_volumes_distrib[(event, side)][(imbalance_bin + 1, 2)] = distribution

    return order_volumes_distrib

def init_deltat_distrib(
    loader: QRParams, ticker: str, bin_edges: np.ndarray, offset: float, rng: np.random.Generator
) -> dict[tuple[int, int], DiscreteDistribution]:
    intensities = loader.compute_intensity(ticker, max_spread=2)
    intensities = intensities.with_columns(pl.col("lambda").mul(offset))
    intensities = intensities.with_columns(
        imbalance_bins=pl.col("imbalance_left").map_elements(
            lambda x: np.digitize(x, bin_edges[:-1]), return_dtype=int
        )
    )
    deltat_distrib = {
        (row["imbalance_bins"], row["spread"]): ExpDistribution(
            lamda=row["lambda"], rng=rng
        )
        for row in intensities.to_dicts()
    }

    return deltat_distrib

def init_event_distrib(
    loader: QRParams, ticker: str, bin_edges: np.ndarray, rng: np.random.Generator
) -> dict[tuple[int, int], DiscreteDistribution]:
    probabilities = loader.compute_probability(ticker, max_spread=2)
    probabilities = probabilities.filter(
        ~(pl.col("spread").eq(2) & pl.col("event_queue_nbr").ne(0))
    ).with_columns(
        pl.when(pl.col("spread").eq(2))
        .then(
            pl.col("probability").truediv(
                pl.col("probability")
                .filter(pl.col("spread").eq(2))
                .sum()
                .over("imbalance_left")
            )
        )
        .otherwise(pl.col("probability"))
        .alias("probability")
    )
    probabilities = probabilities.with_columns(
        imbalance_bins=pl.col("imbalance_left").map_elements(
            lambda x: np.digitize(x, bin_edges[:-1]), return_dtype=int
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
    event_distrib = {
        (row["imbalance_bins"], row["spread"]): DiscreteDistribution(
            values=list(
                zip(row["events"], row["event_sides"], row["event_queue_nbrs"])
            ),
            probabilities=row["probability"],
            rng=rng,
        )
        for row in grouped.to_dicts()
    }

    return probabilities, event_distrib

def init_imbalance_qr_model(
    loader: QRParams, ticker: str, offset: int, rng: np.random.Generator
) -> ImbalanceQR:
    N_BINS = 10
    BIN_EDGES = 2 * np.arange(N_BINS + 1) / N_BINS - 1

    probabilities, event_distrib = init_event_distrib(loader, ticker, BIN_EDGES, rng)
    return ImbalanceQR(
        deltat_distrib=init_deltat_distrib(loader, ticker, BIN_EDGES, offset, rng),
        probabilities=probabilities,
        event_distrib=event_distrib,
        order_volumes_distrib=init_conditional_order_volumes_distrib(
            loader, ticker, BIN_EDGES, rng
        ),
        imbalance_bins=BIN_EDGES[:-1],
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

def init_alpha(params: dict, rng: np.random.Generator) -> Alpha:
    match params["type"]:
        case "imbalance":
            return ImbalanceAlpha()
        case "imbalance_with_jumps":
            return ImbalanceWithJumps(**params["params"], rng=rng)

def init_race(
    params: dict, loader: QRParams, ticker: str, rng: np.random.Generator
) -> Race:
    order_volumes_distrib = init_order_volumes_distrib(loader, ticker, rng)
    match params["type"]:
        case "no_race":
            return NoRace()
        case "simple_race":
            return SimpleRace(**params["params"], order_volumes_distrib=order_volumes_distrib, rng=rng)

def init_trader(params: dict) -> Trader:
    return Trader(**params)

def init_external_event(params: dict, loader: QRParams, ticker: str, rng: np.random.Generator) -> ExternalEvent:
    order_volumes_distrib = init_order_volumes_distrib(loader, ticker, rng)
    match params["type"]:
        case "no_event":
            return NoEvent()
        case "exp_event":
            return ExpEvent(**params["params"], order_volumes_distrib=order_volumes_distrib, rng=rng)
        case "gamma_event":
            return GammaEvent(**params["params"], order_volumes_distrib=order_volumes_distrib, rng=rng)

def parse_config(
    config: dict, loader: QRParams, model_rng: np.random.Generator
) -> dict:
    return {
        "alpha": init_alpha(config["alpha"], model_rng),
        "qr_model": init_imbalance_qr_model(loader, config["ticker"], config["qr_offset"], model_rng),
        "matching_engine": init_matching_engine(
            loader, config["ticker"], config["matching_engine"], model_rng
        ),
        "race_model": init_race(config["race_model"], loader, config["ticker"], model_rng),
        "external_event_model": init_external_event(config["external_event_model"], loader, config["ticker"], model_rng), 
        "trader": init_trader(config["trader"]),
    }

def init_lob(
    loader: QRParams,
    ticker: str,
    bid_prices: list[int],
    bid_volumes: list[int],
    ask_prices: list[int],
    ask_volumes: list[int],
    lob_rng: np.random.Generator,
):
    normalised_queue_sizes = loader.compute_inv_distribution(ticker)
    inv_distributions = compute_inv_distributions(normalised_queue_sizes, lob_rng)
    lob_params = {
        "bid": SortedDict(
            {price: volume for price, volume in zip(bid_prices, bid_volumes)}
        ),
        "ask": SortedDict(
            {price: volume for price, volume in zip(ask_prices, ask_volumes)}
        ),
        "inv_distributions": inv_distributions,
    }
    lob = LimitOrderBook(**lob_params)
    return lob
