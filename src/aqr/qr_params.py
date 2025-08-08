"""Queue Reactive Parameter Estimation Module.

This module provides functionality for estimating and analyzing parameters of the
queue-reactive model. It includes:

- Estimation of event intensities and probabilities
- Visualization utilities

"""

import os
from glob import glob
from itertools import starmap
from functools import reduce

import matplotlib.pyplot as plt
import polars as pl
import numpy as np

from .utils import pl_select


def v1_estimation(df: pl.LazyFrame, n_bins: int = 10) -> pl.LazyFrame:
    """Estimate queue-reactive model parameters from LOB data.

    This function computes event counts and time deltas for each combination of:
    - Imbalance level
    - Spread
    - Event type
    - Queue position

    The estimation focuses on events up to the second limit beyond best bid/ask,
    binning the imbalance into n_bins equal-sized intervals.

    Args:
        df: Market data in the queue-reactive format (output of parquet_to_qr)
        n_bins: Number of imbalance bins, by default 10

    Returns:
        DataFrame containing aggregated information with columns:
        - imbalance_left: Left edge of imbalance bin
        - spread: Bid-ask spread
        - event: Event type
        - event_side: Side of event (A/B)
        - event_queue_nbr: Queue position
        - time: 30-minute time bucket
        - event_size_sum: Sum of event sizes
        - ts_delta_sum: Sum of time deltas
        - ts_delta_count: Number of events
        - symbol: Instrument identifier
        - date: Event date

    Notes
    -----
    - Only considers events within ±1 level of best bid/ask
    - Imbalance is binned into n_bins equal intervals in [-1, 1]
    - Time is binned into 30-minute intervals
    """
    # We only take up to the second limit beyond best bid/ask
    mask: pl.Expr = pl.col("event_queue_nbr").ge(
        pl.col("best_bid_nbr").sub(1)
    ) & pl.col("event_queue_nbr").le(pl.col("best_ask_nbr").add(1))
    df = df.filter(mask)

    # $\Delta_t$ in ns
    ts_delta: pl.Expr = pl.col("ts_event").diff().cast(pl.Int64).alias("ts_delta")
    df = df.with_columns(ts_delta).filter(pl.col("ts_delta").gt(0))

    # We bin the imbalance by the left value of the interval
    bin_edges = 2 * np.arange(n_bins + 1) / n_bins - 1
    imbalance_left: pl.Expr = pl_select(
        condlist=[
            pl.col("imbalance").gt(left) & pl.col("imbalance").le(right)
            for left, right in zip(bin_edges[:-1], bin_edges[1:])
        ],
        choicelist=[pl.lit(left) for left in bin_edges[:-1]],
    ).alias("imbalance_left")

    # Time of day I bin by 30 minutes here but might add it as a param
    time: pl.Expr = (
        pl.col("ts_event").dt.truncate("30m").dt.strftime("%H:%M:%S").alias("time")
    )
    df = df.with_columns(imbalance_left, time)

    estimations = df.group_by(
        [
            "imbalance_left",
            "spread",
            "event",
            "event_side",
            "event_queue_nbr",
            "time",
        ]
    ).agg(
        event_size_sum=pl.col("event_size").sum(),
        ts_delta_sum=pl.col("ts_delta").sum(),
        ts_delta_count=pl.len(),
        symbol=pl.col("symbol").first(),
        date=pl.col("date").first(),
    )

    estimations = estimations.filter(
        pl.col("event_queue_nbr").abs().le(pl.col("spread").add(1).floordiv(2).add(1))
    )

    return estimations


def queue_sizes(df: pl.DataFrame) -> pl.DataFrame:
    queues = dict()
    for i in range(1, 6):
        queues[f"bid_q{i}"] = pl_select(
            condlist=[pl.col("best_bid_nbr").eq(j) for j in range(-5, 0)],
            choicelist=[pl.col(f"Q_{j}") for j in range(-5 - i + 1, -i + 1)],
        ).alias(f"bid_q{i}")
        queues[f"ask_q{i}"] = pl_select(
            condlist=[pl.col("best_ask_nbr").eq(j) for j in range(1, 6)],
            choicelist=[pl.col(f"Q_{j}") for j in range(i, 5 + i)],
        ).alias(f"ask_q{i}")

    global_max = df.select([expr.max() for expr in queues.values()]).max().row(0)[0]
    full_range = pl.DataFrame({"value": pl.arange(0, global_max + 1, eager=True)})
    for name, expr in queues.items():
        vc = (
            df.select(expr.value_counts())
            .unnest(name)
            .rename({name: "value", "count": name})
        )
        full_range = full_range.join(vc, on="value", how="left").fill_null(0)

    full_range = full_range.with_columns(date=df.select("date").unique().item())
    return full_range


class QRParams:
    """Queue Reactive Parameter Manager.

    This class manages the loading and analysis of queue-reactive model parameters.
    It provides methods to:
    - Load parameter estimations from csv files
    - Compute event intensities and probabilities
    - Visualize parameters

    Args:
        estimations_dir: Directory containing parameter estimation files.
            Default is $LOBIB_DATA/QR/v1/estimations/

    Notes
    -----
    The estimation files should follow the structure:
    estimations_dir/ticker/venue/date.csv
    """

    def __init__(
        self,
        data_dir: str,
    ) -> None:
        self.data_dir = data_dir
        self.raw_dir = os.path.join(self.data_dir, "raw/")
        self.estimations_dir = os.path.join(self.data_dir, "estimations/")
        self.queue_sizes_dir = os.path.join(self.data_dir, "queue_sizes/")
        self.raw_files = sorted(glob(self.raw_dir + "**/*.parquet", recursive=True))
        self.estimlation_files = sorted(
            glob(self.estimations_dir + "**/*.csv", recursive=True)
        )
        self.queue_sizes_files = sorted(
            glob(self.queue_sizes_dir + "**/*.csv", recursive=True)
        )
        files = list(
            starmap(
                self.parse_fullfile,
                zip(self.raw_files, self.estimlation_files, self.queue_sizes_files),
            )
        )
        self.files = pl.DataFrame(files)

    @staticmethod
    def parse_fullfile(
        raw_file: str, estimation_file: str, queue_sizes_file: str
    ) -> dict[str, str]:
        """Parse estimation file path into components.

        Args:
            raw_file: Full path to the raw file
            estimation_file: Full path to estimation file
            queue_sizes_file: Full path to queue_sizes file

        Returns:
            Dictionary containing:
            - file: full file path
            - ticker: instrument symbol
            - venue: trading venue
            - date: estimation date
        """
        tk = estimation_file.split("/")[-3]
        ve = estimation_file.split("/")[-2]
        dt = estimation_file.split("/")[-1].split(".")[0]
        return {
            "raw_file": raw_file,
            "estimation_file": estimation_file,
            "queue_sizes_file": queue_sizes_file,
            "ticker": tk,
            "venue": ve,
            "date": dt,
        }

    def parse_dates(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.DataFrame:
        if not start_date:
            start_date = self.files.select(
                pl.col("date").filter(pl.col("ticker").eq(ticker)).min()
            ).item()
        if not end_date:
            end_date = self.files.select(
                pl.col("date").filter(pl.col("ticker").eq(ticker)).max()
            ).item()

        return self.files.filter(
            pl.col("ticker").eq(ticker)
            & pl.col("date").ge(start_date)
            & pl.col("date").le(end_date)
        )

    def load_raw_files(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.DataFrame:
        relevant_files = self.parse_dates(ticker, start_date, end_date)
        return pl.concat([pl.scan_parquet(file) for file in relevant_files["raw_file"]])

    def load_estimations(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.DataFrame:
        """Load parameter estimations for a specific ticker and date range.

        Args:
            ticker: Instrument symbol
            start_date: Start date in YYYY-MM-DD format. If None, uses earliest available
            end_date: End date in YYYY-MM-DD format. If None, uses latest available

        Returns:
            DataFrame containing parameter estimations
        """
        relevant_files = self.parse_dates(ticker, start_date, end_date)
        estimations = pl.concat(
            [pl.scan_csv(file) for file in relevant_files["estimation_file"]]
        )
        return estimations

    def load_queue_sizes(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.DataFrame:
        relevant_files = self.parse_dates(ticker, start_date, end_date)
        queue_sizes = pl.concat(
            [pl.scan_csv(file) for file in relevant_files["queue_sizes_file"]]
        )
        return queue_sizes

    def compute_intensity(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_spread: int = 4,
        freq: str = "daily",
    ) -> pl.DataFrame:
        """Compute event intensities for a specific ticker.

        The intensity is defined as the inverse of the average time between events:

        .. math::

            \Lambda(\text{Imb},n) = \left( \mathbb{E}\left(\Delta t^{e}_k | k\in \mathcal{K}(\text{Imb},n)\right) \right)^{-1}

        Calculates the average time between events for each combination of
        imbalance level and spread, optionally broken down by time of day.

        Args:
            ticker: Instrument symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_spread: Maximum spread to consider, by default 4
            freq: Frequency of analysis: "daily" or "hourly", by default "daily"

        Returns:
            DataFrame with columns:
            - imbalance_left: Left edge of imbalance bin
            - spread: Bid-ask spread
            - lambda: Average time between events
            - time: Time bucket (only for hourly frequency)

        Raises:
            ValueError: If freq is not "daily" or "hourly"
        """
        estimations = self.load_estimations(ticker, start_date, end_date)
        estimations = estimations.with_columns(
            time=pl.col("time").str.replace("30", "00")
        )

        estimations = estimations.group_by(
            [
                "imbalance_left",
                "spread",
                "event",
                "event_side",
                "event_queue_nbr",
                "time",
            ]
        ).agg(
            ts_delta_sum=pl.col("ts_delta_sum").sum(),
            ts_delta_count=pl.col("ts_delta_count").sum(),
        )

        estimations = estimations.filter(
            (
                pl.col("event").is_in(["Create_Ask", "Create_Bid"])
                | (
                    (pl.col("event_queue_nbr").lt(0) & pl.col("event_side").eq("B"))
                    | (pl.col("event_queue_nbr").gt(0) & pl.col("event_side").eq("A"))
                )
            )
            & pl.col("spread").le(max_spread)
        )

        match freq:
            case "daily":
                intensities = estimations.group_by(["imbalance_left", "spread"]).agg(
                    ts_delta_sum=pl.col("ts_delta_sum").sum(),
                    ts_delta_count=pl.col("ts_delta_count").sum(),
                )
                intensities = intensities.select(
                    "imbalance_left",
                    "spread",
                    pl.col("ts_delta_sum")
                    .truediv(pl.col("ts_delta_count"))
                    .alias("lambda"),
                )

            case "hourly":
                intensities = estimations.group_by(
                    ["imbalance_left", "spread", "time"]
                ).agg(
                    ts_delta_sum=pl.col("ts_delta_sum").sum(),
                    ts_delta_count=pl.col("ts_delta_count").sum(),
                )
                intensities = intensities.select(
                    "imbalance_left",
                    "spread",
                    "time",
                    pl.col("ts_delta_sum")
                    .truediv(pl.col("ts_delta_count"))
                    .alias("lambda"),
                )

            case _:
                raise ValueError(
                    f"freq = {freq}, frequency should either be 'daily' or 'hourly'"
                )

        return intensities.collect()

    def compute_probability(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        freq: str = "daily",
        max_spread: int = 4,
    ) -> pl.DataFrame:
        """Compute event probabilities.

        Calculates the probability of each event type for each combination of
        imbalance level and spread, optionally broken down by time of day.

        Args:
            ticker: Instrument symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            freq: Frequency of analysis: "daily" or "hourly", by default "daily"
            max_spread: Maximum spread to consider, by default 4

        Returns:
            DataFrame with columns:
            - imbalance_left: Left edge of imbalance bin
            - spread: Bid-ask spread
            - event: Event type
            - event_side: Side of event (A/B)
            - event_queue_nbr: Queue position
            - probability: Event probability
            - time: Time bucket (only for hourly frequency)

        Raises:
            ValueError: If freq is not "daily" or "hourly"
        """
        estimations = self.load_estimations(ticker, start_date, end_date)
        estimations = estimations.with_columns(
            pl.col("event").replace({"Trd_All": "Trd"})
        ) # Remove Trd_All
        estimations = estimations.with_columns(
            time=pl.col("time").str.replace("30", "00")
        )

        estimations = estimations.group_by(
            [
                "imbalance_left",
                "spread",
                "event",
                "event_side",
                "event_queue_nbr",
                "time",
            ]
        ).agg(
            ts_delta_sum=pl.col("ts_delta_sum").sum(),
            ts_delta_count=pl.col("ts_delta_count").sum(),
        )

        estimations = estimations.filter(
            (
                pl.col("event").is_in(["Create_Ask", "Create_Bid"])
                | (
                    (pl.col("event_queue_nbr").lt(0) & pl.col("event_side").eq("B"))
                    | (pl.col("event_queue_nbr").gt(0) & pl.col("event_side").eq("A"))
                )
            )
            & pl.col("spread").le(max_spread)
        )

        match freq:
            case "daily":
                total_count: pl.Expr = (
                    pl.col("ts_delta_count").sum().over(["imbalance_left", "spread"])
                ).alias("total_count")
                probability: pl.Expr = (
                    pl.col("ts_delta_count").truediv(total_count).alias("probability")
                )
                probabilities = estimations.with_columns(probability)
                return (
                    probabilities.group_by(
                        [
                            "imbalance_left",
                            "spread",
                            "event",
                            "event_side",
                            "event_queue_nbr",
                        ]
                    )
                    .agg(pl.col("probability").sum())
                    .collect()
                )

            case "hourly":
                total_count: pl.Expr = (
                    pl.col("ts_delta_count")
                    .sum()
                    .over(["imbalance_left", "spread", "time"])
                ).alias("total_count")
                probability: pl.Expr = (
                    pl.col("ts_delta_count").truediv(total_count).alias("probability")
                )

                return estimations.select(
                    "imbalance_left",
                    "spread",
                    "event",
                    "event_side",
                    "event_queue_nbr",
                    "time",
                    probability,
                ).collect()

            case _:
                raise ValueError(
                    f"freq = {freq}, frequency should either be 'daily' or 'hourly'"
                )

    def compute_aes(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.DataFrame:
        """Compute average event sizes.

        Args:
            ticker: Instrument symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame containing average event sizes
        """
        df = self.load_raw_files(ticker, start_date, end_date)
        df = df.filter(pl.col("spread").le(2))
        average_event_size = (
            df.group_by(["event_side", "event_queue_nbr"])
            .agg(pl.col("event_size").mean().ceil().cast(int))
        )
        average_event_size = average_event_size.filter(
            (pl.col("event_side").eq("B") & pl.col("event_queue_nbr").lt(0))
            | (pl.col("event_side").eq("A") & pl.col("event_queue_nbr").gt(0))
        )
        return average_event_size.sort(by="event_queue_nbr")

    def compute_mes(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.DataFrame:
        """Compute median event sizes.

        Args:
            ticker: Instrument symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame containing median event sizes
        """
        df = self.load_raw_files(ticker, start_date, end_date)
        df = df.filter(pl.col("spread").le(2))
        average_event_size = df.group_by(["event_side", "event_queue_nbr"]).agg(
            pl.col("event_size").median().ceil().cast(int)
        )
        average_event_size = average_event_size.filter(
            (pl.col("event_side").eq("B") & pl.col("event_queue_nbr").lt(0))
            | (pl.col("event_side").eq("A") & pl.col("event_queue_nbr").gt(0))
        )
        return average_event_size.sort(by="event_queue_nbr")

    def compute_inv_distribution(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.DataFrame:
        """Compute the queue_size invariant or stationnary distribution

        Args:
            ticker: Instrument symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
        """
        average_event_size = self.compute_aes(ticker, start_date, end_date).collect()
        queue_sizes = self.load_queue_sizes(ticker, start_date, end_date).collect()

        queues = [
            *[(f"bid_q{i}", -i) for i in range(1, 6)],
            *[(f"ask_q{i}", i) for i in range(1, 6)],
        ]
        full_range = pl.DataFrame({"value": pl.arange(0, 25, eager=True)})
        for name, queue_nbr in queues:
            aes = average_event_size.filter(pl.col("event_queue_nbr").eq(queue_nbr))[
                "event_size"
            ].item()
            df = queue_sizes.select("value", name)
            df = (
                df.with_columns(pl.col("value").truediv(aes).ceil().cast(int))
                .group_by("value")
                .agg(pl.col(name).sum())
                .sort(by="value")
            )
            df = df.filter(pl.col("value").lt(25)).with_columns(
                pl.col(name).truediv(pl.col(name).sum())
            )
            full_range = full_range.join(df, on="value", how="left").fill_null(0)
        return full_range

    def plot_probabilities(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        spread: int = 1,
        figsize: tuple[int, int] = (15, 5),
    ) -> None:
        """Plot event probabilities against imbalance.

        Creates three sets of plots showing event probabilities:
        1. At empty queues (if spread > 1)
        2. At best limits
        3. At second limits

        Args:
            ticker: Instrument symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            spread: Spread level to analyze, by default 1
            figsize: Figure size (width, height), by default (15, 5)
        """
        probabilities = self.compute_probability(
            ticker, start_date, end_date, max_spread=spread, freq="daily"
        )
        if spread > 1:
            fig_empty_queues, axs_empty_queues = plt.subplots(1, 2, figsize=figsize)
            axs_empty_queues = axs_empty_queues.ravel()
            df_ = probabilities.filter(
                pl.col("spread").eq(spread)
                & pl.col("probability").gt(0)
                & pl.col("event").ne("F")
                & pl.col("event_queue_nbr").abs().lt((spread - 1) // 2 + 1)
            )

            for ax, ((side,), data) in zip(axs_empty_queues, df_.group_by("event")):
                data = data.sort(by=["event_queue_nbr", "imbalance_left"])
                for (queue,), data_ in data.group_by("event_queue_nbr"):
                    ax.plot(
                        data_["imbalance_left"],
                        data_["probability"],
                        label=rf"$Q_{queue}$",
                        marker="o",
                        ms=3,
                        mec="k",
                    )

                ax.set_title(side)
                ax.set_xlabel("imbalance")
                ax.set_ylabel("probability")
                ax.legend()

            fig_empty_queues.suptitle(
                f"{ticker} Event probabilities at empty queues (spread = {spread})"
            )
            fig_empty_queues.tight_layout()

        fig_first_limits, axs_first_limits = plt.subplots(1, 2, figsize=figsize)
        axs_first_limits = axs_first_limits.ravel()

        df_ = probabilities.filter(
            pl.col("spread").eq(spread)
            & pl.col("probability").gt(0)
            & pl.col("event").ne("F")
            & pl.col("event_queue_nbr").abs().eq((spread - 1) // 2 + 1)
        )

        for ax, ((side,), data) in zip(axs_first_limits, df_.sort(by="event_side", descending=True).group_by("event_side")):
            data = data.sort(by=["event", "imbalance_left"])
            for (event,), data_ in data.group_by("event"):
                ax.plot(
                    data_["imbalance_left"],
                    data_["probability"],
                    label=event,
                    marker="o",
                    ms=3,
                    mec="k",
                )
            ax.set_title({"A": "Ask", "B": "Bid"}[side])
            ax.set_xlabel("imbalance")
            ax.set_ylabel("probability")
            ax.legend()

        fig_first_limits.suptitle(
            f"{ticker} Event probabilities at best limit (spread = {spread})"
        )
        fig_first_limits.tight_layout()

        df_ = probabilities.filter(
            pl.col("spread").eq(spread)
            & pl.col("probability").gt(0)
            & pl.col("event").ne("F")
            & pl.col("event_queue_nbr").abs().eq((spread - 1) // 2 + 2)
        )

        fig_second_limits, axs_second_limits = plt.subplots(1, 2, figsize=(15, 5))
        axs_second_limits = axs_second_limits.ravel()

        for ax, ((side,), data) in zip(axs_second_limits, df_.sort(by="event_side", descending=True).group_by("event_side")):
            data = data.sort(by=["event", "imbalance_left"])
            for (event,), data_ in data.group_by("event"):
                ax.plot(
                    data_["imbalance_left"],
                    data_["probability"],
                    label=event,
                    marker="o",
                    ms=3,
                    mec="k",
                )
            ax.set_title({"A": "Ask", "B": "Bid"}[side])
            ax.set_xlabel("imbalance")
            ax.set_ylabel("probability")
            ax.legend()

        fig_second_limits.suptitle(
            f"{ticker} Event probabilities at second limit (spread = {spread})"
        )
        fig_second_limits.tight_layout()

        plt.show()
        plt.close()
