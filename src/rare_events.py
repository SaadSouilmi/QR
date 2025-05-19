"""Queue Reactive Rare Events Module.

This module implements statistical methods for detecting and analyzing rare events
in queue-reactive market data.
"""

import os
from glob import glob
from collections import defaultdict
import datetime as dt

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm

from ..utils import pl_date_quantile, pl_select

normal_quantile = norm.ppf


class RareEvents:
    """Queue Reactive Rare Events Analysis.

    This class provides methods for analyzing rare events in queue-reactive market data.
    In all that follows, the paradigm for splitting the data is either providing a
    start_date/end_date pair that determines the 'training' portion of the data, and the
    relevant indicators will be computed on the data that follows. Or providing an
    à la sklearn train_test_split parameter that can be combined with a start_date (if no start_date
    is provided we set it to the earliest possible date) to determine the end_date.
    """

    def __init__(
        self,
        raw_data_dir: str = os.path.join(os.getenv("LOBIB_DATA"), "QR/v1/raw/"),
        n_bins: int = 10,  # Number of imbalance bins
    ) -> None:
        """Initialize RareEvents.

        Args:
            raw_data_dir: Directory containing raw market data files
            n_bins: Number of bins for imbalance discretization
        """
        self.raw_data_dir = raw_data_dir
        files = glob(self.raw_data_dir + "**/*.parquet", recursive=True)
        files = list(map(self.parse_fullfile, files))
        self.files = pl.DataFrame(files)

        self.n_bins = n_bins
        self.bin_edges = 2 * np.arange(self.n_bins + 1) / self.n_bins - 1
        self.imbalance_left: pl.Expr = pl_select(
            condlist=[
                pl.col("imbalance").gt(left) & pl.col("imbalance").le(right)
                for left, right in zip(self.bin_edges[:-1], self.bin_edges[1:])
            ],
            choicelist=[pl.lit(left) for left in self.bin_edges[:-1]],
        ).alias("imbalance_left")

        self.cache_v1 = defaultdict(
            dict
        )  # Simple Cache for computed quantiles {"sotck": {"start_date - end_date": {"quantiles": DataFrame, "indicator": DataFrame}}}

    @staticmethod
    def parse_fullfile(d):
        """Parse file path into components.

        Args:
            d: Full file path

        Returns:
            Dictionary with file metadata (ticker, venue, date)
        """
        tk = d.split("/")[-3]
        ve = d.split("/")[-2]
        dt = d.split("/")[-1].split(".")[0]
        return {"file": d, "ticker": tk, "venue": ve, "date": dt}

    def clear_cache_v1(self) -> None:
        """Clear the cache of computed quantiles."""
        del self.cache_v1
        self.cache_v1 = defaultdict(dict)

    def parse_dates(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        train_test_split: float | None = None,
    ) -> tuple[str, str]:
        """Parse and validate date range for the 'training' portion of the data.

        If dates are not provided, uses earliest/latest available dates or
        train_test_split to determine the date range.

        Args:
            ticker: Instrument symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            train_test_split: Proportion of data to use for training

        Returns:
            Validated (start_date, end_date) pair
        """
        if not start_date:
            start_date = self.files.select(
                pl.col("date").filter(pl.col("ticker").eq(ticker)).min()
            ).item()
        if not end_date:
            if not train_test_split:
                end_date = self.files.select(
                    pl.col("date").filter(pl.col("ticker").eq(ticker)).max()
                ).item()
            else:
                end_date = pl_date_quantile(
                    self.files.filter(
                        pl.col("ticker").eq(ticker) & pl.col("date").ge(start_date)
                    )["date"],
                    q=train_test_split,
                )

        return start_date, end_date

    def list_oos_dates(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        train_test_split: float = 0.4,
    ) -> list[str]:
        """List out-of-sample dates for a given train/test setting.

        Args:
            ticker: Instrument symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            train_test_split: Proportion of data to use for training

        Returns:
            List of out-of-sample dates
        """
        start_date, end_date = self.parse_dates(
            ticker, start_date, end_date, train_test_split
        )

        oos_start_date = (
            dt.datetime.strptime(end_date, "%Y-%m-%d").date() + dt.timedelta(days=1)
        ).strftime("%Y-%m-%d")

        return (
            self.files.filter(
                pl.col("ticker").eq(ticker) & pl.col("date").ge(oos_start_date)
            )
            .sort(by="date")["date"]
            .to_list()
        )

    @staticmethod
    def window_length(alpha: float, alpha_tilde: float, delta: float) -> int:
        r"""
        Calculate optimal window length for certain confidence level and precision.

        Computes the minimum window size needed to achieve a desired confidence interval width
        for the proportion estimator of rare events. The window length is rounded up to the
        nearest hundred to ensure sufficient sample size.

        Our estimate is :math:`\hat{p}(\alpha)_k = \frac{1}{w}\sum_{i=0}^{w-1} I\left(\Delta
        t_{k-i} \leqslant \text{quantile}(\Delta t \mid \text{LOB}_{k-i}, e_{k-i})\right)`,
        and we wish to control the asymptotic confidence interval of level
        :math:`\tilde\alpha`:

        .. math::

            \mathcal{C}_{\tilde\alpha} = \left[\hat{p}(\alpha)_k \pm q_{1 -\frac{\tilde\alpha}{2}}
            \sqrt{\frac{\alpha(1-\alpha)}{w}}\right]

        Say we want:

        .. math::

            q_{1 -\frac{\tilde\alpha}{2}} \sqrt{\frac{\alpha(1-\alpha)}{w}} \leqslant \delta
            \Longleftrightarrow
            w \geqslant \alpha(1 - \alpha)
            \left(\frac{q_{1 - \frac{\tilde\alpha}{2}}}{\delta}\right)^2

        Args:
            alpha: Target proportion of rare events
            alpha_tilde: Confidence level for the interval
            delta: Desired width of the confidence interval

        Returns:
            Optimal window length rounded up to the nearest hundred
        """
        w = alpha * (1 - alpha) * (normal_quantile(1 - alpha_tilde / 2) / delta) ** 2
        return int(np.ceil(w / 100) * 100)

    def load_data(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.LazyFrame:
        """Load and preprocess market data.

        Args:
            ticker: Instrument symbol
            start_date: Start date in YYYY-MM-DD format, optional
            end_date: End date in YYYY-MM-DD format, optional

        Returns:
            Preprocessed market data as a LazyFrame
        """
        start_date, end_date = self.parse_dates(ticker, start_date, end_date)

        relevant_files = self.files.filter(
            pl.col("ticker").eq(ticker)
            & pl.col("date").ge(start_date)
            & pl.col("date").le(end_date)
        )

        df = pl.concat([pl.scan_parquet(file) for file in relevant_files["file"]])

        # We only take up to the second limit beyond best bid/ask
        mask = pl.col("event_queue_nbr").ge(pl.col("best_bid_nbr").sub(1)) & pl.col(
            "event_queue_nbr"
        ).le(pl.col("best_ask_nbr").add(1))
        df = df.filter(mask)
        ts_delta = (
            pl.col("ts_event").diff().cast(pl.Int64).alias("ts_delta")
        )  # $\Delta_t$ in ns
        df = df.with_columns(ts_delta).filter(pl.col("ts_delta").gt(0))
        return df

    def compute_quantiles_v1(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        train_test_split: float = 0.4,
    ) -> pl.DataFrame:
        """Compute time-based quantiles for market events.

        Args:
            ticker: Instrument symbol
            start_date: Start date in YYYY-MM-DD format, optional
            end_date: End date in YYYY-MM-DD format, optional
            train_test_split: Proportion of data to use for training, optional

        Returns:
            DataFrame containing computed quantiles
        """
        start_date, end_date = self.parse_dates(
            ticker, start_date, end_date, train_test_split
        )

        # If already computed return cached result
        key = f"{start_date} - {end_date}"
        if key in self.cache_v1[ticker]:
            return self.cache_v1[ticker][key]["quantiles"]
        self.cache_v1[ticker][key] = dict()

        df = self.load_data(ticker, start_date, end_date)
        df = df.with_columns(self.imbalance_left)

        # Computing this many quantiles seems overkill but it allows us to recover the cdf if ever needed
        quantiles = (
            df.group_by(
                [
                    "imbalance_left",
                    "spread",
                    "event",
                    "event_side",
                    "event_queue_nbr",
                ]
            )
            .agg(
                [
                    pl.col("ts_delta").quantile(quantile).alias(f"q{quantile:.3f}")
                    for quantile in np.arange(0.005, 1, step=0.005)
                ]
            )
            .collect()
        )

        # Cache the result
        self.cache_v1[ticker][key]["quantiles"] = quantiles

        return quantiles

    def join_quantiles_v1(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        train_test_split: float = 0.4,
    ) -> pl.DataFrame:
        """Join computed quantiles with market data.

        Parameters
        ----------
        ticker : str
            Instrument symbol
        start_date : str | None, optional
            Start date in YYYY-MM-DD format
        end_date : str | None, optional
            End date in YYYY-MM-DD format
        train_test_split : float, optional
            Proportion of data to use for training

        Returns
        -------
        pl.DataFrame
            DataFrame with joined quantiles and market data
        """
        start_date, end_date = self.parse_dates(
            ticker, start_date, end_date, train_test_split
        )
        quantiles = self.compute_quantiles_v1(
            ticker, start_date, end_date, train_test_split
        )
        cols = [
            col
            for col in quantiles.columns
            if not col.startswith("q")
            or float(col[1:]) in [0.01, 0.025, 0.05, 0.075, 0.1]
        ]
        quantiles = quantiles.select(cols)

        key = f"{start_date} - {end_date}"
        if "mkt_data" in self.cache_v1[ticker][key]:
            return self.cache_v1[ticker][key]["mkt_data"]

        start_date = (
            dt.datetime.strptime(end_date, "%Y-%m-%d").date() + dt.timedelta(days=1)
        ).strftime("%Y-%m-%d")

        df = self.load_data(ticker, start_date=start_date)
        df = df.with_columns(self.imbalance_left)

        df = df.join(
            quantiles.lazy(),
            on=[
                "imbalance_left",
                "spread",
                "event",
                "event_side",
                "event_queue_nbr",
            ],
            how="left",
        ).select("date", "ts_event", *cols, "ts_delta")

        df = (
            df.with_columns(
                **{
                    col: pl.col(col).gt(pl.col("ts_delta"))
                    for col in cols
                    if col.startswith("q")
                }
            )
            .drop_nulls()
            .collect()
        )

        self.cache_v1[ticker][key]["mkt_data"] = df

        return df

    def compute_indicators_v1(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        train_test_split: float = 0.4,
        window: int = 1800,
    ) -> pl.DataFrame:
        """Compute statistical indicators for rare events.

        Parameters
        ----------
        ticker : str
            Instrument symbol
        start_date : str | None, optional
            Start date in YYYY-MM-DD format
        end_date : str | None, optional
            End date in YYYY-MM-DD format
        train_test_split : float, optional
            Proportion of data to use for training
        window : int, optional
            Rolling window size

        Returns
        -------
        pl.DataFrame
            DataFrame containing computed indicators
        """
        df = self.join_quantiles_v1(ticker, start_date, end_date, train_test_split)

        indicators = df.with_columns(
            **{
                col: pl.col(col).rolling_mean(window_size=window).over("date")
                for col in df.columns
                if col.startswith("q")
            }
        ).drop_nulls()

        indicators = indicators.with_columns(
            **{
                f"Z{col[1:]}": pl.col(col)
                .sub(float(col[1:]))
                .truediv(np.sqrt(float(col[1:]) * (1 - float(col[1:])) / window))
                for col in df.columns
                if col.startswith("q")
            }
        )

        return indicators

    def plot_indicators_v1(
        self,
        ticker: str,
        date: str,
        start_date: str | None = None,
        end_date: str | None = None,
        train_test_split: float = 0.4,
        window: int = 1800,
        alpha: float = 0.05,
        figsize: tuple[int, int] = (7, 4),
    ) -> None:
        """Plot rare event indicators for a specific date.

        Parameters
        ----------
        ticker : str
            Instrument symbol
        date : str
            Date to plot in YYYY-MM-DD format
        start_date : str | None, optional
            Start date in YYYY-MM-DD format
        end_date : str | None, optional
            End date in YYYY-MM-DD format
        train_test_split : float, optional
            Proportion of data to use for training
        window : int, optional
            Rolling window size
        alpha : float, optional
            Significance level
        figsize : tuple[int, int], optional
            Figure size (width, height)
        """
        if not alpha in [0.01, 0.025, 0.05, 0.075, 0.1]:
            raise ValueError("alpha has to be in [0.01, 0.025, 0.05, 0.075, 0.1]")
        quantile_col = f"q{alpha:.3f}"

        if not date in self.list_oos_dates(
            ticker, start_date, end_date, train_test_split
        ):
            raise ValueError(
                "date should be in oos_dates for given ticker and train_test_split"
            )

        indicators = self.compute_indicators_v1(
            ticker, start_date, end_date, train_test_split, window
        ).filter(pl.col("date").cast(str).eq(date))
        fig = plt.figure(figsize=figsize)

        plt.plot(
            indicators["ts_event"].dt.replace_time_zone(None),
            indicators[quantile_col],
        )
        plt.axhline(alpha, linestyle="--", color="red", label=rf"$\alpha={alpha}$")
        plt.xlabel("time of day")
        plt.ylabel("indicator")
        plt.legend()

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.gcf().autofmt_xdate()

        plt.title(f"ticker = {ticker}, date = {date}")
        plt.show()

    def plot_n_indicators_v1(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        train_test_split: float = 0.4,
        window: int = 1800,
        alpha: float = 0.05,
        n: int = 6,
        highest: bool = True,
    ) -> None:
        """Plot rare event indicators for multiple days.

        Parameters
        ----------
        ticker : str
            Instrument symbol
        start_date : str | None, optional
            Start date in YYYY-MM-DD format
        end_date : str | None, optional
            End date in YYYY-MM-DD format
        train_test_split : float, optional
            Proportion of data to use for training
        window : int, optional
            Rolling window size
        alpha : float, optional
            Significance level
        n : int, optional
            Number of days to plot
        highest : bool, optional
            Whether to plot highest or lowest probability days
        """
        if not alpha in [0.01, 0.025, 0.05, 0.075, 0.1]:
            raise ValueError("alpha has to be in [0.01, 0.025, 0.05, 0.075, 0.1]")
        quantile_col = f"q{alpha:.3f}"

        indicators = self.compute_indicators_v1(
            ticker, start_date, end_date, train_test_split, window
        )

        sorted_dates = (
            indicators.group_by("date")
            .agg(pl.col(quantile_col).max())
            .sort(by=quantile_col)["date"]
            .to_list()
        )
        if highest:
            sorted_dates = reversed(sorted_dates[-n:])
        else:
            sorted_dates = sorted_dates[:n]

        n_cols = 3
        n_rows = (n + n_cols - 1) // n_cols

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows))
        axs = axs.ravel()
        for ax, date in zip(axs, sorted_dates):
            data = indicators.filter(pl.col("date").eq(date))
            ax.plot(data["ts_event"].dt.replace_time_zone(None), data[quantile_col])
            ax.axhline(alpha, linestyle="--", color="red", label=rf"$\alpha={alpha}$")
            ax.set_xlabel("time of day")
            ax.set_ylabel("indicator")
            ax.legend()
            ax.set_title(date)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            # fig.autofmt_xdate()

        fig.suptitle(
            f"Days with the highest probability of rare events, ticker = {ticker}"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0, w_pad=3.0)
        plt.show()
