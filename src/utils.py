from functools import reduce
from itertools import chain
from typing import Callable

import polars as pl
import numpy as np

lob_cols = list(
    chain(
        *[
            [
                f"bid_px_0{i}",
                f"ask_px_0{i}",
                f"bid_sz_0{i}",
                f"ask_sz_0{i}",
                f"bid_ct_0{i}",
                f"ask_ct_0{i}",
            ]
            for i in range(10)
        ]
    )
)


def pl_select(condlist: list[pl.Expr], choicelist: list[pl.Expr]) -> pl.Expr:
    """Implement numpy's select functionality for Polars expressions.

    This function provides similar functionality to numpy.select() but for Polars
    expressions, allowing conditional selection based on multiple conditions.

    Args:
        condlist (list[pl.Expr]): List of conditions as Polars expressions
        choicelist (list[pl.Expr]): List of values to choose from when conditions are met

    Returns:
        pl.Expr: A Polars expression that evaluates to values from choicelist based on
            the first condition in condlist that evaluates to True

    Note:
        Similar to numpy.select (https://numpy.org/doc/stable/reference/generated/numpy.select.html)
        but implemented for Polars expressions
    """
    return reduce(
        lambda expr, cond_choice: expr.when(cond_choice[0]).then(cond_choice[1]),
        zip(condlist, choicelist),
        pl.when(condlist[0]).then(choicelist[0]),
    )


def pl_date_quantile(dates: pl.DataFrame, q: float) -> str:
    """Compute quantile for a column of date strings in Polars.

    This function calculates the quantile of dates by converting them to integers
    (YYYYMMDD format) and then back to a date string.

    Args:
        dates (pl.DataFrame): DataFrame column containing dates in string format (YYYY-MM-DD)
        q (float): Quantile to compute (0 <= q <= 1)

    Returns:
        str: Date string at the specified quantile in YYYY-MM-DD format

    Note:
        Input dates must be in YYYY-MM-DD format with hyphens
    """
    dates = (
        dates.str.replace("-", "").str.replace("-", "").to_numpy().flatten().astype(int)
    )
    quantile = np.quantile(dates, q=q).astype(int)
    date = str(quantile)
    return f"{date[:4]}-{date[4:6]}-{date[6:]}"


def apply_pipeline(df: pl.LazyFrame, pipeline: list[Callable]) -> pl.DataFrame:
    """Apply a pipeline of transforms to a pl.LazyFrame, usefull for preprocessing.

    Args:
        df (pl.LazyFrame): The input dataframe to transform
        pipeline (list[Callable]): A list of functions to apply to the dataframe

    Returns:
        pl.DataFrame: The transformed dataframe
    """

    return reduce(lambda carry, transform: transform(carry), pipeline, df)


def compute_cdf(data) -> tuple[np.ndarray, np.ndarray]:
    """Compute the empirical cumulative distribution function (ECDF).

    Args:
        data: Array-like object containing the data points

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - sorted_data: Sorted input data points
            - cdf: Corresponding CDF values (0 to 1)
    """
    data = np.asarray(data)
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    return sorted_data, cdf


def roll(array, shift):
    """Rotate array elements by a specified shift.

    Args:
        array: Input array to be rotated
        shift: Number of positions to rotate (positive for right rotation,
               negative for left rotation)

    Returns:
        Rotated array where elements are shifted circularly by the specified amount
    """
    return array[-(shift % len(array)) :] + array[: -(shift % len(array))]
