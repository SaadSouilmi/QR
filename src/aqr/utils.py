from copy import deepcopy
from functools import reduce

import numpy as np
import polars as pl

from .orderbook import LimitOrderBook
from .distributions import DiscreteDistribution


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

def copy_lob(lob: LimitOrderBook) -> LimitOrderBook:
    new_lob = deepcopy(lob)
    lob_rng_state = next(iter(lob.inv_distributions.values())).rng.bit_generator.state
    for _, distribution in new_lob.inv_distributions.items():
        distribution.rng.bit_generator.state = deepcopy(lob_rng_state)

    return new_lob


def compute_inv_distributions(
    normalised_queue_sizes: pl.DataFrame, rng: np.random.Generator
):
    queues = [
        *[(f"bid_q{i}", -i) for i in range(1, 5)],
        *[(f"ask_q{i}", i) for i in range(1, 5)],
    ]
    distributions = dict()
    for name, queue_nbr in queues:
        distributions[queue_nbr] = DiscreteDistribution(
            values=normalised_queue_sizes.select("value").to_numpy().flatten(),
            probabilities=normalised_queue_sizes.select(name).to_numpy().flatten(),
            rng=rng,
        )
    return distributions
