from copy import deepcopy

import numpy as np
import polars as pl

from .orderbook import LimitOrderBook
from .distributions import DiscreteDistribution


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
