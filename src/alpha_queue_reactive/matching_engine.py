from typing import Iterable, List, Tuple
from collections import deque

from .orderbook import Order, LimitOrderBook
from .distributions import Distribution


class OrderQueue:
    """FIFO queue of orders. We use a deque here to emphasize the queue aspect
    and for maximum efficiency."""

    def __init__(self) -> None:
        self.queue = deque()
        self.regular_order_count = 0

    @property
    def dt(self) -> int | None:
        """timestamp of the first order in the queue was sent"""
        return self.queue[0].dt if self.queue else None

    @property
    def empty(self) -> bool:
        return not bool(self.queue)

    @property
    def has_regular_order(self) -> bool:
        return self.regular_order_count > 0

    def append_order(self, order: Order | Iterable[Order]) -> None:
        if isinstance(order, Iterable):
            self.queue.extend(order)
            self.regular_order_count += 1
        else:
            self.queue.append(order)

    def pop_order(self) -> Order:
        order = self.queue.popleft()
        if not order.race:
            self.regular_order_count -= 1
        return order

    def clear(self) -> None:
        self.queue.clear()


class MatchingEngine:

    # Latency params
    l1: int
    l2: int
    delta: Distribution
    gamma: Distribution
    lob: LimitOrderBook

    def __init__(
        self,
        l1: int,
        l2: int,
        delta: Distribution,
        gamma: Distribution,
        lob: LimitOrderBook | None = None,
    ) -> None:
        self.l1 = l1
        self.l2 = l2
        self.delta = delta
        self.gamma = gamma
        self.lob = lob

    def process_regular_order(self, order: Order, curr_ts: int) -> Order:
        order.ts = curr_ts
        order.xt = curr_ts + self.l1
        order.dt = order.xt + self.delta.sample().item() + self.l2
        return self.lob.process_order(order)

    def process_race(self, orders: List[Order], curr_ts: int) -> List[Order]:
        processed_orders = deque()
        gammas = self.gamma.sample(n=len(orders)).cumsum() + self.delta.sample()
        for order, gamma in zip(orders, gammas):
            order.ts = curr_ts
            order.xt = curr_ts + self.l1
            order.dt = order.xt + gamma.item() + self.l2
            processed_order = self.lob.process_order(order)
            if not (processed_order is None):
                processed_orders.append(processed_order)

        return processed_orders
