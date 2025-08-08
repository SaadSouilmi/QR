import heapq
from collections import deque
from typing import Iterable, List, Tuple

import numpy as np

from .distributions import Distribution
from .orderbook import Add, LimitOrderBook, Order, Side, Trade


class OrderQueue:
    """Priority queue of orders ordered by timestamp (dt). Uses a min-heap
    to efficiently track the order with the smallest dt."""

    def __init__(self) -> None:
        # Min-heap: (dt, insertion_order, order) insertion_order is a tiebreaker
        self.heap = []  
        self.regular_order_count = 0
        self.trader_order_count = 0
        self._insertion_counter = 0  

    @property
    def dt(self) -> int | float:
        """timestamp of the order with smallest dt in the queue"""
        return self.heap[0][0] if self.heap else float("inf")

    @property
    def empty(self) -> bool:
        return not bool(self.heap)

    @property
    def has_regular_order(self) -> bool:
        return self.regular_order_count > 0

    def append_order(self, order: "Order", regular_order: bool = False) -> None:
        """Add an order to the priority queue"""
        heapq.heappush(self.heap, (order.dt, self._insertion_counter, order))
        self._insertion_counter += 1
        if regular_order:
            self.regular_order_count += 1

    def append_race(self, orders: Iterable["Order"]) -> None:
        """Add multiple orders to the priority queue"""
        for order in orders:
            heapq.heappush(self.heap, (order.dt, self._insertion_counter, order))
            self._insertion_counter += 1
            if order.trader_id == 1:
                self.trader_order_count += 1

    def pop_order(self) -> "Order":
        """Remove and return the order with the smallest dt"""
        if not self.heap:
            raise IndexError("pop from empty queue")

        dt, insertion_order, order = heapq.heappop(self.heap)

        if order.race == 0 and order.trader_id == 0:
            self.regular_order_count -= 1
        if order.trader_id == 1:
            self.trader_order_count -= 1

        return order

    def peek_order(self) -> "Order":
        """Return the order with smallest dt without removing it"""
        if not self.heap:
            raise IndexError("peek from empty queue")
        return self.heap[0][2]

    def clear(self) -> None:
        """Remove all orders from the queue"""
        self.heap.clear()
        self.regular_order_count = 0
        self._insertion_counter = 0

    def __len__(self) -> int:
        """Return the number of orders in the queue"""
        return len(self.heap)


# class OrderQueue:
#     """FIFO queue of orders. We use a deque here to emphasize the queue aspect
#     and for maximum efficiency."""

#     def __init__(self) -> None:
#         self.queue = deque()
#         self.regular_order_count = 0

#     @property
#     def dt(self) -> int | None:
#         """timestamp of the first order in the queue was sent"""
#         return self.queue[0].dt if self.queue else None

#     @property
#     def empty(self) -> bool:
#         return not bool(self.queue)

#     @property
#     def has_regular_order(self) -> bool:
#         return self.regular_order_count > 0

#     def append_order(self, order: Order, regular_order: bool = False) -> None:
#         self.queue.append(order)
#         if regular_order:
#             self.regular_order_count += 1

#     def append_race(self, order: Iterable[Order]) -> None:
#         self.queue.extend(order)

#     def pop_order(self) -> Order:
#         order = self.queue.popleft()
#         if order.race == 0 and order.trader_id == 0:
#             self.regular_order_count -= 1
#         return order

#     def clear(self) -> None:
#         self.queue.clear()


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

    def process_race(self, orders: List[Order], curr_ts: int) -> List[Order]:
        gammas = self.gamma.sample(n=len(orders))
        gammas.sort()
        gammas = gammas.cumsum()
        gammas[0] = 0
        gammas += self.delta.sample()
        for order, gamma in zip(orders, gammas):
            order.ts = curr_ts
            order.xt = curr_ts + self.l1
            order.dt = order.xt + gamma.item() + self.l2