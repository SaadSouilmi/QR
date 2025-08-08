from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from sortedcontainers import SortedDict

from .distributions import Distribution


class Side(Enum):
    B = -1
    A = 1


@dataclass
class LimitOrderBook:
    bid: SortedDict
    ask: SortedDict
    inv_distributions: dict[int, Distribution]

    def __post_init__(self) -> None:
        for i in range(1, 4):
            if self.best_bid_price - i not in self.bid:
                self.bid[self.best_bid_price - i] = (
                    self.inv_distributions[-i - 1].sample().item()
                )
            if self.best_ask_price + i not in self.ask:
                self.ask[self.best_ask_price + i] = (
                    self.inv_distributions[i + 1].sample().item()
                )

    @property
    def best_bid_price(self) -> int:
        return self.bid.peekitem(index=-1)[0]

    @property
    def best_ask_price(self) -> int:
        return self.ask.peekitem(index=0)[0]

    @property
    def best_bid_volume(self) -> int:
        return self.bid.peekitem(index=-1)[1]

    @property
    def best_ask_volume(self) -> int:
        return self.ask.peekitem(index=0)[1]

    @property
    def spread(self) -> int:
        return self.best_ask_price - self.best_bid_price

    @property
    def imbalance(self) -> float:
        return (self.best_bid_volume - self.best_ask_volume) / (
            self.best_bid_volume + self.best_ask_volume
        )

    @property
    def mid_price(self) -> float:
        return (self.best_bid_price + self.best_ask_price) / 2

    def resolve_order_type(self, order: "Order") -> "Order":
        order.bid_event = dict(self.bid)
        order.ask_event = dict(self.ask)
        if order.rejected:
            return order

        order_args = vars(order).copy()

        if isinstance(order, Add):
            if (order.side is Side.B and order.price in self.ask) or (
                order.side is Side.A and order.price in self.bid
            ):
                match order.side:
                    case Side.B:
                        return Trade(**(order_args | {"price": self.best_ask_price}))
                    case Side.A:
                        return Trade(**(order_args | {"price": self.best_bid_price}))

            elif (order.side is Side.B and order.price not in self.bid) or (
                order.side is Side.A and order.price not in self.ask
            ):
                match order.side:
                    case Side.B:
                        return Create_Bid(**order_args)
                    case Side.A:
                        return Create_Ask(**order_args)

        elif isinstance(order, (Create_Ask, Create_Bid)):
            if (order.side is Side.B and order.price in self.bid) or (
                order.side is Side.A and order.price in self.ask
            ):
                return Add(**order_args)
            elif (order.side is Side.B and order.price in self.ask) or (
                order.side is Side.A and order.price in self.bid
            ):
                match order.side:
                    case Side.B:
                        return Trade(**(order_args | {"price": self.best_ask_price}))
                    case Side.A:
                        return Trade(**(order_args | {"price": self.best_bid_price}))

        return order

    def process_order(self, order: "Order") -> "Order":
        order = self.resolve_order_type(order)
        order.apply_order(self)
        return order

    def clean_bid(self) -> None:
        if self.bid[self.best_bid_price] == 0:
            del self.bid[self.best_bid_price]
        if self.bid[self.best_bid_price] == 0:
            self.bid[self.best_bid_price - 1] = (
                self.inv_distributions[-1].sample().item()
            )
            del self.bid[self.best_bid_price]
        if self.best_bid_price - 1 not in self.bid:
            self.bid[self.best_bid_price - 1] = (
                self.inv_distributions[-2].sample().item()
            )
        if self.best_bid_price - 2 not in self.bid:
            self.bid[self.best_bid_price - 2] = (
                self.inv_distributions[-3].sample().item()
            )
        if self.best_bid_price - 3 not in self.bid:
            self.bid[self.best_bid_price - 3] = (
                self.inv_distributions[-4].sample().item()
            )

    def clean_ask(self) -> None:
        if self.ask[self.best_ask_price] == 0:
            del self.ask[self.best_ask_price]
        if self.ask[self.best_ask_price] == 0:
            self.ask[self.best_ask_price + 1] = (
                self.inv_distributions[1].sample().item()
            )
            del self.ask[self.best_ask_price]
        if self.best_ask_price + 1 not in self.ask:
            self.ask[self.best_ask_price + 1] = (
                self.inv_distributions[2].sample().item()
            )
        if self.best_ask_price + 2 not in self.ask:
            self.ask[self.best_ask_price + 2] = (
                self.inv_distributions[3].sample().item()
            )
        if self.best_ask_price + 3 not in self.ask:
            self.ask[self.best_ask_price + 3] = (
                self.inv_distributions[4].sample().item()
            )


@dataclass
class Order(ABC):
    side: Side
    price: int
    spread: int  # At creation
    imbalance: float  # At creation
    alpha: float  # At creation
    ask_sent: SortedDict = None  # At creation
    bid_sent: SortedDict = None  # At creation
    ask_event: SortedDict = None  # At execution
    bid_event: SortedDict = None  # At execution
    size: int = 1
    ts: int = 0  # timestamp order was sent at
    xt: int = 0  # timestamp order was received by matching engine
    dt: int = 0  # timestamp update reaches market participants
    partial: bool = False
    rejected: bool = False
    race: int = 0  # race_id 0 for no race
    trader_id: int = 0

    @property
    @abstractmethod
    def action(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def apply_bid(self, lob: LimitOrderBook) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_ask(self, lob: LimitOrderBook) -> None:
        raise NotImplementedError

    def apply_order(self, lob: LimitOrderBook) -> None:
        match self.side:
            case Side.B:
                self.apply_bid(lob)
            case Side.A:
                self.apply_ask(lob)


class Add(Order):
    action = "Add"

    def apply_bid(self, lob: LimitOrderBook) -> None:
        if self.rejected:
            return

        lob.bid[self.price] += self.size

    def apply_ask(self, lob: LimitOrderBook) -> None:
        if self.rejected:
            return

        lob.ask[self.price] += self.size


class Cancel(Order):
    action = "Can"

    def apply_bid(self, lob: LimitOrderBook) -> None:
        if self.rejected:
            return

        if self.price not in lob.bid:
            self.rejected = True
            return

        if lob.bid[self.price] < self.size:
            self.partial = True

        lob.bid[self.price] = max(0, lob.bid[self.price] - self.size)
        lob.clean_bid()

    def apply_ask(self, lob: LimitOrderBook) -> None:
        if self.rejected:
            return

        if self.price not in lob.ask:
            self.rejected = True
            return

        if lob.ask[self.price] < self.size:
            self.partial = True

        lob.ask[self.price] = max(0, lob.ask[self.price] - self.size)
        lob.clean_ask()


class Trade(Order):
    action = "Trd"

    def apply_bid(self, lob: LimitOrderBook) -> None:
        if self.rejected:
            return

        if self.price < lob.best_bid_price:
            self.rejected = True
            return

        # This would correspond to a race where the volume as been depleted already
        if self.price not in lob.bid:
            self.rejected = True
            return

        # This would correspond to a partial fill
        if lob.bid[self.price] < self.size:
            self.partial = True

        self.size = min(self.size, lob.bid[self.price])
        lob.bid[self.price] = lob.bid[self.price] - self.size
        lob.clean_bid()

    def apply_ask(self, lob: LimitOrderBook) -> None:
        if self.rejected:
            return

        if self.price > lob.best_ask_price:
            self.rejected = True
            return

        # This would correspond to a race where the volume as been depleted already
        if self.price not in lob.ask:
            self.rejected = True
            return

        # This would correspond to a partial fill
        if lob.ask[self.price] < self.size:
            self.partial = True

        self.size = min(self.size, lob.ask[self.price])
        lob.ask[self.price] = lob.ask[self.price] - self.size
        lob.clean_ask()


class TradeAdd(Order):
    action = "Trd_Add"

    def apply_bid(self, lob: LimitOrderBook) -> None:
        if self.rejected:
            return

        if self.price < lob.best_bid_price:
            self.rejected = True
            return

        if self.price not in lob.bid:
            lob.ask[lob.best_ask_price] += self.size
        else:
            lob.bid[self.price] = max(0, lob.bid[self.price] - self.size)
            lob.ask[lob.best_ask_price] += max(0, self.size - lob.bid[self.price])
            lob.clean_bid()

    def apply_ask(self, lob: LimitOrderBook) -> None:
        if self.rejected:
            return

        if self.price > lob.best_ask_price:
            self.rejected = True
            return

        if self.price not in lob.ask:
            lob.bid[lob.best_bid_price] += self.size
        else:
            lob.ask[self.price] -= max(0, lob.ask[self.price] - self.size)
            lob.bid[lob.best_bid_price] += max(0, self.size - lob.ask[self.price])
            lob.clean_ask()

    def adjust_price(self, lob: LimitOrderBook) -> None:
        if self.side is Side.B:  # Buy order
            if self.price < lob.best_bid_price:
                self.price = lob.best_bid_price
            elif self.price > lob.best_ask_price:
                self.price = lob.best_ask_price
        elif self.side is Side.A:  # Sell order  
            if self.price > lob.best_ask_price:
                self.price = lob.best_ask_price
            elif self.price < lob.best_bid_price:
                self.price = lob.best_bid_price


class TradeAll(Order):
    action = "Trd_All"

    def apply_bid(self, lob: LimitOrderBook) -> None:
        if self.rejected:
            return

        if self.price < lob.best_bid_price:
            self.rejected = True
            return

        # This would correspond to a race where the volume as been depleted already
        if self.price not in lob.bid:
            self.rejected = True
            return

        lob.bid[self.price] = 0
        lob.clean_bid()

    def apply_ask(self, lob: LimitOrderBook) -> None:
        if self.rejected:
            return

        if self.price > lob.best_ask_price:
            self.rejected = True
            return

        # This would correspond to a race where the volume as been depleted already
        if self.price not in lob.ask:
            self.rejected = True
            return

        lob.ask[self.price] = 0
        lob.clean_ask()


class Create_Ask(Order):
    action = "Create_Ask"

    def apply_bid(self, lob: LimitOrderBook) -> None:
        raise NotImplementedError

    def apply_ask(self, lob: LimitOrderBook) -> None:
        if self.rejected:
            return
        assert (self.price not in lob.ask) and (self.price not in lob.bid)

        lob.ask[self.price] = self.size
        for i in range(1, 4):
            if self.price + i not in lob.ask:
                lob.ask[self.price + i] = 0

        for price, volume in lob.ask.items()[:]:
            if not (price >= self.price and price <= self.price + 3):
                del lob.ask[price]


class Create_Bid(Order):
    action = "Create_Bid"

    def apply_bid(self, lob: LimitOrderBook) -> None:
        if self.rejected:
            return
        assert (self.price not in lob.ask) and (self.price not in lob.bid)

        lob.bid[self.price] = self.size
        for k in range(1, 4):
            if self.price - k not in lob.bid:
                lob.bid[self.price - k] = 0

        for price, volume in lob.bid.items()[:]:
            if not (price <= self.price and price >= self.price - 3):
                del lob.bid[price]

    def apply_ask(self, lob: LimitOrderBook) -> None:
        raise NotImplementedError


def process_marketable_limit_order(
    lob: LimitOrderBook, order: Order
) -> tuple[Order, Order]:
    order.adjust_price(lob)
    limit_order_side = Side.B if order.side == Side.A else Side.A
    can_trade = (order.side == Side.B and order.price == lob.best_bid_price) or (
        order.side == Side.A and order.price == lob.best_ask_price
    )

    if not can_trade:
        limit_order_size = order.size
        market_order_size = 0
    else:
        market_order_size = (
            lob.best_bid_volume if order.side == Side.B else lob.best_ask_volume
        )
        market_order_size = min(market_order_size, order.size)
        limit_order_size = order.size - market_order_size

    market_order_rejected = market_order_size == 0
    limit_order_rejected = limit_order_size == 0

    market_order = Trade(
        **(vars(order) | {"size": market_order_size, "rejected": market_order_rejected})
    )
    limit_order = Add(
        **(
            vars(order)
            | {
                "size": limit_order_size,
                "side": limit_order_side,
                "rejected": limit_order_rejected,
            }
        )
    )

    return market_order, limit_order
