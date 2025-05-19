from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
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
            if not self.best_bid_price - i in self.bid:
                self.bid[self.best_bid_price - i] = (
                    self.inv_distributions[-i - 1].sample().item()
                )
            if not self.best_ask_price + i in self.ask:
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
        return (self.best_bid[0] + self.best_ask[0]) / 2

    def resolve_order_type(self, order: "Order") -> "Order":
        if order.rejected:
            return order
        
        order_args = dict( # Probably should replace this with vars
            side=order.side,
            ts=order.ts,
            xt=order.xt,
            dt=order.dt,
            price=order.price,
            size=order.size,
            imbalance=order.imbalance,
            spread=order.spread,
            alpha=order.alpha,
            ask=order.ask,
            bid=order.bid,
            partial=order.partial,
            rejected=order.rejected,
            race=order.race,
            trader_id=order.trader_id,
        )
        if isinstance(order, Add):
            if (order.side is Side.B and order.price not in self.bid) or (
                order.side is Side.A and order.price not in self.ask
            ):
                match order.side:
                    case Side.B:
                        return Create_Bid(**order_args)
                    case Side.A:
                        return Create_Ask(**order_args)
            elif (order.side is Side.B and order.price in self.ask) or (
                order.side is Side.A and order.price in self.bid
            ):
                match order.side:
                    case Side.B:
                        return Trade(**(order_args | {"price": self.best_ask_price}))
                    case Side.A:
                        return Trade(**(order_args | {"price": self.best_bid_price}))
            
        elif isinstance(order, (Create_Ask, Create_Bid)):
            if (order.side is Side.B and order.price in self.bid) or (order.side is Side.A and order.price in self.ask):
                return Add(**order_args)
            elif (order.side is Side.B and order.price in self.ask) or (order.side is Side.A and order.price in self.bid):
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
        if not self.best_bid_price - 1 in self.bid:
            self.bid[self.best_bid_price - 1] = (
                self.inv_distributions[-2].sample().item()
            )
        if not self.best_bid_price - 2 in self.bid:
            self.bid[self.best_bid_price - 2] = (
                self.inv_distributions[-3].sample().item()
            )
        if not self.best_bid_price - 3 in self.bid:
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
        if not self.best_ask_price + 1 in self.ask:
            self.ask[self.best_ask_price + 1] = (
                self.inv_distributions[2].sample().item()
            )
        if not self.best_ask_price + 2 in self.ask:
            self.ask[self.best_ask_price + 2] = (
                self.inv_distributions[3].sample().item()
            )
        if not self.best_ask_price + 3 in self.ask:
            self.ask[self.best_ask_price + 3] = (
                self.inv_distributions[4].sample().item()
            )


@dataclass
class Order(ABC):
    side: Side
    price: int
    spread: int # At order creation
    imbalance: float # At order creation
    alpha: float   # At order creation 
    ask: SortedDict
    bid: SortedDict
    size: int = 1
    ts: int = 0  # timestamp order was sent at
    xt: int = 0  # timestamp order was received by matching engine
    dt: int = 0  # timestamp update reaches market participants
    partial: bool = False  # whether the order wall partiall filled
    rejected: bool = False  # whether the order was rejected all together
    race: bool = False  # whether the order is from a race
    trader_id: int = 0  # trader id 1 for informed trader and 0 for other participants

    @property
    @abstractmethod
    def action(self) -> str:
        raise NotImplemented

    @abstractmethod
    def apply_bid(self, lob: LimitOrderBook) -> None:
        raise NotImplemented

    @abstractmethod
    def apply_ask(self, lob: LimitOrderBook) -> None:
        raise NotImplemented

    def apply_order(self, lob: LimitOrderBook) -> None:
        match self.side:
            case Side.B:
                self.apply_bid(lob)
            case Side.A:
                self.apply_ask(lob)


# class OrderType(Enum):
#     Add = "Add"
#     Cancel = "Can"
#     Trade = "Trd"
#     Trade_All = "Trd_All"
#     Create_Ask = "Create_Ask"
#     Create_Bid = "Create_Bid"


@dataclass
class Add(Order):
    @property
    def action(self) -> str:
        return "Add"

    def apply_bid(self, lob: LimitOrderBook) -> None:
        lob.bid[self.price] += self.size

    def apply_ask(self, lob: LimitOrderBook) -> None:
        lob.ask[self.price] += self.size


@dataclass
class Cancel(Order):
    @property
    def action(self) -> str:
        return "Cancel"

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


@dataclass
class Trade(Order):
    @property
    def action(self) -> str:
        return "Trade"

    def apply_bid(self, lob: LimitOrderBook) -> None:
        if self.rejected:
            return

        if self.price < lob.best_bid_price:
            self.rejected = True
            return 
            # raise ValueError(
            #     f"Price is strictly lower than the best bid price, price={self.price}, best_bid={lob.best_bid_price}"
            # )

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
            # raise ValueError(f"Price is not the best ask price, price={self.price}, best_ask={lob.best_ask_price}")

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


@dataclass
class TradeAll(Order):
    @property
    def action(self) -> str:
        return "Trade_All"

    def apply_bid(self, lob: LimitOrderBook) -> None:
        if self.rejected:
            return

        if self.price < lob.best_bid_price:
            self.rejected = True
            return 
            # raise ValueError(
                # f"Price is not the best bid price, price={self.price}, best_bid={lob.best_bid_price}"
            # )

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
            # raise ValueError(
            #     f"Price is not the best ask price, price={self.price}, best_ask={lob.best_ask_price}"
            # )

        # This would correspond to a race where the volume as been depleted already
        if self.price not in lob.ask:
            self.rejected = True
            return

        lob.ask[self.price] = 0
        lob.clean_ask()


@dataclass
class Create_Ask(Order):
    @property
    def action(self) -> str:
        return "Create_Ask"

    def apply_bid(self, lob: LimitOrderBook) -> None:
        raise NotImplementedError

    def apply_ask(self, lob: LimitOrderBook) -> None:
        assert (self.price not in lob.ask) and (self.price not in lob.bid)

        lob.ask[self.price] = self.size
        for i in range(1, 4):
            if not self.price + i in lob.ask:
                lob.ask[self.price + i] = 0

        for price, volume in lob.ask.items()[:]:
            if not (price >= self.price and price <= self.price + 3):
                del lob.ask[price]


@dataclass
class Create_Bid(Order):
    @property
    def action(self) -> str:
        return "Create_Bid"

    def apply_bid(self, lob: LimitOrderBook) -> None:
        assert (self.price not in lob.ask) and (self.price not in lob.bid)

        lob.bid[self.price] = self.size
        for k in range(1, 4):
            if not self.price - k in lob.bid:
                lob.bid[self.price - k] = 0

        for price, volume in lob.bid.items()[:]:
            if not (price <= self.price and price >= self.price - 3):
                del lob.bid[price]

    def apply_ask(self, lob: LimitOrderBook) -> None:
        raise NotImplementedError
