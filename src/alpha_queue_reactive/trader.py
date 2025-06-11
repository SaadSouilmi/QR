from dataclasses import dataclass

from .orderbook import Trade, LimitOrderBook, Side


@dataclass
class Trader:
    trader_id: int
    max_spread: int
    max_volume: int
    alpha_threshold: float
    max_pos: int
    curr_pos: int = 0
    probability_: float = 0.25
    cooldown: int = 10
    can_trade: int = 1

    def send_order(self, lob: LimitOrderBook, alpha: float) -> bool:
        send_ask = alpha > self.alpha_threshold and self.curr_pos < self.max_pos and self.can_trade
        send_bid = alpha < - self.alpha_threshold and self.curr_pos > -self.max_pos and self.can_trade
        return (
            lob.spread <= self.max_spread
            and (send_ask or send_bid)
        )

    def probability(self, lob: LimitOrderBook, alpha: float) -> float:
        return self.send_order(lob, alpha) * self.probability_

    def order(self, lob: LimitOrderBook, alpha: float) -> Trade:
        side = Side.A if alpha > 0 else Side.B
        price = lob.best_ask_price if alpha > 0 else lob.best_bid_price
        remaining_pos = self.max_pos - self.curr_pos if side == Side.A else self.max_pos + self.curr_pos
        size = min(self.max_volume, remaining_pos)
        return Trade(
            side=side,
            price=price,
            size=size,
            imbalance=lob.imbalance,
            spread=lob.spread,
            alpha=alpha,
            ask=lob.ask,
            bid=lob.bid,
            trader_id=self.trader_id,
        )
