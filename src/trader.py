from .orderbook import Trade, LimitOrderBook, Side
from .alpha import Alpha



class Trader:
    def __init__(
        self,
        trader_id: int,
        max_spread: int,
        max_volume: int,
        max_pos: int,
        alpha_threshold: float, 
        probability_: float = 0.25,
        cooldown: int = 10,
        can_trade: int = 1,
        full_alpha: bool = True,
    ) -> None:
        self.trader_id = trader_id
        self.max_spread = max_spread
        self.max_volume = max_volume
        self.alpha_threshold = alpha_threshold
        self.max_pos = max_pos
        self.curr_pos = 0
        self.probability_ = probability_
        self.cooldown = cooldown
        self.can_trade = can_trade
        self.full_alpha = full_alpha

    def send_order(self, lob: LimitOrderBook, alpha: float) -> bool:
        send_ask = alpha > self.alpha_threshold and self.curr_pos < self.max_pos and self.can_trade
        send_bid = alpha < - self.alpha_threshold and self.curr_pos > -self.max_pos and self.can_trade
        return (
            lob.spread <= self.max_spread
            and (send_ask or send_bid)
        )

    def probability(self, lob: LimitOrderBook, alpha: Alpha) -> float:
        if self.full_alpha:
            return self.send_order(lob, alpha.value) * self.probability_
        else:
            return self.send_order(lob, alpha.eps) * self.probability_

    def order(self, lob: LimitOrderBook, alpha: Alpha) -> Trade:
        if self.full_alpha:
            side = Side.A if alpha.value > 0 else Side.B
        else:
            side = Side.A if alpha.eps > 0 else Side.B
        price = lob.best_ask_price if side is Side.A else lob.best_bid_price

        remaining_pos = (
            self.max_pos - self.curr_pos
            if side == Side.A
            else self.max_pos + self.curr_pos
        )
        size = min(self.max_volume, remaining_pos)
        return Trade(
            side=side,
            price=price,
            size=size,
            imbalance=lob.imbalance,
            spread=lob.spread,
            alpha=alpha,
            ask_sent=dict(lob.ask),
            bid_sent=dict(lob.bid),
            trader_id=self.trader_id,
        )
