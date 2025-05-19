from dataclasses import dataclass

from .orderbook import Trade, LimitOrderBook, Side


@dataclass
class Trader:
    trader_id: int
    max_spread: int
    max_volume: int
    alpha_threshold: float
    probability_: float = 0.25

    def send_order(self, lob: LimitOrderBook, alpha: float) -> bool:
        return abs(alpha) > self.alpha_threshold and lob.spread <= self.max_spread
    
    def probability(self, lob: LimitOrderBook, alpha: float) -> float:
        return self.send_order(lob, alpha) * self.probability_
    
    def order(self, lob: LimitOrderBook, alpha: float) -> Trade:
        return Trade(
            side=Side.A if alpha > 0 else Side.B,
            price=lob.best_ask_price if alpha > 0 else lob.best_bid_price,
            size=self.max_volume,
            imbalance=lob.imbalance,
            spread=lob.spread,
            alpha=alpha,
            ask=lob.ask,
            bid=lob.bid,
            trader_id=self.trader_id,
        )