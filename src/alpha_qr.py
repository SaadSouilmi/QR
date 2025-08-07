import numpy as np

from .alpha import Alpha
from .buffer import Buffer
from .matching_engine import MatchingEngine, OrderQueue
from .orderbook import Add, LimitOrderBook, Order, Side, Trade
from .queue_reactive import QRModel
from .race import ExternalEvent, Race
from .trader import Trader
from .utils import copy_lob


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


class AQR:

    def __init__(
        self,
        qr_model: QRModel,
        alpha: Alpha,
        matching_engine: MatchingEngine,
        race_model: Race,
        external_event_model: ExternalEvent,
        trader: Trader,
        rng: np.random.Generator = np.random.default_rng(1337),
    ) -> None:
        self.qr_model = qr_model
        self.alpha = alpha
        self.matching_engine = matching_engine
        self.race_model = race_model
        self.external_event_model = external_event_model
        self.trader = trader
        self.order_queue = OrderQueue()
        self.rng = rng

    def sample(self, lob: LimitOrderBook, max_ts: int, buffer: Buffer) -> None:
        # Initialise state
        self.matching_engine.lob = copy_lob(lob)
        self.order_queue.clear()
        self.alpha.eps = None
        self.alpha.initialise(lob)
        self.trader.curr_pos = 0
        curr_ts, sequence = 0, 0
        t_external_event = self.external_event_model.sample_deltat()
        t_alpha = self.alpha.sample_jump()

        while curr_ts <= max_ts:
            t_reg = (
                self.qr_model.sample_deltat(lob)
                if not self.order_queue.has_regular_order
                else float("inf")
            )
            t1 = self.order_queue.dt

            sample_regular_order = (
                curr_ts + t_reg < t_alpha
                and curr_ts + t_reg < t_external_event
                and curr_ts + t_reg < t1
                and not self.order_queue.has_regular_order
            )
            alpha_jump = t_alpha < t_external_event and t_alpha < t1
            external_race = t_external_event < t1

            if sample_regular_order:
                curr_ts += t_reg 
                reg_order = self.qr_model.sample_order(lob, self.alpha)
                self.matching_engine.process_regular_order(reg_order, curr_ts)
                self.order_queue.append_order(reg_order, regular_order=True)
                continue 
            elif alpha_jump:
                curr_ts = t_alpha
                self.alpha.update_eps()
                self.alpha.compute_value()
                self.qr_model.adjust_event_distrib(self.alpha.eps, exclude_cancels=False)
                t_alpha += self.alpha.sample_jump()
            elif external_race:
                curr_ts = t_external_event
                p_external_event = self.external_event_model.probability(lob)
                if self.rng.uniform() < p_external_event:
                    external_event_orders = self.external_event_model.sample_orders(lob, self.alpha.value)
                    self.matching_engine.process_race(external_event_orders, curr_ts)
                    self.order_queue.append_race(external_event_orders)
                t_external_event += self.external_event_model.sample_deltat()
                continue
            else:
                curr_ts = t1
                pending_order = self.order_queue.pop_order()
                if pending_order.action == "Trd_Add":
                    market_order, limit_order = process_marketable_limit_order(
                        lob, pending_order
                    )

                    market_order = lob.process_order(market_order)
                    buffer.record(lob, self.alpha, market_order, self.trader, sequence)
                    sequence += 1
                    limit_order = lob.process_order(limit_order)
                    buffer.record(lob, self.alpha, limit_order, self.trader, sequence)
                    sequence += 1
                    if limit_order.trader_id == 0 and not limit_order.rejected:
                        self.trader.can_trade += 1
                else:
                    pending_order = lob.process_order(pending_order)
                    buffer.record(lob, self.alpha, pending_order, self.trader, sequence)
                    sequence += 1
                    if pending_order.trader_id == 0 and not pending_order.rejected:
                        self.trader.can_trade += 1
                    if pending_order.trader_id == 1 and not pending_order.rejected:
                        self.trader.curr_pos += (
                            pending_order.size if pending_order.side == Side.A else -pending_order.size
                        )
            
                self.alpha.update_imbalance(lob)
                self.alpha.compute_value()
            
            race_orders = []

            p_trader = self.trader.probability(lob, self.alpha) * (self.order_queue.trader_order_count == 0)
            if self.rng.uniform() < p_trader:
                order = self.trader.order(lob, self.alpha)
                race_orders.append(order)
                self.trader.can_trade = -self.trader.cooldown + 1

            p_race = self.race_model.probability(lob, self.alpha)
            if self.rng.uniform() < p_race:
                orders = self.race_model.sample_race(lob, self.alpha)
                race_orders.extend(orders)

            if len(race_orders) > 0:
                shuffled_race_orders = self.rng.permutation(race_orders).tolist()
                self.matching_engine.process_race(shuffled_race_orders, curr_ts)
                self.order_queue.append_race(shuffled_race_orders)
