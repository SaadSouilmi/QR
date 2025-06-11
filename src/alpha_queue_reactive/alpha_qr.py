import numpy as np

from .orderbook import LimitOrderBook, Side
from .matching_engine import OrderQueue, MatchingEngine
from .alpha import Alpha
from .buffer import Buffer
from .queue_reactive import QRModel
from .race import Race, ExternalEvent
from .trader import Trader
from .utils import copy_lob


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
        self.alpha.initialise(lob)
        self.trader.curr_pos = 0
        curr_ts, sequence = 0, 0
        t_external_event = self.external_event_model.sample_deltat()
        
        while curr_ts <= max_ts:
            t_alpha, jump_value = self.alpha.sample_jump()
            t_reg, order_reg = float("inf"), None
            p_race = self.race_model.probability(lob, self.alpha.value)
            p_trader = self.trader.probability(lob, self.alpha.value)

            race_orders = []
            if self.rng.uniform() < p_trader:
                order = self.trader.order(lob, self.alpha.value)
                race_orders.append(order)
                self.trader.can_trade = - self.trader.cooldown

            if self.rng.uniform() < p_race:
                orders = self.race_model.sample_race(lob, self.alpha.value)
                race_orders.extend(orders)

            if len(race_orders) > 0:
                shuffled_race_orders = self.rng.permutation(race_orders).tolist()
                shuffled_race_orders = self.matching_engine.process_race(
                    shuffled_race_orders, curr_ts
                )
                self.order_queue.append_race(shuffled_race_orders)
                if (
                    race_orders[0].trader_id == self.trader.trader_id
                    and not race_orders[0].rejected
                ):
                    self.trader.curr_pos += (
                        2 * (race_orders[0].side is Side.A) - 1
                    ) * race_orders[0].size

            if not self.order_queue.has_regular_order:
                t_reg = self.qr_model.sample_deltat(lob)
                order_reg = self.qr_model.sample_order(lob, self.alpha.value)

            # If the order queue is empty
            if self.order_queue.empty:
                assert (
                    lob.bid == self.matching_engine.lob.bid
                    and lob.ask == self.matching_engine.lob.ask
                ), "Lob and ME do not match"
                if t_alpha < t_reg and curr_ts + t_alpha < t_external_event:
                    curr_ts += t_alpha
                    self.alpha.value = jump_value
                    orders = self.alpha.race_model.sample_race(lob, self.alpha.value)
                    orders = self.matching_engine.process_race(orders, curr_ts)
                    self.order_queue.append_race(orders)
                elif t_external_event < curr_ts + t_reg:
                    curr_ts = t_external_event
                    orders = self.external_event_model.sample_order(lob, self.alpha.value)
                    orders = self.matching_engine.process_race(orders, curr_ts)
                    self.order_queue.append_race(orders)
                    t_external_event += self.external_event_model.sample_deltat()
                else:
                    curr_ts += t_reg
                    order_reg = self.matching_engine.process_regular_order(
                        order_reg, curr_ts
                    )
                    self.order_queue.append_order(order_reg, regular_order=True)

                t_alpha, t_reg = float("inf"), float("inf")

            t1 = self.order_queue.dt
            if t1 <= curr_ts + t_reg and t1 <= curr_ts + t_alpha and t1 <= t_external_event:
                order = self.order_queue.pop_order()
                buffer.record(
                    lob=lob, alpha=self.alpha, order=order, trader=self.trader, sequence=sequence
                )
                lob.process_order(order)
                sequence += 1
                if order.trader_id == 0 & order.rejected == False:
                    self.trader.can_trade += 1
                self.alpha.sample_value(lob)
                curr_ts = t1
            elif t_alpha < t_reg and curr_ts + t_alpha < t_external_event:
                curr_ts += t_alpha
                self.alpha.value = jump_value
                p_race = self.alpha.race_model.probability(lob, self.alpha.value)
                if self.rng.uniform() < p_race:
                    orders = self.alpha.race_model.sample_race(lob, self.alpha.value)
                    orders = self.matching_engine.process_race(orders, curr_ts)
                    self.order_queue.append_race(orders)
            elif t_external_event < curr_ts + t_reg:
                p_external_event = self.external_event_model.probability(lob)
                if self.rng.uniform() < p_external_event:
                    curr_ts = t_external_event
                    orders = self.external_event_model.sample_order(lob, self.alpha.value)
                    orders = self.matching_engine.process_race(orders, curr_ts)
                    self.order_queue.append_race(orders)
                t_external_event += self.external_event_model.sample_deltat()
            else:
                if not self.order_queue.has_regular_order:
                    curr_ts += t_reg
                    order_reg = self.matching_engine.process_regular_order(
                        order_reg, curr_ts
                    )
                    self.order_queue.append_order(order_reg, regular_order=True)
