import numpy as np

from .orderbook import LimitOrderBook
from .matching_engine import OrderQueue, MatchingEngine
from .alpha import Alpha
from .buffer import Buffer
from .queue_reactive import QRModel
from .race import Race
from .trader import Trader
from .utils import copy_lob


class AQR:

    def __init__(
        self,
        qr_model: QRModel,
        alpha: Alpha,
        matching_engine: MatchingEngine,
        race_model: Race,
        trader: Trader,
        rng: np.random.Generator = np.random.default_rng(1337),
    ) -> None:
        self.qr_model = qr_model
        self.alpha = alpha
        self.matching_engine = matching_engine
        self.race_model = race_model
        self.trader = trader
        self.order_queue = OrderQueue()
        self.rng = rng

    def sample(self, lob: LimitOrderBook, max_ts: int, buffer: Buffer) -> None:
        # Initialise state
        self.matching_engine.lob = copy_lob(lob)
        self.order_queue.clear()
        self.alpha.initialise(lob)
        curr_ts = 0

        while curr_ts <= max_ts:
            t_alpha, jump_value = self.alpha.sample_jump()
            t_reg, order_reg = float("inf"), None
            p_race = self.race_model.probability(lob, self.alpha.value)
            p_trader = self.trader.probability(lob, self.alpha.value)

            # We make the decision to trigger a race or not to
            if self.rng.uniform() < p_trader:
                order = self.trader.order(lob, self.alpha.value)
                order = self.matching_engine.process_regular_order(order, curr_ts)
                self.order_queue.append_order(order)
                
            if self.rng.uniform() < p_race:
                orders = self.race_model.sample_race(lob, self.alpha.value)
                orders = self.matching_engine.process_race(orders, curr_ts)
                self.order_queue.append_order(orders)

            # If no race sample a regular order
            else:
                if self.order_queue.has_regular_order:
                    continue
                t_reg = self.qr_model.sample_deltat(lob)
                order_reg = self.qr_model.sample_order(lob, self.alpha.value)

            # If the order queue is empty
            if self.order_queue.empty:
                assert (
                    lob.bid == self.matching_engine.lob.bid
                    and lob.ask == self.matching_engine.lob.ask
                ), "Lob and ME do not match"
                if t_alpha < t_reg:
                    curr_ts += t_alpha
                    orders = self.race_model.sample_race(lob, self.alpha.value)
                    orders = self.matching_engine.process_race(orders, curr_ts)
                    self.order_queue.append_order(orders)
                else:
                    order_reg = self.matching_engine.process_regular_order(
                        order_reg, curr_ts + t_reg
                    )
                    curr_ts += t_reg
                    self.order_queue.append_order(order_reg)
                # Reset the alpha and reg times
                t_alpha, t_reg = float("inf"), float("inf")

            t1 = self.order_queue.dt
            if t1 <= curr_ts + t_reg and t1 <= curr_ts + t_alpha:
                order = self.order_queue.pop_order()
                buffer.record(lob, self.alpha, order)
                # if not order.rejected:
                lob.process_order(order)
                self.alpha.sample_value(lob)
                curr_ts = t1
            elif t_alpha < t_reg:
                curr_ts += t_alpha
                self.alpha.value = jump_value
                orders = self.race_model.sample_race(lob, self.alpha.value)
                orders = self.matching_engine.process_race(orders, curr_ts)
                self.order_queue.append_order(orders)
            else:
                if self.order_queue.has_regular_order:
                    continue 
                order_reg = self.matching_engine.process_regular_order(
                    order_reg, curr_ts + t_reg
                )
                curr_ts += t_reg
                self.order_queue.append_order(order_reg)
