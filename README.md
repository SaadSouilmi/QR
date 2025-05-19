# QR

The only non standard dependency that I use is [sortedcontainer](https://grantjenks.com/docs/sortedcontainers/). I tried to only include the AlphaQR sampling part of my package to avoid bloating this repo with irrelevant stuff. All the sampling logic is in `src/alpha_queue_reactive`. The other modules I use for parameter estimation and whatnot so they contain some relative imports beyond this repo. `src/alpha_queue_reactive` is self contained though.


I handle randomness by excplicitly passing a `np.random.Generator` object whenever that's needed. In my implementation the `LimitOrderBook` takes in an `rng` in order to sample from some invariant distribution when a new queue is discovered after a shift in the mid price. `LimitOrderBook` should have it's own `rng`, the other objects can share the same `rng` it won't hinder the statistical integrity of the sampling.

The main sampler is in the `src/alpha_queue_reactive/alpha_qr.py`
```python 

def sample(self, lob: LimitOrderBook, max_ts: int, buffer: Buffer) -> None:
    # Initialise state
    self.matching_engine.lob = copy_lob(lob)
    self.order_queue.clear()
    self.alpha.initialise(lob)
    curr_ts = 0
    ... 
```

The `Buffer` object records all the events. I pass it explicitly to `sample` instead of just creating a clean buffer in the initialisation step and returning that, that way I can get intermediate results in case the sampling crashes or I need to interrupt it for some reason.

The matching engine has it\`s own `LimitOrderBook`. I created a little utility to copy the orderbook since `copy.deepcopy` doesn't actually create a new `rng` object and we want both `lob` and `matching_engine.lob` to have two distinct `rng`s that start with the same entropy, that way the market orderbook can mirror the matching-engine's book.


```python
    ...
    while curr_ts <= max_ts:
        t_alpha, jump_value = self.alpha.sample_jump()
        t_reg, order_reg = float("inf"), None
        p_race = self.race_model.probability(lob, self.alpha.value)
        p_trader = self.trader.probability(lob, self.alpha.value)
    ...
```

`t_alpha` and `jump_value`  are jumps in alpha unrelated to the orderbook. These jumps would trigger the indepenant races, still need to include that. Also for `p_trader` I should probably add the current position of the trader as a parameter for it's trading probability.

```python 
        ...
        if self.rng.uniform() < p_trader:
            order = self.trader.order(lob, self.alpha.value)
            order = self.matching_engine.process_regular_order(order, curr_ts)
            self.order_queue.append_order(order)
            
        # We make the decision to trigger a race or not to
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
        ...
```
If the trader decides to send their order, it takes precedence to races/regular orders. I am not sure how to incorporate randomness in whether people involved in races may be faster than our trader. Also I kinda have to include a trader in the sampling, if I actually don't want one we can just set the `probability` to zero. 

```python 
        ...
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
        ...
```
If the `order_queue` is empty, that would first imply that there is no race and the trader did not send an order. And also it the matching-engine's orderbook and that of the market should be in sync, I added an `assert` as a sanity check there.


```python 
        ...
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
```

And finally here, in case there is an order in the queue I execute it against the market's orderbook in case it's received before the pending regular order/alpha jump. Otherwise I just go on with whatever event comes first. One thing that I added recently is the `self.order_queue.has_regular_order` check which helps me ensure the queue has only one regular order at a time. It is a bit of a costly check and adds quite a bit of overhead but it was an easy fix to a situation where I would append to the order queue an order that is received by the market before the order at the far left(top) of the order queue. Another fix would be to make the order queue a `SortedDict` where the key is the `dt` the time an order hits the market, I should probably try that.

I haven't written an extensive documentation or docstrings yet, but I hope this little explanation helps a bit. I have included a notebook with a little example.
