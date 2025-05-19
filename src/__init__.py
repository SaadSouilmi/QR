r"""
Queue-Reactive (QR) model for order book dynamics.

At any time :math:`t`, the order book :math:`\text{LOB}(t)` is projected onto a state
:math:`f(\text{LOB}) = (\text{Imb}, n)` where:

.. math::

    \text{Imb} = \frac{Q^B - Q^A}{Q^B + Q^A}

and :math:`n` is the number of empty price levels between the best bid and best ask.

The possible events occurring on the order book are reduced to:

.. math::

    \mathcal{E} := \{L, C, T\} \times \{B_\ell, A_\ell\}^{1 \leq \ell \leq \overline{L}}
    \cup \{+, -\} \times [\![ -N, N ]\!].

Here, :math:`\{L, C, T\}` are limit orders, cancel orders, and transactions either on
:math:`B_\ell` or :math:`A_\ell` (the :math:`\ell`-th bid or ask level).
Another type of event is a marketable order appearing inside the spread:
buying (:math:`+`) or selling (:math:`-`) at a distance :math:`n \in [\![ -N, N ]\!]`
from the mid-price.

The QR model is defined by the choice of :math:`f(\cdot)` and :math:`\mathcal{E}`.
It assumes that events in :math:`\mathcal{E}` occur as conditionally independent Poisson processes
given the state :math:`f(\text{LOB})`. The intensity is denoted:

.. math::

    \lambda^e(f(\text{LOB}), \ell(e)) \in \mathbb{R}^+, \quad e \in \mathcal{E}

and, in our case:

.. math::

    \lambda^e(f(\text{LOB}), \ell(e)) = \lambda^e(\text{Imb}, n; \ell(e)).

The total intensity over all possible events in a state :math:`(\text{Imb}, n)` is:

.. math::

    \Lambda(\text{Imb}, n) = \sum_{e \in \mathcal{E}} \lambda^e(\text{Imb}, n; \ell(e)).

We define :math:`t^e_k` as the stopping times for event :math:`e`, and:

.. math::

    \Delta t^e_k = t^e_k - t^{e'}_{k-1} \in \mathcal{F}_k,

where :math:`e'` is the event just before :math:`e`. Let :math:`\mathcal{T}` be the set of all stopping times.

The function :math:`\ell(e)` maps an event to a queue index:
- If :math:`e \in \{L, C, T\} \times \{B_\ell, A_\ell\}`, then :math:`\ell(e) = \ell` if on ask side,
  and :math:`\ell(e) = -\ell` otherwise.
- If :math:`e \in \{+, -\} \times [\![ -N, N ]\!]`, then :math:`\ell(e) = 0` if :math:`n = 0`,
  and :math:`\ell(e) = 1/n` otherwise.

Let :math:`\mathcal{K}^e(\text{Imb}, n)` be the set of indices :math:`k` such that an event :math:`e` occurred at
time :math:`t^e_k` and the previous state was :math:`(\text{Imb}, n)`:

.. math::

    \mathcal{K}^e(\text{Imb},n) = \left\{ k : t^e_k \in \mathcal{T}, e' = e,
    \exists e'': f(\text{LOB}(t^{e''}_{k-1})) = (\text{Imb}, n) \right\}.

Let :math:`\mathcal{K}(\text{Imb}, n) = \bigcup_e \mathcal{K}^e(\text{Imb}, n)`.

Estimation of event intensities:

- The event arrival time conditional on state :math:`(\text{Imb}, n)` is:

.. math::

    \Delta t^e_k \mid k \in \mathcal{K}(\text{Imb}, n) \sim \mathrm{Exp}(\Lambda(\text{Imb}, n)),

since it is the minimum among:

.. math::

    \Delta t^e \sim \min_{e' \in \mathcal{E}} \mathrm{Exp}(\lambda^{e'}(\text{Imb}, n; \ell(e))).

- The probability of observing a specific event :math:`e` is:

.. math::

    q^e = \frac{\lambda^e(\text{Imb}, n; \ell(e))}{\Lambda(\text{Imb}, n)}.

Estimates:

.. math::

    \Lambda(\text{Imb}, n) = \left( \mathbb{E}[\Delta t^e_k \mid k \in \mathcal{K}(\text{Imb}, n)] \right)^{-1},

.. math::

    \hat\Lambda(\text{Imb}, n) = \left( \frac{1}{\#\mathcal{K}(\text{Imb}, n)}
    \sum_{e \in \mathcal{E}} \sum_{k \in \mathcal{K}^e(\text{Imb}, n)} \Delta t^e_k \right)^{-1},

.. math::

    \hat\lambda^e(\text{Imb}, n; \ell(e)) = \hat\Lambda(\text{Imb}, n) \cdot
    \underbrace{\frac{\#\mathcal{K}^e(\text{Imb}, n)}{\#\mathcal{K}(\text{Imb}, n)}}_{\hat{q}^e}.

Simulation procedure:

1. Compute state: :math:`(\text{Imb}, n) = f(\text{LOB})`.
2. Sample inter-event times:

.. math::

    \forall e: \Delta \tau^e \sim \mathrm{Exp}(\lambda^e(\text{Imb}, n; \ell(e))).

3. Select event:

.. math::

    e^* = \arg\min_e \Delta \tau^e, \quad
    \Delta t^{e^*}_{k+1} = \Delta \tau^{e^*}, \quad
    t^{e^*}_{k+1} = t_k + \Delta t^{e^*}_{k+1}.

4. Update:

.. math::

    \text{LOB}(t_{k+1}) = \Psi(\text{LOB}(t_k), e^*).

Transition function :math:`\Psi(\text{LOB}, e)`:
- For :math:`e = (s, z) \in \{+, -\} \times [\![ -n, n ]\!]`:
  Add a marketable order at :math:`z` ticks from the mid-price, if spread is wide enough.
- For :math:`e = (t, s) \in \{L, C, T\} \times \{B_\ell, A_\ell\}`:
  Insert/cancel/trade on queue :math:`s`. If it empties the best queue, a new empty price level appears.

Event set :math:`\mathcal{E}|_{\text{LOB}}` depends on:
- Mapping :math:`f(\text{LOB}) = (\text{Imb}, n)`
- Best bid/ask prices :math:`(p^B, p^A)` and tick size :math:`\delta`

Eligible events:
- Always include :math:`\{L, C, T\} \times \{B_\ell, A_\ell\}`.
- Include :math:`\{+, -\} \times [\![ -n, n ]\!]` if :math:`n > 0`, with:
    - If :math:`(p^A - p^B)/\delta` even:

    .. math::

        n \in [\![ -(p^A - p^B)/(2\delta), (p^A - p^B)/(2\delta) ]\!] \setminus \{0\}

    - If odd:

    .. math::

        n \in [\![ -(p^A - p^B - 1)/(2\delta), (p^A - p^B - 1)/(2\delta) ]\!]
"""
