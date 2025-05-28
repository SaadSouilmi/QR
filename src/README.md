# Queue-Reactive (QR) Model for Order Book Dynamics

At any time $t$, the order book $\text{LOB}(t)$ is projected onto a state
$f(\text{LOB}) = (\text{Imb}, n)$ where:

$$\text{Imb} = \frac{Q^B - Q^A}{Q^B + Q^A}$$

and $n$ is the number of empty price levels between the best bid and best ask.

The possible events occurring on the order book are reduced to:

$$\mathcal{E} := \{L, C, T\} \times \{B_\ell, A_\ell\}^{1 \leq \ell \leq \overline{L}}
\cup \{+, -\} \times [\![ -N, N ]\!].$$

Here, $\{L, C, T\}$ are limit orders, cancel orders, and transactions either on
$B_\ell$ or $A_\ell$ (the $\ell$-th bid or ask level).
Another type of event is a marketable order appearing inside the spread:
buying $(+)$ or selling $(-)$ at a distance $n \in [\![ -N, N ]\!]$
from the mid-price.

The QR model is defined by the choice of $f(\cdot)$ and $\mathcal{E}$.
It assumes that events in $\mathcal{E}$ occur as conditionally independent Poisson processes
given the state $f(\text{LOB})$. The intensity is denoted:

$$\lambda^e(f(\text{LOB}), \ell(e)) \in \mathbb{R}^+, \quad e \in \mathcal{E}$$

and, in our case:

$$\lambda^e(f(\text{LOB}), \ell(e)) = \lambda^e(\text{Imb}, n; \ell(e)).$$

The total intensity over all possible events in a state $(\text{Imb}, n)$ is:

$$\Lambda(\text{Imb}, n) = \sum_{e \in \mathcal{E}} \lambda^e(\text{Imb}, n; \ell(e)).$$

We define $t^e_k$ as the stopping times for event $e$, and:

$$\Delta t^e_k = t^e_k - t^{e'}_{k-1} \in \mathcal{F}_k,$$

where $e'$ is the event just before $e$. Let $\mathcal{T}$ be the set of all stopping times.

The function $\ell(e)$ maps an event to a queue index:
- If $e \in \{L, C, T\} \times \{B_\ell, A_\ell\}$, then $\ell(e) = \ell$ if on ask side,
 and $\ell(e) = -\ell$ otherwise.
- If $e \in \{+, -\} \times [\![ -N, N ]\!]$, then $\ell(e) = 0$ if $n = 0$,
 and $\ell(e) = 1/n$ otherwise.

Let $\mathcal{K}^e(\text{Imb}, n)$ be the set of indices $k$ such that an event $e$ occurred at
time $t^e_k$ and the previous state was $(\text{Imb}, n)$:

$$\mathcal{K}^e(\text{Imb},n) = \left\{ k : t^e_k \in \mathcal{T}, e' = e,
\exists e'': f(\text{LOB}(t^{e''}_{k-1})) = (\text{Imb}, n) \right\}.$$

Let $\mathcal{K}(\text{Imb}, n) = \bigcup_e \mathcal{K}^e(\text{Imb}, n)$.

## Estimation of event intensities

- The event arrival time conditional on state $(\text{Imb}, n)$ is:

$$\Delta t^e_k \mid k \in \mathcal{K}(\text{Imb}, n) \sim \mathrm{Exp}(\Lambda(\text{Imb}, n)),$$

since it is the minimum among:

$$\Delta t^e \sim \min_{e' \in \mathcal{E}} \mathrm{Exp}(\lambda^{e'}(\text{Imb}, n; \ell(e))).$$

- The probability of observing a specific event $e$ is:

$$q^e = \frac{\lambda^e(\text{Imb}, n; \ell(e))}{\Lambda(\text{Imb}, n)}.$$

## Estimates

$$\Lambda(\text{Imb}, n) = \left( \mathbb{E}[\Delta t^e_k \mid k \in \mathcal{K}(\text{Imb}, n)] \right)^{-1},$$

$$\hat\Lambda(\text{Imb}, n) = \left( \frac{1}{\#\mathcal{K}(\text{Imb}, n)}
\sum_{e \in \mathcal{E}} \sum_{k \in \mathcal{K}^e(\text{Imb}, n)} \Delta t^e_k \right)^{-1},$$

$$\hat\lambda^e(\text{Imb}, n; \ell(e)) = \hat\Lambda(\text{Imb}, n) \cdot
\underbrace{\frac{\#\mathcal{K}^e(\text{Imb}, n)}{\#\mathcal{K}(\text{Imb}, n)}}_{\hat{q}^e}.$$

## Simulation procedure

1. Compute state: $(\text{Imb}, n) = f(\text{LOB})$.
2. Sample inter-event times:

$$\forall e: \Delta \tau^e \sim \mathrm{Exp}(\lambda^e(\text{Imb}, n; \ell(e))).$$

3. Select event:

$$e^* = \arg\min_e \Delta \tau^e, \quad
\Delta t^{e^*}_{k+1} = \Delta \tau^{e^*}, \quad
t^{e^*}_{k+1} = t_k + \Delta t^{e^*}_{k+1}.$$

4. Update:

$$\text{LOB}(t_{k+1}) = \Psi(\text{LOB}(t_k), e^*).$$

## Transition function $\Psi(\text{LOB}, e)$

- For $e = (s, z) \in \{+, -\} \times [\![ -n, n ]\!]$:
 Add a marketable order at $z$ ticks from the mid-price, if spread is wide enough.
- For $e = (t, s) \in \{L, C, T\} \times \{B_\ell, A_\ell\}$:
 Insert/cancel/trade on queue $s$. If it empties the best queue, a new empty price level appears.

## Event set $\mathcal{E}|_{\text{LOB}}$

Event set depends on:
- Mapping $f(\text{LOB}) = (\text{Imb}, n)$
- Best bid/ask prices $(p^B, p^A)$ and tick size $\delta$

### Eligible events

- Always include $\{L, C, T\} \times \{B_\ell, A_\ell\}$.
- Include $\{+, -\} \times [\![ -n, n ]\!]$ if $n > 0$, with:
   - If $(p^A - p^B)/\delta$ even:

$$n \in [\![ -(p^A - p^B)/(2\delta), (p^A - p^B)/(2\delta) ]\!] \setminus \{0\}$$

   - If odd:

$$n \in [\![ -(p^A - p^B - 1)/(2\delta), (p^A - p^B - 1)/(2\delta) ]\!]$$