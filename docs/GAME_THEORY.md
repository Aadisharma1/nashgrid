# 05 — Game Theory: Formal Model and Derivations

The mathematics behind NashGrid: the matrix game, the LP that solves it
exactly, the best-response attackers, the minimax machinery, and the Value of
Information definition. Everything here is implemented in `src/nashgrid/game/`
and pinned by tests against games with known closed-form solutions.

## 1. The site-based security game

One episode reduces to a simultaneous-move game over `n` protected targets:

- **Defender** picks a target `d` to patrol.
- **Attacker** picks a target `a` to attack.
- Payoffs (defender's perspective):

```
A[d, a] = +R        if d == a   (attack intercepted; R = intercept reward)
A[d, a] = -V[a]     if d != a   (attacker steals target a's value V[a])
```

This is a zero-sum game: the attacker's payoff is exactly `-A[d, a]`. It is
strictly competitive, so von Neumann's minimax theorem applies: there exist
mixed strategies `p*` (defender) and `q*` (attacker) and a value `v` such that

```
max_p min_q E[A] = min_q max_p E[A] = v
```

`build_payoff_matrix` constructs `A` from the target values, and
`solve_zero_sum` computes `p*, q*, v` exactly. The test suite pins the solver
on two textbook games — matching pennies (value 0, uniform 1/2 strategy) and
rock-paper-scissors (value 0, uniform 1/3 strategy) — plus a dominant-column
case with a known pure equilibrium.

## 2. The LP formulation, derived

For the row (defender) player, we want `p` and the largest `v` such that every
pure attacker column yields at least `v`:

```
maximize   v
subject to sum_j A[i, j] * p_j >= v     for every defender pure strategy i
           sum_j p_j = 1,  p_j >= 0
```

Substituting `x = (p_1..p_m, v)` and negating for scipy's minimization form:

```
minimize   c^T x,  c = (0, ..., 0, -1)
subject to -A^T p + v <= 0        (one row per defender pure strategy)
           1^T p = 1
           p >= 0,  v free
```

That is exactly what `solve_zero_sum` builds: `A_ub = hstack([-A.T, ones((n,1))])`,
`A_eq = [1...1, 0]`, `bounds = (0, None) * m + (None, None)`, solved with
scipy's HiGHS. No external solver, no approximation scheme whose convergence
would need its own test — the LP is exact, so the tests can assert equality
with closed-form equilibria up to solver tolerance (1e-6).

**Why the equilibrium has full support here.** For this diagonal-reward
matrix, the attacker is indifferent across columns when the defender's mix
satisfies `p_a(R + V_a) - V_a = v` for every `a`, i.e.

```
p_a = (V_a + v) / (R + V_a) > 0
```

so every target is patrolled with strictly positive probability in equilibrium
— the mathematical reason randomized defense is *not* exploitable the way a
deterministic patrol is: there is no target the attacker can raid with zero
defensive mass. (Whether that defensive mass intercepts in the *spatial*
episode depends on timing — see §5.)

## 3. Attacker strategies and their assumptions

| Strategy | Decision rule | Information assumed |
|---|---|---|
| `RandomAttacker` | uniform target | none |
| `GreedyAttacker` | argmax `V[a] · (1 - p[a])` | knows the defender's *distribution*, not the sampled patrol |
| `BestResponseAttacker` | argmin over columns of `p^T A` | same distributional knowledge, exact payoff-aware version |

The greedy and best-response attackers are deliberately given the defender's
mixed strategy — the standard "worst case for the defender" assumption in
security games. This is what makes the measured comparison between
deterministic and mixed defenders meaningful: against an opponent that knows
your *routine*, randomization is the only remaining defense; against an
opponent that knows your *distribution*, equilibrium randomization is optimal.

## 4. Minimax and alpha-beta

`game/minimax.py` implements depth-limited minimax with alpha-beta pruning
over an explicit game tree (`TreeState`), with both max and min plies. The
tests verify: a hand-built two-ply tree whose value is computable by eye, and
equality with brute-force minimax on randomly generated trees (the pruning
must never change the value, only the work). This module is the correct hook
for future *sequential* spatial games (defender commits, attacker observes and
responds); the current matrix game is simultaneous, which is the standard
first-order abstraction.

## 5. The spatial episode layer

The matrix game is the strategic abstraction; `core/episode.py` realizes one
episode on the grid:

- The attacker spawns 2 cells inside from its target's border cell and walks a
  BFS shortest path (obstacles respected).
- The defender starts at the *next* target over (cyclically) and walks its BFS
  path to its sampled patrol target, then holds.
- **Interception** occurs if the defender ever occupies the attacker's current
  cell — which, given both funnel to the target cell, effectively means the
  defender must arrive no later than the attacker.
- Speeds: the defender moves 2 cells/step by default versus the attacker's 1.
  This asymmetry is load-bearing: with equal speeds the attacker's entry path
  is essentially always shorter than any inter-target patrol path, interception
  becomes near-impossible, and every strategy comparison degenerates to
  "attacker always wins". The 2:1 ratio represents patrol units covering open
  ground faster than an infiltrator moving between cover. It is a modeling
  choice, documented here and in the docstring, not a hidden fudge factor.

Monte Carlo statistics (`simulation/engine.py`) run N seeded episodes per
strategy pairing and report success/interception rates and mean payoffs with
normal-approximation 95% CIs. Same seed → same statistics, asserted by test.

## 6. Value of Information

VoI asks: what is it worth to the attacker to learn the defender's *realized*
patrol target (not just the distribution) before committing? The sweep
(`analysis/voi.py`) measures attacker payoff at observation probability
`p_obs ∈ [0, 1]`:

- With probability `p_obs`: the attacker observes this episode's sampled patrol
  target and attacks the highest-value target the defender is *not* on.
- Otherwise: best response to the distribution only.

Because the observation improves the attacker's decision at no cost, the
measured curve must be (weakly) monotone in `p_obs` on any fixed
configuration — the tests assert the endpoints and the weak-monotonicity
direction as *measured* properties of the constructed scenario, not as
hardcoded values. The value of information is the vertical gap between the
`p_obs = 0` and `p_obs = 1` ends of the curve, reported in
`outputs/experiments.json` and rendered as `outputs/voi.png`.

## 7. Known simplifications

- One attack per episode; no multi-target campaigns or sequential raids.
- Perfect BFS navigation; no attacker confusion or waypoints.
- The defender's patrol target is sampled fresh each episode (no persistence
  or schedule-predictability modeling beyond the deterministic strategy).
- Payoffs are site-based constants; no partial damage or time-discounting.
- The LP solves the simultaneous-move abstraction; the spatial timing layer is
  simulated, not solved — mixed-strategy *spatial* equilibrium would require a
  different (and much larger) solution concept.
