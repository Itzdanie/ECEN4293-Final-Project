"""How much Tier 2 changes if ROUND_WIN_RATE is used as labeled - (attacker's
buy, defender's buy) - instead of as (team A's buy, team B's buy), which is
what tier2/states.py does today (run from the repo root):

    python scripts/check_tier2_side.py

Reuses Tier 2's own states, transitions and Gauss-Seidel solver; only the
round-win probability is swapped per state, by rescaling each branch.
Result on 2026-11-24: mean change 0.05-0.18 pp, max about 1 pp, and 0-0
goes from 50.27% to exactly 50.00%. Model-only: needs no raw data.
"""

import numpy as np

from winprob.numerics.gauss_seidel import gauss_seidel
from winprob.tier1.states import side_for_team_a
from winprob.tier2.economy_data import ROUND_WIN_RATE as R
from winprob.tier2.exact import solve_tier2, win_probability_from_start
from winprob.tier2.states import all_states, boundary_value, equation_terms, is_pistol_next


def p_current(state):
    a, b, buy_a, buy_b, _, _ = state
    return 0.5 if is_pistol_next(a, b) else R[(buy_a, buy_b)]


def p_side_aware(state, a_started):
    a, b, buy_a, buy_b, _, _ = state
    if is_pistol_next(a, b):
        return 0.5
    if side_for_team_a(a + b, a_started) == "atk":
        return R[(buy_a, buy_b)]
    return 1.0 - R[(buy_b, buy_a)]


def solve_side_aware(a_started):
    states = all_states()
    index = {s: i for i, s in enumerate(states)}
    neighbors = [[] for _ in states]
    constants = np.zeros(len(states))
    for s, i in index.items():
        old, new = p_current(s), p_side_aware(s, a_started)
        for nxt, w in equation_terms(s):
            w *= new / old if nxt[0] == s[0] + 1 else (1 - new) / (1 - old)
            bv = boundary_value(nxt[0], nxt[1])
            if bv is not None:
                constants[i] += w * bv
            else:
                neighbors[i].append((index[nxt], w))
    sol, _ = gauss_seidel(neighbors, constants, tol=1e-10, max_iter=1000)
    return {s: float(sol[i]) for s, i in index.items()}


current, _ = solve_tier2(tol=1e-10, max_iter=1000)

print("Per-round gap: R[(x,y)] vs 1 - R[(y,x)] (how asymmetric the table is)")
for (x, y), r in R.items():
    if x < y:
        print(f"  {x:>8} vs {y:<8}  {r:.3f} vs {1 - R[(y, x)]:.3f}  diff {100 * (r - (1 - R[(y, x)])):+.1f} pp")

for a_started in ("atk", "def"):
    fixed = solve_side_aware(a_started)
    diffs = np.array([fixed[s] - current[s] for s in current])
    worst = max(current, key=lambda s: abs(fixed[s] - current[s]))
    print(f"\nYou started on {a_started}:")
    print(f"  from 0-0:        {100 * win_probability_from_start(current):.2f}% -> "
          f"{100 * win_probability_from_start(fixed):.2f}%")
    print(f"  mean |diff|:     {100 * np.abs(diffs).mean():.3f} pp over {len(diffs):,} states")
    print(f"  max |diff|:      {100 * np.abs(diffs).max():.3f} pp at {worst}")
    for s in [(5, 6, "Full buy", "Full buy", 0, 0), (5, 5, "Full buy", "Eco", 0, 2),
              (9, 9, "Semi-buy", "Full buy", 1, 0), (11, 11, "Full buy", "Semi-eco", 0, 1)]:
        print(f"  {s}: {100 * current[s]:.2f}% -> {100 * fixed[s]:.2f}%")
