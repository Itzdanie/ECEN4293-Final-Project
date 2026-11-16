"""Check Tier 3's ult model against real data it was NOT calibrated on
(run from the repo root; needs the gitignored raw CSVs):

    python scripts/validate_ult_model.py

The habits were calibrated on round-to-round transitions (who becomes
ready next round, who uses a ready ult). Here each real team composition is
simulated through a regulation map on ult state alone, and the *shape* of
the ready-by-round curve and each agent's ready share are compared with
(a) the real per-round curve and (b) chiro.gg's separate aggregate table,
``ult_per_round_by_agent.csv``. Charging is simulated independently of who
wins each round (the model has no kills), so some mismatch is expected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from winprob.data.loading import load_maps, load_ult_per_round_by_agent, load_ult_ready
from winprob.tier3.ult_model import SLOTS, slot_params, step_ults
from winprob.ults.prep import clean_ult_rows

REPS = 200
ROUNDS = 24


def main() -> None:
    rows = clean_ult_rows(load_ult_ready(), load_maps())
    first = rows[rows["round"] == 1].sort_values(["match_id", "team", "slot"])
    comps = first.groupby(["match_id", "team"])["agent"].apply(list)
    comps = comps[comps.map(len) == SLOTS]
    print(f"{len(comps)} real team compositions")

    rng = np.random.default_rng(0)
    ready_by_round = np.zeros(ROUNDS)
    per_agent_ready = {}
    per_agent_n = {}
    for agents in comps:
        params = slot_params(agents)
        ready = np.zeros((REPS, SLOTS), dtype=bool)
        age = np.zeros((REPS, SLOTS), dtype=np.int64)
        post = np.zeros((REPS, SLOTS), dtype=bool)
        for r in range(1, ROUNDS + 1):
            if r in (1, 13):
                ready[:] = False
                age[:] = 0
                post[:] = False
            ready_by_round[r - 1] += ready.mean(axis=0).sum() / SLOTS
            for s, a in enumerate(agents):
                if r <= 23 and r != 12:
                    per_agent_ready[a] = per_agent_ready.get(a, 0.0) + ready[:, s].mean()
                    per_agent_n[a] = per_agent_n.get(a, 0) + 1
            _, ready, age, post = step_ults(rng, ready, age, post, params)
    sim_curve = ready_by_round / len(comps)

    real = rows.groupby("round")["ready"].mean()
    print("\nready share by round (real vs simulated), rounds 12/24 excluded from the real data")
    for r in range(1, ROUNDS + 1):
        if r in real.index:
            print(f"  {r:>2}: real {real[r]:.3f}  sim {sim_curve[r - 1]:.3f}")

    table = load_ult_per_round_by_agent().set_index("agent")
    table.index = table.index.str.lower()
    print("\nper-agent ready share (sim vs chiro.gg ready_pct, an independent source)")
    diffs = []
    for a in sorted(per_agent_ready):
        sim = 100 * per_agent_ready[a] / per_agent_n[a]
        if a in table.index:
            diffs.append(sim - table.loc[a, "ready_pct"])
            print(f"  {a:<10} sim {sim:5.1f}  chiro.gg {table.loc[a, 'ready_pct']:5.1f}")
    print(f"mean difference {np.mean(diffs):+.1f} pp, mean abs {np.mean(np.abs(diffs)):.1f} pp")


if __name__ == "__main__":
    main()
