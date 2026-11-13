"""Tier 3's ultimate model: a simplified fallback per CLAUDE.md's
"Open decision: ultimates" (Daniel's call - fit the real regression without
an ults term, set Tier 3's ult effect from ``lift_pp``). This is NOT a
full per-round reconstruction from kills/plants/orbs - that's the optional
stretch goal, built only if time remains.

Tracks each team's count of players with a ready ult (0-5), from
``ult_per_round_by_agent.csv``'s rounds-weighted averages across agents:
~4.8 rounds to charge, ~57.2% of ready ults used per round they're ready,
and a lift of ~+3.8 percentage points in round-win rate when an ult is
used (an *upper bound* on the causal effect - teams choose when to ult,
so this is association, not causation; CLAUDE.md estimates this as
roughly +0.15 on the logit scale near 50%).
"""

from __future__ import annotations

import numpy as np

CHARGE_ROUNDS = 4.8
CHARGE_PROB_PER_ROUND = 1.0 / CHARGE_ROUNDS
USE_PROB_WHEN_READY = 0.572
LIFT_PP = 3.8
MAX_LOGIT_BOOST = 0.15  # the lift_pp upper bound, expressed on the logit scale
MAX_ULTS_READY = 5


def step_ults(rng: np.random.Generator, ults_ready: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One round of using, then charging, ults. Returns ``(used, new_ults_ready)``.

    Only ults that were already ready *entering* this round can be used in
    it - with probability ``1 - (1 - USE_PROB_WHEN_READY) ** ults_ready``
    (at least one of them gets used), consuming one. Afterward, each of a
    team's not-yet-ready player slots independently charges with
    probability ``CHARGE_PROB_PER_ROUND``, becoming usable from next round
    on (an ult can't charge and be used in the same round-step - it needs
    to have been ready coming in).

    This simple two-state-per-slot model doesn't simultaneously reproduce
    all three real aggregate stats - its steady-state "ready" fraction
    comes out higher than the real ~15% ``ready_pct``. It was tuned to
    match ``CHARGE_ROUNDS`` and ``USE_PROB_WHEN_READY`` directly, since
    those are the two inputs that actually drive Tier 3's win-probability
    effect; exactly reproducing ``ready_pct`` isn't needed here.
    """
    p_use_at_least_one = 1.0 - (1.0 - USE_PROB_WHEN_READY) ** ults_ready
    used = (ults_ready > 0) & (rng.random(len(ults_ready)) < p_use_at_least_one)
    ults_ready = np.where(used, ults_ready - 1, ults_ready)

    not_ready = MAX_ULTS_READY - ults_ready
    newly_charged = rng.binomial(not_ready, CHARGE_PROB_PER_ROUND)
    ults_ready = np.minimum(ults_ready + newly_charged, MAX_ULTS_READY)

    return used, ults_ready
