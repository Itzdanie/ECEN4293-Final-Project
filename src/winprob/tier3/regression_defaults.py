"""The Phase 5 round-win model's real fitted coefficients, hardcoded here
so Tier 3 doesn't need to refit on every run (same pattern as Tier 2's
``economy_data`` module).

From ``feature/logistic-regression``'s real-data fit (22,861 rounds,
league + international tier): delta_loadout (per 1000 credits of attacker
advantage) and half2 are map-independent. BASE_INTERCEPT is the raw fitted
intercept relative to "abyss", the reference map that fit dropped - which
turned out to be the single most attacker-favored map of the 12 in the
pool. MAP_EFFECTS gives every map's effect relative to that same abyss
reference (0.0 for abyss itself), straight from the fit, with round
counts from the training data to weight them.

When no specific map is given, Tier 3 falls back to POOL_AVERAGE_MAP_EFFECT
- the real round-count-weighted average across all 12 maps - rather than
silently defaulting to abyss specifically (an earlier version of this
module did exactly that, by omitting map effects entirely while assuming
that was equivalent to an "average" map; it wasn't).
"""

from __future__ import annotations

BASE_INTERCEPT = 0.1279033760495436
DELTA_LOADOUT_COEF = 0.1046087895064958  # per 1000 credits of attacker loadout advantage
HALF2_COEF = 0.008768207764801673

# Each map's effect relative to abyss (the fit's reference level, 0.0 here).
MAP_EFFECTS = {
    "abyss": 0.0,
    "ascent": -0.22896147844930623,
    "bind": -0.011109831638847073,
    "breeze": -0.031847737484890615,
    "corrode": -0.05996534627517346,
    "fracture": 0.056148486622991836,
    "haven": -0.08639487910086907,
    "lotus": -0.07869111936385016,
    "pearl": -0.1949116044377083,
    "split": -0.10318833620409079,
    "summit": -0.15181256358166687,
    "sunset": -0.1763950361119895,
}

# Real round counts per map in the training data (same fit), for the
# pool-average weighting.
_MAP_ROUND_COUNTS = {
    "haven": 3497,
    "split": 3490,
    "lotus": 2821,
    "breeze": 2642,
    "ascent": 1878,
    "pearl": 1667,
    "sunset": 1606,
    "abyss": 1346,
    "summit": 1295,
    "bind": 1210,
    "fracture": 749,
    "corrode": 660,
}
_TOTAL_ROUNDS = sum(_MAP_ROUND_COUNTS.values())

POOL_AVERAGE_MAP_EFFECT = sum(
    (_MAP_ROUND_COUNTS[m] / _TOTAL_ROUNDS) * MAP_EFFECTS[m] for m in MAP_EFFECTS
)
