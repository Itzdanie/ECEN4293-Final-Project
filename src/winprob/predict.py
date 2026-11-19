"""One entry point for "what's each team's chance of winning this map?",
across all three tiers, with the exact/Monte Carlo label and (for Monte
Carlo) the 95% CI. Every state is validated first (Requirement 10)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from winprob import validation as v
from winprob.tier1.exact import solve_linear_system
from winprob.tier1.lookup import exact_win_probability
from winprob.tier1.monte_carlo import simulate_win_probability as tier1_mc
from winprob.tier1.states import map_winner
from winprob.tier2.exact import solve_tier2, win_probability_from_start
from winprob.tier2.monte_carlo import simulate_win_probability as tier2_mc
from winprob.tier3 import simulate_win_probability as tier3_mc
from winprob.tier3.regression_defaults import default_p_atk

_tier2_table_cache: dict | None = None


@dataclass
class Estimate:
    """One answer: ``method`` is ``"exact"`` or ``"monte_carlo"``; ``ci`` is
    the 95% interval on team A's probability (Monte Carlo only)."""

    method: str
    p_a: float
    ci: tuple[float, float] | None = None

    @property
    def p_b(self) -> float:
        return 1.0 - self.p_a


@dataclass
class Prediction:
    tier: int
    estimates: list[Estimate]

    @property
    def exact(self) -> Estimate | None:
        return next((e for e in self.estimates if e.method == "exact"), None)

    @property
    def monte_carlo(self) -> Estimate | None:
        return next((e for e in self.estimates if e.method == "monte_carlo"), None)

    @property
    def difference(self) -> float | None:
        """Monte Carlo minus exact, for compare mode."""
        if self.exact is None or self.monte_carlo is None:
            return None
        return self.monte_carlo.p_a - self.exact.p_a


def _tier2_table() -> dict:
    global _tier2_table_cache
    if _tier2_table_cache is None:
        _tier2_table_cache, _ = solve_tier2()
    return _tier2_table_cache


def _mc(p_hat_ci: tuple[float, float, float]) -> Estimate:
    p, lo, hi = p_hat_ci
    return Estimate("monte_carlo", p, (lo, hi))


def predict_tier1(
    score_a: int,
    score_b: int,
    a_started: str = "atk",
    p_atk: float | None = None,
    map_name: str | None = None,
    method: str = "exact",
    n: int = 100_000,
    seed: int = 0,
) -> Prediction:
    """``method``: ``"exact"``, ``"mc"`` or ``"both"`` (compare mode)."""
    v.validate_map(map_name)
    if p_atk is None:
        p_atk = default_p_atk(map_name)
    v.validate_tier1_state(score_a, score_b, a_started, p_atk)
    _check_method(method)

    estimates = []
    if method in ("exact", "both"):
        estimates.append(Estimate("exact", exact_win_probability(score_a, score_b, p_atk, a_started)))
    if method in ("mc", "both"):
        rng = np.random.default_rng(seed)
        estimates.append(_mc(tier1_mc(p_atk, a_started, score_a, score_b, n, rng)))
    return Prediction(1, estimates)


def predict_tier2(
    score_a: int,
    score_b: int,
    buy_a: str,
    buy_b: str,
    streak_a: int = 0,
    streak_b: int = 0,
    method: str = "exact",
    n: int = 100_000,
    seed: int = 0,
) -> Prediction:
    v.validate_tier2_state(score_a, score_b, buy_a, buy_b, streak_a, streak_b)
    _check_method(method)

    estimates = []
    if method in ("exact", "both"):
        decided = map_winner(score_a, score_b)
        if decided is not None:
            p = decided
        elif (score_a, score_b) == (0, 0):
            p = win_probability_from_start(_tier2_table())
        elif (score_a, score_b) == (12, 12):
            p = 0.5  # overtime: equal fixed credits (see tier2.states.boundary_value)
        else:
            p = _tier2_table()[(score_a, score_b, buy_a, buy_b, streak_a, streak_b)]
        estimates.append(Estimate("exact", p))
    if method in ("mc", "both"):
        rng = np.random.default_rng(seed)
        estimates.append(_mc(tier2_mc(score_a, score_b, buy_a, buy_b, streak_a, streak_b, n, rng)))
    return Prediction(2, estimates)


def predict_tier3(
    score_a: int,
    score_b: int,
    bank_a: float,
    bank_b: float,
    streak_a: int = 0,
    streak_b: int = 0,
    survivors_a: int = 0,
    survivors_b: int = 0,
    a_started: str = "atk",
    map_name: str | None = None,
    agents_a: list[str] | None = None,
    agents_b: list[str] | None = None,
    ready_agents_a: list[str] | None = None,
    ready_agents_b: list[str] | None = None,
    ults_a: int = 0,
    ults_b: int = 0,
    n: int = 100_000,
    seed: int = 0,
    economy_and_ults_affect_outcome: bool = True,
    p_atk: float = 0.5,
) -> Prediction:
    """Tier 3 is always Monte Carlo. ``ready_agents_a`` names which of
    ``agents_a`` have their ult ready (e.g. ``["jett"]``); it replaces the
    plain ``ults_a`` count and must be a subset of ``agents_a``."""
    ready_a, ults_a = _ready_flags("a", agents_a, ready_agents_a, ults_a)
    ready_b, ults_b = _ready_flags("b", agents_b, ready_agents_b, ults_b)
    v.validate_tier3_state(
        score_a, score_b, bank_a, bank_b, ults_a, ults_b, streak_a, streak_b,
        survivors_a, survivors_b, a_started, map_name,
    )
    rng = np.random.default_rng(seed)
    result = tier3_mc(
        score_a=score_a, score_b=score_b, bank_a=bank_a, bank_b=bank_b,
        ults_a=ults_a, ults_b=ults_b, streak_a=streak_a, streak_b=streak_b,
        n=n, rng=rng, survivors_a=survivors_a, survivors_b=survivors_b,
        a_started=a_started, map_name=map_name,
        agents_a=agents_a, agents_b=agents_b, ready_a=ready_a, ready_b=ready_b,
        economy_and_ults_affect_outcome=economy_and_ults_affect_outcome, p_atk=p_atk,
    )
    return Prediction(3, [_mc(result)])


def _ready_flags(team: str, agents, ready_agents, ults: int):
    """Turn "these agents have their ult ready" into the five boolean flags
    Tier 3 expects (or leave it as a plain count when no agents are named)."""
    if ready_agents is None:
        return None, ults
    if agents is None:
        raise v.InvalidStateError(
            f"ready_agents_{team} names which agents have an ult ready, so agents_{team} "
            f"(the team's five agents) is also required"
        )
    if len(agents) != v.PLAYERS_PER_TEAM:
        raise v.InvalidStateError(f"agents_{team} must list {v.PLAYERS_PER_TEAM} agents, got {len(agents)}")
    missing = [a for a in ready_agents if a not in agents]
    if missing:
        raise v.InvalidStateError(f"ready_agents_{team} {missing} aren't on the team: {list(agents)}")
    flags = [a in set(ready_agents) for a in agents]
    return flags, sum(flags)


def _check_method(method: str) -> None:
    if method not in ("exact", "mc", "both"):
        raise v.InvalidStateError(f"method must be 'exact', 'mc' or 'both', got {method!r}")
