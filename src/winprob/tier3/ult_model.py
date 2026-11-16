"""Tier 3's ultimate model: per-player-slot readiness for each team, driven
by the real per-agent habits in :mod:`winprob.tier3.ult_defaults`.

State per team is three (n, 5) arrays over a team's player slots:
``ready`` (ult charged at the start of the round), ``age`` (rounds since
the slot last emptied) and ``post_spend`` (True if it emptied by using the
ult, False if by a half starting). Per round:

1. Readiness at the *start* of the round is what shifts the round-win
   logit (each ready ult adds its agent's ``boost``; CLAUDE.md: readiness,
   not spending, is the regressor).
2. Each ready ult is used with that agent's ``use_prob``. A used ult is
   empty next round (the real data never shows a spent ult ready next
   round); an unused one stays ready (it always does in the real data).
3. Each not-ready slot becomes ready with the pooled charge hazard for its
   ``age``/``post_spend`` times its agent's ``charge_scale``.

Everything resets to not-ready with age 0 at round 1, round 13 and every
overtime round, matching the real data's resets.

Slots with no agent given use the pooled ``GENERIC`` parameters.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from winprob.tier3.ult_defaults import AGENT_PARAMS, GENERIC, POST_SHAPE, START_SHAPE
from winprob.ults.habits import hazard

MAX_ULTS_READY = 5
SLOTS = 5
GENERIC_BOOST = GENERIC["boost"]  # pooled logit effect of one ready ult


def slot_params(agents: Sequence[str] | None) -> dict[str, np.ndarray]:
    """``charge_scale``, ``use_prob`` and ``boost`` arrays of length 5 for a
    team's agents, or the pooled generic values when ``agents`` is None."""
    if agents is None:
        return {
            "charge_scale": np.full(SLOTS, GENERIC["charge_scale"]),
            "use_prob": np.full(SLOTS, GENERIC["use_prob"]),
            "boost": np.full(SLOTS, GENERIC["boost"]),
        }
    if len(agents) != SLOTS:
        raise ValueError(f"a team has {SLOTS} agents, got {len(agents)}: {list(agents)}")
    unknown = sorted(set(agents) - set(AGENT_PARAMS))
    if unknown:
        raise ValueError(f"unknown agents {unknown}; known agents: {sorted(AGENT_PARAMS)}")
    rows = np.array([AGENT_PARAMS[a] for a in agents], dtype=float)
    return {"charge_scale": rows[:, 0], "use_prob": rows[:, 1], "boost": rows[:, 2]}


def initial_ready(ults: int, ready: Sequence[bool] | None) -> np.ndarray:
    """A team's starting ready flags (length 5): ``ready`` if given,
    otherwise the first ``ults`` slots."""
    if ready is not None:
        flags = np.array([bool(x) for x in ready])
        if flags.shape != (SLOTS,):
            raise ValueError(f"ready must have {SLOTS} entries, got {len(ready)}")
        if ults != int(flags.sum()):
            raise ValueError(f"ults={ults} disagrees with ready flags {list(ready)} ({int(flags.sum())} ready)")
        return flags
    if not 0 <= ults <= MAX_ULTS_READY:
        raise ValueError(f"ults must be between 0 and {MAX_ULTS_READY}, got {ults}")
    return np.arange(SLOTS) < ults


def step_ults(
    rng: np.random.Generator,
    ready: np.ndarray,
    age: np.ndarray,
    post_spend: np.ndarray,
    params: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance one team's ult state by one round, for ``n`` simulations at
    once (all arrays ``(n, 5)``). Returns ``(used, ready, age, post_spend)``
    where ``ready`` etc. are next round's state; ``used`` is what was used
    *this* round (``ready`` slots only).
    """
    used = ready & (rng.random(ready.shape) < params["use_prob"])

    h = hazard(START_SHAPE, POST_SHAPE, age, post_spend, params["charge_scale"])
    charged = ~ready & (rng.random(ready.shape) < h)

    next_ready = (ready & ~used) | charged
    next_age = np.where(used | charged, 0, age + 1)
    next_post = np.where(used, True, np.where(charged, False, post_spend))
    return used, next_ready, next_age, next_post
