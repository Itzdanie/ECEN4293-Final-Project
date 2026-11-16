"""Per-agent ult habits from the real readiness export: how fast an ult
charges and how often a ready one gets used.

**Charging** is modeled round to round per player slot. Two real facts
(measured on the export: 37k ready-and-unused and 53k spent transitions):
a ready ult that isn't used stays ready next round *every time*, and a
spent ult is never ready the next round. So only not-ready slots can
change, becoming ready with a hazard that depends on how many rounds
(``age``) have passed since the slot last emptied:

- after a half starts (round 1 / 13): an S-curve, ~2% on the first round,
  climbing to ~25% by age 5 (``start_shape``);
- after the ult is used: roughly flat, ~9-17% (``post_shape``).

Both shapes are pooled across agents; each agent then gets a single
``charge_scale`` multiplying them, shrunk toward its sub-role's scale so
agents with few rounds (and agents with none, like Hush) stay sensible.

**Use** is each agent's own ``use_prob`` = P(spent | ready), shrunk toward
its sub-role's rate the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from winprob.ults.roles import AGENT_SUBROLE, AGENTS

START_MAX_AGE = 6  # ages >= this share one hazard
POST_MAX_AGE = 4
PSEUDO_COUNT = 25.0  # how hard a sparse agent is pulled toward its sub-role
MAX_HAZARD = 0.95


def charge_transitions(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per not-ready player slot in a regulation round whose next
    round (same slot, same agent) is also present: did it become ready, and
    how old (rounds since the slot last emptied) was it?

    ``rows`` is the output of :func:`winprob.ults.prep.clean_ult_rows`.
    A slot whose history has a gap (a dropped incomplete round) has unknown
    age until its next known reset, so those rows are left out.
    """
    keys = ["match_id", "team", "slot"]
    u = rows[rows["round"] <= 23].sort_values(keys + ["round"]).copy()
    g = u.groupby(keys)
    prev_round = g["round"].shift(1)
    prev_spent = g["spent"].shift(1)
    consecutive = prev_round == u["round"] - 1

    half_start = u["round"].isin([1, 13])
    post_spend = consecutive & (prev_spent == 1)
    gap = ~consecutive & ~half_start
    new_run = half_start | post_spend | gap

    u["run"] = new_run.groupby([u[k] for k in keys]).cumsum()
    run_keys = keys + ["run"]
    u["age"] = u.groupby(run_keys).cumcount()
    u["post_spend"] = post_spend.groupby([u[k] for k in run_keys]).transform("first")
    u["known"] = (~gap).groupby([u[k] for k in run_keys]).transform("first")

    for c in ("round", "agent", "ready"):
        u["next_" + c] = g[c].shift(-1)
    ok = (
        u["known"]
        & (u["ready"] == 0)
        & (u["next_round"] == u["round"] + 1)
        & (u["next_agent"] == u["agent"])
    )
    out = u.loc[ok, ["agent", "age", "post_spend", "next_ready"]].copy()
    out["next_ready"] = out["next_ready"].astype(float)
    out["post_spend"] = out["post_spend"].astype(bool)
    return out.reset_index(drop=True)


def _shape(trans: pd.DataFrame, post: bool, max_age: int) -> np.ndarray:
    sub = trans[trans["post_spend"] == post]
    age = np.minimum(sub["age"].to_numpy(), max_age)
    ready = sub["next_ready"].to_numpy()
    return np.array([ready[age == a].mean() for a in range(max_age + 1)])


def hazard(start_shape, post_shape, age: np.ndarray, post_spend: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Chance a not-ready slot is ready next round, for arrays of slots."""
    start = np.asarray(start_shape)[np.minimum(age, len(start_shape) - 1)]
    post = np.asarray(post_shape)[np.minimum(age, len(post_shape) - 1)]
    return np.minimum(scale * np.where(post_spend, post, start), MAX_HAZARD)


def fit_habits(rows: pd.DataFrame) -> dict:
    """Fit the pooled charge shapes plus every agent's ``charge_scale`` and
    ``use_prob`` (both shrunk toward the agent's sub-role).

    Returns ``{"start_shape", "post_shape", "agents": {agent: {...}}, "generic": {...}}``.
    """
    trans = charge_transitions(rows)
    start_shape = _shape(trans, False, START_MAX_AGE)
    post_shape = _shape(trans, True, POST_MAX_AGE)

    age_start = np.minimum(trans["age"].to_numpy(), START_MAX_AGE)
    age_post = np.minimum(trans["age"].to_numpy(), POST_MAX_AGE)
    trans["expected"] = np.where(
        trans["post_spend"], post_shape[age_post], start_shape[age_start]
    )
    trans["sub"] = trans["agent"].map(AGENT_SUBROLE)
    obs_a = trans.groupby("agent")["next_ready"].sum()
    exp_a = trans.groupby("agent")["expected"].sum()
    obs_s = trans.groupby("sub")["next_ready"].sum()
    exp_s = trans.groupby("sub")["expected"].sum()

    # use rate: regulation rounds only (the export is trustworthy there)
    ready_rows = rows[(rows["ready"] == 1) & (rows["round"] <= 23)].copy()
    ready_rows["sub"] = ready_rows["agent"].map(AGENT_SUBROLE)
    used_a = ready_rows.groupby("agent")["spent"].sum()
    n_a = ready_rows.groupby("agent")["spent"].size()
    used_s = ready_rows.groupby("sub")["spent"].sum()
    n_s = ready_rows.groupby("sub")["spent"].size()
    generic_use = float(ready_rows["spent"].mean())

    agents: dict[str, dict] = {}
    for agent in AGENTS:
        sub = AGENT_SUBROLE[agent]
        sub_scale = obs_s[sub] / exp_s[sub] if sub in exp_s.index else 1.0
        o = obs_a.get(agent, 0.0)
        e = exp_a.get(agent, 0.0)
        scale = (o + PSEUDO_COUNT * sub_scale) / (e + PSEUDO_COUNT)

        sub_use = used_s[sub] / n_s[sub] if sub in n_s.index else generic_use
        used = used_a.get(agent, 0.0)
        n = n_a.get(agent, 0)
        use = (used + PSEUDO_COUNT * sub_use) / (n + PSEUDO_COUNT)
        agents[agent] = {
            "charge_scale": float(scale),
            "use_prob": float(use),
            "n_ready_rounds": int(n),
        }
    return {
        "start_shape": start_shape.tolist(),
        "post_shape": post_shape.tolist(),
        "agents": agents,
        "generic": {"charge_scale": 1.0, "use_prob": generic_use},
    }
