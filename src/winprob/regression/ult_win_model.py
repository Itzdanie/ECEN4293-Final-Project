"""The round-win model with an ultimates term, fit on the real per-player
ult-readiness export.

    logit P(atk wins) = b0 + b1*delta_loadout + b_half + b_map
                        + sum over agents of  effect[agent] * delta_ready[agent]

``delta_ready[agent]`` is (attackers with that agent's ult ready) minus
(defenders with it ready) at the start of the round - *ready*, not *spent*,
because readiness is a state while spending is the team's choice
(CLAUDE.md). The per-agent effect is built from three shrunk layers:

    effect[agent] = role + sub_role deviation + agent deviation

The role terms are unpenalized; the sub-role and agent deviations carry
ridge penalties, so an agent with little data stays near its sub-role and a
sub-role with little data stays near its role. The two penalty strengths are
chosen by cross-validation over whole matches. Fit by penalized Newton.

Association, not cause: ults get ready by getting kills, so part of any
effect is "this player is playing well". Rows with an incomplete team-round
are dropped, which can bias the sample (the missing rows skew toward ready
ults), and rounds 12/24 are excluded as an export bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from winprob.numerics.logistic import fit_newton_penalized, log_likelihood
from winprob.regression.dataset import DEFAULT_TIERS, build_training_table
from winprob.regression.round_win_model import _design_matrix
from winprob.ults.prep import EXCLUDED_ROUNDS
from winprob.ults.roles import AGENT_SUBROLE, AGENTS, ROLES, SUB_ROLES, role_of

DEFAULT_SUB_GRID = (100.0, 300.0, 1_000.0, 3_000.0, 10_000.0, 30_000.0)
DEFAULT_AGENT_GRID = (30.0, 100.0, 300.0, 1_000.0, 3_000.0)


def build_ult_training_table(
    rounds: pd.DataFrame,
    buys: pd.DataFrame,
    ult_rows: pd.DataFrame,
    tiers: tuple[str, ...] = DEFAULT_TIERS,
) -> pd.DataFrame:
    """The Phase 5 training table, restricted to rounds where both teams
    have a complete ult record, plus one ``delta_<agent>`` column per agent
    (attacker ready count minus defender ready count for that agent).

    ``ult_rows`` is the output of :func:`winprob.ults.prep.clean_ult_rows`.
    """
    base = build_training_table(rounds, buys, tiers)
    base = base[~base["round"].isin(EXCLUDED_ROUNDS)]
    sides = rounds[["match_id", "round", "atk_team", "def_team"]]
    base = base.merge(sides, on=["match_id", "round"], how="inner")

    keys = ["match_id", "round"]
    u = ult_rows[keys + ["team_name", "agent", "ready"]]
    atk = base[keys + ["atk_team"]].merge(
        u, left_on=keys + ["atk_team"], right_on=keys + ["team_name"]
    )
    dfn = base[keys + ["def_team"]].merge(
        u, left_on=keys + ["def_team"], right_on=keys + ["team_name"]
    )
    atk["signed"] = atk["ready"]
    dfn["signed"] = -dfn["ready"]
    signed = pd.concat([atk[keys + ["agent", "signed"]], dfn[keys + ["agent", "signed"]]])
    delta = signed.pivot_table(
        index=keys, columns="agent", values="signed", aggfunc="sum", fill_value=0
    ).reindex(columns=list(AGENTS), fill_value=0)
    delta.columns = ["delta_" + a for a in delta.columns]

    # both teams must have a complete record for the round
    teams_present = signed.assign(n=1).groupby(keys)["n"].sum()
    both = teams_present[teams_present == 10].index

    out = base.set_index(keys).join(delta, how="inner")
    out = out.loc[out.index.isin(both)].reset_index()
    return out.drop(columns=["atk_team", "def_team"])


def _agent_columns(table: pd.DataFrame) -> np.ndarray:
    return table[["delta_" + a for a in AGENTS]].to_numpy(dtype=float)


def _hierarchy() -> tuple[np.ndarray, np.ndarray]:
    """0/1 matrices mapping agents (rows) onto their role / sub-role columns."""
    to_role = np.array([[role_of(AGENT_SUBROLE[a]) == r for r in ROLES] for a in AGENTS], float)
    to_sub = np.array([[AGENT_SUBROLE[a] == s for s in SUB_ROLES] for a in AGENTS], float)
    return to_role, to_sub


def _design(table: pd.DataFrame, map_levels: list[str]) -> tuple[np.ndarray, int]:
    """Columns: [base | roles | sub-roles | agents]. Returns ``(X, n_base)``."""
    base, _ = _design_matrix(table, map_levels)
    deltas = _agent_columns(table)
    to_role, to_sub = _hierarchy()
    return np.column_stack([base, deltas @ to_role, deltas @ to_sub, deltas]), base.shape[1]


def _penalty(n_base: int, lam_sub: float, lam_agent: float) -> np.ndarray:
    return np.r_[
        np.zeros(n_base + len(ROLES)),
        np.full(len(SUB_ROLES), lam_sub),
        np.full(len(AGENTS), lam_agent),
    ]


def fit_ult_win_model(
    table: pd.DataFrame, lam_sub: float, lam_agent: float, map_levels: list[str] | None = None
) -> dict:
    """Fit the model above at the given penalty strengths. Returns the base
    coefficients and each agent's total ``effect`` (logit units per ready ult)."""
    map_levels = map_levels or sorted(table["map"].unique())
    X, n_base = _design(table, map_levels)
    y = table["atk_won"].to_numpy(dtype=float)
    beta, n_iter, history = fit_newton_penalized(X, y, _penalty(n_base, lam_sub, lam_agent))

    to_role, to_sub = _hierarchy()
    r0 = n_base
    s0 = r0 + len(ROLES)
    a0 = s0 + len(SUB_ROLES)
    effect = to_role @ beta[r0:s0] + to_sub @ beta[s0:a0] + beta[a0:]
    return {
        "beta": beta,
        "n_base": n_base,
        "role_effects": dict(zip(ROLES, beta[r0:s0])),
        "sub_role_deviations": dict(zip(SUB_ROLES, beta[s0:a0])),
        "agent_deviations": dict(zip(AGENTS, beta[a0:])),
        "agent_effects": dict(zip(AGENTS, effect)),
        "map_levels": map_levels,
        "n_iter": n_iter,
        "log_likelihood": log_likelihood(X, y, beta),
        "n_rounds": len(table),
    }


def fit_pooled_ult_effect(table: pd.DataFrame) -> float:
    """One shared logit effect per ready ult (the no-roles baseline), used
    as Tier 3's generic effect when a slot's agent isn't specified."""
    base, _ = _design_matrix(table, sorted(table["map"].unique()))
    total = _agent_columns(table).sum(axis=1)
    X = np.column_stack([base, total])
    y = table["atk_won"].to_numpy(dtype=float)
    beta, _, _ = fit_newton_penalized(X, y, np.zeros(X.shape[1]))
    return float(beta[-1])


def choose_penalties(
    table: pd.DataFrame,
    sub_grid=DEFAULT_SUB_GRID,
    agent_grid=DEFAULT_AGENT_GRID,
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[float, float, pd.DataFrame]:
    """Pick the two penalty strengths by cross-validation, holding out whole
    matches (rounds of one map are correlated). Scores held-out log-likelihood.

    Returns ``(lam_sub, lam_agent, grid)`` where ``grid`` has one row per
    combination, so a best value sitting on the grid's edge is visible.
    """
    map_levels = sorted(table["map"].unique())
    X, n_base = _design(table, map_levels)
    y = table["atk_won"].to_numpy(dtype=float)

    matches = table["match_id"].unique()
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(matches, rng.integers(0, n_folds, len(matches))))
    fold = table["match_id"].map(fold_of).to_numpy()

    rows = []
    for lam_sub in sub_grid:
        for lam_agent in agent_grid:
            pen = _penalty(n_base, lam_sub, lam_agent)
            score = 0.0
            for k in range(n_folds):
                train, test = fold != k, fold == k
                beta, _, _ = fit_newton_penalized(X[train], y[train], pen)
                score += log_likelihood(X[test], y[test], beta)
            rows.append((lam_sub, lam_agent, score))
    grid = pd.DataFrame(rows, columns=["lam_sub", "lam_agent", "cv_log_likelihood"])
    best = grid.loc[grid["cv_log_likelihood"].idxmax()]
    return float(best["lam_sub"]), float(best["lam_agent"]), grid
