"""Match-level reporting from a Rounds CSV: both teams' win probability after
every round (the broadcast-style graph), and a round-by-round step-through of
credits, buy type and loss streak.

Team A is whichever team attacks in the match's first round, so
``a_started = "atk"``. The win probability uses Tier 1's exact answer (score
and side only) at the map's even-gear attacker round-win rate from the
Phase 5 fit, so it shows how the *scoreline* moved the odds, not economy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from winprob.economy.buy_type import classify_buy_type, team_round_loadout
from winprob.tier1.exact import solve_linear_system
from winprob.tier1.lookup import exact_win_probability
from winprob.tier1.states import ROUNDS_PER_HALF, map_winner
from winprob.tier3.regression_defaults import MAP_EFFECTS, default_p_atk

RESET_ROUNDS = (1, ROUNDS_PER_HALF + 1)  # loss streaks restart at each half's pistol round


def load_match_rounds(rounds: pd.DataFrame, match_id: str) -> pd.DataFrame:
    """One match's rounds in order, up to the round that decided the map.

    Rounds with no winner (VLR has no record of them) can't move the score,
    so they stop the history there: everything after an unknown round would
    be a wrong score.
    """
    m = rounds[rounds["match_id"] == match_id].sort_values("round")
    if m.empty:
        known = sorted(rounds["match_id"].unique())[:5]
        raise ValueError(f"match_id {match_id!r} isn't in the rounds file (first few: {known})")
    unknown = m["winner"].isna()
    if unknown.any():
        m = m.iloc[: int(np.argmax(unknown.to_numpy()))]
    if m.empty:
        raise ValueError(f"match {match_id!r} has no rounds with a recorded winner")
    return m.reset_index(drop=True)


def _team_names(match: pd.DataFrame) -> tuple[str, str]:
    first = match.iloc[0]
    return str(first["atk_team"]), str(first["def_team"])


def win_probability_series(match: pd.DataFrame, p_atk: float | None = None) -> pd.DataFrame:
    """Team A's and B's win probability before round 1 and after every round."""
    team_a, team_b = _team_names(match)
    if p_atk is None:
        map_name = str(match["map"].iloc[0]) if "map" in match.columns else None
        p_atk = default_p_atk(map_name if map_name in MAP_EFFECTS else None)

    table = solve_linear_system(p_atk, "atk")
    score_a = score_b = 0
    rows = [(0, "", 0, 0, exact_win_probability(0, 0, p_atk, "atk", table))]
    for _, r in match.iterrows():
        if map_winner(score_a, score_b) is not None:
            break
        if r["winner"] == team_a:
            score_a += 1
        else:
            score_b += 1
        rows.append((int(r["round"]), r["winner"], score_a, score_b,
                     exact_win_probability(score_a, score_b, p_atk, "atk", table)))
    out = pd.DataFrame(rows, columns=["round", "winner", "score_a", "score_b", "p_a"])
    out["p_b"] = 1.0 - out["p_a"]
    out.attrs.update(team_a=team_a, team_b=team_b, p_atk=p_atk)
    return out


def plot_win_probability(series: pd.DataFrame, path: str, title: str | None = None) -> None:
    """Save the broadcast-style graph: both teams' win probability by round."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    team_a, team_b = series.attrs["team_a"], series.attrs["team_b"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.step(series["round"], series["p_a"], where="post", label=team_a, linewidth=2)
    ax.step(series["round"], series["p_b"], where="post", label=team_b, linewidth=2)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(ROUNDS_PER_HALF + 0.5, color="gray", linewidth=0.8, linestyle=":")
    ax.set_ylim(0, 1)
    ax.set_xlabel("round")
    ax.set_ylabel("win probability")
    last = series.iloc[-1]
    ax.set_title(title or f"{team_a} {int(last['score_a'])}-{int(last['score_b'])} {team_b}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def step_through(match: pd.DataFrame, buys: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per round: the score going in, each team's credits at the
    start of the buy phase (spent + bank, from Rounds), its buy type (from
    Buys loadout, only if ``buys`` is given - spent alone understates a
    team holding saved rifles) and the loss streak it entered the round on.
    """
    team_a, team_b = _team_names(match)
    loadout = None
    if buys is not None:
        loadout = team_round_loadout(buys[buys["match_id"] == match["match_id"].iloc[0]])
        loadout = loadout.set_index(["round", "team"])["loadout"]

    score = {team_a: 0, team_b: 0}
    streak = {team_a: 0, team_b: 0}
    rows = []
    for _, r in match.iterrows():
        rnd = int(r["round"])
        if rnd in RESET_ROUNDS or rnd > 2 * ROUNDS_PER_HALF:
            streak = {team_a: 0, team_b: 0}

        def side_values(team: str) -> tuple[float, str]:
            side = "atk" if r["atk_team"] == team else "def"
            credits = r[f"{side}_spent"] + r[f"{side}_bank"] if r["recorded"] == 1 else np.nan
            buy = "-"
            if loadout is not None and (rnd, team) in loadout.index:
                buy = classify_buy_type(loadout[(rnd, team)])
            return credits, buy

        credits_a, buy_a = side_values(team_a)
        credits_b, buy_b = side_values(team_b)
        rows.append({
            "round": rnd,
            "score": f"{score[team_a]}-{score[team_b]}",
            "credits_a": credits_a, "buy_a": buy_a, "streak_a": streak[team_a],
            "credits_b": credits_b, "buy_b": buy_b, "streak_b": streak[team_b],
            "winner": r["winner"],
        })
        loser = team_b if r["winner"] == team_a else team_a
        winner = r["winner"]
        score[winner] += 1
        streak[winner] = 0
        streak[loser] += 1
    out = pd.DataFrame(rows)
    out.attrs.update(team_a=team_a, team_b=team_b)
    return out
