"""Refit the real-data ult model and print the constants that go into
``src/winprob/tier3/ult_defaults.py`` (run from the repo root):

    python scripts/fit_ult_defaults.py

Needs the raw CSVs in ``data/raw/`` (gitignored). Prints the CV grid, then
the model summary, then the ``ult_defaults.py`` source to paste.
"""

from __future__ import annotations

from winprob.data.loading import load_buys, load_maps, load_rounds, load_ult_ready
from winprob.regression.ult_win_model import (
    build_ult_training_table,
    choose_penalties,
    fit_pooled_ult_effect,
    fit_ult_win_model,
)
from winprob.ults.habits import fit_habits
from winprob.ults.prep import clean_ult_rows
from winprob.ults.roles import AGENTS


def main() -> None:
    maps = load_maps()
    rows = clean_ult_rows(load_ult_ready(), maps)
    table = build_ult_training_table(load_rounds(), load_buys(), rows)
    print(f"{len(table)} rounds from {table['match_id'].nunique()} matches")

    lam_sub, lam_agent, grid = choose_penalties(table)
    print(grid.pivot(index="lam_sub", columns="lam_agent", values="cv_log_likelihood").round(1))
    print(f"chosen lam_sub={lam_sub}, lam_agent={lam_agent}")

    model = fit_ult_win_model(table, lam_sub, lam_agent)
    print(f"iterations={model['n_iter']}  log-likelihood={model['log_likelihood']:.1f}")
    print("role effects (logit/ult):", {k: round(v, 3) for k, v in model["role_effects"].items()})

    habits = fit_habits(rows)
    print("\n# ---- paste into src/winprob/tier3/ult_defaults.py ----")
    print(f"START_SHAPE = {[round(x, 4) for x in habits['start_shape']]}")
    print(f"POST_SHAPE = {[round(x, 4) for x in habits['post_shape']]}")
    print(f"GENERIC = {{'charge_scale': 1.0, 'use_prob': {habits['generic']['use_prob']:.4f}, 'boost': {fit_pooled_ult_effect(table):.4f}}}")
    print("AGENT_PARAMS = {")
    for a in AGENTS:
        h = habits["agents"][a]
        print(
            f"    {a!r}: ({h['charge_scale']:.3f}, {h['use_prob']:.3f}, {model['agent_effects'][a]:.4f}),"
        )
    print("}")


if __name__ == "__main__":
    main()
