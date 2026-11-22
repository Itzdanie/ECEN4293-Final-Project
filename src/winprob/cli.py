"""Command-line interface: ``winprob <command> ...``

    winprob predict   win probability from a game state (tier 1, 2 or 3)
    winprob solve     round-win rate needed to reach a target win probability
    winprob converge  Monte Carlo error vs N (log-log plot and slope)
    winprob fit       fit the round-win logistic model on the pro data
    winprob match     win-probability graph / step-through for one match

Impossible states are rejected with an explanation (exit code 2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from winprob import predict as P
from winprob.tier1.convergence import DEFAULT_NS, convergence_study, plot_convergence
from winprob.tier1.exact import solve_linear_system
from winprob.tier1.root_finding import (
    required_round_win_rate_bisection,
    required_round_win_rate_secant,
)
from winprob.tier3.regression_defaults import default_p_atk
from winprob.tier1.states import ROUNDS_PER_HALF
from winprob.validation import InvalidStateError, is_reset_round, validate_scores, validate_tier1_state


def _score(text: str) -> tuple[int, int]:
    try:
        a, b = text.replace(":", "-").split("-")
        return int(a), int(b)
    except ValueError:
        raise argparse.ArgumentTypeError(f"score must look like 7-5, got {text!r}") from None


def _agents(text: str) -> list[str]:
    return [a.strip().lower() for a in text.split(",") if a.strip()]


def _pct(p: float) -> str:
    return f"{100 * p:6.2f}%"


# --- predict ---------------------------------------------------------------


def _print_prediction(pred: P.Prediction, n: int, seed: int) -> None:
    for e in pred.estimates:
        label = "exact" if e.method == "exact" else f"Monte Carlo, N={n:,}, seed={seed}"
        line = f"Tier {pred.tier} [{label}]  Team A {_pct(e.p_a)}   Team B {_pct(e.p_b)}"
        if e.ci:
            line += f"   (95% CI for A: {_pct(e.ci[0])} to {_pct(e.ci[1])})"
        print(line)
    if pred.difference is not None:
        print(f"Difference (Monte Carlo - exact): {100 * pred.difference:+.3f} percentage points")


def _cmd_predict(args) -> int:
    score_a, score_b = args.score
    method = "both" if args.compare else args.method
    validate_scores(score_a, score_b)
    if args.tier == 1:
        pred = P.predict_tier1(score_a, score_b, args.a_started, args.p_atk, args.map, method, args.n, args.seed)
    elif args.tier == 2:
        buy_a, buy_b = args.buy_a, args.buy_b
        if (buy_a is None or buy_b is None) and (score_a + score_b) in (0, ROUNDS_PER_HALF):
            buy_a, buy_b = buy_a or "Eco", buy_b or "Eco"  # a pistol round is 50/50 whatever the buy
        if buy_a is None or buy_b is None:
            raise InvalidStateError(
                "Tier 2 needs --buy-a and --buy-b (Eco, Semi-eco, Semi-buy or Full buy) "
                "unless the next round is a pistol round; use --tier 1 for a score-only answer"
            )
        pred = P.predict_tier2(score_a, score_b, buy_a, buy_b, args.streak_a, args.streak_b, method, args.n, args.seed)
    else:
        if method == "both":
            raise InvalidStateError("Tier 3 has no exact solution to compare against - it is Monte Carlo only")
        if not is_reset_round(score_a, score_b) and (args.bank_a is None or args.bank_b is None):
            raise InvalidStateError(
                "Tier 3 needs --bank-a and --bank-b (each team's total credits) outside pistol/overtime rounds"
            )
        pred = P.predict_tier3(
            score_a, score_b, args.bank_a or 0.0, args.bank_b or 0.0, args.streak_a, args.streak_b,
            args.survivors_a, args.survivors_b, args.a_started, args.map,
            args.agents_a, args.agents_b, args.ready_a, args.ready_b, args.ults_a, args.ults_b,
            args.n, args.seed, not args.no_economy,
            args.p_atk if args.p_atk is not None else 0.5,
        )
    _print_prediction(pred, args.n, args.seed)
    if args.tier == 1:
        print("Note: Tier 1 uses only score and side - it ignores the economy (e.g. who won the pistol). "
              "Use --tier 2 or 3 for analysis.")
    return 0


# --- solve / converge --------------------------------------------------------


def _cmd_solve(args) -> int:
    score_a, score_b = args.score
    validate_tier1_state(score_a, score_b, args.a_started)
    if not 0.0 < args.target < 1.0:
        raise InvalidStateError(f"--target must be strictly between 0 and 1, got {args.target}")
    solver = required_round_win_rate_bisection if args.method == "bisection" else required_round_win_rate_secant
    q = solver(score_a, score_b, args.a_started, args.target)
    check = solve_linear_system(q, args.a_started)[(score_a, score_b)]
    print(f"From {score_a}-{score_b} (team A started on {args.a_started}), team A needs an attacker round-win")
    print(f"rate of {q:.6f} to have a {args.target:.0%} chance of winning the map ({args.method}).")
    print(f"Check: exact win probability at that rate = {check:.6f}")
    return 0


def _cmd_converge(args) -> int:
    score_a, score_b = args.score
    p_atk = args.p_atk if args.p_atk is not None else default_p_atk()
    validate_tier1_state(score_a, score_b, args.a_started, p_atk)
    ns = np.logspace(2, np.log10(args.max_n), 10).astype(int) if args.max_n != DEFAULT_NS[-1] else DEFAULT_NS
    result = convergence_study(p_atk, args.a_started, score_a, score_b, ns, args.trials, np.random.default_rng(args.seed))
    plot_convergence(result, args.out)
    print(f"Exact answer: {result['exact']:.6f}")
    for n, err in zip(result["ns"], result["mean_errors"]):
        print(f"  N = {int(n):>7,}   mean |error| = {err:.5f}")
    print(f"Log-log slope of error vs N: {result['slope']:.3f} (theory: -0.5)")
    print(f"Plot saved to {args.out}")
    return 0


# --- fit -------------------------------------------------------------------------


def format_fit_report(model: dict) -> str:
    lines = [f"Rounds fitted: {model['n_rounds']:,}", "Coefficients (logit of P(attackers win)):"]
    for name, b in zip(model["feature_names"], model["beta"]):
        lines.append(f"  {name:<16}{b:>10.4f}")
    lines.append(f"Log-likelihood: {model['log_likelihood']:.2f}")
    lines.append(f"Newton iterations to converge: {model['n_iter']}")
    return "\n".join(lines)


def _cmd_fit(args) -> int:
    from winprob.data.loading import load_buys, load_rounds
    from winprob.regression import build_training_table, fit_round_win_model

    for path in (args.rounds, args.buys):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found - put the chiro.gg exports in data/raw/ or pass --rounds/--buys")
    table = build_training_table(load_rounds(args.rounds), load_buys(args.buys), tuple(args.tiers))
    print(format_fit_report(fit_round_win_model(table)))
    return 0


# --- match -------------------------------------------------------------------------


def _cmd_match(args) -> int:
    from winprob.data.loading import load_buys, load_rounds
    from winprob.reporting.match import (
        load_match_rounds,
        plot_win_probability,
        step_through,
        win_probability_series,
    )

    if not Path(args.rounds).exists():
        raise FileNotFoundError(f"{args.rounds} not found - pass --rounds with a Rounds CSV")
    match = load_match_rounds(load_rounds(args.rounds), args.match_id)
    series = win_probability_series(match, args.p_atk)
    team_a, team_b = series.attrs["team_a"], series.attrs["team_b"]
    print(f"{team_a} (A) vs {team_b} (B), {match['map'].iloc[0]}; round-win rate p_atk = {series.attrs['p_atk']:.3f}")
    print("round  score   P(A)     P(B)")
    for _, r in series.iterrows():
        print(f"{int(r['round']):>5}  {int(r['score_a']):>2}-{int(r['score_b']):<2}  {_pct(r['p_a'])} {_pct(r['p_b'])}")
    if args.out:
        plot_win_probability(series, args.out)
        print(f"Plot saved to {args.out}")
    if args.step:
        buys = load_buys(args.buys) if args.buys and Path(args.buys).exists() else None
        steps = step_through(match, buys)
        print(f"\nround score  {team_a}: credits buy streak | {team_b}: credits buy streak | winner")
        for _, s in steps.iterrows():
            ca = "n/a" if s["credits_a"] != s["credits_a"] else f"{int(s['credits_a'])}"
            cb = "n/a" if s["credits_b"] != s["credits_b"] else f"{int(s['credits_b'])}"
            print(f"{s['round']:>5} {s['score']:<6} {ca:>8} {s['buy_a']:<9} {s['streak_a']} | {cb:>8} {s['buy_b']:<9} {s['streak_b']} | {s['winner']}")
    return 0


# --- parser ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    raw = Path(__file__).resolve().parents[2] / "data" / "raw"
    parser = argparse.ArgumentParser(prog="winprob", description="Valorant map win-probability engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="win probability from a game state")
    p.add_argument("--tier", type=int, choices=(1, 2, 3), default=2, help="default 2; Tier 1 ignores the economy")
    p.add_argument("--score", type=_score, default=(0, 0), help="team A - team B rounds, e.g. 7-5")
    p.add_argument("--a-started", choices=("atk", "def"), default="atk", help="side team A started on")
    p.add_argument("--p-atk", type=float, help="attacker round-win rate (Tier 1; default: from the fitted model)")
    p.add_argument("--map", help="map name, for its fitted attacker effect")
    p.add_argument("--method", choices=("exact", "mc", "both"), default="exact", help="Tier 1/2 only; Tier 3 is always Monte Carlo")
    p.add_argument("--compare", action="store_true", help="Tier 1/2: show exact, Monte Carlo and their difference")
    p.add_argument("--n", type=int, default=100_000, help="Monte Carlo playouts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--buy-a")
    p.add_argument("--buy-b")
    p.add_argument("--streak-a", type=int, default=0)
    p.add_argument("--streak-b", type=int, default=0)
    p.add_argument("--bank-a", type=float, help="Tier 3: team A's total credits")
    p.add_argument("--bank-b", type=float)
    p.add_argument("--survivors-a", type=int, default=0)
    p.add_argument("--survivors-b", type=int, default=0)
    p.add_argument("--agents-a", type=_agents, help="Tier 3: five agents, e.g. omen,raze,jett,sova,cypher")
    p.add_argument("--agents-b", type=_agents)
    p.add_argument("--ready-a", type=_agents, help="Tier 3: which of --agents-a have their ult ready, e.g. jett,omen")
    p.add_argument("--ready-b", type=_agents)
    p.add_argument("--ults-a", type=int, default=0, help="Tier 3: number of ready ults (if agents aren't named)")
    p.add_argument("--ults-b", type=int, default=0)
    p.add_argument("--no-economy", action="store_true", help="Tier 3: turn economy and ult effects off")
    p.set_defaults(func=_cmd_predict)

    s = sub.add_parser("solve", help="round-win rate needed to reach a target win probability")
    s.add_argument("--score", type=_score, default=(0, 0))
    s.add_argument("--a-started", choices=("atk", "def"), default="atk")
    s.add_argument("--target", type=float, default=0.5)
    s.add_argument("--method", choices=("bisection", "secant"), default="bisection")
    s.set_defaults(func=_cmd_solve)

    c = sub.add_parser("converge", help="Monte Carlo error vs N")
    c.add_argument("--score", type=_score, default=(0, 0))
    c.add_argument("--a-started", choices=("atk", "def"), default="atk")
    c.add_argument("--p-atk", type=float)
    c.add_argument("--trials", type=int, default=10)
    c.add_argument("--max-n", type=int, default=int(DEFAULT_NS[-1]))
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", default="convergence.png")
    c.set_defaults(func=_cmd_converge)

    f = sub.add_parser("fit", help="fit the round-win logistic model on the pro data")
    f.add_argument("--rounds", default=str(raw / "rounds_combined.csv"))
    f.add_argument("--buys", default=str(raw / "buys_combined.csv"))
    f.add_argument("--tiers", nargs="+", default=["league", "international"])
    f.set_defaults(func=_cmd_fit)

    m = sub.add_parser("match", help="win-probability graph / step-through for one match")
    m.add_argument("match_id")
    m.add_argument("--rounds", default=str(raw / "rounds_combined.csv"))
    m.add_argument("--buys", default=str(raw / "buys_combined.csv"), help="optional; adds buy types to --step")
    m.add_argument("--p-atk", type=float)
    m.add_argument("--out", help="save the win-probability graph to this PNG")
    m.add_argument("--step", action="store_true", help="print credits, buy type and loss streak per round")
    m.set_defaults(func=_cmd_match)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (InvalidStateError, FileNotFoundError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
