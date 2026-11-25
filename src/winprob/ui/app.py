"""Streamlit demo UI over the same functions the CLI calls
(``winprob.predict`` and ``winprob.reporting.match``) - no modeling code
lives here.

"Your team" is team A in the model and "Enemy team" is team B.

Run with ``winprob ui`` (or ``streamlit run src/winprob/ui/app.py``).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from winprob import predict as P
from winprob.data.loading import RAW_DATA_DIR, load_buys, load_rounds
from winprob.economy.buy_type import BUY_TYPES, classify_buy_type
from winprob.reporting.match import (
    load_match_rounds,
    step_through,
    win_probability_figure,
    win_probability_series,
)
from winprob.tier1.states import ROUNDS_PER_HALF, map_winner, other_side, side_for_team_a
from winprob.tier3.regression_defaults import MAP_EFFECTS, default_p_atk
from winprob.ults.roles import AGENT_SUBROLE, AGENTS, ROLES, role_of
from winprob.validation import TEAM_CREDIT_CAP, InvalidStateError

TIER_LABELS = {1: "Tier 1 · Score", 2: "Tier 2 · + Buys", 3: "Tier 3 · + Credits & ults"}
TIER_HELP = {
    1: "Score and side only.",
    2: "Adds each team's buy type and loss streak.",
    3: "Adds credits, ults and survivors. Monte Carlo only.",
}
METHOD_LABELS = {"exact": "Exact", "mc": "Monte Carlo", "both": "Compare"}
PLAYOUTS = {1_000: "1k (quick)", 10_000: "10k", 100_000: "100k", 1_000_000: "1M (slow)"}
SIDE_LABELS = {"atk": "Attack", "def": "Defense"}
ROLE_NAMES = {"ctrl": "Controllers", "duel": "Duelists", "init": "Initiators", "sent": "Sentinels"}
DEFAULT_AGENTS = ["omen", "raze", "jett", "sova", "cypher"]
BUY_RULE = "Eco under 5k, Semi-eco under 10k, Semi-buy under 15k, Full buy from 15k (team total)."
TEAMS = {"a": "Your team", "b": "Enemy team"}


def _pct(p: float) -> str:
    return f"{100 * p:.2f}%"


# --- scoreboard ------------------------------------------------------------------------

def _change_score(team: str, step: int) -> None:
    st.session_state[f"score_{team}"] += step


def _your_side(played: int, a_started: str) -> str:
    """Your side for the next round. Overtime starts on the second-half side
    and then swaps every round (the same convention as the Tier 1 model)."""
    if played < 2 * ROUNDS_PER_HALF:
        return side_for_team_a(played, a_started)
    second_half = other_side(a_started)
    return second_half if (played - 2 * ROUNDS_PER_HALF) % 2 == 0 else a_started


def _situation(score_a: int, score_b: int, a_started: str) -> str:
    """One line on what the next round is, e.g. "Round 13 · you're on defense · pistol round"."""
    winner = map_winner(score_a, score_b)
    if winner is not None:
        return "The map is over: " + ("you won." if winner == 1.0 else "the enemy won.")
    played = score_a + score_b
    parts = [f"Round {played + 1}", f"you're on {SIDE_LABELS[_your_side(played, a_started)].lower()}"]
    if played in (0, ROUNDS_PER_HALF):
        parts.append("pistol round")
    if played >= 2 * ROUNDS_PER_HALF:
        parts.append("overtime")
    if map_winner(score_a + 1, score_b) == 1.0:
        parts.append("your match point")
    if map_winner(score_a, score_b + 1) == 0.0:
        parts.append("enemy match point")
    return " · ".join(parts)


def scoreboard(side_matters: bool) -> tuple[int, int, str]:
    st.session_state.setdefault("score_a", 5)
    st.session_state.setdefault("score_b", 6)
    score_a, score_b = st.session_state["score_a"], st.session_state["score_b"]
    over = map_winner(score_a, score_b) is not None

    you, mid, enemy = st.columns([1, 1.2, 1], vertical_alignment="center")
    mid.markdown(f"# {score_a} : {score_b}", text_alignment="center")
    for team, col in (("a", you), ("b", enemy)):
        col.markdown(f"**{TEAMS[team]}**", text_alignment="center")
        minus, plus = col.columns([1, 2])
        minus.button("−1", key=f"minus_{team}", on_click=_change_score, args=(team, -1),
                     disabled=st.session_state[f"score_{team}"] == 0, width="stretch")
        plus.button("+1 round", key=f"plus_{team}", on_click=_change_score, args=(team, 1),
                    disabled=over, width="stretch")

    a_started = st.segmented_control("You started on", list(SIDE_LABELS), format_func=SIDE_LABELS.get,
                                     default="atk", required=True, key="a_started", disabled=not side_matters)
    st.caption(_situation(score_a, score_b, a_started))
    with st.expander("Edit score"):
        c1, c2 = st.columns(2)
        c1.number_input("Your rounds", 0, 30, key="score_a")
        c2.number_input("Enemy rounds", 0, 30, key="score_b")
    return score_a, score_b, a_started


# --- inputs ----------------------------------------------------------------------------

def _count_picker(label: str, top: int, key: str) -> int:
    return st.segmented_control(label, list(range(top + 1)), default=0, required=True, key=key)


def _credits(team: str, key: str) -> float:
    bank = st.number_input(f"{TEAMS[team]} credits", 0, TEAM_CREDIT_CAP, 19000, 500, key=key, help=BUY_RULE)
    st.caption(f"Enough for: **{classify_buy_type(bank)}**")
    return float(bank)


def _agent_picker(team: str) -> tuple[list[str] | None, list[str] | None, bool]:
    """Optional five-agent picker grouped by role, then which of the five have
    their ult ready. Returns (agents, ready, complete)."""
    if not st.checkbox(f"Name {TEAMS[team].lower()}'s five agents", key=f"{team}_named"):
        return None, None, True
    picked = []
    for role in ROLES:
        options = [a for a in AGENTS if role_of(AGENT_SUBROLE[a]) == role]
        default = [a for a in DEFAULT_AGENTS if a in options]
        picked += st.pills(ROLE_NAMES[role], options, selection_mode="multi", default=default,
                           format_func=str.title, key=f"{team}_{role}") or []
    if len(picked) != 5:
        st.warning(f"{TEAMS[team]}: pick exactly five agents ({len(picked)} picked).")
        return None, None, False
    ready = st.pills(f"{TEAMS[team]}: ults ready", picked, selection_mode="multi", format_func=str.title,
                     key=f"{team}_ready") or []
    return picked, ready, True


def tier2_inputs() -> dict:
    args = {}
    for team in TEAMS:
        args[f"buy_{team}"] = st.segmented_control(f"{TEAMS[team]} buy", BUY_TYPES, default="Full buy",
                                                   required=True, key=f"buy_{team}")
    s1, s2 = st.columns(2)
    with s1:
        args["streak_a"] = _count_picker("Your loss streak", 3, "streak_a")
    with s2:
        args["streak_b"] = _count_picker("Enemy loss streak", 3, "streak_b")
    return args


def tier3_inputs(a_started: str, map_name: str | None) -> tuple[dict, bool]:
    args = dict(a_started=a_started, map_name=map_name)
    columns = dict(zip(TEAMS, st.columns(2)))
    for team, col in columns.items():
        with col:
            args[f"bank_{team}"] = _credits(team, f"bank_{team}")
            args[f"streak_{team}"] = _count_picker("Loss streak", 3, f"t3_streak_{team}")
            args[f"survivors_{team}"] = _count_picker("Survivors", 5, f"surv_{team}")
    complete = True
    for team in TEAMS:
        agents, ready, ok = _agent_picker(team)
        complete = complete and ok
        args[f"agents_{team}"], args[f"ready_agents_{team}"] = agents, ready
    for team, col in columns.items():
        args[f"ults_{team}"] = 0
        if args[f"ready_agents_{team}"] is None:
            with col:
                args[f"ults_{team}"] = _count_picker("Ults ready", 5, f"ults_{team}")
    return args, complete


# --- results ---------------------------------------------------------------------------

def _show_prediction(pred: P.Prediction, n: int, seed: int) -> None:
    for est in pred.estimates:
        label = "Exact" if est.method == "exact" else f"Monte Carlo (N={n:,}, seed={seed})"
        st.subheader(label)
        st.progress(est.p_a, text=f"Your chance to win the map: **{_pct(est.p_a)}**")
        a, b = st.columns(2)
        a.metric("Your team", _pct(est.p_a))
        b.metric("Enemy team", _pct(est.p_b))
        if est.ci:
            st.caption(f"95% confidence interval for your team: {_pct(est.ci[0])} to {_pct(est.ci[1])}")
    if pred.difference is not None:
        st.metric("Monte Carlo minus exact", f"{100 * pred.difference:+.3f} percentage points")
    if pred.tier == 1:
        st.info("Tier 1 uses only the score and side. It ignores the economy (for example who won the "
                "pistol), so use Tier 2 or 3 for analysis.")


def game_state_tab() -> None:
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        tier = st.segmented_control("Model", list(TIER_LABELS), format_func=TIER_LABELS.get,
                                    default=2, required=True, key="tier")
        st.caption(TIER_HELP[tier])
        method = "exact"
        if tier < 3:
            method = st.segmented_control("Method", list(METHOD_LABELS), format_func=METHOD_LABELS.get,
                                          default="exact", required=True, key="method")
        n = seed = 0
        if tier == 3 or method != "exact":
            m1, m2 = st.columns([3, 1], vertical_alignment="bottom")
            with m1:
                n = st.segmented_control("Monte Carlo playouts", list(PLAYOUTS), format_func=PLAYOUTS.get,
                                         default=100_000, required=True, key="n")
            seed = m2.number_input("Seed", 0, 10_000, 0, key="seed")
        run_slot = st.container()
        st.divider()

        score_a, score_b, a_started = scoreboard(side_matters=tier != 2)
        map_name = st.selectbox("Map", [None, *sorted(MAP_EFFECTS)], disabled=tier == 2,
                                format_func=lambda m: m.title() if m else "(pool average)", key="map")
        if tier == 2:
            st.caption("Tier 2 doesn't use the map or starting side: its round-win rates come from "
                       "the two buy types alone. Tier 1 and Tier 3 use both.")

        played = score_a + score_b
        pistol_next = played in (0, ROUNDS_PER_HALF)
        complete = True
        if tier == 1:
            if st.checkbox("Use the fitted attacker round-win rate", True, key="auto_p"):
                p_atk = None
                st.caption(f"= {default_p_atk(map_name):.3f} for this map (even gear)")
            else:
                p_atk = st.slider("Attacker round-win rate", 0.30, 0.70, 0.50, 0.005, key="p_atk")
            args = dict(a_started=a_started, p_atk=p_atk, map_name=map_name)
        elif tier == 2:
            if pistol_next:
                st.caption("A pistol round is next, so buy type and streak don't matter.")
            args = tier2_inputs()
        else:
            if pistol_next or played >= 2 * ROUNDS_PER_HALF:
                st.caption("Credits and ults reset this round, so the inputs below are ignored.")
            args, complete = tier3_inputs(a_started, map_name)

        run_now = tier < 3 and method == "exact"
        if not run_now:
            with run_slot:
                run_now = st.button("Run", type="primary", key="run", disabled=not complete, width="stretch")

    with right:
        signature = (tier, int(score_a), int(score_b), method, int(n), int(seed), repr(sorted(args.items(), key=str)))
        if run_now:
            try:
                with st.spinner("Calculating..."):
                    if tier == 1:
                        pred = P.predict_tier1(score_a, score_b, method=method, n=n, seed=seed, **args)
                    elif tier == 2:
                        pred = P.predict_tier2(score_a, score_b, method=method, n=n, seed=seed, **args)
                    else:
                        pred = P.predict_tier3(score_a, score_b, n=n, seed=seed, **args)
                st.session_state["result"] = (signature, pred, n, seed)
            except InvalidStateError as err:
                st.session_state.pop("result", None)
                st.error(str(err))
        saved = st.session_state.get("result")
        if saved is not None:
            sig, pred, n_used, seed_used = saved
            if sig != signature:
                st.warning("The inputs changed since this was run. Press Run to update.")
            _show_prediction(pred, n_used, seed_used)
        elif not run_now:
            st.info("Set up a state and press Run.")


# --- match history ---------------------------------------------------------------------

@st.cache_data(show_spinner="Reading the rounds file...")
def _load_rounds(path: str):
    return load_rounds(path)


@st.cache_data(show_spinner="Reading the (large) buys file...")
def _load_buys(path: str):
    return load_buys(path)


def match_tab() -> None:
    st.caption("Win probability after each round for a real match, from your Rounds export. "
               "This uses Tier 1 (score and side), so it shows how the scoreline moved the odds, not the economy.")
    rounds_path = st.text_input("Rounds CSV", str(RAW_DATA_DIR / "rounds_combined.csv"), key="rounds_path")
    match_id = st.text_input("Match ID (the match_id column)", key="match_id")
    use_buys = st.checkbox("Show buy types (loads the large Buys file)", key="use_buys")
    if not match_id:
        return
    if not Path(rounds_path).exists():
        st.warning(f"{rounds_path} not found.")
        return
    try:
        match = load_match_rounds(_load_rounds(rounds_path), match_id)
        series = win_probability_series(match)
        buys = None
        if use_buys:
            buys_path = str(RAW_DATA_DIR / "buys_combined.csv")
            buys = _load_buys(buys_path) if Path(buys_path).exists() else None
            if buys is None:
                st.warning(f"{buys_path} not found, so buy types are left out.")
        steps = step_through(match, buys)
    except (ValueError, InvalidStateError) as err:
        st.error(str(err))
        return
    team_a, team_b = series.attrs["team_a"], series.attrs["team_b"]
    st.pyplot(win_probability_figure(series), width=800)
    st.write(f"Team A is {team_a} (it attacked round 1); Team B is {team_b}.")
    st.subheader("Round by round")
    if len(steps) == len(series) - 1:  # series has one extra row: before round 1
        steps[f"{team_a} win % after"] = (100 * series["p_a"].iloc[1:]).round(1).to_numpy()
    st.dataframe(steps, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Valorant win probability", layout="wide")
    st.title("Valorant map win probability")
    state_tab, match_tab_ = st.tabs(["Game state", "Match history"])
    with state_tab:
        game_state_tab()
    with match_tab_:
        match_tab()


main()
