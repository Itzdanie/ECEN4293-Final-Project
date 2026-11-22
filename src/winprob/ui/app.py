"""Streamlit demo UI over the same functions the CLI calls
(``winprob.predict`` and ``winprob.reporting.match``) - no modeling code
lives here.

Run with ``winprob ui`` (or ``streamlit run src/winprob/ui/app.py``).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from winprob import predict as P
from winprob.data.loading import RAW_DATA_DIR, load_buys, load_rounds
from winprob.economy.buy_type import BUY_TYPES
from winprob.reporting.match import (
    load_match_rounds,
    step_through,
    win_probability_figure,
    win_probability_series,
)
from winprob.tier1.states import ROUNDS_PER_HALF
from winprob.tier3.regression_defaults import MAP_EFFECTS, default_p_atk
from winprob.ults.roles import AGENTS
from winprob.validation import TEAM_CREDIT_CAP, InvalidStateError

TIER_LABELS = {
    1: "Tier 1: score and side only",
    2: "Tier 2: + buy type and loss streak",
    3: "Tier 3: + credits, ults, survivors (Monte Carlo)",
}
METHOD_LABELS = {"exact": "Exact", "mc": "Monte Carlo", "both": "Compare exact and Monte Carlo"}


def _pct(p: float) -> str:
    return f"{100 * p:.2f}%"


def _show_prediction(pred: P.Prediction, n: int, seed: int) -> None:
    for est in pred.estimates:
        label = "Exact" if est.method == "exact" else f"Monte Carlo (N={n:,}, seed={seed})"
        st.subheader(label)
        a, b = st.columns(2)
        a.metric("Team A", _pct(est.p_a))
        b.metric("Team B", _pct(est.p_b))
        if est.ci:
            st.caption(f"95% confidence interval for Team A: {_pct(est.ci[0])} to {_pct(est.ci[1])}")
    if pred.difference is not None:
        st.metric("Monte Carlo minus exact", f"{100 * pred.difference:+.3f} percentage points")
    if pred.tier == 1:
        st.info("Tier 1 uses only the score and side. It ignores the economy (for example who won the "
                "pistol), so use Tier 2 or 3 for analysis.")


def _team_agents(label: str, key: str) -> tuple[list[str] | None, list[str] | None]:
    """Optional five-agent picker with a ready-ult multiselect."""
    if not st.checkbox(f"Name {label}'s five agents", key=f"{key}_named"):
        return None, None
    cols = st.columns(5)
    defaults = ["omen", "raze", "jett", "sova", "cypher"]
    agents = [
        col.selectbox(f"Agent {i + 1}", AGENTS, index=AGENTS.index(defaults[i]), key=f"{key}_agent{i}")
        for i, col in enumerate(cols)
    ]
    ready = st.multiselect(f"{label}: ults ready", agents, key=f"{key}_ready")
    return agents, ready


def game_state_tab() -> None:
    left, right = st.columns([1, 1.2])
    with left:
        tier = st.radio("Model", list(TIER_LABELS), index=1, format_func=TIER_LABELS.get, key="tier")
        c1, c2 = st.columns(2)
        score_a = c1.number_input("Team A rounds", 0, 30, 5, key="score_a")
        score_b = c2.number_input("Team B rounds", 0, 30, 7, key="score_b")
        a_started = st.radio("Team A started on", ["atk", "def"], horizontal=True, key="a_started")
        map_name = st.selectbox("Map", [None, *sorted(MAP_EFFECTS)], format_func=lambda m: m or "(pool average)", key="map")

        args: dict = {}
        method = "exact"
        pistol_next = score_a + score_b in (0, ROUNDS_PER_HALF)
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
            b1, b2 = st.columns(2)
            buy_a = b1.selectbox("Team A buy", BUY_TYPES, index=3, key="buy_a")
            buy_b = b2.selectbox("Team B buy", BUY_TYPES, index=3, key="buy_b")
            s1, s2 = st.columns(2)
            streak_a = s1.slider("Team A loss streak", 0, 3, 0, key="streak_a")
            streak_b = s2.slider("Team B loss streak", 0, 3, 0, key="streak_b")
            args = dict(buy_a=buy_a, buy_b=buy_b, streak_a=streak_a, streak_b=streak_b)
        else:
            if pistol_next or score_a + score_b >= 2 * ROUNDS_PER_HALF:
                st.caption("Credits and ults reset this round, so the inputs below are ignored.")
            k1, k2 = st.columns(2)
            bank_a = k1.number_input("Team A credits (team total)", 0, TEAM_CREDIT_CAP, 19000, 500, key="bank_a")
            bank_b = k2.number_input("Team B credits (team total)", 0, TEAM_CREDIT_CAP, 19000, 500, key="bank_b")
            s1, s2 = st.columns(2)
            streak_a = s1.slider("Team A loss streak", 0, 3, 0, key="t3_streak_a")
            streak_b = s2.slider("Team B loss streak", 0, 3, 0, key="t3_streak_b")
            v1, v2 = st.columns(2)
            surv_a = v1.slider("Team A survivors", 0, 5, 0, key="surv_a")
            surv_b = v2.slider("Team B survivors", 0, 5, 0, key="surv_b")
            agents_a, ready_a = _team_agents("Team A", "a")
            agents_b, ready_b = _team_agents("Team B", "b")
            u1, u2 = st.columns(2)
            ults_a = u1.slider("Team A ults ready", 0, 5, 0, key="ults_a") if ready_a is None else 0
            ults_b = u2.slider("Team B ults ready", 0, 5, 0, key="ults_b") if ready_b is None else 0
            args = dict(
                bank_a=float(bank_a), bank_b=float(bank_b), streak_a=streak_a, streak_b=streak_b,
                survivors_a=surv_a, survivors_b=surv_b, a_started=a_started, map_name=map_name,
                agents_a=agents_a, agents_b=agents_b, ready_agents_a=ready_a, ready_agents_b=ready_b,
                ults_a=ults_a, ults_b=ults_b,
            )

        if tier < 3:
            method = st.radio("Method", list(METHOD_LABELS), format_func=METHOD_LABELS.get, key="method")
        n = seed = 0
        if tier == 3 or method != "exact":
            n = st.number_input("Monte Carlo playouts", 1_000, 1_000_000, 100_000, 10_000, key="n")
            seed = st.number_input("Seed", 0, 10_000, 0, key="seed")
        run_now = tier < 3 and method == "exact"
        if not run_now:
            run_now = st.button("Run", type="primary", key="run")

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
    st.pyplot(win_probability_figure(series))
    team_a, team_b = series.attrs["team_a"], series.attrs["team_b"]
    st.write(f"Team A is {team_a} (it attacked round 1); Team B is {team_b}.")
    st.subheader("Round by round")
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
