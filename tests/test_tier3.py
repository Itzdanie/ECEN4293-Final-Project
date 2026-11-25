import numpy as np
import pytest

from winprob.economy.constants import (
    TEAM_CREDIT_CAP,
    TEAM_LOSS_BONUS_BY_STREAK,
    TEAM_PISTOL_ROUND_CREDITS,
    TEAM_ROUND_WIN_REWARD,
)
from winprob.tier1.exact import solve_linear_system
from winprob.tier1.states import side_for_team_a
from winprob.tier3 import simulate_win_probability
from winprob.tier3.economy_sim import apply_round_income, decide_spend, draw_survivors
from winprob.tier3.monte_carlo import _a_is_atk, _no_future_value_in_saving, _sigmoid
from winprob.tier3.regression_defaults import (
    BASE_INTERCEPT,
    DELTA_LOADOUT_COEF,
    HALF2_COEF,
    MAP_EFFECTS,
    POOL_AVERAGE_MAP_EFFECT,
)

INTERCEPT = BASE_INTERCEPT + POOL_AVERAGE_MAP_EFFECT  # pool-average, matching map_name=None
from winprob.tier3.ult_defaults import AGENT_PARAMS
from winprob.tier3.ult_model import GENERIC_BOOST, SLOTS, initial_ready, slot_params, step_ults

# 10+ states for Requirement 7, spanning scores, the overtime entry, and a pistol boundary.
SAMPLE_STATES = [
    (0, 0), (5, 5), (3, 8), (8, 3), (11, 5), (5, 11), (12, 12), (9, 9), (1, 0), (7, 7),
]


# --- Requirement 7: economy/ults off -> matches Tier 1 exactly -------------


@pytest.mark.parametrize("score_a,score_b", SAMPLE_STATES)
def test_economy_off_matches_tier1_exact(score_a, score_b):
    p_atk = 0.55
    a_started = "atk"
    exact = solve_linear_system(p_atk, a_started)[(score_a, score_b)]

    rng = np.random.default_rng(1)
    p_hat, ci_low, ci_high = simulate_win_probability(
        score_a=score_a,
        score_b=score_b,
        bank_a=4000,
        bank_b=4000,
        ults_a=0,
        ults_b=0,
        streak_a=0,
        streak_b=0,
        n=100_000,
        rng=rng,
        a_started=a_started,
        economy_and_ults_affect_outcome=False,
        p_atk=p_atk,
    )
    assert abs(p_hat - exact) < 0.02
    assert ci_low <= p_hat <= ci_high


def test_clinched_states_are_exact_without_simulating():
    rng = np.random.default_rng(0)
    p_hat, lo, hi = simulate_win_probability(
        score_a=13, score_b=5, bank_a=0, bank_b=0, ults_a=0, ults_b=0,
        streak_a=0, streak_b=0, n=100, rng=rng,
    )
    assert p_hat == lo == hi == 1.0


# --- economy effect, isolated from the multi-round attacker-swap confound --


def test_a_bigger_loadout_edge_increases_the_attackers_win_probability():
    # Checked in isolation (one round's logit), not a full map simulation,
    # since over a whole map team A's side flips at round 13 and a tied
    # score like 5-5 can show A *disadvantaged* even with better economy
    # that round - verified this matches Tier 1's own model at (5,5) with
    # p_atk > 0.5 (same attacker-favored effect), not a bug.
    logit_even = INTERCEPT + DELTA_LOADOUT_COEF * 0.0
    logit_a_rich = INTERCEPT + DELTA_LOADOUT_COEF * 9.0  # A (attacking) has a 9000-credit edge
    assert _sigmoid(logit_a_rich) > _sigmoid(logit_even)


def test_ult_boost_sensitivity_sweep_is_monotonic():
    # CLAUDE.md asks for a sensitivity analysis from 0 up to the
    # lift_pp-derived logit boost - checked in isolation (one round),
    # since over a full match the effect dilutes the same way the
    # economy's initial lump-sum advantage does (see memory/writeup).
    scales = [0.0, 0.25, 0.5, 0.75, 1.0]
    probs = [_sigmoid(INTERCEPT + scale * GENERIC_BOOST) for scale in scales]
    assert all(b > a for a, b in zip(probs, probs[1:]))


def test_forced_eco_after_a_lost_full_buy_duel_gives_a_real_edge():
    # Both teams spend everything on a contested full buy; A wins. B should
    # go into a genuine eco next round (not a half-measure draw it can't
    # really afford), giving A a real loadout edge - this is the exact
    # swing-round dynamic the bank-conditioned spend decision fixes.
    rng = np.random.default_rng(0)
    n = 20_000
    bank_a = np.full(n, 20000.0)
    bank_b = np.full(n, 20000.0)
    spend_a = decide_spend(bank_a)
    spend_b = decide_spend(bank_b)
    a_wins = np.ones(n, dtype=bool)

    bank_a_next = apply_round_income(bank_a - spend_a, a_wins, np.zeros(n, dtype=np.int64), rng)
    bank_b_next = apply_round_income(bank_b - spend_b, ~a_wins, np.ones(n, dtype=np.int64), rng)
    survivors_a = draw_survivors(rng, a_wins)
    survivors_b = draw_survivors(rng, ~a_wins)

    eff_a = decide_spend(bank_a_next) + survivors_a * 3000
    eff_b = decide_spend(bank_b_next) + survivors_b * 3000
    delta_loadout = ((eff_a - eff_b) / 1000.0).mean()

    assert delta_loadout > 5.0  # a real, large edge - not the ~1.8 the old history-based draw gave


def test_a_is_atk_matches_tier1_side_rule():
    # one swap after round 12, a_started="atk"
    assert _a_is_atk(np.array([0, 11]), True).tolist() == [True, True]
    assert _a_is_atk(np.array([12, 20]), True).tolist() == [False, False]
    # overtime starts on the starting side, then swaps every round
    assert _a_is_atk(np.array([24, 25, 26]), True).tolist() == [True, False, True]
    # and agrees with Tier 1's scalar rule everywhere, both starting sides
    played = np.arange(40)
    for started in ("atk", "def"):
        expected = [side_for_team_a(int(n), started) == "atk" for n in played]
        assert _a_is_atk(played, started == "atk").tolist() == expected


# --- economy constants (checked against real data - see constants.py) -----


def test_loss_bonus_increases_with_streak_and_caps_at_three():
    assert TEAM_LOSS_BONUS_BY_STREAK[1] < TEAM_LOSS_BONUS_BY_STREAK[2] < TEAM_LOSS_BONUS_BY_STREAK[3]
    assert TEAM_LOSS_BONUS_BY_STREAK[3] < TEAM_ROUND_WIN_REWARD  # winning always pays more than any loss bonus


def test_map_name_none_matches_pool_average():
    rng1 = np.random.default_rng(9)
    p_default, _, _ = simulate_win_probability(
        score_a=5, score_b=5, bank_a=19000, bank_b=19000, ults_a=0, ults_b=0,
        streak_a=0, streak_b=0, n=20_000, rng=rng1, ult_boost_scale=0.0,
    )
    rng2 = np.random.default_rng(9)
    p_explicit_none, _, _ = simulate_win_probability(
        score_a=5, score_b=5, bank_a=19000, bank_b=19000, ults_a=0, ults_b=0,
        streak_a=0, streak_b=0, n=20_000, rng=rng2, ult_boost_scale=0.0, map_name=None,
    )
    assert p_default == p_explicit_none


def test_map_name_changes_the_single_round_attacker_edge_correctly():
    # Checked as one round's logit, not a full map simulation: over a whole
    # map, a bigger attacker edge can actually hurt the team that holds it
    # only briefly before switching to defense for a longer stretch (the
    # same effect confirmed in Tier 1 at 5-5 with p_atk > 0.5) - so this
    # has to be isolated from that confound to test cleanly.
    # abyss has the single highest real attacker win rate of the 12 maps;
    # ascent the most negative.
    from winprob.tier3.regression_defaults import BASE_INTERCEPT, MAP_EFFECTS

    logit_abyss = BASE_INTERCEPT + MAP_EFFECTS["abyss"]
    logit_ascent = BASE_INTERCEPT + MAP_EFFECTS["ascent"]
    assert _sigmoid(logit_abyss) > _sigmoid(logit_ascent)


def test_unknown_map_name_raises_a_clear_error():
    with pytest.raises(ValueError, match="isn't one of the maps"):
        simulate_win_probability(
            score_a=5, score_b=5, bank_a=19000, bank_b=19000, ults_a=0, ults_b=0,
            streak_a=0, streak_b=0, n=100, rng=np.random.default_rng(0), map_name="not_a_real_map",
        )


def test_all_12_maps_are_present_in_map_effects():
    expected = {
        "abyss", "ascent", "bind", "breeze", "corrode", "fracture",
        "haven", "lotus", "pearl", "split", "summit", "sunset",
    }
    assert set(MAP_EFFECTS) == expected


def test_no_future_value_in_saving_round_12_always_forces():
    # whoever's playing it, round 12 (total == 11) always forces - round 13
    # resets the economy regardless of outcome
    assert _no_future_value_in_saving(np.array([6]), np.array([5]), np.array([11]))[0]
    assert _no_future_value_in_saving(np.array([0]), np.array([11]), np.array([11]))[0]


def test_no_future_value_in_saving_match_point_forces():
    # my loss would let the opponent clinch at 13 - no future to save for
    assert _no_future_value_in_saving(np.array([9]), np.array([12]), np.array([21]))[0]
    # opponent isn't close enough yet - normal round, no reason to force
    assert not _no_future_value_in_saving(np.array([12]), np.array([9]), np.array([21]))[0]


def test_no_future_value_in_saving_tie_into_overtime_forces():
    # I'm at 12, opponent at 11: winning clinches, but losing just ties
    # 12-12 and overtime resets credits anyway - force either way
    assert _no_future_value_in_saving(np.array([12]), np.array([11]), np.array([23]))[0]


def test_no_future_value_in_saving_is_false_for_an_ordinary_round():
    assert not _no_future_value_in_saving(np.array([6]), np.array([4]), np.array([10]))[0]


def test_force_overrides_eco_to_spend_the_full_bank():
    bank = np.array([2500.0])
    assert decide_spend(bank, np.array([False]))[0] == pytest.approx(2228.0)  # normal eco
    assert decide_spend(bank, np.array([True]))[0] == pytest.approx(2500.0)  # forced: spend it all


def test_decide_spend_is_driven_by_bank_not_history():
    # a broke team spends like a real eco, not a half-measure draw from history
    assert decide_spend(np.array([3278.0]))[0] == pytest.approx(2228.0)
    # a rich team spends like a real full buy
    assert decide_spend(np.array([20000.0]))[0] == pytest.approx(19020.0)
    # spend is capped by bank even within a bucket
    assert decide_spend(np.array([6000.0]))[0] == pytest.approx(6000.0)  # Semi-eco bucket (7232) > bank


def test_draw_survivors_matches_real_win_loss_distributions():
    rng = np.random.default_rng(0)
    n = 50_000
    won_survivors = draw_survivors(rng, np.ones(n, dtype=bool))
    lost_survivors = draw_survivors(rng, np.zeros(n, dtype=bool))
    assert abs(won_survivors.mean() - 2.85) < 0.05
    assert abs(lost_survivors.mean() - 0.17) < 0.05
    assert (won_survivors >= 0).all() and (won_survivors <= 5).all()


def test_apply_round_income_respects_the_credit_cap():
    rng = np.random.default_rng(0)
    bank = np.full(1000, TEAM_CREDIT_CAP - 100.0)
    won = np.ones(1000, dtype=bool)
    new_bank = apply_round_income(
        bank, won, np.zeros(1000, dtype=np.int64), rng
    )
    assert (new_bank <= TEAM_CREDIT_CAP).all()


# --- ult model ---------------------------------------------------------------


def _fresh_state(n, ready_first=0):
    ready = np.zeros((n, SLOTS), dtype=bool)
    ready[:, :ready_first] = True
    return ready, np.zeros((n, SLOTS), dtype=np.int64), np.zeros((n, SLOTS), dtype=bool)


def test_a_ready_unused_ult_stays_ready_and_a_used_one_is_empty_next_round():
    # the two real-data facts the step relies on (37k and 53k observed transitions)
    rng = np.random.default_rng(0)
    params = slot_params(["omen"] * SLOTS)
    ready, age, post = _fresh_state(5000, ready_first=SLOTS)
    used, next_ready, _, next_post = step_ults(rng, ready, age, post, params)
    assert not next_ready[used].any()  # spent -> never ready next round
    assert next_ready[~used].all()  # ready and unused -> stays ready
    assert next_post[used].all()


def test_nothing_is_used_from_a_team_with_no_ready_ults():
    rng = np.random.default_rng(0)
    used, _, _, _ = step_ults(rng, *_fresh_state(500), slot_params(None))
    assert not used.any()


def test_use_rate_differs_by_agent_as_in_the_real_data():
    rng = np.random.default_rng(0)
    ready, age, post = _fresh_state(20_000, ready_first=SLOTS)
    omen_used, *_ = step_ults(rng, ready, age, post, slot_params(["omen"] * SLOTS))
    raze_used, *_ = step_ults(rng, ready, age, post, slot_params(["raze"] * SLOTS))
    assert abs(omen_used.mean() - AGENT_PARAMS["omen"][1]) < 0.01
    assert abs(raze_used.mean() - AGENT_PARAMS["raze"][1]) < 0.01
    assert raze_used.mean() > omen_used.mean() + 0.2  # Omen's is held far more often


def test_slot_params_validation_errors_are_descriptive():
    with pytest.raises(ValueError, match="unknown agents"):
        slot_params(["omen", "raze", "jett", "sova", "not_an_agent"])
    with pytest.raises(ValueError, match="5 agents"):
        slot_params(["omen", "raze"])
    with pytest.raises(ValueError, match="between 0 and 5"):
        initial_ready(6, None)
    with pytest.raises(ValueError, match="disagrees"):
        initial_ready(2, [True, False, False, False, False])


def test_ready_ults_cannot_be_given_at_a_pistol_round():
    for score in [(0, 0), (6, 6)]:
        with pytest.raises(ValueError, match="pistol and overtime"):
            simulate_win_probability(
                score_a=score[0], score_b=score[1], bank_a=800, bank_b=800, ults_a=1, ults_b=0,
                streak_a=0, streak_b=0, n=10, rng=np.random.default_rng(0),
            )


def test_a_team_with_more_ready_ults_has_the_edge_when_all_else_is_equal():
    # whole-map effect is small, so use many playouts at a mid-half state (round 11)
    kw = dict(score_a=5, score_b=5, bank_a=19000, bank_b=19000, streak_a=0, streak_b=0,
              n=200_000, a_started="def", ults_b=0)
    p_more, _, _ = simulate_win_probability(**kw, ults_a=0, rng=np.random.default_rng(3))
    ready = [True] * SLOTS
    p_ult, _, _ = simulate_win_probability(
        **{**kw, "ults_a": SLOTS}, ready_a=ready, agents_a=["chamber", "cypher", "raze", "jett", "veto"],
        rng=np.random.default_rng(3),
    )
    assert p_ult > p_more


def test_ults_reset_to_not_ready_at_the_start_of_overtime():
    # in OT every round starts with nothing ready, so ult state can't matter:
    # a 12-12 start is the same with or without ult knobs on.
    kw = dict(score_a=12, score_b=12, bank_a=0, bank_b=0, ults_a=0, ults_b=0,
              streak_a=0, streak_b=0, n=20_000)
    p0, _, _ = simulate_win_probability(**kw, rng=np.random.default_rng(5), ult_boost_scale=0.0)
    p1, _, _ = simulate_win_probability(**kw, rng=np.random.default_rng(5), ult_boost_scale=1.0)
    assert p0 == p1


# --- team-level economy scale (constants.py: per-player values x 5) -------------


def test_a_won_round_pays_team_scale_income_not_a_single_players():
    rng = np.random.default_rng(0)
    n = 5000
    bank = apply_round_income(
        np.zeros(n), np.ones(n, dtype=bool), np.zeros(n, dtype=np.int64), rng,
    )
    # 5 x 3000 for the win + ~4 kills x 200
    assert bank.mean() == pytest.approx(TEAM_ROUND_WIN_REWARD + 4.0 * 200, abs=60)


def test_a_team_can_afford_a_full_buy_after_a_couple_of_wins():
    rng = np.random.default_rng(1)
    n = 2000
    bank = np.zeros(n)
    won = np.ones(n, dtype=bool)
    for _ in range(2):  # pistol credits spent, then two straight wins
        bank = apply_round_income(bank, won, np.zeros(n, dtype=np.int64), rng)
    assert (decide_spend(bank) > 15_000).mean() > 0.95  # a Full buy, not stuck on an eco


def test_pistol_round_starts_with_the_real_team_total():
    assert TEAM_PISTOL_ROUND_CREDITS == 4000  # real data: round-1 team credits are 5 x 800


def test_simulated_attacker_loadout_edge_matches_the_real_data_not_a_built_in_bias(monkeypatch):
    # Real VCT gun rounds: attacker minus defender loadout averages +0.13k (first
    # half) and +0.34k (second half) credits. With the spike-plant bonus in Tier 3's
    # income it came out ~+1.75k / +1.89k - on top of the attacker edge already in the
    # fitted intercept, a double count - so the plant bonus is left out of income.
    from winprob.tier3 import monte_carlo as mc

    seen = {"total": None, "gaps": []}
    real_a_is_atk, real_sigmoid = mc._a_is_atk, mc._sigmoid

    def spy_a_is_atk(total, started):
        seen["total"] = total.copy()
        return real_a_is_atk(total, started)

    def spy_sigmoid(z):
        total = seen["total"]
        half2 = (total >= 12).astype(float)
        gap = (z - BASE_INTERCEPT - MAP_EFFECTS["haven"] - HALF2_COEF * half2) / DELTA_LOADOUT_COEF
        keep = (total + 1 <= 24) & ~np.isin(total + 1, (1, 13))  # gun rounds, regulation
        seen["gaps"].append(gap[keep])
        return real_sigmoid(z)

    monkeypatch.setattr(mc, "_a_is_atk", spy_a_is_atk)
    monkeypatch.setattr(mc, "_sigmoid", spy_sigmoid)
    simulate_win_probability(
        score_a=0, score_b=0, bank_a=0, bank_b=0, ults_a=0, ults_b=0, streak_a=0, streak_b=0,
        n=6000, rng=np.random.default_rng(0), ult_boost_scale=0.0, map_name="haven",
    )
    mean_gap_thousands = np.concatenate(seen["gaps"]).mean()
    assert abs(mean_gap_thousands) < 0.6  # real: 0.13-0.34; with the plant bonus: ~1.8
