"""Presentation figures (dark theme), drawn from the winprob package
(run from the repo root):  python scripts/make_deck_figures.py <out_dir>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from winprob import predict as P
from winprob.economy.buy_type import BUY_TYPES
from winprob.tier1.convergence import convergence_study
from winprob.tier1.exact import solve_linear_system
from winprob.tier2.economy_data import ROUND_WIN_RATE
from winprob.tier3.regression_defaults import MAP_EFFECTS, default_p_atk

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

BG, INK, BODY, MUTED, GRID = "#18233B", "#EEF1F6", "#B9C3D6", "#8A96AD", "#2C3A5C"
ACCENT, ACCENT2 = "#6B8BCF", "#E8894A"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": BODY, "axes.labelcolor": BODY, "xtick.color": BODY, "ytick.color": BODY,
    "axes.edgecolor": GRID, "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("wrote", name)


# 1. Tier 1 win probability over every regulation score (team A started on attack, pool-average map)
p = default_p_atk(None)
table = solve_linear_system(p, "atk")
grid = np.full((13, 13), np.nan)
for a in range(13):
    for b in range(13):
        if (a, b) in table:
            grid[a, b] = table[(a, b)]
navy = LinearSegmentedColormap.from_list("navy_orange", [ACCENT2, "#2A3555", ACCENT])
fig, ax = plt.subplots(figsize=(9.2, 7.6))
im = ax.imshow(grid, origin="lower", cmap=navy, vmin=0, vmax=1)
for a in range(13):
    for b in range(13):
        v = grid[a, b]
        if not np.isnan(v):
            ax.text(b, a, f"{100 * v:.0f}", ha="center", va="center", fontsize=9.5, color=INK)
ax.set_xlabel("Team B rounds won")
ax.set_ylabel("Team A rounds won")
ax.set_xticks(range(13))
ax.set_yticks(range(13))
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Team A win probability")
cb.outline.set_visible(False)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
cb.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
save(fig, "tier1_heatmap.png")

# 2. Monte Carlo convergence (Requirement 6)
res = convergence_study(0.55, "atk", 5, 8, n_trials=10, rng=np.random.default_rng(1))
ns, err, slope = res["ns"], res["mean_errors"], res["slope"]
fig, ax = plt.subplots(figsize=(9.2, 6.2))
ref = err[0] * (ns / ns[0]) ** -0.5
ax.loglog(ns, ref, "--", color=MUTED, lw=1.5, label="Theory: slope −0.5")
ax.loglog(ns, err, "o-", color=ACCENT, lw=2.5, ms=8, label=f"Measured: slope {slope:.2f}")
ax.set_xlabel("Number of simulated maps, N")
ax.set_ylabel("Mean absolute error vs exact")
ax.grid(True, which="major", color=GRID, lw=0.8)
ax.legend(frameon=False, loc="upper right")
save(fig, "mc_convergence.png")
print("slope", slope)

# 3. Tier 2 buy-matchup round-win rates
labels = BUY_TYPES
m = np.array([[ROUND_WIN_RATE[(r, c)] for c in labels] for r in labels])
fig, ax = plt.subplots(figsize=(8.4, 7.0))
im = ax.imshow(m, cmap=navy, vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        fill = r"$\ast$" if {labels[i], labels[j]} == {"Eco", "Semi-eco"} else ""
        ax.text(j, i, f"{100 * m[i, j]:.0f}%{fill}", ha="center", va="center", fontsize=17, color=INK, weight="bold")
ax.set_xticks(range(4), labels)
ax.set_yticks(range(4), labels)
ax.set_xlabel("Defenders' buy")
ax.set_ylabel("Attackers' buy")
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
save(fig, "buy_matchups.png")

# 4. Attacker round-win rate by map (even gear)
maps = sorted(MAP_EFFECTS, key=lambda k: default_p_atk(k))
vals = [100 * default_p_atk(k) for k in maps]
fig, ax = plt.subplots(figsize=(9.2, 7.4))
colors = [ACCENT2 if k in ("fracture", "ascent") else ACCENT for k in maps]
ax.axvline(50, color=MUTED, lw=1.2)
pool = 100 * default_p_atk(None)
ax.axvline(pool, color=MUTED, lw=1.2, ls="--")
ax.hlines(range(len(maps)), 50, vals, color=GRID, lw=3)
ax.scatter(vals, range(len(maps)), s=150, color=colors, zorder=3)
for i, v in enumerate(vals):
    ax.text(v + (0.25 if v >= 50 else -0.25), i, f"{v:.1f}%", va="center",
            ha="left" if v >= 50 else "right", fontsize=14, color=INK)
ax.set_yticks(range(len(maps)), [k.title() for k in maps])
ax.set_xlim(45.5, 56.5)
ax.set_xlabel("Attackers' round-win rate with equal loadouts")
ax.text(49.9, len(maps) - 0.3, "50% = no edge", color=MUTED, fontsize=12, va="bottom", ha="right")
ax.text(pool + 0.1, len(maps) - 0.3, f"pool average {pool:.1f}%", color=MUTED, fontsize=12, va="bottom", ha="left")
ax.tick_params(axis="y", length=0)
ax.grid(False)
save(fig, "map_attack_rates.png")

# 5. Ult win boost by role (pp per ready ult)
roles = ["Controllers", "Initiators", "Duelists", "Sentinels"]
pp = [1.2, 1.7, 3.5, 4.3]
fig, ax = plt.subplots(figsize=(9.2, 5.6))
bars = ax.barh(roles, pp, color=[ACCENT, ACCENT, ACCENT, ACCENT2], height=0.6)
for b, v in zip(bars, pp):
    ax.text(v + 0.08, b.get_y() + b.get_height() / 2, f"+{v} pp", va="center", fontsize=16, color=INK)
ax.set_xlim(0, 5)
ax.set_xlabel("Round-win boost per ready ult (percentage points)")
ax.tick_params(axis="y", length=0)
save(fig, "ult_roles.png")

# 6. Same state, three tiers: 5-6 (you started on attack), full buys, 19k each
rows = []
for mp in ("ascent", "fracture"):
    t1 = P.predict_tier1(5, 6, map_name=mp).estimates[0].p_a
    t2 = P.predict_tier2(5, 6, buy_a="Full buy", buy_b="Full buy").estimates[0].p_a
    t3 = P.predict_tier3(5, 6, 19000, 19000, map_name=mp, n=100_000, seed=0).estimates[0].p_a
    rows.append((mp.title(), 100 * t1, 100 * t2, 100 * t3))
    print(mp, round(100 * t1, 1), round(100 * t2, 1), round(100 * t3, 1))
fig, ax = plt.subplots(figsize=(9.2, 6.0))
x = np.arange(3)
w = 0.36
for k, (name, *v) in enumerate(rows):
    col = ACCENT if k == 0 else ACCENT2
    bars = ax.bar(x + (k - 0.5) * w, v, w, color=col, label=name)
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, val + 1, f"{val:.1f}%", ha="center", fontsize=14, color=INK)
ax.set_xticks(x, ["Tier 1\nscore + side + map", "Tier 2\nbuys (no map)", "Tier 3\ncredits + map"])
ax.set_ylim(0, 60)
ax.set_ylabel("Team A win probability")
ax.axhline(50, color=MUTED, lw=1, ls="--")
ax.legend(frameon=False, loc="upper center", ncol=2)
save(fig, "tier_compare.png")
