"""Economy constants, checked against patch notes and against the real
data (CLAUDE.md requires both).

Validation method: reconstructed each team's per-round income as
``(next round's spent + bank) - this round's bank`` from Buys, then
isolated each constant by controlling for the others - low starting bank
(so the credit cap doesn't distort it), plant_time specifically (not
``outcome``, which only says how a round ended and can't tell a clean
kill-based win from a plant-then-kill win), and the *actual* consecutive
loss count (``streak entering the round + 1`` if the round was also lost -
an early pass without that ``+ 1`` looked like it needed higher constants
across the board, which turned out to be an off-by-one in loss-streak
counting, not a real discrepancy). Every value below matched exactly once
those were accounted for:

  - Round win:                3000 credits/player (median per-player
    income minus kill credits, wins with no plant, low starting bank)
  - Loss bonus, 1 loss:       1900
  - Loss bonus, 2 losses:     2400
  - Loss bonus, 3+ losses:    2900 (caps here)
  - Kill reward:               200 credits/kill
  - Spike plant bonus:         300 credits/player (plant_time not null:
    median income exactly 300 higher than a no-plant win of the same type)
  - Credit cap:                9000 (bank alone never exceeds this; spent+bank
    together can, briefly, from an income spike before it's spent down)
  - Pistol-round (rounds 1 and 13) starting credits: 800 (median exactly;
    a few players show more from carried-over loadout value on alt accounts
    or data quirks, but the mode and median are unambiguous)

Overtime's fixed credit amount (both teams full reset each OT round, no
carryover) wasn't independently checked against our data - real matches
rarely reach deep overtime, so the sample is too thin to validate reliably
- but it is a specific, consistent claim across patch guides: 5000
credits/player each overtime round.
"""

from __future__ import annotations

ROUND_WIN_REWARD = 3000
LOSS_BONUS_BY_STREAK = {1: 1900, 2: 2400, 3: 2900}  # streak capped at 3
MAX_LOSS_STREAK = 3

KILL_REWARD = 200
PLANT_BONUS = 300

CREDIT_CAP = 9000
PISTOL_ROUND_CREDITS = 800
OVERTIME_ROUND_CREDITS = 5000  # not independently data-checked - see module docstring
