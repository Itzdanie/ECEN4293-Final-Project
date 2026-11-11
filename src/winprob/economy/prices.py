"""Gun and armor prices (credits).

Seeded from ``economy_guns.csv`` plus public patch notes, and checked by
web search for price changes across patches 11.11-13.09 (the window our
Buys data covers): none were found for existing guns or armor - the only
changes in that window are new items being added (Bandit in patch 12.00,
Warden in patch 13.06), which show up in the data naturally from their
release patch onward. So a single flat table is correct for this data,
not an approximation - no per-patch price variation to key on.
"""

from __future__ import annotations

GUN_PRICES: dict[str, int] = {
    "classic": 0,
    "shorty": 300,
    "frenzy": 450,
    "ghost": 500,
    "bandit": 600,
    "sheriff": 800,
    "stinger": 1100,
    "spectre": 1600,
    "bucky": 850,
    "judge": 1850,
    "bulldog": 2050,
    "guardian": 2250,
    "phantom": 2900,
    "vandal": 2900,
    "warden": 2900,
    "marshal": 950,
    "outlaw": 2400,
    "operator": 4700,
    "ares": 1600,
    "odin": 3200,
}

ARMOR_PRICES: dict[str, int] = {
    "light": 400,
    "regen": 650,
    "heavy": 1000,
}


def gun_price(gun: object) -> int:
    """Price of ``gun``, or 0 if missing/unrecognized (no gun bought/equipped)."""
    return GUN_PRICES.get(gun, 0)


def armor_price(armor: object) -> int:
    """Price of ``armor``, or 0 if missing/unrecognized (no armor bought)."""
    return ARMOR_PRICES.get(armor, 0)
