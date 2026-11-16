"""Real-data ultimate constants for Tier 3, hardcoded from
``scripts/fit_ult_defaults.py`` (same pattern as ``regression_defaults``:
Tier 3 doesn't refit on every run).

From the per-player ult-readiness export (17,770 gun rounds, 1,086 matches,
league + international; rounds 12/24, pistols, OT and incomplete team-rounds
excluded). Penalties chosen by 5-fold CV over matches: lam_sub = 30000
(the top of the grid, i.e. the sub-role layer shrinks to ~0 - the data
cannot separate Daniel's sub-roles from their parent role) and lam_agent = 300.

``AGENT_PARAMS[agent] = (charge_scale, use_prob, boost)``:

- ``charge_scale``: multiplies the pooled charge hazard (``START_SHAPE`` after
  a half starts, ``POST_SHAPE`` after the ult was used), indexed by rounds
  since the slot last emptied.
- ``use_prob``: P(a ready ult is used this round).
- ``boost``: logit effect of that agent's ult being ready on the round
  (role + sub-role deviation + agent deviation; ~+0.1 logit is ~+2.4pp near
  50%). Association, not cause.

``GENERIC`` is used for a slot whose agent isn't specified: the pooled fit.
"""

from __future__ import annotations

START_SHAPE = [0.0229, 0.0338, 0.0665, 0.1425, 0.2344, 0.2824, 0.2383]
POST_SHAPE = [0.1342, 0.0958, 0.1533, 0.2245, 0.2468]
GENERIC = {"charge_scale": 1.0, "use_prob": 0.5724, "boost": 0.0971}
AGENT_PARAMS = {
    'astra': (1.169, 0.456, 0.0169),
    'breach': (0.826, 0.709, 0.0345),
    'brimstone': (0.867, 0.627, 0.0620),
    'chamber': (0.998, 0.658, 0.2033),
    'clove': (0.921, 0.656, 0.0450),
    'cypher': (0.985, 0.526, 0.1925),
    'deadlock': (0.886, 0.707, 0.1568),
    'fade': (0.925, 0.657, 0.0536),
    'gekko': (1.241, 0.686, 0.0700),
    'harbor': (1.049, 0.609, 0.0422),
    'hush': (0.990, 0.604, 0.1402),
    'iso': (1.109, 0.742, 0.1364),
    'jett': (0.978, 0.635, 0.1659),
    'kayo': (0.929, 0.630, 0.0557),
    'killjoy': (0.739, 0.595, 0.1767),
    'miks': (0.922, 0.636, 0.0474),
    'neon': (1.065, 0.548, 0.0765),
    'omen': (1.219, 0.465, 0.0693),
    'phoenix': (1.506, 0.698, 0.1322),
    'raze': (0.843, 0.744, 0.1845),
    'reyna': (1.388, 0.557, 0.1401),
    'sage': (1.238, 0.469, 0.1543),
    'skye': (0.793, 0.736, 0.1177),
    'sova': (1.016, 0.614, 0.0421),
    'tejo': (0.772, 0.678, 0.1131),
    'veto': (1.117, 0.595, 0.1951),
    'viper': (0.843, 0.488, 0.0301),
    'vyse': (0.888, 0.575, 0.1293),
    'waylay': (1.015, 0.616, 0.1063),
    'yoru': (0.954, 0.601, 0.1791),
}
