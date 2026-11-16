"""Agent -> sub-role -> role grouping for ultimates.

The sub-role groups are Daniel's own, chosen by how each ult is actually
used (e.g. "dive" duelist ults are kill-oriented, "flex" ones set up the
team). They form the middle layer of the hierarchy the win-boost fit
shrinks through: agent -> sub-role -> role. A sub-role's name starts with
its role's four-letter code (``duel``, ``init``, ``ctrl``, ``sent``).
"""

from __future__ import annotations

SUB_ROLE_AGENTS: dict[str, tuple[str, ...]] = {
    "duel_dive": ("hush", "raze", "neon", "jett"),
    "duel_flex": ("waylay", "yoru", "phoenix"),
    "duel_meh": ("iso", "reyna"),
    "init_info": ("fade", "kayo", "skye"),
    "init_team": ("breach", "gekko", "sova", "tejo"),
    "ctrl_space": ("astra", "harbor", "viper"),
    "ctrl_brim": ("brimstone",),
    "ctrl_meh": ("clove", "omen", "miks"),
    "sent_space": ("killjoy", "vyse", "cypher"),
    "sent_oneoff": ("deadlock", "sage"),
    "sent_other": ("chamber", "veto"),
}

AGENT_SUBROLE: dict[str, str] = {
    agent: sub for sub, agents in SUB_ROLE_AGENTS.items() for agent in agents
}
AGENTS: tuple[str, ...] = tuple(sorted(AGENT_SUBROLE))
SUB_ROLES: tuple[str, ...] = tuple(SUB_ROLE_AGENTS)
ROLES: tuple[str, ...] = ("ctrl", "duel", "init", "sent")


def role_of(sub_role: str) -> str:
    """The four-letter role code a sub-role belongs to."""
    return sub_role.split("_", 1)[0]


def check_known_agents(agents) -> None:
    """Raise a descriptive error if any agent has no sub-role assigned."""
    unknown = sorted(set(agents) - set(AGENT_SUBROLE))
    if unknown:
        raise ValueError(
            f"agents with no sub-role in winprob.ults.roles.SUB_ROLE_AGENTS: {unknown}. "
            "Add them to a sub-role (or a new one) before fitting."
        )
