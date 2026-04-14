"""
Random Market Generation
=========================

Utilities for generating random stable-matching instances with various
preference correlation structures.

Correlation structures:
- **Uniform**: Each agent's preference list is an independent uniformly
  random permutation.
- **Correlated (master list)**: A global "quality" ranking exists; agents'
  preferences are noisy perturbations of this ranking.
- **Tiered**: Agents are divided into tiers; they prefer higher-tier
  partners but are indifferent within tiers.
"""

from __future__ import annotations

import random


def random_market(
    n: int,
    seed: int | None = None,
    proposer_prefix: str = "m",
    receiver_prefix: str = "w",
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Generate a uniformly random complete-list market.

    Parameters
    ----------
    n : int
        Number of agents on each side.
    seed : int or None
        Random seed for reproducibility.
    proposer_prefix, receiver_prefix : str
        Name prefixes for proposers and receivers.

    Returns
    -------
    tuple of dict
        ``(proposer_prefs, receiver_prefs)`` with complete preference lists.
    """
    rng = random.Random(seed)
    proposers = [f"{proposer_prefix}{i+1}" for i in range(n)]
    receivers = [f"{receiver_prefix}{i+1}" for i in range(n)]

    proposer_prefs: dict[str, list[str]] = {}
    for p in proposers:
        order = list(receivers)
        rng.shuffle(order)
        proposer_prefs[p] = order

    receiver_prefs: dict[str, list[str]] = {}
    for r in receivers:
        order = list(proposers)
        rng.shuffle(order)
        receiver_prefs[r] = order

    return proposer_prefs, receiver_prefs


def correlated_market(
    n: int,
    noise: float = 0.3,
    seed: int | None = None,
    proposer_prefix: str = "m",
    receiver_prefix: str = "w",
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Generate a market with correlated preferences (master-list model).

    Each agent has an objective "quality" score.  Preferences are formed
    by sorting agents by ``quality + noise * epsilon`` where ``epsilon``
    is standard normal.

    Parameters
    ----------
    n : int
        Number of agents on each side.
    noise : float
        Standard deviation of the perturbation.  ``noise=0`` gives identical
        preferences; large noise approaches uniform random.
    seed : int or None
        Random seed.

    Returns
    -------
    tuple of dict
        ``(proposer_prefs, receiver_prefs)``.
    """
    rng = random.Random(seed)
    proposers = [f"{proposer_prefix}{i+1}" for i in range(n)]
    receivers = [f"{receiver_prefix}{i+1}" for i in range(n)]

    # Objective quality scores.
    p_quality = {p: rng.gauss(0, 1) for p in proposers}
    r_quality = {r: rng.gauss(0, 1) for r in receivers}

    proposer_prefs: dict[str, list[str]] = {}
    for p in proposers:
        # Each proposer ranks receivers by quality + noise.
        scores = {r: r_quality[r] + noise * rng.gauss(0, 1) for r in receivers}
        proposer_prefs[p] = sorted(receivers, key=lambda r: -scores[r])

    receiver_prefs: dict[str, list[str]] = {}
    for r in receivers:
        scores = {p: p_quality[p] + noise * rng.gauss(0, 1) for p in proposers}
        receiver_prefs[r] = sorted(proposers, key=lambda p: -scores[p])

    return proposer_prefs, receiver_prefs


def tiered_market(
    n: int,
    n_tiers: int = 3,
    seed: int | None = None,
    proposer_prefix: str = "m",
    receiver_prefix: str = "w",
) -> tuple[dict[str, list[list[str]]], dict[str, list[list[str]]]]:
    """Generate a tiered market with weak preferences (ties).

    Agents are divided into tiers.  Each agent prefers higher-tier partners
    but is indifferent among agents in the same tier.

    Parameters
    ----------
    n : int
        Number of agents on each side.
    n_tiers : int
        Number of quality tiers.
    seed : int or None
        Random seed.

    Returns
    -------
    tuple of dict
        ``(proposer_prefs, receiver_prefs)`` with tied preference lists.
        Each value is ``[[tier_1_agents], [tier_2_agents], ...]``.
    """
    rng = random.Random(seed)
    proposers = [f"{proposer_prefix}{i+1}" for i in range(n)]
    receivers = [f"{receiver_prefix}{i+1}" for i in range(n)]

    # Assign tiers.
    p_tier = {p: rng.randint(0, n_tiers - 1) for p in proposers}
    r_tier = {r: rng.randint(0, n_tiers - 1) for r in receivers}

    proposer_prefs: dict[str, list[list[str]]] = {}
    for p in proposers:
        # Group receivers by tier, order tiers from best (0) to worst.
        tiers: dict[int, list[str]] = {t: [] for t in range(n_tiers)}
        for r in receivers:
            tiers[r_tier[r]].append(r)
        # Shuffle within each tier.
        groups = []
        for t in range(n_tiers):
            if tiers[t]:
                tier_list = list(tiers[t])
                rng.shuffle(tier_list)
                groups.append(tier_list)
        proposer_prefs[p] = groups

    receiver_prefs: dict[str, list[list[str]]] = {}
    for r in receivers:
        tiers_map: dict[int, list[str]] = {t: [] for t in range(n_tiers)}
        for p in proposers:
            tiers_map[p_tier[p]].append(p)
        groups = []
        for t in range(n_tiers):
            if tiers_map[t]:
                tier_list = list(tiers_map[t])
                rng.shuffle(tier_list)
                groups.append(tier_list)
        receiver_prefs[r] = groups

    return proposer_prefs, receiver_prefs


def random_hospital_market(
    n_residents: int,
    n_hospitals: int,
    min_quota: int = 1,
    max_quota: int = 3,
    list_length: int | None = None,
    seed: int | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, int]]:
    """Generate a random hospital-resident market.

    Parameters
    ----------
    n_residents : int
        Number of residents.
    n_hospitals : int
        Number of hospitals.
    min_quota, max_quota : int
        Range for hospital quotas.
    list_length : int or None
        Maximum length of preference lists.  If None, lists are complete.
    seed : int or None
        Random seed.

    Returns
    -------
    tuple
        ``(resident_prefs, hospital_prefs, quotas)``.
    """
    rng = random.Random(seed)
    residents = [f"r{i+1}" for i in range(n_residents)]
    hospitals = [f"h{i+1}" for i in range(n_hospitals)]

    quotas = {h: rng.randint(min_quota, max_quota) for h in hospitals}

    max_len_r = list_length or n_hospitals
    max_len_h = list_length or n_residents

    resident_prefs: dict[str, list[str]] = {}
    for r in residents:
        order = list(hospitals)
        rng.shuffle(order)
        resident_prefs[r] = order[:max_len_r]

    hospital_prefs: dict[str, list[str]] = {}
    for h in hospitals:
        order = list(residents)
        rng.shuffle(order)
        hospital_prefs[h] = order[:max_len_h]

    return resident_prefs, hospital_prefs, quotas


def incomplete_market(
    n: int,
    accept_prob: float = 0.7,
    seed: int | None = None,
    proposer_prefix: str = "m",
    receiver_prefix: str = "w",
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Generate a market with incomplete preference lists.

    Each potential partner is included in the list with probability
    ``accept_prob``.  This can result in some agents being unmatched.

    Parameters
    ----------
    n : int
        Number of agents on each side.
    accept_prob : float
        Probability that each partner is acceptable.
    seed : int or None
        Random seed.

    Returns
    -------
    tuple of dict
        ``(proposer_prefs, receiver_prefs)`` with possibly incomplete lists.
    """
    rng = random.Random(seed)
    proposers = [f"{proposer_prefix}{i+1}" for i in range(n)]
    receivers = [f"{receiver_prefix}{i+1}" for i in range(n)]

    proposer_prefs: dict[str, list[str]] = {}
    for p in proposers:
        acceptable = [r for r in receivers if rng.random() < accept_prob]
        rng.shuffle(acceptable)
        proposer_prefs[p] = acceptable

    receiver_prefs: dict[str, list[str]] = {}
    for r in receivers:
        acceptable = [p for p in proposers if rng.random() < accept_prob]
        rng.shuffle(acceptable)
        receiver_prefs[r] = acceptable

    return proposer_prefs, receiver_prefs
