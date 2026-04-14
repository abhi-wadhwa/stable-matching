"""
Stability Checker
=================

Determines whether a matching is stable by exhaustively checking for
blocking pairs.

A pair ``(m, w)`` **blocks** matching ``mu`` if:
  1. ``m`` prefers ``w`` to ``mu(m)`` (or ``m`` is unmatched), AND
  2. ``w`` prefers ``m`` to ``mu(w)`` (or ``w`` is unmatched).

A matching is **stable** iff it admits no blocking pair.
"""

from __future__ import annotations


def _rank_map(prefs: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """Build ``{agent: {other: rank}}`` lookup.  Lower rank = more preferred."""
    return {
        agent: {other: idx for idx, other in enumerate(plist)}
        for agent, plist in prefs.items()
    }


def find_blocking_pairs(
    matching: dict[str, str],
    proposer_prefs: dict[str, list[str]],
    receiver_prefs: dict[str, list[str]],
) -> list[tuple[str, str]]:
    """Find all blocking pairs for a given matching.

    Parameters
    ----------
    matching : dict
        ``{proposer: receiver}`` -- the matching to check.
    proposer_prefs : dict
        Strict proposer preferences (most preferred first).
    receiver_prefs : dict
        Strict receiver preferences (most preferred first).

    Returns
    -------
    list of tuple
        Each tuple ``(proposer, receiver)`` is a blocking pair.
    """
    p_rank = _rank_map(proposer_prefs)
    r_rank = _rank_map(receiver_prefs)

    # Inverse matching: receiver -> proposer.
    inv_matching: dict[str, str | None] = {}
    for p, r in matching.items():
        inv_matching[r] = p
    # Ensure all receivers are in the inverse mapping.
    for r in receiver_prefs:
        if r not in inv_matching:
            inv_matching[r] = None

    blocking: list[tuple[str, str]] = []

    for m in proposer_prefs:
        current_r = matching.get(m)
        # Rank of current partner for m (infinity if unmatched).
        if current_r:
            m_current_rank = p_rank[m].get(current_r, len(proposer_prefs[m]))
        else:
            m_current_rank = len(proposer_prefs[m])

        for w in proposer_prefs[m]:
            # m must prefer w to current partner.
            m_w_rank = p_rank[m].get(w)
            if m_w_rank is None:
                continue
            if m_w_rank >= m_current_rank:
                continue  # m does not prefer w to current partner.

            # w must prefer m to current partner.
            current_p = inv_matching.get(w)
            w_current_rank = (
                r_rank[w].get(current_p, len(receiver_prefs[w]))
                if current_p
                else len(receiver_prefs[w])
            )
            w_m_rank = r_rank[w].get(m)
            if w_m_rank is None:
                continue
            if w_m_rank < w_current_rank:
                blocking.append((m, w))

    return blocking


def is_stable(
    matching: dict[str, str],
    proposer_prefs: dict[str, list[str]],
    receiver_prefs: dict[str, list[str]],
) -> bool:
    """Return True iff the matching has no blocking pairs."""
    return len(find_blocking_pairs(matching, proposer_prefs, receiver_prefs)) == 0


def find_weakly_blocking_pairs(
    matching: dict[str, str],
    proposer_prefs: dict[str, list[list[str]]],
    receiver_prefs: dict[str, list[list[str]]],
) -> list[tuple[str, str]]:
    """Find blocking pairs under weak preferences (with ties).

    A pair ``(m, w)`` **weakly blocks** if both ``m`` *strictly* prefers
    ``w`` to ``mu(m)`` and ``w`` *strictly* prefers ``m`` to ``mu(w)``.

    Parameters
    ----------
    matching : dict
        ``{proposer: receiver}``
    proposer_prefs : dict
        ``{proposer: [[tied_group, ...], ...]}``
    receiver_prefs : dict
        ``{receiver: [[tied_group, ...], ...]}``

    Returns
    -------
    list of tuple
        Weakly blocking pairs.
    """

    def rank_from_groups(groups: list[list[str]]) -> dict[str, int]:
        """Map each agent to its tier index (lower = more preferred)."""
        ranks: dict[str, int] = {}
        for tier_idx, group in enumerate(groups):
            for agent in group:
                ranks[agent] = tier_idx
        return ranks

    p_rank = {p: rank_from_groups(groups) for p, groups in proposer_prefs.items()}
    r_rank = {r: rank_from_groups(groups) for r, groups in receiver_prefs.items()}

    inv_matching: dict[str, str | None] = {}
    for p, r in matching.items():
        inv_matching[r] = p
    for r in receiver_prefs:
        if r not in inv_matching:
            inv_matching[r] = None

    n_p_tiers = {p: len(groups) for p, groups in proposer_prefs.items()}
    n_r_tiers = {r: len(groups) for r, groups in receiver_prefs.items()}

    blocking: list[tuple[str, str]] = []

    for m, m_ranks in p_rank.items():
        current_r = matching.get(m)
        m_current_tier = m_ranks.get(current_r, n_p_tiers[m]) if current_r else n_p_tiers[m]

        for w in m_ranks:
            if m_ranks[w] >= m_current_tier:
                continue  # Not strictly better.

            current_p = inv_matching.get(w)
            w_current_tier = (
                r_rank[w].get(current_p, n_r_tiers[w])
                if current_p
                else n_r_tiers[w]
            )
            w_m_tier = r_rank[w].get(m)
            if w_m_tier is not None and w_m_tier < w_current_tier:
                blocking.append((m, w))

    return blocking
