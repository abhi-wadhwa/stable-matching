"""
Gale-Shapley Deferred Acceptance Algorithm
===========================================

Implements the classic Gale-Shapley algorithm for one-to-one stable matching.

Given two disjoint sets of agents (proposers and receivers), each with strict
preference orderings over the other side, the algorithm finds a stable matching
where no pair of agents would prefer each other over their assigned partners.

Key results:
- The proposer-optimal stable matching is simultaneously the receiver-pessimal
  stable matching (and vice versa).
- Every execution of DA with the same proposer side yields the same result.

References:
    Gale, D. & Shapley, L. S. (1962). College Admissions and the Stability
    of Marriage. *The American Mathematical Monthly*, 69(1), 9--15.
"""

from __future__ import annotations

from collections import deque


def gale_shapley(
    proposer_prefs: dict[str, list[str]],
    receiver_prefs: dict[str, list[str]],
) -> dict[str, str]:
    """Run the Gale-Shapley deferred-acceptance algorithm.

    Parameters
    ----------
    proposer_prefs : dict
        Maps each proposer to a strict preference list over receivers
        (most preferred first).
    receiver_prefs : dict
        Maps each receiver to a strict preference list over proposers
        (most preferred first).

    Returns
    -------
    dict
        A dict ``{proposer: receiver}`` representing the proposer-optimal
        stable matching.  Unmatched agents (when lists are incomplete) are
        omitted.

    Examples
    --------
    >>> m_prefs = {"m1": ["w1", "w2"], "m2": ["w2", "w1"]}
    >>> w_prefs = {"w1": ["m1", "m2"], "w2": ["m1", "m2"]}
    >>> gale_shapley(m_prefs, w_prefs)
    {'m1': 'w1', 'm2': 'w2'}
    """
    # Build rank lookup for each receiver for O(1) comparisons.
    receiver_rank: dict[str, dict[str, int]] = {}
    for r, pref_list in receiver_prefs.items():
        receiver_rank[r] = {p: idx for idx, p in enumerate(pref_list)}

    # Track how far down each proposer's list they have gone.
    next_proposal: dict[str, int] = {p: 0 for p in proposer_prefs}

    # Current tentative matching (receiver -> proposer currently held).
    current_holder: dict[str, str | None] = {r: None for r in receiver_prefs}

    # Queue of free proposers who still have someone to propose to.
    free: deque[str] = deque(proposer_prefs.keys())

    while free:
        proposer = free.popleft()
        prefs = proposer_prefs[proposer]

        # Proposer may have exhausted their list.
        if next_proposal[proposer] >= len(prefs):
            continue

        receiver = prefs[next_proposal[proposer]]
        next_proposal[proposer] += 1

        # If receiver finds proposer unacceptable, skip.
        if proposer not in receiver_rank.get(receiver, {}):
            free.append(proposer)
            continue

        current = current_holder[receiver]

        if current is None:
            # Receiver is free -- accept tentatively.
            current_holder[receiver] = proposer
        elif receiver_rank[receiver][proposer] < receiver_rank[receiver][current]:
            # Receiver prefers new proposer -- switch.
            current_holder[receiver] = proposer
            free.append(current)
        else:
            # Receiver prefers current holder -- reject.
            free.append(proposer)

    # Build proposer -> receiver mapping.
    matching: dict[str, str] = {}
    for receiver, held_proposer in current_holder.items():
        if held_proposer is not None:
            matching[held_proposer] = receiver
    return matching


def gale_shapley_weak(
    proposer_prefs: dict[str, list[list[str]]],
    receiver_prefs: dict[str, list[list[str]]],
) -> dict[str, str]:
    """Gale-Shapley for preferences with ties (weakly stable matching).

    Preferences are given as lists of *tie groups* (most preferred first).
    Within a tie group, agents are considered equally preferred.

    We break ties arbitrarily (by order within the tie group) and then
    run standard DA.  The result is a weakly stable matching -- no
    *strict* blocking pair exists.

    Parameters
    ----------
    proposer_prefs : dict
        ``{proposer: [[tied_receivers, ...], [next_tier, ...], ...]}``
    receiver_prefs : dict
        ``{receiver: [[tied_proposers, ...], [next_tier, ...], ...]}``

    Returns
    -------
    dict
        ``{proposer: receiver}`` -- a weakly stable matching.
    """

    def flatten(grouped: dict[str, list[list[str]]]) -> dict[str, list[str]]:
        flat: dict[str, list[str]] = {}
        for agent, groups in grouped.items():
            flat[agent] = [item for group in groups for item in group]
        return flat

    return gale_shapley(flatten(proposer_prefs), flatten(receiver_prefs))


def receiver_optimal(
    proposer_prefs: dict[str, list[str]],
    receiver_prefs: dict[str, list[str]],
) -> dict[str, str]:
    """Compute the receiver-optimal (proposer-pessimal) stable matching.

    This is equivalent to running DA with receivers as proposers.

    Returns
    -------
    dict
        ``{proposer: receiver}`` (same orientation as :func:`gale_shapley`).
    """
    # Run DA with roles swapped.
    rev_matching = gale_shapley(receiver_prefs, proposer_prefs)
    # Invert: receiver->proposer becomes proposer->receiver.
    return {v: k for k, v in rev_matching.items()}


def da_trace(
    proposer_prefs: dict[str, list[str]],
    receiver_prefs: dict[str, list[str]],
) -> tuple[dict[str, str], list[dict]]:
    """Run DA and return a round-by-round trace for visualization.

    Returns
    -------
    tuple
        ``(matching, rounds)`` where *rounds* is a list of dicts, each with:
        - ``proposals``: list of ``(proposer, receiver)``
        - ``rejections``: list of ``(proposer, receiver)``
        - ``holds``: list of ``(receiver, proposer)``
    """
    receiver_rank: dict[str, dict[str, int]] = {}
    for r, pref_list in receiver_prefs.items():
        receiver_rank[r] = {p: idx for idx, p in enumerate(pref_list)}

    next_proposal: dict[str, int] = {p: 0 for p in proposer_prefs}
    current_holder: dict[str, str | None] = {r: None for r in receiver_prefs}
    free: deque[str] = deque(proposer_prefs.keys())

    rounds: list[dict] = []

    while free:
        proposals: list[tuple[str, str]] = []
        rejections: list[tuple[str, str]] = []
        holds: list[tuple[str, str]] = []

        # Collect all proposals for this round.
        proposers_this_round = list(free)
        free.clear()

        for proposer in proposers_this_round:
            prefs = proposer_prefs[proposer]
            if next_proposal[proposer] >= len(prefs):
                continue
            receiver = prefs[next_proposal[proposer]]
            next_proposal[proposer] += 1

            if proposer not in receiver_rank.get(receiver, {}):
                proposals.append((proposer, receiver))
                rejections.append((proposer, receiver))
                free.append(proposer)
                continue

            proposals.append((proposer, receiver))
            current = current_holder[receiver]

            if current is None:
                current_holder[receiver] = proposer
                holds.append((receiver, proposer))
            elif receiver_rank[receiver][proposer] < receiver_rank[receiver][current]:
                current_holder[receiver] = proposer
                holds.append((receiver, proposer))
                rejections.append((current, receiver))
                free.append(current)
            else:
                rejections.append((proposer, receiver))
                free.append(proposer)

        if proposals:
            rounds.append(
                {
                    "proposals": proposals,
                    "rejections": rejections,
                    "holds": holds,
                }
            )

    matching: dict[str, str] = {}
    for receiver, held_proposer in current_holder.items():
        if held_proposer is not None:
            matching[held_proposer] = receiver
    return matching, rounds
