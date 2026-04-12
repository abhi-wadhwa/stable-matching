"""
Stable Matching Lattice
========================

The set of all stable matchings for an instance of the stable marriage
problem forms a *distributive lattice* under the partial order:

    ``M1 <= M2``  iff  every proposer weakly prefers their partner in ``M1``
    to their partner in ``M2``.

The proposer-optimal matching is the *top* of the lattice and the
receiver-optimal matching is the *bottom*.

This module enumerates all stable matchings (feasible for small instances)
and provides lattice join/meet operations.

References:
    Knuth, D. E. (1976). Mariages stables. Les Presses de l'Universite de Montreal.
    Irving, R. W. & Leather, P. (1986). The Complexity of Counting Stable Marriages.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from src.core.gale_shapley import gale_shapley, receiver_optimal
from src.core.rotations import (
    Rotation,
    build_rotation_poset,
    eliminate_rotation,
    find_rotations,
)
from src.core.stability import is_stable


def enumerate_stable_matchings(
    proposer_prefs: Dict[str, List[str]],
    receiver_prefs: Dict[str, List[str]],
) -> List[Dict[str, str]]:
    """Enumerate all stable matchings for a given instance.

    Uses BFS over rotation eliminations starting from the proposer-optimal
    matching.  Suitable for small instances only (the number of stable
    matchings can be exponential).

    Parameters
    ----------
    proposer_prefs, receiver_prefs : dict
        Strict preference lists.

    Returns
    -------
    list of dict
        All stable matchings ``{proposer: receiver}``.
    """
    m0 = gale_shapley(proposer_prefs, receiver_prefs)
    all_matchings: List[Dict[str, str]] = [m0]
    visited: Set[str] = set()

    def matching_key(m: Dict[str, str]) -> str:
        return str(sorted(m.items()))

    visited.add(matching_key(m0))

    from collections import deque

    queue: deque[Dict[str, str]] = deque([m0])

    while queue:
        current = queue.popleft()
        rotations = find_rotations(proposer_prefs, receiver_prefs, current)

        for rot in rotations:
            new_matching = eliminate_rotation(
                current, rot, proposer_prefs, receiver_prefs
            )
            mk = matching_key(new_matching)
            if mk not in visited:
                visited.add(mk)
                all_matchings.append(new_matching)
                queue.append(new_matching)

    return all_matchings


def _dominance_key(
    matching: Dict[str, str],
    proposer_prefs: Dict[str, List[str]],
) -> Dict[str, int]:
    """Map each proposer to the rank of their partner (lower = better)."""
    p_rank: Dict[str, Dict[str, int]] = {
        p: {r: i for i, r in enumerate(pl)} for p, pl in proposer_prefs.items()
    }
    return {
        m: p_rank[m].get(w, len(proposer_prefs[m]))
        for m, w in matching.items()
    }


def lattice_join(
    m1: Dict[str, str],
    m2: Dict[str, str],
    proposer_prefs: Dict[str, List[str]],
    receiver_prefs: Dict[str, List[str]],
) -> Dict[str, str]:
    """Compute the lattice join (proposer-best) of two stable matchings.

    For each proposer, take the partner they prefer more.  The result is
    a stable matching (this is a fundamental property of the lattice).

    Parameters
    ----------
    m1, m2 : dict
        Two stable matchings ``{proposer: receiver}``.
    proposer_prefs, receiver_prefs : dict
        Full preference lists.

    Returns
    -------
    dict
        The join ``m1 \\vee m2`` in the stable matching lattice.
    """
    p_rank: Dict[str, Dict[str, int]] = {
        p: {r: i for i, r in enumerate(pl)} for p, pl in proposer_prefs.items()
    }

    join: Dict[str, str] = {}
    for m in proposer_prefs:
        w1 = m1.get(m)
        w2 = m2.get(m)
        if w1 is None and w2 is None:
            continue
        if w1 is None:
            join[m] = w2  # type: ignore
            continue
        if w2 is None:
            join[m] = w1
            continue
        # Take the one m prefers more (lower rank).
        r1 = p_rank[m].get(w1, len(proposer_prefs[m]))
        r2 = p_rank[m].get(w2, len(proposer_prefs[m]))
        join[m] = w1 if r1 <= r2 else w2

    return join


def lattice_meet(
    m1: Dict[str, str],
    m2: Dict[str, str],
    proposer_prefs: Dict[str, List[str]],
    receiver_prefs: Dict[str, List[str]],
) -> Dict[str, str]:
    """Compute the lattice meet (proposer-worst) of two stable matchings.

    For each proposer, take the partner they like *less*.  The result is
    a stable matching.

    Parameters
    ----------
    m1, m2 : dict
        Two stable matchings ``{proposer: receiver}``.
    proposer_prefs, receiver_prefs : dict
        Full preference lists.

    Returns
    -------
    dict
        The meet ``m1 \\wedge m2`` in the stable matching lattice.
    """
    p_rank: Dict[str, Dict[str, int]] = {
        p: {r: i for i, r in enumerate(pl)} for p, pl in proposer_prefs.items()
    }

    meet: Dict[str, str] = {}
    for m in proposer_prefs:
        w1 = m1.get(m)
        w2 = m2.get(m)
        if w1 is None and w2 is None:
            continue
        if w1 is None:
            meet[m] = w2  # type: ignore
            continue
        if w2 is None:
            meet[m] = w1
            continue
        r1 = p_rank[m].get(w1, len(proposer_prefs[m]))
        r2 = p_rank[m].get(w2, len(proposer_prefs[m]))
        meet[m] = w1 if r1 >= r2 else w2

    return meet


def hasse_diagram(
    matchings: List[Dict[str, str]],
    proposer_prefs: Dict[str, List[str]],
) -> List[Tuple[int, int]]:
    """Compute the Hasse diagram edges for a list of stable matchings.

    Returns edges ``(i, j)`` meaning matching ``i`` covers matching ``j``
    (``i`` is directly above ``j`` in the lattice -- proposers prefer ``i``).

    Parameters
    ----------
    matchings : list of dict
        All stable matchings.
    proposer_prefs : dict
        Proposer preference lists.

    Returns
    -------
    list of tuple
        Edges ``(i, j)`` in the Hasse diagram.
    """
    n = len(matchings)
    keys = [_dominance_key(m, proposer_prefs) for m in matchings]
    proposers = list(proposer_prefs.keys())

    # m_i dominates m_j if every proposer weakly prefers m_i.
    def dominates(i: int, j: int) -> bool:
        for p in proposers:
            if keys[i].get(p, 999) > keys[j].get(p, 999):
                return False
        return True

    # Build dominance relation (i >= j means proposers prefer or equal i).
    dom: List[List[bool]] = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and dominates(i, j):
                dom[i][j] = True

    # Hasse: i covers j if i > j and there is no k with i > k > j.
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if not dom[i][j]:
                continue
            # Check no intermediate k.
            covered = True
            for k in range(n):
                if k != i and k != j and dom[i][k] and dom[k][j]:
                    covered = False
                    break
            if covered:
                edges.append((i, j))

    return edges
