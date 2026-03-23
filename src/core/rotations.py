"""
Rotations for Stable Matching
==============================

A *rotation* is a cyclic sequence of (proposer, receiver) pairs that can be
"eliminated" from the proposer-optimal matching to obtain another stable
matching.  The set of all rotations, together with a partial order, fully
characterises the lattice of stable matchings.

Algorithm outline
-----------------
1. Start from the proposer-optimal matching ``M_0``.
2. Build the *reduced preference lists*: for each proposer ``m``, remove
   every receiver that ``m`` prefers to ``M_0(m)``; for each receiver ``w``,
   remove every proposer that ``w`` likes less than ``M_0(w)`` (with respect
   to the proposer-optimal matching context).
3. Identify *rotations* as directed cycles in the "next-partner" graph.
4. Eliminating a rotation yields the next stable matching.

The rotation poset is built by tracking which rotations must be eliminated
before others become *exposed*.

References:
    Irving, R. W. & Leather, P. (1986). The Complexity of Counting Stable
    Marriages. *SIAM J. Comput.*, 15(3), 655--667.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from src.core.gale_shapley import gale_shapley, receiver_optimal


# Type alias: a rotation is a list of (proposer, receiver) pairs.
Rotation = List[Tuple[str, str]]


def _build_reduced_lists(
    proposer_prefs: Dict[str, List[str]],
    receiver_prefs: Dict[str, List[str]],
    m0: Dict[str, str],
    m_z: Dict[str, str],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build reduced preference lists bounded by M_0 and M_z.

    For proposer m: keep only receivers between M_0(m) and M_z(m) inclusive.
    For receiver w: keep only proposers between M_0(w) and M_z(w) inclusive.
    """
    p_rank: Dict[str, Dict[str, int]] = {
        p: {r: i for i, r in enumerate(pl)} for p, pl in proposer_prefs.items()
    }
    r_rank: Dict[str, Dict[str, int]] = {
        r: {p: i for i, p in enumerate(pl)} for r, pl in receiver_prefs.items()
    }

    inv_m0 = {v: k for k, v in m0.items()}
    inv_mz = {v: k for k, v in m_z.items()}

    reduced_p: Dict[str, List[str]] = {}
    for m, plist in proposer_prefs.items():
        best = p_rank[m].get(m0.get(m, ""), len(plist))
        worst = p_rank[m].get(m_z.get(m, ""), -1)
        reduced_p[m] = [w for w in plist if best <= p_rank[m][w] <= worst]

    reduced_r: Dict[str, List[str]] = {}
    for w, plist in receiver_prefs.items():
        best = r_rank[w].get(inv_mz.get(w, ""), len(plist))
        worst = r_rank[w].get(inv_m0.get(w, ""), -1)
        reduced_r[w] = [m for m in plist if best <= r_rank[w][m] <= worst]

    return reduced_p, reduced_r


def find_rotations(
    proposer_prefs: Dict[str, List[str]],
    receiver_prefs: Dict[str, List[str]],
    matching: Optional[Dict[str, str]] = None,
) -> List[Rotation]:
    """Find all exposed rotations from a given stable matching.

    Parameters
    ----------
    proposer_prefs, receiver_prefs : dict
        Full preference lists.
    matching : dict or None
        Current stable matching ``{proposer: receiver}``.
        If None, uses the proposer-optimal matching.

    Returns
    -------
    list of Rotation
        Each rotation is a list of ``(proposer, receiver)`` pairs forming
        a directed cycle.
    """
    if matching is None:
        matching = gale_shapley(proposer_prefs, receiver_prefs)

    p_rank: Dict[str, Dict[str, int]] = {
        p: {r: i for i, r in enumerate(pl)} for p, pl in proposer_prefs.items()
    }
    r_rank: Dict[str, Dict[str, int]] = {
        r: {p: i for i, p in enumerate(pl)} for r, pl in receiver_prefs.items()
    }

    inv_matching = {v: k for k, v in matching.items()}

    # For each proposer m matched to w = matching[m], find the *next* receiver
    # on m's list after w who still finds m acceptable and prefers m to her
    # current partner... actually, we find the next woman on m's reduced list.
    # "next(m)" = first w' after matching[m] on m's pref list such that
    # w' prefers m to her current partner in this matching.
    def next_receiver(m: str) -> Optional[str]:
        w = matching[m]
        start = p_rank[m][w]
        for idx in range(start + 1, len(proposer_prefs[m])):
            w_prime = proposer_prefs[m][idx]
            if w_prime not in r_rank:
                continue
            m_prime = inv_matching.get(w_prime)
            if m_prime is None:
                continue
            if m in r_rank[w_prime] and (
                m_prime not in r_rank[w_prime]
                or r_rank[w_prime][m] < r_rank[w_prime][m_prime]
            ):
                return w_prime
        return None

    # Build the "next" graph: m -> m' where m' = current partner of next(m).
    next_map: Dict[str, Optional[str]] = {}
    next_w: Dict[str, Optional[str]] = {}
    for m in matching:
        w_prime = next_receiver(m)
        if w_prime is not None:
            next_map[m] = inv_matching[w_prime]
            next_w[m] = w_prime
        else:
            next_map[m] = None
            next_w[m] = None

    # Find all cycles in next_map.
    visited: Set[str] = set()
    rotations: List[Rotation] = []

    for start_m in matching:
        if start_m in visited or next_map[start_m] is None:
            continue

        # Trace until we find a cycle or hit None.
        path: List[str] = []
        path_set: Set[str] = set()
        current: Optional[str] = start_m

        while current is not None and current not in visited and current not in path_set:
            path.append(current)
            path_set.add(current)
            current = next_map.get(current)

        if current is not None and current in path_set:
            # Extract the cycle.
            cycle_start = path.index(current)
            cycle = path[cycle_start:]
            rotation: Rotation = [(m, matching[m]) for m in cycle]
            rotations.append(rotation)
            visited.update(cycle)

        visited.update(path)

    return rotations


def eliminate_rotation(
    matching: Dict[str, str],
    rotation: Rotation,
    proposer_prefs: Dict[str, List[str]],
    receiver_prefs: Dict[str, List[str]],
) -> Dict[str, str]:
    """Eliminate a rotation from a matching to produce a new stable matching.

    If rotation = [(m_0, w_0), (m_1, w_1), ..., (m_{k-1}, w_{k-1})],
    then in the new matching each m_i is matched to the *next* preferred
    receiver after w_i who prefers m_i to her current partner.

    Parameters
    ----------
    matching : dict
        Current stable matching.
    rotation : Rotation
        Rotation to eliminate.
    proposer_prefs, receiver_prefs : dict
        Full preference lists.

    Returns
    -------
    dict
        New stable matching after rotation elimination.
    """
    p_rank: Dict[str, Dict[str, int]] = {
        p: {r: i for i, r in enumerate(pl)} for p, pl in proposer_prefs.items()
    }
    r_rank: Dict[str, Dict[str, int]] = {
        r: {p: i for i, p in enumerate(pl)} for r, pl in receiver_prefs.items()
    }
    inv_matching = {v: k for k, v in matching.items()}

    new_matching = dict(matching)
    k = len(rotation)

    for i in range(k):
        m_i = rotation[i][0]
        w_i = rotation[i][1]
        # m_i gets matched to w_{i+1 mod k}'s "first choice that prefers m_i"
        # Actually, in the standard rotation elimination:
        # m_i is re-matched to w_{(i+1) mod k}.
        w_next = rotation[(i + 1) % k][1]
        new_matching[m_i] = w_next

    return new_matching


def build_rotation_poset(
    proposer_prefs: Dict[str, List[str]],
    receiver_prefs: Dict[str, List[str]],
) -> Tuple[List[Rotation], Dict[int, Set[int]]]:
    """Build the rotation poset for a stable matching instance.

    Explores all rotations reachable from the proposer-optimal matching
    by BFS through rotation eliminations.

    Returns
    -------
    tuple
        ``(rotations, predecessors)`` where:
        - ``rotations`` is a list of all rotations found.
        - ``predecessors[i]`` is the set of rotation indices that must be
          eliminated before rotation ``i`` becomes exposed.
    """
    m0 = gale_shapley(proposer_prefs, receiver_prefs)

    all_rotations: List[Rotation] = []
    rotation_to_idx: Dict[str, int] = {}
    predecessors: Dict[int, Set[int]] = {}

    def rot_key(rot: Rotation) -> str:
        return str(sorted(rot))

    # BFS over matchings.
    from collections import deque

    queue: deque[Tuple[Dict[str, str], Set[int]]] = deque()
    queue.append((m0, set()))
    visited_matchings: Set[str] = set()

    def matching_key(m: Dict[str, str]) -> str:
        return str(sorted(m.items()))

    visited_matchings.add(matching_key(m0))

    while queue:
        current_matching, eliminated_so_far = queue.popleft()
        exposed = find_rotations(proposer_prefs, receiver_prefs, current_matching)

        for rot in exposed:
            rk = rot_key(rot)
            if rk not in rotation_to_idx:
                idx = len(all_rotations)
                all_rotations.append(rot)
                rotation_to_idx[rk] = idx
                predecessors[idx] = set(eliminated_so_far)
            else:
                idx = rotation_to_idx[rk]
                # Keep the minimal set of predecessors.
                predecessors[idx] = predecessors[idx] & eliminated_so_far

            new_matching = eliminate_rotation(
                current_matching, rot, proposer_prefs, receiver_prefs
            )
            mk = matching_key(new_matching)
            if mk not in visited_matchings:
                visited_matchings.add(mk)
                new_eliminated = eliminated_so_far | {idx}
                queue.append((new_matching, new_eliminated))

    return all_rotations, predecessors
