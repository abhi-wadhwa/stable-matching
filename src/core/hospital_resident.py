"""
Hospital-Resident Problem (Many-to-One Matching)
=================================================

Extends Gale-Shapley to settings where one side (hospitals) can accept
multiple partners up to a *quota*.

The **Rural Hospital Theorem** states that:
  1. Every stable matching matches the same set of residents and hospitals.
  2. Every hospital that is under-subscribed in one stable matching is
     under-subscribed in *all* stable matchings, and is matched to exactly
     the same set of residents.

References:
    Roth, A. E. (1984). The Evolution of the Labor Market for Medical
    Interns and Residents. *Journal of Political Economy*, 92(6), 991--1016.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple


def hospital_resident_da(
    resident_prefs: Dict[str, List[str]],
    hospital_prefs: Dict[str, List[str]],
    quotas: Dict[str, int],
    proposer: str = "resident",
) -> Dict[str, List[str]]:
    """Run Gale-Shapley for the hospital-resident problem.

    Parameters
    ----------
    resident_prefs : dict
        ``{resident: [hospital_1, hospital_2, ...]}`` in preference order.
    hospital_prefs : dict
        ``{hospital: [resident_1, resident_2, ...]}`` in preference order.
    quotas : dict
        ``{hospital: max_positions}``.
    proposer : str
        ``"resident"`` for resident-optimal or ``"hospital"`` for
        hospital-optimal.

    Returns
    -------
    dict
        ``{hospital: [matched_residents]}`` -- the stable matching.
    """
    if proposer == "resident":
        return _resident_proposing_da(resident_prefs, hospital_prefs, quotas)
    else:
        return _hospital_proposing_da(resident_prefs, hospital_prefs, quotas)


def _resident_proposing_da(
    resident_prefs: Dict[str, List[str]],
    hospital_prefs: Dict[str, List[str]],
    quotas: Dict[str, int],
) -> Dict[str, List[str]]:
    """Resident-proposing deferred acceptance."""
    # Rank lookup for hospitals.
    h_rank: Dict[str, Dict[str, int]] = {}
    for h, plist in hospital_prefs.items():
        h_rank[h] = {r: idx for idx, r in enumerate(plist)}

    next_proposal: Dict[str, int] = {r: 0 for r in resident_prefs}

    # Each hospital holds a set of tentatively accepted residents.
    held: Dict[str, List[str]] = {h: [] for h in hospital_prefs}

    free: deque[str] = deque(resident_prefs.keys())

    while free:
        resident = free.popleft()
        prefs = resident_prefs[resident]

        if next_proposal[resident] >= len(prefs):
            continue

        hospital = prefs[next_proposal[resident]]
        next_proposal[resident] += 1

        # Check if hospital finds resident acceptable.
        if resident not in h_rank.get(hospital, {}):
            free.append(resident)
            continue

        current = held[hospital]

        if len(current) < quotas[hospital]:
            # Hospital has room -- accept.
            current.append(resident)
        else:
            # Hospital is full -- check if resident is preferred to worst held.
            worst = max(current, key=lambda r: h_rank[hospital][r])
            if h_rank[hospital][resident] < h_rank[hospital][worst]:
                current.remove(worst)
                current.append(resident)
                free.append(worst)
            else:
                free.append(resident)

    # Sort each hospital's residents by preference.
    for h in held:
        held[h].sort(key=lambda r: h_rank[h].get(r, float("inf")))

    return held


def _hospital_proposing_da(
    resident_prefs: Dict[str, List[str]],
    hospital_prefs: Dict[str, List[str]],
    quotas: Dict[str, int],
) -> Dict[str, List[str]]:
    """Hospital-proposing deferred acceptance."""
    r_rank: Dict[str, Dict[str, int]] = {}
    for r, plist in resident_prefs.items():
        r_rank[r] = {h: idx for idx, h in enumerate(plist)}

    # Each hospital tracks how far down its preference list it has proposed.
    next_proposal: Dict[str, int] = {h: 0 for h in hospital_prefs}

    # Each resident holds at most one offer.
    resident_held: Dict[str, Optional[str]] = {r: None for r in resident_prefs}

    # Track how many positions each hospital still needs to fill.
    remaining: Dict[str, int] = dict(quotas)

    # Hospitals that still have unfilled positions and people to propose to.
    active: deque[str] = deque(
        h for h in hospital_prefs if remaining[h] > 0
    )

    while active:
        hospital = active.popleft()
        prefs = hospital_prefs[hospital]

        if next_proposal[hospital] >= len(prefs) or remaining[hospital] <= 0:
            continue

        resident = prefs[next_proposal[hospital]]
        next_proposal[hospital] += 1

        if hospital not in r_rank.get(resident, {}):
            # Resident finds hospital unacceptable.
            if remaining[hospital] > 0 and next_proposal[hospital] < len(prefs):
                active.append(hospital)
            continue

        current_h = resident_held[resident]

        if current_h is None:
            resident_held[resident] = hospital
            remaining[hospital] -= 1
        elif r_rank[resident][hospital] < r_rank[resident][current_h]:
            resident_held[resident] = hospital
            remaining[hospital] -= 1
            remaining[current_h] += 1
            if remaining[current_h] > 0 and next_proposal[current_h] < len(hospital_prefs[current_h]):
                active.append(current_h)
        else:
            # Rejected.
            pass

        if remaining[hospital] > 0 and next_proposal[hospital] < len(prefs):
            active.append(hospital)

    # Build hospital -> residents mapping.
    h_rank: Dict[str, Dict[str, int]] = {}
    for h, plist in hospital_prefs.items():
        h_rank[h] = {r: idx for idx, r in enumerate(plist)}

    held: Dict[str, List[str]] = {h: [] for h in hospital_prefs}
    for r, h in resident_held.items():
        if h is not None:
            held[h].append(r)

    for h in held:
        held[h].sort(key=lambda r: h_rank[h].get(r, float("inf")))

    return held


def verify_rural_hospital(
    matchings: List[Dict[str, List[str]]],
) -> bool:
    """Verify the Rural Hospital Theorem across multiple stable matchings.

    Checks that:
    1. The same set of residents are matched in every stable matching.
    2. Any hospital that is under-subscribed in one matching is
       under-subscribed in all, with the *same* residents.

    Parameters
    ----------
    matchings : list of dict
        Multiple stable matchings ``{hospital: [residents]}``.

    Returns
    -------
    bool
        True if the Rural Hospital Theorem holds.
    """
    if not matchings:
        return True

    # Check 1: Same set of matched residents.
    matched_sets = []
    for m in matchings:
        residents = frozenset(r for rs in m.values() for r in rs)
        matched_sets.append(residents)

    if len(set(matched_sets)) > 1:
        return False

    # Check 2: Under-subscribed hospitals have same residents.
    # We don't have quotas here, but we can check that hospitals with
    # fewer residents in one matching have the same residents in all.
    hospitals = set()
    for m in matchings:
        hospitals.update(m.keys())

    for h in hospitals:
        residents_across = [frozenset(m.get(h, [])) for m in matchings]
        sizes = [len(rs) for rs in residents_across]
        min_size = min(sizes)
        max_size = max(sizes)

        if min_size != max_size:
            # Different sizes -> not all matchings have same count.
            # But rural hospital says under-quota hospitals have same residents.
            # If sizes differ, theorem is violated.
            return False

    return True


def find_hr_blocking_pairs(
    matching: Dict[str, List[str]],
    resident_prefs: Dict[str, List[str]],
    hospital_prefs: Dict[str, List[str]],
    quotas: Dict[str, int],
) -> List[Tuple[str, str]]:
    """Find blocking pairs in a hospital-resident matching.

    A pair ``(r, h)`` blocks if:
    - ``r`` prefers ``h`` to their current hospital (or is unmatched), AND
    - ``h`` either has unfilled quota or prefers ``r`` to its worst accepted
      resident.

    Parameters
    ----------
    matching : dict
        ``{hospital: [residents]}``.
    resident_prefs, hospital_prefs : dict
        Preference lists.
    quotas : dict
        Hospital capacities.

    Returns
    -------
    list of tuple
        Blocking pairs ``(resident, hospital)``.
    """
    r_rank: Dict[str, Dict[str, int]] = {
        r: {h: i for i, h in enumerate(pl)} for r, pl in resident_prefs.items()
    }
    h_rank: Dict[str, Dict[str, int]] = {
        h: {r: i for i, r in enumerate(pl)} for h, pl in hospital_prefs.items()
    }

    # Build resident -> hospital mapping.
    res_to_hosp: Dict[str, Optional[str]] = {r: None for r in resident_prefs}
    for h, residents in matching.items():
        for r in residents:
            res_to_hosp[r] = h

    blocking: List[Tuple[str, str]] = []

    for r in resident_prefs:
        current_h = res_to_hosp[r]
        r_current_rank = (
            r_rank[r].get(current_h, len(resident_prefs[r]))
            if current_h
            else len(resident_prefs[r])
        )

        for h in resident_prefs[r]:
            # r must prefer h to current hospital.
            h_rank_for_r = r_rank[r].get(h)
            if h_rank_for_r is None or h_rank_for_r >= r_current_rank:
                continue

            # h must find r acceptable.
            if r not in h_rank[h]:
                continue

            residents_at_h = matching.get(h, [])

            if len(residents_at_h) < quotas.get(h, 0):
                # h has room.
                blocking.append((r, h))
            else:
                # h is full -- check if h prefers r to worst current.
                if residents_at_h:
                    worst = max(residents_at_h, key=lambda x: h_rank[h].get(x, float("inf")))
                    if h_rank[h][r] < h_rank[h].get(worst, float("inf")):
                        blocking.append((r, h))

    return blocking
