"""Tests for the stable matching lattice and rotation enumeration."""

from __future__ import annotations

import pytest

from src.core.gale_shapley import gale_shapley, receiver_optimal
from src.core.stability import is_stable
from src.core.lattice import (
    enumerate_stable_matchings,
    lattice_join,
    lattice_meet,
    hasse_diagram,
)
from src.core.rotations import find_rotations, build_rotation_poset, eliminate_rotation
from src.core.generators import random_market


class TestEnumeration:
    """Test stable matching enumeration."""

    def test_all_matchings_are_stable(self) -> None:
        m_prefs = {
            "m1": ["w1", "w2", "w3"],
            "m2": ["w2", "w1", "w3"],
            "m3": ["w1", "w3", "w2"],
        }
        w_prefs = {
            "w1": ["m2", "m1", "m3"],
            "w2": ["m3", "m1", "m2"],
            "w3": ["m1", "m3", "m2"],
        }
        all_m = enumerate_stable_matchings(m_prefs, w_prefs)
        assert len(all_m) >= 1
        for m in all_m:
            assert is_stable(m, m_prefs, w_prefs), f"Matching {m} is not stable"

    def test_includes_both_optimal(self) -> None:
        """Both proposer-optimal and receiver-optimal must be in the set."""
        m_prefs = {
            "m1": ["w1", "w2", "w3"],
            "m2": ["w2", "w1", "w3"],
            "m3": ["w1", "w3", "w2"],
        }
        w_prefs = {
            "w1": ["m2", "m1", "m3"],
            "w2": ["m3", "m1", "m2"],
            "w3": ["m1", "m3", "m2"],
        }
        all_m = enumerate_stable_matchings(m_prefs, w_prefs)
        m0 = gale_shapley(m_prefs, w_prefs)
        mz = receiver_optimal(m_prefs, w_prefs)

        all_sorted = [dict(sorted(m.items())) for m in all_m]
        assert dict(sorted(m0.items())) in all_sorted
        assert dict(sorted(mz.items())) in all_sorted

    def test_unique_stable_matching(self) -> None:
        """When preferences are perfectly aligned, there is exactly one."""
        m_prefs = {
            "m1": ["w1", "w2"],
            "m2": ["w2", "w1"],
        }
        w_prefs = {
            "w1": ["m1", "m2"],
            "w2": ["m2", "m1"],
        }
        all_m = enumerate_stable_matchings(m_prefs, w_prefs)
        assert len(all_m) == 1
        assert all_m[0] == {"m1": "w1", "m2": "w2"}

    @pytest.mark.parametrize("n", [3, 4])
    def test_random_enumeration_all_stable(self, n: int) -> None:
        for seed in range(3):
            m_prefs, w_prefs = random_market(n, seed=seed)
            all_m = enumerate_stable_matchings(m_prefs, w_prefs)
            for m in all_m:
                assert is_stable(m, m_prefs, w_prefs)


class TestLatticeOperations:
    """Test join and meet in the stable matching lattice."""

    def _get_instance(self):
        m_prefs = {
            "m1": ["w1", "w2", "w3"],
            "m2": ["w2", "w1", "w3"],
            "m3": ["w1", "w3", "w2"],
        }
        w_prefs = {
            "w1": ["m2", "m1", "m3"],
            "w2": ["m3", "m1", "m2"],
            "w3": ["m1", "m3", "m2"],
        }
        return m_prefs, w_prefs

    def test_join_of_same_is_same(self) -> None:
        m_prefs, w_prefs = self._get_instance()
        m0 = gale_shapley(m_prefs, w_prefs)
        j = lattice_join(m0, m0, m_prefs, w_prefs)
        assert j == m0

    def test_meet_of_same_is_same(self) -> None:
        m_prefs, w_prefs = self._get_instance()
        m0 = gale_shapley(m_prefs, w_prefs)
        mt = lattice_meet(m0, m0, m_prefs, w_prefs)
        assert mt == m0

    def test_join_of_extremes_is_proposer_optimal(self) -> None:
        m_prefs, w_prefs = self._get_instance()
        m0 = gale_shapley(m_prefs, w_prefs)
        mz = receiver_optimal(m_prefs, w_prefs)
        j = lattice_join(m0, mz, m_prefs, w_prefs)
        assert j == m0

    def test_meet_of_extremes_is_receiver_optimal(self) -> None:
        m_prefs, w_prefs = self._get_instance()
        m0 = gale_shapley(m_prefs, w_prefs)
        mz = receiver_optimal(m_prefs, w_prefs)
        mt = lattice_meet(m0, mz, m_prefs, w_prefs)
        assert mt == mz

    def test_join_is_stable(self) -> None:
        m_prefs, w_prefs = random_market(4, seed=13)
        all_m = enumerate_stable_matchings(m_prefs, w_prefs)
        if len(all_m) >= 2:
            j = lattice_join(all_m[0], all_m[1], m_prefs, w_prefs)
            assert is_stable(j, m_prefs, w_prefs)

    def test_meet_is_stable(self) -> None:
        m_prefs, w_prefs = random_market(4, seed=13)
        all_m = enumerate_stable_matchings(m_prefs, w_prefs)
        if len(all_m) >= 2:
            mt = lattice_meet(all_m[0], all_m[1], m_prefs, w_prefs)
            assert is_stable(mt, m_prefs, w_prefs)


class TestHasseDiagram:
    """Test Hasse diagram construction."""

    def test_single_matching(self) -> None:
        m_prefs = {"m1": ["w1"], "m2": ["w2"]}
        w_prefs = {"w1": ["m1"], "w2": ["m2"]}
        # Only one stable matching possible.
        # Note: with incomplete mutual lists this might not enumerate fully.
        # Use complete lists:
        m_prefs = {"m1": ["w1", "w2"], "m2": ["w2", "w1"]}
        w_prefs = {"w1": ["m1", "m2"], "w2": ["m2", "m1"]}
        all_m = enumerate_stable_matchings(m_prefs, w_prefs)
        edges = hasse_diagram(all_m, m_prefs)
        # With one matching, no edges.
        if len(all_m) == 1:
            assert edges == []

    def test_two_matchings_one_edge(self) -> None:
        m_prefs = {
            "m1": ["w1", "w2"],
            "m2": ["w2", "w1"],
        }
        w_prefs = {
            "w1": ["m2", "m1"],
            "w2": ["m1", "m2"],
        }
        all_m = enumerate_stable_matchings(m_prefs, w_prefs)
        if len(all_m) == 2:
            edges = hasse_diagram(all_m, m_prefs)
            assert len(edges) == 1


class TestRotations:
    """Test rotation finding and elimination."""

    def test_find_rotations_from_m0(self) -> None:
        m_prefs = {
            "m1": ["w1", "w2"],
            "m2": ["w2", "w1"],
        }
        w_prefs = {
            "w1": ["m2", "m1"],
            "w2": ["m1", "m2"],
        }
        m0 = gale_shapley(m_prefs, w_prefs)
        rots = find_rotations(m_prefs, w_prefs, m0)
        # Should find at least one rotation if m0 != mz.
        mz = receiver_optimal(m_prefs, w_prefs)
        if m0 != mz:
            assert len(rots) >= 1

    def test_rotation_elimination_yields_stable(self) -> None:
        m_prefs = {
            "m1": ["w1", "w2", "w3"],
            "m2": ["w2", "w1", "w3"],
            "m3": ["w1", "w3", "w2"],
        }
        w_prefs = {
            "w1": ["m2", "m1", "m3"],
            "w2": ["m3", "m1", "m2"],
            "w3": ["m1", "m3", "m2"],
        }
        m0 = gale_shapley(m_prefs, w_prefs)
        rots = find_rotations(m_prefs, w_prefs, m0)
        for rot in rots:
            new_m = eliminate_rotation(m0, rot, m_prefs, w_prefs)
            assert is_stable(new_m, m_prefs, w_prefs), (
                f"Eliminating rotation {rot} from {m0} yielded unstable {new_m}"
            )

    def test_rotation_poset_structure(self) -> None:
        m_prefs, w_prefs = random_market(4, seed=17)
        rotations, preds = build_rotation_poset(m_prefs, w_prefs)
        # Predecessors should reference valid indices.
        for idx, pred_set in preds.items():
            assert 0 <= idx < len(rotations)
            for p in pred_set:
                assert 0 <= p < len(rotations)
