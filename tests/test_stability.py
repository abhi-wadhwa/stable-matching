"""Tests for the stability checker."""

from __future__ import annotations

import pytest

from src.core.gale_shapley import gale_shapley, receiver_optimal
from src.core.stability import find_blocking_pairs, is_stable, find_weakly_blocking_pairs
from src.core.generators import random_market


class TestBlockingPairs:
    """Test blocking pair detection."""

    def test_stable_matching_has_no_blocking_pairs(self) -> None:
        m_prefs = {
            "m1": ["w1", "w2", "w3"],
            "m2": ["w2", "w3", "w1"],
            "m3": ["w3", "w1", "w2"],
        }
        w_prefs = {
            "w1": ["m1", "m2", "m3"],
            "w2": ["m2", "m3", "m1"],
            "w3": ["m3", "m1", "m2"],
        }
        matching = gale_shapley(m_prefs, w_prefs)
        bp = find_blocking_pairs(matching, m_prefs, w_prefs)
        assert bp == []

    def test_unstable_matching_has_blocking_pairs(self) -> None:
        m_prefs = {
            "m1": ["w1", "w2"],
            "m2": ["w1", "w2"],
        }
        w_prefs = {
            "w1": ["m1", "m2"],
            "w2": ["m1", "m2"],
        }
        # Force an unstable matching: m1->w2, m2->w1.
        matching = {"m1": "w2", "m2": "w1"}
        bp = find_blocking_pairs(matching, m_prefs, w_prefs)
        # (m1, w1) is a blocking pair: m1 prefers w1, w1 prefers m1.
        assert len(bp) >= 1
        assert ("m1", "w1") in bp

    def test_no_false_positives(self) -> None:
        """A stable matching should never report blocking pairs."""
        for seed in range(10):
            m_prefs, w_prefs = random_market(5, seed=seed)
            matching = gale_shapley(m_prefs, w_prefs)
            assert is_stable(matching, m_prefs, w_prefs)

    def test_single_pair(self) -> None:
        m_prefs = {"m1": ["w1"]}
        w_prefs = {"w1": ["m1"]}
        matching = {"m1": "w1"}
        assert is_stable(matching, m_prefs, w_prefs)


class TestIsStable:
    """Test the is_stable convenience function."""

    def test_stable(self) -> None:
        m_prefs = {"m1": ["w1", "w2"], "m2": ["w2", "w1"]}
        w_prefs = {"w1": ["m1", "m2"], "w2": ["m2", "m1"]}
        matching = {"m1": "w1", "m2": "w2"}
        assert is_stable(matching, m_prefs, w_prefs) is True

    def test_unstable(self) -> None:
        m_prefs = {"m1": ["w1", "w2"], "m2": ["w1", "w2"]}
        w_prefs = {"w1": ["m1", "m2"], "w2": ["m1", "m2"]}
        matching = {"m1": "w2", "m2": "w1"}
        assert is_stable(matching, m_prefs, w_prefs) is False


class TestWeakBlocking:
    """Test blocking pair detection with ties."""

    def test_weakly_stable(self) -> None:
        m_prefs = {
            "m1": [["w1", "w2"]],
            "m2": [["w2"], ["w1"]],
        }
        w_prefs = {
            "w1": [["m1"], ["m2"]],
            "w2": [["m2"], ["m1"]],
        }
        matching = {"m1": "w1", "m2": "w2"}
        bp = find_weakly_blocking_pairs(matching, m_prefs, w_prefs)
        assert bp == []

    def test_weakly_unstable(self) -> None:
        m_prefs = {
            "m1": [["w2"], ["w1"]],
            "m2": [["w1"], ["w2"]],
        }
        w_prefs = {
            "w1": [["m1"], ["m2"]],
            "w2": [["m2"], ["m1"]],
        }
        # m1 matched to w1, m2 matched to w2 -- but m1 prefers w2 and w2 prefers m2.
        # Actually m1 prefers w2 (tier 0) to w1 (tier 1), w2 prefers m2 (tier 0) to m1 (tier 1).
        # So (m1, w2) is NOT blocking because w2 prefers m2 (her current partner).
        # Let's check (m2, w1): m2 prefers w1 (tier 0) to w2 (tier 1), w1 prefers m1 (her current, tier 0) to m2 (tier 1).
        # So (m2, w1) is NOT blocking because w1 prefers m1.
        # This matching is actually weakly stable.
        matching = {"m1": "w1", "m2": "w2"}
        bp = find_weakly_blocking_pairs(matching, m_prefs, w_prefs)
        assert bp == []

    def test_clearly_unstable_weak(self) -> None:
        m_prefs = {
            "m1": [["w2"], ["w1"]],
            "m2": [["w1"], ["w2"]],
        }
        w_prefs = {
            "w1": [["m2"], ["m1"]],
            "w2": [["m1"], ["m2"]],
        }
        # m1->w1, m2->w2 but m1 prefers w2 and w2 prefers m1,
        # and m2 prefers w1 and w1 prefers m2.
        matching = {"m1": "w1", "m2": "w2"}
        bp = find_weakly_blocking_pairs(matching, m_prefs, w_prefs)
        assert len(bp) >= 1


class TestEdgeCases:
    """Edge cases for stability checking."""

    def test_empty_market(self) -> None:
        assert is_stable({}, {}, {})

    def test_incomplete_lists_unmatched(self) -> None:
        m_prefs = {"m1": ["w1"], "m2": []}
        w_prefs = {"w1": ["m1"], "w2": []}
        matching = {"m1": "w1"}
        # m2 and w2 are unmatched but have empty lists, so no blocking pair.
        assert is_stable(matching, m_prefs, w_prefs)

    def test_receiver_optimal_is_stable(self) -> None:
        m_prefs, w_prefs = random_market(6, seed=99)
        matching = receiver_optimal(m_prefs, w_prefs)
        assert is_stable(matching, m_prefs, w_prefs)
