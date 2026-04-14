"""Tests for the Gale-Shapley deferred acceptance algorithm."""

from __future__ import annotations

import pytest

from src.core.gale_shapley import da_trace, gale_shapley, gale_shapley_weak, receiver_optimal
from src.core.generators import random_market
from src.core.stability import is_stable


class TestBasicDA:
    """Basic correctness tests for Gale-Shapley."""

    def test_simple_2x2(self) -> None:
        m_prefs = {"m1": ["w1", "w2"], "m2": ["w2", "w1"]}
        w_prefs = {"w1": ["m1", "m2"], "w2": ["m1", "m2"]}
        matching = gale_shapley(m_prefs, w_prefs)
        assert matching == {"m1": "w1", "m2": "w2"}

    def test_simple_3x3(self) -> None:
        m_prefs = {
            "m1": ["w1", "w2", "w3"],
            "m2": ["w2", "w3", "w1"],
            "m3": ["w3", "w1", "w2"],
        }
        w_prefs = {
            "w1": ["m2", "m3", "m1"],
            "w2": ["m3", "m1", "m2"],
            "w3": ["m1", "m2", "m3"],
        }
        matching = gale_shapley(m_prefs, w_prefs)
        # Must be stable.
        assert is_stable(matching, m_prefs, w_prefs)
        # Must be a perfect matching.
        assert len(matching) == 3
        assert set(matching.values()) == {"w1", "w2", "w3"}

    def test_everyone_agrees(self) -> None:
        """When all proposers have the same top choice but receivers disagree."""
        m_prefs = {
            "m1": ["w1", "w2"],
            "m2": ["w1", "w2"],
        }
        w_prefs = {
            "w1": ["m1", "m2"],
            "w2": ["m2", "m1"],
        }
        matching = gale_shapley(m_prefs, w_prefs)
        assert matching == {"m1": "w1", "m2": "w2"}
        assert is_stable(matching, m_prefs, w_prefs)


class TestProposerOptimality:
    """The proposer-optimal matching gives each proposer their best
    stable partner."""

    def test_proposer_optimal_is_best_for_proposers(self) -> None:
        m_prefs = {
            "m1": ["w1", "w2", "w3"],
            "m2": ["w1", "w3", "w2"],
            "m3": ["w2", "w1", "w3"],
        }
        w_prefs = {
            "w1": ["m2", "m1", "m3"],
            "w2": ["m3", "m1", "m2"],
            "w3": ["m1", "m3", "m2"],
        }
        m0 = gale_shapley(m_prefs, w_prefs)
        mz = receiver_optimal(m_prefs, w_prefs)

        # Build rank lookup.
        p_rank = {
            p: {r: i for i, r in enumerate(pl)} for p, pl in m_prefs.items()
        }

        # Every proposer must weakly prefer m0 to mz.
        for m in m_prefs:
            assert p_rank[m][m0[m]] <= p_rank[m][mz[m]], (
                f"{m} prefers {mz[m]} (rank {p_rank[m][mz[m]]}) "
                f"to {m0[m]} (rank {p_rank[m][m0[m]]}) in m0"
            )

    def test_receiver_optimal_is_best_for_receivers(self) -> None:
        m_prefs = {
            "m1": ["w1", "w2", "w3"],
            "m2": ["w1", "w3", "w2"],
            "m3": ["w2", "w1", "w3"],
        }
        w_prefs = {
            "w1": ["m2", "m1", "m3"],
            "w2": ["m3", "m1", "m2"],
            "w3": ["m1", "m3", "m2"],
        }
        m0 = gale_shapley(m_prefs, w_prefs)
        mz = receiver_optimal(m_prefs, w_prefs)
        inv_m0 = {v: k for k, v in m0.items()}
        inv_mz = {v: k for k, v in mz.items()}

        r_rank = {
            r: {p: i for i, p in enumerate(pl)} for r, pl in w_prefs.items()
        }

        # Every receiver must weakly prefer mz to m0.
        for w in w_prefs:
            assert r_rank[w][inv_mz[w]] <= r_rank[w][inv_m0[w]]


class TestDATrace:
    """Test that da_trace produces the same matching and valid trace."""

    def test_trace_matches_result(self) -> None:
        m_prefs, w_prefs = random_market(5, seed=123)
        matching, rounds = da_trace(m_prefs, w_prefs)
        expected = gale_shapley(m_prefs, w_prefs)
        assert matching == expected

    def test_trace_has_rounds(self) -> None:
        m_prefs, w_prefs = random_market(4, seed=7)
        _, rounds = da_trace(m_prefs, w_prefs)
        assert len(rounds) >= 1
        for rd in rounds:
            assert "proposals" in rd
            assert "rejections" in rd
            assert "holds" in rd


class TestRandomMarkets:
    """Statistical tests on random markets."""

    @pytest.mark.parametrize("n", [3, 5, 8, 10])
    def test_random_market_always_stable(self, n: int) -> None:
        for seed in range(5):
            m_prefs, w_prefs = random_market(n, seed=seed)
            matching = gale_shapley(m_prefs, w_prefs)
            assert is_stable(matching, m_prefs, w_prefs), (
                f"Unstable matching for n={n}, seed={seed}"
            )

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_receiver_optimal_always_stable(self, n: int) -> None:
        for seed in range(5):
            m_prefs, w_prefs = random_market(n, seed=seed)
            matching = receiver_optimal(m_prefs, w_prefs)
            assert is_stable(matching, m_prefs, w_prefs)

    @pytest.mark.parametrize("n", [3, 5, 10])
    def test_perfect_matching(self, n: int) -> None:
        m_prefs, w_prefs = random_market(n, seed=42)
        matching = gale_shapley(m_prefs, w_prefs)
        assert len(matching) == n


class TestWeakPreferences:
    """Test Gale-Shapley with ties."""

    def test_weak_simple(self) -> None:
        m_prefs = {
            "m1": [["w1", "w2"]],  # Indifferent between w1 and w2.
            "m2": [["w2"], ["w1"]],
        }
        w_prefs = {
            "w1": [["m1", "m2"]],
            "w2": [["m1"], ["m2"]],
        }
        matching = gale_shapley_weak(m_prefs, w_prefs)
        assert len(matching) == 2
        assert set(matching.values()) == {"w1", "w2"}

    def test_weak_3x3(self) -> None:
        m_prefs = {
            "m1": [["w1"], ["w2", "w3"]],
            "m2": [["w2"], ["w1"], ["w3"]],
            "m3": [["w3", "w1"], ["w2"]],
        }
        w_prefs = {
            "w1": [["m1", "m2"], ["m3"]],
            "w2": [["m3"], ["m2"], ["m1"]],
            "w3": [["m1"], ["m3", "m2"]],
        }
        matching = gale_shapley_weak(m_prefs, w_prefs)
        assert len(matching) == 3
