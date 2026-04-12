"""Tests for the hospital-resident matching algorithm."""

from __future__ import annotations

import pytest

from src.core.hospital_resident import (
    hospital_resident_da,
    find_hr_blocking_pairs,
    verify_rural_hospital,
)
from src.core.generators import random_hospital_market


class TestHospitalResidentDA:
    """Basic hospital-resident tests."""

    def test_simple_instance(self) -> None:
        r_prefs = {
            "r1": ["h1", "h2"],
            "r2": ["h1", "h2"],
            "r3": ["h2", "h1"],
        }
        h_prefs = {
            "h1": ["r1", "r2", "r3"],
            "h2": ["r3", "r1", "r2"],
        }
        quotas = {"h1": 2, "h2": 1}

        matching = hospital_resident_da(r_prefs, h_prefs, quotas, proposer="resident")

        # Check no blocking pairs.
        bp = find_hr_blocking_pairs(matching, r_prefs, h_prefs, quotas)
        assert bp == [], f"Blocking pairs: {bp}"

        # Check quotas respected.
        for h, rs in matching.items():
            assert len(rs) <= quotas[h]

        # Check all residents are matched (since there's room).
        all_matched = [r for rs in matching.values() for r in rs]
        assert len(all_matched) == 3

    def test_hospital_proposing(self) -> None:
        r_prefs = {
            "r1": ["h1", "h2"],
            "r2": ["h2", "h1"],
            "r3": ["h1", "h2"],
        }
        h_prefs = {
            "h1": ["r1", "r3", "r2"],
            "h2": ["r2", "r1", "r3"],
        }
        quotas = {"h1": 2, "h2": 2}

        matching = hospital_resident_da(r_prefs, h_prefs, quotas, proposer="hospital")
        bp = find_hr_blocking_pairs(matching, r_prefs, h_prefs, quotas)
        assert bp == []

    @pytest.mark.parametrize("seed", range(10))
    def test_random_resident_optimal_stable(self, seed: int) -> None:
        r_prefs, h_prefs, quotas = random_hospital_market(
            8, 3, min_quota=2, max_quota=4, seed=seed
        )
        matching = hospital_resident_da(r_prefs, h_prefs, quotas, proposer="resident")
        bp = find_hr_blocking_pairs(matching, r_prefs, h_prefs, quotas)
        assert bp == [], f"Seed {seed}: blocking pairs {bp}"

    @pytest.mark.parametrize("seed", range(10))
    def test_random_hospital_optimal_stable(self, seed: int) -> None:
        r_prefs, h_prefs, quotas = random_hospital_market(
            8, 3, min_quota=2, max_quota=4, seed=seed
        )
        matching = hospital_resident_da(r_prefs, h_prefs, quotas, proposer="hospital")
        bp = find_hr_blocking_pairs(matching, r_prefs, h_prefs, quotas)
        assert bp == [], f"Seed {seed}: blocking pairs {bp}"


class TestQuotas:
    """Test quota enforcement."""

    def test_quota_respected(self) -> None:
        r_prefs = {
            "r1": ["h1"],
            "r2": ["h1"],
            "r3": ["h1"],
        }
        h_prefs = {
            "h1": ["r1", "r2", "r3"],
        }
        quotas = {"h1": 2}

        matching = hospital_resident_da(r_prefs, h_prefs, quotas)
        assert len(matching["h1"]) <= 2

    def test_quota_one(self) -> None:
        """Hospital with quota 1 behaves like one-to-one."""
        r_prefs = {
            "r1": ["h1", "h2"],
            "r2": ["h1", "h2"],
        }
        h_prefs = {
            "h1": ["r1", "r2"],
            "h2": ["r2", "r1"],
        }
        quotas = {"h1": 1, "h2": 1}
        matching = hospital_resident_da(r_prefs, h_prefs, quotas)
        assert len(matching["h1"]) == 1
        assert len(matching["h2"]) == 1


class TestRuralHospitalTheorem:
    """Test the Rural Hospital Theorem."""

    def test_rural_hospital_basic(self) -> None:
        """Both stable matchings should match the same residents."""
        r_prefs = {
            "r1": ["h1", "h2"],
            "r2": ["h1", "h2"],
            "r3": ["h2", "h1"],
        }
        h_prefs = {
            "h1": ["r1", "r2", "r3"],
            "h2": ["r3", "r1", "r2"],
        }
        quotas = {"h1": 2, "h2": 1}

        m_res = hospital_resident_da(r_prefs, h_prefs, quotas, proposer="resident")
        m_hosp = hospital_resident_da(r_prefs, h_prefs, quotas, proposer="hospital")

        assert verify_rural_hospital([m_res, m_hosp])

    @pytest.mark.parametrize("seed", range(5))
    def test_rural_hospital_random(self, seed: int) -> None:
        r_prefs, h_prefs, quotas = random_hospital_market(
            6, 3, min_quota=1, max_quota=3, seed=seed
        )
        m_res = hospital_resident_da(r_prefs, h_prefs, quotas, proposer="resident")
        m_hosp = hospital_resident_da(r_prefs, h_prefs, quotas, proposer="hospital")

        # Same set of matched residents.
        res_set_1 = set(r for rs in m_res.values() for r in rs)
        res_set_2 = set(r for rs in m_hosp.values() for r in rs)
        assert res_set_1 == res_set_2, (
            f"Seed {seed}: different matched residents: {res_set_1} vs {res_set_2}"
        )


class TestBlockingPairsHR:
    """Test blocking pair detection for hospital-resident."""

    def test_stable_has_no_blocking(self) -> None:
        r_prefs = {
            "r1": ["h1", "h2"],
            "r2": ["h2", "h1"],
        }
        h_prefs = {
            "h1": ["r1", "r2"],
            "h2": ["r2", "r1"],
        }
        quotas = {"h1": 1, "h2": 1}
        matching = hospital_resident_da(r_prefs, h_prefs, quotas)
        bp = find_hr_blocking_pairs(matching, r_prefs, h_prefs, quotas)
        assert bp == []

    def test_unstable_has_blocking(self) -> None:
        r_prefs = {
            "r1": ["h1", "h2"],
            "r2": ["h1", "h2"],
        }
        h_prefs = {
            "h1": ["r1", "r2"],
            "h2": ["r1", "r2"],
        }
        quotas = {"h1": 1, "h2": 1}
        # Force unstable: r1->h2, r2->h1 (but both prefer h1, and h1 prefers r1).
        matching = {"h1": ["r2"], "h2": ["r1"]}
        bp = find_hr_blocking_pairs(matching, r_prefs, h_prefs, quotas)
        assert len(bp) >= 1
        assert ("r1", "h1") in bp
