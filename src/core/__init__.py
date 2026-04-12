"""Core algorithms for stable matching."""

from src.core.gale_shapley import gale_shapley, gale_shapley_weak
from src.core.stability import find_blocking_pairs, is_stable
from src.core.lattice import enumerate_stable_matchings, lattice_join, lattice_meet
from src.core.rotations import find_rotations, build_rotation_poset
from src.core.hospital_resident import hospital_resident_da
from src.core.generators import (
    random_market,
    correlated_market,
    random_hospital_market,
)

__all__ = [
    "gale_shapley",
    "gale_shapley_weak",
    "find_blocking_pairs",
    "is_stable",
    "enumerate_stable_matchings",
    "lattice_join",
    "lattice_meet",
    "find_rotations",
    "build_rotation_poset",
    "hospital_resident_da",
    "random_market",
    "correlated_market",
    "random_hospital_market",
]
