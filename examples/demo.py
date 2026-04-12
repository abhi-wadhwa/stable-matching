"""
Demonstration of the stable matching engine.

Run with: python examples/demo.py
"""

from src.core.gale_shapley import gale_shapley, receiver_optimal, da_trace
from src.core.stability import find_blocking_pairs, is_stable
from src.core.lattice import enumerate_stable_matchings, lattice_join, lattice_meet, hasse_diagram
from src.core.hospital_resident import hospital_resident_da, find_hr_blocking_pairs, verify_rural_hospital
from src.core.generators import random_market, correlated_market, random_hospital_market


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def demo_basic_gale_shapley() -> None:
    section("1. Basic Gale-Shapley (Stable Marriage)")

    m_prefs = {
        "Alice":   ["Xavier", "Yves", "Zeus"],
        "Beatrix": ["Yves", "Xavier", "Zeus"],
        "Cleo":    ["Xavier", "Zeus", "Yves"],
    }
    w_prefs = {
        "Xavier": ["Beatrix", "Alice", "Cleo"],
        "Yves":   ["Alice", "Cleo", "Beatrix"],
        "Zeus":   ["Alice", "Beatrix", "Cleo"],
    }

    print("Proposer preferences:")
    for p, pl in m_prefs.items():
        print(f"  {p}: {' > '.join(pl)}")
    print("\nReceiver preferences:")
    for r, pl in w_prefs.items():
        print(f"  {r}: {' > '.join(pl)}")

    # Proposer-optimal.
    m0 = gale_shapley(m_prefs, w_prefs)
    print(f"\nProposer-optimal matching:")
    for p, r in sorted(m0.items()):
        print(f"  {p} <-> {r}")
    print(f"  Stable: {is_stable(m0, m_prefs, w_prefs)}")

    # Receiver-optimal.
    mz = receiver_optimal(m_prefs, w_prefs)
    print(f"\nReceiver-optimal matching:")
    for p, r in sorted(mz.items()):
        print(f"  {p} <-> {r}")
    print(f"  Stable: {is_stable(mz, m_prefs, w_prefs)}")


def demo_trace() -> None:
    section("2. DA Round-by-Round Trace")

    m_prefs = {
        "m1": ["w1", "w2", "w3"],
        "m2": ["w1", "w3", "w2"],
        "m3": ["w2", "w1", "w3"],
    }
    w_prefs = {
        "w1": ["m2", "m3", "m1"],
        "w2": ["m1", "m2", "m3"],
        "w3": ["m3", "m1", "m2"],
    }

    matching, rounds = da_trace(m_prefs, w_prefs)
    for i, rd in enumerate(rounds):
        print(f"Round {i+1}:")
        print(f"  Proposals:  {rd['proposals']}")
        print(f"  Rejections: {rd['rejections']}")
        print(f"  Holds:      {rd['holds']}")
    print(f"\nFinal matching: {matching}")


def demo_lattice() -> None:
    section("3. Stable Matching Lattice")

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
    print(f"Found {len(all_m)} stable matching(s):\n")
    for i, m in enumerate(all_m):
        bp = find_blocking_pairs(m, m_prefs, w_prefs)
        print(f"  M{i}: {dict(sorted(m.items()))}  (blocking pairs: {len(bp)})")

    if len(all_m) >= 2:
        edges = hasse_diagram(all_m, m_prefs)
        print(f"\nHasse diagram edges (i covers j): {edges}")

        j = lattice_join(all_m[0], all_m[-1], m_prefs, w_prefs)
        mt = lattice_meet(all_m[0], all_m[-1], m_prefs, w_prefs)
        print(f"\nJoin(M0, M{len(all_m)-1}) = {dict(sorted(j.items()))}")
        print(f"Meet(M0, M{len(all_m)-1}) = {dict(sorted(mt.items()))}")


def demo_hospital_resident() -> None:
    section("4. Hospital-Resident Problem")

    r_prefs = {
        "r1": ["h1", "h2", "h3"],
        "r2": ["h1", "h3", "h2"],
        "r3": ["h2", "h1", "h3"],
        "r4": ["h3", "h2", "h1"],
        "r5": ["h1", "h2", "h3"],
        "r6": ["h2", "h3", "h1"],
    }
    h_prefs = {
        "h1": ["r1", "r2", "r5", "r3", "r4", "r6"],
        "h2": ["r3", "r6", "r1", "r4", "r2", "r5"],
        "h3": ["r4", "r2", "r6", "r5", "r1", "r3"],
    }
    quotas = {"h1": 2, "h2": 2, "h3": 2}

    print("Quotas:", quotas)
    print()

    # Resident-optimal.
    m_res = hospital_resident_da(r_prefs, h_prefs, quotas, proposer="resident")
    print("Resident-optimal matching:")
    for h, rs in sorted(m_res.items()):
        print(f"  {h}: {rs}")
    bp = find_hr_blocking_pairs(m_res, r_prefs, h_prefs, quotas)
    print(f"  Blocking pairs: {len(bp)}")

    # Hospital-optimal.
    m_hosp = hospital_resident_da(r_prefs, h_prefs, quotas, proposer="hospital")
    print("\nHospital-optimal matching:")
    for h, rs in sorted(m_hosp.items()):
        print(f"  {h}: {rs}")
    bp2 = find_hr_blocking_pairs(m_hosp, r_prefs, h_prefs, quotas)
    print(f"  Blocking pairs: {len(bp2)}")

    # Rural Hospital Theorem.
    rht = verify_rural_hospital([m_res, m_hosp])
    print(f"\nRural Hospital Theorem holds: {rht}")


def demo_random_market() -> None:
    section("5. Random Market Statistics")

    n = 10
    num_trials = 20

    total_matchings = 0
    for seed in range(num_trials):
        m_prefs, w_prefs = random_market(n, seed=seed)
        all_m = enumerate_stable_matchings(m_prefs, w_prefs)
        total_matchings += len(all_m)

    avg = total_matchings / num_trials
    print(f"Average stable matchings for n={n} over {num_trials} trials: {avg:.1f}")

    # Correlated market.
    print("\nCorrelated market (noise=0.1):")
    m_prefs, w_prefs = correlated_market(n, noise=0.1, seed=42)
    m0 = gale_shapley(m_prefs, w_prefs)
    mz = receiver_optimal(m_prefs, w_prefs)
    same = m0 == mz
    print(f"  Proposer-optimal == Receiver-optimal: {same}")


if __name__ == "__main__":
    demo_basic_gale_shapley()
    demo_trace()
    demo_lattice()
    demo_hospital_resident()
    demo_random_market()

    section("Done!")
    print("Run the Streamlit app with: streamlit run src/viz/app.py")
