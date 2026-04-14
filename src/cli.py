"""
Command-line interface for the stable matching engine.
"""

from __future__ import annotations

import argparse
import json

from src.core.gale_shapley import gale_shapley, receiver_optimal
from src.core.generators import random_hospital_market, random_market
from src.core.hospital_resident import find_hr_blocking_pairs, hospital_resident_da
from src.core.lattice import enumerate_stable_matchings
from src.core.stability import find_blocking_pairs


def cmd_solve(args: argparse.Namespace) -> None:
    """Solve a stable matching instance."""
    if args.input:
        with open(args.input) as f:
            data = json.load(f)
        m_prefs = data["proposer_prefs"]
        w_prefs = data["receiver_prefs"]
    else:
        n = args.n or 4
        seed = args.seed
        m_prefs, w_prefs = random_market(n, seed=seed)
        print(f"Generated random market (n={n}, seed={seed})")

    print("\n=== Proposer Preferences ===")
    for p, pl in sorted(m_prefs.items()):
        print(f"  {p}: {' > '.join(pl)}")

    print("\n=== Receiver Preferences ===")
    for r, pl in sorted(w_prefs.items()):
        print(f"  {r}: {' > '.join(pl)}")

    # Proposer-optimal.
    m0 = gale_shapley(m_prefs, w_prefs)
    print("\n=== Proposer-Optimal Matching ===")
    for p, r in sorted(m0.items()):
        print(f"  {p} <-> {r}")
    bp0 = find_blocking_pairs(m0, m_prefs, w_prefs)
    print(f"  Blocking pairs: {len(bp0)}")

    # Receiver-optimal.
    mz = receiver_optimal(m_prefs, w_prefs)
    print("\n=== Receiver-Optimal Matching ===")
    for p, r in sorted(mz.items()):
        print(f"  {p} <-> {r}")
    bpz = find_blocking_pairs(mz, m_prefs, w_prefs)
    print(f"  Blocking pairs: {len(bpz)}")

    if args.enumerate:
        all_m = enumerate_stable_matchings(m_prefs, w_prefs)
        print(f"\n=== All Stable Matchings ({len(all_m)}) ===")
        for i, m in enumerate(all_m):
            print(f"  M{i}: {dict(sorted(m.items()))}")


def cmd_hospital(args: argparse.Namespace) -> None:
    """Solve a hospital-resident instance."""
    if args.input:
        with open(args.input) as f:
            data = json.load(f)
        r_prefs = data["resident_prefs"]
        h_prefs = data["hospital_prefs"]
        quotas = data["quotas"]
    else:
        nr = args.residents or 6
        nh = args.hospitals or 3
        r_prefs, h_prefs, quotas = random_hospital_market(nr, nh, seed=args.seed)
        print(f"Generated random hospital market ({nr} residents, {nh} hospitals)")

    print("\n=== Resident Preferences ===")
    for r, pl in sorted(r_prefs.items()):
        print(f"  {r}: {' > '.join(pl)}")

    print("\n=== Hospital Preferences ===")
    for h, pl in sorted(h_prefs.items()):
        print(f"  {h}: {' > '.join(pl)}")

    print("\n=== Quotas ===")
    for h, q in sorted(quotas.items()):
        print(f"  {h}: {q}")

    matching = hospital_resident_da(r_prefs, h_prefs, quotas, proposer="resident")
    print("\n=== Resident-Optimal Matching ===")
    for h, rs in sorted(matching.items()):
        print(f"  {h}: {rs}")

    bp = find_hr_blocking_pairs(matching, r_prefs, h_prefs, quotas)
    print(f"  Blocking pairs: {len(bp)}")


def cmd_check(args: argparse.Namespace) -> None:
    """Check stability of a matching."""
    with open(args.input) as f:
        data = json.load(f)

    matching = data["matching"]
    m_prefs = data["proposer_prefs"]
    w_prefs = data["receiver_prefs"]

    bp = find_blocking_pairs(matching, m_prefs, w_prefs)
    if bp:
        print(f"UNSTABLE -- {len(bp)} blocking pair(s):")
        for p, r in bp:
            print(f"  ({p}, {r})")
    else:
        print("STABLE -- no blocking pairs.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stable Matching Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # solve
    p_solve = sub.add_parser("solve", help="Solve a stable matching instance")
    p_solve.add_argument("-i", "--input", help="JSON input file")
    p_solve.add_argument("-n", type=int, help="Number of agents (random)")
    p_solve.add_argument("--seed", type=int, default=42)
    p_solve.add_argument("--enumerate", action="store_true",
                         help="Enumerate all stable matchings")

    # hospital
    p_hosp = sub.add_parser("hospital", help="Hospital-resident matching")
    p_hosp.add_argument("-i", "--input", help="JSON input file")
    p_hosp.add_argument("--residents", type=int)
    p_hosp.add_argument("--hospitals", type=int)
    p_hosp.add_argument("--seed", type=int, default=42)

    # check
    p_check = sub.add_parser("check", help="Check stability of a matching")
    p_check.add_argument("-i", "--input", required=True, help="JSON input file")

    args = parser.parse_args()

    if args.command == "solve":
        cmd_solve(args)
    elif args.command == "hospital":
        cmd_hospital(args)
    elif args.command == "check":
        cmd_check(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
