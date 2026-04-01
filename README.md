# stable-matching

gale-shapley and friends. the algorithm that matches medical residents to hospitals and earned shapley a nobel prize.

## what this is

- **deferred acceptance** — the classic gale-shapley algorithm. proposer-optimal, receiver-pessimal. O(n²) and strategy-proof for the proposing side
- **stable matching lattice** — all stable matchings form a distributive lattice. this computes the full lattice structure
- **hospital-resident problem** — the many-to-one extension. hospitals have capacities, residents have preferences. rural hospital theorem included

## running it

```bash
pip install -r requirements.txt
python main.py
```

## the idea

a matching is **stable** if there's no blocking pair — no unmatched pair who'd both rather be with each other than their current partners. gale-shapley proves stable matchings always exist by having one side propose and the other side hold/reject.

the deep result: the set of stable matchings has lattice structure. the proposer-optimal matching is simultaneously the receiver-pessimal matching, and vice versa. this means there's a fundamental conflict of interest between the two sides of the market.

the real-world version (NRMP, school choice, kidney exchange) adds complications — couples, quotas, incentive compatibility — but the core algorithm is still gale-shapley under the hood.
