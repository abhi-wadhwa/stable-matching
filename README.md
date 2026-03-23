# Stable Matching Engine

A comprehensive implementation of stable matching algorithms, including the
Gale-Shapley deferred acceptance algorithm, stable matching lattice
enumeration via rotations, the hospital-resident problem, and an interactive
Streamlit visualization.

## Background

### The Stable Marriage Problem

Given two disjoint sets of agents (e.g., *proposers* and *receivers*), each of
size $n$, where every agent has a strict preference ranking over agents on the
other side, a **matching** $\mu$ pairs each proposer with exactly one receiver.
A matching is **stable** if there is no *blocking pair*:

$$
(m, w) \text{ blocks } \mu \iff m \text{ prefers } w \text{ to } \mu(m) \;\wedge\; w \text{ prefers } m \text{ to } \mu(w).
$$

Gale and Shapley (1962) proved that a stable matching always exists and gave a
constructive $O(n^2)$ algorithm -- **Deferred Acceptance (DA)** -- to find one.

### Deferred Acceptance

The DA algorithm operates in rounds:

1. Each unmatched proposer **proposes** to the most-preferred receiver they
   have not yet proposed to.
2. Each receiver **holds** the best proposal received so far and rejects the
   rest.
3. Rejected proposers return to step 1.

The algorithm terminates in at most $n^2$ proposals. The resulting matching is
**proposer-optimal**: every proposer gets their best achievable stable partner.
Symmetrically, running DA with the roles swapped yields the
**receiver-optimal** stable matching.

### The Stable Matching Lattice

The set of all stable matchings forms a **distributive lattice** under the
partial order where $M_1 \leq M_2$ iff every proposer weakly prefers $M_1$
to $M_2$. Key properties:

- The **proposer-optimal** matching $M_0$ is the lattice **top** (supremum).
- The **receiver-optimal** matching $M_z$ is the lattice **bottom** (infimum).
- **Join** $M_1 \vee M_2$: each proposer gets the partner they prefer more.
- **Meet** $M_1 \wedge M_2$: each proposer gets the partner they prefer less.
- Both join and meet of stable matchings are themselves stable.

### Rotations

A **rotation** $\rho = ((m_0, w_0), (m_1, w_1), \ldots, (m_{k-1}, w_{k-1}))$
is a cyclic sequence where eliminating $\rho$ from a stable matching $M$
re-matches each $m_i$ to $w_{(i+1) \bmod k}$, producing another stable
matching. The set of all rotations, with a partial order capturing
dependencies, fully characterises the lattice structure. All stable matchings
correspond to *closed subsets* of the rotation poset.

### Hospital-Resident Problem

The **many-to-one** extension where hospitals have quotas $q_h$. Modified DA
lets hospitals hold up to $q_h$ residents. The **Rural Hospital Theorem**
states:

1. The same set of residents is matched in every stable matching.
2. Any hospital under-subscribed in one stable matching is under-subscribed in
   all, with the same set of residents.

## Features

| Module | Description |
|--------|-------------|
| `gale_shapley.py` | Proposer-optimal and receiver-optimal DA, round-by-round trace, weak preferences |
| `stability.py` | Blocking pair detection, weak stability checking |
| `rotations.py` | Rotation finding, rotation elimination, rotation poset construction |
| `lattice.py` | Enumerate all stable matchings, lattice join/meet, Hasse diagram |
| `hospital_resident.py` | Many-to-one DA (resident/hospital proposing), Rural Hospital Theorem verification |
| `generators.py` | Random markets (uniform, correlated/master-list, tiered, incomplete) |
| `viz/app.py` | Streamlit interactive dashboard |
| `cli.py` | Command-line interface |

## Installation

```bash
# Clone the repository
git clone https://github.com/abhi-wadhwa/stable-matching.git
cd stable-matching

# Install in development mode
pip install -e ".[dev]"
```

## Usage

### Command Line

```bash
# Solve a random instance
stable-matching solve -n 5 --seed 42

# Enumerate all stable matchings
stable-matching solve -n 4 --enumerate

# Hospital-resident problem
stable-matching hospital --residents 8 --hospitals 3

# Check stability from JSON
stable-matching check -i instance.json
```

### Python API

```python
from src.core.gale_shapley import gale_shapley, receiver_optimal
from src.core.stability import find_blocking_pairs, is_stable
from src.core.lattice import enumerate_stable_matchings

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

# Proposer-optimal stable matching
m0 = gale_shapley(m_prefs, w_prefs)
print(m0)  # {'Alice': 'Xavier', 'Beatrix': 'Yves', 'Cleo': 'Zeus'}

# Check stability
assert is_stable(m0, m_prefs, w_prefs)

# Receiver-optimal
mz = receiver_optimal(m_prefs, w_prefs)

# Enumerate all stable matchings
all_matchings = enumerate_stable_matchings(m_prefs, w_prefs)
print(f"Found {len(all_matchings)} stable matchings")
```

### Streamlit App

```bash
streamlit run src/viz/app.py
```

The interactive dashboard provides:
- **Matching Visualizer**: Bipartite graph with animated proposal rounds
- **Lattice Viewer**: Hasse diagram of all stable matchings
- **Stability Checker**: JSON input with blocking pair highlighting
- **Hospital-Resident**: Many-to-one matching with quota management

### Docker

```bash
docker build -t stable-matching .
docker run -p 8501:8501 stable-matching
```

## Testing

```bash
# Run all tests
make test

# With coverage
make test-cov

# Lint and type check
make lint
make typecheck
```

## Project Structure

```
stable-matching/
├── src/
│   ├── core/
│   │   ├── gale_shapley.py      # DA algorithm
│   │   ├── stability.py         # Blocking pair checker
│   │   ├── rotations.py         # Rotation finding
│   │   ├── lattice.py           # Lattice enumeration
│   │   ├── hospital_resident.py # Many-to-one matching
│   │   └── generators.py        # Random market generation
│   ├── viz/
│   │   └── app.py               # Streamlit dashboard
│   └── cli.py                   # CLI entry point
├── tests/
│   ├── test_gale_shapley.py
│   ├── test_stability.py
│   ├── test_lattice.py
│   └── test_hospital.py
├── examples/
│   └── demo.py
├── pyproject.toml
├── Makefile
├── Dockerfile
└── .github/workflows/ci.yml
```

## References

- Gale, D. & Shapley, L. S. (1962). College Admissions and the Stability of
  Marriage. *The American Mathematical Monthly*, 69(1), 9--15.
- Roth, A. E. (1984). The Evolution of the Labor Market for Medical Interns
  and Residents. *Journal of Political Economy*, 92(6), 991--1016.
- Irving, R. W. & Leather, P. (1986). The Complexity of Counting Stable
  Marriages. *SIAM J. Comput.*, 15(3), 655--667.
- Knuth, D. E. (1976). *Mariages stables*. Les Presses de l'Universite de
  Montreal.

## License

MIT
