"""
Streamlit Visualization App
============================

Interactive dashboard for exploring stable matching algorithms:
- Preference table editor
- Animated proposal rounds (bipartite graph)
- Lattice viewer (Hasse diagram of all stable matchings)
- Stability checker with highlighted blocking pairs
"""

from __future__ import annotations

import json

import graphviz
import streamlit as st

from src.core.gale_shapley import da_trace, receiver_optimal
from src.core.generators import correlated_market, random_market
from src.core.hospital_resident import find_hr_blocking_pairs, hospital_resident_da
from src.core.lattice import enumerate_stable_matchings, hasse_diagram
from src.core.stability import find_blocking_pairs, is_stable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def draw_bipartite_matching(
    proposers: list[str],
    receivers: list[str],
    matching: dict[str, str],
    blocking_pairs: list[tuple[str, str]] | None = None,
    proposals: list[tuple[str, str]] | None = None,
    rejections: list[tuple[str, str]] | None = None,
    title: str = "Matching",
) -> graphviz.Digraph:
    """Draw a bipartite matching using graphviz."""
    dot = graphviz.Digraph(comment=title, engine="dot")
    dot.attr(rankdir="LR", label=title, fontsize="16", labelloc="t")

    # Left column: proposers.
    with dot.subgraph(name="cluster_proposers") as c:
        c.attr(label="Proposers", style="dashed", color="blue")
        for p in proposers:
            c.node(p, p, shape="circle", style="filled", fillcolor="lightblue")

    # Right column: receivers.
    with dot.subgraph(name="cluster_receivers") as c:
        c.attr(label="Receivers", style="dashed", color="pink")
        for r in receivers:
            c.node(r, r, shape="circle", style="filled", fillcolor="lightyellow")

    # Matching edges.
    for p, r in matching.items():
        dot.edge(p, r, color="green", penwidth="2.5")

    # Blocking pair edges.
    if blocking_pairs:
        for p, r in blocking_pairs:
            dot.edge(p, r, color="red", style="dashed", penwidth="1.5",
                     label="blocks", fontcolor="red", fontsize="10")

    # Proposal edges (for animation).
    if proposals:
        for p, r in proposals:
            is_rejected = rejections and (p, r) in rejections
            color = "gray" if is_rejected else "orange"
            style = "dotted" if is_rejected else "bold"
            dot.edge(p, r, color=color, style=style, penwidth="1.0")

    return dot


def draw_hasse_diagram(
    matchings: list[dict[str, str]],
    edges: list[tuple[int, int]],
) -> graphviz.Digraph:
    """Draw the Hasse diagram of the stable matching lattice."""
    dot = graphviz.Digraph(comment="Stable Matching Lattice")
    dot.attr(rankdir="TB", label="Stable Matching Lattice", fontsize="16", labelloc="t")

    for i, m in enumerate(matchings):
        label = "\\n".join(f"{k}->{v}" for k, v in sorted(m.items()))
        shape = "box"
        color = "lightblue"
        if i == 0:
            color = "lightgreen"
            label = f"M0 (proposer-optimal)\\n{label}"
        elif i == len(matchings) - 1 and len(matchings) > 1:
            color = "lightyellow"
        dot.node(str(i), label, shape=shape, style="filled", fillcolor=color)

    for i, j in edges:
        dot.edge(str(i), str(j))

    return dot


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_matching_visualizer() -> None:
    """Main matching visualizer page."""
    st.header("Matching Visualizer")

    col1, col2 = st.columns(2)
    with col1:
        n = st.slider("Number of agents per side", 2, 8, 4)
    with col2:
        market_type = st.selectbox("Market type", ["Uniform Random", "Correlated"])

    seed = st.number_input("Random seed", value=42, step=1)

    if market_type == "Uniform Random":
        m_prefs, w_prefs = random_market(n, seed=int(seed))
    else:
        noise = st.slider("Correlation noise", 0.0, 2.0, 0.3, 0.1)
        m_prefs, w_prefs = correlated_market(n, noise=noise, seed=int(seed))

    # Display preferences.
    st.subheader("Preferences")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Proposer preferences:**")
        for p, pl in sorted(m_prefs.items()):
            st.text(f"  {p}: {' > '.join(pl)}")
    with col2:
        st.write("**Receiver preferences:**")
        for r, pl in sorted(w_prefs.items()):
            st.text(f"  {r}: {' > '.join(pl)}")

    # Run DA with trace.
    matching, rounds = da_trace(m_prefs, w_prefs)
    proposers = sorted(m_prefs.keys())
    receivers = sorted(w_prefs.keys())

    st.subheader("Proposal Rounds (Animation)")

    if rounds:
        round_idx = st.slider("Round", 0, len(rounds) - 1, 0)
        rd = rounds[round_idx]
        dot = draw_bipartite_matching(
            proposers, receivers, {},
            proposals=rd["proposals"],
            rejections=rd["rejections"],
            title=f"Round {round_idx + 1}",
        )
        st.graphviz_chart(dot)

        st.write(f"**Proposals:** {rd['proposals']}")
        st.write(f"**Rejections:** {rd['rejections']}")
        st.write(f"**Holds:** {rd['holds']}")

    st.subheader("Final Matching")
    blocking = find_blocking_pairs(matching, m_prefs, w_prefs)
    dot = draw_bipartite_matching(
        proposers, receivers, matching,
        blocking_pairs=blocking,
        title="Proposer-Optimal Stable Matching",
    )
    st.graphviz_chart(dot)

    if blocking:
        st.error(f"Found {len(blocking)} blocking pair(s): {blocking}")
    else:
        st.success("Matching is stable (no blocking pairs).")

    st.write("**Matching:**", matching)

    # Receiver-optimal.
    r_matching = receiver_optimal(m_prefs, w_prefs)
    st.subheader("Receiver-Optimal Matching")
    r_blocking = find_blocking_pairs(r_matching, m_prefs, w_prefs)
    dot2 = draw_bipartite_matching(
        proposers, receivers, r_matching,
        blocking_pairs=r_blocking,
        title="Receiver-Optimal Stable Matching",
    )
    st.graphviz_chart(dot2)
    st.write("**Matching:**", r_matching)


def page_lattice_viewer() -> None:
    """Lattice viewer page."""
    st.header("Stable Matching Lattice")
    st.write(
        "Enumerate all stable matchings and view the Hasse diagram. "
        "Only feasible for small instances."
    )

    n = st.slider("Number of agents per side", 2, 5, 3, key="lattice_n")
    seed = st.number_input("Random seed", value=7, step=1, key="lattice_seed")
    m_prefs, w_prefs = random_market(n, seed=int(seed))

    st.subheader("Preferences")
    col1, col2 = st.columns(2)
    with col1:
        for p, pl in sorted(m_prefs.items()):
            st.text(f"  {p}: {' > '.join(pl)}")
    with col2:
        for r, pl in sorted(w_prefs.items()):
            st.text(f"  {r}: {' > '.join(pl)}")

    with st.spinner("Enumerating stable matchings..."):
        all_matchings = enumerate_stable_matchings(m_prefs, w_prefs)

    st.write(f"**Found {len(all_matchings)} stable matching(s).**")

    for i, m in enumerate(all_matchings):
        stable = is_stable(m, m_prefs, w_prefs)
        label = "Stable" if stable else "UNSTABLE"
        st.write(f"  M{i}: {dict(sorted(m.items()))} -- {label}")

    if len(all_matchings) > 1:
        edges = hasse_diagram(all_matchings, m_prefs)
        dot = draw_hasse_diagram(all_matchings, edges)
        st.graphviz_chart(dot)
    else:
        st.info("Only one stable matching exists -- lattice is trivial.")


def page_stability_checker() -> None:
    """Stability checker page."""
    st.header("Stability Checker")
    st.write("Enter a matching and preferences to check for blocking pairs.")

    st.subheader("Example (editable)")

    default_m = '{"m1": ["w1", "w2", "w3"], "m2": ["w2", "w1", "w3"], "m3": ["w1", "w3", "w2"]}'
    default_w = '{"w1": ["m2", "m1", "m3"], "w2": ["m1", "m3", "m2"], "w3": ["m3", "m2", "m1"]}'
    default_match = '{"m1": "w1", "m2": "w2", "m3": "w3"}'

    m_prefs_str = st.text_area("Proposer prefs (JSON)", default_m)
    w_prefs_str = st.text_area("Receiver prefs (JSON)", default_w)
    matching_str = st.text_area("Matching (JSON)", default_match)

    if st.button("Check stability"):
        try:
            m_prefs = json.loads(m_prefs_str)
            w_prefs = json.loads(w_prefs_str)
            matching = json.loads(matching_str)

            blocking = find_blocking_pairs(matching, m_prefs, w_prefs)
            proposers = sorted(m_prefs.keys())
            receivers = sorted(w_prefs.keys())

            dot = draw_bipartite_matching(
                proposers, receivers, matching,
                blocking_pairs=blocking,
                title="Stability Check",
            )
            st.graphviz_chart(dot)

            if blocking:
                st.error(f"UNSTABLE -- {len(blocking)} blocking pair(s):")
                for bp in blocking:
                    st.write(f"  ({bp[0]}, {bp[1]})")
            else:
                st.success("Matching is STABLE.")
        except Exception as e:
            st.error(f"Error: {e}")


def page_hospital_resident() -> None:
    """Hospital-resident page."""
    st.header("Hospital-Resident Matching")

    n_res = st.slider("Residents", 3, 10, 6)
    n_hosp = st.slider("Hospitals", 2, 5, 3)
    seed = st.number_input("Random seed", value=42, step=1, key="hr_seed")

    from src.core.generators import random_hospital_market

    r_prefs, h_prefs, quotas = random_hospital_market(
        n_res, n_hosp, min_quota=1, max_quota=3, seed=int(seed)
    )

    st.subheader("Market")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Resident prefs:**")
        for r, pl in sorted(r_prefs.items()):
            st.text(f"  {r}: {' > '.join(pl)}")
    with col2:
        st.write("**Hospital prefs:**")
        for h, pl in sorted(h_prefs.items()):
            st.text(f"  {h}: {' > '.join(pl)}")
    with col3:
        st.write("**Quotas:**")
        for h, q in sorted(quotas.items()):
            st.text(f"  {h}: {q}")

    # Resident-optimal.
    r_matching = hospital_resident_da(r_prefs, h_prefs, quotas, proposer="resident")
    st.subheader("Resident-Optimal Matching")
    for h, rs in sorted(r_matching.items()):
        st.write(f"  {h} ({quotas[h]} slots): {rs}")

    blocking = find_hr_blocking_pairs(r_matching, r_prefs, h_prefs, quotas)
    if blocking:
        st.error(f"Found {len(blocking)} blocking pair(s): {blocking}")
    else:
        st.success("Matching is stable.")

    # Hospital-optimal.
    h_matching = hospital_resident_da(r_prefs, h_prefs, quotas, proposer="hospital")
    st.subheader("Hospital-Optimal Matching")
    for h, rs in sorted(h_matching.items()):
        st.write(f"  {h} ({quotas[h]} slots): {rs}")

    h_blocking = find_hr_blocking_pairs(h_matching, r_prefs, h_prefs, quotas)
    if h_blocking:
        st.error(f"Found {len(h_blocking)} blocking pair(s): {h_blocking}")
    else:
        st.success("Matching is stable.")


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Stable Matching Engine",
        page_icon="<>",
        layout="wide",
    )
    st.title("Stable Matching Engine")
    st.markdown(
        "Interactive exploration of Gale-Shapley deferred acceptance, "
        "stable matching lattices, and the hospital-resident problem."
    )

    page = st.sidebar.radio(
        "Navigate",
        [
            "Matching Visualizer",
            "Lattice Viewer",
            "Stability Checker",
            "Hospital-Resident",
        ],
    )

    if page == "Matching Visualizer":
        page_matching_visualizer()
    elif page == "Lattice Viewer":
        page_lattice_viewer()
    elif page == "Stability Checker":
        page_stability_checker()
    elif page == "Hospital-Resident":
        page_hospital_resident()


if __name__ == "__main__":
    main()
