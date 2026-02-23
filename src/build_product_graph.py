"""
src/build_product_graph.py
--------------------------
Builds a unified cross-retailer product graph from matched_products.csv.

Input format (long):
    retailer, product_id, product_url, brand, brand_normalized,
    product_name, category, price, rating

    — One row per product, retailer column distinguishes Sephora vs Ulta.
    — 803 Sephora rows + 3,812 Ulta rows = combined catalog of brands present
      on both platforms. No pre-existing product-level pairings.

What this script does:
    1. Splits into Sephora / Ulta catalogs
    2. Cleans product names (strips brand suffixes, size info)
    3. Fuzzy-matches Sephora → Ulta within each shared brand
    4. Builds a NetworkX DiGraph from matched pairs + Ulta-only products
    5. Exports graph artifacts and four notebook-ready visualizations

Catalog philosophy — Ulta is PRIMARY:
    Ulta has ~4.7× more products than Sephora. The graph treats Ulta as the
    canonical catalog. Every Ulta product becomes a node (matched or ulta_only).
    Sephora data layers on top where a match exists.

Graph node types:
    brand           — one per brand (keyed on brand_normalized)
    matched_product — one per Sephora↔Ulta pair
      ├── sephora_sku
      └── ulta_sku
    ulta_only       — Ulta product with no Sephora match

Outputs  →  data/processed/matched/:
    matched_pairs.csv        — the fuzzy-matched product pairs (new file)
    graph_nodes.csv
    graph_edges.csv
    product_graph.gpickle
    graph_summary.txt
    cross_retailer_graph.png

Notebook usage:
    from src.build_product_graph import load_graph, visualize_graph, \\
                                        query_brand, get_cross_platform_pairs, \\
                                        get_ulta_only_products

    G   = load_graph()                                   # auto-builds if needed
    fig = visualize_graph(G, mode="brand_network")
    fig = visualize_graph(G, mode="category_heatmap")
    fig = visualize_graph(G, mode="price_delta")
    fig = visualize_graph(G, mode="coverage", top_n=25)
    fig.savefig("my_plot.png", dpi=150, bbox_inches="tight")
"""

import difflib
import pickle
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
MATCHED_DIR = BASE_DIR / "data" / "processed" / "matched"
OUT_DIR     = MATCHED_DIR
MATCHED_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_PATH = MATCHED_DIR / "matched_products.csv"   # long-format combined catalog

# ── Tunable matching threshold ─────────────────────────────────────────────────
# Pairs with similarity < MATCH_THRESHOLD are kept as ulta_only.
# 0.65 is a good balance: eliminates clearly wrong matches while allowing
# minor wording differences (e.g. "Nourishing Hair Mask" vs "Nourishing Mask").
MATCH_THRESHOLD = 0.65

# ── Colour palette (shared across all viz modes) ───────────────────────────────
COLORS = {
    "brand":           "#2196F3",
    "matched_product": "#FF9800",
    "sephora_sku":     "#E91E63",
    "ulta_sku":        "#9C27B0",
    "ulta_only":       "#CE93D8",
}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — NAME CLEANING
# ══════════════════════════════════════════════════════════════════════════════
def clean_product_name(name: str, brand: str) -> str:
    """
    Normalise a product name for fuzzy comparison.

    Sephora appends  ' - Brand Name'  to every product title.
    Ulta often appends  ' - 5.0 oz'  or similar size info.
    Both are stripped before comparison.
    """
    name = str(name).strip()

    # Strip Sephora's '- Brand Name' suffix (case-insensitive)
    escaped = re.escape(str(brand).strip())
    name = re.sub(r"\s*-\s*" + escaped + r"\s*$", "", name, flags=re.IGNORECASE)

    # Strip size / volume suffixes: '- 5.0 oz', '200ml', '8 fl oz', etc.
    name = re.sub(
        r"\s*-?\s*[\d.]+\s*(oz|fl\.?\s*oz|ml|g|lb|ct|count)\s*$",
        "", name, flags=re.IGNORECASE
    )

    # Collapse whitespace and lower-case
    return re.sub(r"\s+", " ", name).strip().lower()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — FUZZY MATCHING
# ══════════════════════════════════════════════════════════════════════════════
def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def match_products(
    sephora_df: pd.DataFrame,
    ulta_df: pd.DataFrame,
    threshold: float = MATCH_THRESHOLD,
) -> pd.DataFrame:
    """
    For every Sephora product, find the best-matching Ulta product within the
    same brand (brand_normalized).  Returns a DataFrame of matched pairs.

    Matching is intentionally conservative:
      - Only compares products within the same brand_normalized bucket
      - Uses SequenceMatcher on cleaned names
      - Requires similarity >= threshold (default 0.65)
      - Each Ulta product can only be matched once (greedy, score-ordered)
    """
    sephora_df = sephora_df.copy()
    ulta_df    = ulta_df.copy()

    sephora_df["_clean"] = sephora_df.apply(
        lambda r: clean_product_name(r["product_name"], r["brand"]), axis=1
    )
    ulta_df["_clean"] = ulta_df.apply(
        lambda r: clean_product_name(r["product_name"], r["brand"]), axis=1
    )

    shared_brands = (
        set(sephora_df["brand_normalized"].unique()) &
        set(ulta_df["brand_normalized"].unique())
    )

    candidates = []   # (score, sephora_idx, ulta_idx)
    for brand in shared_brands:
        s_sub = sephora_df[sephora_df["brand_normalized"] == brand]
        u_sub = ulta_df[ulta_df["brand_normalized"] == brand]
        u_names = u_sub["_clean"].tolist()
        u_idxs  = u_sub.index.tolist()

        for s_idx, s_row in s_sub.iterrows():
            best_score, best_u_idx = 0.0, None
            for u_idx, u_name in zip(u_idxs, u_names):
                sc = _similarity(s_row["_clean"], u_name)
                if sc > best_score:
                    best_score, best_u_idx = sc, u_idx
            if best_score >= threshold:
                candidates.append((best_score, s_idx, best_u_idx))

    # Greedy deduplication: highest-score matches get priority.
    # Each Ulta product may only appear in one pair.
    candidates.sort(key=lambda x: -x[0])
    used_ulta: set = set()
    pairs = []
    mid = 0
    for score, s_idx, u_idx in candidates:
        if u_idx in used_ulta:
            continue
        used_ulta.add(u_idx)
        s_row = sephora_df.loc[s_idx]
        u_row = ulta_df.loc[u_idx]
        pairs.append({
            "match_id":             mid,
            "similarity_score":     round(score, 4),
            "sephora_product_id":   s_row["product_id"],
            "sephora_product_name": s_row["product_name"],
            "sephora_clean_name":   s_row["_clean"],
            "sephora_brand":        s_row["brand"],
            "sephora_brand_normalized": s_row["brand_normalized"],
            "sephora_category":     s_row["category"],
            "sephora_price":        s_row["price"],
            "sephora_rating":       s_row["rating"],
            "sephora_product_url":  s_row["product_url"],
            "ulta_product_id":      u_row["product_id"],
            "ulta_product_name":    u_row["product_name"],
            "ulta_clean_name":      u_row["_clean"],
            "ulta_brand":           u_row["brand"],
            "ulta_brand_normalized":u_row["brand_normalized"],
            "ulta_category":        u_row["category"],
            "ulta_price":           u_row["price"],
            "ulta_rating":          u_row["rating"],
            "ulta_product_url":     u_row["product_url"],
        })
        mid += 1

    return pd.DataFrame(pairs)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — BUILD GRAPH
# ══════════════════════════════════════════════════════════════════════════════
def build_graph(threshold: float = MATCH_THRESHOLD) -> nx.DiGraph:
    print("=" * 64)
    print("  Cross-Retailer Product Graph Builder  (Ulta-primary)")
    print("=" * 64)

    # ── Load catalog ───────────────────────────────────────────
    print(f"\n[1/6] Loading {CATALOG_PATH.name} ...")
    catalog = pd.read_csv(CATALOG_PATH, low_memory=False)
    print(f"      {len(catalog):,} total rows | cols: {list(catalog.columns)}")

    sephora_df = catalog[catalog["retailer"].str.lower().str.startswith("seph")].copy()
    ulta_df    = catalog[catalog["retailer"].str.lower().str.startswith("ulta")].copy()
    print(f"      Sephora: {len(sephora_df):,} products")
    print(f"      Ulta:    {len(ulta_df):,} products  (primary catalog)")

    shared_brands = (
        set(sephora_df["brand_normalized"].unique()) &
        set(ulta_df["brand_normalized"].unique())
    )
    print(f"      Shared brands: {len(shared_brands):,}")

    # ── Fuzzy match ────────────────────────────────────────────
    print(f"\n[2/6] Fuzzy-matching products within shared brands "
          f"(threshold={threshold}) ...")
    pairs_df = match_products(sephora_df, ulta_df, threshold=threshold)

    matched_ulta_ids   = set(pairs_df["ulta_product_id"].astype(str).tolist())
    matched_sephora_ids = set(pairs_df["sephora_product_id"].astype(str).tolist())

    print(f"      Matched pairs:         {len(pairs_df):,}")
    print(f"      Sephora unmatched:     {len(sephora_df) - len(pairs_df):,}  "
          f"({(len(sephora_df)-len(pairs_df))/len(sephora_df)*100:.1f}%)")
    print(f"      Ulta-only (unmatched): "
          f"{len(ulta_df) - len(matched_ulta_ids):,}  "
          f"({(len(ulta_df)-len(matched_ulta_ids))/len(ulta_df)*100:.1f}%)")

    # Save pairs for transparency / downstream use
    pairs_path = OUT_DIR / "matched_pairs.csv"
    pairs_df.to_csv(pairs_path, index=False)
    print(f"      Saved → {pairs_path.name}")

    # ── Build NetworkX graph ───────────────────────────────────
    print("\n[3/6] Building NetworkX DiGraph ...")
    G = nx.DiGraph()
    node_records: list[dict] = []
    edge_records: list[dict] = []

    def _add_node(nid: str, attrs: dict):
        G.add_node(nid, **attrs)
        node_records.append({"node_id": nid, **attrs})

    def _add_edge(src: str, tgt: str, **attrs):
        G.add_edge(src, tgt, **attrs)
        edge_records.append({"source": src, "target": tgt, **attrs})

    def _ensure_brand(brand_norm: str, brand_display: str) -> str:
        bid = f"brand::{brand_norm}"
        if not G.has_node(bid):
            _add_node(bid, {
                "node_type":        "brand",
                "brand_name":       brand_display,
                "brand_normalized": brand_norm,
                "label":            brand_display,
            })
        return bid

    # ── Pass 1: matched pairs ──────────────────────────────────
    for _, row in pairs_df.iterrows():
        mid   = str(row["match_id"])
        brand_id = _ensure_brand(
            row["ulta_brand_normalized"], row["ulta_brand"]
        )

        # Price / rating deltas
        price_delta = rating_delta = None
        try:
            price_delta  = float(row["sephora_price"])  - float(row["ulta_price"])
        except (ValueError, TypeError):
            pass
        try:
            rating_delta = float(row["sephora_rating"]) - float(row["ulta_rating"])
        except (ValueError, TypeError):
            pass

        # matched_product node
        mp_id = f"matched::{mid}"
        _add_node(mp_id, {
            "node_type":        "matched_product",
            "match_id":         mid,
            "similarity_score": row["similarity_score"],
            "product_name":     row["ulta_product_name"],    # Ulta canonical
            "label":            row["ulta_product_name"],
            "brand":            row["ulta_brand"],
            "brand_normalized": row["ulta_brand_normalized"],
            "category":         row["ulta_category"],
            "sephora_price":    row["sephora_price"],
            "ulta_price":       row["ulta_price"],
            "price_delta":      price_delta,
            "sephora_rating":   row["sephora_rating"],
            "ulta_rating":      row["ulta_rating"],
            "rating_delta":     rating_delta,
        })

        # Sephora SKU node
        s_nid = f"sephora::{row['sephora_product_id']}"
        _add_node(s_nid, {
            "node_type":    "sephora_sku",
            "retailer":     "sephora",
            "product_id":   str(row["sephora_product_id"]),
            "product_name": row["sephora_product_name"],
            "label":        row["sephora_product_name"],
            "brand":        row["sephora_brand"],
            "category":     row["sephora_category"],
            "price":        row["sephora_price"],
            "rating":       row["sephora_rating"],
            "product_url":  row["sephora_product_url"],
        })

        # Ulta SKU node
        u_nid = f"ulta::{row['ulta_product_id']}"
        _add_node(u_nid, {
            "node_type":    "ulta_sku",
            "retailer":     "ulta",
            "product_id":   str(row["ulta_product_id"]),
            "product_name": row["ulta_product_name"],
            "label":        row["ulta_product_name"],
            "brand":        row["ulta_brand"],
            "category":     row["ulta_category"],
            "price":        row["ulta_price"],
            "rating":       row["ulta_rating"],
            "product_url":  row["ulta_product_url"],
        })

        # Edges
        _add_edge(brand_id, mp_id,  edge_type="has_product")
        _add_edge(mp_id,    s_nid,  edge_type="sephora_listing",
                  price=row["sephora_price"], rating=row["sephora_rating"])
        _add_edge(mp_id,    u_nid,  edge_type="ulta_listing",
                  price=row["ulta_price"],    rating=row["ulta_rating"])
        _add_edge(s_nid,    u_nid,  edge_type="cross_platform_match",
                  match_id=mid, similarity=row["similarity_score"],
                  price_delta=price_delta, rating_delta=rating_delta)
        _add_edge(u_nid,    s_nid,  edge_type="cross_platform_match",
                  match_id=mid, similarity=row["similarity_score"],
                  price_delta=price_delta, rating_delta=rating_delta)

    matched_count = sum(1 for _, d in G.nodes(data=True)
                        if d.get("node_type") == "matched_product")

    # ── Pass 2: Ulta-only products ─────────────────────────────
    ulta_only_df = ulta_df[
        ~ulta_df["product_id"].astype(str).isin(matched_ulta_ids)
    ].copy()

    for _, row in ulta_only_df.iterrows():
        uid      = str(row["product_id"])
        brand_id = _ensure_brand(
            str(row["brand_normalized"]), str(row["brand"])
        )
        u_only_id = f"ulta_only::{uid}"
        _add_node(u_only_id, {
            "node_type":        "ulta_only",
            "retailer":         "ulta",
            "product_id":       uid,
            "product_name":     row["product_name"],
            "label":            row["product_name"],
            "brand":            row["brand"],
            "brand_normalized": row["brand_normalized"],
            "category":         row["category"],
            "price":            row["price"],
            "rating":           row["rating"],
            "product_url":      row["product_url"],
            "sephora_match":    False,
        })
        _add_edge(brand_id, u_only_id, edge_type="has_product")

    ulta_only_count = sum(1 for _, d in G.nodes(data=True)
                          if d.get("node_type") == "ulta_only")

    print(f"      Pass 1 → {matched_count:,} matched_product nodes")
    print(f"      Pass 2 → {ulta_only_count:,} ulta_only nodes")
    print(f"      Total  → {G.number_of_nodes():,} nodes | "
          f"{G.number_of_edges():,} edges")

    # ── Export CSVs ────────────────────────────────────────────
    print("\n[4/6] Exporting graph CSVs ...")
    nodes_df = pd.DataFrame(node_records).drop_duplicates(subset=["node_id"])
    edges_df = pd.DataFrame(edge_records)
    nodes_df.to_csv(OUT_DIR / "graph_nodes.csv", index=False)
    edges_df.to_csv(OUT_DIR / "graph_edges.csv", index=False)
    print(f"      graph_nodes.csv → {len(nodes_df):,} rows")
    print(f"      graph_edges.csv → {len(edges_df):,} rows")

    with open(OUT_DIR / "product_graph.gpickle", "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("      product_graph.gpickle saved")

    # ── Summary ────────────────────────────────────────────────
    print("\n[5/6] Writing graph_summary.txt ...")
    _write_summary(G, len(ulta_df), matched_count, ulta_only_count, threshold)

    # ── Default viz ────────────────────────────────────────────
    print("\n[6/6] Saving default visualization ...")
    fig = visualize_graph(G, mode="brand_network", top_n=15, show=False)
    fig.savefig(OUT_DIR / "cross_retailer_graph.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      cross_retailer_graph.png saved")

    print("\n✓  All outputs written to:", OUT_DIR)
    for f in ["matched_pairs.csv", "graph_nodes.csv", "graph_edges.csv",
              "product_graph.gpickle", "graph_summary.txt",
              "cross_retailer_graph.png"]:
        print(f"   ├── {f}")

    return G


# ── Summary ────────────────────────────────────────────────────────────────────
def _write_summary(
    G: nx.DiGraph,
    ulta_total: int,
    matched_count: int,
    ulta_only_count: int,
    threshold: float,
):
    brand_nodes   = [n for n, d in G.nodes(data=True) if d.get("node_type") == "brand"]
    sephora_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "sephora_sku"]
    ulta_nodes    = [n for n, d in G.nodes(data=True) if d.get("node_type") == "ulta_sku"]

    match_pct = matched_count / ulta_total * 100 if ulta_total else 0.0

    price_deltas  = [d["price_delta"]  for _, d in G.nodes(data=True)
                     if d.get("node_type") == "matched_product"
                     and d.get("price_delta")  is not None]
    rating_deltas = [d["rating_delta"] for _, d in G.nodes(data=True)
                     if d.get("node_type") == "matched_product"
                     and d.get("rating_delta") is not None]

    brand_counts = {}
    for b in brand_nodes:
        m  = sum(1 for n in G.successors(b) if G.nodes[n].get("node_type") == "matched_product")
        uo = sum(1 for n in G.successors(b) if G.nodes[n].get("node_type") == "ulta_only")
        brand_counts[b] = (m + uo, m, uo)
    top = sorted(brand_counts.items(), key=lambda x: -x[1][0])[:15]

    lines = [
        "=" * 64,
        "  Cross-Retailer Product Graph — Summary",
        f"  Ulta-primary | fuzzy match threshold = {threshold}",
        "=" * 64, "",
        "Node counts:",
        f"  Brand nodes              : {len(brand_nodes):>6,}",
        f"  Matched product nodes    : {matched_count:>6,}",
        f"  Sephora SKU nodes        : {len(sephora_nodes):>6,}",
        f"  Ulta SKU nodes           : {len(ulta_nodes):>6,}",
        f"  Ulta-only nodes          : {ulta_only_count:>6,}",
        f"  Total nodes              : {G.number_of_nodes():>6,}", "",
        "Ulta catalog coverage:",
        f"  Total Ulta products      : {ulta_total:>6,}",
        f"  Matched to Sephora       : {matched_count:>6,}  ({match_pct:.1f}%)",
        f"  Ulta-only (unmatched)    : {ulta_only_count:>6,}  ({100-match_pct:.1f}%)", "",
        "Edge counts:",
    ]
    for etype in ["has_product", "sephora_listing", "ulta_listing",
                  "cross_platform_match"]:
        ct = sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") == etype)
        lines.append(f"  {etype:<30}: {ct:>6,}")
    lines += [f"  {'Total':<30}: {G.number_of_edges():>6,}", ""]

    if price_deltas:
        arr = np.array(price_deltas)
        lines += [
            "Price delta (Sephora − Ulta), matched products only:",
            f"  Mean          : ${arr.mean():+.2f}",
            f"  Median        : ${np.median(arr):+.2f}",
            f"  Sephora higher: {(arr > 0).sum():,}  ({(arr > 0).mean()*100:.1f}%)",
            f"  Ulta higher   : {(arr < 0).sum():,}  ({(arr < 0).mean()*100:.1f}%)",
            f"  Parity |Δ|≤$1 : {(np.abs(arr)<=1).sum():,}  "
            f"({(np.abs(arr)<=1).mean()*100:.1f}%)", "",
        ]
    if rating_deltas:
        arr = np.array(rating_deltas)
        lines += [
            "Rating delta (Sephora − Ulta), matched products only:",
            f"  Mean          : {arr.mean():+.3f} ★",
            f"  Median        : {np.median(arr):+.3f} ★",
            f"  Sephora higher: {(arr > 0).sum():,}",
            f"  Ulta higher   : {(arr < 0).sum():,}", "",
        ]

    lines.append("Top 15 brands by total product count:")
    lines.append(f"  {'Brand':<35} {'Total':>6}  {'Matched':>8}  {'Ulta-only':>10}")
    lines.append("  " + "-" * 62)
    for b, (total, m, uo) in top:
        bname = G.nodes[b].get("brand_name", b)
        lines.append(f"  {bname:<35} {total:>6,}  {m:>8,}  {uo:>10,}")

    text = "\n".join(lines)
    print(text)
    with open(OUT_DIR / "graph_summary.txt", "w") as f:
        f.write(text)


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATIONS  —  all return fig for notebook use
# ══════════════════════════════════════════════════════════════════════════════
def visualize_graph(
    G: nx.DiGraph,
    mode: str = "brand_network",
    top_n: int = 15,
    figsize: tuple = None,
    show: bool = True,
) -> plt.Figure:
    """
    Generate a visualization of the cross-retailer product graph.

    Parameters
    ----------
    G       : nx.DiGraph  — from build_graph() or load_graph()
    mode    : str
        "brand_network"    — spring-layout network of top-N brands
        "category_heatmap" — matched vs ulta-only count per category
        "price_delta"      — histogram + donut of Sephora−Ulta price gaps
        "coverage"         — horizontal bar of Ulta match rate per brand
    top_n   : int         — number of brands to show (brand_network, coverage)
    figsize : tuple|None  — override default figure size
    show    : bool        — call plt.show() (set False in scripts)

    Returns
    -------
    matplotlib.figure.Figure
        Always returned.  Notebook display:  display(fig)  or just  fig
        Save from notebook:  fig.savefig("out.png", dpi=150, bbox_inches="tight")
    """
    modes = ("brand_network", "category_heatmap", "price_delta", "coverage")
    if mode not in modes:
        raise ValueError(f"mode must be one of {modes}, got '{mode}'")

    defaults = {
        "brand_network":    (22, 15),
        "category_heatmap": (14, 8),
        "price_delta":      (14, 6),
        "coverage":         (14, 7),
    }
    fs = figsize or defaults[mode]
    dispatch = {
        "brand_network":    lambda: _viz_brand_network(G, top_n, fs),
        "category_heatmap": lambda: _viz_category_heatmap(G, fs),
        "price_delta":      lambda: _viz_price_delta(G, fs),
        "coverage":         lambda: _viz_coverage(G, top_n, fs),
    }
    fig = dispatch[mode]()
    if show:
        plt.show()
    return fig


def _viz_brand_network(G, top_n, figsize):
    brand_nodes  = [n for n, d in G.nodes(data=True) if d.get("node_type") == "brand"]
    brand_counts = {b: sum(1 for _ in G.successors(b)) for b in brand_nodes}
    top_brands   = [b for b, _ in
                    sorted(brand_counts.items(), key=lambda x: -x[1])[:top_n]]

    viz_nodes = set(top_brands)
    for b in top_brands:
        for child in G.successors(b):
            viz_nodes.add(child)
            if G.nodes[child].get("node_type") == "matched_product":
                for sku in G.successors(child):
                    viz_nodes.add(sku)

    H   = G.subgraph(viz_nodes).copy()
    pos = nx.spring_layout(H, k=1.8, seed=42)

    node_color = [COLORS.get(H.nodes[n].get("node_type", ""), "#AAA") for n in H.nodes()]
    node_size  = [900 if H.nodes[n].get("node_type") == "brand" else
                  350 if H.nodes[n].get("node_type") == "matched_product" else 80
                  for n in H.nodes()]

    fig, ax = plt.subplots(figsize=figsize)
    for etype, (style, alpha, color) in {
        "has_product":          ("solid",  0.55, "#888"),
        "sephora_listing":      ("solid",  0.40, COLORS["sephora_sku"]),
        "ulta_listing":         ("solid",  0.40, COLORS["ulta_sku"]),
        "cross_platform_match": ("dashed", 0.15, "#AAA"),
    }.items():
        elist = [(u, v) for u, v, d in H.edges(data=True)
                 if d.get("edge_type") == etype]
        if elist:
            nx.draw_networkx_edges(H, pos, edgelist=elist, style=style,
                                   alpha=alpha, edge_color=color,
                                   arrows=True, arrowsize=8, ax=ax)

    nx.draw_networkx_nodes(H, pos, node_color=node_color,
                           node_size=node_size, ax=ax)
    nx.draw_networkx_labels(
        H, pos,
        labels={n: H.nodes[n].get("brand_name", n)
                for n in H.nodes() if H.nodes[n].get("node_type") == "brand"},
        font_size=7, font_weight="bold", ax=ax,
    )

    ax.legend(handles=[
        mpatches.Patch(color=COLORS["brand"],           label="Brand"),
        mpatches.Patch(color=COLORS["matched_product"], label="Matched product"),
        mpatches.Patch(color=COLORS["sephora_sku"],     label="Sephora SKU"),
        mpatches.Patch(color=COLORS["ulta_sku"],        label="Ulta SKU"),
        mpatches.Patch(color=COLORS["ulta_only"],       label="Ulta-only"),
    ], loc="upper left", fontsize=9, framealpha=0.9)

    n_m  = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "matched_product")
    n_uo = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "ulta_only")
    ax.set_title(
        f"Cross-Retailer Product Graph — Top {top_n} Brands  (Ulta-primary)\n"
        f"{len(brand_nodes):,} brands  |  {n_m:,} matched  |  {n_uo:,} Ulta-only",
        fontsize=13, pad=14,
    )
    ax.axis("off")
    fig.tight_layout()
    return fig


def _viz_category_heatmap(G, figsize):
    rows = [
        {"category": d.get("category") or "Unknown",
         "status":   "Matched" if d["node_type"] == "matched_product" else "Ulta-only"}
        for _, d in G.nodes(data=True)
        if d.get("node_type") in ("matched_product", "ulta_only")
    ]
    if not rows:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    df    = pd.DataFrame(rows)
    pivot = df.groupby(["category", "status"]).size().unstack(fill_value=0)
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total").drop(columns="_total").tail(22)

    fig, ax = plt.subplots(figsize=figsize)
    pivot.plot(kind="barh", stacked=True, ax=ax,
               color=[COLORS["matched_product"], COLORS["ulta_only"]][:len(pivot.columns)])
    ax.set_xlabel("Number of Products", fontsize=11)
    ax.set_title("Product Coverage by Category\n(Matched vs. Ulta-Only)",
                 fontsize=13, pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _viz_price_delta(G, figsize):
    deltas = [d["price_delta"] for _, d in G.nodes(data=True)
              if d.get("node_type") == "matched_product"
              and d.get("price_delta") is not None]
    if not deltas:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No price delta data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    arr = np.array(deltas)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.hist(arr, bins=40, color=COLORS["matched_product"],
             edgecolor="white", alpha=0.85)
    ax1.axvline(0,              color="black",             lw=1.5, ls="--",
                label="Parity ($0)")
    ax1.axvline(arr.mean(),     color=COLORS["sephora_sku"], lw=1.5, ls=":",
                label=f"Mean {arr.mean():+.2f}")
    ax1.axvline(np.median(arr), color=COLORS["ulta_sku"],   lw=1.5, ls="-.",
                label=f"Median {np.median(arr):+.2f}")
    ax1.set_xlabel("Price Delta: Sephora − Ulta ($)", fontsize=11)
    ax1.set_ylabel("Products", fontsize=11)
    ax1.set_title("Price Gap Distribution", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    sizes  = [(arr > 1).sum(), (np.abs(arr) <= 1).sum(), (arr < -1).sum()]
    labels = ["Sephora higher", "Parity (|Δ|≤$1)", "Ulta higher"]
    _, _, autotexts = ax2.pie(
        sizes, labels=labels, autopct="%1.1f%%", startangle=90,
        colors=[COLORS["sephora_sku"], COLORS["matched_product"], COLORS["ulta_sku"]],
        wedgeprops=dict(width=0.55),
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax2.set_title("Price Advantage by Retailer", fontsize=12)

    fig.suptitle(f"Price Delta — {len(arr):,} Matched Products (Sephora − Ulta)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def _viz_coverage(G, top_n, figsize):
    brand_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "brand"]
    rows = []
    for b in brand_nodes:
        m  = sum(1 for n in G.successors(b)
                 if G.nodes[n].get("node_type") == "matched_product")
        uo = sum(1 for n in G.successors(b)
                 if G.nodes[n].get("node_type") == "ulta_only")
        tot = m + uo
        if tot > 0:
            rows.append({"brand": G.nodes[b].get("brand_name", b),
                         "matched": m, "ulta_only": uo, "total": tot,
                         "match_rate": m / tot})
    if not rows:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    df = (pd.DataFrame(rows)
          .sort_values("total", ascending=False)
          .head(top_n)
          .sort_values("match_rate"))

    fig, ax = plt.subplots(figsize=figsize)
    y = range(len(df))
    ax.barh(y, df["total"],   color=COLORS["ulta_only"],       alpha=0.85,
            label="Ulta-only")
    ax.barh(y, df["matched"], color=COLORS["matched_product"], alpha=0.95,
            label="Matched")
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["brand"], fontsize=9)
    ax.set_xlabel("Number of Products", fontsize=11)
    ax.set_title(f"Ulta Catalog Match Rate — Top {top_n} Brands\n"
                 "(Ulta-primary: orange = matched, purple = Ulta-only)",
                 fontsize=12, pad=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    for i, row in enumerate(df.itertuples()):
        ax.text(row.total * 1.01, i, f"{row.match_rate*100:.0f}%",
                va="center", fontsize=8, color="#444")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  I/O HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_graph(auto_build: bool = True) -> nx.DiGraph:
    """
    Load the saved product graph.  Runs build_graph() automatically if the
    gpickle doesn't exist yet (auto_build=True, default).
    """
    gpickle_path = OUT_DIR / "product_graph.gpickle"
    if not gpickle_path.exists():
        if auto_build:
            print("product_graph.gpickle not found — running build_graph() ...\n")
            return build_graph()
        raise FileNotFoundError(
            f"Graph not built yet. Run build_graph() first.\n"
            f"Expected: {gpickle_path}"
        )
    with open(gpickle_path, "rb") as f:
        return pickle.load(f)


def query_brand(G: nx.DiGraph, brand_name: str) -> pd.DataFrame:
    """All products (matched + ulta_only) for a given brand."""
    bid = f"brand::{brand_name.lower()}"
    if bid not in G:
        candidates = [n for n in G if n.startswith("brand::") and
                      brand_name.lower() in n.lower()]
        if not candidates:
            print(f"Brand '{brand_name}' not found in graph.")
            return pd.DataFrame()
        bid = candidates[0]
    return pd.DataFrame([dict(G.nodes[n]) for n in G.successors(bid)])


def get_cross_platform_pairs(G: nx.DiGraph) -> pd.DataFrame:
    """DataFrame of all matched Sephora↔Ulta pairs with price/rating deltas."""
    rows = []
    for u, v, d in G.edges(data=True):
        if d.get("edge_type") == "cross_platform_match" and \
                u.startswith("sephora::"):
            sa, ua = G.nodes[u], G.nodes[v]
            rows.append({
                "sephora_node":         u,
                "ulta_node":            v,
                "sephora_product_name": sa.get("product_name"),
                "ulta_product_name":    ua.get("product_name"),
                "sephora_price":        sa.get("price"),
                "ulta_price":           ua.get("price"),
                "price_delta":          d.get("price_delta"),
                "sephora_rating":       sa.get("rating"),
                "ulta_rating":          ua.get("rating"),
                "rating_delta":         d.get("rating_delta"),
                "similarity":           d.get("similarity"),
            })
    return pd.DataFrame(rows)


def get_ulta_only_products(G: nx.DiGraph) -> pd.DataFrame:
    """All Ulta products with no Sephora match."""
    return pd.DataFrame([
        dict(G.nodes[n]) for n, d in G.nodes(data=True)
        if d.get("node_type") == "ulta_only"
    ])


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    G = build_graph()