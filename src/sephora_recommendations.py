# sephora_recommendations.py
# 1.3 — Sephora Item-to-Item Marketing Recommendation System
#
# Three recommendation intents:
#   1. Close Substitutes    — same need, similar perception
#   2. Complementary        — different need, shared audience
#   3. Trade-Up / Trade-Down — same experience, different price tier
#
# Consumes:
#   1.1 → data/processed/sephora_segmentation.csv
#   1.2 → brand health & sentiment (embedded in segmentation)
#
# Outputs:
#   data/processed/Sephora/sephora_recommendations.csv
#
# Usage:
#   As script (runs full pipeline):
#     python src/sephora_recommendations.py
#
#   As import (dashboard only, loads from pre-built CSVs):
#     from src.sephora_recommendations import show_dashboard         # Jupyter notebook (interactive dropdown)
#     from src.sephora_recommendations import write_product_html     # browser — single product
#     from src.sephora_recommendations import write_all_product_htmls # browser — all products + index page

import pandas as pd
import numpy as np
import re
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed" / "Sephora"
OUTPUT_DIR = _PROJECT_ROOT / "notebooks" / "independent" / "outputs"
PRODUCT_DASHBOARD_DIR = OUTPUT_DIR / "product_dashboards"
MIN_REVIEWS = 20
TOP_N_SUB = 5
TOP_N_COMP = 5
TOP_N_TRADE = 3

# Module-level cache (populated by run_pipeline() or _load_data())
_data = {}


# DATA LOADING (lazy, for dashboard)

def _load_data():
    """Load pre-built CSV for dashboard use. Called lazily on first dashboard call."""
    if _data:
        return

    t0 = time.time()
    print("Loading Sephora recommendation data from CSVs...")

    _data["seg"] = pd.read_csv(PROCESSED_DIR / "sephora_segmentation.csv")
    _data["seg"]["price"] = pd.to_numeric(_data["seg"]["price"], errors="coerce")
    _data["seg"]["review_count"] = pd.to_numeric(
        _data["seg"].get("review_count", 0), errors="coerce"
    ).fillna(0)

    _data["recs"] = pd.read_csv(PROCESSED_DIR / "sephora_recommendations.csv")

    name_col = "product_name" if "product_name" in _data["seg"].columns else "name"
    _data["name_col"] = name_col

    # Build product list for dropdown
    seg = _data["seg"]
    products = []
    for _, row in seg.iterrows():
        label = (f"{row.get('brand', '')} — {row.get(name_col, '')} "
                 f"(${row.get('price', 0):.0f})")
        products.append((label, str(row["product_id"])))
    products.sort(key=lambda x: x[0])
    _data["product_list"] = products

    elapsed = time.time() - t0
    print(f"  Loaded {len(products):,} products in {elapsed:.1f}s")


def _slug(text):
    """Convert a string to a safe filename slug."""
    return re.sub(r"[^\w\-]", "_", str(text)).strip("_")


# FEATURE BLOCKS & SIMILARITY (for run_pipeline)

def _build_feature_blocks(df):
    blocks = {}

    cat_dummies = pd.get_dummies(df["category"], prefix="cat").astype(float)
    skin_cols = [c for c in df.columns if c.startswith("pct_skin_tone_")
                 or c.startswith("pct_skin_type_")
                 or c in ["pct_has_skin_tone", "pct_has_skin_type"]]
    struct_extra = df[skin_cols].fillna(0) if skin_cols else pd.DataFrame(index=df.index)
    structural = pd.concat([cat_dummies, struct_extra], axis=1).fillna(0)
    blocks["structural"] = structural.values
    print(f"    A. Structural:  {structural.shape[1]} features")

    sentiment_cols = [
        "avg_sentiment", "sentiment_std", "pct_positive", "pct_negative", "pct_neutral",
        "avg_rating", "rating_std",
        "pct_5_star", "pct_4_star", "pct_3_star", "pct_2_star", "pct_1_star",
        "rating_dispersion_over_time"
    ]
    sentiment_cols = [c for c in sentiment_cols if c in df.columns]
    if "avg_rating" in df.columns and "avg_sentiment" in df.columns:
        df = df.copy()
        df["rating_sentiment_mismatch"] = abs(
            (df["avg_rating"] - 1) / 4 - (df["avg_sentiment"] + 1) / 2
        )
        sentiment_cols.append("rating_sentiment_mismatch")
    blocks["sentiment"] = df[sentiment_cols].fillna(0).values
    print(f"    B. Sentiment:   {len(sentiment_cols)} features")

    topic_cols = [c for c in df.columns if c == "top_topic_prevalence"]
    topic_dummies = pd.DataFrame(index=df.index)
    if "dominant_topic_mode" in df.columns:
        topic_dummies = pd.get_dummies(
            df["dominant_topic_mode"].astype(int), prefix="dom_topic"
        ).astype(float)
    topic_extra = df[topic_cols].fillna(0) if topic_cols else pd.DataFrame(index=df.index)
    content = pd.concat([topic_dummies, topic_extra], axis=1).fillna(0)
    blocks["content"] = content.values
    print(f"    C. Content:     {content.shape[1]} features")

    price_features = pd.DataFrame(index=df.index)
    if "price" in df.columns:
        price_features["log_price"] = np.log1p(df["price"].fillna(0))
        price_features["price_pctile"] = df["price"].rank(pct=True)
    if "price_vs_category_median" in df.columns:
        price_features["price_vs_cat"] = df["price_vs_category_median"].fillna(1.0)
    blocks["price"] = price_features.fillna(0).values
    print(f"    D. Price:       {price_features.shape[1]} features")

    return blocks, df


def _compute_similarities(blocks):
    sims = {}
    scaler = StandardScaler()
    for name, matrix in blocks.items():
        if matrix.shape[1] == 0:
            continue
        scaled = scaler.fit_transform(np.nan_to_num(matrix, 0))
        sim = cosine_similarity(scaled)
        np.fill_diagonal(sim, 0)
        sims[name] = sim
        print(f"    {name:<15} → {sim.shape[0]}×{sim.shape[1]}")
    return sims


def _compute_scores(sims, df):
    n = len(df)
    content = sims.get("content", np.zeros((n, n)))
    sentiment = sims.get("sentiment", np.zeros((n, n)))
    price_sim = sims.get("price", np.zeros((n, n)))
    structural = sims.get("structural", np.zeros((n, n)))

    cat_arr = np.array(df["category"].values)
    same_cat = (cat_arr[:, None] == cat_arr[None, :]).astype(float)
    cross_cat = 1.0 - same_cat * 0.5

    rc = df["review_count"].fillna(0).values.astype(float)
    stability = np.sqrt(rc) / (np.sqrt(rc).max() + 1e-8)
    penalty = 1.0 - (rc < MIN_REVIEWS).astype(float) * 0.7
    tw = stability * penalty

    sub = (0.40 * content + 0.30 * sentiment + 0.15 * price_sim + 0.15 * same_cat)
    sub *= (same_cat * 0.5 + 0.5)
    sub *= tw[np.newaxis, :]

    comp = (0.35 * content + 0.30 * sentiment + 0.20 * structural + 0.15 * cross_cat)
    comp *= cross_cat
    comp *= tw[np.newaxis, :]

    base = (0.60 * content + 0.40 * sentiment) * (same_cat * 0.7 + 0.3) * tw[np.newaxis, :]
    prices = df["price"].fillna(0).values.astype(float)
    trade_up = base * (prices[np.newaxis, :] > prices[:, np.newaxis] * 1.15).astype(float)
    trade_dn = base * (prices[np.newaxis, :] < prices[:, np.newaxis] * 0.85).astype(float)

    return sub, comp, trade_up, trade_dn


def _generate_csv(df, sub, comp, trade_up, trade_dn):
    n = len(df)
    name_col = "product_name" if "product_name" in df.columns else "name"
    pids = df["product_id"].astype(str).values
    names = df[name_col].fillna("").values if name_col in df.columns else [""] * n
    brands = df["brand"].fillna("").values if "brand" in df.columns else [""] * n
    cats = df["category"].fillna("").values
    prices = df["price"].fillna(0).values.astype(float)
    ratings = df["avg_rating"].fillna(0).values.astype(float)
    sents = df["avg_sentiment"].fillna(0).values.astype(float)
    rcounts = df["review_count"].fillna(0).values.astype(int)

    rows = []
    for i in range(n):
        row = {
            "product_id": pids[i], "product_name": names[i], "brand": brands[i],
            "category": cats[i], "price": prices[i], "avg_rating": ratings[i],
            "avg_sentiment": sents[i], "review_count": int(rcounts[i]),
        }
        for prefix, matrix, top_n in [("sub", sub, TOP_N_SUB), ("comp", comp, TOP_N_COMP),
                                       ("tradeup", trade_up, TOP_N_TRADE),
                                       ("tradedn", trade_dn, TOP_N_TRADE)]:
            top_j = np.argsort(matrix[i])[::-1][:top_n]
            for rank, j in enumerate(top_j):
                row[f"{prefix}_{rank+1}_id"] = pids[j]
                row[f"{prefix}_{rank+1}_name"] = names[j]
                row[f"{prefix}_{rank+1}_brand"] = brands[j]
                row[f"{prefix}_{rank+1}_score"] = round(float(matrix[i, j]), 4)
                row[f"{prefix}_{rank+1}_price"] = prices[j]
                row[f"{prefix}_{rank+1}_sentiment"] = round(sents[j], 3)
                row[f"{prefix}_{rank+1}_category"] = cats[j]
                row[f"{prefix}_{rank+1}_rating"] = round(ratings[j], 2)
                row[f"{prefix}_{rank+1}_reviews"] = int(rcounts[j])
        rows.append(row)
    return pd.DataFrame(rows)


# PIPELINE

def run_pipeline():
    """Run full computation pipeline. Saves CSV. Then use show_dashboard() or write_all_product_htmls()."""
    print("█" * 70)
    print("  SEPHORA — RECOMMENDATION SYSTEM (1.3)")
    print("█" * 70)

    print("\n  Loading segmentation...")
    df = pd.read_csv(PROCESSED_DIR / "sephora_segmentation.csv")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["review_count"] = pd.to_numeric(df.get("review_count", 0), errors="coerce").fillna(0)
    print(f"  {len(df):,} products × {len(df.columns)} features")

    print("\n  Building feature blocks...")
    blocks, df = _build_feature_blocks(df)

    print("\n  Computing similarities...")
    sims = _compute_similarities(blocks)

    print("\n  Computing scores...")
    sub, comp, trade_up, trade_dn = _compute_scores(sims, df)

    print("\n  Generating CSV...")
    recs_df = _generate_csv(df, sub, comp, trade_up, trade_dn)

    # Validate
    print(f"\n  {'─'*60}")
    print(f"  VALIDATION")
    id_to_cat = dict(zip(df["product_id"].astype(str), df["category"]))
    total, same = 0, 0
    for _, r in recs_df.iterrows():
        src = id_to_cat.get(str(r["product_id"]), "")
        for rk in range(1, TOP_N_SUB + 1):
            sid = r.get(f"sub_{rk}_id")
            if pd.notna(sid):
                total += 1
                if id_to_cat.get(str(sid), "") == src:
                    same += 1
    if total:
        print(f"  Substitute same-category: {same/total*100:.1f}%")

    up_ok, dn_ok, up_n, dn_n = 0, 0, 0, 0
    for _, r in recs_df.iterrows():
        sp = r.get("price", 0)
        if pd.isna(sp) or sp == 0:
            continue
        for rk in range(1, TOP_N_TRADE + 1):
            u = r.get(f"tradeup_{rk}_price", np.nan)
            if pd.notna(u):
                up_n += 1; up_ok += int(u > sp)
            d = r.get(f"tradedn_{rk}_price", np.nan)
            if pd.notna(d):
                dn_n += 1; dn_ok += int(d < sp)
    if up_n:
        print(f"  Trade-up price correct:   {up_ok/up_n*100:.1f}%")
    if dn_n:
        print(f"  Trade-down price correct:  {dn_ok/dn_n*100:.1f}%")

    csv_path = PROCESSED_DIR / "sephora_recommendations.csv"
    recs_df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Saved: {csv_path}  ({recs_df.shape})")
    print(f"\n  Notebook (interactive dropdown):")
    print(f"    from src.sephora_recommendations import show_dashboard")
    print(f"    show_dashboard()")
    print(f"\n  Browser (per-product HTML files):")
    print(f"    from src.sephora_recommendations import write_all_product_htmls")
    print(f"    write_all_product_htmls()")
    print(f"\n{'█'*70}\n")


# DASHBOARD: build_product_dashboard() → plotly figure

def build_product_dashboard(product_id):
    """
    Build a multi-panel product recommendation dashboard for a single product.
    Returns a Plotly Figure — call .show() in a notebook or
    .write_html(path) to save as a browser-accessible file.
    Loads from CSVs if data not already in memory.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _load_data()
    recs = _data["recs"]
    name_col = _data["name_col"]

    match = recs[recs["product_id"].astype(str) == str(product_id)]
    if len(match) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Product not found", showarrow=False, font=dict(size=20))
        return fig

    row = match.iloc[0]

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Close Substitutes — Price",
            "Close Substitutes — Sentiment",
            "Complements — Price",
            "Complements — Sentiment",
            "Trade-Up — Price",
            "Trade-Down — Price",
            "Avg Score by Intent", "",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.20,
        row_heights=[0.30, 0.30, 0.20, 0.20],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "table"}],
        ],
    )

    src_price = float(row.get("price", 0) or 0)
    src_sent = float(row.get("avg_sentiment", 0) or 0)

    def _extract(prefix, top_n):
        items = []
        for rk in range(1, top_n + 1):
            n = row.get(f"{prefix}_{rk}_name", "")
            if pd.isna(n) or n == "":
                continue
            items.append(dict(
                name=str(n), brand=str(row.get(f"{prefix}_{rk}_brand", "")),
                price=float(row.get(f"{prefix}_{rk}_price", 0) or 0),
                sent=float(row.get(f"{prefix}_{rk}_sentiment", 0) or 0),
                score=float(row.get(f"{prefix}_{rk}_score", 0) or 0),
                cat=str(row.get(f"{prefix}_{rk}_category", "")),
                rating=float(row.get(f"{prefix}_{rk}_rating", 0) or 0),
                reviews=int(row.get(f"{prefix}_{rk}_reviews", 0) or 0),
            ))
        short = [f"#{i+1} {it['brand'][:18]}" for i, it in enumerate(items)]
        hover = [
            f"<b>{it['brand']}</b><br>{it['name']}<br>"
            f"Category: {it['cat']}<br>"
            f"${it['price']:.0f} · ★{it['rating']:.2f} · Sent {it['sent']:.3f}<br>"
            f"{it['reviews']:,} reviews · Score {it['score']:.4f}"
            for it in items
        ]
        return items, short, hover

    # Substitutes (row 1)
    items, short, hover = _extract("sub", TOP_N_SUB)
    if items:
        fig.add_trace(go.Bar(
            y=short[::-1], x=[it["price"] for it in items][::-1], orientation="h",
            marker_color="#2d6a4f", showlegend=False,
            text=[f"${it['price']:.0f}" for it in items][::-1],
            textposition="inside", textfont=dict(color="white", size=11),
            hovertext=hover[::-1], hoverinfo="text",
        ), row=1, col=1)
        fig.add_shape(type="line", x0=src_price, x1=src_price, y0=-0.5, y1=len(items)-0.5,
                      line=dict(color="red", dash="dash", width=2), row=1, col=1)

        colors_s = ["#27ae60" if it["sent"] >= src_sent else "#e74c3c" for it in items]
        fig.add_trace(go.Bar(
            y=short[::-1], x=[it["sent"] for it in items][::-1], orientation="h",
            marker_color=colors_s[::-1], showlegend=False,
            text=[f"{it['sent']:.3f}" for it in items][::-1],
            textposition="inside", textfont=dict(color="white", size=11),
            hovertext=hover[::-1], hoverinfo="text",
        ), row=1, col=2)
        fig.add_shape(type="line", x0=src_sent, x1=src_sent, y0=-0.5, y1=len(items)-0.5,
                      line=dict(color="red", dash="dash", width=2), row=1, col=2)

    # Complements (row 2)
    items, short, hover = _extract("comp", TOP_N_COMP)
    if items:
        short_c = [f"#{i+1} {it['cat'][:20]}" for i, it in enumerate(items)]
        fig.add_trace(go.Bar(
            y=short_c[::-1], x=[it["price"] for it in items][::-1], orientation="h",
            marker_color="#7b2d8b", showlegend=False,
            text=[f"${it['price']:.0f}" for it in items][::-1],
            textposition="inside", textfont=dict(color="white", size=11),
            hovertext=hover[::-1], hoverinfo="text",
        ), row=2, col=1)

        fig.add_trace(go.Bar(
            y=short_c[::-1], x=[it["sent"] for it in items][::-1], orientation="h",
            marker_color="#7b2d8b", showlegend=False, opacity=0.7,
            text=[f"{it['sent']:.3f}" for it in items][::-1],
            textposition="inside", textfont=dict(color="white", size=11),
            hovertext=hover[::-1], hoverinfo="text",
        ), row=2, col=2)

    # Trade-Up (row 3, col 1)
    items, short, hover = _extract("tradeup", TOP_N_TRADE)
    if items:
        fig.add_trace(go.Bar(
            y=short[::-1], x=[it["price"] for it in items][::-1], orientation="h",
            marker_color="#1a73e8", showlegend=False,
            text=[f"${it['price']:.0f} (+${it['price']-src_price:.0f})" for it in items][::-1],
            textposition="inside", textfont=dict(color="white", size=11),
            hovertext=hover[::-1], hoverinfo="text",
        ), row=3, col=1)
        fig.add_shape(type="line", x0=src_price, x1=src_price, y0=-0.5, y1=len(items)-0.5,
                      line=dict(color="red", dash="dash", width=2), row=3, col=1)

    # Trade-Down (row 3, col 2)
    items, short, hover = _extract("tradedn", TOP_N_TRADE)
    if items:
        fig.add_trace(go.Bar(
            y=short[::-1], x=[it["price"] for it in items][::-1], orientation="h",
            marker_color="#e67700", showlegend=False,
            text=[f"${it['price']:.0f} (-${src_price-it['price']:.0f})" for it in items][::-1],
            textposition="inside", textfont=dict(color="white", size=11),
            hovertext=hover[::-1], hoverinfo="text",
        ), row=3, col=2)
        fig.add_shape(type="line", x0=src_price, x1=src_price, y0=-0.5, y1=len(items)-0.5,
                      line=dict(color="red", dash="dash", width=2), row=3, col=2)

    # Score summary + table (row 4)
    intent_names, intent_scores = [], []
    for prefix, label, top_n in [("sub", "Substitutes", TOP_N_SUB),
                                  ("comp", "Complements", TOP_N_COMP),
                                  ("tradeup", "Trade-Up", TOP_N_TRADE),
                                  ("tradedn", "Trade-Down", TOP_N_TRADE)]:
        sc = [float(row.get(f"{prefix}_{rk}_score", 0) or 0) for rk in range(1, top_n + 1)]
        intent_names.append(label)
        intent_scores.append(np.mean(sc) if sc else 0)

    fig.add_trace(go.Bar(
        x=intent_names, y=intent_scores,
        marker_color=["#2d6a4f", "#7b2d8b", "#1a73e8", "#e67700"],
        text=[f"{s:.4f}" for s in intent_scores], textposition="auto",
        showlegend=False,
    ), row=4, col=1)

    # Full detail table
    t_intent, t_brand, t_name, t_cat, t_price, t_score = [], [], [], [], [], []
    for prefix, ilabel, top_n in [("sub", "Substitute", TOP_N_SUB),
                                   ("comp", "Complement", TOP_N_COMP),
                                   ("tradeup", "Trade-Up", TOP_N_TRADE),
                                   ("tradedn", "Trade-Down", TOP_N_TRADE)]:
        for rk in range(1, top_n + 1):
            n = row.get(f"{prefix}_{rk}_name", "")
            if pd.isna(n) or n == "":
                continue
            t_intent.append(ilabel)
            t_brand.append(str(row.get(f"{prefix}_{rk}_brand", ""))[:20])
            t_name.append(str(n)[:28])
            t_cat.append(str(row.get(f"{prefix}_{rk}_category", ""))[:18])
            t_price.append(f"${float(row.get(f'{prefix}_{rk}_price', 0) or 0):.0f}")
            t_score.append(f"{float(row.get(f'{prefix}_{rk}_score', 0) or 0):.4f}")

    ic = {"Substitute": "#e8f5e9", "Complement": "#f3e5f5",
          "Trade-Up": "#e3f2fd", "Trade-Down": "#fff3e0"}
    fc = [ic.get(i, "#fff") for i in t_intent]

    fig.add_trace(go.Table(
        header=dict(
            values=["Type", "Brand", "Product", "Category", "Price", "Score"],
            font=dict(size=10, color="white"), fill_color="#1a1a1a", align="left",
        ),
        cells=dict(
            values=[t_intent, t_brand, t_name, t_cat, t_price, t_score],
            font=dict(size=9), fill_color=[fc] * 6, align="left",
        ),
    ), row=4, col=2)

    # Layout
    product_name = row.get("product_name", "")
    brand = row.get("brand", "")
    category = row.get("category", "")
    rating = float(row.get("avg_rating", 0) or 0)
    sentiment = float(row.get("avg_sentiment", 0) or 0)
    review_count = int(row.get("review_count", 0) or 0)

    kpi = (
        f"<b style='font-size:16px'>{brand} — {product_name}</b><br>"
        f"<span style='font-size:11px; color:#555;'>"
        f"Category: {category} &nbsp;|&nbsp; "
        f"Price: ${src_price:.0f} &nbsp;|&nbsp; "
        f"Rating: {rating:.2f} &nbsp;|&nbsp; "
        f"Sentiment: {sentiment:.3f} &nbsp;|&nbsp; "
        f"Reviews: {review_count:,}</span><br>"
        f"<span style='font-size:10px; color:#999;'>"
        f"Red dashed = source product &nbsp;|&nbsp; Hover bars for full details</span>"
    )

    fig.update_layout(
        height=1800, width=1300,
        title=dict(text=kpi, font=dict(size=12), x=0.01, y=0.99),
        template="plotly_white",
        showlegend=False,
        margin=dict(t=100, l=160, r=40, b=30),
    )
    fig.update_yaxes(tickfont=dict(size=11))

    return fig


# BROWSER EXPORT FUNCTIONS

def write_product_html(product_id, output_dir=None):
    """
    Save a single product's recommendation dashboard as a self-contained HTML
    file that can be opened in any browser without a running Jupyter server.

    Args:
        product_id:   Product ID string (must exist in sephora_recommendations.csv).
        output_dir:   Path to save the file. Defaults to
                      notebooks/independent/outputs/product_dashboards/.

    Returns:
        Path object pointing to the written file.

    Usage:
        from src.sephora_recommendations import write_product_html
        write_product_html("P12345678")
        write_product_html("P12345678", output_dir="my_reports/")
    """
    _load_data()
    out = Path(output_dir) if output_dir else PRODUCT_DASHBOARD_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Look up brand and name for a readable filename
    recs = _data["recs"]
    match = recs[recs["product_id"].astype(str) == str(product_id)]
    if len(match) > 0:
        r = match.iloc[0]
        brand = _slug(str(r.get("brand", "")))
        name = _slug(str(r.get("product_name", product_id))[:40])
        filename = f"{brand}_{name}_{_slug(str(product_id))}.html"
    else:
        filename = f"{_slug(str(product_id))}.html"

    fig = build_product_dashboard(product_id)
    path = out / filename
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"  -> {path}")
    return path


def write_all_product_htmls(output_dir=None):
    """
    Save a self-contained HTML dashboard for every product in the catalog,
    then generate a linked index page (sephora_product_dashboards_index.html) listing
    all products with their key metrics. All files can be opened in any browser
    without a running Jupyter server.

    Note: the Sephora catalog is large. This function may take several minutes
    to complete. For a targeted subset, call write_product_html() directly for
    each product ID of interest.

    Args:
        output_dir:   Directory to save product HTML files. Defaults to
                      notebooks/independent/outputs/product_dashboards/.
                      The index page is always written one level up
                      (notebooks/independent/outputs/sephora_product_dashboards_index.html).

    Usage:
        from src.sephora_recommendations import write_all_product_htmls
        write_all_product_htmls()
        write_all_product_htmls(output_dir="my_reports/products/")
    """
    _load_data()
    recs = _data["recs"]
    out = Path(output_dir) if output_dir else PRODUCT_DASHBOARD_DIR
    out.mkdir(parents=True, exist_ok=True)

    total = len(recs)
    print(f"Writing {total:,} product dashboards to {out}/")
    print("This may take several minutes for the full Sephora catalog.")

    written = []
    for i, (_, rec_row) in enumerate(recs.iterrows(), 1):
        pid = str(rec_row["product_id"])
        try:
            path = write_product_html(pid, output_dir=out)
            written.append((rec_row, path.name))
        except Exception as e:
            print(f"  WARNING: skipped {pid!r} — {e}")
        if i % 100 == 0:
            print(f"  {i:,}/{total:,} complete...")

    # Build index page
    index_path = out.parent / "sephora_product_dashboards_index.html"

    rows_html = ""
    for rec_row, filename in written:
        brand = str(rec_row.get("brand", ""))
        name = str(rec_row.get("product_name", ""))
        category = str(rec_row.get("category", ""))
        try:
            price = f"${float(rec_row.get('price', 0) or 0):.0f}"
            rating = f"{float(rec_row.get('avg_rating', 0) or 0):.2f}"
            sentiment = f"{float(rec_row.get('avg_sentiment', 0) or 0):.3f}"
            reviews = f"{int(rec_row.get('review_count', 0) or 0):,}"
        except Exception:
            price = rating = sentiment = reviews = "—"

        rows_html += (
            f"<tr>"
            f"<td><a href='product_dashboards/{filename}'>{brand}</a></td>"
            f"<td>{name}</td>"
            f"<td>{category}</td>"
            f"<td>{price}</td>"
            f"<td>{rating}</td>"
            f"<td>{sentiment}</td>"
            f"<td>{reviews}</td>"
            f"</tr>\n"
        )

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sephora Recommendations — Product Index</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1300px; margin: 40px auto; padding: 0 20px; color: #2C3E50; }}
    h1 {{ color: #2C3E50; border-bottom: 2px solid #2d6a4f; padding-bottom: 10px; }}
    p.subtitle {{ color: #7f8c8d; margin-top: -8px; }}
    input {{ width: 100%; padding: 8px; margin-bottom: 16px; box-sizing: border-box;
             border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ background: #2C3E50; color: white; padding: 10px; text-align: left; position: sticky; top: 0; }}
    td {{ padding: 8px 10px; border-bottom: 1px solid #ecf0f1; }}
    tr:hover td {{ background: #f0f4f8; }}
    a {{ color: #2d6a4f; text-decoration: none; font-weight: bold; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>Sephora Recommendations — Product Index</h1>
  <p class="subtitle">
    {len(written):,} products &nbsp;|&nbsp;
    Click any brand name to open its full recommendation dashboard
  </p>
  <input type="text" id="search" placeholder="Filter by brand, product, or category..." onkeyup="filterTable()">
  <table id="productTable">
    <thead>
      <tr>
        <th>Brand</th>
        <th>Product</th>
        <th>Category</th>
        <th>Price</th>
        <th>Avg Rating</th>
        <th>Avg Sentiment</th>
        <th>Reviews</th>
      </tr>
    </thead>
    <tbody>
{rows_html}    </tbody>
  </table>
  <script>
    function filterTable() {{
      const q = document.getElementById("search").value.toLowerCase();
      document.querySelectorAll("#productTable tbody tr").forEach(row => {{
        const text = Array.from(row.cells).slice(0, 3).map(c => c.textContent).join(" ").toLowerCase();
        row.style.display = text.includes(q) ? "" : "none";
      }});
    }}
  </script>
</body>
</html>"""

    index_path.write_text(index_html, encoding="utf-8")
    print(f"\nIndex page written: {index_path}")
    print(f"Done. {len(written):,}/{total:,} product dashboards saved.")
    return index_path


# DASHBOARD: show_dashboard() — ipywidgets dropdown

def show_dashboard():
    """
    Launch interactive product recommendation dashboard in a Jupyter notebook.
    Dropdown toggles between ALL Sephora products.
    Requires a running Jupyter server with ipywidgets enabled.

    For browser access without a notebook, use write_all_product_htmls() instead.

    Usage:
        from src.sephora_recommendations import show_dashboard
        show_dashboard()
    """
    import ipywidgets as widgets
    from IPython.display import display, HTML

    _load_data()
    products = _data["product_list"]

    display(HTML(
        "<h2 style='text-align:center; color:#1a1a1a;'>"
        "Sephora — Product Recommendation Dashboard</h2>"
        "<p style='text-align:center; color:#7f8c8d;'>"
        f"Select from {len(products):,} products. "
        "Dashboard rebuilds on each selection.</p>"
    ))

    dropdown = widgets.Dropdown(
        options=products,
        value=products[0][1],
        description="Product:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="600px"),
    )

    output = widgets.Output()

    def on_change(change):
        with output:
            output.clear_output(wait=True)
            build_product_dashboard(change["new"]).show()

    dropdown.observe(on_change, names="value")
    display(dropdown, output)

    with output:
        build_product_dashboard(products[0][1]).show()


if __name__ == "__main__":
    run_pipeline()