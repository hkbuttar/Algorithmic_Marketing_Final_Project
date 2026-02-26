# src/joint_recommendations.py
# 2.5 — Combined Recommendation Engine (Sephora + Ulta)

#   Write dashboards (first 200):
#     python src/joint_recommendations.py write_htmls --limit 200
#
#   Notebook dropdown:
#     from src.joint_recommendations import show_dashboard
#     show_dashboard()

import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -------------------------------------------------
# Paths
# -------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DATA_DIR = _PROJECT_ROOT / "data" / "processed"
MATCHED_DIR = DATA_DIR / "Matched"
SEPHORA_DIR = DATA_DIR / "Sephora"
ULTA_DIR = DATA_DIR / "Ulta"
JOINT_DIR = DATA_DIR / "Joint"
JOINT_RECS_PATH = JOINT_DIR / "joint_recommendations.csv"

OUTPUT_DIR = _PROJECT_ROOT / "notebooks" / "joint" / "outputs"
PRODUCT_DASHBOARD_DIR = OUTPUT_DIR / "product_dashboards"

TOP_N_SUB = 5
TOP_N_COMP = 5
TOP_N_TRADE = 3

_data = {}


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _slug(text):
    return re.sub(r"[^\w\-]", "_", str(text)).strip("_")


def _to_string(s: pd.Series) -> pd.Series:
    # robust string coercion (prevents float-vs-str merge errors)
    return s.astype("string")


# -------------------------------------------------
# A) PIPELINE: build joint_recommendations.csv
# -------------------------------------------------
def run_pipeline():
    """
    Build joint_recommendations.csv by attaching cross-retailer match info
    to BOTH Sephora and Ulta catalogs, then concat.
    This is the part that answers your lead: YES, it uses matched_products.
    """
    t0 = time.time()
    JOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    sephora_recs = pd.read_csv(SEPHORA_DIR / "sephora_recommendations.csv")
    ulta_recs = pd.read_csv(ULTA_DIR / "ulta_recommendations.csv")

    matched_pairs = pd.read_csv(MATCHED_DIR / "matched_pairs.csv")
    matched_products = pd.read_csv(MATCHED_DIR / "matched_products.csv")

    print("Loaded datasets:")
    print("  Sephora recs:", sephora_recs.shape)
    print("  Ulta recs:", ulta_recs.shape)
    print("  Matched pairs:", matched_pairs.shape)
    print("  Matched products:", matched_products.shape)

    # Normalize IDs to string (critical)
    sephora_recs["product_id"] = _to_string(sephora_recs["product_id"])
    ulta_recs["product_id"] = _to_string(ulta_recs["product_id"])

    # Cross-retailer mapping table
    cross_map = matched_pairs[["sephora_product_id", "ulta_product_id", "similarity_score"]].copy()
    cross_map["sephora_product_id"] = _to_string(cross_map["sephora_product_id"])
    cross_map["ulta_product_id"] = _to_string(cross_map["ulta_product_id"])
    print("Cross-retailer matches:", cross_map.shape)

    sephora_to_ulta = dict(zip(cross_map["sephora_product_id"], cross_map["ulta_product_id"]))
    ulta_to_sephora = dict(zip(cross_map["ulta_product_id"], cross_map["sephora_product_id"]))

    # Build basic lookup tables from matched_products (THIS is what your lead asked about)
    matched_products["product_id"] = _to_string(matched_products["product_id"])

    # Sephora basic info
    sephora_basic = matched_products[matched_products["retailer"] == "sephora"][
        ["product_id", "product_name", "brand", "category", "price", "rating"]
    ].copy()
    sephora_basic = sephora_basic.rename(
        columns={
            "product_id": "matched_sephora_id",
            "product_name": "matched_sephora_name",
            "brand": "matched_sephora_brand",
            "category": "matched_sephora_category",
            "price": "matched_sephora_price",
            "rating": "matched_sephora_rating",
        }
    )
    sephora_basic["matched_sephora_id"] = _to_string(sephora_basic["matched_sephora_id"])

    # Ulta basic info
    ulta_basic = matched_products[matched_products["retailer"] == "ulta"][
        ["product_id", "product_name", "brand", "category", "price", "rating"]
    ].copy()
    ulta_basic = ulta_basic.rename(
        columns={
            "product_id": "matched_ulta_id",
            "product_name": "matched_ulta_name",
            "brand": "matched_ulta_brand",
            "category": "matched_ulta_category",
            "price": "matched_ulta_price",
            "rating": "matched_ulta_rating",
        }
    )
    ulta_basic["matched_ulta_id"] = _to_string(ulta_basic["matched_ulta_id"])

    # -------------------------------------------------
    # Sephora -> attach matched Ulta info
    # -------------------------------------------------
    sephora_joint = sephora_recs.copy()
    sephora_joint["matched_ulta_id"] = _to_string(
        sephora_joint["product_id"].map(sephora_to_ulta)
    )

    sephora_joint = sephora_joint.merge(ulta_basic, on="matched_ulta_id", how="left")
    print("Sephora joint shape:", sephora_joint.shape)
    print("Sephora matched rows:", int(sephora_joint["matched_ulta_id"].notna().sum()))

    sephora_joint["price"] = pd.to_numeric(sephora_joint.get("price", np.nan), errors="coerce")
    sephora_joint["matched_ulta_price"] = pd.to_numeric(sephora_joint.get("matched_ulta_price", np.nan), errors="coerce")
    sephora_joint["avg_rating"] = pd.to_numeric(sephora_joint.get("avg_rating", np.nan), errors="coerce")
    sephora_joint["matched_ulta_rating"] = pd.to_numeric(sephora_joint.get("matched_ulta_rating", np.nan), errors="coerce")

    sephora_joint["price_diff_vs_ulta"] = sephora_joint["price"] - sephora_joint["matched_ulta_price"]
    sephora_joint["rating_diff_vs_ulta"] = sephora_joint["avg_rating"] - sephora_joint["matched_ulta_rating"]

    sephora_joint["ulta_cheaper"] = sephora_joint["price_diff_vs_ulta"] > 0
    sephora_joint["ulta_higher_rated"] = sephora_joint["rating_diff_vs_ulta"] < 0

    def _tag_sephora(r):
        if pd.isna(r["matched_ulta_id"]) or str(r["matched_ulta_id"]) in ("<NA>", "nan"):
            return "no_cross_match"
        if r["ulta_cheaper"] and r["ulta_higher_rated"]:
            return "ulta_better_and_cheaper"
        if r["ulta_cheaper"]:
            return "ulta_trade_down"
        if r["ulta_higher_rated"]:
            return "ulta_trade_up"
        return "similar"

    sephora_joint["cross_platform_signal"] = sephora_joint.apply(_tag_sephora, axis=1)
    print("Sephora cross_platform_signal counts:")
    print(sephora_joint["cross_platform_signal"].value_counts())

    sephora_joint["retailer"] = "sephora"

    # -------------------------------------------------
    # Ulta -> attach matched Sephora info
    # -------------------------------------------------
    ulta_joint = ulta_recs.copy()
    ulta_joint["matched_sephora_id"] = _to_string(
        ulta_joint["product_id"].map(ulta_to_sephora)
    )

    ulta_joint = ulta_joint.merge(sephora_basic, on="matched_sephora_id", how="left")
    print("Ulta joint shape:", ulta_joint.shape)
    print("Ulta matched rows:", int(ulta_joint["matched_sephora_id"].notna().sum()))

    ulta_joint["price"] = pd.to_numeric(ulta_joint.get("price", np.nan), errors="coerce")
    ulta_joint["matched_sephora_price"] = pd.to_numeric(ulta_joint.get("matched_sephora_price", np.nan), errors="coerce")
    ulta_joint["avg_rating"] = pd.to_numeric(ulta_joint.get("avg_rating", np.nan), errors="coerce")
    ulta_joint["matched_sephora_rating"] = pd.to_numeric(ulta_joint.get("matched_sephora_rating", np.nan), errors="coerce")

    ulta_joint["price_diff_vs_sephora"] = ulta_joint["price"] - ulta_joint["matched_sephora_price"]
    ulta_joint["rating_diff_vs_sephora"] = ulta_joint["avg_rating"] - ulta_joint["matched_sephora_rating"]

    ulta_joint["sephora_cheaper"] = ulta_joint["price_diff_vs_sephora"] > 0
    ulta_joint["sephora_higher_rated"] = ulta_joint["rating_diff_vs_sephora"] < 0

    def _tag_ulta(r):
        if pd.isna(r["matched_sephora_id"]) or str(r["matched_sephora_id"]) in ("<NA>", "nan"):
            return "no_cross_match"
        if r["sephora_cheaper"] and r["sephora_higher_rated"]:
            return "sephora_better_and_cheaper"
        if r["sephora_cheaper"]:
            return "sephora_trade_down"
        if r["sephora_higher_rated"]:
            return "sephora_trade_up"
        return "similar"

    ulta_joint["cross_platform_signal"] = ulta_joint.apply(_tag_ulta, axis=1)
    print("Ulta cross_platform_signal counts:")
    print(ulta_joint["cross_platform_signal"].value_counts())

    ulta_joint["retailer"] = "ulta"

    # Save outputs
    sephora_out = JOINT_DIR / "joint_recommendations_sephora_view.csv"
    ulta_out = JOINT_DIR / "joint_recommendations_ulta_view.csv"

    sephora_joint.to_csv(sephora_out, index=False)
    ulta_joint.to_csv(ulta_out, index=False)

    combined = pd.concat([sephora_joint, ulta_joint], ignore_index=True)
    combined.to_csv(JOINT_RECS_PATH, index=False)

    print("Saved:")
    print(" -", sephora_out)
    print(" -", ulta_out)
    print(" -", JOINT_RECS_PATH)
    print("Combined shape:", combined.shape)
    print(f"Done in {time.time() - t0:.1f}s")


# -------------------------------------------------
# B) DASHBOARD: load from joint_recommendations.csv
# -------------------------------------------------
def _load_data():
    if _data:
        return

    t0 = time.time()
    if not JOINT_RECS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {JOINT_RECS_PATH}. Run: python src/joint_recommendations.py build_csv"
        )

    print("Loading JOINT recommendation data from CSV...")
    recs = pd.read_csv(JOINT_RECS_PATH)
    recs["product_id"] = _to_string(recs["product_id"])

    if "retailer" not in recs.columns:
        recs["retailer"] = "unknown"

    _data["recs"] = recs

    # Build dropdown list
    products = []
    for _, row in recs.iterrows():
        brand = str(row.get("brand", ""))
        name = str(row.get("product_name", ""))
        price = float(pd.to_numeric(row.get("price", 0), errors="coerce") or 0)
        ret = str(row.get("retailer", "")).lower()
        label = f"[{ret}] {brand} — {name} (${price:.0f})"
        products.append((label, str(row["product_id"])))
    products.sort(key=lambda x: x[0])
    _data["product_list"] = products

    print(f"  Loaded {len(products):,} products in {time.time() - t0:.1f}s")


def _extract_items(row, prefix, top_n, include_verif=False):
    items = []
    for rk in range(1, top_n + 1):
        n = row.get(f"{prefix}_{rk}_name", "")
        if pd.isna(n) or n == "":
            continue

        it = dict(
            name=str(n),
            brand=str(row.get(f"{prefix}_{rk}_brand", "")),
            price=float(pd.to_numeric(row.get(f"{prefix}_{rk}_price", 0), errors="coerce") or 0),
            sent=float(pd.to_numeric(row.get(f"{prefix}_{rk}_sentiment", 0), errors="coerce") or 0),
            score=float(pd.to_numeric(row.get(f"{prefix}_{rk}_score", 0), errors="coerce") or 0),
            cat=str(row.get(f"{prefix}_{rk}_category", "")),
            rating=float(pd.to_numeric(row.get(f"{prefix}_{rk}_rating", 0), errors="coerce") or 0),
            reviews=int(pd.to_numeric(row.get(f"{prefix}_{rk}_reviews", 0), errors="coerce") or 0),
        )

        if include_verif:
            vb_val = float(pd.to_numeric(row.get(f"{prefix}_{rk}_verified_buyer", 0), errors="coerce") or 0)
            disc_val = float(pd.to_numeric(row.get(f"{prefix}_{rk}_disclosure", 0), errors="coerce") or 0)
            if disc_val > 0.20:
                vtag = f"Seeded {disc_val:.0%}"
            elif vb_val > 0.20:
                vtag = f"Verified {vb_val:.0%}"
            else:
                vtag = "Low verif."
            it.update(vb=vb_val, disc=disc_val, vtag=vtag)

        items.append(it)

    short = [f"#{i+1} {it['brand'][:18]}" for i, it in enumerate(items)]
    hover = [
        f"<b>{it['brand']}</b><br>{it['name']}<br>"
        f"Category: {it['cat']}<br>"
        f"${it['price']:.0f} · ★{it['rating']:.2f} · Sent {it['sent']:.3f}<br>"
        f"{it['reviews']:,} reviews · Score {it['score']:.4f}"
        + (f"<br><b>{it['vtag']}</b>" if include_verif else "")
        for it in items
    ]
    return items, short, hover


def build_product_dashboard(product_id):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _load_data()
    recs = _data["recs"]

    match = recs[recs["product_id"].astype(str) == str(product_id)]
    if len(match) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Product not found", showarrow=False, font=dict(size=20))
        return fig

    row = match.iloc[0]
    retailer = str(row.get("retailer", "")).lower()

    # Theme by retailer
    if retailer == "ulta":
        theme_color = "#880e4f"
        header_fill = "#880e4f"
        include_verif = True
    else:
        theme_color = "#2d6a4f"
        header_fill = "#1a1a1a"
        include_verif = False

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Close Substitutes — Price",
            "Close Substitutes — Sentiment",
            "Complements — Price",
            "Complements — Sentiment",
            "Trade-Up — Price",
            "Trade-Down — Price",
            "Avg Score by Intent",
            "Cross-Retailer Match (if any)",
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

    src_price = float(pd.to_numeric(row.get("price", 0), errors="coerce") or 0)
    src_sent = float(pd.to_numeric(row.get("avg_sentiment", 0), errors="coerce") or 0)

    # Substitutes
    items, short, hover = _extract_items(row, "sub", TOP_N_SUB, include_verif=include_verif)
    if items:
        fig.add_trace(go.Bar(
            y=short[::-1], x=[it["price"] for it in items][::-1], orientation="h",
            marker_color=theme_color, showlegend=False,
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

    # Complements
    items, short, hover = _extract_items(row, "comp", TOP_N_COMP, include_verif=include_verif)
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

    # Trade-Up / Trade-Down
    items, short, hover = _extract_items(row, "tradeup", TOP_N_TRADE, include_verif=include_verif)
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

    items, short, hover = _extract_items(row, "tradedn", TOP_N_TRADE, include_verif=include_verif)
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

    # Avg score by intent
    intent_names, intent_scores = [], []
    for prefix, label, top_n in [
        ("sub", "Substitutes", TOP_N_SUB),
        ("comp", "Complements", TOP_N_COMP),
        ("tradeup", "Trade-Up", TOP_N_TRADE),
        ("tradedn", "Trade-Down", TOP_N_TRADE),
    ]:
        sc = [float(pd.to_numeric(row.get(f"{prefix}_{rk}_score", 0), errors="coerce") or 0) for rk in range(1, top_n + 1)]
        intent_names.append(label)
        intent_scores.append(float(np.mean(sc)) if sc else 0.0)

    fig.add_trace(go.Bar(
        x=intent_names, y=intent_scores,
        marker_color=["#2d6a4f", "#7b2d8b", "#1a73e8", "#e67700"],
        text=[f"{s:.4f}" for s in intent_scores], textposition="auto",
        showlegend=False,
    ), row=4, col=1)

    # Cross-retailer match table (based on columns created in run_pipeline)
    if retailer == "ulta":
        mid = row.get("matched_sephora_id", "")
        mname = row.get("matched_sephora_name", "")
        mbrand = row.get("matched_sephora_brand", "")
        mcat = row.get("matched_sephora_category", "")
        mprice = row.get("matched_sephora_price", np.nan)
        mrating = row.get("matched_sephora_rating", np.nan)
        other = "Sephora"
    else:
        mid = row.get("matched_ulta_id", "")
        mname = row.get("matched_ulta_name", "")
        mbrand = row.get("matched_ulta_brand", "")
        mcat = row.get("matched_ulta_category", "")
        mprice = row.get("matched_ulta_price", np.nan)
        mrating = row.get("matched_ulta_rating", np.nan)
        other = "Ulta"

    if pd.isna(mid) or str(mid) in ("<NA>", "nan", ""):
        header_vals = ["Cross-Match", "", "", "", "", ""]
        table_vals = [["No cross-retailer match found"], [""], [""], [""], [""], [""]]
    else:
        header_vals = ["Retailer", "Brand", "Product", "Category", "Price", "Rating"]
        table_vals = [
            [other],
            [str(mbrand)],
            [str(mname)[:40]],
            [str(mcat)[:30]],
            [f"${float(pd.to_numeric(mprice, errors='coerce')):.0f}" if pd.notna(mprice) else "—"],
            [f"{float(pd.to_numeric(mrating, errors='coerce')):.2f}" if pd.notna(mrating) else "—"],
        ]

    fig.add_trace(go.Table(
        header=dict(values=header_vals, font=dict(size=10, color="white"),
                    fill_color=header_fill, align="left"),
        cells=dict(values=table_vals, font=dict(size=9),
                   fill_color="#fff", align="left"),
    ), row=4, col=2)

    # KPI title
    product_name = row.get("product_name", "")
    brand = row.get("brand", "")
    category = row.get("category", "")
    rating = float(pd.to_numeric(row.get("avg_rating", 0), errors="coerce") or 0)
    sentiment = float(pd.to_numeric(row.get("avg_sentiment", 0), errors="coerce") or 0)
    review_count = int(pd.to_numeric(row.get("review_count", 0), errors="coerce") or 0)

    kpi = (
        f"<b style='font-size:16px'>[{retailer.upper()}] {brand} — {product_name}</b><br>"
        f"<span style='font-size:11px; color:#555;'>"
        f"Category: {category} &nbsp;|&nbsp; "
        f"Price: ${src_price:.0f} &nbsp;|&nbsp; "
        f"Rating: {rating:.2f} &nbsp;|&nbsp; "
        f"Sentiment: {sentiment:.3f} &nbsp;|&nbsp; "
        f"Reviews: {review_count:,}</span><br>"
        f"<span style='font-size:10px; color:#999;'>"
        f"Red dashed = source product &nbsp;|&nbsp; Hover bars for details</span>"
    )

    # Optional Ulta verification badge (only if columns exist)
    if retailer == "ulta":
        src_vb = float(pd.to_numeric(row.get("pct_verified_buyer", 0), errors="coerce") or 0)
        src_disc = float(pd.to_numeric(row.get("pct_has_disclosure", 0), errors="coerce") or 0)
        if src_disc > 0.20:
            badge = f"⚠️ Seeded ({src_disc:.0%} disclosure)"
        elif src_vb > 0.20:
            badge = f"✓ Verified ({src_vb:.0%} buyers)"
        else:
            badge = "○ Low verification"
        kpi += f"<br><span style='font-size:11px; color:#880e4f; font-weight:bold;'>{badge}</span>"

    fig.update_layout(
        height=1800, width=1300,
        title=dict(text=kpi, font=dict(size=12), x=0.01, y=0.99),
        template="plotly_white",
        showlegend=False,
        margin=dict(t=110, l=160, r=40, b=30),
    )
    fig.update_yaxes(tickfont=dict(size=11))
    return fig


def write_product_html(product_id, output_dir=None):
    _load_data()
    import plotly.io as pio  # noqa: F401

    out = Path(output_dir) if output_dir else PRODUCT_DASHBOARD_DIR
    out.mkdir(parents=True, exist_ok=True)

    recs = _data["recs"]
    match = recs[recs["product_id"].astype(str) == str(product_id)]
    if len(match) > 0:
        r = match.iloc[0]
        retailer = str(r.get("retailer", ""))
        brand = _slug(str(r.get("brand", "")))
        name = _slug(str(r.get("product_name", product_id))[:40])
        filename = f"{_slug(retailer)}_{brand}_{name}_{_slug(str(product_id))}.html"
    else:
        filename = f"{_slug(str(product_id))}.html"

    fig = build_product_dashboard(product_id)
    path = out / filename
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(" ->", path)
    return path


def write_all_product_htmls(output_dir=None, limit=None):
    """
    Writes dashboards for all products (or first N if limit is set),
    then writes index page to notebooks/joint/outputs/joint_product_dashboards_index.html
    """
    _load_data()
    recs = _data["recs"]

    out = Path(output_dir) if output_dir else PRODUCT_DASHBOARD_DIR
    out.mkdir(parents=True, exist_ok=True)

    total = len(recs) if limit is None else min(int(limit), len(recs))
    print(f"Writing {total:,} joint product dashboards to {out}/")

    written = []
    for i, (_, rec_row) in enumerate(recs.head(total).iterrows(), 1):
        pid = str(rec_row["product_id"])
        try:
            path = write_product_html(pid, output_dir=out)
            written.append((rec_row, path.name))
        except Exception as e:
            print(f"  WARNING: skipped {pid!r} — {e}")
        if i % 200 == 0:
            print(f"  {i:,}/{total:,} complete...")

    index_path = out.parent / "joint_product_dashboards_index.html"

    rows_html = ""
    for rec_row, filename in written:
        retailer = str(rec_row.get("retailer", "")).lower()
        brand = str(rec_row.get("brand", ""))
        name = str(rec_row.get("product_name", ""))
        category = str(rec_row.get("category", ""))
        signal = str(rec_row.get("cross_platform_signal", ""))

        try:
            price = f"${float(pd.to_numeric(rec_row.get('price', 0), errors='coerce') or 0):.0f}"
            rating = f"{float(pd.to_numeric(rec_row.get('avg_rating', 0), errors='coerce') or 0):.2f}"
            sentiment = f"{float(pd.to_numeric(rec_row.get('avg_sentiment', 0), errors='coerce') or 0):.3f}"
            reviews = f"{int(pd.to_numeric(rec_row.get('review_count', 0), errors='coerce') or 0):,}"
        except Exception:
            price = rating = sentiment = reviews = "—"

        rows_html += (
            f"<tr>"
            f"<td><a href='product_dashboards/{filename}'>[{retailer}] {brand}</a></td>"
            f"<td>{name}</td>"
            f"<td>{category}</td>"
            f"<td>{price}</td>"
            f"<td>{rating}</td>"
            f"<td>{sentiment}</td>"
            f"<td>{reviews}</td>"
            f"<td>{signal}</td>"
            f"</tr>\n"
        )

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Joint Recommendations — Product Index</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1500px; margin: 40px auto; padding: 0 20px; color: #2C3E50; }}
    h1 {{ color: #2C3E50; border-bottom: 2px solid #2C3E50; padding-bottom: 10px; }}
    p.subtitle {{ color: #7f8c8d; margin-top: -8px; }}
    input {{ width: 100%; padding: 8px; margin-bottom: 16px; box-sizing: border-box;
             border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ background: #2C3E50; color: white; padding: 10px; text-align: left; position: sticky; top: 0; }}
    td {{ padding: 8px 10px; border-bottom: 1px solid #ecf0f1; }}
    tr:hover td {{ background: #f0f4f8; }}
    a {{ color: #2C3E50; text-decoration: none; font-weight: bold; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>Joint Recommendations — Product Index</h1>
  <p class="subtitle">
    {len(written):,} products &nbsp;|&nbsp;
    Click any brand to open its full recommendation dashboard (Sephora + Ulta)
  </p>
  <input type="text" id="search" placeholder="Filter by retailer, brand, product, or category..." onkeyup="filterTable()">
  <table id="productTable">
    <thead>
      <tr>
        <th>Retailer + Brand</th>
        <th>Product</th>
        <th>Category</th>
        <th>Price</th>
        <th>Avg Rating</th>
        <th>Avg Sentiment</th>
        <th>Reviews</th>
        <th>Cross Signal</th>
      </tr>
    </thead>
    <tbody>
{rows_html}    </tbody>
  </table>
  <script>
    function filterTable() {{
      const q = document.getElementById("search").value.toLowerCase();
      document.querySelectorAll("#productTable tbody tr").forEach(row => {{
        const text = Array.from(row.cells).slice(0, 4).map(c => c.textContent).join(" ").toLowerCase();
        row.style.display = text.includes(q) ? "" : "none";
      }});
    }}
  </script>
</body>
</html>"""

    index_path.write_text(index_html, encoding="utf-8")
    print("\nIndex page written:", index_path)
    print(f"Done. {len(written):,}/{total:,} joint product dashboards saved.")
    return index_path


def show_dashboard():
    import ipywidgets as widgets
    from IPython.display import display, HTML

    _load_data()
    products = _data["product_list"]

    display(HTML(
        "<h2 style='text-align:center; color:#2C3E50;'>"
        "Joint Recommendations — Product Dashboard</h2>"
        "<p style='text-align:center; color:#7f8c8d;'>"
        f"Select from {len(products):,} products (Sephora + Ulta).</p>"
    ))

    dropdown = widgets.Dropdown(
        options=products,
        value=products[0][1],
        description="Product:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="700px"),
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


# -------------------------------------------------
# CLI entry
# -------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", nargs="?", default="help",
                        choices=["help", "build_csv", "write_htmls"])
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    if args.cmd == "build_csv":
        run_pipeline()
    elif args.cmd == "write_htmls":
        write_all_product_htmls(limit=args.limit)
    else:
        print("Commands:")
        print("  python src/joint_recommendations.py build_csv")
        print("  python src/joint_recommendations.py write_htmls --limit 200")
        print("Notebook:")
        print("  from src.joint_recommendations import show_dashboard; show_dashboard()")