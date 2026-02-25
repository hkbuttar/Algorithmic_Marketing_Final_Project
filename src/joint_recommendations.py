# joint_recommendations.py
# 2.5 — Combined (Sephora + Ulta) Product Dashboards

import pandas as pd
import numpy as np
import re
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

JOINT_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed" / "Joint"
JOINT_RECS_PATH = JOINT_PROCESSED_DIR / "joint_recommendations.csv"

OUTPUT_DIR = _PROJECT_ROOT / "notebooks" / "joint" / "outputs"
PRODUCT_DASHBOARD_DIR = OUTPUT_DIR / "product_dashboards"

# keep same constants as your lead’s scripts
TOP_N_SUB = 5
TOP_N_COMP = 5
TOP_N_TRADE = 3

_data = {}

def _slug(text):
    return re.sub(r"[^\w\-]", "_", str(text)).strip("_")

def _load_data():
    if _data:
        return
    t0 = time.time()
    print("Loading JOINT recommendation data from CSV...")
    recs = pd.read_csv(JOINT_RECS_PATH)

    # Normalize ids
    recs["product_id"] = recs["product_id"].astype("string")

    # Ensure retailer exists
    if "retailer" not in recs.columns:
        # fallback: if you used source_retailer earlier
        if "source_retailer" in recs.columns:
            recs["retailer"] = recs["source_retailer"]
        else:
            recs["retailer"] = "unknown"

    _data["recs"] = recs

    # Build dropdown list (optional)
    products = []
    for _, row in recs.iterrows():
        brand = str(row.get("brand", ""))
        name = str(row.get("product_name", ""))
        price = float(row.get("price", 0) or 0)
        ret = str(row.get("retailer", "")).lower()
        label = f"[{ret}] {brand} — {name} (${price:.0f})"
        products.append((label, str(row["product_id"])))
    products.sort(key=lambda x: x[0])
    _data["product_list"] = products

    elapsed = time.time() - t0
    print(f"  Loaded {len(products):,} products in {elapsed:.1f}s")

def _extract_items(row, prefix, top_n, include_verif=False):
    items = []
    for rk in range(1, top_n + 1):
        n = row.get(f"{prefix}_{rk}_name", "")
        if pd.isna(n) or n == "":
            continue
        it = dict(
            name=str(n),
            brand=str(row.get(f"{prefix}_{rk}_brand", "")),
            price=float(row.get(f"{prefix}_{rk}_price", 0) or 0),
            sent=float(row.get(f"{prefix}_{rk}_sentiment", 0) or 0),
            score=float(row.get(f"{prefix}_{rk}_score", 0) or 0),
            cat=str(row.get(f"{prefix}_{rk}_category", "")),
            rating=float(row.get(f"{prefix}_{rk}_rating", 0) or 0),
            reviews=int(row.get(f"{prefix}_{rk}_reviews", 0) or 0),
        )

        if include_verif:
            vb_val = float(row.get(f"{prefix}_{rk}_verified_buyer", 0) or 0)
            disc_val = float(row.get(f"{prefix}_{rk}_disclosure", 0) or 0)
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

    # Theme (match your lead’s look)
    if retailer == "ulta":
        theme_color = "#880e4f"
        header_fill = "#880e4f"
        row_hover = "#fdf0f5"
        include_verif = True
    else:
        theme_color = "#2d6a4f"
        header_fill = "#1a1a1a"
        row_hover = "#f0f4f8"
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

    src_price = float(row.get("price", 0) or 0)
    src_sent = float(row.get("avg_sentiment", 0) or 0)

    # Substitutes
    items, short, hover = _extract_items(row, "sub", TOP_N_SUB, include_verif=include_verif)
    if items:
        fig.add_trace(go.Bar(
            y=short[::-1], x=[it["price"] for it in items][::-1], orientation="h",
            marker_color=theme_color if retailer != "ulta" else ["#3498db"] * len(items),
            showlegend=False,
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
    for prefix, label, top_n in [("sub", "Substitutes", TOP_N_SUB),
                                ("comp", "Complements", TOP_N_COMP),
                                ("tradeup", "Trade-Up", TOP_N_TRADE),
                                ("tradedn", "Trade-Down", TOP_N_TRADE)]:
        sc = [float(row.get(f"{prefix}_{rk}_score", 0) or 0) for rk in range(1, top_n + 1)]
        intent_names.append(label)
        intent_scores.append(float(np.mean(sc)) if sc else 0.0)

    fig.add_trace(go.Bar(
        x=intent_names, y=intent_scores,
        marker_color=["#2d6a4f", "#7b2d8b", "#1a73e8", "#e67700"],
        text=[f"{s:.4f}" for s in intent_scores], textposition="auto",
        showlegend=False,
    ), row=4, col=1)

    # Cross-retailer match table
    # We use the columns produced by your joint_recommendations.py
    if retailer == "ulta":
        mid = row.get("matched_sephora_id", "")
        mname = row.get("matched_sephora_name", "")
        mbrand = row.get("matched_sephora_brand", "")
        mcat = row.get("matched_sephora_category", "")
        mprice = row.get("matched_sephora_price", np.nan)
        mrating = row.get("matched_sephora_rating", np.nan)
        signal = row.get("cross_platform_signal", "")
        other = "Sephora"
    else:
        mid = row.get("matched_ulta_id", "")
        mname = row.get("matched_ulta_name", "")
        mbrand = row.get("matched_ulta_brand", "")
        mcat = row.get("matched_ulta_category", "")
        mprice = row.get("matched_ulta_price", np.nan)
        mrating = row.get("matched_ulta_rating", np.nan)
        signal = row.get("cross_platform_signal", "")
        other = "Ulta"

    if pd.isna(mid) or str(mid) == "<NA>" or str(mid) == "":
        table_vals = [["No cross-retailer match found"], [""], [""], [""], [""], [""]]
        header_vals = ["Cross-Match", "", "", "", "", ""]
    else:
        table_vals = [
            [other],
            [str(mbrand)],
            [str(mname)[:40]],
            [str(mcat)[:30]],
            [f"${float(mprice):.0f}" if pd.notna(mprice) else "—"],
            [f"{float(mrating):.2f}" if pd.notna(mrating) else "—"],
        ]
        header_vals = ["Retailer", "Brand", "Product", "Category", "Price", "Rating"]

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
    rating = float(row.get("avg_rating", 0) or 0)
    sentiment = float(row.get("avg_sentiment", 0) or 0)
    review_count = int(row.get("review_count", 0) or 0)

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

    # Add Ulta verification badge if present
    if retailer == "ulta":
        src_vb = float(row.get("pct_verified_buyer", 0) or 0)
        src_disc = float(row.get("pct_has_disclosure", 0) or 0)
        if src_disc > 0.20:
            badge = f"⚠️ Seeded ({src_disc:.0%} disclosure)"
        elif src_vb > 0.20:
            badge = f"✓ Verified ({src_vb:.0%} buyers)"
        else:
            badge = "○ Low verification"
        kpi += (
            f"<br><span style='font-size:11px; color:#880e4f; font-weight:bold;'>"
            f"{badge}</span>"
        )

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
    print(f"  -> {path}")
    return path

def write_all_product_htmls(output_dir=None, limit=None):
    """
    Writes dashboards for all products (or first N if limit is set),
    then writes an index page to notebooks/joint/outputs/joint_product_dashboards_index.html
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
            price = f"${float(rec_row.get('price', 0) or 0):.0f}"
            rating = f"{float(rec_row.get('avg_rating', 0) or 0):.2f}"
            sentiment = f"{float(rec_row.get('avg_sentiment', 0) or 0):.3f}"
            reviews = f"{int(rec_row.get('review_count', 0) or 0):,}"
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
    print(f"\nIndex page written: {index_path}")
    print(f"Done. {len(written):,}/{total:,} joint product dashboards saved.")
    return index_path

if __name__ == "__main__":
    # small sanity run (don’t do full by default)
    _load_data()
    print("Tip: from src.joint_dashboards import write_all_product_htmls; write_all_product_htmls(limit=50)")