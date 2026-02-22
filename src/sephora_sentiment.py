"""
Brand Health & Sentiment Analysis — Sephora
=============================================

Usage:
  As script (runs full pipeline):
    python src/sephora_brand_health.py

  As import (dashboard only, loads from pre-built CSVs):
    from sephora_brand_health import show_dashboard
    show_dashboard()

Pipeline (Steps 1–9, runs via script or run_pipeline()):
  1. Text preprocessing & review filtering
  2. Sentiment scoring (VADER)
  3. Rating–sentiment mismatch detection
  4. Topic modeling (NMF on TF-IDF)
  5. Topic–sentiment linkage (drivers, not themes)
  6. Delighters vs. disappointers per brand
  7. Complaint concentration analysis
  8. Price/value perception diagnostics
  9. Brand-level aggregation → CSVs

Dashboard (Step 10, runs via show_dashboard()):
  Interactive Plotly + ipywidgets brand toggle over ALL brands.

Inputs:
  data/processed/sephora_products.csv
  data/processed/sephora_reviews.csv

Outputs (CSVs):
  data/processed/sephora_brand_health.csv
  data/processed/sephora_reviews_enriched.csv
  data/processed/sephora_topic_drivers.csv
  data/processed/sephora_brand_topic_labels.csv
  data/processed/sephora_delighters_disappointers.csv
  data/processed/sephora_complaint_concentration.csv
  data/processed/sephora_value_perception.csv

Outputs (HTML):
  outputs/sephora_brand_health_overview.html
"""

import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# Constants
# ============================================================
_SCRIPT_DIR = Path(__file__).resolve().parent          # src/
PROJECT_ROOT = _SCRIPT_DIR.parent                       # Algorithmic_Marketing_Final_Project/
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MIN_BRAND_REVIEWS = 30
N_TOPICS = 12
MIN_REVIEW_LENGTH = 20
VALUE_KEYWORDS = [
    "worth", "expensive", "cheap", "overpriced", "price", "value",
    "money", "cost", "pricey", "affordable", "waste", "ripoff",
    "rip off", "not worth", "pay", "paid", "dollar", "buck",
]

# ============================================================
# Module-level data (populated by run_pipeline() or _load_data())
# ============================================================
_data = {}


def _load_data():
    """Load pre-built CSVs for dashboard use. Called lazily on first dashboard call."""
    if _data:
        return  # already loaded

    import time
    t0 = time.time()
    print("Loading dashboard data from CSVs...")

    _data["brand_agg"] = pd.read_csv(PROCESSED_DIR / "sephora_brand_health.csv")

    # Only load the 11 columns the dashboard actually uses (not all 30+)
    review_cols = [
        "brand", "Rating", "sentiment_compound", "sentiment_label",
        "mismatch_type", "dominant_topic", "submission_date",
        "mentions_value", "rating_sentiment_gap", "text_length", "Helpfulness",
    ]
    print("  Loading reviews (subset columns)...")
    _data["reviews"] = pd.read_csv(
        PROCESSED_DIR / "sephora_reviews_enriched.csv",
        usecols=review_cols,
    )

    # Pre-index reviews by brand for fast lookup
    print("  Indexing by brand...")
    _data["brand_reviews"] = dict(tuple(_data["reviews"].groupby("brand")))

    _data["topic_drivers_df"] = pd.read_csv(PROCESSED_DIR / "sephora_topic_drivers.csv")
    _data["dd_df"] = pd.read_csv(PROCESSED_DIR / "sephora_delighters_disappointers.csv")

    tl = pd.read_csv(PROCESSED_DIR / "sephora_brand_topic_labels.csv")
    _data["topic_labels"] = dict(zip(tl["topic_id"], tl["keywords"]))

    _data["dashboard_brands"] = _data["brand_agg"]["brand"].tolist()

    elapsed = time.time() - t0
    print(f"  Loaded {len(_data['dashboard_brands'])} brands, "
          f"{len(_data['reviews']):,} reviews in {elapsed:.1f}s")


# ============================================================
# PIPELINE: run_pipeline()
# ============================================================
def run_pipeline():
    """Run the full brand health pipeline (Steps 1–9) and save CSVs."""

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    from scipy.stats import entropy
    import plotly.graph_objects as go

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # STEP 1: Load, Merge, Clean
    # ============================================================
    print("=" * 60)
    print("STEP 1: Loading and cleaning data")
    print("=" * 60)

    products = pd.read_csv(PROCESSED_DIR / "sephora_products.csv")
    reviews = pd.read_csv(PROCESSED_DIR / "sephora_reviews.csv")

    reviews = reviews.merge(
        products[["product_id", "brand", "product_name", "category", "price"]],
        left_on="pd_id", right_on="product_id", how="left",
    ).drop(columns="product_id")

    print(f"  Raw reviews: {len(reviews):,}")
    print(f"  Matched to brand: {reviews['brand'].notna().sum():,}")

    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"[^\w\s.,!?'-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    reviews["clean_text"] = reviews["ReviewText"].apply(clean_text)
    reviews["text_length"] = reviews["clean_text"].str.len()

    before = len(reviews)
    reviews = reviews[reviews["text_length"] >= MIN_REVIEW_LENGTH].copy()
    print(f"  After filtering short (<{MIN_REVIEW_LENGTH} chars): {len(reviews):,} ({before - len(reviews):,} removed)")

    reviews["submission_date"] = pd.to_datetime(reviews["SubmissionTime"], errors="coerce")
    print(f"  Unique brands: {reviews['brand'].nunique()}")
    print(f"  Unique products: {reviews['pd_id'].nunique()}")

    # ============================================================
    # STEP 2: Sentiment Scoring (VADER)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: Sentiment scoring (VADER)")
    print("=" * 60)

    analyzer = SentimentIntensityAnalyzer()

    def vader_compound(text):
        if not text:
            return 0.0
        return analyzer.polarity_scores(text)["compound"]

    reviews["sentiment_compound"] = reviews["clean_text"].apply(vader_compound)
    reviews["sentiment_label"] = pd.cut(
        reviews["sentiment_compound"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"],
    )
    reviews["sentiment_intensity"] = reviews["sentiment_compound"].abs()

    print(f"  Positive: {(reviews['sentiment_label'] == 'positive').mean():.1%}")
    print(f"  Neutral:  {(reviews['sentiment_label'] == 'neutral').mean():.1%}")
    print(f"  Negative: {(reviews['sentiment_label'] == 'negative').mean():.1%}")
    print(f"  Mean compound: {reviews['sentiment_compound'].mean():.3f}")

    # ============================================================
    # STEP 3: Rating–Sentiment Mismatch Detection
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: Rating-sentiment mismatch detection")
    print("=" * 60)

    reviews["rating_normalized"] = (reviews["Rating"] - 3) / 2
    reviews["rating_sentiment_gap"] = reviews["rating_normalized"] - reviews["sentiment_compound"]

    reviews["mismatch_type"] = "aligned"
    reviews.loc[
        (reviews["Rating"] >= 4) & (reviews["sentiment_compound"] < -0.05),
        "mismatch_type",
    ] = "overrated_unhappy"
    reviews.loc[
        (reviews["Rating"] <= 2) & (reviews["sentiment_compound"] > 0.05),
        "mismatch_type",
    ] = "underrated_positive"
    reviews.loc[
        (reviews["Rating"] == 3) & (reviews["sentiment_compound"].abs() > 0.5),
        "mismatch_type",
    ] = "ambivalent"

    for mt, count in reviews["mismatch_type"].value_counts().items():
        print(f"    {mt}: {count:,} ({count / len(reviews):.1%})")

    # ============================================================
    # STEP 4: Topic Modeling (NMF on TF-IDF)
    # ============================================================
    print("\n" + "=" * 60)
    print(f"STEP 4: Topic modeling (NMF, {N_TOPICS} topics)")
    print("=" * 60)

    tfidf = TfidfVectorizer(
        max_features=8000, min_df=10, max_df=0.85,
        ngram_range=(1, 2), stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    tfidf_matrix = tfidf.fit_transform(reviews["clean_text"])
    feature_names = tfidf.get_feature_names_out()
    print(f"  TF-IDF matrix: {tfidf_matrix.shape}")

    nmf = NMF(n_components=N_TOPICS, random_state=42, max_iter=300, init="nndsvda")
    W = nmf.fit_transform(tfidf_matrix)
    H = nmf.components_

    topic_labels = {}
    for i, topic_vec in enumerate(H):
        top_idx = topic_vec.argsort()[-10:][::-1]
        top_words = [feature_names[j] for j in top_idx]
        topic_labels[i] = ", ".join(top_words)
        print(f"  Topic {i}: {topic_labels[i]}")

    topic_cols = [f"topic_{i}" for i in range(N_TOPICS)]
    for i in range(N_TOPICS):
        reviews[f"topic_{i}"] = W[:, i]
    reviews["dominant_topic"] = W.argmax(axis=1)
    reviews["topic_confidence"] = W.max(axis=1)

    # ============================================================
    # STEP 5: Topic–Sentiment Linkage (Drivers)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 5: Topic-sentiment driver analysis")
    print("=" * 60)

    print("\n  Global topic-sentiment correlations:")
    for col in topic_cols:
        corr = reviews[col].corr(reviews["sentiment_compound"])
        tid = int(col.split("_")[1])
        print(f"    Topic {tid}: {corr:+.3f} — {topic_labels[tid][:50]}")

    brand_topic_drivers = []
    for brand, bdf in reviews.groupby("brand"):
        if len(bdf) < MIN_BRAND_REVIEWS:
            continue
        for i in range(N_TOPICS):
            corr = bdf[f"topic_{i}"].corr(bdf["sentiment_compound"])
            avg_sent = bdf.loc[bdf["dominant_topic"] == i, "sentiment_compound"].mean()
            prev = (bdf["dominant_topic"] == i).mean()
            brand_topic_drivers.append({
                "brand": brand, "topic_id": i,
                "topic_keywords": topic_labels[i],
                "correlation_with_sentiment": corr if not np.isnan(corr) else 0,
                "avg_sentiment_when_dominant": avg_sent if not np.isnan(avg_sent) else 0,
                "topic_prevalence": prev, "n_reviews": len(bdf),
            })

    topic_drivers_df = pd.DataFrame(brand_topic_drivers)
    print(f"\n  Brand x topic driver matrix: {topic_drivers_df.shape}")

    # ============================================================
    # STEP 6: Delighters vs. Disappointers per Brand
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 6: Delighters vs. disappointers")
    print("=" * 60)

    delighters_disappointers = []
    for brand, bdf in reviews.groupby("brand"):
        if len(bdf) < MIN_BRAND_REVIEWS:
            continue
        brand_avg = bdf["sentiment_compound"].mean()
        for i in range(N_TOPICS):
            mask = bdf["dominant_topic"] == i
            if mask.sum() < 5:
                continue
            topic_sent = bdf.loc[mask, "sentiment_compound"].mean()
            lift = topic_sent - brand_avg
            if lift > 0.1:
                role = "delighter"
            elif lift < -0.1:
                role = "disappointer"
            else:
                role = "neutral"
            brand_neg = bdf[bdf["sentiment_label"] == "negative"]
            neg_mass = (brand_neg["dominant_topic"] == i).sum() / max(len(brand_neg), 1)
            delighters_disappointers.append({
                "brand": brand, "topic_id": i,
                "topic_keywords": topic_labels[i], "role": role,
                "sentiment_lift": lift, "topic_avg_sentiment": topic_sent,
                "topic_sentiment_std": bdf.loc[mask, "sentiment_compound"].std(),
                "topic_prevalence": mask.mean(),
                "neg_mass_contribution": neg_mass,
                "n_topic_reviews": mask.sum(),
            })

    dd_df = pd.DataFrame(delighters_disappointers)
    for role in ["delighter", "disappointer", "neutral"]:
        print(f"  {role}: {(dd_df['role'] == role).sum()} brand-topic pairs")

    # ============================================================
    # STEP 7: Complaint Concentration Analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 7: Complaint concentration analysis")
    print("=" * 60)

    complaint_rows = []
    for brand, bdf in reviews.groupby("brand"):
        if len(bdf) < MIN_BRAND_REVIEWS:
            continue
        neg = bdf[bdf["sentiment_label"] == "negative"]
        n_neg = len(neg)
        if n_neg < 3:
            complaint_rows.append({
                "brand": brand, "n_negative_reviews": n_neg,
                "complaint_entropy": np.nan, "complaint_entropy_normalized": np.nan,
                "complaint_concentration_label": "insufficient_data",
                "top_complaint_topic": np.nan, "top_complaint_topic_keywords": np.nan,
                "top_complaint_share": np.nan, "top_2_complaint_share": np.nan,
            })
            continue
        dist = neg["dominant_topic"].value_counts(normalize=True)
        ent = entropy(dist.values, base=2)
        norm_ent = ent / np.log2(N_TOPICS) if np.log2(N_TOPICS) > 0 else 0
        top_t = dist.index[0]
        top_s = dist.values[0]
        top2_s = dist.values[:2].sum() if len(dist) >= 2 else top_s
        label = "concentrated" if norm_ent < 0.5 else "moderate" if norm_ent < 0.75 else "diffuse"
        complaint_rows.append({
            "brand": brand, "n_negative_reviews": n_neg,
            "complaint_entropy": ent, "complaint_entropy_normalized": norm_ent,
            "complaint_concentration_label": label,
            "top_complaint_topic": int(top_t),
            "top_complaint_topic_keywords": topic_labels[int(top_t)],
            "top_complaint_share": top_s, "top_2_complaint_share": top2_s,
        })

    complaint_df = pd.DataFrame(complaint_rows)
    for label, count in complaint_df["complaint_concentration_label"].value_counts().items():
        print(f"    {label}: {count} brands")

    # ============================================================
    # STEP 8: Price/Value Perception Diagnostics
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 8: Price/value perception diagnostics")
    print("=" * 60)

    value_pattern = "|".join(VALUE_KEYWORDS)
    reviews["mentions_value"] = reviews["clean_text"].str.lower().str.contains(
        value_pattern, regex=True, na=False,
    )
    print(f"  Reviews mentioning value/price: {reviews['mentions_value'].sum():,} ({reviews['mentions_value'].mean():.1%})")

    value_rows = []
    for brand, bdf in reviews.groupby("brand"):
        if len(bdf) < MIN_BRAND_REVIEWS:
            continue
        val = bdf[bdf["mentions_value"]]
        nval = bdf[~bdf["mentions_value"]]
        overall = bdf["sentiment_compound"].mean()
        v_sent = val["sentiment_compound"].mean() if len(val) > 0 else np.nan
        nv_sent = nval["sentiment_compound"].mean() if len(nval) > 0 else np.nan
        gap = v_sent - nv_sent if not np.isnan(v_sent) and not np.isnan(nv_sent) else np.nan
        neg = bdf[bdf["sentiment_label"] == "negative"]
        pct_neg_val = neg["mentions_value"].mean() if len(neg) > 0 else 0
        if not np.isnan(gap):
            if gap < -0.15:
                vd = "price_driven_negativity"
            elif gap < -0.05:
                vd = "mild_price_concern"
            elif gap > 0.05:
                vd = "perceived_good_value"
            else:
                vd = "price_neutral"
        else:
            vd = "insufficient_data"
        value_rows.append({
            "brand": brand, "overall_sentiment": overall,
            "value_mention_rate": len(val) / len(bdf),
            "value_review_sentiment": v_sent,
            "non_value_review_sentiment": nv_sent,
            "value_sentiment_gap": gap,
            "pct_negative_about_value": pct_neg_val,
            "value_driver_label": vd, "n_value_reviews": len(val),
        })

    value_df = pd.DataFrame(value_rows)
    for label, count in value_df["value_driver_label"].value_counts().items():
        print(f"    {label}: {count} brands")

    # ============================================================
    # STEP 9: Brand-Level Aggregation
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 9: Brand-level aggregation")
    print("=" * 60)

    brand_agg = reviews.groupby("brand").agg(
        n_reviews=("clean_text", "count"),
        n_products=("pd_id", "nunique"),
        avg_rating=("Rating", "mean"),
        rating_std=("Rating", "std"),
        pct_5_star=("Rating", lambda x: (x == 5).mean()),
        pct_1_star=("Rating", lambda x: (x == 1).mean()),
        avg_sentiment=("sentiment_compound", "mean"),
        sentiment_std=("sentiment_compound", "std"),
        avg_sentiment_intensity=("sentiment_intensity", "mean"),
        pct_positive=("sentiment_label", lambda x: (x == "positive").mean()),
        pct_negative=("sentiment_label", lambda x: (x == "negative").mean()),
        pct_neutral=("sentiment_label", lambda x: (x == "neutral").mean()),
        pct_overrated_unhappy=("mismatch_type", lambda x: (x == "overrated_unhappy").mean()),
        pct_underrated_positive=("mismatch_type", lambda x: (x == "underrated_positive").mean()),
        pct_ambivalent=("mismatch_type", lambda x: (x == "ambivalent").mean()),
        total_mismatch_rate=("mismatch_type", lambda x: (x != "aligned").mean()),
        dominant_topic_mode=("dominant_topic", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else -1),
        avg_topic_confidence=("topic_confidence", "mean"),
        avg_helpfulness=("Helpfulness", "mean"),
        avg_price=("price", "mean"),
        median_price=("price", "median"),
        value_mention_rate=("mentions_value", "mean"),
    ).reset_index()

    brand_agg = brand_agg[brand_agg["n_reviews"] >= MIN_BRAND_REVIEWS].copy()
    print(f"  Brands with >= {MIN_BRAND_REVIEWS} reviews: {len(brand_agg)}")

    brand_agg = brand_agg.merge(
        complaint_df[["brand", "n_negative_reviews", "complaint_entropy_normalized",
                      "complaint_concentration_label", "top_complaint_topic",
                      "top_complaint_topic_keywords", "top_complaint_share"]],
        on="brand", how="left",
    )
    brand_agg = brand_agg.merge(
        value_df[["brand", "value_review_sentiment", "value_sentiment_gap",
                  "pct_negative_about_value", "value_driver_label"]],
        on="brand", how="left",
    )

    brand_agg["sentiment_polarization"] = brand_agg["pct_positive"] - brand_agg["pct_negative"]
    brand_agg["rating_sentiment_alignment"] = 1 - brand_agg["total_mismatch_rate"]

    top_del = (
        dd_df[dd_df["role"] == "delighter"]
        .sort_values("sentiment_lift", ascending=False)
        .groupby("brand").first().reset_index()
        [["brand", "topic_id", "topic_keywords", "sentiment_lift"]]
    )
    top_del.columns = ["brand", "top_delighter_topic", "top_delighter_keywords", "top_delighter_lift"]

    top_dis = (
        dd_df[dd_df["role"] == "disappointer"]
        .sort_values("sentiment_lift", ascending=True)
        .groupby("brand").first().reset_index()
        [["brand", "topic_id", "topic_keywords", "sentiment_lift"]]
    )
    top_dis.columns = ["brand", "top_disappointer_topic", "top_disappointer_keywords", "top_disappointer_lift"]

    brand_agg = brand_agg.merge(top_del, on="brand", how="left")
    brand_agg = brand_agg.merge(top_dis, on="brand", how="left")
    brand_agg = brand_agg.sort_values("n_reviews", ascending=False).reset_index(drop=True)
    print(f"  Dashboard shape: {brand_agg.shape}")

    # ============================================================
    # Save CSVs
    # ============================================================
    print("\n" + "=" * 60)
    print("Saving CSVs")
    print("=" * 60)

    brand_agg.to_csv(PROCESSED_DIR / "sephora_brand_health.csv", index=False)
    print(f"  -> sephora_brand_health.csv ({len(brand_agg)} brands)")

    review_export_cols = [
        "pd_id", "brand", "product_name", "category", "price",
        "Rating", "clean_text", "text_length", "submission_date",
        "Helpfulness", "skinTone", "skinType",
        "sentiment_compound", "sentiment_label", "sentiment_intensity",
        "rating_normalized", "rating_sentiment_gap", "mismatch_type",
        "dominant_topic", "topic_confidence", "mentions_value",
    ] + topic_cols
    reviews[review_export_cols].to_csv(PROCESSED_DIR / "sephora_reviews_enriched.csv", index=False)
    print(f"  -> sephora_reviews_enriched.csv ({len(reviews):,} reviews)")

    topic_drivers_df.to_csv(PROCESSED_DIR / "sephora_topic_drivers.csv", index=False)
    print(f"  -> sephora_topic_drivers.csv ({len(topic_drivers_df):,} rows)")

    pd.DataFrame([{"topic_id": k, "keywords": v} for k, v in topic_labels.items()]).to_csv(
        PROCESSED_DIR / "sephora_brand_topic_labels.csv", index=False,
    )
    print(f"  -> sephora_brand_topic_labels.csv ({N_TOPICS} topics)")

    dd_df.to_csv(PROCESSED_DIR / "sephora_delighters_disappointers.csv", index=False)
    print(f"  -> sephora_delighters_disappointers.csv ({len(dd_df):,} rows)")

    complaint_df.to_csv(PROCESSED_DIR / "sephora_complaint_concentration.csv", index=False)
    print(f"  -> sephora_complaint_concentration.csv ({len(complaint_df)} brands)")

    value_df.to_csv(PROCESSED_DIR / "sephora_value_perception.csv", index=False)
    print(f"  -> sephora_value_perception.csv ({len(value_df)} brands)")

    # Save overview HTML
    fig_overview = _build_overview(brand_agg)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_overview.write_html(OUTPUT_DIR / "sephora_brand_health_overview.html")
    print(f"  -> {OUTPUT_DIR}/sephora_brand_health_overview.html")

    print("\n" + "=" * 60)
    print("Pipeline complete. In a notebook, call show_dashboard()")
    print("=" * 60)


# ############################################################
#
#  DASHBOARD FUNCTIONS (work on import — load from CSVs)
#
# ############################################################

def _build_overview(brand_agg):
    """Build the brand health overview scatter."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=brand_agg["avg_sentiment"],
        y=brand_agg["avg_rating"],
        mode="markers",
        marker=dict(
            size=np.clip(brand_agg["n_reviews"] / brand_agg["n_reviews"].max() * 40, 5, 40),
            color=brand_agg["total_mismatch_rate"],
            colorscale="RdYlGn_r",
            colorbar=dict(title="Mismatch<br>Rate"),
            opacity=0.7, line=dict(width=1, color="white"),
        ),
        text=brand_agg["brand"],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Sentiment: %{x:.3f}<br>"
            "Rating: %{y:.2f}<br>"
            "Reviews: %{customdata[0]:,.0f}<br>"
            "Mismatch: %{customdata[1]:.1%}<br>"
            "Complaints: %{customdata[2]}<br>"
            "Value: %{customdata[3]}<extra></extra>"
        ),
        customdata=np.stack([
            brand_agg["n_reviews"],
            brand_agg["total_mismatch_rate"],
            brand_agg["complaint_concentration_label"].fillna("N/A"),
            brand_agg["value_driver_label"].fillna("N/A"),
        ], axis=-1),
    ))
    fig.update_layout(
        title="Sephora Brand Health Overview<br><sup>Size = review volume | Color = mismatch rate</sup>",
        xaxis_title="Avg Sentiment (VADER Compound)",
        yaxis_title="Avg Star Rating",
        template="plotly_white", height=700, width=1100,
    )
    return fig


def build_brand_dashboard(brand_name):
    """
    Build a 12-panel brand health dashboard for a single brand.
    Loads from CSVs if data not already in memory.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _load_data()
    brand_agg = _data["brand_agg"]
    topic_drivers_df = _data["topic_drivers_df"]
    dd_df = _data["dd_df"]

    b = brand_agg[brand_agg["brand"] == brand_name].iloc[0]
    br = _data["brand_reviews"].get(brand_name, pd.DataFrame())
    b_drv = topic_drivers_df[topic_drivers_df["brand"] == brand_name].sort_values(
        "correlation_with_sentiment"
    )
    b_dd = dd_df[dd_df["brand"] == brand_name]

    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=(
            "Star Rating Distribution",
            "Sentiment Distribution",
            "Rating vs Sentiment (Mismatch)",
            "Topic Prevalence",
            "Topic\u2013Sentiment Drivers",
            "Delighters vs Disappointers",
            "Sentiment Over Time",
            "Complaint Concentration",
            "Value Perception",
            "Rating\u2013Sentiment Gap",
            "Review Length vs Sentiment",
            "Helpfulness vs Sentiment",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "pie"}, {"type": "bar"}],
            [{"type": "histogram"}, {"type": "scatter"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    n_br = len(br)

    # R1C1: Star Rating
    stars = br["Rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    fig.add_trace(go.Bar(
        x=stars.index, y=stars.values,
        marker_color=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"],
        text=[f"{v / max(n_br, 1):.0%}" for v in stars.values],
        textposition="outside", showlegend=False,
    ), row=1, col=1)

    # R1C2: Sentiment Distribution
    sent_bins = pd.cut(br["sentiment_compound"], bins=20)
    sent_hist = sent_bins.value_counts().sort_index()
    mids = [(iv.left + iv.right) / 2 for iv in sent_hist.index]
    clrs = ["#e74c3c" if m < -0.05 else "#95a5a6" if m <= 0.05 else "#27ae60" for m in mids]
    fig.add_trace(go.Bar(x=mids, y=sent_hist.values, marker_color=clrs, showlegend=False), row=1, col=2)

    # R1C3: Rating vs Sentiment Scatter (Mismatch)
    samp = br.sample(min(500, n_br), random_state=42)
    cmap = {"aligned": "#bdc3c7", "overrated_unhappy": "#e74c3c",
            "underrated_positive": "#2ecc71", "ambivalent": "#f39c12"}
    for mt, color in cmap.items():
        m = samp["mismatch_type"] == mt
        if m.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=samp.loc[m, "Rating"] + np.random.normal(0, 0.08, m.sum()),
            y=samp.loc[m, "sentiment_compound"],
            mode="markers", marker=dict(color=color, size=4, opacity=0.5),
            name=mt, legendgroup="mm", showlegend=True,
        ), row=1, col=3)

    # R2C1: Topic Prevalence
    tp = br["dominant_topic"].value_counts(normalize=True).sort_index()
    fig.add_trace(go.Bar(
        x=[f"T{i}" for i in tp.index], y=tp.values,
        marker_color="#3498db", showlegend=False,
        text=[f"{v:.0%}" for v in tp.values], textposition="outside",
    ), row=2, col=1)

    # R2C2: Topic–Sentiment Drivers
    if len(b_drv) > 0:
        fig.add_trace(go.Bar(
            y=[f"T{int(r['topic_id'])}" for _, r in b_drv.iterrows()],
            x=b_drv["correlation_with_sentiment"], orientation="h",
            marker_color=["#27ae60" if v > 0 else "#e74c3c" for v in b_drv["correlation_with_sentiment"]],
            showlegend=False,
        ), row=2, col=2)

    # R2C3: Delighters vs Disappointers
    if len(b_dd) > 0:
        bdd_s = b_dd.sort_values("sentiment_lift")
        fig.add_trace(go.Bar(
            y=[f"T{int(r['topic_id'])}" for _, r in bdd_s.iterrows()],
            x=bdd_s["sentiment_lift"], orientation="h",
            marker_color=[
                "#27ae60" if r["role"] == "delighter"
                else "#e74c3c" if r["role"] == "disappointer"
                else "#95a5a6"
                for _, r in bdd_s.iterrows()
            ],
            showlegend=False,
        ), row=2, col=3)

    # R3C1: Sentiment Over Time
    br_copy = br.copy()
    br_copy["submission_date"] = pd.to_datetime(br_copy["submission_date"], errors="coerce")
    brt = br_copy.dropna(subset=["submission_date"])
    if len(brt) > 0:
        brt = brt.copy()
        brt["month"] = brt["submission_date"].dt.to_period("M").dt.to_timestamp()
        mo = brt.groupby("month").agg(
            avg_sent=("sentiment_compound", "mean"),
            avg_rat=("Rating", "mean"),
            n=("Rating", "count"),
        ).reset_index()
        mo = mo[mo["n"] >= 3]
        if len(mo) > 0:
            fig.add_trace(go.Scatter(
                x=mo["month"], y=mo["avg_sent"], mode="lines+markers",
                name="Sentiment", line=dict(color="#3498db", width=2),
                legendgroup="time", showlegend=True,
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=mo["month"], y=(mo["avg_rat"] - 3) / 2, mode="lines+markers",
                name="Rating (norm)", line=dict(color="#e67e22", width=2, dash="dash"),
                legendgroup="time", showlegend=True,
            ), row=3, col=1)

    # R3C2: Complaint Concentration Pie
    neg_br = br[br["sentiment_label"] == "negative"]
    if len(neg_br) >= 3:
        nt = neg_br["dominant_topic"].value_counts().head(6)
        fig.add_trace(go.Pie(
            labels=[f"T{t}" for t in nt.index], values=nt.values,
            hole=0.4, showlegend=False, textinfo="label+percent",
        ), row=3, col=2)

    # R3C3: Value Perception
    vr = br[br["mentions_value"] == True]
    nvr = br[br["mentions_value"] == False]
    if len(vr) > 0 and len(nvr) > 0:
        fig.add_trace(go.Bar(
            x=["Value Reviews", "Non-Value Reviews"],
            y=[vr["sentiment_compound"].mean(), nvr["sentiment_compound"].mean()],
            marker_color=["#e74c3c", "#3498db"],
            text=[f"{vr['sentiment_compound'].mean():.3f}", f"{nvr['sentiment_compound'].mean():.3f}"],
            textposition="outside", showlegend=False,
        ), row=3, col=3)

    # R4C1: Rating–Sentiment Gap
    fig.add_trace(go.Histogram(
        x=br["rating_sentiment_gap"], nbinsx=30,
        marker_color="#8e44ad", showlegend=False, opacity=0.7,
    ), row=4, col=1)

    # R4C2: Review Length vs Sentiment
    s2 = br.sample(min(500, n_br), random_state=42)
    fig.add_trace(go.Scatter(
        x=s2["text_length"], y=s2["sentiment_compound"],
        mode="markers", marker=dict(color="#3498db", size=3, opacity=0.4),
        showlegend=False,
    ), row=4, col=2)

    # R4C3: Helpfulness vs Sentiment
    fig.add_trace(go.Scatter(
        x=s2["Helpfulness"], y=s2["sentiment_compound"],
        mode="markers", marker=dict(color="#e67e22", size=3, opacity=0.4),
        showlegend=False,
    ), row=4, col=3)

    # KPI header
    del_kw = b.get("top_delighter_keywords", "\u2014")
    dis_kw = b.get("top_disappointer_keywords", "\u2014")
    if pd.isna(del_kw):
        del_kw = "\u2014"
    else:
        # Shorten to first 5 keywords
        del_kw = ", ".join(str(del_kw).split(", ")[:5])
    if pd.isna(dis_kw):
        dis_kw = "\u2014"
    else:
        dis_kw = ", ".join(str(dis_kw).split(", ")[:5])

    kpi = (
        f"<b style='font-size:16px'>{brand_name}</b><br>"
        f"<span style='font-size:10px; color:#555;'>"
        f"Reviews: {b['n_reviews']:,.0f} &nbsp;|&nbsp; "
        f"Products: {b['n_products']:.0f} &nbsp;|&nbsp; "
        f"Rating: {b['avg_rating']:.2f} &nbsp;|&nbsp; "
        f"Sentiment: {b['avg_sentiment']:.3f} &nbsp;|&nbsp; "
        f"Mismatch: {b['total_mismatch_rate']:.1%} &nbsp;|&nbsp; "
        f"Value: {b.get('value_driver_label', 'N/A')}"
        f"</span><br>"
        f"<span style='font-size:10px; color:#555;'>"
        f"Complaints: {b.get('complaint_concentration_label', 'N/A')} &nbsp;|&nbsp; "
        f"Delighter: {del_kw}"
        f"</span><br>"
        f"<span style='font-size:10px; color:#555;'>"
        f"Disappointer: {dis_kw}"
        f"</span>"
    )

    fig.update_layout(
        height=1700, width=1400,
        title=dict(text=kpi, font=dict(size=12), x=0.01, y=0.99),
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.02, x=0.5, xanchor="center"),
        margin=dict(t=130),
    )
    fig.update_xaxes(title_text="Stars", row=1, col=1)
    fig.update_xaxes(title_text="Sentiment", row=1, col=2)
    fig.update_xaxes(title_text="Star Rating", row=1, col=3)
    fig.update_yaxes(title_text="Sentiment", row=1, col=3)
    fig.update_xaxes(title_text="Corr w/ Sentiment", row=2, col=2)
    fig.update_xaxes(title_text="Sentiment Lift", row=2, col=3)
    fig.update_yaxes(title_text="Sentiment", row=3, col=1)
    fig.update_xaxes(title_text="Gap (Rating \u2212 Sentiment)", row=4, col=1)
    fig.update_xaxes(title_text="Review Length (chars)", row=4, col=2)
    fig.update_yaxes(title_text="Sentiment", row=4, col=2)
    fig.update_xaxes(title_text="Helpfulness", row=4, col=3)
    fig.update_yaxes(title_text="Sentiment", row=4, col=3)

    return fig


def show_overview():
    """Show the brand health overview scatter in a notebook."""
    _load_data()
    fig = _build_overview(_data["brand_agg"])
    fig.show()


def show_dashboard():
    """
    Launch the interactive brand health dashboard in a notebook.
    Dropdown toggles between ALL qualifying Sephora brands.

    Usage:
        from sephora_brand_health import show_dashboard
        show_dashboard()
    """
    import ipywidgets as widgets
    from IPython.display import display, HTML

    _load_data()
    brands = _data["dashboard_brands"]

    display(HTML(
        "<h2 style='text-align:center; color:#2C3E50;'>"
        "Sephora Brand Health Dashboard</h2>"
        "<p style='text-align:center; color:#7f8c8d;'>"
        f"Select from {len(brands)} brands. "
        "Dashboard rebuilds on each selection.</p>"
    ))

    dropdown = widgets.Dropdown(
        options=brands,
        value=brands[0],
        description="Brand:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="450px"),
    )

    output = widgets.Output()

    def on_change(change):
        with output:
            output.clear_output(wait=True)
            build_brand_dashboard(change["new"]).show()

    dropdown.observe(on_change, names="value")
    display(dropdown, output)

    with output:
        build_brand_dashboard(brands[0]).show()


# ============================================================
# Script entry point
# ============================================================
if __name__ == "__main__":
    run_pipeline()