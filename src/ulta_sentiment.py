"""
Brand Health & Sentiment Analysis — Ulta
=========================================

Ulta-specific enrichments vs Sephora pipeline:
  - Headline sentiment (VADER on review headlines)
  - Combined sentiment (0.7 × body + 0.3 × headline)
  - Verified buyer segmentation (all / verified / unverified)
  - Dashboard toggles: Brand × Verified Segment

Usage:
  As script (runs full pipeline):
    python src/ulta_sentiment.py

  As import (dashboard only, loads from pre-built CSVs):
    from ulta_sentiment import show_dashboard
    show_dashboard()

Pipeline (Steps 1–9, runs via run_pipeline()):
  1. Text preprocessing & review filtering
  2. Sentiment scoring (VADER — body + headline + combined)
  3. Rating–sentiment mismatch detection
  4. Topic modeling (NMF on TF-IDF, body + headline concatenated)
  5. Topic–sentiment linkage (drivers)
  6. Delighters vs. disappointers per brand
  7. Complaint concentration analysis
  8. Price/value perception diagnostics
  9. Brand-level aggregation → CSVs (with verified segmentation)

Dashboard (Step 10, runs via show_dashboard()):
  Interactive Plotly + ipywidgets brand toggle × verified toggle.

Inputs:
  data/processed/ulta_products.csv
  data/processed/ulta_reviews.csv

Outputs (CSVs):
  data/processed/ulta_brand_health.csv
  data/processed/ulta_reviews_enriched.csv
  data/processed/ulta_topic_drivers.csv
  data/processed/ulta_brand_topic_labels.csv
  data/processed/ulta_delighters_disappointers.csv
  data/processed/ulta_complaint_concentration.csv
  data/processed/ulta_value_perception.csv
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
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
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
        return

    import time
    t0 = time.time()
    print("Loading Ulta dashboard data from CSVs...")

    _data["brand_agg"] = pd.read_csv(PROCESSED_DIR / "ulta_brand_health.csv")

    review_cols = [
        "brand", "Rating", "sentiment_compound", "headline_sentiment",
        "combined_sentiment", "sentiment_label",
        "mismatch_type", "dominant_topic", "submission_date",
        "mentions_value", "rating_sentiment_gap", "text_length",
        "helpful_votes", "is_verified_buyer",
    ]
    print("  Loading reviews (subset columns)...")
    _data["reviews"] = pd.read_csv(
        PROCESSED_DIR / "ulta_reviews_enriched.csv",
        usecols=review_cols,
    )

    print("  Indexing by brand...")
    _data["brand_reviews"] = dict(tuple(_data["reviews"].groupby("brand")))

    _data["topic_drivers_df"] = pd.read_csv(PROCESSED_DIR / "ulta_topic_drivers.csv")
    _data["dd_df"] = pd.read_csv(PROCESSED_DIR / "ulta_delighters_disappointers.csv")

    tl = pd.read_csv(PROCESSED_DIR / "ulta_brand_topic_labels.csv")
    _data["topic_labels"] = dict(zip(tl["topic_id"], tl["keywords"]))

    _data["dashboard_brands"] = sorted(
        _data["brand_agg"]["brand"].unique().tolist()
    )

    elapsed = time.time() - t0
    print(f"  Loaded {len(_data['dashboard_brands'])} brands, "
          f"{len(_data['reviews']):,} reviews in {elapsed:.1f}s")


# ============================================================
# PIPELINE: run_pipeline()
# ============================================================
def run_pipeline():
    """Run the full Ulta brand health pipeline (Steps 1–9) and save CSVs."""

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

    products = pd.read_csv(PROCESSED_DIR / "ulta_products.csv")
    reviews = pd.read_csv(PROCESSED_DIR / "ulta_reviews.csv")

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
    reviews["clean_headline"] = reviews["headline"].apply(clean_text)
    reviews["text_length"] = reviews["clean_text"].str.len()

    before = len(reviews)
    reviews = reviews[reviews["text_length"] >= MIN_REVIEW_LENGTH].copy()
    print(f"  After filtering short (<{MIN_REVIEW_LENGTH} chars): {len(reviews):,} "
          f"({before - len(reviews):,} removed)")

    # Parse submission time (Ulta uses epoch milliseconds)
    reviews["submission_date"] = pd.to_datetime(
        pd.to_numeric(reviews["SubmissionTime"], errors="coerce"),
        unit="ms", errors="coerce",
    )

    # Ensure boolean verified buyer flag
    reviews["is_verified_buyer"] = reviews["is_verified_buyer"].fillna(False).astype(bool)

    print(f"  Unique brands: {reviews['brand'].nunique()}")
    print(f"  Unique products: {reviews['pd_id'].nunique()}")
    print(f"  Verified buyers: {reviews['is_verified_buyer'].sum():,} "
          f"({reviews['is_verified_buyer'].mean():.1%})")
    print(f"  Reviews with headlines: {(reviews['clean_headline'].str.len() > 0).sum():,}")

    # ============================================================
    # STEP 2: Sentiment Scoring (VADER — Body + Headline + Combined)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: Sentiment scoring (VADER — body + headline + combined)")
    print("=" * 60)

    analyzer = SentimentIntensityAnalyzer()

    def vader_compound(text):
        if not text:
            return 0.0
        return analyzer.polarity_scores(text)["compound"]

    # Body sentiment
    reviews["sentiment_compound"] = reviews["clean_text"].apply(vader_compound)

    # Headline sentiment (Ulta-specific)
    reviews["headline_sentiment"] = reviews["clean_headline"].apply(vader_compound)

    # Combined: 70% body + 30% headline (headline is more emotionally concentrated)
    has_headline = reviews["clean_headline"].str.len() > 0
    reviews["combined_sentiment"] = reviews["sentiment_compound"]  # default to body
    reviews.loc[has_headline, "combined_sentiment"] = (
        0.7 * reviews.loc[has_headline, "sentiment_compound"]
        + 0.3 * reviews.loc[has_headline, "headline_sentiment"]
    )

    # Sentiment labels (based on combined)
    reviews["sentiment_label"] = pd.cut(
        reviews["combined_sentiment"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"],
    )
    reviews["sentiment_intensity"] = reviews["combined_sentiment"].abs()

    # Headline-body divergence
    reviews["headline_body_gap"] = reviews["headline_sentiment"] - reviews["sentiment_compound"]

    print(f"  Body sentiment   — mean: {reviews['sentiment_compound'].mean():.3f}")
    print(f"  Headline sentiment — mean: {reviews['headline_sentiment'].mean():.3f}")
    print(f"  Combined sentiment — mean: {reviews['combined_sentiment'].mean():.3f}")
    print(f"  Positive: {(reviews['sentiment_label'] == 'positive').mean():.1%}")
    print(f"  Neutral:  {(reviews['sentiment_label'] == 'neutral').mean():.1%}")
    print(f"  Negative: {(reviews['sentiment_label'] == 'negative').mean():.1%}")

    # Verified vs unverified sentiment
    for seg, mask in [("Verified", reviews["is_verified_buyer"]),
                      ("Unverified", ~reviews["is_verified_buyer"])]:
        if mask.sum() > 0:
            print(f"  {seg} — n={mask.sum():,}, "
                  f"body={reviews.loc[mask, 'sentiment_compound'].mean():.3f}, "
                  f"headline={reviews.loc[mask, 'headline_sentiment'].mean():.3f}, "
                  f"combined={reviews.loc[mask, 'combined_sentiment'].mean():.3f}")

    # ============================================================
    # STEP 3: Rating–Sentiment Mismatch Detection
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: Rating-sentiment mismatch detection")
    print("=" * 60)

    reviews["rating_normalized"] = (reviews["Rating"] - 3) / 2
    reviews["rating_sentiment_gap"] = reviews["rating_normalized"] - reviews["combined_sentiment"]

    reviews["mismatch_type"] = "aligned"
    reviews.loc[
        (reviews["Rating"] >= 4) & (reviews["combined_sentiment"] < -0.05),
        "mismatch_type",
    ] = "overrated_unhappy"
    reviews.loc[
        (reviews["Rating"] <= 2) & (reviews["combined_sentiment"] > 0.05),
        "mismatch_type",
    ] = "underrated_positive"
    reviews.loc[
        (reviews["Rating"] == 3) & (reviews["combined_sentiment"].abs() > 0.5),
        "mismatch_type",
    ] = "ambivalent"

    for mt, count in reviews["mismatch_type"].value_counts().items():
        print(f"    {mt}: {count:,} ({count / len(reviews):.1%})")

    # ============================================================
    # STEP 4: Topic Modeling (NMF on TF-IDF — body + headline)
    # ============================================================
    print("\n" + "=" * 60)
    print(f"STEP 4: Topic modeling (NMF, {N_TOPICS} topics)")
    print("=" * 60)

    # Concatenate headline + body for richer topic signal
    reviews["topic_text"] = (
        reviews["clean_headline"].fillna("") + " " + reviews["clean_text"]
    ).str.strip()

    tfidf = TfidfVectorizer(
        max_features=8000, min_df=10, max_df=0.85,
        ngram_range=(1, 2), stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    tfidf_matrix = tfidf.fit_transform(reviews["topic_text"])
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

    print("\n  Global topic-sentiment correlations (combined):")
    for col in topic_cols:
        corr = reviews[col].corr(reviews["combined_sentiment"])
        tid = int(col.split("_")[1])
        print(f"    Topic {tid}: {corr:+.3f} — {topic_labels[tid][:50]}")

    brand_topic_drivers = []
    for brand, bdf in reviews.groupby("brand"):
        if len(bdf) < MIN_BRAND_REVIEWS:
            continue
        for i in range(N_TOPICS):
            corr = bdf[f"topic_{i}"].corr(bdf["combined_sentiment"])
            avg_sent = bdf.loc[bdf["dominant_topic"] == i, "combined_sentiment"].mean()
            prev = (bdf["dominant_topic"] == i).mean()
            brand_topic_drivers.append({
                "brand": brand, "topic_id": i,
                "topic_keywords": topic_labels[i],
                "correlation_with_sentiment": corr if not np.isnan(corr) else 0,
                "avg_sentiment_when_dominant": avg_sent if not np.isnan(avg_sent) else 0,
                "topic_prevalence": prev, "n_reviews": len(bdf),
            })

    topic_drivers_df = pd.DataFrame(brand_topic_drivers)
    print(f"\n  Brand × topic driver matrix: {topic_drivers_df.shape}")

    # ============================================================
    # STEP 6: Delighters vs. Disappointers
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 6: Delighters vs. disappointers")
    print("=" * 60)

    delighters_disappointers = []
    for brand, bdf in reviews.groupby("brand"):
        if len(bdf) < MIN_BRAND_REVIEWS:
            continue
        brand_avg = bdf["combined_sentiment"].mean()
        for i in range(N_TOPICS):
            mask = bdf["dominant_topic"] == i
            if mask.sum() < 5:
                continue
            topic_sent = bdf.loc[mask, "combined_sentiment"].mean()
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
                "topic_sentiment_std": bdf.loc[mask, "combined_sentiment"].std(),
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
    # Check both body and headline for value mentions
    reviews["mentions_value"] = (
        reviews["clean_text"].str.lower().str.contains(value_pattern, regex=True, na=False)
        | reviews["clean_headline"].str.lower().str.contains(value_pattern, regex=True, na=False)
    )
    print(f"  Reviews mentioning value/price: {reviews['mentions_value'].sum():,} "
          f"({reviews['mentions_value'].mean():.1%})")

    value_rows = []
    for brand, bdf in reviews.groupby("brand"):
        if len(bdf) < MIN_BRAND_REVIEWS:
            continue
        val = bdf[bdf["mentions_value"]]
        nval = bdf[~bdf["mentions_value"]]
        overall = bdf["combined_sentiment"].mean()
        v_sent = val["combined_sentiment"].mean() if len(val) > 0 else np.nan
        nv_sent = nval["combined_sentiment"].mean() if len(nval) > 0 else np.nan
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
    # STEP 9: Brand-Level Aggregation (with Verified Segmentation)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 9: Brand-level aggregation (all + verified + unverified)")
    print("=" * 60)

    def aggregate_segment(df, segment_label):
        """Aggregate review metrics for a subset of reviews."""
        agg = df.groupby("brand").agg(
            n_reviews=("clean_text", "count"),
            n_products=("pd_id", "nunique"),
            avg_rating=("Rating", "mean"),
            rating_std=("Rating", "std"),
            pct_5_star=("Rating", lambda x: (x == 5).mean()),
            pct_1_star=("Rating", lambda x: (x == 1).mean()),
            avg_sentiment=("combined_sentiment", "mean"),
            avg_body_sentiment=("sentiment_compound", "mean"),
            avg_headline_sentiment=("headline_sentiment", "mean"),
            sentiment_std=("combined_sentiment", "std"),
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
            avg_helpful_votes=("helpful_votes", "mean"),
            avg_price=("price", "mean"),
            median_price=("price", "median"),
            value_mention_rate=("mentions_value", "mean"),
            avg_headline_body_gap=("headline_body_gap", "mean"),
            pct_verified=("is_verified_buyer", "mean"),
        ).reset_index()

        agg = agg[agg["n_reviews"] >= MIN_BRAND_REVIEWS].copy()
        agg["verified_segment"] = segment_label

        agg["sentiment_polarization"] = agg["pct_positive"] - agg["pct_negative"]
        agg["rating_sentiment_alignment"] = 1 - agg["total_mismatch_rate"]

        return agg

    # Build three segments
    agg_all = aggregate_segment(reviews, "all")
    agg_ver = aggregate_segment(reviews[reviews["is_verified_buyer"]], "verified")
    agg_unv = aggregate_segment(reviews[~reviews["is_verified_buyer"]], "unverified")

    brand_agg = pd.concat([agg_all, agg_ver, agg_unv], ignore_index=True)

    print(f"  All:        {len(agg_all)} brands")
    print(f"  Verified:   {len(agg_ver)} brands")
    print(f"  Unverified: {len(agg_unv)} brands")
    print(f"  Total rows: {len(brand_agg)}")

    # Merge complaint + value + delighter/disappointer (on "all" segment, then broadcast)
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
    brand_agg = brand_agg.sort_values(["brand", "verified_segment"]).reset_index(drop=True)
    print(f"  Final brand_agg shape: {brand_agg.shape}")

    # ============================================================
    # Save CSVs
    # ============================================================
    print("\n" + "=" * 60)
    print("Saving CSVs")
    print("=" * 60)

    brand_agg.to_csv(PROCESSED_DIR / "ulta_brand_health.csv", index=False)
    print(f"  -> ulta_brand_health.csv ({len(brand_agg)} rows)")

    review_export_cols = [
        "pd_id", "brand", "product_name", "category", "price",
        "Rating", "clean_text", "clean_headline", "text_length",
        "submission_date", "helpful_votes", "is_verified_buyer",
        "sentiment_compound", "headline_sentiment", "combined_sentiment",
        "headline_body_gap", "sentiment_label", "sentiment_intensity",
        "rating_normalized", "rating_sentiment_gap", "mismatch_type",
        "dominant_topic", "topic_confidence", "mentions_value",
    ] + topic_cols
    reviews[review_export_cols].to_csv(PROCESSED_DIR / "ulta_reviews_enriched.csv", index=False)
    print(f"  -> ulta_reviews_enriched.csv ({len(reviews):,} reviews)")

    topic_drivers_df.to_csv(PROCESSED_DIR / "ulta_topic_drivers.csv", index=False)
    print(f"  -> ulta_topic_drivers.csv ({len(topic_drivers_df):,} rows)")

    pd.DataFrame([{"topic_id": k, "keywords": v} for k, v in topic_labels.items()]).to_csv(
        PROCESSED_DIR / "ulta_brand_topic_labels.csv", index=False,
    )
    print(f"  -> ulta_brand_topic_labels.csv ({N_TOPICS} topics)")

    dd_df.to_csv(PROCESSED_DIR / "ulta_delighters_disappointers.csv", index=False)
    print(f"  -> ulta_delighters_disappointers.csv ({len(dd_df):,} rows)")

    complaint_df.to_csv(PROCESSED_DIR / "ulta_complaint_concentration.csv", index=False)
    print(f"  -> ulta_complaint_concentration.csv ({len(complaint_df)} brands)")

    value_df.to_csv(PROCESSED_DIR / "ulta_value_perception.csv", index=False)
    print(f"  -> ulta_value_perception.csv ({len(value_df)} brands)")

    # Save overview HTML
    fig_overview = _build_overview(
        brand_agg[brand_agg["verified_segment"] == "all"]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_overview.write_html(OUTPUT_DIR / "ulta_brand_health_overview.html")
    print(f"  -> {OUTPUT_DIR}/ulta_brand_health_overview.html")

    print("\n" + "=" * 60)
    print("Pipeline complete. In a notebook, call show_dashboard()")
    print("=" * 60)


# ############################################################
#
#  DASHBOARD FUNCTIONS
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
            "Combined Sentiment: %{x:.3f}<br>"
            "Rating: %{y:.2f}<br>"
            "Reviews: %{customdata[0]:,.0f}<br>"
            "Mismatch: %{customdata[1]:.1%}<br>"
            "Verified%: %{customdata[2]:.1%}<extra></extra>"
        ),
        customdata=np.stack([
            brand_agg["n_reviews"],
            brand_agg["total_mismatch_rate"],
            brand_agg["pct_verified"],
        ], axis=-1),
    ))
    fig.update_layout(
        title="Ulta Brand Health Overview<br>"
              "<sup>Size = review volume | Color = mismatch rate</sup>",
        xaxis_title="Avg Combined Sentiment (0.7×body + 0.3×headline)",
        yaxis_title="Avg Star Rating",
        template="plotly_white", height=700, width=1100,
    )
    return fig


def build_brand_dashboard(brand_name, verified_segment="all"):
    """
    Build a 12-panel brand health dashboard for a single brand.

    Ulta-specific panels vs Sephora:
      - R1C2: Body vs Headline sentiment overlay
      - R3C3: Verified vs Unverified sentiment comparison
      - R4C2: Headline vs Body scatter (replaces Review Length vs Sentiment)
      - R4C3: Helpful Votes vs Sentiment

    Parameters
    ----------
    brand_name : str
    verified_segment : str
        "all", "verified", or "unverified"
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _load_data()
    brand_agg = _data["brand_agg"]
    topic_drivers_df = _data["topic_drivers_df"]
    dd_df = _data["dd_df"]

    # Get brand-level row for this segment
    mask = (brand_agg["brand"] == brand_name) & (brand_agg["verified_segment"] == verified_segment)
    if mask.sum() == 0:
        # Fall back to "all"
        mask = (brand_agg["brand"] == brand_name) & (brand_agg["verified_segment"] == "all")
    b = brand_agg[mask].iloc[0]

    # Filter reviews by brand + verified segment
    br = _data["brand_reviews"].get(brand_name, pd.DataFrame())
    if len(br) > 0 and verified_segment == "verified":
        br = br[br["is_verified_buyer"] == True]
    elif len(br) > 0 and verified_segment == "unverified":
        br = br[br["is_verified_buyer"] == False]
    # "all" keeps everything

    b_drv = topic_drivers_df[topic_drivers_df["brand"] == brand_name].sort_values(
        "correlation_with_sentiment"
    )
    b_dd = dd_df[dd_df["brand"] == brand_name]

    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=(
            "Star Rating Distribution",
            "Body vs Headline Sentiment",
            "Rating vs Sentiment (Mismatch)",
            "Topic Prevalence",
            "Topic\u2013Sentiment Drivers",
            "Delighters vs Disappointers",
            "Sentiment Over Time",
            "Complaint Concentration",
            "Verified vs Unverified",
            "Rating\u2013Sentiment Gap",
            "Headline vs Body Sentiment",
            "Helpful Votes vs Sentiment",
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
    if n_br == 0:
        fig.update_layout(
            height=400, title=f"No {verified_segment} reviews for {brand_name}",
            template="plotly_white",
        )
        return fig

    # R1C1: Star Rating
    stars = br["Rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    fig.add_trace(go.Bar(
        x=stars.index, y=stars.values,
        marker_color=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"],
        text=[f"{v / max(n_br, 1):.0%}" for v in stars.values],
        textposition="outside", showlegend=False,
    ), row=1, col=1)

    # R1C2: Body vs Headline Sentiment (overlaid histograms)
    fig.add_trace(go.Histogram(
        x=br["sentiment_compound"], nbinsx=25, name="Body",
        marker_color="rgba(52, 152, 219, 0.5)", showlegend=True,
        legendgroup="sent",
    ), row=1, col=2)
    fig.add_trace(go.Histogram(
        x=br["headline_sentiment"], nbinsx=25, name="Headline",
        marker_color="rgba(231, 76, 60, 0.5)", showlegend=True,
        legendgroup="sent",
    ), row=1, col=2)

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
            y=samp.loc[m, "combined_sentiment"],
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
            marker_color=["#27ae60" if v > 0 else "#e74c3c"
                          for v in b_drv["correlation_with_sentiment"]],
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
    br_ts = br.copy()
    br_ts["submission_date"] = pd.to_datetime(br_ts["submission_date"], errors="coerce")
    brt = br_ts.dropna(subset=["submission_date"])
    if len(brt) > 0:
        brt = brt.copy()
        brt["month"] = brt["submission_date"].dt.to_period("M").dt.to_timestamp()
        mo = brt.groupby("month").agg(
            avg_body=("sentiment_compound", "mean"),
            avg_headline=("headline_sentiment", "mean"),
            avg_combined=("combined_sentiment", "mean"),
            n=("Rating", "count"),
        ).reset_index()
        mo = mo[mo["n"] >= 3]
        if len(mo) > 0:
            fig.add_trace(go.Scatter(
                x=mo["month"], y=mo["avg_combined"], mode="lines+markers",
                name="Combined", line=dict(color="#3498db", width=2),
                legendgroup="time", showlegend=True,
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=mo["month"], y=mo["avg_headline"], mode="lines+markers",
                name="Headline", line=dict(color="#e74c3c", width=2, dash="dash"),
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

    # R3C3: Verified vs Unverified Comparison (always uses full brand data)
    br_full = _data["brand_reviews"].get(brand_name, pd.DataFrame())
    if len(br_full) > 0:
        ver = br_full[br_full["is_verified_buyer"] == True]
        unv = br_full[br_full["is_verified_buyer"] == False]
        labels, vals, colors = [], [], []
        if len(ver) > 0:
            labels.append(f"Verified (n={len(ver):,})")
            vals.append(ver["combined_sentiment"].mean())
            colors.append("#27ae60")
        if len(unv) > 0:
            labels.append(f"Unverified (n={len(unv):,})")
            vals.append(unv["combined_sentiment"].mean())
            colors.append("#e74c3c")
        if labels:
            fig.add_trace(go.Bar(
                x=labels, y=vals, marker_color=colors,
                text=[f"{v:.3f}" for v in vals],
                textposition="outside", showlegend=False,
            ), row=3, col=3)

    # R4C1: Rating–Sentiment Gap
    fig.add_trace(go.Histogram(
        x=br["rating_sentiment_gap"], nbinsx=30,
        marker_color="#8e44ad", showlegend=False, opacity=0.7,
    ), row=4, col=1)

    # R4C2: Headline vs Body Sentiment scatter (Ulta-specific)
    s2 = br.sample(min(500, n_br), random_state=42)
    fig.add_trace(go.Scatter(
        x=s2["sentiment_compound"], y=s2["headline_sentiment"],
        mode="markers", marker=dict(color="#3498db", size=3, opacity=0.4),
        showlegend=False,
    ), row=4, col=2)
    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[-1, 1], y=[-1, 1], mode="lines",
        line=dict(color="gray", width=1, dash="dash"),
        showlegend=False,
    ), row=4, col=2)

    # R4C3: Helpful Votes vs Sentiment
    fig.add_trace(go.Scatter(
        x=s2["helpful_votes"], y=s2["combined_sentiment"],
        mode="markers", marker=dict(color="#e67e22", size=3, opacity=0.4),
        showlegend=False,
    ), row=4, col=3)

    # KPI header
    del_kw = b.get("top_delighter_keywords", "\u2014")
    dis_kw = b.get("top_disappointer_keywords", "\u2014")
    if pd.isna(del_kw):
        del_kw = "\u2014"
    else:
        del_kw = ", ".join(str(del_kw).split(", ")[:5])
    if pd.isna(dis_kw):
        dis_kw = "\u2014"
    else:
        dis_kw = ", ".join(str(dis_kw).split(", ")[:5])

    seg_label = verified_segment.upper()
    kpi = (
        f"<b style='font-size:16px'>{brand_name}</b> "
        f"<span style='font-size:12px; color:#2980b9;'>({seg_label})</span><br>"
        f"<span style='font-size:10px; color:#555;'>"
        f"Reviews: {b['n_reviews']:,.0f} &nbsp;|&nbsp; "
        f"Products: {b['n_products']:.0f} &nbsp;|&nbsp; "
        f"Rating: {b['avg_rating']:.2f} &nbsp;|&nbsp; "
        f"Combined Sent: {b['avg_sentiment']:.3f} &nbsp;|&nbsp; "
        f"Body: {b['avg_body_sentiment']:.3f} &nbsp;|&nbsp; "
        f"Headline: {b['avg_headline_sentiment']:.3f}"
        f"</span><br>"
        f"<span style='font-size:10px; color:#555;'>"
        f"Mismatch: {b['total_mismatch_rate']:.1%} &nbsp;|&nbsp; "
        f"Verified: {b.get('pct_verified', 0):.1%} &nbsp;|&nbsp; "
        f"Value: {b.get('value_driver_label', 'N/A')} &nbsp;|&nbsp; "
        f"Complaints: {b.get('complaint_concentration_label', 'N/A')}"
        f"</span><br>"
        f"<span style='font-size:10px; color:#555;'>"
        f"Delighter: {del_kw} &nbsp;|&nbsp; "
        f"Disappointer: {dis_kw}"
        f"</span>"
    )

    fig.update_layout(
        height=1700, width=1400,
        title=dict(text=kpi, font=dict(size=12), x=0.01, y=0.99),
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.02, x=0.5, xanchor="center"),
        margin=dict(t=140),
    )
    fig.update_xaxes(title_text="Stars", row=1, col=1)
    fig.update_xaxes(title_text="Sentiment Score", row=1, col=2)
    fig.update_xaxes(title_text="Star Rating", row=1, col=3)
    fig.update_yaxes(title_text="Combined Sentiment", row=1, col=3)
    fig.update_xaxes(title_text="Corr w/ Sentiment", row=2, col=2)
    fig.update_xaxes(title_text="Sentiment Lift", row=2, col=3)
    fig.update_yaxes(title_text="Combined Sentiment", row=3, col=1)
    fig.update_xaxes(title_text="Avg Combined Sentiment", row=3, col=3)
    fig.update_xaxes(title_text="Gap (Rating \u2212 Sentiment)", row=4, col=1)
    fig.update_xaxes(title_text="Body Sentiment", row=4, col=2)
    fig.update_yaxes(title_text="Headline Sentiment", row=4, col=2)
    fig.update_xaxes(title_text="Helpful Votes", row=4, col=3)
    fig.update_yaxes(title_text="Combined Sentiment", row=4, col=3)

    return fig


def show_overview():
    """Show the brand health overview scatter in a notebook."""
    _load_data()
    agg_all = _data["brand_agg"][_data["brand_agg"]["verified_segment"] == "all"]
    fig = _build_overview(agg_all)
    fig.show()


def show_dashboard():
    """
    Launch the interactive Ulta brand health dashboard in a notebook.
    Two dropdowns: Brand × Verified Segment.

    Usage:
        from ulta_sentiment import show_dashboard
        show_dashboard()
    """
    import ipywidgets as widgets
    from IPython.display import display, HTML

    _load_data()
    brands = _data["dashboard_brands"]
    segments = ["all", "verified", "unverified"]

    display(HTML(
        "<h2 style='text-align:center; color:#2C3E50;'>"
        "Ulta Brand Health Dashboard</h2>"
        "<p style='text-align:center; color:#7f8c8d;'>"
        f"Select from {len(brands)} brands × 3 verified segments. "
        "Dashboard rebuilds on each selection.</p>"
    ))

    brand_dropdown = widgets.Dropdown(
        options=brands,
        value=brands[0],
        description="Brand:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="450px"),
    )

    segment_toggle = widgets.ToggleButtons(
        options=[("All Reviews", "all"),
                 ("Verified Buyers", "verified"),
                 ("Unverified", "unverified")],
        value="all",
        description="Segment:",
        style={"description_width": "initial", "button_width": "140px"},
    )

    output = widgets.Output()

    def on_change(_):
        with output:
            output.clear_output(wait=True)
            build_brand_dashboard(
                brand_dropdown.value, segment_toggle.value
            ).show()

    brand_dropdown.observe(on_change, names="value")
    segment_toggle.observe(on_change, names="value")
    display(brand_dropdown, segment_toggle, output)

    with output:
        build_brand_dashboard(brands[0], "all").show()


# ============================================================
# Script entry point
# ============================================================
if __name__ == "__main__":
    run_pipeline()