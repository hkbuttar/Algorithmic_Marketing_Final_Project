# segmentation_features.py
# Builds product-level feature CSVs for market segmentation by aggregating
# review metrics (sentiment, topics, ratings, velocity) to the product level.
#
# Inputs:
#   data/processed/Sephora/sephora_products.csv
#   data/processed/Sephora/sephora_reviews.csv
#   data/processed/Ulta/ulta_products.csv
#   data/processed/Ulta/ulta_reviews.csv
#
# Outputs:
#   data/processed/Sephora/sephora_segmentation.csv
#   data/processed/Ulta/ulta_segmentation.csv

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

import nltk
nltk.download("vader_lexicon", quiet=True)

PROCESSED_DIR = Path("data/processed")
N_TOPICS = 12


# -------------------------
# Text preprocessing
# -------------------------

def clean_text(text):
    """
    Basic text cleaning for NLP: lowercase, remove non-alpha, collapse spaces.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Sentiment scoring

def score_sentiment(reviews_df, text_col="ReviewText"):
    """
    Add VADER sentiment scores to each review.
    """
    sia = SentimentIntensityAnalyzer()

    print("  Scoring sentiment...")
    reviews_df["sentiment_compound"] = (
        reviews_df[text_col]
        .fillna("")
        .apply(lambda x: sia.polarity_scores(str(x))["compound"])
    )

    reviews_df["sentiment_label"] = pd.cut(
        reviews_df["sentiment_compound"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"]
    )

    return reviews_df


# Topic modeling

def extract_topics(reviews_df, text_col="ReviewText", n_topics=N_TOPICS):
    """
    Run NMF topic modeling on review text. Assigns each review a dominant topic
    and returns topic label mapping.
    """
    print(f"  Extracting {n_topics} topics via NMF...")

    reviews_df["text_clean"] = reviews_df[text_col].apply(clean_text)

    # Filter out empty text
    valid_mask = reviews_df["text_clean"].str.len() > 10
    valid_text = reviews_df.loc[valid_mask, "text_clean"]

    if len(valid_text) < n_topics:
        print(f"  WARNING: Only {len(valid_text)} valid reviews, skipping topic modeling")
        reviews_df["dominant_topic"] = -1
        return reviews_df, {}

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
    )
    tfidf_matrix = tfidf.fit_transform(valid_text)

    nmf = NMF(n_components=n_topics, random_state=42, max_iter=300)
    topic_matrix = nmf.fit_transform(tfidf_matrix)

    # Assign dominant topic to valid reviews
    reviews_df["dominant_topic"] = -1
    reviews_df.loc[valid_mask, "dominant_topic"] = topic_matrix.argmax(axis=1)

    # Build topic labels from top words
    feature_names = tfidf.get_feature_names_out()
    topic_labels = {}
    print(f"\n  Topic keywords:")
    for i, topic in enumerate(nmf.components_):
        top_words = [feature_names[j] for j in topic.argsort()[-8:]]
        topic_labels[i] = ", ".join(top_words)
        print(f"    Topic {i}: {topic_labels[i]}")
    print()

    reviews_df.drop(columns=["text_clean"], inplace=True)
    return reviews_df, topic_labels


# Product-level aggregation

def aggregate_to_product(products_df, reviews_df, retailer):
    """
    Aggregate review-level metrics to product-level for segmentation.
    """
    print(f"  Aggregating review features to product level...")

    # Ensure types
    reviews_df["Rating"] = pd.to_numeric(reviews_df["Rating"], errors="coerce")
    reviews_df["pd_id"] = reviews_df["pd_id"].astype(str).str.strip()
    products_df["product_id"] = products_df["product_id"].astype(str).str.strip()

    # Parse submission time
    reviews_df["submission_ts"] = pd.to_numeric(reviews_df["SubmissionTime"], errors="coerce")
    # Bazaarvoice/PowerReviews timestamps are in milliseconds
    reviews_df["submission_date"] = pd.to_datetime(
        reviews_df["submission_ts"], unit="ms", errors="coerce"
    )

    # --- Per-product aggregation ---
    agg = reviews_df.groupby("pd_id").agg(
        # Review count
        review_count=("Rating", "size"),

        # Rating stats
        avg_rating=("Rating", "mean"),
        rating_std=("Rating", "std"),
        pct_5_star=("Rating", lambda x: (x == 5).mean()),
        pct_4_star=("Rating", lambda x: (x == 4).mean()),
        pct_3_star=("Rating", lambda x: (x == 3).mean()),
        pct_2_star=("Rating", lambda x: (x == 2).mean()),
        pct_1_star=("Rating", lambda x: (x == 1).mean()),

        # Sentiment
        avg_sentiment=("sentiment_compound", "mean"),
        sentiment_std=("sentiment_compound", "std"),
        pct_positive=("sentiment_label", lambda x: (x == "positive").mean()),
        pct_negative=("sentiment_label", lambda x: (x == "negative").mean()),
        pct_neutral=("sentiment_label", lambda x: (x == "neutral").mean()),

        # Topic — top topic prevalence
        dominant_topic_mode=("dominant_topic", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else -1),

        # Time range
        first_review=("submission_date", "min"),
        last_review=("submission_date", "max"),
    ).reset_index()

    # Review velocity
    agg["review_span_days"] = (agg["last_review"] - agg["first_review"]).dt.days
    agg["review_span_months"] = agg["review_span_days"] / 30.44  # avg days per month
    agg["reviews_per_month"] = np.where(
        agg["review_span_months"] > 0,
        agg["review_count"] / agg["review_span_months"],
        agg["review_count"]  # all reviews in same month
    )

    # Normalized review count (0-1 within this retailer)
    agg["review_count_normalized"] = (
        (agg["review_count"] - agg["review_count"].min()) /
        (agg["review_count"].max() - agg["review_count"].min())
    ).fillna(0)

    # Rating dispersion over time (std of monthly average ratings)
    monthly_ratings = reviews_df.copy()
    monthly_ratings["year_month"] = monthly_ratings["submission_date"].dt.to_period("M")
    monthly_avg = monthly_ratings.groupby(["pd_id", "year_month"])["Rating"].mean().reset_index()
    rating_dispersion = monthly_avg.groupby("pd_id")["Rating"].std().reset_index()
    rating_dispersion.columns = ["pd_id", "rating_dispersion_over_time"]
    agg = agg.merge(rating_dispersion, on="pd_id", how="left")
    agg["rating_dispersion_over_time"] = agg["rating_dispersion_over_time"].fillna(0)

    # Top topic prevalence (what % of reviews belong to the dominant topic)
    topic_prevalence = reviews_df.groupby("pd_id").apply(
        lambda x: x["dominant_topic"].value_counts(normalize=True).iloc[0]
        if len(x) > 0 and x["dominant_topic"].iloc[0] != -1 else 0
    ).reset_index()
    topic_prevalence.columns = ["pd_id", "top_topic_prevalence"]
    agg = agg.merge(topic_prevalence, on="pd_id", how="left")

    # Helpfulness engagement rate (Ulta only)
    if retailer == "ulta":
        reviews_df["helpful_votes"] = pd.to_numeric(reviews_df["helpful_votes"], errors="coerce").fillna(0)
        reviews_df["not_helpful_votes"] = pd.to_numeric(reviews_df["not_helpful_votes"], errors="coerce").fillna(0)
        reviews_df["total_votes"] = reviews_df["helpful_votes"] + reviews_df["not_helpful_votes"]

        helpfulness = reviews_df.groupby("pd_id").agg(
            total_helpful=("helpful_votes", "sum"),
            total_votes=("total_votes", "sum"),
        ).reset_index()
        helpfulness["helpfulness_rate"] = np.where(
            helpfulness["total_votes"] > 0,
            helpfulness["total_helpful"] / helpfulness["total_votes"],
            0
        )
        agg = agg.merge(helpfulness[["pd_id", "helpfulness_rate"]], on="pd_id", how="left")

        # Ulta-specific: verified buyer, recommendation, disclosure ratios
        reviews_df["is_verified_buyer"] = reviews_df["is_verified_buyer"].astype(str).str.strip().str.lower()
        reviews_df["is_verified_reviewer"] = reviews_df["is_verified_reviewer"].astype(str).str.strip().str.lower()
        reviews_df["bottom_line"] = reviews_df["bottom_line"].astype(str).str.strip().str.lower()

        ulta_specific = reviews_df.groupby("pd_id").agg(
            pct_verified_buyer=("is_verified_buyer", lambda x: (x == "true").mean()),
            pct_verified_reviewer=("is_verified_reviewer", lambda x: (x == "true").mean()),
            pct_would_recommend=("bottom_line", lambda x: (x == "yes").mean()),
            pct_has_disclosure=("disclosure_code", lambda x: x.notna().mean()),
        ).reset_index()
        agg = agg.merge(ulta_specific, on="pd_id", how="left")

    else:
        # Sephora — Helpfulness is a single field, not votes
        reviews_df["Helpfulness"] = pd.to_numeric(reviews_df["Helpfulness"], errors="coerce").fillna(0)
        helpfulness = reviews_df.groupby("pd_id")["Helpfulness"].mean().reset_index()
        helpfulness.columns = ["pd_id", "avg_helpfulness"]
        agg = agg.merge(helpfulness, on="pd_id", how="left")

        # Sephora-specific: skin tone and skin type distributions
        # Skin tone
        skin_tone_vals = reviews_df[reviews_df["skinTone"].notna()].copy()
        if len(skin_tone_vals) > 0:
            skin_tone_dist = (
                skin_tone_vals.groupby(["pd_id", "skinTone"]).size()
                .unstack(fill_value=0)
            )
            # Normalize to percentages
            skin_tone_dist = skin_tone_dist.div(skin_tone_dist.sum(axis=1), axis=0)
            skin_tone_dist.columns = ["pct_skin_tone_" + str(c).lower().replace(" ", "_") for c in skin_tone_dist.columns]
            skin_tone_dist = skin_tone_dist.reset_index()
            agg = agg.merge(skin_tone_dist, on="pd_id", how="left")

            # Also add % of reviews that have skin tone data
            tone_coverage = reviews_df.groupby("pd_id")["skinTone"].apply(
                lambda x: x.notna().mean()
            ).reset_index()
            tone_coverage.columns = ["pd_id", "pct_has_skin_tone"]
            agg = agg.merge(tone_coverage, on="pd_id", how="left")

        # Skin type
        skin_type_vals = reviews_df[reviews_df["skinType"].notna()].copy()
        if len(skin_type_vals) > 0:
            skin_type_dist = (
                skin_type_vals.groupby(["pd_id", "skinType"]).size()
                .unstack(fill_value=0)
            )
            skin_type_dist = skin_type_dist.div(skin_type_dist.sum(axis=1), axis=0)
            skin_type_dist.columns = ["pct_skin_type_" + str(c).lower().replace(" ", "_") for c in skin_type_dist.columns]
            skin_type_dist = skin_type_dist.reset_index()
            agg = agg.merge(skin_type_dist, on="pd_id", how="left")

            # % of reviews that have skin type data
            type_coverage = reviews_df.groupby("pd_id")["skinType"].apply(
                lambda x: x.notna().mean()
            ).reset_index()
            type_coverage.columns = ["pd_id", "pct_has_skin_type"]
            agg = agg.merge(type_coverage, on="pd_id", how="left")

    # Drop intermediate time columns
    agg.drop(columns=["first_review", "last_review", "review_span_days", "review_span_months"], inplace=True)

    # Merge with product data
    merged = products_df.merge(agg, left_on="product_id", right_on="pd_id", how="inner")
    merged.drop(columns=["pd_id"], inplace=True)

    # Price buckets
    merged["price"] = pd.to_numeric(merged["price"], errors="coerce")
    merged["price_bucket"] = (merged["price"] // 10 * 10).astype(int).astype(str) + "-" + \
                             ((merged["price"] // 10 * 10) + 10).astype(int).astype(str)

    # Price relative to category median
    merged["price_vs_category_median"] = merged["price"] / merged.groupby("category")["price"].transform("median")

    return merged


# Main

def main():
    # Load data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    seph_products = pd.read_csv(PROCESSED_DIR / "Sephora" / "sephora_products.csv", dtype=str)
    seph_reviews = pd.read_csv(PROCESSED_DIR / "Sephora" / "sephora_reviews.csv", dtype=str)
    ulta_products = pd.read_csv(PROCESSED_DIR / "Ulta" / "ulta_products.csv", dtype=str)
    ulta_reviews = pd.read_csv(PROCESSED_DIR / "Ulta" / "ulta_reviews.csv", dtype=str)

    print(f"  Sephora: {len(seph_products):,} products | {len(seph_reviews):,} reviews")
    print(f"  Ulta:    {len(ulta_products):,} products | {len(ulta_reviews):,} reviews")

    # --- Sephora ---
    print("\n" + "=" * 60)
    print("SEPHORA")
    print("=" * 60)

    seph_reviews = score_sentiment(seph_reviews)
    seph_reviews, seph_topic_labels = extract_topics(seph_reviews)
    seph_seg = aggregate_to_product(seph_products, seph_reviews, retailer="sephora")

    print(f"\n  Sephora segmentation: {len(seph_seg):,} products x {len(seph_seg.columns)} features")

    # --- Ulta ---
    print("\n" + "=" * 60)
    print("ULTA")
    print("=" * 60)

    ulta_reviews = score_sentiment(ulta_reviews)
    ulta_reviews, ulta_topic_labels = extract_topics(ulta_reviews)
    ulta_seg = aggregate_to_product(ulta_products, ulta_reviews, retailer="ulta")

    print(f"\n  Ulta segmentation: {len(ulta_seg):,} products x {len(ulta_seg.columns)} features")

    # Save
    print("\n" + "=" * 60)
    print("SAVING")
    print("=" * 60)

    seph_seg.to_csv(PROCESSED_DIR / "Sephora/sephora_segmentation.csv", index=False)
    ulta_seg.to_csv(PROCESSED_DIR / "Ulta/ulta_segmentation.csv", index=False)

    print(f"  Saved sephora_segmentation.csv ({len(seph_seg):,} rows)")
    print(f"  Saved ulta_segmentation.csv ({len(ulta_seg):,} rows)")

    # Save topic labels for reference
    for name, labels in [("sephora", seph_topic_labels), ("ulta", ulta_topic_labels)]:
        if labels:
            topic_df = pd.DataFrame([
                {"topic_id": k, "keywords": v} for k, v in labels.items()
            ])
            topic_df.to_csv(PROCESSED_DIR / f"{name}_topic_labels.csv", index=False)
            print(f"  Saved {name}_topic_labels.csv")

    # Summary
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)

    print("\n  Sephora columns:")
    for col in seph_seg.columns:
        print(f"    {col}")

    print(f"\n  Ulta columns:")
    for col in ulta_seg.columns:
        print(f"    {col}")

    # Quick stats
    for name, df in [("Sephora", seph_seg), ("Ulta", ulta_seg)]:
        print(f"\n  {name} feature stats:")
        print(f"    Avg reviews/product:   {df['review_count'].mean():.1f}")
        print(f"    Avg reviews/month:     {df['reviews_per_month'].mean():.1f}")
        print(f"    Avg sentiment:         {df['avg_sentiment'].mean():.3f}")
        print(f"    Avg rating:            {df['avg_rating'].mean():.2f}")
        print(f"    Price range:           ${df['price'].min():.0f} - ${df['price'].max():.0f}")
        print(f"    Top topic prevalence:  {df['top_topic_prevalence'].mean():.2%}")
        print(f"    Price buckets:         {df['price_bucket'].nunique()} buckets")

    print("\nDone.")


if __name__ == "__main__":
    main()