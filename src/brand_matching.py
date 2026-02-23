# brand_matching.py
# Finds brands present in both Sephora and Ulta using multi-strategy brand
# name matching (exact, containment, fuzzy), copies all products and reviews
# under matched brands into unified CSVs with normalized column names.
#
# Inputs:
#   data/processed/Sephora/sephora_products.csv
#   data/processed/Sephora/sephora_reviews.csv
#   data/processed/Ulta/ulta_products.csv
#   data/processed/Ulta/ulta_reviews.csv
#
# Outputs:
#   data/processed/Matched/matched_products.csv
#   data/processed/Matched/matched_reviews.csv
#   data/processed/Matched/brand_mapping.csv  (brand name crosswalk for reference)

import pandas as pd
import numpy as np
import re
from pathlib import Path
from difflib import SequenceMatcher

PROCESSED_DIR = Path("data/processed")

# Minimum fuzzy similarity score (0-1) to consider a match
FUZZY_THRESHOLD = 0.80


# Brand name normalization

def normalize_brand(name):
    """
    Normalize brand names for matching across retailers.
    Lowercases, strips whitespace, removes punctuation, collapses spaces.
    """
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    name = re.sub(r"[''`]", "", name)          # remove apostrophes
    name = re.sub(r"[^\w\s]", " ", name)       # punctuation to space
    name = re.sub(r"\s+", " ", name).strip()   # collapse spaces
    return name


# Multi-strategy brand matching

def match_brands(sephora_brands, ulta_brands):
    """
    Match brand names across retailers using 3 strategies:
      1. Exact match on normalized names
      2. Containment — one name fully contains the other
         (e.g., "Rare Beauty" in "Rare Beauty by Selena Gomez")
      3. Fuzzy match on remaining unmatched using token similarity

    Returns a dict mapping: normalized_sephora_brand -> normalized_ulta_brand
    """
    matches = {}           # sephora_norm -> ulta_norm
    matched_seph = set()
    matched_ulta = set()

    # --- Strategy 1: Exact match ---
    exact = sephora_brands & ulta_brands
    for brand in exact:
        matches[brand] = brand
        matched_seph.add(brand)
        matched_ulta.add(brand)

    print(f"\n  Strategy 1 (Exact):       {len(exact)} matches")

    # --- Strategy 2: Containment match ---
    unmatched_seph = sephora_brands - matched_seph
    unmatched_ulta = ulta_brands - matched_ulta
    containment_count = 0

    for s_brand in sorted(unmatched_seph):
        for u_brand in sorted(unmatched_ulta):
            # Check if one contains the other
            if s_brand in u_brand or u_brand in s_brand:
                matches[s_brand] = u_brand
                matched_seph.add(s_brand)
                matched_ulta.add(u_brand)
                containment_count += 1
                break  # take first containment match

    print(f"  Strategy 2 (Containment): {containment_count} matches")

    # --- Strategy 3: Fuzzy match on remaining ---
    unmatched_seph = sephora_brands - matched_seph
    unmatched_ulta = ulta_brands - matched_ulta
    fuzzy_count = 0

    for s_brand in sorted(unmatched_seph):
        best_score = 0
        best_match = None

        for u_brand in sorted(unmatched_ulta):
            # Token set similarity — handles word reordering
            score = token_set_similarity(s_brand, u_brand)
            if score > best_score:
                best_score = score
                best_match = u_brand

        if best_score >= FUZZY_THRESHOLD and best_match:
            matches[s_brand] = best_match
            matched_seph.add(s_brand)
            matched_ulta.add(best_match)
            fuzzy_count += 1

    print(f"  Strategy 3 (Fuzzy):       {fuzzy_count} matches")
    print(f"\n  Total matched brands:     {len(matches)}")

    return matches


def token_set_similarity(a, b):
    """
    Compare two strings using token set similarity.
    Splits into word tokens, computes similarity on the intersection
    and remainders. Handles cases like word reordering and extra words.
    """
    tokens_a = set(a.split())
    tokens_b = set(b.split())

    common = tokens_a & tokens_b
    if not common:
        return SequenceMatcher(None, a, b).ratio()

    common_str = " ".join(sorted(common))
    combined_a = common_str + " " + " ".join(sorted(tokens_a - common))
    combined_b = common_str + " " + " ".join(sorted(tokens_b - common))

    scores = [
        SequenceMatcher(None, common_str, combined_a.strip()).ratio(),
        SequenceMatcher(None, common_str, combined_b.strip()).ratio(),
        SequenceMatcher(None, combined_a.strip(), combined_b.strip()).ratio(),
    ]

    return max(scores)


# Normalize reviews to shared schema

def normalize_sephora_reviews(df):
    """
    Map Sephora review columns to shared schema.
    """
    df = df.copy()
    df["retailer"] = "sephora"
    df["review_id"] = df.index.astype(str) + "_sephora"
    df["headline"] = np.nan
    df["nickname"] = np.nan
    df["location"] = np.nan
    df["bottom_line"] = np.nan
    df["helpful_votes"] = df["Helpfulness"]
    df["not_helpful_votes"] = np.nan
    df["is_verified_buyer"] = np.nan
    df["is_verified_reviewer"] = np.nan
    df["disclosure_code"] = np.nan

    df = df.rename(columns={
        "pd_id": "pd_id",
        "Rating": "rating",
        "ReviewText": "review_text",
        "SubmissionTime": "submission_time",
        "skinTone": "skin_tone",
        "skinType": "skin_type",
    })

    return df


def normalize_ulta_reviews(df):
    """
    Map Ulta review columns to shared schema.
    """
    df = df.copy()
    df["retailer"] = "ulta"
    df["skin_tone"] = np.nan
    df["skin_type"] = np.nan

    df = df.rename(columns={
        "pd_id": "pd_id",
        "review_id": "review_id",
        "Rating": "rating",
        "headline": "headline",
        "ReviewText": "review_text",
        "SubmissionTime": "submission_time",
        "nickname": "nickname",
        "location": "location",
        "bottom_line": "bottom_line",
        "helpful_votes": "helpful_votes",
        "not_helpful_votes": "not_helpful_votes",
        "is_verified_buyer": "is_verified_buyer",
        "is_verified_reviewer": "is_verified_reviewer",
        "disclosure_code": "disclosure_code",
    })

    return df


REVIEW_COLS = [
    "retailer", "pd_id", "review_id", "rating", "headline", "review_text",
    "submission_time", "nickname", "location", "bottom_line",
    "helpful_votes", "not_helpful_votes",
    "is_verified_buyer", "is_verified_reviewer", "disclosure_code",
    "skin_tone", "skin_type",
]


# Main

def main():
    # Load processed data
    print("LOADING PROCESSED DATA")

    seph_products = pd.read_csv(PROCESSED_DIR / "Sephora" / "sephora_products.csv", dtype=str)
    seph_reviews = pd.read_csv(PROCESSED_DIR / "Sephora" / "sephora_reviews.csv", dtype=str)
    ulta_products = pd.read_csv(PROCESSED_DIR / "Ulta" / "ulta_products.csv", dtype=str)
    ulta_reviews = pd.read_csv(PROCESSED_DIR / "Ulta" / "ulta_reviews.csv", dtype=str)

    print(f"  Sephora: {len(seph_products):,} products | {len(seph_reviews):,} reviews")
    print(f"  Ulta:    {len(ulta_products):,} products | {len(ulta_reviews):,} reviews")

    # Normalize brand names
    print("MATCHING BRANDS")

    seph_products["brand_normalized"] = seph_products["brand"].apply(normalize_brand)
    ulta_products["brand_normalized"] = ulta_products["brand"].apply(normalize_brand)

    sephora_brands = set(seph_products["brand_normalized"].unique()) - {""}
    ulta_brands = set(ulta_products["brand_normalized"].unique()) - {""}

    print(f"  Sephora unique brands: {len(sephora_brands):,}")
    print(f"  Ulta unique brands:    {len(ulta_brands):,}")

    # Run multi-strategy matching
    brand_map = match_brands(sephora_brands, ulta_brands)

    # Build reverse map (ulta_norm -> sephora_norm) for lookups
    ulta_to_seph = {v: k for k, v in brand_map.items()}

    # Show match examples
    print(f"\n  Sample matches:")
    for i, (s, u) in enumerate(sorted(brand_map.items())):
        if s != u:  # only show non-exact matches
            # Get original brand names for display
            s_orig = seph_products[seph_products["brand_normalized"] == s]["brand"].iloc[0]
            u_orig = ulta_products[ulta_products["brand_normalized"] == u]["brand"].iloc[0]
            print(f"    Sephora: {s_orig:<40} <-> Ulta: {u_orig}")
            if i > 20:
                print(f"    ... and more")
                break

    # Save brand mapping for reference
    brand_mapping_df = pd.DataFrame([
        {
            "sephora_brand": seph_products[seph_products["brand_normalized"] == s]["brand"].iloc[0],
            "ulta_brand": ulta_products[ulta_products["brand_normalized"] == u]["brand"].iloc[0],
            "sephora_normalized": s,
            "ulta_normalized": u,
            "match_type": "exact" if s == u else ("containment" if (s in u or u in s) else "fuzzy"),
        }
        for s, u in sorted(brand_map.items())
    ])
    brand_mapping_df.to_csv(PROCESSED_DIR / "Matched" / "brand_mapping.csv", index=False)
    print(f"\n  Saved brand_mapping.csv ({len(brand_mapping_df)} brand pairs)")

    # Filter products to matched brands
    print("FILTERING TO MATCHED BRANDS")

    matched_seph_brands = set(brand_map.keys())
    matched_ulta_brands = set(brand_map.values())

    seph_matched = seph_products[seph_products["brand_normalized"].isin(matched_seph_brands)].copy()
    ulta_matched = ulta_products[ulta_products["brand_normalized"].isin(matched_ulta_brands)].copy()

    print(f"  Sephora products from matched brands: {len(seph_matched):,}")
    print(f"  Ulta products from matched brands:    {len(ulta_matched):,}")

    # Add retailer column
    seph_matched["retailer"] = "sephora"
    ulta_matched["retailer"] = "ulta"

    product_cols = ["retailer", "product_id", "product_url", "brand",
                    "brand_normalized", "product_name", "category", "price", "rating"]

    matched_products = pd.concat([
        seph_matched[product_cols],
        ulta_matched[product_cols]
    ], ignore_index=True)

    print(f"  Combined matched products: {len(matched_products):,}")

    # Filter reviews to matched products
    print("FILTERING REVIEWS TO MATCHED PRODUCTS")

    seph_matched_pids = set(seph_matched["product_id"].astype(str).str.strip())
    ulta_matched_pids = set(ulta_matched["product_id"].astype(str).str.strip())

    seph_reviews["pd_id"] = seph_reviews["pd_id"].astype(str).str.strip()
    ulta_reviews["pd_id"] = ulta_reviews["pd_id"].astype(str).str.strip()

    seph_reviews_matched = seph_reviews[seph_reviews["pd_id"].isin(seph_matched_pids)].copy()
    ulta_reviews_matched = ulta_reviews[ulta_reviews["pd_id"].isin(ulta_matched_pids)].copy()

    print(f"  Sephora reviews from matched products: {len(seph_reviews_matched):,}")
    print(f"  Ulta reviews from matched products:    {len(ulta_reviews_matched):,}")

    # Normalize review columns to shared schema
    print("NORMALIZING REVIEW COLUMNS")

    seph_reviews_norm = normalize_sephora_reviews(seph_reviews_matched)
    ulta_reviews_norm = normalize_ulta_reviews(ulta_reviews_matched)

    matched_reviews = pd.concat([
        seph_reviews_norm[REVIEW_COLS],
        ulta_reviews_norm[REVIEW_COLS]
    ], ignore_index=True)

    print(f"  Combined matched reviews: {len(matched_reviews):,}")

    # Save
    print("SAVING")

    matched_products.to_csv(PROCESSED_DIR / "Matched" / "matched_products.csv", index=False)
    matched_reviews.to_csv(PROCESSED_DIR / "Matched" / "matched_reviews.csv", index=False)

    print(f"  Saved matched_products.csv ({len(matched_products):,} rows)")
    print(f"  Saved matched_reviews.csv ({len(matched_reviews):,} rows)")

    # Summary
    print("SUMMARY")
    print(f"  Matched brands: {len(brand_map):,}")
    print(f"  Matched products: {len(matched_products):,} ({len(seph_matched):,} Sephora + {len(ulta_matched):,} Ulta)")
    print(f"  Matched reviews:  {len(matched_reviews):,} ({len(seph_reviews_matched):,} Sephora + {len(ulta_reviews_matched):,} Ulta)")

    # Unmatched brands for review
    sephora_only = sephora_brands - matched_seph_brands
    ulta_only = ulta_brands - matched_ulta_brands

    print(f"\n  Unmatched Sephora brands: {len(sephora_only)}")
    if sephora_only:
        sample = sorted(sephora_only)[:15]
        originals = []
        for b in sample:
            orig = seph_products[seph_products["brand_normalized"] == b]["brand"].iloc[0]
            originals.append(orig)
        print(f"    Sample: {originals}")

    print(f"  Unmatched Ulta brands: {len(ulta_only)}")
    if ulta_only:
        sample = sorted(ulta_only)[:15]
        originals = []
        for b in sample:
            orig = ulta_products[ulta_products["brand_normalized"] == b]["brand"].iloc[0]
            originals.append(orig)
        print(f"    Sample: {originals}")

    # Per-brand breakdown
    print(f"\n  Top 10 matched brands by total products:")
    brand_counts = matched_products.groupby("brand_normalized").agg(
        sephora=("retailer", lambda x: (x == "sephora").sum()),
        ulta=("retailer", lambda x: (x == "ulta").sum()),
        total=("retailer", "count"),
    ).sort_values("total", ascending=False).head(10)
    print(brand_counts.to_string(index=True))

    print("\nDone.")


if __name__ == "__main__":
    main()