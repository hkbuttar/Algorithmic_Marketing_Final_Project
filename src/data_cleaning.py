# data_cleaning.py
# Merges raw batch CSVs into consolidated files, fixes multi-line and comma-broken
# review parsing, removes products with missing info or no reviews, and drops
# incomplete reviews while preserving a minimum of 20 reviews per product.
#
# Auto-detects Sephora vs Ulta review files by header inspection.
#
# Outputs:
#   data/processed/Sephora/sephora_products.csv
#   data/processed/Sephora/sephora_reviews.csv
#   data/processed/Ulta/ulta_products.csv
#   data/processed/Ulta/ulta_reviews.csv

import pandas as pd
import numpy as np
import csv
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MIN_REVIEWS_PER_PRODUCT = 20

SEPHORA_REVIEW_COLS = ["pd_id", "Rating", "ReviewText", "SubmissionTime",
                       "Helpfulness", "skinTone", "skinType"]

ULTA_REVIEW_COLS = ["pd_id", "review_id", "Rating", "headline", "ReviewText",
                    "SubmissionTime", "nickname", "location", "bottom_line",
                    "helpful_votes", "not_helpful_votes", "is_verified_buyer",
                    "is_verified_reviewer", "disclosure_code"]


# Auto-detect retailer

def detect_review_type(filepath):
    """
    Read the header row and determine if it's Sephora or Ulta reviews.
    """
    with open(filepath, "r") as f:
        header = f.readline().strip().split(",")

    if "review_id" in header:
        return "ulta", ULTA_REVIEW_COLS
    else:
        return "sephora", SEPHORA_REVIEW_COLS


# Safe CSV loader

def load_reviews_safe(filepath, expected_cols=None):
    """
    Load a reviews CSV handling commas and newlines inside text fields.

    Strategy:
      1. Try pandas with quoting enabled (handles properly quoted fields)
      2. If row counts look wrong, fall back to manual parsing that
         reconstructs broken rows using the expected column count
    """
    print(f"\nLoading: {filepath}")

    # Attempt 1: Read with explicit quoting rules
    try:
        df = pd.read_csv(
            filepath,
            engine="python",
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
            on_bad_lines="warn",
            dtype=str,
        )
        print(f"  Attempt 1 (QUOTE_ALL): {len(df):,} rows, {len(df.columns)} cols")

        if expected_cols and list(df.columns) == expected_cols:
            print(f"  Columns match expected. Using this parse.")
            return df
    except Exception as e:
        print(f"  Attempt 1 failed: {e}")
        df = None

    # Attempt 2: Read with QUOTE_MINIMAL (default quoting)
    try:
        df = pd.read_csv(
            filepath,
            engine="python",
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="warn",
            dtype=str,
        )
        print(f"  Attempt 2 (QUOTE_MINIMAL): {len(df):,} rows, {len(df.columns)} cols")

        if expected_cols and list(df.columns) == expected_cols:
            print(f"  Columns match expected. Using this parse.")
            return df
    except Exception as e:
        print(f"  Attempt 2 failed: {e}")

    # Attempt 3: Manual reconstruction for badly formatted CSVs
    print(f"  Attempt 3: Manual row reconstruction...")
    df = reconstruct_csv(filepath, expected_cols)

    return df


# Manual CSV reconstruction

def reconstruct_csv(filepath, expected_cols):
    """
    Manually parse a CSV where commas and newlines inside text fields
    broke the row structure. Reconstructs rows by counting columns.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    if not raw_lines:
        return pd.DataFrame()

    if expected_cols:
        header = expected_cols
    else:
        header = raw_lines[0].strip().split(",")

    n_cols = len(header)
    print(f"  Expected columns: {n_cols} -> {header}")

    rows = []
    i = 1  # skip header
    buffer = ""

    while i < len(raw_lines):
        buffer = buffer + raw_lines[i] if buffer else raw_lines[i]

        # Count quotes — if odd number, the line is split mid-field
        quote_count = buffer.count('"')
        if quote_count % 2 != 0:
            i += 1
            continue

        # Try to parse the buffered line
        try:
            parsed = list(csv.reader([buffer.strip()]))[0]
        except Exception:
            buffer = ""
            i += 1
            continue

        if len(parsed) == n_cols:
            rows.append(parsed)
        elif len(parsed) > n_cols:
            fixed = merge_extra_columns(parsed, n_cols, header)
            if fixed:
                rows.append(fixed)
        elif len(parsed) < n_cols:
            i += 1
            continue

        buffer = ""
        i += 1

    df = pd.DataFrame(rows, columns=header)
    print(f"  Reconstructed: {len(df):,} rows")
    return df


def merge_extra_columns(parsed, expected_n, header):
    """
    When a row has too many columns due to unquoted commas in text,
    find the text field and merge the extra parts back together.
    """
    text_fields = {"ReviewText", "headline"}

    text_idx = None
    for i, col in enumerate(header):
        if col in text_fields:
            text_idx = i
            break

    if text_idx is None:
        text_idx = len(header) // 2

    extra = len(parsed) - expected_n

    before = parsed[:text_idx]
    text_parts = parsed[text_idx:text_idx + 1 + extra]
    after = parsed[text_idx + 1 + extra:]

    merged_text = ",".join(text_parts)
    fixed = before + [merged_text] + after

    if len(fixed) == expected_n:
        return fixed
    return None


# Multiline fix

def fix_multiline_reviews(df, text_col="ReviewText"):
    """
    Collapse newlines/carriage returns within text column into spaces.
    """
    if text_col not in df.columns:
        return df

    df[text_col] = (
        df[text_col]
        .astype(str)
        .str.replace(r"\r\n|\r|\n", " ", regex=True)
        .str.replace(r"\s{2,}", " ", regex=True)
        .str.strip()
    )

    df.loc[df[text_col] == "nan", text_col] = np.nan
    return df


# Merge batch files

def merge_product_files(pattern, id_col="product_id"):
    """
    Merge all product CSVs matching a glob pattern. Deduplicates on id_col.
    """
    files = sorted(RAW_DIR.glob(pattern))
    if not files:
        print(f"  No files found for pattern: {pattern}")
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, engine="python", on_bad_lines="warn")
            print(f"  Loaded {len(df):,} rows from {f.name}")
            frames.append(df)
        except Exception as e:
            print(f"  Failed to load {f.name}: {e}")

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=id_col)
    after = len(merged)

    if before != after:
        print(f"  Removed {before - after:,} duplicate products")
    print(f"  Merged total: {after:,} products")
    return merged


def merge_review_files(pattern):
    """
    Merge all review CSVs matching a glob pattern.
    Auto-detects Sephora vs Ulta and uses safe parsing.
    """
    files = sorted(RAW_DIR.glob(pattern))
    if not files:
        print(f"  No files found for pattern: {pattern}")
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            retailer, expected = detect_review_type(f)
            df = load_reviews_safe(f, expected_cols=expected)
            if df is not None and len(df) > 0:
                print(f"  Loaded {len(df):,} rows from {f.name} ({retailer})")
                frames.append(df)
        except Exception as e:
            print(f"  Failed to load {f.name}: {e}")

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)

    # Deduplicate: Ulta has review_id, Sephora doesn't
    if "review_id" in merged.columns:
        merged = merged.drop_duplicates(subset="review_id")
    else:
        merged = merged.drop_duplicates()

    after = len(merged)
    if before != after:
        print(f"  Removed {before - after:,} duplicate reviews")
    print(f"  Merged total: {after:,} reviews")
    return merged


# Clean products

def clean_products(df, retailer):
    """
    Drop products missing any field. Clean price and rating to numeric.
    """
    before = len(df)

    df = df.dropna()

    # Clean price — extract numeric value
    df["price"] = df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    # Clean rating — numeric
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    after = len(df)
    print(f"  {retailer} products: {before:,} -> {after:,} (dropped {before - after:,})")
    return df


# Remove products with no reviews & orphan reviews

def remove_products_without_reviews(products_df, reviews_df, pid_col="pd_id", product_id_col="product_id"):
    """
    Drop products that have zero reviews.
    Drop reviews that don't have a matching product.
    """
    products_df[product_id_col] = products_df[product_id_col].astype(str).str.strip()
    reviews_df[pid_col] = reviews_df[pid_col].astype(str).str.strip()

    product_ids_with_reviews = set(reviews_df[pid_col].unique())
    review_pids_with_products = set(products_df[product_id_col].unique())

    before_products = len(products_df)
    products_df = products_df[products_df[product_id_col].isin(product_ids_with_reviews)]
    print(f"  Products with reviews: {before_products:,} -> {len(products_df):,}")

    before_reviews = len(reviews_df)
    reviews_df = reviews_df[reviews_df[pid_col].isin(review_pids_with_products)]
    print(f"  Reviews with matching products: {before_reviews:,} -> {len(reviews_df):,}")

    return products_df, reviews_df


# Drop incomplete reviews (min 20 per product)

def clean_reviews_with_floor(df, required_cols, pid_col="pd_id", min_reviews=20):
    """
    Drop reviews with missing values in required columns, but never let
    a product fall below min_reviews. If a product already has fewer than
    min_reviews total, keep all its reviews.
    """
    before = len(df)
    cleaned = []

    for pid, group in df.groupby(pid_col):
        has_missing = group[required_cols].isnull().any(axis=1)
        complete = group[~has_missing]
        incomplete = group[has_missing]

        total = len(group)
        n_complete = len(complete)

        if n_complete >= min_reviews:
            cleaned.append(complete)
        elif total >= min_reviews:
            need = min_reviews - n_complete
            incomplete_sorted = incomplete.copy()
            incomplete_sorted["_null_count"] = incomplete_sorted[required_cols].isnull().sum(axis=1)
            incomplete_sorted = incomplete_sorted.sort_values("_null_count")
            keep_incomplete = incomplete_sorted.head(need).drop(columns=["_null_count"])
            cleaned.append(pd.concat([complete, keep_incomplete]))
        else:
            cleaned.append(group)

    result = pd.concat(cleaned, ignore_index=True)
    after = len(result)
    print(f"  Reviews cleaned: {before:,} -> {after:,} (dropped {before - after:,})")
    return result


# Main

def main():
    # Step 1: Merge batch files
    print("STEP 1: MERGING BATCH FILES")

    print("\n--- Sephora Products ---")
    seph_products = merge_product_files("sephora_products*.csv")

    print("\n--- Sephora Reviews ---")
    seph_reviews = merge_review_files("sephora_reviews*.csv")

    print("\n--- Ulta Products ---")
    ulta_products = merge_product_files("ulta_products*.csv")

    print("\n--- Ulta Reviews ---")
    ulta_reviews = merge_review_files("ulta_reviews*.csv")

    # Step 2: Fix multi-line reviews
    print("STEP 2: FIXING MULTI-LINE REVIEWS")

    seph_reviews = fix_multiline_reviews(seph_reviews, text_col="ReviewText")
    print(f"  Sephora: fixed multi-line ReviewText")

    ulta_reviews = fix_multiline_reviews(ulta_reviews, text_col="ReviewText")
    if "headline" in ulta_reviews.columns:
        ulta_reviews = fix_multiline_reviews(ulta_reviews, text_col="headline")
    print(f"  Ulta: fixed multi-line ReviewText + headline")

    # Step 3: Clean products
    print("STEP 3: REMOVING PRODUCTS WITH MISSING INFO")

    seph_products = clean_products(seph_products, "Sephora")
    ulta_products = clean_products(ulta_products, "Ulta")

    # Step 4: Remove products with no reviews & orphan reviews
    print("STEP 4: REMOVING PRODUCTS WITH NO REVIEWS")

    print("  Sephora:")
    seph_products, seph_reviews = remove_products_without_reviews(
        seph_products, seph_reviews, pid_col="pd_id", product_id_col="product_id"
    )
    print("  Ulta:")
    ulta_products, ulta_reviews = remove_products_without_reviews(
        ulta_products, ulta_reviews, pid_col="pd_id", product_id_col="product_id"
    )

    # Step 5: Drop incomplete reviews with floor protection
    print("STEP 5: CLEANING REVIEWS (MIN 20 PER PRODUCT)")

    sephora_required = [c for c in SEPHORA_REVIEW_COLS if c != "pd_id"]
    ulta_required = [c for c in ULTA_REVIEW_COLS if c not in ("pd_id", "disclosure_code")]

    print("  Sephora:")
    seph_reviews = clean_reviews_with_floor(
        seph_reviews, required_cols=sephora_required,
        pid_col="pd_id", min_reviews=MIN_REVIEWS_PER_PRODUCT
    )

    print("  Ulta:")
    ulta_reviews = clean_reviews_with_floor(
        ulta_reviews, required_cols=ulta_required,
        pid_col="pd_id", min_reviews=MIN_REVIEWS_PER_PRODUCT
    )

    # Step 6: Final reconciliation
    print("STEP 6: FINAL RECONCILIATION")

    print("  Sephora:")
    seph_products, seph_reviews = remove_products_without_reviews(
        seph_products, seph_reviews, pid_col="pd_id", product_id_col="product_id"
    )
    print("  Ulta:")
    ulta_products, ulta_reviews = remove_products_without_reviews(
        ulta_products, ulta_reviews, pid_col="pd_id", product_id_col="product_id"
    )

    # Save
    print("SAVING TO data/processed/")

    seph_products.to_csv(PROCESSED_DIR / "Sephora" / "sephora_products.csv", index=False)
    seph_reviews.to_csv(PROCESSED_DIR / "Sephora" / "sephora_reviews.csv", index=False)
    ulta_products.to_csv(PROCESSED_DIR / "Ulta" / "ulta_products.csv", index=False)
    ulta_reviews.to_csv(PROCESSED_DIR / "Ulta" / "ulta_reviews.csv", index=False)

    print(f"  Saved 4 files to {PROCESSED_DIR}/")

    # Summary
    print("SUMMARY")
    print(f"  Sephora: {len(seph_products):,} products | {len(seph_reviews):,} reviews")
    print(f"  Ulta:    {len(ulta_products):,} products | {len(ulta_reviews):,} reviews")
    print(f"\n  Avg reviews per product:")
    print(f"    Sephora: {len(seph_reviews) / max(len(seph_products), 1):.1f}")
    print(f"    Ulta:    {len(ulta_reviews) / max(len(ulta_products), 1):.1f}")

    for name, rev_df in [("Sephora", seph_reviews), ("Ulta", ulta_reviews)]:
        counts = rev_df.groupby("pd_id").size()
        print(f"\n  {name} reviews per product distribution:")
        print(f"    Min: {counts.min()}  |  Median: {counts.median():.0f}  |  Max: {counts.max()}")
        print(f"    Products under {MIN_REVIEWS_PER_PRODUCT} reviews: {(counts < MIN_REVIEWS_PER_PRODUCT).sum()}")

    print("\nDone.")


if __name__ == "__main__":
    main()