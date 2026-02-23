"""
build_review_timeseries.py

Reads raw per-review CSVs for Sephora and Ulta, parses review dates,
and aggregates to monthly product-level timeseries.

Inputs:
    data/processed/Sephora/sephora_reviews.csv          (or sephora_reviews_raw.csv)
    data/processed/Ulta/ulta_reviews.csv                (or ulta_reviews_raw.csv)

Input columns (auto-detected via aliases — see COL_ALIASES below):
    product_id    : product identifier
                    aliases: product_id, productid, product_code, sku, id
    review_date   : date of review (any parseable format)
                    aliases: review_date, date, reviewdate, created_at,
                             submission_time, review_time, timestamp
    rating        : numeric star rating (1–5)
                    aliases: rating, stars, star_rating, review_rating,
                             overall_rating, score
    sentiment     : pre-computed sentiment score (optional — derived from
                    review_text via VADER if absent)
                    aliases: avg_sentiment, sentiment, sentiment_score,
                             compound, vader_compound
    review_text   : raw review text (optional — used to compute sentiment
                    if no sentiment column is present)
                    aliases: review_text, text, body, review_body,
                             comment, review_content

Outputs:
    data/processed/Sephora/sephora_reviews_timeseries.csv
    data/processed/Ulta/ulta_reviews_timeseries.csv

Output columns (one row per product × calendar month):
    product_id          : product identifier
    year_month          : calendar month (YYYY-MM)
    review_count        : number of reviews submitted that month
    avg_rating          : mean star rating that month
    rating_std          : standard deviation of ratings that month
    pct_5star           : share of 5-star reviews that month
    pct_1star           : share of 1-star reviews that month
    avg_sentiment       : mean VADER compound score that month
    pct_positive        : share of reviews with compound >= 0.05
    pct_negative        : share of reviews with compound <= -0.05
    pct_neutral         : share of reviews with -0.05 < compound < 0.05
    velocity_3m         : 3-month rolling average of review_count
    velocity_delta      : month-over-month change in review_count
    velocity_delta_pct  : month-over-month % change in review_count
    avg_rating_3m       : 3-month rolling average of avg_rating
    rating_trend        : OLS slope of avg_rating over trailing 3 months
                          (positive = improving, negative = declining)
    pct_negative_3m     : 3-month rolling average of pct_negative
    negative_shock      : 1 if pct_negative > 1.5x its own 3m average
    velocity_lag1       : review_count from 1 month prior
    velocity_lag2       : review_count from 2 months prior
    neg_shock_lag1      : negative_shock flag from 1 month prior
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "data" / "processed"

CONFIGS = [
    {
        "platform":     "Sephora",
        "review_file":  PROCESSED / "Sephora" / "sephora_reviews.csv",
        "product_file": PROCESSED / "Sephora" / "sephora_products.csv",
        "out_file":     PROCESSED / "Sephora" / "sephora_reviews_timeseries.csv",
    },
    {
        "platform":     "Ulta",
        "review_file":  PROCESSED / "Ulta" / "ulta_reviews.csv",
        "product_file": PROCESSED / "Ulta" / "ulta_products.csv",
        "out_file":     PROCESSED / "Ulta" / "ulta_reviews_timeseries.csv",
    },
]

# Column alias resolution
COL_ALIASES = {
    "product_id":   ["product_id", "productid", "product_code", "sku", "id", "pd_id"],
    "review_date":  ["review_date", "date", "reviewdate", "created_at",
                     "submission_time", "review_time", "timestamp",
                     "SubmissionTime"],
    "rating":       ["rating", "stars", "star_rating", "review_rating",
                     "overall_rating", "score", "Rating"],
    "sentiment":    ["avg_sentiment", "sentiment", "sentiment_score",
                     "compound", "vader_compound"],
    "review_text":  ["review_text", "text", "body", "review_body",
                     "comment", "review_content", "ReviewText"],
}

def resolve_col(df: pd.DataFrame, canonical: str):
    for alias in COL_ALIASES.get(canonical, [canonical]):
        if alias in df.columns:
            return alias
    return None


def vader_sentiment(texts: pd.Series) -> pd.Series:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
        except ImportError:
            print("  VADER not available — sentiment will be NaN")
            return pd.Series(np.nan, index=texts.index)
    sia = SentimentIntensityAnalyzer()
    return texts.fillna("").apply(
        lambda t: sia.polarity_scores(str(t))["compound"]
    )


# Core build function 
def build_timeseries(config: dict):
    platform     = config["platform"]
    review_path  = config["review_file"]
    product_path = config["product_file"]
    out_file     = config["out_file"]

    # 1. Load reviews
    if not review_path.exists():
        print(f"[{platform}] Review file not found: {review_path}")
        return None

    print(f"[{platform}] Loading reviews : {review_path.name}")
    df = pd.read_csv(review_path, dtype=str, low_memory=False)
    print(f"[{platform}] Raw review rows : {len(df):,}")
    print(f"[{platform}] Columns         : {list(df.columns)}")

    # 2. Join product metadata
    if product_path.exists():
        print(f"[{platform}] Loading products: {product_path.name}")
        products = pd.read_csv(product_path, dtype=str, low_memory=False)

        pid_p = resolve_col(products, "product_id")
        if pid_p is None:
            print(f"[{platform}] Cannot find product_id in products file — skipping join")
        else:
            if pid_p != "product_id":
                products = products.rename(columns={pid_p: "product_id"})

            meta_want = ["brand", "brand_normalized", "category",
                         "price", "price_vs_category_median"]
            meta_cols = ["product_id"] + [
                c for c in meta_want
                if c in products.columns and c not in df.columns
            ]
            prod_id_col = resolve_col(products, "product_id")
            rev_id_col  = resolve_col(df, "product_id")
            if prod_id_col and rev_id_col:
                if prod_id_col != "product_id":
                    products = products.rename(columns={prod_id_col: "product_id"})
                if rev_id_col != "product_id":
                    df = df.rename(columns={rev_id_col: "product_id"})
                meta_cols = [c for c in ["product_id", "brand", "category", "price"]
                             if c in products.columns]
                df = df.merge(products[meta_cols], on="product_id", how="left")
            print(f"[{platform}] After product join — columns: {list(df.columns)}")
    else:
        print(f"[{platform}] Product file not found ({product_path.name}) "
              f"— continuing without metadata")

    # 3. Resolve and rename core columns
    col_id   = resolve_col(df, "product_id")
    col_date = resolve_col(df, "review_date")
    col_rate = resolve_col(df, "rating")
    col_sent = resolve_col(df, "sentiment")
    col_text = resolve_col(df, "review_text")

    missing = [k for k, v in {"product_id": col_id,
                               "review_date": col_date,
                               "rating": col_rate}.items() if v is None]
    if missing:
        print(f"[{platform}] Missing required columns: {missing}")
        print(f"[{platform}] Available columns: {list(df.columns)}")
        return None

    print(f"[{platform}] Resolved → "
          f"id={col_id}, date={col_date}, rating={col_rate}, "
          f"sentiment={col_sent}, text={col_text}")

    rename_map = {
        col_id:   "product_id",
        col_date: "review_date",
        col_rate: "rating",
    }
    if col_sent and col_sent != "sentiment":
        rename_map[col_sent] = "sentiment"
    if col_text and col_text != "review_text":
        rename_map[col_text] = "review_text"
    df = df.rename(columns=rename_map)

    # 4. Parse and clean
    raw = df["review_date"]
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().mean() > 0.5:
        # Majority are numeric — Unix ms timestamps (Ulta)
        df["review_date"] = pd.to_datetime(numeric, unit="ms", errors="coerce")
    else:
        # String dates (Sephora)
        df["review_date"] = pd.to_datetime(raw, errors="coerce")
    n_bad = df["review_date"].isna().sum()
    if n_bad:
        print(f"[{platform}] Dropping {n_bad:,} rows with unparseable dates")
    df = df.dropna(subset=["review_date"])

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").clip(1, 5)
    df = df.dropna(subset=["rating"])

    df["year_month"] = df["review_date"].dt.to_period("M").astype(str)
    print(f"[{platform}] Date range: "
          f"{df['review_date'].min().date()} → {df['review_date'].max().date()}")

    # 5. Sentiment
    if "sentiment" in df.columns:
        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
        print(f"[{platform}] Using existing sentiment scores "
              f"(non-null: {df['sentiment'].notna().sum():,})")
    elif "review_text" in df.columns:
        print(f"[{platform}] Computing VADER sentiment from review text...")
        df["sentiment"] = vader_sentiment(df["review_text"])
    else:
        df["sentiment"] = np.nan
        print(f"[{platform}] No sentiment source — using rating-based proxies")

    if df["sentiment"].notna().any():
        df["is_positive"] = (df["sentiment"] >=  0.05).astype(float)
        df["is_negative"] = (df["sentiment"] <= -0.05).astype(float)
        df["is_neutral"]  = (
            (df["sentiment"] > -0.05) & (df["sentiment"] < 0.05)
        ).astype(float)
    else:
        df["is_positive"] = (df["rating"] >= 4).astype(float)
        df["is_negative"] = (df["rating"] <= 2).astype(float)
        df["is_neutral"]  = (df["rating"] == 3).astype(float)

    df["is_5star"] = (df["rating"] == 5).astype(float)
    df["is_1star"] = (df["rating"] == 1).astype(float)
    # 6. Monthly aggregation
    print(f"[{platform}] Aggregating to product × month...")

    stable_meta = [c for c in ["brand", "brand_normalized", "category", "price"]
                   if c in df.columns]
    group_cols  = ["product_id", "year_month"] + stable_meta

    agg = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            review_count  = ("rating",       "count"),
            avg_rating    = ("rating",        "mean"),
            rating_std    = ("rating",        "std"),
            pct_5star     = ("is_5star",      "mean"),
            pct_1star     = ("is_1star",      "mean"),
            avg_sentiment = ("sentiment",     "mean"),
            pct_positive  = ("is_positive",   "mean"),
            pct_negative  = ("is_negative",   "mean"),
            pct_neutral   = ("is_neutral",    "mean"),
        )
    )
    print(f"[{platform}] Agg columns: {list(agg.columns)}")

    # 7. Rolling & lagged features per product
    agg = agg.sort_values(["product_id", "year_month"]).reset_index(drop=True)

    def add_rolling(grp):
        grp = grp.sort_values("year_month").copy()

        # Velocity
        grp["velocity_3m"]        = grp["review_count"].rolling(3, min_periods=1).mean()
        grp["velocity_delta"]     = grp["review_count"].diff()
        grp["velocity_delta_pct"] = grp["review_count"].pct_change() * 100

        # Rating
        grp["avg_rating_3m"] = grp["avg_rating"].rolling(3, min_periods=1).mean()
        grp["rating_trend"]  = (
            grp["avg_rating"]
            .rolling(3, min_periods=2)
            .apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0]
                          if len(x) >= 2 else np.nan,
                raw=True,
            )
        )

        # Negative shock: pct_negative spikes > 1.5× its own 3m rolling mean
        grp["pct_negative_3m"] = grp["pct_negative"].rolling(3, min_periods=1).mean()
        grp["negative_shock"]  = (
            grp["pct_negative"] > (grp["pct_negative_3m"] * 1.5)
        ).astype(int)

        # Lags for shock-response modelling in notebook 3.3
        grp["velocity_lag1"]  = grp["review_count"].shift(1)
        grp["velocity_lag2"]  = grp["review_count"].shift(2)
        grp["neg_shock_lag1"] = grp["negative_shock"].shift(1)

        return grp

    agg = agg.sort_values(["product_id", "year_month"]).reset_index(drop=True)
    g = agg.groupby("product_id")

    agg["velocity_3m"]        = g["review_count"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    agg["velocity_delta"]     = g["review_count"].transform(lambda x: x.diff())
    agg["velocity_delta_pct"] = g["review_count"].transform(lambda x: x.pct_change() * 100)
    agg["avg_rating_3m"]      = g["avg_rating"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    agg["rating_trend"]       = g["avg_rating"].transform(
        lambda x: x.rolling(3, min_periods=2)
                    .apply(lambda w: np.polyfit(range(len(w)), w, 1)[0]
                                    if len(w) >= 2 else np.nan, raw=True)
    )
    agg["pct_negative_3m"]    = g["pct_negative"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    agg["negative_shock"]     = (agg["pct_negative"] > (agg["pct_negative_3m"] * 1.5)).astype(int)
    agg["velocity_lag1"]      = g["review_count"].transform(lambda x: x.shift(1))
    agg["velocity_lag2"]      = g["review_count"].transform(lambda x: x.shift(2))
    agg["neg_shock_lag1"]     = g["negative_shock"].transform(lambda x: x.shift(1))

    # 8. Save
    out_file.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_file, index=False)

    n_products = agg["product_id"].nunique()
    n_months   = agg["year_month"].nunique()
    date_range = f"{agg['year_month'].min()} → {agg['year_month'].max()}"
    shock_rate = agg["negative_shock"].mean()

    print(f"[{platform}] ✓  Saved: {out_file.name}")
    print(f"             Rows       : {len(agg):,}")
    print(f"             Products   : {n_products:,}")
    print(f"             Months     : {n_months}")
    print(f"             Range      : {date_range}")
    print(f"             Shock rate : {shock_rate:.1%} of product-months flagged")
    print()
    return agg


if __name__ == "__main__":
    print("  build_review_timeseries.py")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = {}
    for cfg in CONFIGS:
        ts = build_timeseries(cfg)
        if ts is not None:
            results[cfg["platform"]] = ts

    print(f"  Completed {len(results)}/2 platforms")

    for platform, ts in results.items():
        print(f"\n{platform} — sample (first product, up to 6 months):")
        pid = ts["product_id"].iloc[0]
        display_cols = ["product_id", "year_month", "review_count",
                        "avg_rating", "pct_negative", "negative_shock",
                        "velocity_3m", "rating_trend"]
        display_cols = [c for c in display_cols if c in ts.columns]
        print(ts[ts["product_id"] == pid][display_cols]
              .head(6).to_string(index=False))