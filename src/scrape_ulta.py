"""
Scrapes Ulta Beauty product listings and reviews.
Outputs:
- data/raw/ulta_products.csv
- data/raw/ulta_reviews.csv
"""

import requests
import time
import random
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path

# -------------------------
# Config
# -------------------------

BASE_URL = "https://www.ulta.com"
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.ulta.com/",
    "Origin": "https://www.ulta.com",
}

# -------------------------
# Helpers
# -------------------------

def polite_sleep(a=1.5, b=3.5):
    time.sleep(random.uniform(a, b))


def safe_get_text(el):
    return el.get_text(strip=True) if el else None


# -------------------------
# Product Listing Scraper
# -------------------------

def scrape_ulta_category(category, max_pages=5):
    products = []

    for page in range(max_pages):
        params = {
            "q": category,
            "page": page,
            "size": 48,
            "type": "product"
        }

        r = requests.get(
            "https://www.ulta.com/api/search",
            headers=HEADERS,
            params=params,
            timeout=10
        )

        # ---- SAFETY CHECKS ----
        if r.status_code != 200:
            print(f"Ulta API blocked (status {r.status_code}) for {category}")
            break

        try:
            data = r.json()
        except Exception:
            print(f"Non-JSON response for {category}, page {page}")
            break

        items = data.get("products", [])
        if not items:
            break

        for p in items:
            products.append({
                "retailer": "ulta",
                "category": category,
                "product_id": p.get("productId"),
                "brand": p.get("brandName"),
                "product_name": p.get("displayName"),
                "price": p.get("price"),
                "rating": p.get("rating"),
                "review_count": p.get("reviewCount"),
                "product_url": "https://www.ulta.com" + p.get("url")
            })

        polite_sleep(2.0, 4.0)

    return pd.DataFrame(products)


# -------------------------
# Product Detail Scraper
# -------------------------

def scrape_ulta_product_details(product_url):
    r = requests.get(product_url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    return {
        "price": safe_get_text(soup.select_one("span.ProductPricing__price")),
        "rating": safe_get_text(soup.select_one("span.ReviewStars__rating")),
        "review_count": safe_get_text(soup.select_one("span.ReviewStars__count")),
        "description": safe_get_text(soup.select_one("div.ProductDetail__description")),
        "ingredients": safe_get_text(soup.select_one("div.ProductDetail__ingredients"))
    }


# -------------------------
# Review Scraper
# -------------------------

def scrape_ulta_reviews(product_id, max_pages=5):
    reviews = []

    for page in range(max_pages):
        params = {
            "Filter": f"ProductId:{product_id}",
            "Sort": "SubmissionTime:desc",
            "Limit": 20,
            "Offset": page * 20,
            "Include": "Products",
            "Stats": "Reviews",
            "passkey": "ulta",
            "apiversion": "5.4"
        }

        r = requests.get(
            "https://api.bazaarvoice.com/data/reviews.json",
            headers=HEADERS,
            params=params
        )

        if r.status_code != 200:
            break

        data = r.json()
        results = data.get("Results", [])

        if not results:
            break

        for rev in results:
            reviews.append({
                "rating": rev.get("Rating"),
                "title": rev.get("Title"),
                "text": rev.get("ReviewText"),
                "date": rev.get("SubmissionTime"),
                "helpful_votes": rev.get("TotalPositiveFeedbackCount"),
                "product_id": product_id
            })

        polite_sleep()

    return pd.DataFrame(reviews)


# -------------------------
# Main Pipeline
# -------------------------

def main():
    categories = [
        "skincare",
        "haircare",
        "makeup",
        "fragrance",
        "body care",
        "wellness",
        "tools"
    ]

    all_products = []

    # -------------------------
    # Product scraping
    # -------------------------
    for cat in categories:
        print(f"Scraping category: {cat}")
        df = scrape_ulta_category(cat, max_pages=4)
        df["category"] = cat
        all_products.append(df)

    products_df = pd.concat(all_products, ignore_index=True)

    # Save products immediately (important for debugging)
    products_df.to_csv(OUTPUT_DIR / "ulta_products.csv", index=False)

    # -------------------------
    # Reviews
    # -------------------------
    all_reviews = []

    for _, row in products_df.iterrows():
        try:
            if pd.notna(row.get("product_id")):
                df_reviews = scrape_ulta_reviews(
                    row["product_id"],
                    max_pages=3
                )

                if not df_reviews.empty:
                    df_reviews["product_id"] = row["product_id"]
                    df_reviews["brand"] = row["brand"]
                    df_reviews["product_name"] = row["product_name"]
                    df_reviews["category"] = row["category"]
                    all_reviews.append(df_reviews)

        except Exception as e:
            print(f"Review scrape failed for {row.get('product_name')}: {e}")

        polite_sleep()

    if all_reviews:
        reviews_df = pd.concat(all_reviews, ignore_index=True)
        reviews_df.to_csv(OUTPUT_DIR / "ulta_reviews.csv", index=False)

    print("Ulta scraping complete.")


if __name__ == "__main__":
    main()