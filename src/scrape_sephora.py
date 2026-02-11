# scrape_sephora.py
# Sephora scraper using undetected-chromedriver + Bazaarvoice reviews
# Designed for small-to-medium scale academic use

import time
import random
import requests
import pandas as pd
import json
import re
from bs4 import BeautifulSoup
import undetected_chromedriver as uc

# -------------------------
# Helpers
# -------------------------

def polite_sleep(a=3, b=7):
    time.sleep(random.uniform(a, b))

# -------------------------
# Driver (anti-bot safe)
# -------------------------

def make_driver():
    driver = uc.Chrome(headless=False)
    driver.get("https://www.sephora.com/")
    time.sleep(10)  # critical: Akamai + consent cookies
    return driver

# -------------------------
# Step 1: Get brand URLs
# -------------------------

def get_brand_urls(driver, limit=6):
    driver.get("https://www.sephora.com/brands-list")
    time.sleep(8)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    links = soup.select("a[href^='/brand/']")

    brand_urls = []
    for a in links:
        href = a.get("href")
        if href:
            brand_urls.append("https://www.sephora.com" + href)
        if len(brand_urls) >= limit:
            break

    return brand_urls

# -------------------------
# Step 2: Get product URLs
# -------------------------

def get_product_urls_from_brand(driver, brand_url, limit=25):
    base = brand_url.split("?")[0]
    parts = base.split("/")
    base = "/".join(parts[:5])  # https://www.sephora.com/brand/<brand>

    driver.get(base)
    time.sleep(10)

    if "error/404" in driver.current_url.lower():
        print("⚠️ 404 page detected. Skipping brand.")
        return []

    if "Access Denied" in driver.page_source or "Reference #" in driver.page_source:
        print("⚠️ Access Denied detected. Skipping brand.")
        return []

    # Human-like scrolling
    for _ in range(3):
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight / 2);")
        time.sleep(4)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    product_urls = set()
    for a in soup.select("a[href^='/product/']"):
        href = a.get("href")
        if href:
            product_urls.add("https://www.sephora.com" + href.split("?")[0])
        if len(product_urls) >= limit:
            break

    return list(product_urls)

# -------------------------
# Price extraction (JSON)
# -------------------------

def extract_price_from_json(html):
    try:
        start = html.find("__NEXT_DATA__")
        if start == -1:
            return None

        json_start = html.find("{", start)
        json_end = html.find("</script>", json_start)
        data = json.loads(html[json_start:json_end])

        price = (
            data.get("props", {})
                .get("pageProps", {})
                .get("product", {})
                .get("currentSku", {})
                .get("listPrice")
        )

        return f"${price}" if price else None
    except Exception:
        return None

# -------------------------
# Step 3: Scrape product metadata
# -------------------------

def scrape_product_page(driver, url):
    driver.get(url)
    time.sleep(8)

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    def safe(sel):
        el = soup.select_one(sel)
        return el.get_text(strip=True) if el else None

    match = re.search(r"P(\d+)", url)
    product_id = match.group(1) if match else None

    price = extract_price_from_json(html)

    return {
        "product_id": product_id,
        "product_url": url,
        "brand": safe("a[data-at='brand_name']"),
        "product_name": safe("span[data-at='product_name']"),
        "category": safe("nav a:last-child"),
        "price": price,
        "rating": safe("span[data-at='rating']"),
        "review_count": safe("span[data-at='review_count']")
    }

# -------------------------
# Step 4: Reviews via Bazaarvoice
# -------------------------

def fetch_reviews(product_id, max_pages=2):
    reviews = []

    if not product_id:
        return reviews

    product_ids_to_try = [product_id, f"P{product_id}"]

    for pid in product_ids_to_try:
        for page in range(max_pages):
            params = {
                "Filter": f"ProductId:{pid}",
                "Sort": "SubmissionTime:desc",
                "Limit": 20,
                "Offset": page * 20,
                "passkey": "sephora",
                "apiversion": "5.4"
            }

            r = requests.get(
                "https://api.bazaarvoice.com/data/reviews.json",
                params=params,
                timeout=10
            )

            if r.status_code != 200:
                break

            data = r.json().get("Results", [])
            if not data:
                break

            for rev in data:
                reviews.append({
                    "product_id": pid,
                    "rating": rev.get("Rating"),
                    "text": rev.get("ReviewText"),
                    "date": rev.get("SubmissionTime")
                })

            polite_sleep()

        if reviews:
            break

    return reviews

# -------------------------
# Main pipeline
# -------------------------

def main():
    driver = make_driver()

    print("Collecting brands...")
    brand_urls = get_brand_urls(driver, limit=1)

    products = []
    reviews = []

    for brand in brand_urls:
        print("Brand:", brand)
        product_urls = get_product_urls_from_brand(driver, brand, limit=1)

        for url in product_urls:
            # Filter non-products early
            if "subscription" in url.lower():
                continue

            print("  Product:", url)
            prod = scrape_product_page(driver, url)

            if prod["brand"] is None:
                continue

            products.append(prod)

            revs = fetch_reviews(prod["product_id"], max_pages=2)
            reviews.extend(revs)

            polite_sleep()

    driver.quit()

    pd.DataFrame(products).to_csv("data/raw/sephora_products.csv", index=False)
    pd.DataFrame(reviews).to_csv("data/raw/sephora_reviews.csv", index=False)

    print("Done.")
    print("Products:", len(products))
    print("Reviews:", len(reviews))

if __name__ == "__main__":
    main()
