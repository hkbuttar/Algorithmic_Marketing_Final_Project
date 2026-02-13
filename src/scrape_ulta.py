# scrape_ulta.py
# Selenium (undetected-chromedriver) Ulta scraper
# - Collects brand URLs from /brands
# - Visits each brand landing page, scrolls to load product grid, extracts product URLs
# - Visits each product page to extract: brand, name, category (breadcrumb), price, rating, review_count
# - Pulls reviews via PowerReviews Read API (pure requests, no Selenium needed)
#
# Outputs:
#   data/raw/ulta_products.csv
#   data/raw/ulta_reviews.csv

import time
import random
import re
import json
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# -------------------------
# Config
# -------------------------

BASE = "https://www.ulta.com"
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tune these for your scale
BRAND_LIMIT         = 500      # how many brands to scrape
PRODUCTS_PER_BRAND  = 50       # max product URLs per brand
MAX_BRAND_SCROLLS   = 40       # scroll iterations per brand page
PRODUCT_LIMIT_TOTAL = 5000     # global cap across brands (safety)
MAX_REVIEWS_PER_PRODUCT = 300  # max reviews per product

# Selenium timeouts
PAGE_LOAD_SLEEP      = 6
WAIT_GRID_SECONDS    = 20

# PowerReviews config
PR_MERCHANT_ID = "6406"
PR_API_KEY     = "daa0f241-c242-4483-afb7-4449942d1a2b"
PR_PAGE_SIZE   = 25            # max per request (PowerReviews cap)
PR_TIMEOUT     = 20


# -------------------------
# Helpers
# -------------------------

def polite_sleep(a=3.0, b=7.0):
    time.sleep(random.uniform(a, b))

def norm_url(u: str) -> str:
    if not u:
        return u
    u = u.split("?")[0].strip()
    if u.startswith("/"):
        u = BASE + u
    return u

def extract_product_id(url: str):
    """Extract pimprod ID from Ulta URL like /p/product-name-pimprod2015889"""
    m = re.search(r"(pimprod\d+)", url or "")
    return m.group(1) if m else None

def is_access_denied(html: str) -> bool:
    if not html:
        return False
    h = html.lower()
    return ("access denied" in h) or ("reference #" in h) or ("errors.edgesuite.net" in h)

def is_404_url(current_url: str) -> bool:
    return "error/404" in (current_url or "").lower() or "/404" in (current_url or "")

def safe_text(soup, selector):
    el = soup.select_one(selector)
    return el.get_text(strip=True) if el else None

def safe_attr(soup, selector, attr):
    el = soup.select_one(selector)
    return el.get(attr) if el and el.has_attr(attr) else None


# -------------------------
# Selenium driver
# -------------------------

def make_driver(headless=False):
    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")

    driver = uc.Chrome(options=options, headless=headless)
    driver.set_page_load_timeout(60)

    # Warm-up
    driver.get(BASE + "/")
    time.sleep(10)
    driver.execute_script("window.scrollBy(0, 300);")
    time.sleep(3)
    return driver


# -------------------------
# Brand URL collection
# -------------------------

def get_brand_urls(driver, limit=500):
    driver.get(BASE + "/brands")
    time.sleep(PAGE_LOAD_SLEEP)

    # Scroll to load all brand links
    for _ in range(10):
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
        time.sleep(2)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    links = soup.select("a[href*='/brand/']")

    urls = []
    seen = set()

    for a in links:
        href = a.get("href")
        if not href:
            continue
        full = norm_url(href)

        # Only keep direct brand pages like /brand/brand-name
        path = full.replace(BASE, "").rstrip("/")
        parts = path.strip("/").split("/")
        if len(parts) != 2 or parts[0] != "brand":
            continue

        slug = parts[-1].lower()

        # Skip non-brand pages
        if slug in {"", "offers", "sale", "new", "gifts", "brands"}:
            continue

        if full not in seen:
            seen.add(full)
            urls.append(full)

        if len(urls) >= limit:
            break

    return urls


# -------------------------
# Product URL extraction from brand page (rendered grid)
# -------------------------

def wait_for_any(driver, selectors, timeout=20):
    """Wait until ANY selector is present in DOM."""
    end = time.time() + timeout
    while time.time() < end:
        for sel in selectors:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            if els:
                return sel
        time.sleep(0.5)
    return None

def scroll_and_collect_product_links(driver, limit=50, max_scrolls=40):
    """
    Collect product links from rendered page.
    Ulta product URLs look like: /p/product-name-pimprod2015889
    """
    product_urls = []
    seen = set()

    last_count = 0
    stagnant_rounds = 0

    for _ in range(max_scrolls):
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        for a in soup.select("a[href*='/p/']"):
            href = a.get("href")
            if not href:
                continue
            full = norm_url(href)
            if not full:
                continue
            # Must contain pimprod ID
            if not re.search(r"pimprod\d+", full):
                continue
            if full not in seen:
                seen.add(full)
                product_urls.append(full)
                if len(product_urls) >= limit:
                    return product_urls

        # Scroll down to trigger lazy loading
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
        time.sleep(2.5)

        # Detect stagnation
        if len(product_urls) == last_count:
            stagnant_rounds += 1
        else:
            stagnant_rounds = 0
            last_count = len(product_urls)

        if stagnant_rounds >= 3:
            break

    return product_urls

def get_product_urls_from_brand(driver, brand_url, limit=50):
    """
    Visit brand URL and extract product URLs.
    Includes retries if page partially hydrates.
    """
    base = brand_url.split("?")[0].rstrip("/")

    for attempt in range(1, 4):
        driver.get(base)
        time.sleep(PAGE_LOAD_SLEEP)

        cur = driver.current_url
        html = driver.page_source

        if is_404_url(cur):
            print("    ⚠️ 404 page detected. Skipping brand.")
            return []

        if is_access_denied(html):
            print("    ⚠️ Access Denied detected. Skipping brand.")
            return []

        # Wait for product grid
        grid_sel = wait_for_any(
            driver,
            selectors=[
                "div[class*='ProductListingResults']",
                "div[class*='product-listing']",
                "div[class*='ProductCard']",
                "a[href*='pimprod']",
                "main",
            ],
            timeout=WAIT_GRID_SECONDS,
        )

        if not grid_sel:
            time.sleep(3)

        urls = scroll_and_collect_product_links(driver, limit=limit, max_scrolls=MAX_BRAND_SCROLLS)

        if urls:
            return urls

        print(f"    ⚠️ No product tiles found (attempt {attempt}/3). Refreshing…")
        driver.refresh()
        time.sleep(6)

    return []


# -------------------------
# Product page scraping (price + meta)
# -------------------------

def scrape_product_page(driver, url):
    driver.get(url)
    time.sleep(PAGE_LOAD_SLEEP)

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    pid = extract_product_id(url)

    brand = None
    name = None
    price = None
    rating = None
    category = None
    review_count = None

    # --- Strategy 1: JSON-LD structured data ---
    for script in soup.select("script[type='application/ld+json']"):
        try:
            ld = json.loads(script.string)
            if isinstance(ld, list):
                ld = ld[0]
            if ld.get("@type") != "Product":
                continue
            if not name:
                name = ld.get("name")
            if not brand:
                b = ld.get("brand", {})
                brand = b.get("name") if isinstance(b, dict) else b
            if not price:
                offers = ld.get("offers", {})
                if isinstance(offers, list):
                    offers = offers[0]
                if isinstance(offers, dict):
                    p = offers.get("price") or offers.get("lowPrice")
                    if p:
                        price = f"${p}"
            if not rating:
                agg = ld.get("aggregateRating", {})
                if isinstance(agg, dict) and agg.get("ratingValue"):
                    rating = agg.get("ratingValue")
                    review_count = agg.get("reviewCount")
            if not category:
                category = ld.get("category")
        except Exception:
            continue

    # --- Strategy 2: CSS selectors ---
    if not name:
        for sel in [
            "h1.ProductMainSection__productName",
            "h1[class*='productName']",
            "h1[class*='ProductName']",
            "span[class*='ProductName']",
            "h1",
        ]:
            name = safe_text(soup, sel)
            if name and len(name) > 2 and name.lower() != "ulta":
                break
            name = None

    if not brand:
        for sel in [
            "a.ProductMainSection__brandName",
            "a[class*='brandName']",
            "a[class*='BrandName']",
            "span[class*='brandName']",
        ]:
            brand = safe_text(soup, sel)
            if brand:
                break

    if not price:
        for sel in [
            "span[class*='Price']",
            "div[class*='Price']",
            "span.ProductPricingPanel__price",
            "[data-testid='product-price']",
        ]:
            price = safe_text(soup, sel)
            if price and "$" in price:
                break
            price = None

    # --- Strategy 3: Meta tags ---
    if not name:
        og_title = safe_attr(soup, "meta[property='og:title']", "content")
        if og_title:
            name = og_title.split("|")[0].split("-")[0].strip()

    if not brand:
        brand = safe_attr(soup, "meta[property='product:brand']", "content")

    if not price:
        p = (
            safe_attr(soup, "meta[property='product:price:amount']", "content")
            or safe_attr(soup, "meta[property='og:price:amount']", "content")
        )
        if p:
            price = f"${p}" if not p.startswith("$") else p

    # --- Strategy 4: Regex on raw HTML ---
    if not price:
        m = re.search(r'"price"\s*:\s*"?\$?(\d+\.?\d*)"?', html)
        if m:
            price = f"${m.group(1)}"

    if not rating:
        m = re.search(r'"average_rating"\s*:\s*([\d.]+)', html)
        if m:
            rating = m.group(1)

    # --- Category from breadcrumbs ---
    if not category:
        bc_items = soup.select("nav[aria-label='breadcrumb'] li, nav.Breadcrumb li, ol.breadcrumb li")
        if bc_items:
            texts = [li.get_text(strip=True) for li in bc_items if li.get_text(strip=True).lower() != "home"]
            if texts:
                category = texts[-1]

    if not category:
        bc = driver.find_elements(By.CSS_SELECTOR, "nav[aria-label='breadcrumb'] li")
        if bc:
            category = bc[-1].text.strip()

    return {
        "product_id": pid,
        "product_url": url,
        "brand": brand,
        "product_name": name,
        "category": category,
        "price": price,
        "rating": rating,
    }


# -------------------------
# PowerReviews: Fetch reviews
# -------------------------

def fetch_reviews(product_id, max_reviews=300):
    """
    Fetch reviews from PowerReviews Read API.
    No Selenium needed — pure HTTP requests.
    Max 300 reviews per product, 25 per page.
    """
    if not product_id:
        return []

    base_url = (
        f"https://readservices-b2c.powerreviews.com"
        f"/m/{PR_MERCHANT_ID}/l/en_US/product/{product_id}/reviews"
    )

    all_reviews = []
    max_pages = (max_reviews + PR_PAGE_SIZE - 1) // PR_PAGE_SIZE  # ceiling division = 12

    for page in range(max_pages):
        offset = page * PR_PAGE_SIZE
        params = {
            "apikey": PR_API_KEY,
            "paging.size": PR_PAGE_SIZE,
            "paging.from": offset,
            "sort": "Newest",
            "image_only": "false",
        }

        try:
            r = requests.get(base_url, params=params, timeout=PR_TIMEOUT)
            if r.status_code != 200:
                print(f"    API Error {r.status_code} for {product_id}")
                break

            data = r.json()

            results = data.get("results", [])
            if not results:
                break

            # Reviews are nested: results[0]["reviews"]
            reviews_list = results[0].get("reviews", [])
            if not reviews_list:
                break

            total_results = data.get("paging", {}).get("total_results", 0)

            for rev in reviews_list:
                details = rev.get("details", {})
                metrics = rev.get("metrics", {})
                badges = rev.get("badges", {})

                all_reviews.append({
                    "pd_id": product_id,
                    "review_id": rev.get("ugc_id"),
                    "Rating": metrics.get("rating"),
                    "headline": details.get("headline"),
                    "ReviewText": details.get("comments"),
                    "SubmissionTime": details.get("created_date"),
                    "nickname": details.get("nickname"),
                    "location": details.get("location"),
                    "bottom_line": details.get("bottom_line"),
                    "helpful_votes": metrics.get("helpful_votes", 0),
                    "not_helpful_votes": metrics.get("not_helpful_votes", 0),
                    "is_verified_buyer": badges.get("is_verified_buyer", False),
                    "is_verified_reviewer": badges.get("is_verified_reviewer", False),
                    "disclosure_code": details.get("disclosure_code"),
                })

            if len(all_reviews) >= total_results or len(all_reviews) >= max_reviews:
                break

            time.sleep(0.3)  # Light throttle (1800 calls / 5 min is generous)

        except Exception as e:
            print(f"    Request failed: {e}")
            break

    # Trim to exact max
    all_reviews = all_reviews[:max_reviews]

    if all_reviews:
        print(f"    ✅ Found {len(all_reviews)} reviews for {product_id}")

    return all_reviews


# -------------------------
# Main
# -------------------------

def main():
    driver = make_driver(headless=False)

    # Load ALL existing product IDs across all files
    all_existing_pids = set()

    for csv_file in OUTPUT_DIR.glob("ulta_products*.csv"):
        try:
            df = pd.read_csv(csv_file)
            all_existing_pids.update(df["product_id"].dropna().astype(str))
            print(f"📂 Loaded {len(df)} products from {csv_file.name}")
        except Exception:
            continue

    print(f"📂 Total already-scraped products: {len(all_existing_pids)}")

    # Track scraped brand slugs using exact URL slugs
    brands_done_path = OUTPUT_DIR / "ulta_scraped_brand_slugs.txt"
    if brands_done_path.exists():
        all_existing_brand_slugs = set(brands_done_path.read_text().strip().splitlines())
    else:
        all_existing_brand_slugs = set()

    print(f"📂 Already-scraped brand slugs: {len(all_existing_brand_slugs)}")

    print("Collecting brands...")
    brand_urls = get_brand_urls(driver, limit=BRAND_LIMIT)
    print(f"✅ Collected {len(brand_urls)} real brand URLs")

    # Output files
    products_path = OUTPUT_DIR / "ulta_products.csv"
    reviews_path  = OUTPUT_DIR / "ulta_reviews.csv"

    # Load partial progress
    if products_path.exists():
        try:
            existing_run = pd.read_csv(products_path)
            all_existing_pids.update(existing_run["product_id"].dropna().astype(str))
            print(f"📂 Resuming: {len(existing_run)} products already in {products_path.name}")
        except Exception:
            pass

    seen_product_ids = set(all_existing_pids)
    new_products_this_run = 0

    for brand_url in brand_urls:
        brand_slug = brand_url.rstrip("/").split("/")[-1].lower()

        # Skip brands already scraped
        if brand_slug in all_existing_brand_slugs:
            print(f"⏭️  Skipping already-scraped brand: {brand_slug}")
            continue

        if len(seen_product_ids) >= PRODUCT_LIMIT_TOTAL:
            print("Reached global product cap; stopping.")
            break

        print(f"\nBrand: {brand_url}")

        # Pace between brands
        polite_sleep(5.0, 12.0)

        product_urls = get_product_urls_from_brand(driver, brand_url, limit=PRODUCTS_PER_BRAND)

        if not product_urls:
            if is_access_denied(driver.page_source):
                print("    🚫 Access Denied! Restarting browser with fresh session...")
                driver.quit()
                time.sleep(random.uniform(60, 90))
                driver = make_driver(headless=False)
                product_urls = get_product_urls_from_brand(driver, brand_url, limit=PRODUCTS_PER_BRAND)

            if not product_urls:
                print("    ⚠️ No product URLs found. Skipping brand.")
                all_existing_brand_slugs.add(brand_slug)
                with open(brands_done_path, "a") as f:
                    f.write(brand_slug + "\n")
                continue

        # Early exit: if first 5 products are all already known, skip
        first_pids = []
        for u in product_urls[:5]:
            pid = extract_product_id(u)
            if pid:
                first_pids.append(pid)
        if first_pids and all(pid in seen_product_ids for pid in first_pids):
            print(f"  ⏭️  All sample products already scraped. Skipping brand.")
            all_existing_brand_slugs.add(brand_slug)
            with open(brands_done_path, "a") as f:
                f.write(brand_slug + "\n")
            continue

        brand_products = []
        brand_reviews = []

        for url in product_urls:
            if len(seen_product_ids) >= PRODUCT_LIMIT_TOTAL:
                break

            pid = extract_product_id(url)
            if pid and pid in seen_product_ids:
                print(f"  ⏭️  Already have {url}, skipping.")
                continue

            print(f"  Product: {url}")

            prod = scrape_product_page(driver, url)

            if not prod or not prod.get("product_id"):
                print("    ⚠️ Failed to parse product page.")
                continue

            pid = prod["product_id"]
            if pid in seen_product_ids:
                continue
            seen_product_ids.add(pid)

            brand_products.append(prod)

            # Fetch reviews via PowerReviews API (no Selenium!)
            revs = fetch_reviews(pid, max_reviews=MAX_REVIEWS_PER_PRODUCT)

            if revs:
                brand_reviews.extend(revs)
            else:
                print(f"    ⚠️ No reviews found for {pid}")

            polite_sleep(5.0, 12.0)

        # --- Save after every brand (incremental / crash-safe) ---
        if brand_products:
            pd.DataFrame(brand_products).to_csv(
                products_path,
                mode="a",
                header=not products_path.exists(),
                index=False,
            )
            new_products_this_run += len(brand_products)

        if brand_reviews:
            pd.DataFrame(brand_reviews).to_csv(
                reviews_path,
                mode="a",
                header=not reviews_path.exists(),
                index=False,
            )

        # Mark brand as done
        all_existing_brand_slugs.add(brand_slug)
        with open(brands_done_path, "a") as f:
            f.write(brand_slug + "\n")

        print(f"  💾 Saved {len(brand_products)} products, {len(brand_reviews)} reviews for {brand_slug}")

    driver.quit()

    print("\n--- Scrape Complete ---")
    print(f"New products this run: {new_products_this_run}")
    print(f"Total products across all files: {len(seen_product_ids)}")


if __name__ == "__main__":
    main()