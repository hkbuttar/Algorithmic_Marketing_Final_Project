markdown

# Algorithmic Marketing Optimization in the Beauty Industry

### A Cross-Retailer Framework: Sephora x Ulta

**Segmentation, Positioning, Perceived Value, Price Context, and Causal Attribution**

---

## Project Overview

This project builds an end-to-end algorithmic marketing framework for beauty products using publicly available e-commerce product and review data from **Sephora** and **Ulta** -- the two dominant specialty beauty retailers in the United States. The objective is to demonstrate how modern data science techniques can be applied to real-world digital marketing problems by jointly modeling consumer perception, perceived value, price context, and demand dynamics **within and across** retail platforms.

The beauty industry is highly competitive, review-driven, and reputation-sensitive. While prices are publicly observable, true demand, willingness-to-pay, and consumer expectations are not -- and they may differ systematically between Sephora and Ulta customer bases. Brands must therefore make product, pricing, and marketing decisions under uncertainty that varies by retail channel. This project addresses that challenge by combining large-scale review data with machine learning, natural language processing (NLP), pricing context, and causal inference to generate actionable marketing insights grounded in how consumers perceive value relative to price -- and how those perceptions diverge or converge across platforms.

### Why Two Retailers?

Sephora and Ulta serve overlapping but distinct consumer segments. Sephora skews prestige and trend-forward; Ulta spans mass-to-prestige with a broader price range. Many brands sell on both platforms, creating a natural laboratory for cross-retailer comparison. By analyzing each platform independently and then jointly, this project surfaces insights that a single-retailer analysis would miss: channel-specific consumer expectations, cross-platform pricing inconsistencies, and retailer-dependent brand perception gaps.

---

## Unifying Business Problem

**How should beauty brands optimize their product, pricing, and marketing strategies across Sephora and Ulta to maximize perceived value, consumer engagement, and competitive positioning -- given that consumer expectations, sentiment drivers, and price sensitivity may differ by retail channel?**

This question anchors all analyses. Each component contributes a piece of the answer, and the project converges on a set of cross-retailer strategic recommendations for brand and product management.

---

## Data Sources

All data is sourced from publicly accessible e-commerce platforms. No login-restricted, proprietary, or personal user data is collected.

### Sephora

**Product Data:**
| Field | Description |
|-------|-------------|
| `product_id` | Unique product identifier (P-number from URL) |
| `product_url` | Public product page URL |
| `brand` | Brand name |
| `product_name` | Product display name |
| `category` | Product category (from breadcrumbs) |
| `price` | Listed price |
| `rating` | Average star rating |

**Review Data:**
| Field | Description |
|-------|-------------|
| `pd_id` | Product identifier (joins to product data) |
| `Rating` | Individual review star rating |
| `ReviewText` | Full review text |
| `SubmissionTime` | Review timestamp |
| `Helpfulness` | Helpfulness votes |
| `skinTone` | Reviewer self-reported skin tone |
| `skinType` | Reviewer self-reported skin type |

### Ulta

**Product Data:**
| Field | Description |
|-------|-------------|
| `product_id` | Unique product identifier (pimprod ID from URL) |
| `product_url` | Public product page URL |
| `brand` | Brand name |
| `product_name` | Product display name |
| `category` | Product category (from breadcrumbs) |
| `price` | Listed price |
| `rating` | Average star rating |

**Review Data:**
| Field | Description |
|-------|-------------|
| `pd_id` | Product identifier (joins to product data) |
| `review_id` | Unique review identifier |
| `Rating` | Individual review star rating |
| `headline` | Review headline/title |
| `ReviewText` | Full review text |
| `SubmissionTime` | Review timestamp |
| `nickname` | Reviewer display name |
| `location` | Reviewer location |
| `bottom_line` | Would-recommend indicator |
| `helpful_votes` | Count of helpful votes |
| `not_helpful_votes` | Count of not-helpful votes |
| `is_verified_buyer` | Verified purchase flag |
| `is_verified_reviewer` | Verified reviewer flag |
| `disclosure_code` | Disclosure/incentive indicator |

### Demand and Value Proxies

Demand and consumer response are proxied using:

- Review volume and velocity over time
- Rating levels and dispersion
- Sentiment and topic dynamics
- Helpfulness and engagement signals
- Price relative to peer products and segment norms
- Verified vs. unverified reviewer behavior (Ulta)

---

## Data Collection Architecture

Both scrapers follow a shared pipeline pattern with retailer-specific adaptations:

**Shared Pattern:** Brand list collection, brand page scroll-to-load, product page multi-strategy extraction, review API ingestion, and incremental CSV writes (crash-safe, resumable via brand slug tracking).

**Sephora (`src/scrape_sephora.py`):**
- Brand URLs collected from `/brands-list`
- Product IDs extracted as P-numbers from URLs (e.g., `P504942`)
- Product metadata extracted via cascading strategies: Selenium JS context (`window.Sephora.productPage`), CSS selectors, meta tags, JSON-LD, `__NEXT_DATA__` JSON, and regex fallbacks
- Reviews fetched via **Bazaarvoice API** (`api.bazaarvoice.com/data/reviews.json`), paginated at 100 per request, with `ContextDataValues` for skin tone and skin type

**Ulta (`src/scrape_ulta.py`):**
- Brand URLs collected from `/brand/all`
- Product IDs extracted as pimprod identifiers from URLs (e.g., `pimprod2015889`)
- Product metadata extracted via cascading strategies: JSON-LD, CSS selectors, meta tags, and regex fallbacks
- Reviews fetched via **PowerReviews Read API** (`readservices-b2c.powerreviews.com`), paginated at 25 per request, with verified buyer badges, helpfulness votes, and disclosure codes

**Anti-detection and resilience:** Both scrapers use `undetected-chromedriver` with warm-up navigation, randomized polite sleeps, access-denial detection with automatic session rotation, and stagnation-aware scroll loops.

---

## Getting Started

### Prerequisites

- Python 3.9+
- Google Chrome installed

### Setup

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Connect to a VPN** to reduce the likelihood of IP-based rate limiting or access blocks during scraping.

### Running the Scrapers

**Scrape Sephora:**
```bash
python src/scrape_sephora.py
```

**Scrape Ulta:**
```bash
python src/scrape_ulta.py
```

Both scrapers are resumable -- if interrupted, they will skip already-scraped brands and products on the next run. Raw output files are written incrementally to `data/raw/`.

---

## Project Architecture

The project is organized into three analysis tiers that build on each other:

**Part 1 -- Independent Analyses:** Sephora and Ulta are each analyzed separately for segmentation, sentiment, brand health, and positioning to establish retailer-specific baselines.

**Part 2 -- Joint Cross-Retailer Analyses:** Brand/product linkage across platforms enables cross-platform segmentation comparison, sentiment gap analysis, price consistency checks, and a combined recommendation engine.

**Part 3 -- Comparative and Convergent Strategy:** Dual-retailer perceptual maps, demand sensitivity modeling, causal attribution with cross-platform spillover analysis, and a unified strategic recommendation framework.

---

## Part 1: Independent Retailer Analyses

Each retailer is analyzed independently to establish baseline insights before cross-platform comparison.

### 1.1 Market Segmentation (per retailer)

Product-level segmentation using clustering algorithms:

- K-Means
- Hierarchical Clustering
- Gaussian Mixture Models

Segments are defined by review volume and velocity, rating distributions and stability, sentiment scores, topic prevalence from review text, and price levels with relative positioning. Resulting segments represent distinct consumer expectation and value-for-money profiles within each retailer ecosystem.

**Sephora-specific features:** Skin tone and skin type dimensions enable demographic-aware segmentation.

**Ulta-specific features:** Verified buyer flags and recommendation indicators (`bottom_line`) allow segmentation by purchase-confirmed behavior and advocacy.

### 1.2 Brand Health and Sentiment Analysis (per retailer)

NLP applied to customer reviews for sentiment scoring, topic modeling, and identification of brand delighters, brand disappointers, and mismatches between star ratings and written sentiment.

Brand health dashboards capture experience drivers, complaint concentration, and whether negative sentiment is driven by performance issues or perceived price unfairness.

**Ulta-specific enrichment:** Review headlines provide an additional signal layer for topic extraction. Verified buyer vs. non-buyer sentiment comparison tests whether incentivized or unverified reviews skew brand perception.

### 1.3 Recommendation Systems (per retailer)

Item-to-item product recommendations using content-based similarity across product attributes, review text embeddings, sentiment and topic alignment, and price proximity and substitution bands. Used to identify complementary products, close substitutes, and trade-up/trade-down opportunities within each retailer catalog.

---

## Part 2: Joint Cross-Retailer Analyses

### 2.1 Brand and Product Linkage

Brands and products appearing on both Sephora and Ulta are matched to create a unified cross-retailer product graph. This linkage enables all downstream comparative analyses.

Matching approach: Brand name normalization, product name fuzzy matching, price and category validation.

### 2.2 Cross-Platform Segmentation Comparison

For matched products, segment assignments from each retailer are compared. Key questions include whether the same product falls into different consumer perception segments depending on the platform, whether Sephora segments skew toward premium expectations while Ulta segments reflect value-orientation, and which brands are perceived consistently vs. inconsistently across retailers.

### 2.3 Sentiment Gap Analysis

For brands present on both platforms, sentiment profiles are compared to surface divergence in consumer language and experience themes, platform-specific delighters and disappointers, and whether the same product receives systematically different ratings or sentiment by channel.

### 2.4 Price Consistency and Perceived Value Divergence

Cross-retailer price comparison for matched products to identify price parity vs. discrepancy, whether price differences correlate with sentiment or rating differences, and how perceived value efficiency (sentiment per dollar, rating per dollar) varies by platform.

### 2.5 Combined Recommendation Engine

A unified recommendation system that operates across both catalogs, enabling cross-retailer substitution recommendations, platform-aware trade-up/trade-down paths, and discovery of products that perform well on one platform but are under-reviewed on the other.

---

## Part 3: Comparative and Convergent Strategy

### 3.1 Competitive Positioning and Perceptual Mapping

Competitive landscape analysis using PCA and distance-based methods. Brands and products are compared across perceived efficacy, experience quality, consumer polarization, market traction (review intensity), and price positioning with perceived value efficiency.

**Dual-retailer perceptual maps** position brands in a shared space, revealing whether a brand occupies different competitive positions on Sephora vs. Ulta, overpriced vs. underpriced perception zones by retailer, perceptual white spaces that exist on one platform but not the other, and strategic repositioning opportunities that are channel-specific.

### 3.2 Perceived Value, Price Context and Demand Sensitivity

Rather than estimating classical price elasticity from sales data, this project examines demand sensitivity to perceived value conditional on price. Models analyze how review velocity, sentiment, and rating behavior respond to negative review shocks, changes in topic prevalence, rating declines, and price deviations from segment norms.

Products and segments are classified as price-resilient, value-fragile, or reputation-sensitive -- with classifications compared across retailers.

**Cross-retailer simulations** explore whether a reputation shock on one platform spills over to the other, whether price sensitivity differs by channel for the same brand, and optimal expectation management and value communication strategies by retailer.

### 3.3 Experimentation and Causal Attribution

Quasi-experimental designs leveraging natural variation in review and pricing dynamics using Difference-in-Differences and matching approaches.

Used to estimate incremental changes in consumer response, brand recovery trajectories after perception shocks, interaction effects between price level and reputation damage, and **cross-retailer spillover effects** -- whether a brand crisis on Sephora affects Ulta performance and vice versa.

Outputs support ROI-driven prioritization of marketing, pricing, and brand investment strategies **differentiated by retail channel**.

### 3.4 Convergent Strategic Recommendations

All analyses converge into a unified strategic framework addressing:

- **For brands on both platforms:** How to optimize positioning, pricing, and messaging per channel while maintaining brand coherence.
- **For brands on one platform:** Whether cross-listing represents an opportunity, and what the target segment and positioning should be.
- **For category managers:** Where the competitive gaps are, which brands are over/under-indexed by platform, and where marketing investment will yield the highest incremental return.
- **For pricing strategy:** Where price harmonization vs. channel-specific pricing is optimal given divergent consumer expectations.

---

## Tools and Methods

- **Language:** Python
- **Data Collection:**
  - **Selenium** via `undetected-chromedriver` -- automated browser sessions with anti-detection, warm-up navigation, scroll-to-load pagination, and session rotation on access denial
  - **BeautifulSoup** -- HTML parsing with multi-strategy extraction (CSS selectors, meta tags, JSON-LD, regex fallbacks)
  - **Bazaarvoice API** -- Sephora review ingestion (`reviews.json` endpoint, paginated, with `ContextDataValues` for skin tone/type)
  - **PowerReviews Read API** -- Ulta review ingestion (paginated via `readservices-b2c.powerreviews.com`, includes verified buyer badges, helpfulness votes, disclosure codes)
  - **Incremental/crash-safe writes** -- per-brand CSV appends with brand slug tracking for resumable runs
- **Data Processing:** pandas, NumPy
- **Machine Learning:** scikit-learn (clustering, classification, regression)
- **NLP:** NLTK, spaCy, TF-IDF, sentiment analysis, topic modeling
- **Statistical Modeling:** statsmodels (regression, causal inference)
- **Causal Inference:** Difference-in-Differences, propensity score matching
- **Visualization:** matplotlib, seaborn, perceptual maps
- **Infrastructure:** PySpark (for scalable processing)

---

## Deliverables

- **Executive-ready marketing insights** and cross-retailer strategic recommendations
- **Reproducible data pipelines** for both Sephora and Ulta scraping, processing, and analysis
- **Visual dashboards** for:
  - Market segmentation (per retailer and comparative)
  - Sentiment and brand health (per retailer and gap analysis)
  - Competitive positioning (dual-retailer perceptual maps)
  - Perceived value vs. price efficiency (cross-platform)
  - Demand sensitivity and causal attribution
- **Technical appendix** documenting assumptions, limitations, and validation
- **Managerial guidance** for beauty brand, product, and pricing strategy across retail channels

---

## Repository Structure

```
data/
    raw/
        sephora_products.csv
        sephora_reviews.csv
        ulta_products.csv
        ulta_reviews.csv
        scraped_brand_slugs.txt
        ulta_scraped_brand_slugs.txt
    processed/
        sephora/
        ulta/
        matched/                  # Cross-retailer linked products
src/
    scrape_sephora.py
    scrape_ulta.py
notebooks/
    02_independent/
        sephora_segmentation.ipynb
        sephora_sentiment.ipynb
        ulta_segmentation.ipynb
        ulta_sentiment.ipynb
    03_joint/
        brand_product_linkage.ipynb
        cross_platform_segmentation.ipynb
        sentiment_gap_analysis.ipynb
        combined_recommendations.ipynb
    04_comparative/
        dual_retailer_positioning.ipynb
        demand_sensitivity.ipynb
        causal_attribution.ipynb
        convergent_strategy.ipynb
outputs/                          # Figures, dashboards, reports
README.md
requirements.txt
.gitignore                        # Excludes data/, outputs/, credentials
```

---

## Use Cases

This project is relevant for digital marketing and growth analytics, consumer packaged goods (CPG) analytics, e-commerce and marketplace strategy teams, brand and product management, pricing and value strategy across retail channels, and applied data science and marketing analytics coursework.

---

## Disclaimer

This project is conducted for educational and research purposes only. All data is sourced from publicly accessible pages and APIs. No proprietary, confidential, or personally identifiable user information is collected.