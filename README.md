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
- Jupyter Lab or Jupyter Notebook (for interactive dashboards)

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

### Running the Pipeline

**Step 1 — Scrape raw data:**
```bash
python src/scrape_sephora.py
python src/scrape_ulta.py
```
Both scrapers are resumable — if interrupted, they will skip already-scraped brands and products on the next run. Raw output files are written incrementally to `data/raw/`.

**Step 2 — Clean and match:**
```bash
python src/data_cleaning.py
python src/brand_matching.py
```

**Step 3 — Build features:**
```bash
python src/build_product_graph.py
python src/segmentation_features.py
```

**Step 4 — Run sentiment pipelines:**
```bash
python src/sephora_sentiment.py
python src/ulta_sentiment.py
```

**Step 5 — Build recommendation systems:**
```bash
python src/sephora_recommendations.py
python src/ulta_recommendations.py
```

**Step 6 — Run notebooks** in the order listed under [Execution Order](#execution-order) to generate segmentation, cross-platform analysis, and strategic outputs.

### Accessing Dashboards

All interactive dashboards are available in two modes:

**Jupyter notebook** (requires a running Jupyter server with `ipywidgets`):
```python
from src.sephora_sentiment import show_dashboard
show_dashboard()
```

**Browser** (no server required — open any exported file directly):
```python
from src.sephora_sentiment import write_all_brand_htmls
write_all_brand_htmls()   # saves to notebooks/independent/outputs/brand_dashboards/
```

See the [Code Reference](#code-reference) section for the equivalent calls for Ulta sentiment, Sephora recommendations, and Ulta recommendations.

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
ALGORITHMIC_MARKETING_FINAL_PROJECT/
├── data/
│   ├── processed/
│   │   ├── Matched/
│   │   │   ├── brand_mapping.csv
│   │   │   ├── cross_retailer_graph.png
│   │   │   ├── graph_edges.csv
│   │   │   ├── graph_nodes.csv
│   │   │   ├── graph_summary.txt
│   │   │   ├── matched_pairs.csv
│   │   │   ├── matched_products.csv
│   │   │   ├── matched_reviews.csv
│   │   │   └── product_graph.gpickle
│   │   ├── Sephora/
│   │   │   ├── sephora_brand_health.csv
│   │   │   ├── sephora_brand_topic_labels.csv
│   │   │   ├── sephora_cluster_labels.csv
│   │   │   ├── sephora_complaint_concentration.csv
│   │   │   ├── sephora_delighters_disappointers.csv
│   │   │   ├── sephora_products.csv
│   │   │   ├── sephora_recommendations.csv
│   │   │   ├── sephora_reviews_enriched.csv
│   │   │   ├── sephora_reviews.csv
│   │   │   ├── sephora_segmentation.csv
│   │   │   ├── sephora_topic_drivers.csv
│   │   │   ├── sephora_topic_labels.csv
│   │   │   └── sephora_value_perception.csv
│   │   └── Ulta/
│   │       ├── ulta_brand_health.csv
│   │       ├── ulta_brand_topic_labels.csv
│   │       ├── ulta_cluster_labels.csv
│   │       ├── ulta_complaint_concentration.csv
│   │       ├── ulta_delighters_disappointers.csv
│   │       ├── ulta_products.csv
│   │       ├── ulta_recommendations.csv
│   │       ├── ulta_reviews_enriched.csv
│   │       ├── ulta_reviews.csv
│   │       ├── ulta_segmentation.csv
│   │       ├── ulta_topic_drivers.csv
│   │       ├── ulta_topic_labels.csv
│   │       └── ulta_value_perception.csv
│   └── raw/
│       ├── scraped_brand_slugs.txt
│       ├── sephora_products.csv
│       ├── sephora_products2.csv
│       ├── sephora_reviews.csv
│       ├── sephora_reviews2.csv
│       ├── ulta_products.csv
│       ├── ulta_products3.csv
│       ├── ulta_products4.csv
│       ├── ulta_products5.csv
│       ├── ulta_reviews.csv
│       ├── ulta_reviews3.csv
│       ├── ulta_reviews4.csv
│       ├── ulta_reviews5.csv
│       └── ulta_scraped_brand_slugs.txt
├── notebooks/
│   ├── comparative/
│   │   ├── outputs/
│   │   │   ├── 3.1_biplot.png
│   │   │   ├── 3.1_cross_platform_overlay.png
│   │   │   ├── 3.1_dendrogram_sephora.png
│   │   │   ├── 3.1_dendrogram_ulta.png
│   │   │   ├── 3.1_dimension_heatmap_sephora.png
│   │   │   ├── 3.1_dimension_heatmap_ulta.png
│   │   │   ├── 3.1_perceptual_map.png
│   │   │   ├── 3.1_price_perception_shift.png
│   │   │   ├── 3.1_price_perception_zones.png
│   │   │   ├── 3.1_radar_sephora.png
│   │   │   ├── 3.1_radar_ulta.png
│   │   │   ├── 3.1_repositioning_matrix.png
│   │   │   ├── 3.1_shared_perceptual_space.png
│   │   │   ├── 3.1_white_space_map.png
│   │   │   ├── 3.1_whitespace_candidates.png
│   │   │   ├── 3.2_archetype_by_price_tier.png
│   │   │   ├── 3.2_crossplatform_archetype.png
│   │   │   ├── 3.2_expectation_management.png
│   │   │   ├── 3.2_fragility_map.png
│   │   │   ├── 3.2_granger_causality.png
│   │   │   ├── 3.2_price_deviation_velocity.png
│   │   │   ├── 3.2_price_perception_zones.png
│   │   │   ├── 3.2_price_sensitivity_divergence.png
│   │   │   ├── 3.2_segment_sensitivity_sephora.png
│   │   │   ├── 3.2_segment_sensitivity_ulta.png
│   │   │   ├── 3.2_sensitivity_distributions.png
│   │   │   ├── 3.2_shock_response.png
│   │   │   ├── 3.2_spillover_event_study.png
│   │   │   ├── 3.2_strategy_matrix.png
│   │   │   ├── 3.2_value_communication_sephora.png
│   │   │   ├── 3.2_value_communication_ulta.png
│   │   │   ├── 3.3_did_estimates.png
│   │   │   ├── 3.3_event_study_coefficients.png
│   │   │   ├── 3.3_price_reputation_interaction.png
│   │   │   ├── 3.3_recovery_trajectories.png
│   │   │   ├── 3.3_roi_matrix_sephora.png
│   │   │   ├── 3.3_roi_matrix_ulta.png
│   │   │   └── 3.3_spillover_did.png
│   │   ├── 3-1_combined_perceptual_map.ipynb
│   │   ├── 3-1_competitive_positioning.ipynb
│   │   ├── 3-2_coss_retailer_simulations.ipynb
│   │   ├── 3-2_perceived_value.ipynb
│   │   ├── 3-3_causal_attribution.ipynb
│   │   └── 3-4_convergent_strategic_recommendation.ipynb
│   ├── independent/
│   │   ├── outputs/
│   │   │   ├── brand_dashboards/
│   │   │   ├── product_dashboards/
│   │   │   ├── sephora_segmentation/
│   │   │   │   ├── gmm_uw_bic_aic_silhouette.png
│   │   │   │   ├── gmm_w_bic_aic_silhouette.png
│   │   │   │   ├── hc_uw_dendrogram.png
│   │   │   │   ├── hc_uw_silhouette.png
│   │   │   │   ├── hc_w_dendrogram.png
│   │   │   │   ├── hc_w_silhouette.png
│   │   │   │   ├── km_uw_elbow_silhouette.png
│   │   │   │   ├── km_w_elbow_silhouette.png
│   │   │   │   ├── profile_gmm_unweighted.png
│   │   │   │   ├── profile_gmm_weighted.png
│   │   │   │   ├── profile_hierarchical_unweighted.png
│   │   │   │   ├── profile_hierarchical_weighted.png
│   │   │   │   ├── profile_kmeans_unweighted.png
│   │   │   │   └── profile_kmeans_weighted.png
│   │   │   ├── ulta_segmentation/
│   │   │   │   ├── gmm_uw_bic_aic_silhouette.png
│   │   │   │   ├── gmm_w_bic_aic_silhouette.png
│   │   │   │   ├── hc_uw_dendrogram.png
│   │   │   │   ├── hc_uw_silhouette.png
│   │   │   │   ├── hc_w_dendrogram.png
│   │   │   │   ├── hc_w_silhouette.png
│   │   │   │   ├── km_uw_elbow_silhouette.png
│   │   │   │   ├── km_w_elbow_silhouette.png
│   │   │   │   ├── profile_gmm_unweighted.png
│   │   │   │   ├── profile_gmm_weighted.png
│   │   │   │   ├── profile_hierarchical_unweighted.png
│   │   │   │   ├── profile_hierarchical_weighted.png
│   │   │   │   ├── profile_kmeans_unweighted.png
│   │   │   │   └── profile_kmeans_weighted.png
│   │   │   ├── sephora_brand_dashboards_index.html
│   │   │   ├── sephora_brand_health_overview.html
│   │   │   ├── sephora_product_dashboards_index.html
│   │   │   ├── ulta_brand_dashboards_index.html
│   │   │   ├── ulta_brand_health_overview.html
│   │   │   └── ulta_product_dashboards_index.html
│   │   ├── sephora_recommendations.ipynb
│   │   ├── sephora_segmentation.ipynb
│   │   ├── sephora_sentiment.ipynb
│   │   ├── ulta_recommendations.ipynb
│   │   ├── ulta_segmentation.ipynb
│   │   └── ulta_sentiment.ipynb
│   └── joint/
│       ├── outputs/
│       │   ├── 2.1_product_graph_brand_networrk.png
│       │   ├── 2.1_product_graph_category_heatmap.png
│       │   ├── 2.1_product_graph_coverage.png
│       │   ├── 2.1_product_graph_price_delta.png
│       │   ├── 2.2_brand_consistency.png
│       │   ├── 2.2_segment_divergence.png
│       │   ├── 2.2_segment_flow.png
│       │   ├── 2.2_segment_heatmap.png
│       │   ├── 2.3_brand_sentiment_gap.png
│       │   ├── 2.3_category_sentiment_gap.png
│       │   ├── 2.3_delta_distributions.png
│       │   ├── 2.3_gap_scatter.png
│       │   ├── 2.3_platform_delighters.png
│       │   ├── 2.3_platform_profile.png
│       │   ├── 2.4_brand_price_gap.png
│       │   ├── 2.4_category_price_gap.png
│       │   ├── 2.4_price_parity_overview.png
│       │   ├── 2.4_price_tier_analysis.png
│       │   ├── 2.4_price_vs_experience_delta.png
│       │   ├── 2.4_price_vs_value_scatter.png
│       │   └── 2.4_value_efficiency.png
│       ├── 2-1_product_graph.ipynb
│       ├── 2-2_matched_segmentation.ipynb
│       ├── 2-3_setiment_gap.ipynb
│       └── 2-4_perceived_value.ipynb
├── src/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── brand_matching.py
│   ├── build_product_graph.py
│   ├── data_cleaning.py
│   ├── joint_recommendations.py
│   ├── scrape_sephora.py
│   ├── scrape_ulta.py
│   ├── segmentation_features.py
│   ├── sephora_recommendations.py
│   ├── sephora_sentiment.py
│   ├── ulta_recommendations.py
│   ├── ulta_sentiment.py
│   └── utils.py
├── venv_scrape/
├── .gitattributes
├── README.md
└── requirements.txt
```
---

---

## Code Reference

This section documents every source file and notebook in the repository — what each one does, what it produces, and how to run it. Files are grouped by tier and ordered by intended execution sequence.

---

### Source Modules (`src/`)

These Python modules contain the core data collection, processing, and modeling logic. They are imported by notebooks and can also be run directly from the command line where noted. All scripts should be run from the project root with the virtual environment activated.

---

#### `src/scrape_sephora.py`

**What it does:** End-to-end Sephora data collector. Navigates the `/brands-list` page to enumerate all brand URLs, then iterates through each brand's product pages using scroll-to-load pagination. For each product, it cascades through multiple extraction strategies — Selenium JavaScript context (`window.Sephora.productPage`), CSS selectors, meta tags, JSON-LD, `__NEXT_DATA__` JSON, and regex fallbacks — to capture product ID, name, category, price, and aggregate rating. Reviews are fetched via the Bazaarvoice API (`api.bazaarvoice.com/data/reviews.json`) in pages of 100, capturing full review text, star rating, submission timestamp, helpfulness votes, and reviewer-reported skin tone and skin type. Output is written incrementally per brand to prevent data loss on interruption. Completed brand slugs are tracked so the scraper is fully resumable.

**Outputs:**
- `data/raw/sephora_products.csv` — product-level metadata
- `data/raw/sephora_reviews.csv` — individual review records with demographic attributes
- `data/raw/scraped_brand_slugs.txt` — checkpoint file for resumable runs

**Usage:**
```bash
python src/scrape_sephora.py
```

**Notes:** Requires Google Chrome and `undetected-chromedriver`. Uses randomized polite sleeps between requests. If IP-based rate limiting is encountered, connecting to a VPN before running is recommended. The scraper automatically rotates sessions on access denial. Do not interrupt mid-brand if possible; interruptions between brands are safe.

---

#### `src/scrape_ulta.py`

**What it does:** End-to-end Ulta data collector. Navigates the `/brand/all` brand directory, collects brand page URLs, and iterates through each brand's product listings using scroll-to-load pagination. Product metadata — ID (pimprod identifier), name, category, price, and rating — is extracted via cascading fallbacks including JSON-LD, CSS selectors, meta tags, and regex. Reviews are fetched via the PowerReviews Read API (`readservices-b2c.powerreviews.com`) in pages of 25 and include review text, headline, star rating, submission timestamp, reviewer nickname and location, would-recommend indicator (`bottom_line`), helpful and not-helpful vote counts, verified buyer status, verified reviewer status, and disclosure codes. Output is written incrementally per brand with brand slug tracking for resumable runs.

**Outputs:**
- `data/raw/ulta_products.csv` — product-level metadata (multiple shards if run in stages)
- `data/raw/ulta_reviews.csv` — individual review records with engagement and verification signals
- `data/raw/ulta_scraped_brand_slugs.txt` — checkpoint file for resumable runs

**Usage:**
```bash
python src/scrape_ulta.py
```

**Notes:** Same anti-detection setup as the Sephora scraper. Ulta's PowerReviews API returns 25 reviews per page versus Bazaarvoice's 100, so Ulta collection is proportionally slower per product for high-review-count SKUs. Multiple raw output shards (`ulta_products3.csv`, etc.) are expected artifacts of staged collection runs and are merged during cleaning.

---

#### `src/data_cleaning.py`

**What it does:** Consolidates raw scraped output into clean, analysis-ready CSVs. For both retailers, it merges multi-shard raw files, deduplicates product and review records, standardizes column names and data types, parses submission timestamps to datetime, imputes or flags missing prices and ratings, strips HTML artifacts from review text, and normalizes brand names to a canonical form. Produces a single clean product file and review file per retailer.

**Outputs:**
- `data/processed/Sephora/sephora_products.csv`
- `data/processed/Sephora/sephora_reviews.csv`
- `data/processed/Ulta/ulta_products.csv`
- `data/processed/Ulta/ulta_reviews.csv`

**Usage:**
```bash
python src/data_cleaning.py
```

**Notes:** Must be run before any notebooks. Expects raw files in `data/raw/`. If additional scraping shards are added, re-running this script will automatically incorporate them.

---

#### `src/brand_matching.py`

**What it does:** Identifies brands and products that appear on both Sephora and Ulta to enable cross-retailer analysis. Implements a multi-stage matching pipeline: brand name normalization (lowercasing, punctuation stripping, common abbreviation expansion), fuzzy string matching via token set ratio to handle minor naming differences, and price-and-category validation to confirm candidate matches. For matched brands, performs product-level fuzzy matching on product names within the same price band and category. Outputs a brand mapping table, a matched pairs table at the brand level, and a matched products table at the SKU level.

**Outputs:**
- `data/processed/Matched/brand_mapping.csv` — canonical brand name to Sephora/Ulta brand name mapping
- `data/processed/Matched/matched_pairs.csv` — confirmed cross-retailer brand pairs
- `data/processed/Matched/matched_products.csv` — matched product records with both platforms' prices and ratings

**Usage:**
```bash
python src/brand_matching.py
```

**Notes:** Must be run after `data_cleaning.py`. The fuzzy matching threshold is tunable; the default is calibrated for the beauty retail naming conventions in this dataset. Manual review of low-confidence matches is recommended before downstream use.

---

#### `src/build_product_graph.py`

**What it does:** Constructs a bipartite graph connecting Sephora and Ulta products through their matched relationships. Nodes represent individual products on each platform with attributes including brand, category, price, and average rating. Edges connect matched product pairs with edge weights reflecting match confidence, price delta, and rating delta. The graph is serialized as a NetworkX `gpickle` for use in downstream notebooks, and summary statistics and a visual overview are exported.

**Outputs:**
- `data/processed/Matched/product_graph.gpickle` — serialized NetworkX graph
- `data/processed/Matched/graph_nodes.csv` — node attribute table
- `data/processed/Matched/graph_edges.csv` — edge attribute table
- `data/processed/Matched/graph_summary.txt` — graph statistics (node count, edge count, density, connected components)
- `data/processed/Matched/cross_retailer_graph.png` — visual overview of the brand network

**Usage:**
```bash
python src/build_product_graph.py
```

**Notes:** Must be run after `brand_matching.py`. The graph is the foundation for Section 2 joint analyses; all cross-retailer notebooks load the `gpickle` rather than re-running matching.

---

#### `src/segmentation_features.py`

**What it does:** Constructs the feature matrix used as input to all segmentation notebooks. For each product on each platform, it computes review volume, review velocity (reviews per month), average and variance of star ratings, VADER-based average sentiment score, topic prevalence scores from LDA, price tier assignment, relative price positioning within category, and (for Sephora) skin-tone-weighted sentiment breakdowns. Outputs separate feature matrices for Sephora and Ulta, normalized and ready for clustering. Also exports the feature column definitions and scaling parameters for reproducibility.

**Outputs:**
- `data/processed/Sephora/sephora_segmentation.csv` — Sephora product feature matrix
- `data/processed/Ulta/ulta_segmentation.csv` — Ulta product feature matrix

**Usage:**
```bash
python src/segmentation_features.py
```

**Notes:** Must be run after `data_cleaning.py`. Sentiment scoring in this module uses VADER on the full review corpus and may take several minutes for large datasets. Topic modeling uses a fixed random seed for reproducibility.

---

---

#### `src/sephora_sentiment.py`

**What it does:** Full brand health and sentiment pipeline for Sephora, covering nine processing steps and an interactive dashboard layer. Steps 1–9 run as a script: text cleaning and review filtering, VADER sentiment scoring, rating–sentiment mismatch detection, NMF topic modeling on TF-IDF features, topic–sentiment driver linkage, per-brand delighter and disappointer identification, complaint concentration analysis via entropy scoring, price/value perception diagnostics, and brand-level aggregation. Step 9 also writes the static HTML brand health overview — average VADER sentiment vs. average star rating scatter sized by review volume and colored by mismatch rate — which is saved as a standalone file viewable in any browser. Step 10 is available in two modes. In a Jupyter notebook, `show_dashboard()` launches an ipywidgets dropdown that toggles a 12-panel per-brand dashboard across all qualifying brands. For browser access without a notebook, `write_brand_html()` saves a single brand's dashboard as a self-contained HTML, and `write_all_brand_htmls()` saves every qualifying brand and generates a linked index page listing all brands.

**Outputs:**
- `data/processed/Sephora/sephora_brand_health.csv`
- `data/processed/Sephora/sephora_reviews_enriched.csv`
- `data/processed/Sephora/sephora_topic_drivers.csv`
- `data/processed/Sephora/sephora_brand_topic_labels.csv`
- `data/processed/Sephora/sephora_delighters_disappointers.csv`
- `data/processed/Sephora/sephora_complaint_concentration.csv`
- `data/processed/Sephora/sephora_value_perception.csv`
- `notebooks/independent/outputs/sephora_brand_health_overview.html` — static brand health scatter; self-contained, opens in any browser
- `notebooks/independent/outputs/brand_dashboards/<brand>.html` — per-brand dashboards; self-contained, open in any browser
- `notebooks/independent/outputs/sephora_brand_dashboards_index.html` — index page linking all brands; self-contained, opens in any browser

**Usage:**

As a script (runs Steps 1–9, saves all CSVs and the HTML overview):
```bash
python src/sephora_sentiment.py
```

As an import in a notebook (launches the interactive per-brand dashboard):
```python
from src.sephora_sentiment import show_dashboard
show_dashboard()
```

To show just the overview scatter inside a notebook:
```python
from src.sephora_sentiment import show_overview
show_overview()
```

To save a single brand dashboard as a browser-accessible HTML:
```python
from src.sephora_sentiment import write_brand_html
write_brand_html("Charlotte Tilbury")
write_brand_html("NARS", output_dir="my_reports/")
```

To save all brand dashboards with a linked index page:
```python
from src.sephora_sentiment import write_all_brand_htmls
write_all_brand_htmls()
write_all_brand_htmls(output_dir="my_reports/brands/")
```

**Notes:** `show_dashboard()` requires a running Jupyter server with `ipywidgets` enabled. The browser export functions have no such requirement — all HTML files are self-contained via CDN-loaded Plotly. Unlike the Ulta equivalent, Sephora produces one HTML file per brand with no segment dimension, so the index page uses a simpler single-link-per-brand layout. `N_TOPICS` (default 12) and `MIN_BRAND_REVIEWS` (default 30) are configurable constants at the top of the file.

---

#### `src/ulta_sentiment.py`

**What it does:** Full brand health and sentiment pipeline for Ulta, structurally parallel to `sephora_sentiment.py` with Ulta-specific enrichments throughout. Steps 1–9 run as a script: text cleaning and review filtering, VADER sentiment scoring on both review body and headline with a combined score weighted 70% body and 30% headline, rating–sentiment mismatch detection using the combined score, NMF topic modeling on concatenated headline and body text for richer topic signal, topic–sentiment driver linkage, per-brand delighter and disappointer identification, complaint concentration analysis via entropy scoring, price/value perception diagnostics checking both body and headline for value mentions, and brand-level aggregation across three verified buyer segments — all reviews, verified buyers only, and unverified only. Step 9 also writes the static HTML brand health overview — combined sentiment vs. star rating scatter sized by review volume and colored by mismatch rate — which is saved as a standalone file viewable in any browser. Step 10 is available in two modes. In a Jupyter notebook, `show_dashboard()` launches an ipywidgets brand dropdown paired with a verified segment toggle that rebuilds the 12-panel per-brand dashboard on each selection. For browser access without a notebook, `write_brand_html()` saves a single brand's dashboard for a specific segment as a self-contained HTML, and `write_all_brand_htmls()` saves every qualifying brand across all three segments and generates a linked index page grouping all segment links per brand.

**Outputs:**
- `data/processed/Ulta/ulta_brand_health.csv`
- `data/processed/Ulta/ulta_reviews_enriched.csv`
- `data/processed/Ulta/ulta_topic_drivers.csv`
- `data/processed/Ulta/ulta_brand_topic_labels.csv`
- `data/processed/Ulta/ulta_delighters_disappointers.csv`
- `data/processed/Ulta/ulta_complaint_concentration.csv`
- `data/processed/Ulta/ulta_value_perception.csv`
- `notebooks/independent/outputs/ulta_brand_health_overview.html` — static brand health scatter; self-contained, opens in any browser
- `notebooks/independent/outputs/brand_dashboards/<brand>_<segment>.html` — per-brand per-segment dashboards; self-contained, open in any browser
- `notebooks/independent/outputs/ulta_brand_dashboards_index.html` — index page linking all brands and their segment dashboards; self-contained, opens in any browser

**Usage:**

As a script (runs Steps 1–9, saves all CSVs and the HTML overview):
```bash
python src/ulta_sentiment.py
```

As an import in a notebook (launches the interactive per-brand dashboard with segment toggle):
```python
from src.ulta_sentiment import show_dashboard
show_dashboard()
```

To show just the overview scatter inside a notebook:
```python
from src.ulta_sentiment import show_overview
show_overview()
```

To save a single brand dashboard as a browser-accessible HTML:
```python
from src.ulta_sentiment import write_brand_html
write_brand_html("Sol de Janeiro")                                   # defaults to "all" segment
write_brand_html("NARS", verified_segment="verified")               # verified buyers only
write_brand_html("Charlotte Tilbury", output_dir="my_reports/")
```

To save all brand dashboards across all segments with a linked index page:
```python
from src.ulta_sentiment import write_all_brand_htmls
write_all_brand_htmls()                           # all brands × all 3 segments
write_all_brand_htmls(segments=("all",))          # all brands, full-corpus view only
write_all_brand_htmls(output_dir="my_reports/brands/")
```

**Notes:** `show_dashboard()` requires a running Jupyter server with `ipywidgets` enabled. The browser export functions have no such requirement — all HTML files are self-contained via CDN-loaded Plotly. The Ulta index page includes columns for body sentiment, headline sentiment, combined sentiment, and % verified buyers that are not present in the Sephora equivalent. The headline weighting (0.3) and body weighting (0.7) are configurable constants at the top of the file.

---

---

#### `src/sephora_recommendations.py`

**What it does:** Builds a content-based product recommendation engine for the Sephora catalog across three recommendation intents — close substitutes (same need, similar perception), complementary products (different need, shared audience), and trade-up/trade-down paths (same experience, different price tier). Constructs four feature blocks per product: structural (category dummies and skin tone/type distributions), sentiment (VADER scores, rating distributions, and rating-sentiment mismatch), content (dominant topic and topic prevalence), and price (log-transformed price and category-relative positioning). Computes cosine similarity matrices per block, then combines them into intent-specific composite scores with block weights tuned per intent. Exports a precomputed recommendations table mapping each product to its top substitutes, complements, trade-up, and trade-down matches. The dashboard layer is available in two modes. In a Jupyter notebook, `show_dashboard()` launches an ipywidgets dropdown that toggles an 8-panel per-product dashboard across the full catalog. For browser access without a notebook, `write_product_html()` saves a single product's dashboard as a self-contained HTML, and `write_all_product_htmls()` saves every product in the catalog and generates a linked index page searchable by brand, product name, and category.

**Outputs:**
- `data/processed/Sephora/sephora_recommendations.csv` — product-to-product recommendation table with scores, prices, ratings, and sentiment for each recommended product
- `notebooks/independent/outputs/product_dashboards/<brand>_<product>_<id>.html` — per-product recommendation dashboards; self-contained, open in any browser
- `notebooks/independent/outputs/sephora_product_dashboards_index.html` — index page linking all products; self-contained, opens in any browser

**Usage:**

As a script (runs full pipeline and saves the CSV):
```bash
python src/sephora_recommendations.py
```

As an import in a notebook (launches the interactive product dropdown):
```python
from src.sephora_recommendations import show_dashboard
show_dashboard()
```

To save a single product dashboard as a browser-accessible HTML:
```python
from src.sephora_recommendations import write_product_html
write_product_html("P12345678")
write_product_html("P12345678", output_dir="my_reports/")
```

To save all product dashboards with a linked index page:
```python
from src.sephora_recommendations import write_all_product_htmls
write_all_product_htmls()
write_all_product_htmls(output_dir="my_reports/products/")
```

**Notes:** `show_dashboard()` requires a running Jupyter server with `ipywidgets` enabled. The browser export functions have no such requirement. `write_all_product_htmls()` scales with the full product catalog and may take several minutes to complete — for targeted exports, call `write_product_html()` directly for each product ID of interest. HTML filenames are constructed from brand, product name, and product ID for readability in the output folder.

---

#### `src/ulta_recommendations.py`

**What it does:** Builds a content-based product recommendation engine for the Ulta catalog across three recommendation intents — close substitutes, complementary products, and trade-up/trade-down paths. Structurally parallel to `sephora_recommendations.py` with one key Ulta-specific distinction: verification provenance (percent verified buyers, verified reviewers, would-recommend rate, and disclosure rate) is encoded as a dedicated structural feature block and given 25% weight in the substitute and complement scoring. Two products with identical ratings but mismatched verification profiles are penalized as substitutes, reflecting that organic and seeded review bases signal meaningfully different audience trust dynamics. Substitute bar colors in the dashboard encode provenance directly — green for organic verified, orange for seeded, blue for low verification — and are preserved in all browser exports. The dashboard layer is available in two modes. In a Jupyter notebook, `show_dashboard()` launches an ipywidgets dropdown that toggles an 8-panel per-product dashboard across the full catalog. For browser access without a notebook, `write_product_html()` saves a single product's dashboard as a self-contained HTML, and `write_all_product_htmls()` saves every product and generates a linked index page with a verification provenance column and legend.

**Outputs:**
- `data/processed/Ulta/ulta_recommendations.csv` — product-to-product recommendation table with scores, prices, ratings, sentiment, verified buyer rate, and disclosure rate for each recommended product
- `notebooks/independent/outputs/ulta_product_dashboards/<brand>_<product>_<id>.html` — per-product recommendation dashboards; self-contained, open in any browser
- `notebooks/independent/outputs/ulta_product_dashboards_index.html` — index page linking all products with verification provenance column; self-contained, opens in any browser

**Usage:**

As a script (runs full pipeline and saves the CSV):

```bash
python src/ulta_recommendations.py
```

As an import in a notebook (launches the interactive product dropdown):

```python
from src.ulta_recommendations import show_dashboard
show_dashboard()
```

To save a single product dashboard as a browser-accessible HTML:

```python
from src.ulta_recommendations import write_product_html
write_product_html("P12345678")
write_product_html("P12345678", output_dir="my_reports/")
```

To save all product dashboards with a linked index page:

```python
from src.ulta_recommendations import write_all_product_htmls
write_all_product_htmls()
write_all_product_htmls(output_dir="my_reports/ulta_products/")
```

**Notes:** `show_dashboard()` requires a running Jupyter server with `ipywidgets` enabled. The browser export functions have no such requirement. `write_all_product_htmls()` may take several minutes for the full catalog — use `write_product_html()` in a loop for targeted subsets. The pipeline also reports a substitute verification gap metric (average percentage-point difference in verified buyer rate between source and recommended products) as a quality check.

---

#### `src/__init__.py`

**What it does:** Marks `src/` as a Python package, enabling relative imports across modules. Contains no logic. Required for `from src.utils import ...` style imports to resolve correctly from the project root.

---

### Notebooks

All notebooks are self-contained and produce their outputs to the `outputs/` subfolder within their tier directory. They should be run in the order presented below, as later notebooks depend on processed files produced by earlier ones. Notebooks can be run via Jupyter Lab, Jupyter Notebook, or VS Code's notebook interface.

```bash
jupyter lab
```

---

### Part 1 — Independent Retailer Analyses (`notebooks/independent/`)

---

#### `sephora_segmentation.ipynb`

**What it does:** Applies three clustering algorithms — K-Means, Hierarchical Clustering (Ward linkage), and Gaussian Mixture Models — to the Sephora product feature matrix to identify distinct consumer-perception segments within the Sephora catalog. For each algorithm, runs model selection diagnostics (elbow plots, silhouette scores, BIC/AIC curves) in both weighted and unweighted feature configurations to select optimal cluster count. Profiles each resulting segment across review velocity, rating level, sentiment, dominant topics, and price tier. Assigns final cluster labels using K-Means as the primary solution. The four final segments are labeled: Quiet Underperformers (C0), Mass-Market Satisfied (C1), Niche Beloved (C2), and Disappointed & Vocal (C3).

**Inputs:**
- `data/processed/Sephora/sephora_segmentation.csv`

**Outputs:**
- `data/processed/Sephora/sephora_cluster_labels.csv` — product-level cluster assignments
- `outputs/sephora_segmentation/` — 14 diagnostic and profile visualizations (elbow/silhouette/BIC plots per algorithm and weighting, cluster profile radar charts)

**Usage:** Run all cells top to bottom. Model selection plots are generated mid-notebook to inform the cluster count choice before final labeling.

---

#### `ulta_segmentation.ipynb`

**What it does:** Identical segmentation pipeline applied to the Ulta product feature matrix. Incorporates Ulta-specific features including the `bottom_line` recommendation rate and verified buyer sentiment. Runs the same three algorithms with the same diagnostics. Final segments reflect Ulta's broader catalog — the Mass Satisfied and Value Fragile groupings at Ulta have different compositions than their Sephora counterparts due to Ulta's wider price range.

**Inputs:**
- `data/processed/Ulta/ulta_segmentation.csv`

**Outputs:**
- `data/processed/Ulta/ulta_cluster_labels.csv`
- `outputs/ulta_segmentation/` — 14 diagnostic and profile visualizations

**Usage:** Run all cells top to bottom. Segment labels are assigned in the final cells and written to `ulta_cluster_labels.csv`.

---

#### `sephora_sentiment.ipynb`

**What it does:** Interactive exploration of the Sephora brand health pipeline. Calls `run_pipeline()` to execute all nine processing steps and generate the CSV outputs and static HTML overview, then launches the full interactive dashboard via `show_dashboard()`. The overview scatter — average VADER sentiment vs. average star rating, bubble-sized by review volume and colored by mismatch rate — plots composite health scores across all ranked brands and is saved as a browser-accessible HTML. The interactive dashboard adds a brand dropdown that renders a 12-panel per-brand deep-dive on selection; this layer requires the notebook environment. To share individual brand dashboards outside the notebook, use the browser export functions after the pipeline has run.

**Inputs:**
- `data/processed/Sephora/sephora_products.csv`
- `data/processed/Sephora/sephora_reviews.csv`

**Outputs:**
- All CSVs listed under `src/sephora_sentiment.py`
- `outputs/sephora_brand_health_overview.html` — static overview scatter; self-contained, opens in any browser

**Usage:** Run all cells. To access the interactive per-brand dashboard after the pipeline has already been run, import directly without re-running the pipeline:

```python
from src.sephora_sentiment import show_dashboard
show_dashboard()
```

To export browser-accessible dashboards after the pipeline has run:

```python
from src.sephora_sentiment import write_all_brand_htmls
write_all_brand_htmls()
```

---

#### `ulta_sentiment.ipynb`

**What it does:** Interactive exploration of the Ulta brand health pipeline. Calls `run_pipeline()` to execute all nine processing steps and generate the CSV outputs and static HTML overview, then launches the full interactive dashboard via `show_dashboard()`. The overview scatter — average combined sentiment (0.7 body + 0.3 headline) vs. average star rating, bubble-sized by review volume and colored by mismatch rate — plots composite health scores across all ranked brands and is saved as a browser-accessible HTML. The interactive dashboard adds a brand dropdown paired with a verified segment toggle (All Reviews / Verified Buyers / Unverified) that renders a 12-panel per-brand deep-dive on each selection; this layer requires the notebook environment. To share individual brand dashboards outside the notebook, use the browser export functions after the pipeline has run.

**Inputs:**
- `data/processed/Ulta/ulta_products.csv`
- `data/processed/Ulta/ulta_reviews.csv`

**Outputs:**
- All CSVs listed under `src/ulta_sentiment.py`
- `outputs/ulta_brand_health_overview.html` — static overview scatter; self-contained, opens in any browser

**Usage:** Run all cells. To access the interactive per-brand dashboard after the pipeline has already been run, import directly without re-running the pipeline:

```python
from src.ulta_sentiment import show_dashboard
show_dashboard()
```

To export browser-accessible dashboards after the pipeline has run:

```python
from src.ulta_sentiment import write_all_brand_htmls
write_all_brand_htmls()
```

---

#### `sephora_recommendations.ipynb`

**What it does:** Demonstrates and evaluates the Sephora recommendation engine. Loads the precomputed recommendation table and renders category-specific substitute clusters, trade-up/trade-down paths, and similarity network visualizations for selected brands and products. Launches the interactive product lookup via `show_dashboard()`, which provides a dropdown to select any product in the catalog and renders its 8-panel recommendation dashboard — covering close substitutes by price and sentiment, complementary products by price and sentiment, trade-up and trade-down paths, an average score summary by intent, and a full detail table of all recommendations. To share dashboards outside the notebook, use the browser export functions after the pipeline has run.

**Inputs:**
- `data/processed/Sephora/sephora_recommendations.csv`
- `data/processed/Sephora/sephora_products.csv`

**Outputs:** Visualizations rendered inline. To produce shareable files:
- `outputs/product_dashboards/<brand>_<product>_<id>.html` — per-product, browser-accessible
- `outputs/sephora_product_dashboards_index.html` — linked index page, browser-accessible

**Usage:** Run all cells. To access the interactive dashboard or export browser files after the pipeline has already been run, import directly without re-running:

```python
from src.sephora_recommendations import show_dashboard
show_dashboard()

from src.sephora_recommendations import write_all_product_htmls
write_all_product_htmls()
```

---

#### `ulta_recommendations.ipynb`

**What it does:** Demonstrates and evaluates the Ulta recommendation engine. Loads the precomputed recommendation table and renders category-specific substitute clusters, trade-up/trade-down paths, and verification-stratified views for selected brands and products. Launches the interactive product lookup via `show_dashboard()`, which provides a dropdown to select any product in the catalog and renders its 8-panel recommendation dashboard — covering close substitutes by price and sentiment (bars color-coded by verification provenance), complementary products, trade-up and trade-down paths, an average score summary by intent, and a full detail table including a verification label column. To share dashboards outside the notebook, use the browser export functions after the pipeline has run.

**Inputs:**
- `data/processed/Ulta/ulta_recommendations.csv`
- `data/processed/Ulta/ulta_products.csv`

**Outputs:** Visualizations rendered inline. To produce shareable files:
- `outputs/ulta_product_dashboards/<brand>_<product>_<id>.html` — per-product, browser-accessible
- `outputs/ulta_product_dashboards_index.html` — linked index page with verification column, browser-accessible

**Usage:** Run all cells. To access the interactive dashboard or export browser files after the pipeline has already been run, import directly without re-running:

```python
from src.ulta_recommendations import show_dashboard
show_dashboard()

from src.ulta_recommendations import write_all_product_htmls
write_all_product_htmls()
```

---

#### `2-1_product_graph.ipynb`

**What it does:** Loads and visualizes the cross-retailer product graph constructed by `src/build_product_graph.py`. Renders four graph-level diagnostic views: a brand network showing which brands have matched products across both platforms, a category heatmap showing cross-platform coverage density by category, a product coverage chart comparing matched vs. unmatched catalog fractions per retailer, and a price delta distribution across matched product pairs. Summarizes the match corpus: 379 matched products across 142 shared brands and 110 matched brand pairs.

**Inputs:**
- `data/processed/Matched/product_graph.gpickle`
- `data/processed/Matched/graph_nodes.csv`
- `data/processed/Matched/graph_edges.csv`

**Outputs:**
- `outputs/2.1_product_graph_brand_network.png`
- `outputs/2.1_product_graph_category_heatmap.png`
- `outputs/2.1_product_graph_coverage.png`
- `outputs/2.1_product_graph_price_delta.png`

**Usage:** Run all cells. This notebook is primarily diagnostic and does not require parameter tuning.

---

#### `2-2_matched_segmentation.ipynb`

**What it does:** Compares segment assignments for the 379 matched products across both platforms to measure cross-retailer segmentation consistency. For each matched product, retrieves its Sephora cluster label and Ulta cluster label and computes a consistency score. Renders a Sankey flow diagram showing how Sephora segments map to Ulta segments, a cross-tabulation heatmap of segment co-occurrences, a brand-level consistency ranking, and a divergence index for brands where the same products land in systematically different segments by platform. Key finding: 33.5% overall consistency; Sephora C2 Niche Beloved is the strongest cross-platform predictor (88% land in Ulta positive tiers).

**Inputs:**
- `data/processed/Matched/matched_products.csv`
- `data/processed/Sephora/sephora_cluster_labels.csv`
- `data/processed/Ulta/ulta_cluster_labels.csv`

**Outputs:**
- `outputs/2.2_segment_flow.png`
- `outputs/2.2_segment_heatmap.png`
- `outputs/2.2_segment_divergence.png`
- `outputs/2.2_brand_consistency.png`

**Usage:** Run all cells. The Sankey diagram requires `plotly`; ensure it is installed.

---

#### `2-3_setiment_gap.ipynb`

**What it does:** Computes and visualizes sentiment and rating gaps for matched products across both platforms. For each matched product, calculates the Sephora-minus-Ulta delta in average sentiment score and average star rating. Runs paired t-tests to assess statistical significance of platform-level differences. Renders delta distribution plots, a gap scatter comparing sentiment delta vs. rating delta, platform-specific delighter heatmaps showing which categories and brands skew positive on each platform, and a brand-level sentiment gap ranking. Key findings: Sephora shows higher sentiment language but lower star ratings than Ulta for the same products; functional skincare shows the largest Ulta-favoring rating gaps.

**Inputs:**
- `data/processed/Matched/matched_products.csv`
- `data/processed/Matched/matched_reviews.csv`
- `data/processed/Sephora/sephora_reviews_enriched.csv`
- `data/processed/Ulta/ulta_reviews_enriched.csv`

**Outputs:**
- `outputs/2.3_delta_distributions.png`
- `outputs/2.3_gap_scatter.png`
- `outputs/2.3_brand_sentiment_gap.png`
- `outputs/2.3_category_sentiment_gap.png`
- `outputs/2.3_platform_delighters.png`
- `outputs/2.3_platform_profile.png`

**Usage:** Run all cells. Statistical test results are printed inline and should be reviewed before interpreting the visualizations.

---

#### `2-4_perceived_value.ipynb`

**What it does:** Cross-platform price consistency and perceived value efficiency analysis for the 379 matched products. Computes price deltas (Sephora price minus Ulta price), classifies products into parity (|Δ| ≤ $1), Sephora-higher, and Ulta-higher categories, and breaks down price architecture by price tier and by category. Computes rating-per-dollar and sentiment-per-dollar efficiency metrics for both platforms and runs t-tests on the difference. Correlates price delta with rating delta and sentiment delta to test whether paying more on one platform buys a better consumer experience. Key finding: price premium carries no measurable consumer experience premium — rating and sentiment gaps operate independently of price.

**Inputs:**
- `data/processed/Matched/matched_products.csv`
- `data/processed/Matched/matched_reviews.csv`

**Outputs:**
- `outputs/2.4_price_parity_overview.png`
- `outputs/2.4_price_tier_analysis.png`
- `outputs/2.4_brand_price_gap.png`
- `outputs/2.4_category_price_gap.png`
- `outputs/2.4_value_efficiency.png`
- `outputs/2.4_price_vs_experience_delta.png`
- `outputs/2.4_price_vs_value_scatter.png`

**Usage:** Run all cells. The price tier boundaries (Budget/Mid/Premium/Luxury/Ultra-lux) are defined in `src/utils.py` and applied consistently across all notebooks.

---

### Part 3 — Comparative and Convergent Strategy (`notebooks/comparative/`)

---

#### `3-1_competitive_positioning.ipynb`

**What it does:** Constructs dual-retailer perceptual maps using PCA on the brand-level feature space (efficacy perception, experience quality, consumer polarization, review intensity, and price positioning). Renders platform-specific PCA biplots and radar charts profiling each brand on the five competitive dimensions. Runs hierarchical clustering on the PCA-reduced space to identify competitive peer groupings within each platform. Produces a repositioning matrix flagging brands whose PCA position is inconsistent with their price tier — revealing over-positioned and under-positioned brands relative to their price point.

**Inputs:**
- `data/processed/Sephora/sephora_brand_health.csv`
- `data/processed/Sephora/sephora_cluster_labels.csv`
- `data/processed/Ulta/ulta_brand_health.csv`
- `data/processed/Ulta/ulta_cluster_labels.csv`

**Outputs:**
- `outputs/3.1_biplot.png`
- `outputs/3.1_dendrogram_sephora.png`
- `outputs/3.1_dendrogram_ulta.png`
- `outputs/3.1_dimension_heatmap_sephora.png`
- `outputs/3.1_dimension_heatmap_ulta.png`
- `outputs/3.1_radar_sephora.png`
- `outputs/3.1_radar_ulta.png`
- `outputs/3.1_repositioning_matrix.png`
- `outputs/3.1_price_perception_zones.png`
- `outputs/3.1_price_perception_shift.png`

**Usage:** Run all cells. PCA components are printed with explained variance; interpret biplots in light of the top two component loadings.

---

#### `3-1_combined_perceptual_map.ipynb`

**What it does:** Extends `3-1_competitive_positioning.ipynb` by projecting both Sephora and Ulta brands into a shared perceptual space via joint PCA on the combined brand feature matrix. Reveals which brands occupy consistent positions across platforms and which shift substantially by retailer context. Renders the shared perceptual map, white space identification overlays showing unoccupied competitive zones, brand-level position shift vectors, and a whitespace candidates chart highlighting the most promising uncontested positioning opportunities by category. Key white spaces identified: prestige haircare tools on Sephora, accessible skincare on Ulta, and fragrance adjacencies on Ulta.

**Inputs:**
- `data/processed/Sephora/sephora_brand_health.csv`
- `data/processed/Ulta/ulta_brand_health.csv`
- `data/processed/Matched/matched_pairs.csv`

**Outputs:**
- `outputs/3.1_shared_perceptual_space.png`
- `outputs/3.1_cross_platform_overlay.png`
- `outputs/3.1_white_space_map.png`
- `outputs/3.1_whitespace_candidates.png`

**Usage:** Run after `3-1_competitive_positioning.ipynb`. The joint PCA requires both platforms' brand health files to be current.

---

#### `3-2_perceived_value.ipynb`

**What it does:** Models demand sensitivity to perceived value on each platform. Classifies brands and products as price-resilient, value-fragile, or reputation-sensitive based on how their review velocity, sentiment, and rating respond to negative review shocks. Generates expectation management quadrant maps placing brands into well-managed, underselling, overpromising, and misaligned categories separately for each retailer. Computes price deviation vs. velocity scatter plots to measure how distance from category price norms correlates with review engagement rates. Produces value communication heatmaps showing which brand archetypes respond most to positive vs. negative review dynamics.

**Inputs:**
- `data/processed/Sephora/sephora_reviews_enriched.csv`
- `data/processed/Sephora/sephora_brand_health.csv`
- `data/processed/Ulta/ulta_reviews_enriched.csv`
- `data/processed/Ulta/ulta_brand_health.csv`

**Outputs:**
- `outputs/3.2_expectation_management.png`
- `outputs/3.2_fragility_map.png`
- `outputs/3.2_price_deviation_velocity.png`
- `outputs/3.2_price_sensitivity_divergence.png`
- `outputs/3.2_segment_sensitivity_sephora.png`
- `outputs/3.2_segment_sensitivity_ulta.png`
- `outputs/3.2_sensitivity_distributions.png`
- `outputs/3.2_strategy_matrix.png`
- `outputs/3.2_value_communication_sephora.png`
- `outputs/3.2_value_communication_ulta.png`
- `outputs/3.2_price_perception_zones.png`
- `outputs/3.2_archetype_by_price_tier.png`
- `outputs/3.2_crossplatform_archetype.png`

**Usage:** Run all cells. Sensitivity classifications are computed using a rolling-window review velocity measure; the window size (default: 30 days) is configurable at the top of the notebook.

---

#### `3-2_coss_retailer_simulations.ipynb`

**What it does:** Simulates cross-platform sentiment spillover effects and Granger causality tests. Uses monthly time-series review data to test whether sentiment shocks on one platform predict subsequent sentiment or velocity changes on the other platform for matched brands. Runs shock response simulations — injecting a synthetic negative review shock and tracking the resulting velocity and sentiment trajectory on the same platform and cross-platform. Identifies which brands exhibit statistically significant Granger causality in cross-platform sentiment propagation. Key finding: Sephora shock leads to 2.47× velocity increase at Ulta; Ulta shock leads to 2.05× at Sephora. Ten percent of brands show significant cross-platform Granger causality.

**Inputs:**
- `data/processed/Matched/matched_reviews.csv`
- `data/processed/Sephora/sephora_reviews_enriched.csv`
- `data/processed/Ulta/ulta_reviews_enriched.csv`

**Outputs:**
- `outputs/3.2_shock_response.png`
- `outputs/3.2_spillover_event_study.png`
- `outputs/3.2_granger_causality.png`

**Usage:** Run all cells. Granger causality tests use a lag order of 3 months by default, configurable at the top of the notebook. Brands with fewer than 12 monthly observations are excluded from time-series tests.

---

#### `3-3_causal_attribution.ipynb`

**What it does:** Implements quasi-experimental causal inference using Difference-in-Differences (DiD) with propensity score matching to estimate the causal effect of negative review shocks on brand outcomes. Constructs treatment and control groups using matched brands that did and did not experience shocks in a given period. Estimates average treatment effects (ATTs) for review velocity, star rating, and sentiment score separately by platform and by price tier. Validates parallel pre-trends assumption and produces event study coefficient plots. Estimates brand recovery trajectories post-shock. Key findings: shocks causally increase velocity but suppress rating and sentiment; the Sephora $40–75 premium tier is uniquely vulnerable with velocity ATT = −3.57 (p < 0.001).

**Inputs:**
- `data/processed/Sephora/sephora_reviews_enriched.csv`
- `data/processed/Ulta/ulta_reviews_enriched.csv`
- `data/processed/Matched/matched_reviews.csv`

**Outputs:**
- `outputs/3.3_did_estimates.png`
- `outputs/3.3_event_study_coefficients.png`
- `outputs/3.3_price_reputation_interaction.png`
- `outputs/3.3_recovery_trajectories.png`
- `outputs/3.3_roi_matrix_sephora.png`
- `outputs/3.3_roi_matrix_ulta.png`
- `outputs/3.3_spillover_did.png`

**Usage:** Run all cells in sequence. Propensity score matching requires `scikit-learn` and `statsmodels`. Pre-trend validation plots are generated mid-notebook and should be reviewed before interpreting ATT estimates — if parallel trends is violated for a given outcome, interpret that ATT with caution.

---

#### `3-4_convergent_strategic_recommendation.ipynb`

**What it does:** Synthesizes all prior analyses into a unified strategic recommendation framework. Loads outputs from all preceding notebooks and applies a rule-based tagging system to classify each brand into actionable strategic profiles: cross-platform harmonizers, channel-specific specialists, at-risk brands requiring intervention, and white-space opportunity candidates. Generates a comprehensive strategy matrix positioning brands on two axes — cross-platform consistency vs. platform-specific value capture potential — with quadrant-specific action recommendations. Also produces the final ROI prioritization framework linking causal ATT estimates to marketing investment recommendations by retailer, price tier, and brand archetype.

**Inputs:**
- All `sephora_brand_health.csv`, `ulta_brand_health.csv`, `sephora_cluster_labels.csv`, `ulta_cluster_labels.csv`, matched product and review files, and selected outputs from Sections 3.1–3.3

**Outputs:**
- `outputs/3.2_strategy_matrix.png` *(shared output path with Section 3.2)*
- Inline recommendation tables and brand-level action summaries

**Usage:** Run all cells. This notebook is primarily analytical and produces its most important outputs as rendered tables and inline text rather than saved image files. Review the printed brand-level recommendation tables carefully — they are the primary deliverable of the full project.

---

### Execution Order Reference

For a clean run from raw data to final recommendations, execute files in this order:

1.  python src/scrape_sephora.py
2.  python src/scrape_ulta.py
3.  python src/data_cleaning.py
4.  python src/brand_matching.py
5.  python src/build_product_graph.py
6.  python src/segmentation_features.py
7.  python src/sephora_sentiment.py
8.  python src/ulta_sentiment.py
9.  python src/sephora_recommendations.py
10. python src/ulta_recommendations.py

Then open Jupyter and run notebooks in this order:

11. notebooks/independent/sephora_segmentation.ipynb
12. notebooks/independent/ulta_segmentation.ipynb
13. notebooks/independent/sephora_sentiment.ipynb
14. notebooks/independent/ulta_sentiment.ipynb
15. notebooks/independent/sephora_recommendations.ipynb
16. notebooks/independent/ulta_recommendations.ipynb
17. notebooks/joint/2-1_product_graph.ipynb
18. notebooks/joint/2-2_matched_segmentation.ipynb
19. notebooks/joint/2-3_setiment_gap.ipynb
20. notebooks/joint/2-4_perceived_value.ipynb
21. notebooks/comparative/3-1_competitive_positioning.ipynb
22. notebooks/comparative/3-1_combined_perceptual_map.ipynb
23. notebooks/comparative/3-2_perceived_value.ipynb
24. notebooks/comparative/3-2_coss_retailer_simulations.ipynb
25. notebooks/comparative/3-3_causal_attribution.ipynb
26. notebooks/comparative/3-4_convergent_strategic_recommendation.ipynb

---

## Use Cases

This project is relevant for digital marketing and growth analytics, consumer packaged goods (CPG) analytics, e-commerce and marketplace strategy teams, brand and product management, pricing and value strategy across retail channels, and applied data science and marketing analytics coursework.

---

## Disclaimer

This project is conducted for educational and research purposes only. All data is sourced from publicly accessible pages and APIs. No proprietary, confidential, or personally identifiable user information is collected.