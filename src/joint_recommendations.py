# src/joint_recommendations.py
# 2.5 — Joint Recommendation Engine (Sephora + Ulta)
#
# Architecture:
#   Loads BOTH segmentation CSVs, merges into a single joint catalog, builds
#   a shared feature space (Sephora skin-tone cols + Ulta verification cols,
#   zero-filled where absent), computes cross-catalog cosine similarities,
#   and generates recommendations that can span both retailers.
#
#   Deduplication: matched_products.csv / matched_pairs.csv are used to
#   identify products that are the same item sold at both retailers. When
#   building recommendations for product X, any cross-retailer match of X
#   is always excluded from the results — you will never be recommended the
#   same product you already have under a different retailer.
#
# Consumes:
#   data/processed/Sephora/sephora_segmentation.csv
#   data/processed/Ulta/ulta_segmentation.csv
#   data/processed/Matched/matched_pairs.csv       (sephora_product_id, ulta_product_id, similarity_score)
#   data/processed/Matched/matched_products.csv    (product_id, retailer, product_name, brand, …)
#
# Outputs:
#   data/processed/Joint/joint_recommendations.csv
#
# Usage:
#   As script:
#     python src/joint_recommendations.py build_csv
#     python src/joint_recommendations.py write_htmls --limit 200
#
#   As import:
#     from src.joint_recommendations import run_pipeline      # build CSV
#     from src.joint_recommendations import show_dashboard    # Jupyter dropdown
#     from src.joint_recommendations import write_product_html
#     from src.joint_recommendations import write_all_product_htmls

import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DATA_DIR      = _PROJECT_ROOT / "data" / "processed"
SEPHORA_DIR   = DATA_DIR / "Sephora"
ULTA_DIR      = DATA_DIR / "Ulta"
MATCHED_DIR   = DATA_DIR / "Matched"
JOINT_DIR     = DATA_DIR / "Joint"

JOINT_RECS_PATH       = JOINT_DIR / "joint_recommendations.csv"
OUTPUT_DIR            = _PROJECT_ROOT / "notebooks" / "joint" / "outputs"
PRODUCT_DASHBOARD_DIR = OUTPUT_DIR / "product_dashboards"

MIN_REVIEWS = 20
TOP_N_SUB   = 5
TOP_N_COMP  = 5
TOP_N_TRADE = 3

# Retailer display config
_THEME = {
    "sephora": dict(
        color        = "#2d6a4f",
        header_fill  = "#1a1a1a",
        hover_bg     = "#f0f4f8",
        link_color   = "#2d6a4f",
        h1_color     = "#2C3E50",
        h1_border    = "#2d6a4f",
        has_verif    = False,
    ),
    "ulta": dict(
        color        = "#880e4f",
        header_fill  = "#880e4f",
        hover_bg     = "#fdf0f5",
        link_color   = "#880e4f",
        h1_color     = "#880e4f",
        h1_border    = "#880e4f",
        has_verif    = True,
    ),
}

_data: dict = {}
_cat_to_gid = {}   # populated by _compute_scores; used by _generate_csv


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _slug(text: str) -> str:
    return re.sub(r"[^\w\-]", "_", str(text)).strip("_")


# ── Category normalization & alias matching ──────────────────────────────────
_CAT_RE = re.compile(r"[^\w\s]")


def _norm_cat(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(_CAT_RE.sub(" ", str(s).lower()).split())


# Explicit cross-retailer aliases: Sephora and Ulta often name the same shelf
# differently. Both sides are normalised before comparison.
# Add more pairs here as you discover mismatches in the overlap report.
# Each entry is a SET of all category names (Sephora + Ulta) that refer to
# the same product shelf. All members share one group id. Built from the
# actual unique category values in both segmentation CSVs.
_CAT_GROUPS: list = [
    # ── Fragrance ─────────────────────────────────────────────────────────
    # S: Cologne, Cologne Gift Sets, Perfume, Perfume Gift Sets, Rollerballs & Travel Size,
    #    Unisex / Genderless
    # U: Cologne, Cologne Gift Sets, Fragrance, Fragrance Gifts, Perfume,
    #    Unisex Fragrance, Women's Fragrance
    {"cologne", "perfume", "fragrance",
     "cologne gift sets", "perfume gift sets", "fragrance gifts",
     "rollerballs  travel size", "rollerballs and travel size",
     "unisex   genderless", "unisex fragrance", "women s fragrance",
     "unisex genderless"},

    # ── BB & CC Cream ─────────────────────────────────────────────────────
    # S: BB & CC Cream   U: BB & CC Creams
    {"bb   cc cream", "bb   cc creams"},

    # ── Blush ─────────────────────────────────────────────────────────────
    # S: Blush   U: Blush
    {"blush"},

    # ── Bronzer ───────────────────────────────────────────────────────────
    # S: Bronzer   U: Bronzer
    {"bronzer"},

    # ── Highlighter ───────────────────────────────────────────────────────
    # S: Highlighter   U: Highlighter
    {"highlighter"},

    # ── Concealer ─────────────────────────────────────────────────────────
    # S: Concealer, Under-Eye Concealer   U: Concealer
    {"concealer", "under eye concealer"},

    # ── Contour ───────────────────────────────────────────────────────────
    # S: Contour   U: Contouring
    {"contour", "contouring"},

    # ── Foundation ────────────────────────────────────────────────────────
    # S: Foundation   U: Foundation
    {"foundation"},

    # ── Tinted Moisturizer ────────────────────────────────────────────────
    # S: Tinted Moisturizer   U: Tinted Moisturizer
    {"tinted moisturizer"},

    # ── Color Correct ─────────────────────────────────────────────────────
    # S: Color Correct   U: Color Correcting
    {"color correct", "color correcting"},

    # ── Face Primer ───────────────────────────────────────────────────────
    # S: Face Primer   U: Face Primer
    {"face primer"},

    # ── Setting Spray & Powder ────────────────────────────────────────────
    # S: Setting Spray & Powder   U: Setting Spray & Powder
    {"setting spray   powder"},

    # ── Cheek Palettes ────────────────────────────────────────────────────
    # S: Cheek Palettes   (no Ulta equiv — solo group)
    {"cheek palettes"},

    # ── Eyeshadow ─────────────────────────────────────────────────────────
    # S: Eyeshadow, Eye Palettes   U: Eyeshadow, Eyeshadow Palettes
    {"eyeshadow", "eye palettes", "eyeshadow palettes"},

    # ── Eyeliner ─────────────────────────────────────────────────────────
    # S: Eyeliner   U: Eyeliner
    {"eyeliner"},

    # ── Mascara ───────────────────────────────────────────────────────────
    # S: Mascara   U: Mascara
    {"mascara"},

    # ── Eyebrow ───────────────────────────────────────────────────────────
    # S: Eyebrow   U: Eyebrows
    {"eyebrow", "eyebrows"},

    # ── False Eyelashes ───────────────────────────────────────────────────
    # S: False Eyelashes   U: Eyelashes
    {"false eyelashes", "eyelashes"},

    # ── Eye Creams ────────────────────────────────────────────────────────
    # S: Eye Creams & Treatments   U: Eye Cream, Eye Serums, Eye Treatments
    {"eye creams   treatments", "eye cream", "eye serums", "eye treatments"},

    # ── Eye Masks ─────────────────────────────────────────────────────────
    # S: Eye Masks   U: Eye Masks
    {"eye masks"},

    # ── Lip Gloss ─────────────────────────────────────────────────────────
    # S: Lip Gloss   U: Lip Gloss, Gloss & Shine
    {"lip gloss", "gloss   shine"},

    # ── Lipstick ──────────────────────────────────────────────────────────
    # S: Lipstick, Liquid Lipstick   U: Lipstick
    {"lipstick", "liquid lipstick"},

    # ── Lip Liner ─────────────────────────────────────────────────────────
    # S: Lip Liner   U: Lip Liner
    {"lip liner"},

    # ── Lip Oil ───────────────────────────────────────────────────────────
    # S: Lip Oil   U: Lip Oil
    {"lip oil"},

    # ── Lip Balm ─────────────────────────────────────────────────────────
    # S: Lip Balms & Treatments   U: Lip Balms
    {"lip balms   treatments", "lip balms"},

    # ── Lip Plumper ───────────────────────────────────────────────────────
    # S: Lip Plumper   U: Lip Plumpers
    {"lip plumper", "lip plumpers"},

    # ── Lip Stain ─────────────────────────────────────────────────────────
    # S: Lip Stain   U: Lip Stain
    {"lip stain"},

    # ── Makeup Palettes ───────────────────────────────────────────────────
    # S: Makeup Palettes   U: Makeup Palettes
    {"makeup palettes"},

    # ── Makeup Removers ───────────────────────────────────────────────────
    # S: Makeup Removers   U: Makeup Remover, Face Wipes
    {"makeup removers", "makeup remover", "face wipes"},

    # ── Moisturizers ─────────────────────────────────────────────────────
    # S: Moisturizers, Face Creams, Night Creams
    # U: Face Moisturizer, Moisturizers, Moisturizers & Treatments, Night Cream, Hydration
    {"moisturizers", "face creams", "night creams",
     "face moisturizer", "moisturizers   treatments", "night cream", "hydration"},

    # ── Face Serums ───────────────────────────────────────────────────────
    # S: Face Serums   U: Face Serums, Treatment & Serums, Treatments & Serums, Oils & Serums
    {"face serums", "treatment   serums", "treatments   serums", "oils   serums",
     "treatment", "treatments"},

    # ── Face Oils ─────────────────────────────────────────────────────────
    # S: Face Oils   U: Face Oils
    {"face oils"},

    # ── Face Masks ────────────────────────────────────────────────────────
    # S: Face Masks   U: Face Masks, Masks
    {"face masks", "masks"},

    # ── Sheet Masks ───────────────────────────────────────────────────────
    # S: Sheet Masks   U: Sheet Masks
    {"sheet masks"},

    # ── Cleansers ─────────────────────────────────────────────────────────
    # S: Cleansers, Face Wash & Cleansers
    # U: Cleansers, Face Wash, Cleansing Balms & Oils, Cleansing Exfoliators
    {"cleansers", "face wash   cleansers", "face wash",
     "cleansing balms   oils", "cleansing exfoliators"},

    # ── Exfoliators ───────────────────────────────────────────────────────
    # S: Exfoliators, Facial Peels, Scrub & Exfoliants
    # U: Face Peels & Exfoliators, Cleansing Exfoliators, Body Scrubs & Exfoliants
    {"exfoliators", "facial peels", "scrub   exfoliants",
     "face peels   exfoliators"},

    # ── Toners ────────────────────────────────────────────────────────────
    # S: Toners, Mists & Essences   U: Toner, Face Mists & Essences
    {"toners", "toner", "mists   essences", "face mists   essences"},

    # ── Sunscreen ─────────────────────────────────────────────────────────
    # S: Face Sunscreen   U: Sunscreen
    {"face sunscreen", "sunscreen"},

    # ── Anti-Aging ────────────────────────────────────────────────────────
    # S: Anti-Aging   U: Anti-Aging
    {"anti aging"},

    # ── Blemish & Acne ────────────────────────────────────────────────────
    # S: Blemish & Acne Treatments   U: Acne & Blemish Treatments
    {"blemish   acne treatments", "acne   blemish treatments"},

    # ── Decollete & Neck ─────────────────────────────────────────────────
    # S: Decollete & Neck Creams   U: Neck Cream
    {"decollete   neck creams", "neck cream"},

    # ── Scalp ────────────────────────────────────────────────────────────
    # S: Scalp Treatments   U: Scalp Care
    {"scalp treatments", "scalp care"},

    # ── Shampoo ───────────────────────────────────────────────────────────
    # S: Shampoo   U: Shampoo, Co-Wash, Dry Shampoo (only exact match here)
    {"shampoo"},

    # ── Dry Shampoo ───────────────────────────────────────────────────────
    # S: Dry Shampoo   U: Dry Shampoo
    {"dry shampoo"},

    # ── Conditioner ───────────────────────────────────────────────────────
    # S: Conditioner   U: Conditioner
    {"conditioner"},

    # ── Leave-In Conditioner ─────────────────────────────────────────────
    # S: Leave-In Conditioner   U: Leave-In Conditioner, Leave-In Treatment
    {"leave in conditioner", "leave in treatment"},

    # ── Hair Masks ────────────────────────────────────────────────────────
    # S: Hair Masks   U: (no exact, but Masks overlaps)
    {"hair masks"},

    # ── Hair Oil ─────────────────────────────────────────────────────────
    # S: Hair Oil   U: (none exact — solo)
    {"hair oil"},

    # ── Hair Styling Products ────────────────────────────────────────────
    # S: Hair Styling Products, Hair Styling & Treatments, Hair Spray, Hair Primers
    # U: Styling Products, Styling, Hairspray, Heat Protectant, Wax & Pomade,
    #    Volume & Texture, Smoothing, Curl Enhancing
    {"hair styling products", "hair styling   treatments", "hair spray", "hair primers",
     "styling products", "styling", "hairspray", "heat protectant",
     "wax   pomade", "volume   texture", "smoothing", "curl enhancing"},

    # ── Hair Dryers ───────────────────────────────────────────────────────
    # S: Hair Dryers   U: Hair Dryers
    {"hair dryers"},

    # ── Curling Irons ─────────────────────────────────────────────────────
    # S: Curling Irons   U: Curling Irons & Stylers
    {"curling irons", "curling irons   stylers"},

    # ── Flat Irons ────────────────────────────────────────────────────────
    # S: Hair Straighteners & Flat Irons   U: Flat Irons
    {"hair straighteners   flat irons", "flat irons"},

    # ── Hair Brushes & Combs ─────────────────────────────────────────────
    # S: Brushes & Combs   U: Hair Brushes & Combs
    {"brushes   combs", "hair brushes   combs"},

    # ── Hair Color ────────────────────────────────────────────────────────
    # S: Hair Dye & Root Touch-Ups   U: Hair Color, Hair Color & Bleach, Root Touch Up
    {"hair dye   root touch ups", "hair color", "hair color   bleach", "root touch up"},

    # ── Hair Supplements ─────────────────────────────────────────────────
    # S: Hair Supplements   U: Hair Thinning & Hair Loss
    {"hair supplements", "hair thinning   hair loss"},

    # ── Body Lotion ───────────────────────────────────────────────────────
    # S: Body Lotions & Body Oils   U: Body Lotion, Body Lotion & Creams, Body Lotions, Body Butters
    {"body lotions   body oils", "body lotion", "body lotion   creams",
     "body lotions", "body butters"},

    # ── Body Wash ─────────────────────────────────────────────────────────
    # S: Body Wash & Shower Gel   U: Shower Gel & Body Wash
    {"body wash   shower gel", "shower gel   body wash"},

    # ── Body Scrubs ───────────────────────────────────────────────────────
    # S: Scrub & Exfoliants   U: Body Scrubs & Exfoliants
    {"scrub   exfoliants", "body scrubs   exfoliants"},

    # ── Body Serums ───────────────────────────────────────────────────────
    # S: For Body   U: Body Serums & Oils, Body Treatments
    {"for body", "body serums   oils", "body treatments"},

    # ── Body Mist ────────────────────────────────────────────────────────
    # S: Body Mist & Hair Mist   U: Body Mist & Hair Mist
    {"body mist   hair mist"},

    # ── Bath Soaks ────────────────────────────────────────────────────────
    # S: Bath Soaks & Bubble Bath   U: Bubble Bath & Soaks, Bath Bombs & Shower Steamers
    {"bath soaks   bubble bath", "bubble bath   soaks", "bath bombs   shower steamers"},

    # ── Hand & Foot ───────────────────────────────────────────────────────
    # S: Hand Cream & Foot Cream   U: Hand Cream & Foot Cream, Hand & Foot Treatment
    {"hand cream   foot cream", "hand   foot treatment"},

    # ── Hand Soap & Sanitizer ─────────────────────────────────────────────
    # S: Hand Sanitizer & Hand Soap   U: Hand Soap & Sanitizers
    {"hand sanitizer   hand soap", "hand soap   sanitizers"},

    # ── Deodorant ────────────────────────────────────────────────────────
    # S: Deodorant & Antiperspirant   U: Deodorant
    {"deodorant   antiperspirant", "deodorant"},

    # ── Candles ───────────────────────────────────────────────────────────
    # S: Candles   U: Candles & Home Fragrance
    {"candles", "candles   home fragrance"},

    # ── Makeup Brushes ────────────────────────────────────────────────────
    # S: Face Brushes   U: Makeup Brushes, Brush Sets
    {"face brushes", "makeup brushes", "brush sets"},

    # ── Brush Cleaners ────────────────────────────────────────────────────
    # S: Brush Cleaners   U: Brush Cleaner
    {"brush cleaners", "brush cleaner"},

    # ── Sponges & Applicators ────────────────────────────────────────────
    # S: Sponges & Applicators   U: Sponges & Applicators
    {"sponges   applicators"},

    # ── Face Brushes/Tools ───────────────────────────────────────────────
    # S: Facial Cleansing Brushes   U: Cleansing Brushes, Facial Rollers, Skincare Tools
    {"facial cleansing brushes", "cleansing brushes", "facial rollers", "skincare tools"},

    # ── Beauty Supplements ───────────────────────────────────────────────
    # S: Beauty Supplements   U: Beauty Supplements, Daily Vitamins & Supplements, Supplements
    {"beauty supplements", "daily vitamins   supplements", "supplements"},

    # ── Nail ─────────────────────────────────────────────────────────────
    # S: Nail   U: Nail Polish, Nail Care, Gel Nail Polish, Top & Base Coats,
    #              Nail Art & Design, Press On Nails, Nail Polish Stickers
    {"nail", "nail polish", "nail care", "gel nail polish",
     "top   base coats", "nail art   design", "press on nails",
     "nail polish stickers"},

    # ── Accessories ──────────────────────────────────────────────────────
    # S: Accessories   U: Accessories
    {"accessories"},

    # ── Hair Accessories ─────────────────────────────────────────────────
    # S: Hair Clips & Claw Clips, Scrunchies & Hair Ties
    # U: Clips & Bobby Pins, Elastics, Headbands, Styling Accessories
    {"hair clips   claw clips", "scrunchies   hair ties",
     "clips   bobby pins", "elastics", "headbands", "styling accessories"},

    # ── Intimate Care ────────────────────────────────────────────────────
    # S: Intimate Care   U: Intimate Wellness, Sexual Wellness
    {"intimate care", "intimate wellness", "sexual wellness"},

    # ── Self-Tanner ───────────────────────────────────────────────────────
    # S: (none)   U: Self-Tanning & Bronzing, After Sun Care
    {"self tanning   bronzing", "after sun care"},
]

_CAT_GROUP: dict = {}
for _gid, _group_set in enumerate(_CAT_GROUPS):
    for _cat in _group_set:
        _n = _norm_cat(_cat)
        _CAT_GROUP.setdefault(_n, _gid)


def _same_category(a: str, b: str) -> bool:
    """True if two category strings refer to the same product shelf."""
    na, nb = _norm_cat(a), _norm_cat(b)
    if na == nb:
        return True
    ga, gb = _CAT_GROUP.get(na), _CAT_GROUP.get(nb)
    return ga is not None and ga == gb


def _build_gid_arr(cat_values) -> "np.ndarray":
    """
    Convert an array of category strings into a group-ID integer array.
    Products in the same category (or alias group) get the same integer.
    Populates the module-level _cat_to_gid dict as a side-effect so that
    _generate_csv can reuse it for O(1) per-product checks.
    Returns a numpy int32 array of length len(cat_values).
    """
    global _cat_to_gid
    _cat_to_gid = {}
    next_gid = [0]

    def _gid_for(nc: str) -> int:
        if nc in _cat_to_gid:
            return _cat_to_gid[nc]
        ag = _CAT_GROUP.get(nc)
        if ag is not None:
            existing = [_cat_to_gid[c] for c in _cat_to_gid if _CAT_GROUP.get(c) == ag]
            gid = existing[0] if existing else next_gid[0]
            if not existing:
                next_gid[0] += 1
        else:
            gid = next_gid[0]
            next_gid[0] += 1
        _cat_to_gid[nc] = gid
        return gid

    # Two-pass: first populate dict with all unique cats, then build array
    unique = list(dict.fromkeys(_norm_cat(c) for c in cat_values))
    for uc in unique:
        _gid_for(uc)

    return np.array([_cat_to_gid[_norm_cat(c)] for c in cat_values], dtype=np.int32)


def _to_str(s: pd.Series) -> pd.Series:
    return s.astype("string")


def _safe_float(v, default=0.0) -> float:
    try:
        return float(pd.to_numeric(v, errors="coerce") or default)
    except Exception:
        return default


def _safe_int(v, default=0) -> int:
    try:
        return int(pd.to_numeric(v, errors="coerce") or default)
    except Exception:
        return default


def _verif_color(vb: float, disc: float) -> str:
    if disc > 0.20: return "#e67e22"   # seeded
    if vb  > 0.20: return "#27ae60"   # organic verified
    return "#3498db"                   # low


def _verif_tag(vb: float, disc: float) -> str:
    if disc > 0.20: return f"Seeded {disc:.0%}"
    if vb  > 0.20: return f"Verified {vb:.0%}"
    return "Low verif."


def _retailer_badge(retailer: str) -> str:
    if retailer == "ulta":
        return "<span style='background:#880e4f;color:white;padding:1px 6px;border-radius:3px;font-size:10px;'>ULTA</span>"
    return "<span style='background:#2d6a4f;color:white;padding:1px 6px;border-radius:3px;font-size:10px;'>SEPHORA</span>"


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-MATCH LOOKUP  (product_id → set of excluded product_ids)
# ══════════════════════════════════════════════════════════════════════════════

def _build_exclusion_map(matched_pairs: pd.DataFrame) -> dict:
    """
    Returns {product_id_str: {matched_id_str, ...}}
    A product should never recommend its own cross-retailer match.
    """
    excl: dict = {}
    for _, row in matched_pairs.iterrows():
        s = str(row.get("sephora_product_id", ""))
        u = str(row.get("ulta_product_id",   ""))
        if s and u:
            excl.setdefault(s, set()).add(u)
            excl.setdefault(u, set()).add(s)
    return excl


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BLOCKS  (joint, retailer-aware)
# ══════════════════════════════════════════════════════════════════════════════

def _build_feature_blocks(df: pd.DataFrame):
    """
    Build feature blocks on the combined (Sephora + Ulta) catalog.
    Sephora-only and Ulta-only columns are zero-filled for the other retailer.
    """
    blocks = {}

    # ── A. Structural ─────────────────────────────────────────────────────────
    cat_dummies = pd.get_dummies(df["category"], prefix="cat").astype(float)

    # Sephora skin signals (zero for Ulta rows)
    skin_cols = [c for c in df.columns if
                 c.startswith("pct_skin_tone_") or c.startswith("pct_skin_type_")
                 or c in ("pct_has_skin_tone", "pct_has_skin_type")]
    skin_block = df[skin_cols].fillna(0) if skin_cols else pd.DataFrame(index=df.index)

    # Ulta verification signals (zero for Sephora rows)
    verif_cols = [c for c in ("pct_verified_buyer", "pct_verified_reviewer",
                               "pct_would_recommend", "pct_has_disclosure",
                               "helpfulness_rate") if c in df.columns]
    verif_block = df[verif_cols].fillna(0) if verif_cols else pd.DataFrame(index=df.index)

    structural = pd.concat([cat_dummies, skin_block, verif_block], axis=1).fillna(0)
    blocks["structural"] = structural.values
    print(f"    A. Structural:  {structural.shape[1]} features  "
          f"({len(skin_cols)} skin · {len(verif_cols)} verif)")

    # ── B. Sentiment ──────────────────────────────────────────────────────────
    sentiment_cols = [c for c in (
        "avg_sentiment", "sentiment_std", "pct_positive", "pct_negative", "pct_neutral",
        "avg_rating", "rating_std",
        "pct_5_star", "pct_4_star", "pct_3_star", "pct_2_star", "pct_1_star",
        "rating_dispersion_over_time",
    ) if c in df.columns]
    if "avg_rating" in df.columns and "avg_sentiment" in df.columns:
        df = df.copy()
        df["rating_sentiment_mismatch"] = abs(
            (df["avg_rating"] - 1) / 4 - (df["avg_sentiment"] + 1) / 2
        )
        sentiment_cols.append("rating_sentiment_mismatch")
    blocks["sentiment"] = df[sentiment_cols].fillna(0).values
    print(f"    B. Sentiment:   {len(sentiment_cols)} features")

    # ── C. Content (LDA topics) ───────────────────────────────────────────────
    topic_dummies = pd.DataFrame(index=df.index)
    if "dominant_topic_mode" in df.columns:
        topic_dummies = pd.get_dummies(
            df["dominant_topic_mode"].astype(int), prefix="dom_topic"
        ).astype(float)
    topic_extra = df[["top_topic_prevalence"]].fillna(0) \
        if "top_topic_prevalence" in df.columns else pd.DataFrame(index=df.index)
    content = pd.concat([topic_dummies, topic_extra], axis=1).fillna(0)
    blocks["content"] = content.values
    print(f"    C. Content:     {content.shape[1]} features")

    # ── D. Price ──────────────────────────────────────────────────────────────
    price_f = pd.DataFrame(index=df.index)
    if "price" in df.columns:
        price_f["log_price"]    = np.log1p(df["price"].fillna(0))
        price_f["price_pctile"] = df["price"].rank(pct=True)
    if "price_vs_category_median" in df.columns:
        price_f["price_vs_cat"] = df["price_vs_category_median"].fillna(1.0)
    blocks["price"] = price_f.fillna(0).values
    print(f"    D. Price:       {price_f.shape[1]} features")

    return blocks, df


# ══════════════════════════════════════════════════════════════════════════════
# SIMILARITIES
# ══════════════════════════════════════════════════════════════════════════════

def _compute_similarities(blocks: dict) -> dict:
    sims, scaler = {}, StandardScaler()
    for name, matrix in blocks.items():
        if matrix.shape[1] == 0:
            continue
        scaled = scaler.fit_transform(np.nan_to_num(matrix, 0))
        sim = cosine_similarity(scaled)
        np.fill_diagonal(sim, 0)
        sims[name] = sim
        print(f"    {name:<15} → {sim.shape[0]}×{sim.shape[1]}")
    return sims


# ══════════════════════════════════════════════════════════════════════════════
# SCORES
# ══════════════════════════════════════════════════════════════════════════════

def _compute_scores(sims: dict, df: pd.DataFrame):
    """
    Joint scoring — ALL four intents are strictly same-category.

    same_cat is multiplied into every score matrix so off-category pairs are
    zeroed at the matrix level. _generate_csv additionally checks cats[j] != src_cat
    as a belt-and-suspenders guard during top-N selection.

    Complements (within same category) are differentiated from Substitutes by
    relying more on content/topic divergence — e.g., a concealer recommended
    alongside a foundation, both in Face Makeup.
    """
    n          = len(df)
    content    = sims.get("content",    np.zeros((n, n)))
    sentiment  = sims.get("sentiment",  np.zeros((n, n)))
    price_sim  = sims.get("price",      np.zeros((n, n)))
    structural = sims.get("structural", np.zeros((n, n)))

    # Build same_cat matrix — O(n), vectorised.
    # _build_gid_arr populates module-level _cat_to_gid and returns an int array.
    # same_cat[i,j]==1 iff products i and j share a category (exact or alias).
    gid_arr  = _build_gid_arr(df["category"].values)
    same_cat = (gid_arr[:, None] == gid_arr[None, :]).astype(float)
    # cross_cat removed — all intents are same-category

    ret_arr   = np.array(df["retailer"].values)
    same_ret  = (ret_arr[:, None] == ret_arr[None, :]).astype(float)
    diff_ret  = 1.0 - same_ret                # bonus for cross-retailer recs

    rc        = df["review_count"].fillna(0).values.astype(float)
    stability = np.sqrt(rc) / (np.sqrt(rc).max() + 1e-8)
    penalty   = 1.0 - (rc < MIN_REVIEWS).astype(float) * 0.7
    tw        = stability * penalty

    # All intents are same-category only.
    # Hard same-category masking is also applied in _generate_csv (belt-and-suspenders).

    # Substitutes — same category, similar perception + price tier
    sub  = (0.35 * content + 0.25 * sentiment + 0.20 * structural
            + 0.10 * price_sim + 0.10 * same_cat)
    sub *= same_cat                  # hard zero for any different-category item
    sub *= tw[np.newaxis, :]

    # Complements — same category, different sentiment/topic profile;
    # slight nudge toward cross-retailer discovery within the same category
    comp  = (0.35 * content + 0.25 * sentiment + 0.20 * structural
             + 0.15 * same_cat + 0.05 * diff_ret)
    comp *= same_cat                 # hard zero for any different-category item
    comp *= tw[np.newaxis, :]

    # Trade — same category, price filter applied below
    base      = (0.50 * content + 0.30 * sentiment + 0.20 * structural)
    base     *= same_cat * tw[np.newaxis, :]   # hard zero for different-category items
    prices    = df["price"].fillna(0).values.astype(float)
    trade_up  = base * (prices[np.newaxis, :] > prices[:, np.newaxis] * 1.15).astype(float)
    trade_dn  = base * (prices[np.newaxis, :] < prices[:, np.newaxis] * 0.85).astype(float)

    return sub, comp, trade_up, trade_dn


# ══════════════════════════════════════════════════════════════════════════════
# CSV GENERATION  (with cross-match deduplication)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_csv(df: pd.DataFrame, sub, comp, trade_up, trade_dn,
                  excl_map: dict) -> pd.DataFrame:
    """
    excl_map: {product_id_str: {excluded_id_str, ...}}
    For each source product, any id in its exclusion set is skipped when
    selecting top-N recommendations (cross-retailer duplicates).
    """
    n        = len(df)
    name_col = "product_name" if "product_name" in df.columns else "name"

    pids     = df["product_id"].astype(str).values
    names    = df[name_col].fillna("").values if name_col in df.columns else [""] * n
    brands   = df["brand"].fillna("").values  if "brand"   in df.columns else [""] * n
    cats     = df["category"].fillna("").values
    # pre-build group-id array for O(1) same-category checks in the inner loop
    # (reuse _cat_to_gid populated during _compute_scores; fall back to _gid_for)
    _gid_arr = np.array([_cat_to_gid.get(_norm_cat(c), -i-9999)
                         for i, c in enumerate(cats)], dtype=np.int32)
    rets     = df["retailer"].fillna("").values
    prices   = df["price"].fillna(0).values.astype(float)
    ratings  = df["avg_rating"].fillna(0).values.astype(float)
    sents    = df["avg_sentiment"].fillna(0).values.astype(float)
    rcounts  = df["review_count"].fillna(0).values.astype(int)

    vb   = df["pct_verified_buyer"].fillna(0).values  if "pct_verified_buyer"  in df.columns else np.zeros(n)
    disc = df["pct_has_disclosure"].fillna(0).values  if "pct_has_disclosure"  in df.columns else np.zeros(n)

    # pid → column index for fast exclusion lookup
    pid_to_idx = {pids[j]: j for j in range(n)}

    rows = []
    for i in range(n):
        src_pid  = pids[i]
        excluded = excl_map.get(src_pid, set()) | {src_pid}  # always exclude self too

        row = dict(
            product_id    = src_pid,
            product_name  = names[i],
            brand         = brands[i],
            category      = cats[i],
            retailer      = rets[i],
            price         = prices[i],
            avg_rating    = ratings[i],
            avg_sentiment = sents[i],
            review_count  = int(rcounts[i]),
            pct_verified_buyer = round(float(vb[i]),   4),
            pct_has_disclosure = round(float(disc[i]), 4),
        )

        for prefix, matrix, top_n in [
            ("sub",     sub,      TOP_N_SUB),
            ("comp",    comp,     TOP_N_COMP),
            ("tradeup", trade_up, TOP_N_TRADE),
            ("tradedn", trade_dn, TOP_N_TRADE),
        ]:
            # Two-pass selection:
            # Pass 1 — fill up to ceil(top_n/2) slots from the OTHER retailer (same cat)
            # Pass 2 — fill remaining slots from ANY retailer (same cat), best score first
            # This guarantees cross-retailer representation whenever products exist.
            src_cat  = cats[i]
            _src_gid = _gid_arr[i]
            src_ret  = rets[i]
            sorted_j = np.argsort(matrix[i])[::-1]

            cross, same_ret_pool = [], []
            for j in sorted_j:
                if pids[j] in excluded:
                    continue
                if _gid_arr[j] != _src_gid:   # vectorised same-category check
                    continue
                if rets[j] != src_ret:
                    cross.append(j)
                else:
                    same_ret_pool.append(j)

            # Reserve up to ceil(top_n/2) slots for cross-retailer; fill rest same-retailer
            cross_slots = (top_n + 1) // 2   # ceil(top_n / 2)
            cross_take  = cross[:cross_slots]
            # Fill remaining from same-retailer, keeping overall order by score
            remaining   = top_n - len(cross_take)
            same_take   = same_ret_pool[:remaining]

            # Merge and re-sort by score so the final ranking is score-ordered
            combined_idx = cross_take + same_take
            combined_idx.sort(key=lambda j: matrix[i, j], reverse=True)
            selected = combined_idx[:top_n]

            for rank, j in enumerate(selected):
                p = f"{prefix}_{rank+1}_"
                row[p+"id"]              = pids[j]
                row[p+"name"]            = names[j]
                row[p+"brand"]           = brands[j]
                row[p+"retailer"]        = rets[j]
                row[p+"score"]           = round(float(matrix[i, j]), 4)
                row[p+"price"]           = prices[j]
                row[p+"sentiment"]       = round(sents[j],   3)
                row[p+"category"]        = cats[j]
                row[p+"rating"]          = round(ratings[j], 2)
                row[p+"reviews"]         = int(rcounts[j])
                row[p+"verified_buyer"]  = round(float(vb[j]),   4)
                row[p+"disclosure"]      = round(float(disc[j]), 4)
        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    """
    Build joint_recommendations.csv from scratch:
      1. Load both segmentation CSVs and tag with retailer column
      2. Load matched_pairs for cross-match deduplication
      3. Merge into one joint catalog (shared feature space, zero-filled)
      4. Build features, compute similarities, score
      5. Generate CSV — cross-retailer duplicates excluded from every product's recs
    """
    t0 = time.time()
    JOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("█" * 70)
    print("  JOINT RECOMMENDATION SYSTEM (2.5)")
    print("█" * 70)

    # ── 1. Load segmentations ─────────────────────────────────────────────────
    print("\n  Loading segmentation files...")
    seph = pd.read_csv(SEPHORA_DIR / "sephora_segmentation.csv")
    ulta = pd.read_csv(ULTA_DIR    / "ulta_segmentation.csv")
    seph["retailer"] = "sephora"
    ulta["retailer"] = "ulta"
    seph["product_id"] = _to_str(seph["product_id"])
    ulta["product_id"] = _to_str(ulta["product_id"])

    for df_r in (seph, ulta):
        df_r["price"]        = pd.to_numeric(df_r["price"],        errors="coerce")
        df_r["review_count"] = pd.to_numeric(df_r.get("review_count", 0), errors="coerce").fillna(0)

    print(f"    Sephora: {len(seph):,} products × {seph.shape[1]} cols")
    print(f"    Ulta:    {len(ulta):,} products × {ulta.shape[1]} cols")

    # ── 2. Load cross-match data ───────────────────────────────────────────────
    print("\n  Loading matched pairs...")
    matched_pairs = pd.read_csv(MATCHED_DIR / "matched_pairs.csv")
    matched_pairs["sephora_product_id"] = _to_str(matched_pairs["sephora_product_id"])
    matched_pairs["ulta_product_id"]    = _to_str(matched_pairs["ulta_product_id"])
    print(f"    {len(matched_pairs):,} cross-retailer matches loaded")

    excl_map = _build_exclusion_map(matched_pairs)
    print(f"    {len(excl_map):,} products have at least one cross-match to exclude")

    # ── 3. Merge into joint catalog ────────────────────────────────────────────
    print("\n  Merging into joint catalog...")
    # Rename Ulta name column if needed
    for df_r in (seph, ulta):
        if "name" in df_r.columns and "product_name" not in df_r.columns:
            df_r.rename(columns={"name": "product_name"}, inplace=True)

    combined = pd.concat([seph, ulta], ignore_index=True, sort=False).fillna(0)
    # Restore proper NaN for price (filled 0 is fine for ranking; flag for display)
    combined["price"] = pd.to_numeric(combined["price"], errors="coerce")

    # Normalise category column in place so feature blocks & scoring are consistent
    combined["category"] = combined["category"].apply(_norm_cat)
    print(f"    Joint catalog: {len(combined):,} products × {combined.shape[1]} cols")

    # ── 3b. Category overlap report ───────────────────────────────────────────
    print("\n  Category overlap (Sephora ↔ Ulta):")
    seph_cats = set(combined.loc[combined["retailer"]=="sephora","category"].unique())
    ulta_cats = set(combined.loc[combined["retailer"]=="ulta",   "category"].unique())
    exact_match  = seph_cats & ulta_cats
    alias_only_s = set()   # Sephora cats matched via alias but not exact
    alias_only_u = set()
    for sc in seph_cats - exact_match:
        for uc in ulta_cats - exact_match:
            if _same_category(sc, uc):
                alias_only_s.add(sc); alias_only_u.add(uc)
    seph_only    = seph_cats - exact_match - alias_only_s
    ulta_only    = ulta_cats - exact_match - alias_only_u
    print(f"    Exact match:  {len(exact_match)} categories")
    if exact_match:
        print("      " + ", ".join(sorted(exact_match)[:20]))
    print(f"    Alias match:  {len(alias_only_s)} Sephora / {len(alias_only_u)} Ulta categories")
    if alias_only_s:
        pairs = [(s,u) for s in alias_only_s for u in alias_only_u if _same_category(s,u)]
        for s,u in sorted(pairs)[:10]:
            print(f"      '{s}'  ←→  '{u}'")
    print(f"    Sephora-only: {len(seph_only)} categories")
    if seph_only:
        print("      " + ", ".join(sorted(seph_only)[:15]))
    print(f"    Ulta-only:    {len(ulta_only)} categories")
    if ulta_only:
        print("      " + ", ".join(sorted(ulta_only)[:15]))
    print(f"    NOTE: products in Sephora-only or Ulta-only categories will receive")
    print(f"          same-retailer recommendations only. Add aliases above to fix.")

    # ── 4. Build features ─────────────────────────────────────────────────────
    print("\n  Building feature blocks...")
    blocks, combined = _build_feature_blocks(combined)

    print("\n  Computing similarities...")
    sims = _compute_similarities(blocks)

    print("\n  Computing scores...")
    sub, comp, trade_up, trade_dn = _compute_scores(sims, combined)

    print("\n  Generating recommendations CSV (with cross-match deduplication)...")
    recs_df = _generate_csv(combined, sub, comp, trade_up, trade_dn, excl_map)

    # ── 5. Validate ───────────────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print("  VALIDATION")

    # Confirm no cross-match duplicates slipped through
    pid_to_retailer = dict(zip(combined["product_id"].astype(str), combined["retailer"]))
    dup_found = 0
    for _, row in recs_df.iterrows():
        src = str(row["product_id"])
        excluded = excl_map.get(src, set())
        for prefix, top_n in [("sub",TOP_N_SUB),("comp",TOP_N_COMP),
                               ("tradeup",TOP_N_TRADE),("tradedn",TOP_N_TRADE)]:
            for rk in range(1, top_n + 1):
                rid = str(row.get(f"{prefix}_{rk}_id",""))
                if rid in excluded:
                    dup_found += 1
    print(f"  Cross-match duplicates in recs: {dup_found}  (should be 0)")

    # Same-category rate across ALL intents (should be 100%)
    id_to_cat = dict(zip(combined["product_id"].astype(str), combined["category"]))
    for prefix, label, top_n in [("sub","Substitutes",TOP_N_SUB),
                                  ("comp","Complements",TOP_N_COMP),
                                  ("tradeup","Trade-Up",TOP_N_TRADE),
                                  ("tradedn","Trade-Down",TOP_N_TRADE)]:
        total, same = 0, 0
        for _, row in recs_df.iterrows():
            src_cat = id_to_cat.get(str(row["product_id"]), "")
            for rk in range(1, top_n + 1):
                sid = row.get(f"{prefix}_{rk}_id")
                if pd.notna(sid):
                    total += 1
                    if _same_category(id_to_cat.get(str(sid), ""), src_cat):
                        same += 1
        pct = same/total*100 if total else 0
        flag = "✓" if pct == 100.0 else "✗"
        print(f"  {flag} {label:<14} same-category: {same}/{total} ({pct:.1f}%)")

    # Cross-retailer coverage in substitutes
    cross_ret = 0
    for _, row in recs_df.iterrows():
        src_ret = str(row.get("retailer", ""))
        for rk in range(1, TOP_N_SUB + 1):
            rret = str(row.get(f"sub_{rk}_retailer",""))
            if rret and rret != src_ret:
                cross_ret += 1
    total_sub_slots = len(recs_df) * TOP_N_SUB
    print(f"  Cross-retailer substitutes:    {cross_ret:,}/{total_sub_slots:,} "
          f"({cross_ret/total_sub_slots*100:.1f}%)")

    recs_df.to_csv(JOINT_RECS_PATH, index=False)
    print(f"\n  ✓ Saved: {JOINT_RECS_PATH}  ({recs_df.shape})")
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"\n  Notebook:  show_dashboard()")
    print(f"  Browser:   write_all_product_htmls(limit=200)")
    print(f"\n{'█'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (lazy)
# ══════════════════════════════════════════════════════════════════════════════

def _load_data():
    if _data:
        return

    if not JOINT_RECS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {JOINT_RECS_PATH}\n"
            "Run:  python src/joint_recommendations.py build_csv"
        )

    t0   = time.time()
    print("Loading joint recommendation data...")
    recs = pd.read_csv(JOINT_RECS_PATH)
    recs["product_id"] = _to_str(recs["product_id"])
    if "retailer" not in recs.columns:
        recs["retailer"] = "unknown"

    _data["recs"] = recs

    products = []
    for _, row in recs.iterrows():
        ret   = str(row.get("retailer","")).lower()
        brand = str(row.get("brand",""))
        name  = str(row.get("product_name",""))
        price = _safe_float(row.get("price", 0))
        label = f"[{ret.upper()}] {brand} — {name} (${price:.0f})"
        products.append((label, str(row["product_id"])))
    products.sort(key=lambda x: x[0])
    _data["product_list"] = products

    print(f"  Loaded {len(products):,} products in {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# ITEM EXTRACTION  (shared by dashboard + export)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_items(row: pd.Series, prefix: str, top_n: int) -> tuple:
    items = []
    for rk in range(1, top_n + 1):
        nm = row.get(f"{prefix}_{rk}_name", "")
        if pd.isna(nm) or nm == "":
            continue
        ret    = str(row.get(f"{prefix}_{rk}_retailer", "")).lower()
        vb_v   = _safe_float(row.get(f"{prefix}_{rk}_verified_buyer", 0))
        disc_v = _safe_float(row.get(f"{prefix}_{rk}_disclosure",     0))
        items.append(dict(
            name    = str(nm),
            brand   = str(row.get(f"{prefix}_{rk}_brand",     "")),
            retailer= ret,
            price   = _safe_float(row.get(f"{prefix}_{rk}_price",     0)),
            sent    = _safe_float(row.get(f"{prefix}_{rk}_sentiment",  0)),
            score   = _safe_float(row.get(f"{prefix}_{rk}_score",      0)),
            cat     = str(row.get(f"{prefix}_{rk}_category", "")),
            rating  = _safe_float(row.get(f"{prefix}_{rk}_rating",     0)),
            reviews = _safe_int(  row.get(f"{prefix}_{rk}_reviews",    0)),
            vb      = vb_v,
            disc    = disc_v,
            vtag    = _verif_tag(vb_v, disc_v) if ret == "ulta" else "",
        ))

    ret_label = {
        "sephora": "<b style='color:#2d6a4f'>[S]</b>",
        "ulta":    "<b style='color:#880e4f'>[U]</b>",
    }
    short = [f"#{i+1} {ret_label.get(it['retailer'],'')} {it['brand'][:14]}"
             for i, it in enumerate(items)]
    hover = []
    for it in items:
        tag  = f"[{it['retailer'].upper()}] " if it["retailer"] else ""
        h    = (f"<b>{tag}{it['brand']}</b><br>{it['name']}<br>"
                f"Category: {it['cat']}<br>"
                f"${it['price']:.0f} · ★{it['rating']:.2f} · Sent {it['sent']:.3f}<br>"
                f"{it['reviews']:,} reviews · Score {it['score']:.4f}")
        if it["retailer"] == "ulta" and it["vtag"]:
            h += f"<br><b>{it['vtag']}</b>"
        hover.append(h)

    return items, short, hover


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def build_product_dashboard(product_id: str):
    """
    Build a cross-catalog recommendation dashboard for one product.
    Recommendations can come from EITHER retailer; cross-retailer duplicates
    are already excluded in the CSV. Ulta items show verification provenance;
    Sephora items do not.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _load_data()
    recs = _data["recs"]

    match = recs[recs["product_id"].astype(str) == str(product_id)]
    if len(match) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Product not found", showarrow=False, font=dict(size=20))
        return fig

    row      = match.iloc[0]
    retailer = str(row.get("retailer", "")).lower()
    theme    = _THEME.get(retailer, _THEME["sephora"])

    src_price = _safe_float(row.get("price",         0))
    src_sent  = _safe_float(row.get("avg_sentiment", 0))
    src_vb    = _safe_float(row.get("pct_verified_buyer", 0))
    src_disc  = _safe_float(row.get("pct_has_disclosure",  0))

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Close Substitutes — Price",      "Close Substitutes — Sentiment",
            "Complements — Price",            "Complements — Sentiment",
            "Trade-Up — Price",               "Trade-Down — Price",
            "Avg Score by Intent",            "All Recommendations",
        ),
        vertical_spacing=0.10, horizontal_spacing=0.20,
        row_heights=[0.30, 0.30, 0.20, 0.20],
        specs=[
            [{"type":"bar"},   {"type":"bar"}],
            [{"type":"bar"},   {"type":"bar"}],
            [{"type":"bar"},   {"type":"bar"}],
            [{"type":"bar"},   {"type":"table"}],
        ],
    )

    def _bar_color(it: dict, fallback: str) -> str:
        """Per-item color: Ulta items colored by verification, Sephora by theme."""
        if it["retailer"] == "ulta":
            return _verif_color(it["vb"], it["disc"])
        return fallback

    # ── Row 1: Substitutes ──────────────────────────────────────────────────
    items, short, hover = _extract_items(row, "sub", TOP_N_SUB)
    if items:
        bar_cols = [_bar_color(it, theme["color"]) for it in items]
        fig.add_trace(go.Bar(
            y=short[::-1], x=[it["price"] for it in items][::-1], orientation="h",
            marker_color=bar_cols[::-1], showlegend=False,
            text=[f"${it['price']:.0f}" for it in items][::-1],
            textposition="inside", textfont=dict(color="white", size=11),
            hovertext=hover[::-1], hoverinfo="text",
        ), row=1, col=1)
        fig.add_shape(type="line", x0=src_price, x1=src_price,
                      y0=-0.5, y1=len(items)-0.5,
                      line=dict(color="red", dash="dash", width=2), row=1, col=1)

        colors_s = ["#27ae60" if it["sent"] >= src_sent else "#e74c3c" for it in items]
        fig.add_trace(go.Bar(
            y=short[::-1], x=[it["sent"] for it in items][::-1], orientation="h",
            marker_color=colors_s[::-1], showlegend=False,
            text=[f"{it['sent']:.3f}" for it in items][::-1],
            textposition="inside", textfont=dict(color="white", size=11),
            hovertext=hover[::-1], hoverinfo="text",
        ), row=1, col=2)
        fig.add_shape(type="line", x0=src_sent, x1=src_sent,
                      y0=-0.5, y1=len(items)-0.5,
                      line=dict(color="red", dash="dash", width=2), row=1, col=2)

    # ── Row 2: Complements ──────────────────────────────────────────────────
    items, short, hover = _extract_items(row, "comp", TOP_N_COMP)
    if items:
        short_c  = [f"#{i+1} {it['retailer'].upper()[:1]} {it['cat'][:18]}"
                    for i, it in enumerate(items)]
        bar_cols = [_bar_color(it, "#7b2d8b") for it in items]
        fig.add_trace(go.Bar(
            y=short_c[::-1], x=[it["price"] for it in items][::-1], orientation="h",
            marker_color=bar_cols[::-1], showlegend=False,
            text=[f"${it['price']:.0f}" for it in items][::-1],
            textposition="inside", textfont=dict(color="white", size=11),
            hovertext=hover[::-1], hoverinfo="text",
        ), row=2, col=1)
        fig.add_trace(go.Bar(
            y=short_c[::-1], x=[it["sent"] for it in items][::-1], orientation="h",
            marker_color=bar_cols[::-1], showlegend=False, opacity=0.75,
            text=[f"{it['sent']:.3f}" for it in items][::-1],
            textposition="inside", textfont=dict(color="white", size=11),
            hovertext=hover[::-1], hoverinfo="text",
        ), row=2, col=2)

    # ── Row 3: Trade-Up / Trade-Down ───────────────────────────────────────
    for intent, base_col, col_n in [("tradeup","#1a73e8",1), ("tradedn","#e67700",2)]:
        items, short, hover = _extract_items(row, intent, TOP_N_TRADE)
        if items:
            bar_cols = [_bar_color(it, base_col) for it in items]
            sign     = "+" if intent == "tradeup" else "-"
            texts    = [f"${it['price']:.0f} ({sign}${abs(it['price']-src_price):.0f})"
                        for it in items]
            fig.add_trace(go.Bar(
                y=short[::-1], x=[it["price"] for it in items][::-1], orientation="h",
                marker_color=bar_cols[::-1], showlegend=False,
                text=texts[::-1], textposition="inside",
                textfont=dict(color="white", size=11),
                hovertext=hover[::-1], hoverinfo="text",
            ), row=3, col=col_n)
            fig.add_shape(type="line", x0=src_price, x1=src_price,
                          y0=-0.5, y1=len(items)-0.5,
                          line=dict(color="red", dash="dash", width=2), row=3, col=col_n)

    # ── Row 4, Col 1: Avg score by intent ──────────────────────────────────
    intent_names, intent_scores = [], []
    for prefix, label, top_n in [("sub","Substitutes",TOP_N_SUB),
                                  ("comp","Complements",TOP_N_COMP),
                                  ("tradeup","Trade-Up",TOP_N_TRADE),
                                  ("tradedn","Trade-Down",TOP_N_TRADE)]:
        sc = [_safe_float(row.get(f"{prefix}_{rk}_score", 0)) for rk in range(1, top_n+1)]
        intent_names.append(label)
        intent_scores.append(float(np.mean(sc)) if sc else 0.0)

    fig.add_trace(go.Bar(
        x=intent_names, y=intent_scores,
        marker_color=["#2d6a4f","#7b2d8b","#1a73e8","#e67700"],
        text=[f"{s:.4f}" for s in intent_scores], textposition="auto",
        showlegend=False,
    ), row=4, col=1)

    # ── Row 4, Col 2: Full detail table ────────────────────────────────────
    t_intent=[];t_ret=[];t_brand=[];t_name=[];t_cat=[];t_price=[];t_score=[];t_verif=[]
    for prefix, ilabel, top_n in [("sub","Substitute",TOP_N_SUB),
                                   ("comp","Complement",TOP_N_COMP),
                                   ("tradeup","Trade-Up",TOP_N_TRADE),
                                   ("tradedn","Trade-Down",TOP_N_TRADE)]:
        for rk in range(1, top_n + 1):
            nm = row.get(f"{prefix}_{rk}_name", "")
            if pd.isna(nm) or nm == "":
                continue
            t_intent.append(ilabel)
            t_ret.append(  str(row.get(f"{prefix}_{rk}_retailer", "")).upper()[:1])
            t_brand.append(str(row.get(f"{prefix}_{rk}_brand",    ""))[:18])
            t_name.append( str(nm)[:26])
            t_cat.append(  str(row.get(f"{prefix}_{rk}_category", ""))[:16])
            t_price.append(f"${_safe_float(row.get(f'{prefix}_{rk}_price',0)):.0f}")
            t_score.append(f"{_safe_float(row.get(f'{prefix}_{rk}_score', 0)):.4f}")
            vb_v   = _safe_float(row.get(f"{prefix}_{rk}_verified_buyer", 0))
            disc_v = _safe_float(row.get(f"{prefix}_{rk}_disclosure",     0))
            ret_v  = str(row.get(f"{prefix}_{rk}_retailer","")).lower()
            if ret_v == "ulta":
                t_verif.append("Seeded" if disc_v > 0.2 else
                                "Verified" if vb_v > 0.2 else "Low")
            else:
                t_verif.append("—")

    ic = {"Substitute":"#e8f5e9","Complement":"#f3e5f5",
          "Trade-Up":"#e3f2fd","Trade-Down":"#fff3e0"}
    fc = [ic.get(i,"#fff") for i in t_intent]

    fig.add_trace(go.Table(
        header=dict(
            values=["Intent","Rtlr","Brand","Product","Cat.","Price","Score","Verif."],
            font=dict(size=9, color="white"),
            fill_color=theme["header_fill"], align="left",
        ),
        cells=dict(
            values=[t_intent,t_ret,t_brand,t_name,t_cat,t_price,t_score,t_verif],
            font=dict(size=8),
            fill_color=[fc]*8, align="left",
        ),
    ), row=4, col=2)

    # ── KPI header ─────────────────────────────────────────────────────────
    brand        = str(row.get("brand",""))
    product_name = str(row.get("product_name",""))
    category     = str(row.get("category",""))
    rating       = _safe_float(row.get("avg_rating",   0))
    review_count = _safe_int(  row.get("review_count", 0))

    if retailer == "ulta":
        if src_disc > 0.20:
            verif_line = (f"<span style='font-size:11px;color:#880e4f;font-weight:bold;'>"
                          f"⚠️ Seeded ({src_disc:.0%} disclosure)</span>")
        elif src_vb > 0.20:
            verif_line = (f"<span style='font-size:11px;color:#880e4f;font-weight:bold;'>"
                          f"✓ Verified ({src_vb:.0%} buyers)</span>")
        else:
            verif_line = ("<span style='font-size:11px;color:#880e4f;'>"
                          "○ Low verification</span>")
        legend_line = (
            "<span style='font-size:10px;color:#999;'>"
            "Bars: "
            "<span style='color:#27ae60;'>■</span> Organic "
            "<span style='color:#e67e22;'>■</span> Seeded "
            "<span style='color:#3498db;'>■</span> Low "
            "<span style='color:#2d6a4f;'>■</span> Sephora &nbsp;|&nbsp; "
            "<span style='color:red;'>---</span> Source &nbsp;|&nbsp; "
            "Labels: [S]=Sephora [U]=Ulta</span>"
        )
        extra = f"{verif_line}<br>{legend_line}"
    else:
        extra = (
            "<span style='font-size:10px;color:#999;'>"
            "Red dashed = source product &nbsp;|&nbsp; "
            "Labels: [S]=Sephora [U]=Ulta &nbsp;|&nbsp; "
            "Hover bars for full details</span>"
        )

    kpi = (
        f"{_retailer_badge(retailer)} "
        f"<b style='font-size:16px'>{brand} — {product_name}</b><br>"
        f"<span style='font-size:11px;color:#555;'>"
        f"Category: {category} &nbsp;|&nbsp; "
        f"Price: ${src_price:.0f} &nbsp;|&nbsp; "
        f"Rating: {rating:.2f} &nbsp;|&nbsp; "
        f"Sentiment: {src_sent:.3f} &nbsp;|&nbsp; "
        f"Reviews: {review_count:,}</span><br>"
        f"{extra}"
    )

    fig.update_layout(
        height=1800, width=1350,
        title=dict(text=kpi, font=dict(size=12), x=0.01, y=0.99),
        template="plotly_white",
        showlegend=False,
        margin=dict(t=120, l=170, r=40, b=30),
    )
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# BROWSER EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_product_html(product_id: str, output_dir=None) -> Path:
    """
    Save a single product's joint recommendation dashboard as a self-contained
    HTML file (no Jupyter required).
    """
    _load_data()
    out = Path(output_dir) if output_dir else PRODUCT_DASHBOARD_DIR
    out.mkdir(parents=True, exist_ok=True)

    recs  = _data["recs"]
    match = recs[recs["product_id"].astype(str) == str(product_id)]
    if len(match) > 0:
        r        = match.iloc[0]
        retailer = _slug(str(r.get("retailer",      "")))
        brand    = _slug(str(r.get("brand",         "")))
        name     = _slug(str(r.get("product_name",  product_id))[:40])
        filename = f"{retailer}_{brand}_{name}_{_slug(str(product_id))}.html"
    else:
        filename = f"{_slug(str(product_id))}.html"

    fig  = build_product_dashboard(product_id)
    path = out / filename
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"  -> {path}")
    return path


def write_all_product_htmls(output_dir=None, limit: int = None) -> Path:
    """
    Write a dashboard HTML for every (or first `limit`) products, then build
    a linked index page at notebooks/joint/outputs/joint_product_dashboards_index.html.
    """
    _load_data()
    recs  = _data["recs"]
    out   = Path(output_dir) if output_dir else PRODUCT_DASHBOARD_DIR
    out.mkdir(parents=True, exist_ok=True)

    total = len(recs) if limit is None else min(int(limit), len(recs))
    print(f"Writing {total:,} joint dashboards to {out}/")

    written = []
    for i, (_, rec_row) in enumerate(recs.head(total).iterrows(), 1):
        pid = str(rec_row["product_id"])
        try:
            path = write_product_html(pid, output_dir=out)
            written.append((rec_row, path.name))
        except Exception as e:
            print(f"  WARNING: skipped {pid!r} — {e}")
        if i % 200 == 0:
            print(f"  {i:,}/{total:,} complete...")

    # ── Index page ─────────────────────────────────────────────────────────
    index_path = out.parent / "joint_product_dashboards_index.html"
    rows_html  = ""

    for rec_row, filename in written:
        retailer = str(rec_row.get("retailer","")).lower()
        brand    = str(rec_row.get("brand",""))
        name     = str(rec_row.get("product_name",""))
        category = str(rec_row.get("category",""))
        signal   = str(rec_row.get("cross_platform_signal","—"))

        try:
            price     = f"${_safe_float(rec_row.get('price',         0)):.0f}"
            rating    = f"{_safe_float(rec_row.get('avg_rating',     0)):.2f}"
            sentiment = f"{_safe_float(rec_row.get('avg_sentiment',  0)):.3f}"
            reviews   = f"{_safe_int(  rec_row.get('review_count',   0)):,}"
        except Exception:
            price = rating = sentiment = reviews = "—"

        ret_badge = (
            f"<span style='background:{'#880e4f' if retailer=='ulta' else '#2d6a4f'};"
            f"color:white;padding:1px 5px;border-radius:3px;font-size:11px;'>"
            f"{'ULTA' if retailer=='ulta' else 'SEPHORA'}</span>"
        )

        # Verification column (Ulta only)
        vb   = _safe_float(rec_row.get("pct_verified_buyer", 0))
        disc = _safe_float(rec_row.get("pct_has_disclosure",  0))
        if retailer == "ulta":
            if disc > 0.20:
                vlabel, vcol = f"Seeded ({disc:.0%})", "#e67e22"
            elif vb > 0.20:
                vlabel, vcol = f"Verified ({vb:.0%})", "#27ae60"
            else:
                vlabel, vcol = "Low", "#3498db"
            verif_cell = f"<td><span style='color:{vcol};font-weight:bold;'>{vlabel}</span></td>"
        else:
            verif_cell = "<td>—</td>"

        rows_html += (
            f"<tr>"
            f"<td>{ret_badge}</td>"
            f"<td><a href='product_dashboards/{filename}'>{brand}</a></td>"
            f"<td>{name}</td><td>{category}</td>"
            f"<td>{price}</td><td>{rating}</td><td>{sentiment}</td><td>{reviews}</td>"
            f"{verif_cell}"
            f"</tr>\n"
        )

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Joint Recommendations — Product Index</title>
  <style>
    body  {{ font-family: Arial, sans-serif; max-width: 1500px; margin: 40px auto;
             padding: 0 20px; color: #2C3E50; }}
    h1    {{ color: #2C3E50; border-bottom: 3px solid #2C3E50; padding-bottom: 10px; }}
    p.sub {{ color: #7f8c8d; margin-top: -8px; font-size:13px; }}
    p.leg {{ font-size: 12px; }}
    input {{ width:100%; padding:8px; margin-bottom:16px; box-sizing:border-box;
             border:1px solid #ccc; border-radius:4px; font-size:14px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th    {{ background:#2C3E50; color:white; padding:10px; text-align:left;
             position:sticky; top:0; }}
    td    {{ padding:7px 10px; border-bottom:1px solid #ecf0f1; }}
    tr:hover td {{ background:#f5f5f5; }}
    a     {{ color:#2C3E50; text-decoration:none; font-weight:bold; }}
    a:hover {{ text-decoration:underline; }}
  </style>
</head>
<body>
  <h1>Joint Recommendations — Product Index</h1>
  <p class="sub">
    {len(written):,} products (Sephora + Ulta) &nbsp;|&nbsp;
    Click any brand to open its cross-catalog recommendation dashboard
  </p>
  <p class="leg">
    <span style="background:#2d6a4f;color:white;padding:1px 5px;border-radius:3px;">SEPHORA</span>
    &nbsp;
    <span style="background:#880e4f;color:white;padding:1px 5px;border-radius:3px;">ULTA</span>
    &nbsp;&nbsp; Ulta verification:
    <span style="color:#27ae60;font-weight:bold;">■ Organic</span> &nbsp;
    <span style="color:#e67e22;font-weight:bold;">■ Seeded</span> &nbsp;
    <span style="color:#3498db;font-weight:bold;">■ Low</span>
  </p>
  <input type="text" id="search"
         placeholder="Filter by retailer, brand, product, or category..."
         onkeyup="filterTable()">
  <table id="productTable">
    <thead>
      <tr>
        <th>Retailer</th><th>Brand</th><th>Product</th><th>Category</th>
        <th>Price</th><th>Rating</th><th>Sentiment</th><th>Reviews</th>
        <th>Verification</th>
      </tr>
    </thead>
    <tbody>
{rows_html}    </tbody>
  </table>
  <script>
    function filterTable() {{
      const q = document.getElementById("search").value.toLowerCase();
      document.querySelectorAll("#productTable tbody tr").forEach(row => {{
        const text = Array.from(row.cells).slice(0,4).map(c=>c.textContent).join(" ").toLowerCase();
        row.style.display = text.includes(q) ? "" : "none";
      }});
    }}
  </script>
</body>
</html>"""

    index_path.write_text(index_html, encoding="utf-8")
    print(f"\nIndex page: {index_path}")
    print(f"Done. {len(written):,}/{total:,} dashboards saved.")
    return index_path


# ══════════════════════════════════════════════════════════════════════════════
# JUPYTER DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def show_dashboard():
    """
    Interactive Jupyter dropdown across the full joint catalog (Sephora + Ulta).
    Requires ipywidgets. For browser export use write_all_product_htmls().
    """
    import ipywidgets as widgets
    from IPython.display import display, HTML

    _load_data()
    products = _data["product_list"]

    display(HTML(
        "<h2 style='text-align:center;color:#2C3E50;'>"
        "Joint Recommendations — Sephora &amp; Ulta</h2>"
        "<p style='text-align:center;color:#7f8c8d;'>"
        f"Select from {len(products):,} products. "
        "Recommendations span both catalogs; matched duplicates excluded.</p>"
    ))

    dropdown = widgets.Dropdown(
        options=products, value=products[0][1],
        description="Product:",
        style={"description_width":"initial"},
        layout=widgets.Layout(width="750px"),
    )
    output = widgets.Output()

    def on_change(change):
        with output:
            output.clear_output(wait=True)
            build_product_dashboard(change["new"]).show()

    dropdown.observe(on_change, names="value")
    display(dropdown, output)
    with output:
        build_product_dashboard(products[0][1]).show()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Joint Recommendation Engine")
    parser.add_argument("cmd", nargs="?", default="help",
                        choices=["help","build_csv","write_htmls"])
    parser.add_argument("--limit", type=int, default=200,
                        help="Max products to export as HTML (default: 200)")
    args = parser.parse_args()

    if args.cmd == "build_csv":
        run_pipeline()
    elif args.cmd == "write_htmls":
        write_all_product_htmls(limit=args.limit)
    else:
        print("Commands:")
        print("  python src/joint_recommendations.py build_csv")
        print("  python src/joint_recommendations.py write_htmls --limit 200")
        print("\nNotebook:")
        print("  from src.joint_recommendations import show_dashboard")
        print("  show_dashboard()")