"""
OCR-based wine label text extraction and fuzzy matching.
Phase 2: Promoted to primary matching signal with fuzzy text matching.
"""
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

import easyocr
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ─── Text normalization ────────────────────────────────────────────

# Cyrillic -> Latin transliteration table
_CYR_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def transliterate_cyrillic(text: str) -> str:
    """Transliterate Cyrillic characters to Latin."""
    result = []
    for ch in text.lower():
        result.append(_CYR_TO_LAT.get(ch, ch))
    return "".join(result)


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching: lowercase, strip, normalize unicode."""
    if not text:
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_and_transliterate(text: str) -> str:
    """Normalize text AND produce a transliterated Latin version for cross-script matching."""
    norm = normalize_text(text)
    # If text contains Cyrillic, also produce transliterated version
    has_cyrillic = any("а" <= ch <= "я" or "а" <= ch <= "я" for ch in norm)
    if has_cyrillic:
        return transliterate_cyrillic(norm)
    return norm


def extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from normalized text, removing stop words."""
    stop_words = {
        "и", "в", "на", "с", "по", "для", "от", "из", "к", "о", "у", "за",
        "не", "что", "это", "как", "да", "нет", "но", "а", "то", "все",
        "the", "and", "of", "for", "in", "with", "by", "de", "la", "le",
        "no", "al", "el", "du", "au", "is", "or", "it",
    }
    words = text.split()
    return {w for w in words if len(w) > 1 and w not in stop_words}


# ─── Fuzzy matching ───────────────────────────────────────────────

def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity (0.0-1.0)."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # DP matrix
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,      # deletion
                matrix[i][j - 1] + 1,      # insertion
                matrix[i - 1][j - 1] + cost  # substitution
            )

    distance = matrix[len1][len2]
    max_len = max(len1, len2)
    return 1.0 - (distance / max_len)


def _token_sort_ratio(text1: str, text2: str) -> float:
    """Sort tokens alphabetically, then compute similarity. Handles word reordering."""
    sorted1 = " ".join(sorted(text1.split()))
    sorted2 = " ".join(sorted(text2.split()))
    return _levenshtein_ratio(sorted1, sorted2)


def _substring_match_score(query_keywords: set[str], candidate_text: str) -> float:
    """Score based on how many query keywords appear as substrings in candidate."""
    if not query_keywords or not candidate_text:
        return 0.0
    matches = sum(1 for kw in query_keywords if kw in candidate_text)
    return matches / len(query_keywords) if query_keywords else 0.0


def fuzzy_match_score(query_text: str, candidate_text: str) -> float:
    """
    Compute fuzzy match score between query OCR text and candidate text.
    Handles both Cyrillic and Latin scripts via transliteration.
    Returns 0.0-1.0.
    """
    q_norm = normalize_text(query_text)
    c_norm = normalize_text(candidate_text)

    if not q_norm or not c_norm:
        return 0.0

    # Try direct match first
    best = _fuzzy_score_pair(q_norm, c_norm)

    # Try cross-script: transliterate query to Latin, compare with candidate
    q_translit = transliterate_cyrillic(q_norm)
    if q_translit != q_norm:
        best = max(best, _fuzzy_score_pair(q_translit, c_norm))

    # Try cross-script: transliterate candidate to Latin, compare with query
    c_translit = transliterate_cyrillic(c_norm)
    if c_translit != c_norm:
        best = max(best, _fuzzy_score_pair(q_norm, c_translit))
        # Also compare both transliterated
        if q_translit != q_norm:
            best = max(best, _fuzzy_score_pair(q_translit, c_translit))

    return best


def _fuzzy_score_pair(q_norm: str, c_norm: str) -> float:
    """Compute fuzzy score between two normalized texts (same script)."""
    if not q_norm or not c_norm:
        return 0.0

    if q_norm == c_norm:
        return 1.0

    # Strategy 1: Token sort ratio (handles word reordering)
    tsr = _token_sort_ratio(q_norm, c_norm)

    # Strategy 2: Substring keyword matching
    q_keywords = extract_keywords(q_norm)
    sub_score = _substring_match_score(q_keywords, c_norm)

    # Strategy 3: Word-level matching
    q_words = set(q_norm.split())
    c_words = set(c_norm.split())
    if q_words and c_words:
        intersection = q_words & c_words
        word_score = len(intersection) / max(len(q_words), len(c_words))
    else:
        word_score = 0.0

    # Strategy 4: Best single-word fuzzy match
    best_word_match = 0.0
    for qw in q_words:
        if len(qw) < 2:
            continue
        for cw in c_words:
            if len(cw) < 2:
                continue
            ratio = _levenshtein_ratio(qw, cw)
            if ratio > best_word_match:
                best_word_match = ratio

    # Combine
    score = (
        0.35 * tsr +
        0.30 * sub_score +
        0.20 * word_score +
        0.15 * best_word_match
    )

    return min(score, 1.0)


# ─── WineOCR class ────────────────────────────────────────────────

class WineOCR:
    """Extract and compare text from wine labels using EasyOCR + fuzzy matching."""

    def __init__(self, languages: list[str] = None):
        self.languages = languages or ["ru", "en"]
        self.reader = None
        self.catalog_texts: dict[str, str] = {}  # slug -> precomputed OCR text from images
        self.catalog_metadata: dict[str, str] = {}  # slug -> metadata text (name+winery+grape)

    def _init_reader(self):
        """Lazy-initialize EasyOCR reader (heavy, ~1.5GB RAM)."""
        if self.reader is None:
            logger.info(f"Initializing EasyOCR reader for {self.languages}...")
            self.reader = easyocr.Reader(self.languages, gpu=False)
            logger.info("EasyOCR reader initialized")

    def extract_text(self, image: Image.Image) -> str:
        """Extract text from a wine label image."""
        self._init_reader()
        img_array = np.array(image.convert("RGB"))
        try:
            results = self.reader.readtext(img_array, detail=0)
            text = " ".join(results).lower()
            text = "".join(c for c in text if c.isalnum() or c.isspace())
            return text
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return ""

    def extract_text_from_label(self, label_image: Image.Image) -> str:
        """Extract text specifically from an already-cropped label region."""
        # The label is already extracted — just run OCR directly
        return self.extract_text(label_image)

    # ─── Metadata text ─────────────────────────────────────────────

    def build_catalog_metadata(self, catalog):
        """
        Build metadata text for all wines in catalog from DB fields.
        Includes both Cyrillic original AND transliterated Latin version.
        """
        self.catalog_metadata = {}
        for slug, wine in catalog.wines.items():
            parts = [
                wine.name or "",
                wine.producer or "",
                wine.grape or "",
                wine.category or "",
                wine.slug or "",
            ]
            text = " ".join(parts)
            text_normalized = normalize_text(text)
            # Also include transliterated version for cross-script matching
            text_translit = transliterate_cyrillic(text_normalized)
            # Store as "normalized|translit" for dual comparison
            self.catalog_metadata[slug] = f"{text_normalized} {text_translit}"
        logger.info(f"Built metadata text for {len(self.catalog_metadata)} wines")

    def save_metadata(self, path: Path):
        """Save catalog metadata texts to disk."""
        with open(path, "w") as f:
            json.dump(self.catalog_metadata, f, ensure_ascii=False)
        logger.info(f"Saved catalog metadata texts to {path}")

    def load_metadata(self, path: Path) -> bool:
        """Load catalog metadata texts from disk."""
        if not path.exists():
            return False
        try:
            with open(path) as f:
                self.catalog_metadata = json.load(f)
            logger.info(f"Loaded catalog metadata texts: {len(self.catalog_metadata)} wines")
            return True
        except Exception as e:
            logger.warning(f"Failed to load metadata texts: {e}")
            return False

    # ─── Catalog image OCR precomputation ──────────────────────────

    def precompute_catalog_texts(self, catalog_images: dict[str, Image.Image]):
        """Precompute OCR text for all catalog images at startup."""
        self._init_reader()
        total = len(catalog_images)
        logger.info(f"Precomputing OCR text for {total} catalog images...")

        for i, (slug, image) in enumerate(catalog_images.items()):
            try:
                text = self.extract_text(image)
                self.catalog_texts[slug] = text
            except Exception as e:
                logger.warning(f"OCR failed for {slug}: {e}")
                self.catalog_texts[slug] = ""

            if (i + 1) % 100 == 0:
                logger.info(f"  OCR progress: {i + 1}/{total}")

        logger.info(f"Precomputed OCR for {len(self.catalog_texts)} wines")

    def save_texts(self, path: Path):
        """Save precomputed OCR texts to disk."""
        with open(path, "w") as f:
            json.dump(self.catalog_texts, f, ensure_ascii=False)
        logger.info(f"Saved OCR texts to {path}")

    def load_texts(self, path: Path) -> bool:
        """Load precomputed OCR texts from disk."""
        if not path.exists():
            return False
        try:
            with open(path) as f:
                self.catalog_texts = json.load(f)
            logger.info(f"Loaded OCR texts: {len(self.catalog_texts)} wines")
            return True
        except Exception as e:
            logger.warning(f"Failed to load OCR texts: {e}")
            return False

    # ─── Scoring ───────────────────────────────────────────────────

    def compute_ocr_scores_robust(
        self,
        query_image: Image.Image,
        candidate_slugs: list[str],
        catalog=None,
    ) -> dict[str, float]:
        """
        Compute fuzzy OCR scores for multiple candidates.
        Uses both catalog image OCR texts AND metadata texts.
        Returns {slug: score} where score is 0.0-1.0.
        """
        query_text = self.extract_text(query_image)
        if not query_text:
            return {s: 0.0 for s in candidate_slugs}

        results = {}
        for slug in candidate_slugs:
            # Try image OCR text first
            catalog_text = self.catalog_texts.get(slug, "")

            # Fallback to metadata text
            if not catalog_text:
                catalog_text = self.catalog_metadata.get(slug, "")

            # Last resort: build from catalog wine object
            if not catalog_text and catalog:
                wine = catalog.get(slug)
                if wine:
                    parts = [wine.name or "", wine.grape or "", wine.winery or ""]
                    catalog_text = normalize_text(" ".join(parts))

            results[slug] = fuzzy_match_score(query_text, catalog_text)

        return results

    def get_query_text(self, query_image: Image.Image) -> str:
        """Extract and return normalized text from query image."""
        raw = self.extract_text(query_image)
        return normalize_text(raw)

    # ─── Backward-compatible methods ───────────────────────────────

    def compute_ocr_score(self, query_image: Image.Image, candidate_slug: str) -> float:
        """Compute OCR similarity between query and a catalog candidate (backward compat)."""
        query_text = self.extract_text(query_image)
        catalog_text = self.catalog_texts.get(candidate_slug, "")
        if not catalog_text:
            catalog_text = self.catalog_metadata.get(candidate_slug, "")
        if not query_text or not catalog_text:
            return 0.0
        return fuzzy_match_score(query_text, catalog_text)

    def compute_ocr_scores_batch(
        self,
        query_image: Image.Image,
        candidate_slugs: list[str],
        catalog=None,
    ) -> dict[str, float]:
        """Backward-compatible batch scoring."""
        return self.compute_ocr_scores_robust(query_image, candidate_slugs, catalog)

    def text_similarity(self, text1: str, text2: str) -> float:
        """Backward-compatible text similarity."""
        return fuzzy_match_score(text1, text2)
