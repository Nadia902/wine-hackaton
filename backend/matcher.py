"""
Three-stage wine label matcher with pgvector:
  Stage 1: pgvector HNSW search — find nearest neighbors by CLIP embedding
  Stage 2: ORB visual re-ranking — refine with local features
  Stage 3: OCR text matching — fuzzy text comparison for real-world photos
"""
import gc
import json
import time
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import open_clip
import psycopg2
from PIL import Image

from .config import DB_CONFIG, DB_SCHEMA, FINETUNED_MODEL_PATH
from .preprocess import (
    preprocess_image, extract_label_robust, remove_glare,
    suppress_background, normalize_for_ocr,
)
from .catalog import WineCatalog

logger = logging.getLogger(__name__)


# ─── ORB helpers ───────────────────────────────────────────────────

ORB_FEATURES = 500
THUMB_SIZE = (128, 128)


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def image_to_thumb(cv2_img: np.ndarray, size=THUMB_SIZE) -> np.ndarray:
    return cv2.resize(cv2_img, size, interpolation=cv2.INTER_AREA)


def thumb_hash(thumb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY) if len(thumb.shape) == 3 else thumb
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    mean_val = small.mean()
    return (small > mean_val).flatten().astype(np.uint8)


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    return int(np.sum(h1 != h2))


def orb_score_pair(q_desc, q_thumb, q_hash,
                   c_desc, c_thumb, c_hash,
                   orb_matcher) -> float:
    """Compute ORB+pixel+hash score between query and catalog entry."""
    if q_hash is not None and c_hash is not None:
        hash_dist = hamming_distance(q_hash, c_hash)
        hash_score = 1.0 - hash_dist / 256.0
    else:
        hash_score = 0.0

    orb_s = 0.0
    if q_desc is not None and c_desc is not None and len(q_desc) >= 2 and len(c_desc) >= 2:
        try:
            k = min(2, len(c_desc))
            matches = orb_matcher.knnMatch(q_desc, c_desc, k=k)
            good = [m for mg in matches if len(mg) == 2
                    for m, n in [mg] if m.distance < 0.75 * n.distance]
            max_kp = max(len(q_desc), len(c_desc))
            orb_s = len(good) / max_kp if max_kp > 0 else 0.0
        except Exception:
            pass

    pixel_s = 0.0
    if q_thumb is not None and c_thumb is not None:
        diff = cv2.absdiff(q_thumb, c_thumb)
        pixel_s = 1.0 - (diff.mean() / 255.0)

    return 0.60 * orb_s + 0.25 * pixel_s + 0.15 * hash_score


# ─── TTA augmentations (for raw images — no preprocessing) ──────────

def _augment_query(image: Image.Image) -> list[Image.Image]:
    """Generate augmented variants of the raw query for CLIP matching.
    No preprocessing — DB embeddings are from raw photos."""
    from PIL import ImageEnhance

    variants = [image]

    # Center crop
    w, h = image.size
    crop_pct = 0.12
    cropped = image.crop((int(w * crop_pct), int(h * crop_pct),
                          int(w * (1 - crop_pct)), int(h * (1 - crop_pct))))
    variants.append(cropped)

    # Slight rotation
    rotated = image.rotate(3, expand=True, fillcolor=(180, 180, 180))
    variants.append(rotated)

    # Brightness variations
    bright = ImageEnhance.Brightness(image).enhance(1.2)
    variants.append(bright)
    dark = ImageEnhance.Brightness(image).enhance(0.8)
    variants.append(dark)

    # Contrast boost
    high_contrast = ImageEnhance.Contrast(image).enhance(1.25)
    variants.append(high_contrast)

    return variants


# ─── Main matcher ──────────────────────────────────────────────────

class WineMatcher:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
        db_config: Optional[dict] = None,
        db_schema: str = "just_vine_it",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CLIP model {model_name} on {self.device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()

        # Load fine-tuned weights if available
        if FINETUNED_MODEL_PATH.exists():
            logger.info(f"Loading fine-tuned model from {FINETUNED_MODEL_PATH}")
            state = torch.load(FINETUNED_MODEL_PATH, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
            logger.info("Fine-tuned model loaded successfully")

        self.catalog: Optional[WineCatalog] = None
        self.series_groups: dict[str, list[str]] = {}
        self.ocr = None  # WineOCR instance

        # pgvector DB connection
        self.db_config = db_config or DB_CONFIG
        self.db_schema = db_schema or DB_SCHEMA

        # Adaptive threshold tracking
        self._recent_confidences: list[float] = []
        self._max_recent = 20

        # ORB index data
        self.orb_data: dict[str, dict] = {}
        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        self.orb_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # ─── CLIP encoding ─────────────────────────────────────────────

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        processed = self.preprocess(image).unsqueeze(0).to(self.device)
        embedding = self.model.encode_image(processed)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().squeeze()

    @torch.no_grad()
    def encode_with_tta(self, image: Image.Image) -> np.ndarray:
        """Encode raw query with TTA — no preprocessing, DB embeddings are raw."""
        variants = _augment_query(image)
        all_embs = []
        for v in variants:
            processed = self.preprocess(v).unsqueeze(0).to(self.device)
            emb = self.model.encode_image(processed)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu().numpy().squeeze())
        avg_emb = np.mean(all_embs, axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        return avg_emb

    @torch.no_grad()
    def encode_label_crop(self, label_image: Image.Image) -> np.ndarray:
        """Encode just the extracted label region — reduces domain gap."""
        processed = self.preprocess(label_image).unsqueeze(0).to(self.device)
        emb = self.model.encode_image(processed)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().squeeze()

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            processed = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
            embeddings = self.model.encode_image(processed)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

    # ─── pgvector search ────────────────────────────────────────────

    def pgvector_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 30,
    ) -> list[tuple[str, float]]:
        """Search wines by CLIP embedding similarity using pgvector HNSW index.

        Returns: [(slug, similarity_score), ...] sorted by descending similarity.
        """
        # Convert numpy array to pgvector string format: "[0.1,0.2,...]"
        vec_str = "[" + ",".join(f"{x:.8f}" for x in query_embedding.tolist()) + "]"

        sql = f"""
            SELECT slug, 1 - (embedding <=> %s::vector) AS similarity
            FROM {self.db_schema}.wines
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        conn = psycopg2.connect(**self.db_config)
        try:
            cur = conn.cursor()
            cur.execute(sql, (vec_str, vec_str, top_k))
            results = [(row[0].strip(), float(row[1])) for row in cur.fetchall()]
            return results
        except Exception as e:
            logger.error(f"pgvector search failed: {e}")
            return []
        finally:
            conn.close()

    # ─── Index (kept as fallback) ──────────────────────────────────

    def build_index(self, catalog: WineCatalog):
        self.catalog = catalog
        self.series_groups = catalog.series_groups
        self.slugs = []
        images = []
        for slug, wine in catalog.wines.items():
            if not wine.image_path:
                continue
            try:
                img = Image.open(wine.image_path)
                try:
                    label_img, score, _ = extract_label_robust(img)
                    if score > 0.3:
                        img = preprocess_image(label_img)
                    else:
                        img = preprocess_image(img)
                except Exception:
                    img = preprocess_image(img)
                images.append(img)
                self.slugs.append(slug)
            except Exception as e:
                logger.warning(f"Failed to load image for {slug}: {e}")
        logger.info(f"Encoding {len(images)} wine images...")
        start = time.time()
        self.embeddings = self.encode_images(images)
        logger.info(f"Encoded {len(images)} images in {time.time() - start:.1f}s")

    def save_index(self, index_dir: Path):
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self.embeddings)
        with open(index_dir / "mappings.json", "w") as f:
            json.dump(self.slugs, f)

    def load_index(self, index_dir: Path):
        emb_path = index_dir / "embeddings.npy"
        map_path = index_dir / "mappings.json"
        if not emb_path.exists() or not map_path.exists():
            raise FileNotFoundError(f"Index not found in {index_dir}")
        self.embeddings = np.load(emb_path)
        with open(map_path) as f:
            self.slugs = json.load(f)
        logger.info(f"Loaded CLIP index: {len(self.slugs)} wines")

    def load_orb_index(self, sift_index_path: Path):
        """Load pre-computed ORB data from SIFT index JSON."""
        if not sift_index_path.exists():
            logger.warning(f"ORB index not found: {sift_index_path}")
            return
        with open(sift_index_path) as f:
            data = json.load(f)
        loaded = 0
        for item in data:
            slug = item["slug"]
            orb_entry = {}
            if item.get("hash") is not None:
                orb_entry["hash"] = np.array(item["hash"], dtype=np.uint8)
            if item.get("descriptors") is not None:
                orb_entry["descriptors"] = np.array(item["descriptors"], dtype=np.uint8)
            if item.get("thumb") is not None:
                thumb_arr = np.array(item["thumb"], dtype=np.uint8)
                orb_entry["thumb"] = cv2.imdecode(thumb_arr, cv2.IMREAD_COLOR)
            self.orb_data[slug] = orb_entry
            loaded += 1
        logger.info(f"Loaded ORB index: {loaded} wines")

    # ─── Match (pgvector + ORB + OCR pipeline) ─────────────────────

    def match(
        self,
        query_image: Image.Image,
        top_k: int = 5,
        clip_candidates: int = 30,
        match_threshold: float = 0.25,
        gap_threshold: float = 0.005,
        use_ocr: bool = False,
    ) -> dict:
        start = time.time()

        # ── Step 1: CLIP TTA encoding ──
        query_embedding = self.encode_with_tta(query_image)

        # ── Step 2: pgvector HNSW search (replaces numpy dot product) ──
        pgvector_results = self.pgvector_search(query_embedding, top_k=clip_candidates)

        if not pgvector_results:
            logger.warning("pgvector search returned no results")
            return {
                "slug": None,
                "confidence": 0.0,
                "ocr_score": 0.0,
                "gap": 0.0,
                "confidence_margin": 0.0,
                "in_catalog": False,
                "series_confusion": False,
                "top_k": [],
                "match_time_ms": round((time.time() - start) * 1000),
            }

        clip_results = [(0, score, slug) for slug, score in pgvector_results]

        del query_embedding
        gc.collect()

        # ── Step 3: ORB re-ranking on top candidates ──
        cv2_query = pil_to_cv2(query_image)
        q_thumb = image_to_thumb(cv2_query)
        q_hash = thumb_hash(q_thumb)
        gray_thumb = cv2.cvtColor(q_thumb, cv2.COLOR_BGR2GRAY)
        _, q_desc = self.orb.detectAndCompute(gray_thumb, None)

        # ── Step 4: Combine CLIP + ORB scores ──
        combined_results = []
        for idx, clip_score, slug in clip_results:
            orb_entry = self.orb_data.get(slug, {})
            c_desc = orb_entry.get("descriptors")
            c_thumb = orb_entry.get("thumb")
            c_hash = orb_entry.get("hash")

            visual_score = orb_score_pair(
                q_desc, q_thumb, q_hash,
                c_desc, c_thumb, c_hash,
                self.orb_matcher,
            )

            combined_results.append({
                "slug": slug,
                "clip_score": clip_score,
                "visual_score": visual_score,
                "ocr_score": 0.0,
                "score": clip_score,  # Will be updated after OCR
            })

        # ── Step 5: ORB re-ranking sort ──
        combined_results.sort(key=lambda x: x["clip_score"], reverse=True)

        # ── Step 6: OCR scoring (optional, very slow on CPU ~18s) ──
        # EasyOCR inference is too slow for real-time use on CPU.
        # Enable with use_ocr=True for accuracy-critical scenarios.
        ocr_scores = {}
        if use_ocr and self.ocr:
            try:
                query_text = self.ocr.get_query_text(query_image)
                if query_text:
                    from .matcher_ocr import fuzzy_match_score
                    top_for_ocr = combined_results[:top_k]
                    for r in top_for_ocr:
                        slug = r["slug"]
                        catalog_text = self.ocr.catalog_metadata.get(slug, "")
                        if catalog_text:
                            score = fuzzy_match_score(query_text, catalog_text)
                            r["ocr_score"] = score
                            ocr_scores[slug] = score
            except Exception as e:
                logger.warning(f"OCR scoring failed: {e}")

        # ── Step 7: Final fusion — CLIP + ORB + OCR ──
        for result in combined_results:
            slug = result["slug"]
            clip_s = result["clip_score"]
            orb_s = result["visual_score"]
            ocr_s = ocr_scores.get(slug, 0.0)
            result["ocr_score"] = ocr_s

            # Adaptive fusion weights based on confidence
            if ocr_s >= 0.70:
                # Strong OCR match — trust text
                combined = 0.40 * clip_s + 0.15 * orb_s + 0.45 * ocr_s
            elif ocr_s >= 0.40:
                # Medium OCR — balanced
                combined = 0.50 * clip_s + 0.20 * orb_s + 0.30 * ocr_s
            elif clip_s >= 0.85:
                # High CLIP confidence
                combined = 0.65 * clip_s + 0.15 * orb_s + 0.20 * ocr_s
            else:
                # Default — CLIP primary
                combined = 0.55 * clip_s + 0.25 * orb_s + 0.20 * ocr_s

            result["score"] = combined

        combined_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = combined_results[:top_k]

        best = top_results[0] if top_results else None
        best_slug = best["slug"] if best else None
        best_score = best["score"] if best else 0.0

        # Gap: top-1 vs top-2
        if len(top_results) > 1:
            gap = best_score - top_results[1]["score"]
        else:
            gap = best_score

        # Series-aware check
        series_confusion = False
        if len(top_results) > 1 and self.series_groups:
            slug1, slug2 = top_results[0]["slug"], top_results[1]["slug"]
            for series_key, members in self.series_groups.items():
                if slug1 in members and slug2 in members:
                    series_confusion = True
                    break

        # Adaptive threshold
        avg_recent = 0.0
        if self._recent_confidences:
            avg_recent = sum(self._recent_confidences) / len(self._recent_confidences)

        base_threshold = 0.65
        if avg_recent > 0.85:
            base_threshold = 0.62
        elif avg_recent > 0 and avg_recent < 0.70:
            base_threshold = 0.68

        score_threshold = 0.70 if (series_confusion and gap < 0.02) else base_threshold

        # Acceptance logic — trust CLIP when it's confident, or OCR, or visual
        in_catalog = (
            best_score >= score_threshold or
            (best.get("clip_score", 0) >= 0.78 and gap >= 0.002) or
            (best.get("clip_score", 0) >= 0.85) or
            (best.get("ocr_score", 0) >= 0.70 and gap >= 0.02) or
            (best_score >= match_threshold and gap >= gap_threshold) or
            (best and best.get("visual_score", 0) >= 0.50)
        )

        # Update rolling confidence tracker
        if in_catalog:
            self._recent_confidences.append(best_score)
            if len(self._recent_confidences) > self._max_recent:
                self._recent_confidences.pop(0)

        elapsed = time.time() - start
        logger.info(
            f"Match: slug={best_slug} combined={best_score:.4f} "
            f"clip={best.get('clip_score',0):.4f} orb={best.get('visual_score',0):.4f} "
            f"ocr={best.get('ocr_score',0):.4f} gap={gap:.4f} "
            f"series_conf={series_confusion} threshold={score_threshold:.2f} "
            f"accepted={in_catalog} time={elapsed*1000:.0f}ms"
        )

        return {
            "slug": best_slug if in_catalog else None,
            "confidence": best_score,
            "ocr_score": best.get("ocr_score", 0.0),
            "gap": float(gap),
            "confidence_margin": float(gap),
            "in_catalog": in_catalog,
            "series_confusion": series_confusion,
            "top_k": top_results if in_catalog else [],
            "match_time_ms": round(elapsed * 1000),
        }
