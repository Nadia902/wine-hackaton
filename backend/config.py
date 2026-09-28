import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Dataset — images stored locally
DATASET_DIR = PROJECT_ROOT / "opt_dataset"
CATALOG_IMAGES_DIR = DATASET_DIR / "clean_uploads"

# PostgreSQL database
DB_CONFIG = {
    "dbname": os.environ.get("DB_NAME", "wine_db"),
    "user": os.environ.get("DB_USER", "wine_user"),
    "password": os.environ.get("DB_PASSWORD", "wine_secret_2026"),
    "host": os.environ.get("DB_HOST", "127.0.0.1"),
    "port": os.environ.get("DB_PORT", "5432"),
}
DB_SCHEMA = os.environ.get("DB_SCHEMA", "just_vine_it")

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Vision model config — SigLIP for compatibility with pgvector embeddings in DB.
# DB embeddings were computed with google/siglip-base-patch16-224 (768-dim).
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL", "ViT-B-16-SigLIP")
CLIP_PRETRAINED = os.environ.get("CLIP_PRETRAINED", "webli")
CLIP_EMBEDDING_DIM = 768  # SigLIP base = 768

# FAISS index
FAISS_INDEX_FILE = DATA_DIR / "wine_index.faiss"
FAISS_MAPPINGS_FILE = DATA_DIR / "wine_mappings.json"

# Image preprocessing
IMAGE_SIZE = (224, 224)  # ViT-B-32 input size

# Matching thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.30  # Cosine similarity above this = confident match
LOW_CONFIDENCE_THRESHOLD = 0.15   # Below this = wine not found

# Server
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))

# Fine-tuned model
FINETUNED_MODEL_PATH = MODELS_DIR / "siglip_finetuned" / "model_state.pt"
