# Architecture — Своё Вино (Wine Label Scanner)

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker (docker-compose)               │
│                                                         │
│  ┌──────────────┐         ┌──────────────────────┐      │
│  │  wine-app    │────────▶│  wine-db             │      │
│  │  :8000       │         │  PostgreSQL 16       │      │
│  │              │         │  + pgvector          │      │
│  │  FastAPI     │         │  :5432               │      │
│  │  + SigLIP    │         │                      │      │
│  │  + ORB       │         │  HNSW index on       │      │
│  │  + EasyOCR   │         │  vector(768)         │      │
│  └──────┬───────┘         └──────────────────────┘      │
│         │                                               │
│  ┌──────▼───────┐                                       │
│  │  Frontend    │  Vanilla HTML/CSS/JS                   │
│  │  (static)    │  mobile-first                         │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
```

## Recognition Pipeline

```
Photo → Preprocess → SigLIP TTA (6 augmentations)
                          │
                          ▼
                    pgvector HNSW
                    (top-30 candidates, ~20ms)
                          │
                          ▼
                    ORB Re-ranking
                    (top-20, 128×128 thumbnails)
                          │
                          ▼
                    OCR Text Match
                    (EasyOCR ru+en, optional)
                          │
                          ▼
                    Fusion Score
                    CLIP(55%) + ORB(25%) + OCR(20%)
                          │
                          ▼
                    Wine Card + Alternatives
```

## Modules

| Module | Purpose |
|--------|---------|
| `backend/app.py` | FastAPI server, endpoints, static file serving |
| `backend/config.py` | Configuration (DB, model paths, thresholds) |
| `backend/catalog.py` | WineCatalog — PostgreSQL data loading, search |
| `backend/matcher.py` | SigLIP + pgvector + ORB 3-stage matcher |
| `backend/matcher_ocr.py` | EasyOCR + fuzzy text matching |
| `backend/preprocess.py` | Image preprocessing (label extraction, glare removal) |
| `backend/fine_tune.py` | SigLIP fine-tuning on labeled pairs |
| `frontend/` | Mobile-first HTML/CSS/JS scanner UI |

## Data Flow

1. **User uploads photo** → `POST /api/scan`
2. **Preprocessing**: EXIF transpose, letterbox resize (336×336), lighting normalization, label extraction, glare removal
3. **CLIP encoding**: 6 TTA augmentations → 768-dim embeddings
4. **Vector search**: pgvector HNSW nearest-neighbor query → top-30 candidates
5. **ORB re-ranking**: Visual feature matching on 128×128 thumbnails → top-20
6. **OCR matching** (optional): EasyOCR text extraction + fuzzy comparison
7. **Fusion**: Weighted score combination with adaptive thresholds
8. **Response**: Wine card + up to 4 alternatives

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/scan` | Full scan (card + alternatives) |
| `POST` | `/api/scan/flat` | Flat scan → `{"slug": "..."}` |
| `POST` | `/v1/eval/predict` | Evaluation endpoint |
| `GET` | `/api/wine/{slug}` | Wine card by slug |
| `GET` | `/api/catalog?q=...` | Search/catalog list |
| `GET` | `/api/stats` | Catalog statistics |

## Database Schema

**Table:** `just_vine_it.wines`

| Column | Type | Description |
|--------|------|-------------|
| `id` | serial | Primary key |
| `name` | text | Wine name |
| `slug` | text | URL identifier (unique) |
| `category` | text | Type (красное, белое, розовое...) |
| `color` | text | Color |
| `region` | text | Region |
| `grape_variety` | text | Grape variety |
| `description` | text | Description |
| `winery` | text | Producer |
| `image_filename` | text | Image filename |
| `image_path` | text | Image path |
| `embedding` | vector(768) | SigLIP embedding |
| `rating_roskachestvo` | float | Roskachestvo rating (3.9–5.0) |
| `rating_sommelier` | float | Sommelier rating (60–100) |
| `rating_people` | float | People rating (4.0–5.0) |

**Index:** HNSW on `embedding` (vector_cosine_ops)

## Tech Stack

- **Backend**: Python 3.12, FastAPI, Uvicorn
- **ML**: PyTorch (CPU), open-clip-torch, SigLIP ViT-B-16
- **OCR**: EasyOCR (ru+en)
- **Database**: PostgreSQL 16 + pgvector (HNSW index)
- **Frontend**: Vanilla HTML/CSS/JS (no framework)
- **Deploy**: Docker Compose
