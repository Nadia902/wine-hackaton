import gc
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager
from asyncio import Lock

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .config import (
    PROJECT_ROOT, DATASET_DIR, CATALOG_IMAGES_DIR, DB_CONFIG, DB_SCHEMA,
    DATA_DIR, MODELS_DIR, HOST, PORT,
    CLIP_MODEL_NAME, CLIP_PRETRAINED,
)
from .catalog import WineCatalog
from .matcher import WineMatcher
from .matcher_ocr import WineOCR
from .preprocess import decode_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

catalog: Optional[WineCatalog] = None
matcher: Optional[WineMatcher] = None
_match_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global catalog, matcher

    logger.info("Loading wine catalog from database...")
    catalog = WineCatalog(CATALOG_IMAGES_DIR, DB_CONFIG, schema=DB_SCHEMA)
    catalog.load()
    logger.info(f"Loaded {len(catalog)} wines")

    logger.info("Initializing CLIP matcher with pgvector...")
    matcher = WineMatcher(
        model_name=CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
        db_config=DB_CONFIG,
        db_schema=DB_SCHEMA,
    )
    matcher.catalog = catalog
    matcher.series_groups = catalog.series_groups

    # Load ORB index for visual re-ranking
    orb_index_path = DATA_DIR / "sift_index" / "sift_index.json"
    logger.info("Loading ORB index for visual re-ranking...")
    matcher.load_orb_index(orb_index_path)

    # Initialize OCR with metadata texts (fast) and optional image OCR texts
    ocr = WineOCR()

    # Build catalog metadata texts from DB (fast, no OCR needed)
    metadata_path = DATA_DIR / "catalog_metadata.json"
    if not ocr.load_metadata(metadata_path):
        logger.info("Building catalog metadata texts from database...")
        ocr.build_catalog_metadata(catalog)
        ocr.save_metadata(metadata_path)
    else:
        logger.info(f"Loaded catalog metadata texts: {len(ocr.catalog_metadata)} wines")

    # Load precomputed image OCR texts if available
    ocr_texts_path = DATA_DIR / "ocr_texts.json"
    if ocr.load_texts(ocr_texts_path):
        logger.info("Loaded precomputed OCR texts")
    else:
        logger.info("No precomputed OCR texts")

    # Pre-initialize EasyOCR reader at startup (avoids ~15s delay on first scan)
    logger.info("Pre-initializing EasyOCR reader...")
    ocr._init_reader()
    logger.info("EasyOCR reader ready")

    matcher.ocr = ocr

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Своё Вино — Wine Scanner API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── API Response Models ───

class WineCard(BaseModel):
    slug: str
    name: str
    producer: str
    region: str
    grape: str
    year: Optional[int] = None
    category: str = ""
    rating: Optional[float] = None
    description: str = ""
    food_pairing: str = ""
    alcohol: Optional[float] = None
    price: Optional[str] = None
    image_url: str = ""
    rating_roskachestvo: Optional[float] = None
    rating_sommelier: Optional[float] = None
    rating_people: Optional[float] = None


class ScanResult(BaseModel):
    slug: Optional[str] = None
    confidence: float = 0.0
    sift_score: float = 0.0
    ocr_score: float = 0.0
    inlier_count: int = 0
    in_catalog: bool = False
    wine: Optional[WineCard] = None
    alternatives: list[WineCard] = []
    match_time_ms: int = 0
    f1_top1: Optional[float] = None
    f1_top5: Optional[float] = None


class CatalogEntry(BaseModel):
    slug: str
    name: str
    producer: str
    region: str
    grape: str
    year: Optional[int] = None
    category: str = ""
    rating: Optional[float] = None
    image_url: str = ""


# ─── Helpers ───

def wine_to_card(wine) -> WineCard:
    image_url = ""
    if wine.image_path:
        image_url = f"/images/{Path(wine.image_path).name}"
    return WineCard(
        slug=wine.slug,
        name=wine.name,
        producer=wine.producer,
        region=wine.region,
        grape=wine.grape,
        year=wine.year,
        category=wine.category,
        rating=wine.rating,
        description=wine.description,
        food_pairing=wine.food_pairing,
        alcohol=wine.alcohol,
        price=wine.price,
        image_url=image_url,
        rating_roskachestvo=wine.rating_roskachestvo,
        rating_sommelier=wine.rating_sommelier,
        rating_people=wine.rating_people,
    )


# ─── Endpoints ───

@app.post("/api/scan", response_model=ScanResult)
async def scan_wine(image: UploadFile = File(...)):
    """
    Scan a wine label image and return the matching wine card.
    This is the primary endpoint used by the evaluation script.
    Returns flat JSON: {"slug": "wine-slug"} in the response.
    """
    start = time.time()

    async with _match_lock:
        contents = await image.read()
        query_img = decode_image(contents)

        result = matcher.match(query_img, top_k=5)
        gc.collect()

    # Get wine card
    wine_card = None
    if result["slug"]:
        wine = catalog.get(result["slug"])
        if wine:
            wine_card = wine_to_card(wine)

    # Get alternatives (top-2 through top-5 that are different wines)
    alternatives = []
    seen_slugs = {result["slug"]}
    for item in result["top_k"][1:5]:
        if item["slug"] not in seen_slugs:
            wine = catalog.get(item["slug"])
            if wine:
                alternatives.append(wine_to_card(wine))
                seen_slugs.add(item["slug"])

    # Compute approximate F1 metrics
    f1_top1 = result["confidence"] if result["in_catalog"] else 0.0
    f1_top5 = 1.0 if result["in_catalog"] else 0.0

    total_time = round((time.time() - start) * 1000)

    return ScanResult(
        slug=result["slug"],
        confidence=result["confidence"],
        sift_score=result.get("sift_score", 0.0),
        ocr_score=result.get("ocr_score", 0.0),
        inlier_count=result.get("inlier_count", 0),
        in_catalog=result["in_catalog"],
        wine=wine_card,
        alternatives=alternatives,
        match_time_ms=total_time,
        f1_top1=f1_top1,
        f1_top5=f1_top5,
    )


@app.post("/api/scan/flat")
async def scan_wine_flat(image: UploadFile = File(...)):
    """
    Flat endpoint for the evaluation bash script.
    Returns: {"slug": "wine-slug"} or {"slug": ""} if not found.
    """
    async with _match_lock:
        contents = await image.read()
        query_img = decode_image(contents)
        result = matcher.match(query_img, top_k=1)
        gc.collect()

    return {"slug": result["slug"] or ""}


@app.post("/v1/eval/predict")
async def eval_predict(image: UploadFile = File(...)):
    """Evaluation endpoint for participant_test.sh."""
    async with _match_lock:
        contents = await image.read()
        query_img = decode_image(contents)
        result = matcher.match(query_img, top_k=1)
        gc.collect()

    return {"slug": result["slug"] or ""}


@app.get("/api/wine/{slug}")
async def get_wine(slug: str):
    """Get wine card by slug."""
    wine = catalog.get(slug)
    if not wine:
        raise HTTPException(status_code=404, detail=f"Wine '{slug}' not found")
    return wine_to_card(wine)


@app.get("/api/catalog", response_model=list[CatalogEntry])
async def list_catalog(
    q: Optional[str] = Query(None, description="Search query"),
    limit: int = Query(50, ge=1, le=500),
):
    """List or search wines in the catalog."""
    if q:
        wines = catalog.search_text(q, limit=limit)
    else:
        wines = list(catalog.wines.values())[:limit]

    return [
        CatalogEntry(
            slug=w.slug,
            name=w.name,
            producer=w.producer,
            region=w.region,
            grape=w.grape,
            year=w.year,
            category=w.category,
            rating=w.rating,
            image_url=f"/images/{Path(w.image_path).name}" if w.image_path else "",
        )
        for w in wines
    ]


@app.get("/api/stats")
async def get_stats():
    """Return catalog statistics."""
    wines_with_embedding = sum(1 for w in catalog.wines.values() if w.embedding)
    return {
        "total_wines": len(catalog),
        "wines_with_embedding": wines_with_embedding,
        "series_count": len(catalog.series_groups),
        "matcher": "pgvector + CLIP + ORB + OCR",
    }


# ─── Static Files ───

# Serve wine images
app.mount("/images", StaticFiles(directory=str(CATALOG_IMAGES_DIR)), name="catalog_images")

# Serve frontend
FRONTEND_DIR = PROJECT_ROOT / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/wine/{slug}")
async def serve_wine_page(slug: str):
    return FileResponse(str(FRONTEND_DIR / "wine.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
