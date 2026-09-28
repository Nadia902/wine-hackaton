import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import psycopg2


@dataclass
class Wine:
    slug: str
    name: str
    producer: str
    region: str
    grape: str
    year: Optional[int] = None
    category: str = ""
    rating: Optional[float] = None
    description: str = ""
    image_path: str = ""
    food_pairing: str = ""
    alcohol: Optional[float] = None
    price: Optional[str] = None
    series: str = ""
    embedding: Optional[list] = None
    rating_roskachestvo: Optional[float] = None
    rating_sommelier: Optional[float] = None
    rating_people: Optional[float] = None


class WineCatalog:
    def __init__(self, images_dir: Path, db_config: dict, schema: str = "just_vine_it"):
        self.images_dir = images_dir
        self.db_config = db_config
        self.schema = schema
        self.wines: dict[str, Wine] = {}
        self.series_groups: dict[str, list[str]] = {}

    def load(self):
        conn = psycopg2.connect(**self.db_config)
        try:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT name, category, color, region, grape_variety,
                       description, winery, slug, image_filename, image_path,
                       embedding, rating_roskachestvo, rating_sommelier, rating_people
                FROM {self.schema}.wines
                ORDER BY id
            """)
            for row in cur.fetchall():
                (name, category, color, region, grape, description,
                 winery, slug, image_filename, image_path,
                 embedding, rating_roskachestvo, rating_sommelier, rating_people) = row

                if not slug:
                    continue

                slug = slug.strip()
                image_filename = (image_filename or "").strip()

                # Parse pgvector embedding string "[0.1,0.2,...]" to list of floats
                embedding_list = None
                if embedding is not None:
                    try:
                        if isinstance(embedding, str):
                            embedding_list = [float(x) for x in embedding.strip("[]").split(",")]
                        elif isinstance(embedding, list):
                            embedding_list = [float(x) for x in embedding]
                    except (ValueError, TypeError):
                        embedding_list = None

                # Resolve image file path
                full_image_path = None
                if image_filename:
                    candidate = self.images_dir / image_filename
                    if candidate.is_file():
                        full_image_path = candidate
                    else:
                        stem = Path(image_filename).stem
                        for ext in [".webp", ".png", ".jpg", ".jpeg"]:
                            candidate = self.images_dir / f"{stem}{ext}"
                            if candidate.is_file():
                                full_image_path = candidate
                                break

                # Extract year from name if present (e.g. "Алиготе Баррель, 2024")
                year = None
                name_str = (name or slug).strip()
                name_parts = name_str.rsplit(",", 1)
                if len(name_parts) > 1:
                    try:
                        year = int(name_parts[-1].strip())
                    except ValueError:
                        pass

                wine = Wine(
                    slug=slug,
                    name=name_str,
                    producer=(winery or "").strip(),
                    region=(region or "").strip(),
                    grape=(grape or "").strip(),
                    year=year,
                    category=(category or "").strip(),
                    rating=None,
                    description=(description or "").strip(),
                    image_path=str(full_image_path) if full_image_path else "",
                    food_pairing="",
                    alcohol=None,
                    price=None,
                    series="",
                    embedding=embedding_list,
                    rating_roskachestvo=rating_roskachestvo,
                    rating_sommelier=rating_sommelier,
                    rating_people=rating_people,
                )
                self.wines[slug] = wine

                series_key = wine.series or wine.name
                if series_key not in self.series_groups:
                    self.series_groups[series_key] = []
                self.series_groups[series_key].append(slug)
        finally:
            conn.close()

    def get(self, slug: str) -> Optional[Wine]:
        return self.wines.get(slug)

    def get_seriesmates(self, slug: str) -> list[str]:
        wine = self.wines.get(slug)
        if not wine:
            return []
        series_key = wine.series or wine.name
        return [s for s in self.series_groups.get(series_key, []) if s != slug]

    def search_text(self, query: str, limit: int = 10) -> list[Wine]:
        query_lower = query.lower()
        results = []
        for wine in self.wines.values():
            score = 0
            if query_lower in wine.name.lower():
                score += 3
            if query_lower in wine.producer.lower():
                score += 2
            if query_lower in wine.grape.lower():
                score += 1
            if score > 0:
                results.append((score, wine))
        results.sort(key=lambda x: x[0], reverse=True)
        return [w for _, w in results[:limit]]

    def __len__(self):
        return len(self.wines)
