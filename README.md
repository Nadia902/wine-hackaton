# 🍷 Своё Вино — Wine Label Scanner

Сервис для мгновенного распознавания российских вин по фотографии этикетки.

## Быстрый старт (Docker)

### 1. Клонировать проект

```bash
cd svoe-vino
git clone https://github.com/Nadia902/wine-hackaton.git
```

### 2. Настроить окружение

```bash
cp .env.example .env
# отредактируйте .env при необходимости
```

## Модель
Скачайте здесь: https://disk.yandex.ru/d/S8NQeiH2_KKzjA
Размер: ~750 MB
Разархивируйте и положите в корень проекта

### 3. Запустить

```bash
docker compose up -d
```

Сервер будет доступен на `http://localhost:8000`.

загружает модель SigLIP (~60 сек). Проверьте:

```bash
curl http://localhost:8000/api/stats
```

### 4. Остановить

```bash
docker compose down
```

Данные PostgreSQL сохраняются в Docker volume `wine-db-data`.

## Требования

- Docker + Docker Compose v2
- 4GB RAM (для backend) + 1GB RAM (для PostgreSQL)
- ~3GB диска (образ) + ~1GB (данные БД)

## Переменные окружения

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `POSTGRES_USER` | `wine_user` | Пользователь PostgreSQL |
| `POSTGRES_PASSWORD` | `wine_secret_2026` | Пароль PostgreSQL |
| `POSTGRES_DB` | `wine_db` | Имя базы данных |
| `PORT` | `8000` | Порт сервера |

## Архитектура

См. [ARCHITECTURE.md](ARCHITECTURE.md).

## API Endpoints

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/scan` | Сканирование (карточка + альтернативы) |
| `POST` | `/api/scan/flat` | Сканирование → `{"slug": "..."}` |
| `POST` | `/v1/eval/predict` | Evaluation endpoint |
| `GET` | `/api/wine/{slug}` | Карточка вина по slug |
| `GET` | `/api/catalog?q=...` | Поиск/список каталога |
| `GET` | `/api/stats` | Статистика каталога |

### Пример запроса

```bash
curl -X POST http://localhost:8000/api/scan \
  -F "image=@bottle_photo.jpg"
```

### Пример ответа

```json
{
    "slug": "abrau-durso-classic",
    "confidence": 0.85,
    "in_catalog": true,
    "wine": {
        "name": "Абрау-Дюрсо Классик",
        "producer": "Абрау-Дюрсо",
        "region": "Краснодарский край",
        "grape": "Пино Нуар, Шардоне",
        "rating_roskachestvo": 4.5,
        "rating_sommelier": 87.0,
        "rating_people": 4.2
    },
    "alternatives": [],
    "match_time_ms": 1500
}
```

## Технологии

- **Vision модель:** Google SigLIP (`ViT-B-16-SigLIP`) — 768-мерные эмбеддинги
- **Векторный поиск:** PostgreSQL + pgvector с HNSW индексом
- **Визуальный ре-рейтинг:** ORB локальные особенности (128x128 thumbnails)
- **OCR (опционально):** EasyOCR (ru+en) для текстового сопоставления
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Vanilla HTML/CSS/JS (mobile-first)
- **Deploy:** Docker Compose

## Pipeline распознавания

1. **SigLIP TTA** — 6 аугментаций запроса (яркость, контраст, кроп, поворот)
2. **pgvector HNSW** — поиск ближайших соседей в БД (~20ms)
3. **ORB ре-рейтинг** — визуальное сопоставление топ-20 кандидатов
4. **Fusion** — CLIP (55%) + ORB (25%) + OCR (20%, опционально)

## Структура проекта

```
├── backend/
│   ├── app.py          # FastAPI сервер, endpoints
│   ├── config.py       # Конфигурация (DB, модель, пути)
│   ├── catalog.py      # Менеджер каталога (PostgreSQL)
│   ├── matcher.py      # SigLIP + pgvector + ORB matcher
│   ├── matcher_ocr.py  # EasyOCR + fuzzy matching
│   ├── preprocess.py   # Предобработка изображений
│   └── fine_tune.py    # Fine-tuning SigLIP
├── frontend/
│   ├── index.html      # Главная страница (сканер)
│   ├── wine.html       # Страница карточки вина
│   ├── style.css       # Стили (mobile-first)
│   └── app.js          # Логика фронтенда
├── data/
│   └── sift_index/     # ORB дескрипторы
├── models/
│   └── siglip_finetuned/  # Fine-tuned SigLIP модель
├── opt_dataset/
│   └── clean_uploads/  # Каталог изображений вин
├── database/
│   └── init.sql        # Schema + данные PostgreSQL
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── ARCHITECTURE.md
└── README.md
```

## Development (без Docker)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# PostgreSQL (нужен отдельно)
export DB_HOST=127.0.0.1

python -m backend.app
```

## Evaluation

```bash
./eval/participant_test.sh \
  --images-dir ./eval/queries \
  --manifest ./eval/queries.tsv \
  --endpoint 'http://127.0.0.1:8000/v1/eval/predict' \
  --output ./eval/predictions.jsonl
```
