"""
Fine-tune SigLIP image encoder on labeled (real photo → wine slug) pairs.

Usage:
    python -m backend.fine_tune              # train and save model
    python -m backend.fine_tune --update-db  # also recompute all catalog embeddings in DB
"""
import argparse
import csv
import logging
import random
from pathlib import Path

import numpy as np
import psycopg2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import open_clip

from .config import (
    PROJECT_ROOT, CATALOG_IMAGES_DIR, DB_CONFIG, DB_SCHEMA,
    CLIP_MODEL_NAME, CLIP_PRETRAINED, CLIP_EMBEDDING_DIM,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABELS_FILE = PROJECT_ROOT / "labels_template.csv"
PHOTOS_DIR = PROJECT_ROOT / "real_photo"
MODEL_SAVE_DIR = PROJECT_ROOT / "models" / "siglip_finetuned"


# ─── Dataset ───────────────────────────────────────────────────────

class WineLabelDataset(Dataset):
    """Pairs of (real photo, catalog embedding) for contrastive training."""

    def __init__(self, pairs: list[tuple[str, np.ndarray]], preprocess):
        """
        Args:
            pairs: [(photo_path, target_embedding), ...]
            preprocess: SigLIP image preprocessing transform
        """
        self.pairs = pairs
        self.preprocess = preprocess

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        photo_path, target_emb = self.pairs[idx]
        img = Image.open(photo_path).convert("RGB")
        img = self.preprocess(img)
        target = torch.tensor(target_emb, dtype=torch.float32)
        return img, target


# ─── Load data ─────────────────────────────────────────────────────

def load_labeled_pairs(labels_path: Path, photos_dir: Path, db_config: dict, schema: str):
    """
    Load labeled pairs: (photo_path, catalog_embedding) from labels + DB.
    Returns list of (photo_path, embedding_768d).
    """
    # Load slugs from labels
    labels = []
    with open(labels_path, "r") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            slug = row["slug"].strip()
            if slug and slug != "Nan":
                labels.append((row["image_filename"].strip(), slug.lower()))

    logger.info(f"Loaded {len(labels)} labeled pairs from {labels_path}")

    # Load embeddings from DB
    conn = psycopg2.connect(**db_config)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT slug, embedding FROM {schema}.wines WHERE embedding IS NOT NULL")
        slug_to_emb = {}
        for slug, embedding in cur.fetchall():
            if isinstance(embedding, str):
                emb = np.array([float(x) for x in embedding.strip("[]").split(",")], dtype=np.float32)
            elif isinstance(embedding, list):
                emb = np.array(embedding, dtype=np.float32)
            else:
                emb = np.array(list(embedding), dtype=np.float32)
            slug_to_emb[slug.strip()] = emb
    finally:
        conn.close()

    logger.info(f"Loaded {len(slug_to_emb)} catalog embeddings from DB")

    # Build pairs
    pairs = []
    missing = []
    for filename, slug in labels:
        photo_path = photos_dir / filename
        if not photo_path.exists():
            missing.append(filename)
            continue
        if slug not in slug_to_emb:
            missing.append(f"{filename} -> {slug} (slug not in DB)")
            continue
        pairs.append((str(photo_path), slug_to_emb[slug]))

    if missing:
        logger.warning(f"Skipped {len(missing)} entries: {missing[:5]}...")

    logger.info(f"Final training pairs: {len(pairs)}")
    return pairs


# ─── Training ──────────────────────────────────────────────────────

def train(
    model, preprocess, device, pairs,
    epochs=5, lr=1e-5, batch_size=4, val_split=0.2,
):
    """Fine-tune image encoder with contrastive loss against frozen catalog embeddings."""
    random.seed(42)
    random.shuffle(pairs)

    split = int(len(pairs) * (1 - val_split))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    logger.info(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    train_ds = WineLabelDataset(train_pairs, preprocess)
    val_ds = WineLabelDataset(val_pairs, preprocess)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_imgs, batch_targets in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_targets = batch_targets.to(device)

            # Encode images
            image_emb = model.encode_image(batch_imgs)
            image_emb = F.normalize(image_emb, dim=-1)

            # L2 normalize target embeddings
            batch_targets = F.normalize(batch_targets, dim=-1)

            # Contrastive loss: we want each image embedding to be close
            # to its target embedding and far from others
            # Similar to InfoNCE: similarity matrix vs identity
            logits = image_emb @ batch_targets.T  # (batch, batch)
            logits = logits * 10.0  # temperature scaling

            labels = torch.arange(len(batch_imgs), device=device)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        # Validation
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        if val_loader:
            model.eval()
            with torch.no_grad():
                for batch_imgs, batch_targets in val_loader:
                    batch_imgs = batch_imgs.to(device)
                    batch_targets = batch_targets.to(device)

                    image_emb = model.encode_image(batch_imgs)
                    image_emb = F.normalize(image_emb, dim=-1)
                    batch_targets = F.normalize(batch_targets, dim=-1)

                    logits = image_emb @ batch_targets.T
                    logits = logits * 10.0

                    labels = torch.arange(len(batch_imgs), device=device)
                    loss = F.cross_entropy(logits, labels)
                    val_loss += loss.item()

                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total += len(labels)

            val_loss /= max(len(val_loader), 1)
            val_acc = val_correct / max(val_total, 1)
        else:
            val_loss = avg_train_loss
            val_acc = 0.0

        logger.info(
            f"Epoch {epoch+1}/{epochs} — "
            f"train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.2%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state


# ─── Update DB embeddings ──────────────────────────────────────────

def update_db_embeddings(model, preprocess, device, db_config, schema):
    """Recompute embeddings for ALL catalog wines using the fine-tuned model and update DB."""
    logger.info("Loading catalog images for embedding recomputation...")

    # Load catalog from DB to get slug → image_filename mapping
    conn = psycopg2.connect(**db_config)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT slug, image_filename FROM {schema}.wines ORDER BY id")
        wines = [(row[0].strip(), row[1]) for row in cur.fetchall() if row[0] and row[1]]
    finally:
        conn.close()

    logger.info(f"Encoding {len(wines)} wines with fine-tuned model...")

    model.eval()
    embeddings = {}

    with torch.no_grad():
        for i, (slug, image_filename) in enumerate(wines):
            image_path = CATALOG_IMAGES_DIR / image_filename.strip()
            if not image_path.exists():
                # Try alternative extensions
                stem = Path(image_filename).stem
                for ext in [".webp", ".png", ".jpg", ".jpeg"]:
                    candidate = CATALOG_IMAGES_DIR / f"{stem}{ext}"
                    if candidate.is_file():
                        image_path = candidate
                        break
                else:
                    continue

            try:
                img = Image.open(image_path).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                emb = model.encode_image(img_tensor)
                emb = F.normalize(emb, dim=-1)
                embeddings[slug] = emb.cpu().numpy().squeeze()
            except Exception as e:
                logger.warning(f"Failed to encode {slug}: {e}")

            if (i + 1) % 200 == 0:
                logger.info(f"  Progress: {i+1}/{len(wines)}")

    logger.info(f"Encoded {len(embeddings)} wines. Updating DB...")

    # Update DB
    conn = psycopg2.connect(**db_config)
    try:
        cur = conn.cursor()
        updated = 0
        for slug, emb in embeddings.items():
            emb_str = "[" + ",".join(f"{x:.8f}" for x in emb.tolist()) + "]"
            cur.execute(
                f"UPDATE {schema}.wines SET embedding = %s::vector WHERE slug = %s",
                (emb_str, slug),
            )
            updated += cur.rowcount
        conn.commit()
        logger.info(f"Updated {updated} embeddings in DB")
    finally:
        conn.close()


# ─── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SigLIP on wine label data")
    parser.add_argument("--update-db", action="store_true", help="Recompute all catalog embeddings and update DB")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load model
    logger.info(f"Loading SigLIP model {CLIP_MODEL_NAME}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device
    )

    # Load labeled pairs
    pairs = load_labeled_pairs(LABELS_FILE, PHOTOS_DIR, DB_CONFIG, DB_SCHEMA)
    if len(pairs) < 2:
        logger.error("Not enough labeled pairs to train")
        return

    # Train
    logger.info("Starting fine-tuning...")
    best_state = train(
        model, preprocess, device, pairs,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )

    # Save model
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_SAVE_DIR / "model_state.pt"
    torch.save(best_state, save_path)
    logger.info(f"Saved fine-tuned model to {save_path}")

    # Load best state back for potential DB update
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    if args.update_db:
        update_db_embeddings(model, preprocess, device, DB_CONFIG, DB_SCHEMA)


if __name__ == "__main__":
    main()
