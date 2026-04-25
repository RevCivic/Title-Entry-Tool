#!/usr/bin/env python3
"""Training data export and OCR fine-tuning pipeline.

Usage
-----
Export all approved ground-truth corrections as image/label pairs::

    python train_ocr.py export --output training_data/

Show statistics about the current ground-truth dataset::

    python train_ocr.py stats

Export only specific fields (e.g. vin and make)::

    python train_ocr.py export --fields vin,make --output training_data/

The exported directory contains:
    training_data/
        manifest.json           – list of all samples with metadata
        samples/
            <uuid>.png          – source title image (one per correction)
            <uuid>.json         – ground-truth labels for that image

The manifest and per-image JSON files are compatible with common
vision-language fine-tuning frameworks (e.g. LLaVA-style LoRA training).
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
TRAINING_DATA_DIR = os.path.join(DATA_DIR, "training_data")

# ---------------------------------------------------------------------------
# Database helpers (inline to keep this script self-contained)
# ---------------------------------------------------------------------------

try:
    from title_entry_tool import create_connection_from_env, list_ground_truth_corrections
except ImportError:
    sys.exit(
        "ERROR: Cannot import title_entry_tool. "
        "Run this script from the project root directory."
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_stats(args: argparse.Namespace) -> None:
    conn = create_connection_from_env()
    try:
        records = list_ground_truth_corrections(conn)
    finally:
        conn.close()

    print(f"\nGround-truth corrections: {len(records)}")

    from collections import Counter

    field_counts: Counter = Counter(r["field_name"] for r in records)
    print("\nBy field:")
    for field, count in sorted(field_counts.items(), key=lambda x: -x[1]):
        print(f"  {field:<20} {count}")

    with_images = sum(1 for r in records if r.get("source_file_path") and Path(r["source_file_path"]).exists())
    print(f"\nWith source image on disk: {with_images}/{len(records)}")


def cmd_export(args: argparse.Namespace) -> None:
    output_dir = Path(args.output or TRAINING_DATA_DIR)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    filter_fields = set(args.fields.split(",")) if args.fields else None

    conn = create_connection_from_env()
    try:
        corrections = list_ground_truth_corrections(conn)
    finally:
        conn.close()

    if filter_fields:
        corrections = [c for c in corrections if c["field_name"] in filter_fields]

    # Group corrections by record_id so each image gets all its labels.
    from collections import defaultdict
    import uuid

    by_record: dict = defaultdict(list)
    record_image: dict = {}
    for corr in corrections:
        rid = corr["record_id"]
        by_record[rid].append(corr)
        if corr.get("source_file_path"):
            record_image[rid] = corr["source_file_path"]

    manifest = []
    exported = 0

    for record_id, corrs in by_record.items():
        src_path = record_image.get(record_id)
        sample_id = str(uuid.uuid4())

        image_dest: str | None = None
        if src_path and Path(src_path).exists():
            ext = Path(src_path).suffix or ".png"
            image_dest = str(samples_dir / f"{sample_id}{ext}")
            shutil.copy2(src_path, image_dest)

        labels = {c["field_name"]: c["corrected_value"] for c in corrs}
        label_path = samples_dir / f"{sample_id}.json"
        label_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")

        manifest.append(
            {
                "id": sample_id,
                "record_id": record_id,
                "image": str(image_dest) if image_dest else None,
                "labels": labels,
                "fields": list(labels.keys()),
            }
        )
        exported += 1

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"total": exported, "samples": manifest}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nExported {exported} training samples to {output_dir}")
    print(f"Manifest: {manifest_path}")
    print("\nNext steps:")
    print("  1. Review samples in", samples_dir)
    print("  2. Use manifest.json to drive your fine-tuning framework.")
    print("  3. For Tesseract fine-tuning, convert images+labels to .box/.tif pairs.")
    print("  4. For LoRA/LLaVA fine-tuning, use the image+JSON pairs directly.")
    print("  5. After fine-tuning, update AI_MODEL in .env and restart the stack.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Title Entry Tool – training data export pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # stats
    subparsers.add_parser("stats", help="Show ground-truth correction statistics")

    # export
    export_parser = subparsers.add_parser(
        "export", help="Export approved corrections as image/label training data"
    )
    export_parser.add_argument(
        "--output",
        metavar="DIR",
        help=f"Output directory (default: {TRAINING_DATA_DIR})",
    )
    export_parser.add_argument(
        "--fields",
        metavar="FIELDS",
        default="",
        help="Comma-separated list of fields to export (default: all)",
    )

    args = parser.parse_args()

    if args.command == "stats":
        cmd_stats(args)
    elif args.command == "export":
        cmd_export(args)


if __name__ == "__main__":
    main()
