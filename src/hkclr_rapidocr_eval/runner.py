"""Evaluation run orchestration."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .core import (
    ADAPTER_VERSION,
    OCREngine,
    atomic_write_json,
    cache_key,
    discover_images,
    extract_fields,
    inspect_image,
    profile_for,
)


def run_scan(
    *,
    source: Path,
    output: Path,
    requested_profile: str,
    min_score: float,
    limit: int | None,
    force: bool,
    visualize: bool,
    dry_run: bool,
    engine: OCREngine | None,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")

    images = discover_images(source)
    if limit is not None:
        images = images[:limit]
    output.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    cache_hits = 0
    errors = 0
    total_elapsed = 0.0
    profile_counts: dict[str, int] = {}
    pass_counts: dict[str, int] = {}

    for index, path in enumerate(images, 1):
        image = inspect_image(path, source)
        profile = profile_for(path, requested_profile)
        profile_counts[profile] = profile_counts.get(profile, 0) + 1
        key = cache_key(image, profile=profile, min_score=min_score)
        object_path = output / "objects" / f"{key}.ocr.json"
        row: dict[str, Any] = {
            "index": index,
            "source_path": str(path),
            "relative_path": image.relative_path,
            "sha256": image.sha256,
            "width": image.width,
            "height": image.height,
            "profile": profile,
            "cache_key": key,
            "result_path": str(object_path),
        }

        if image.image_error:
            row.update(status="error", error=image.image_error)
            errors += 1
        elif dry_run:
            row.update(status="dry_run")
        elif object_path.exists() and not force:
            cached = json.loads(object_path.read_text(encoding="utf-8"))
            row.update(status="cached", extracted_fields=cached.get("extracted_fields", {}))
            cache_hits += 1
        else:
            if engine is None:
                raise RuntimeError("An OCR engine is required unless --dry-run is used")
            try:
                visual_path = output / "visualizations" / f"{key}.jpg" if visualize else None
                raw = engine.recognize(path, visualize_path=visual_path)
                lines = [line for line in raw["lines"] if float(line["score"]) >= min_score]
                fields = extract_fields(lines, profile)
                elapsed = raw.get("elapsed_seconds")
                if elapsed is not None:
                    total_elapsed += float(elapsed)
                payload = {
                    "schema": "hkclr.rapidocr.result.v1",
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "adapter_version": ADAPTER_VERSION,
                    "engine": {"name": engine.name, "model": engine.model},
                    "source": {
                        "path": str(path),
                        "relative_path": image.relative_path,
                        "sha256": image.sha256,
                        "width": image.width,
                        "height": image.height,
                    },
                    "settings": {"profile": profile, "min_score": min_score},
                    "lines": lines,
                    "full_text": "\n".join(line["text"] for line in lines),
                    "extracted_fields": fields,
                    "timing": {
                        "elapsed_seconds": elapsed,
                        "stage_elapsed_seconds": raw.get("stage_elapsed_seconds", []),
                    },
                    "language": raw.get("language"),
                    "visualization_path": str(visual_path) if visual_path else None,
                }
                atomic_write_json(object_path, payload)
                row.update(status="ok", extracted_fields=fields)
            except Exception as error:
                row.update(status="error", error=f"{type(error).__name__}: {error}")
                errors += 1

        check = row.get("extracted_fields", {}).get("profile_check")
        if check:
            pass_counts[check] = pass_counts.get(check, 0) + 1
        manifest_rows.append(row)

    manifest_path = output / "ocr-manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    summary = {
        "schema": "hkclr.rapidocr.summary.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source),
        "output": str(output),
        "dry_run": dry_run,
        "images_discovered": len(images),
        "processed": sum(row["status"] == "ok" for row in manifest_rows),
        "cache_hits": cache_hits,
        "errors": errors,
        "profiles": profile_counts,
        "profile_checks": pass_counts,
        "ocr_elapsed_seconds": round(total_elapsed, 3),
        "manifest": str(manifest_path),
    }
    atomic_write_json(output / "ocr-summary.json", summary)
    return summary

