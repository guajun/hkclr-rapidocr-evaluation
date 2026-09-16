"""Evaluation run orchestration for directory scans and typed job manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from .core import (
    OCREngine,
    atomic_write_json,
    cache_key,
    discover_images,
    extract_fields,
    inspect_image,
)
from .jobs import OCRJob, load_job_manifest
from .schemas import (
    ADAPTER_REGISTRY_VERSION,
    ADAPTER_SCHEMA,
    CONFIG_SCHEMA,
    JOB_SCHEMA,
    RESULT_SCHEMA,
    RUN_RECORD_SCHEMA,
    SUMMARY_SCHEMA,
)

RAW_CACHE_SCHEMA = "hkclr.rapidocr.raw-cache.v1"


def _engine_identity(engine: OCREngine | None) -> dict[str, Any]:
    if engine is None:
        return {}
    return dict(getattr(engine, "cache_identity", {"name": engine.name, "model": engine.model}))


def _raw_cache_key(source_sha256: str, identity: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"schema": RAW_CACHE_SCHEMA, "source_sha256": source_sha256, "engine": identity},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_raw_cache(path: Path, source_sha256: str, identity: dict[str, Any]) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("schema") != RAW_CACHE_SCHEMA
            or payload.get("source_sha256") != source_sha256
            or payload.get("engine") != identity):
            return None
        raw = payload["raw"]
        if not isinstance(raw.get("lines"), list):
            return None
        if not all(isinstance(line, dict) and isinstance(line.get("text"), str)
                   and isinstance(line.get("score"), (int, float)) for line in raw["lines"]):
            return None
        return raw
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _stable_evidence_id(path: Path) -> str:
    canonical = str(path.resolve()).casefold().encode("utf-8")
    return f"scan-{hashlib.sha256(canonical).hexdigest()[:20]}"


def _apply_result_to_row(
    row: dict[str, Any], payload: dict[str, Any], *, status: str
) -> None:
    profile_support = payload.get("profile_support", {})
    row.update(
        status=status,
        ocr_status=payload.get("ocr", {}).get("status", "unknown"),
        profile=profile_support.get("resolved", row["requested_profile"]),
        profile_support_status=profile_support.get("status", "unknown"),
        profile_check=profile_support.get("check", "unknown"),
        extracted_fields=payload.get("extracted_fields", {}),
        transactions=payload.get("transactions", []),
        candidate_totals=payload.get("candidate_totals", []),
        adapter_warnings=payload.get("adapter_warnings", []),
    )


def _run_jobs(
    *,
    jobs: Iterable[OCRJob],
    output: Path,
    source_description: str,
    min_score: float,
    limit: int | None,
    force: bool,
    visualize: bool,
    dry_run: bool,
    engine: OCREngine | None,
) -> dict[str, Any]:
    if not 0 <= min_score <= 1:
        raise ValueError("minimum score must be between 0 and 1")

    selected_jobs = list(jobs)
    if limit is not None:
        selected_jobs = selected_jobs[:limit]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    configuration = {
        "schema": CONFIG_SCHEMA,
        "minimum_score": min_score,
        "adapter_registry_version": ADAPTER_REGISTRY_VERSION,
    }
    manifest_rows: list[dict[str, Any]] = []
    unique_hashes: set[str] = set()
    cache_hits = 0
    raw_cache_hits = 0
    errors = 0
    total_elapsed = 0.0
    identity = _engine_identity(engine)

    for index, job in enumerate(selected_jobs, 1):
        path = job.source_path.resolve()
        if not path.is_file():
            row = {
                "schema": RUN_RECORD_SCHEMA,
                "index": index,
                "evidence_id": job.evidence_id,
                "source_path": str(path),
                "requested_profile": job.requested_profile,
                "status": "error",
                "ocr_status": "error",
                "profile_support_status": "not_evaluated",
                "error": f"Source is not a file: {path}",
            }
            manifest_rows.append(row)
            errors += 1
            continue

        image = inspect_image(path, path)
        unique_hashes.add(image.sha256)
        key = cache_key(
            image,
            profile=job.requested_profile,
            min_score=min_score,
            evidence_id=job.evidence_id,
            business_context=job.business_context,
            expected_fields=job.expected_fields,
            engine_identity=identity,
        )
        object_path = output / "objects" / f"{key}.ocr.json"
        row: dict[str, Any] = {
            "schema": RUN_RECORD_SCHEMA,
            "index": index,
            "evidence_id": job.evidence_id,
            "source_path": str(path),
            "relative_path": image.relative_path,
            "source_sha256": image.sha256,
            "width": image.width,
            "height": image.height,
            "requested_profile": job.requested_profile,
            "expected_fields": list(job.expected_fields),
            "cache_key": key,
            "result_path": str(object_path),
        }

        if image.image_error:
            row.update(
                status="error",
                ocr_status="error",
                profile_support_status="not_evaluated",
                error=image.image_error,
            )
            errors += 1
        elif dry_run:
            row.update(
                status="dry_run",
                ocr_status="not_run",
                profile_support_status="not_evaluated",
            )
        else:
            cached = None
            if object_path.exists() and not force:
                try:
                    candidate = json.loads(object_path.read_text(encoding="utf-8"))
                    cached = (
                        candidate if isinstance(candidate, dict)
                        and candidate.get("schema") == RESULT_SCHEMA
                        and isinstance(candidate.get("source"), dict) else None
                    )
                except (OSError, json.JSONDecodeError):
                    cached = None

            if cached is not None:
                cached["job"] = job.to_dict()
                cached["source"].update(
                    path=str(path),
                    sha256=image.sha256,
                    width=image.width,
                    height=image.height,
                )
                atomic_write_json(object_path, cached)
                _apply_result_to_row(row, cached, status="cached")
                cache_hits += 1
                manifest_rows.append(row)
                continue

            if engine is None:
                raise RuntimeError("An OCR engine is required unless --dry-run is used")
            try:
                raw_key = _raw_cache_key(image.sha256, identity)
                raw_path = output / "raw-objects" / f"{raw_key}.ocr.json"
                visual_path = (
                    output / "visualizations" / f"{raw_key}.jpg" if visualize else None
                )
                raw = None if force else _load_raw_cache(raw_path, image.sha256, identity)
                if visualize and visual_path is not None and not visual_path.exists():
                    raw = None
                raw_cached = raw is not None
                if raw is None:
                    raw = engine.recognize(path, visualize_path=visual_path)
                    atomic_write_json(raw_path, {
                        "schema": RAW_CACHE_SCHEMA, "source_sha256": image.sha256,
                        "engine": identity, "created_at": _now(), "raw": raw,
                    })
                else:
                    raw_cache_hits += 1
                lines = [
                    line for line in raw["lines"] if float(line["score"]) >= min_score
                ]
                extraction = extract_fields(
                    lines,
                    job.requested_profile,
                    source_path=path,
                    business_context=job.business_context,
                    expected_fields=job.expected_fields,
                )
                elapsed = raw.get("elapsed_seconds")
                if elapsed is not None and not raw_cached:
                    total_elapsed += float(elapsed)
                payload = {
                    "schema": RESULT_SCHEMA,
                    "schemas": {
                        "job": JOB_SCHEMA,
                        "configuration": CONFIG_SCHEMA,
                        "adapter": ADAPTER_SCHEMA,
                        "result": RESULT_SCHEMA,
                    },
                    "created_at": _now(),
                    "job": job.to_dict(),
                    "configuration": configuration,
                    "source": {
                        "path": str(path),
                        "sha256": image.sha256,
                        "width": image.width,
                        "height": image.height,
                    },
                    "ocr": {
                        "status": "ok",
                        "engine": {"name": engine.name, "model": engine.model},
                        "elapsed_seconds": elapsed,
                        "stage_elapsed_seconds": raw.get("stage_elapsed_seconds", []),
                        "language": raw.get("language"),
                        "raw_cache_hit": raw_cached,
                        "raw_cache_key": raw_key,
                    },
                    "profile_support": {
                        "requested": job.requested_profile,
                        "resolved": extraction["profile"],
                        "status": extraction["support_status"],
                        "check": extraction["profile_check"],
                        "adapter": extraction["adapter"],
                        "markers_found": extraction.get("support_markers_found", []),
                        "expected_fields": extraction["expected_fields"],
                        "missing_expected_fields": extraction[
                            "missing_expected_fields"
                        ],
                    },
                    "extracted_fields": extraction["fields"],
                    "transactions": extraction["transactions"],
                    "candidate_totals": extraction["candidate_totals"],
                    "adapter_warnings": extraction["warnings"],
                    "lines": lines,
                    "full_text": "\n".join(str(line.get("text", "")) for line in lines),
                    "visualization_path": str(visual_path) if visual_path else None,
                }
                atomic_write_json(object_path, payload)
                _apply_result_to_row(row, payload, status="ok")
            except Exception as error:  # noqa: BLE001 - isolate failures to one evidence job
                row.update(
                    status="error",
                    ocr_status="error",
                    profile_support_status="not_evaluated",
                    error=f"{type(error).__name__}: {error}",
                )
                errors += 1
        manifest_rows.append(row)

    manifest_path = output / "ocr-manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )

    profile_counts: dict[str, int] = {}
    support_counts: dict[str, int] = {}
    check_counts: dict[str, int] = {}
    for row in manifest_rows:
        profile = row.get("profile", row.get("requested_profile", "unknown"))
        profile_counts[profile] = profile_counts.get(profile, 0) + 1
        support = row.get("profile_support_status")
        if support:
            support_counts[support] = support_counts.get(support, 0) + 1
        check = row.get("profile_check")
        if check:
            check_counts[check] = check_counts.get(check, 0) + 1

    summary = {
        "schema": SUMMARY_SCHEMA,
        "schemas": {
            "job": JOB_SCHEMA,
            "configuration": CONFIG_SCHEMA,
            "adapter": ADAPTER_SCHEMA,
            "result": RESULT_SCHEMA,
        },
        "created_at": _now(),
        "source": source_description,
        "output": str(output),
        "configuration": configuration,
        "dry_run": dry_run,
        "image_path_count": len(selected_jobs),
        "unique_image_count": len(unique_hashes),
        "images_discovered": len(selected_jobs),
        "processed": sum(row["status"] == "ok" for row in manifest_rows),
        "cache_hits": cache_hits,
        "raw_cache_hits": raw_cache_hits,
        "errors": errors,
        "profiles": profile_counts,
        "profile_support": support_counts,
        "profile_checks": check_counts,
        "ocr_elapsed_seconds": round(total_elapsed, 3),
        "manifest": str(manifest_path),
    }
    atomic_write_json(output / "ocr-summary.json", summary)
    return summary


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
    if source.is_dir() and source == output:
        raise ValueError("Output directory must differ from the source directory")

    images = discover_images(source, excluded_roots=(output,))
    jobs = (
        OCRJob(
            evidence_id=_stable_evidence_id(path),
            source_path=path.resolve(),
            requested_profile=requested_profile,
            expected_fields=(),
        )
        for path in images
    )
    return _run_jobs(
        jobs=jobs,
        output=output,
        source_description=str(source),
        min_score=min_score,
        limit=limit,
        force=force,
        visualize=visualize,
        dry_run=dry_run,
        engine=engine,
    )


def run_manifest(
    *,
    manifest_path: Path,
    output: Path,
    min_score: float | None,
    limit: int | None,
    force: bool,
    visualize: bool,
    dry_run: bool,
    engine: OCREngine | None,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = load_job_manifest(manifest_path)
    configured_score = manifest.configuration.get("minimum_score", 0.5)
    effective_score = float(configured_score if min_score is None else min_score)
    return _run_jobs(
        jobs=manifest.jobs,
        output=output,
        source_description=str(manifest_path),
        min_score=effective_score,
        limit=limit,
        force=force,
        visualize=visualize,
        dry_run=dry_run,
        engine=engine,
    )
