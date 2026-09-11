"""Image discovery, normalized OCR results, and versioned caching."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

from . import adapters as _adapters
from .schemas import (
    ADAPTER_REGISTRY_VERSION,
    ADAPTER_SCHEMA,
    CONFIG_SCHEMA,
    JOB_SCHEMA,
    RESULT_SCHEMA,
)

ADAPTER_VERSION = ADAPTER_REGISTRY_VERSION
compact_text = _adapters.compact_text
extract_fields = _adapters.extract_fields
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pgm"}
DERIVED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "objects",
    "runs",
    "visualizations",
}


class OCREngine(Protocol):
    name: str
    model: str

    def recognize(
        self, path: Path, *, visualize_path: Path | None = None
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    relative_path: str
    sha256: str
    width: int | None
    height: int | None
    image_error: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dimensions(path: Path) -> tuple[int | None, int | None, str | None]:
    try:
        with Image.open(path) as image:
            width, height = image.size
        return width, height, None
    except (OSError, UnidentifiedImageError) as error:
        return None, None, str(error)


def discover_images(root: Path, *, excluded_roots: tuple[Path, ...] = ()) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_SUFFIXES else []
    resolved_exclusions = tuple(path.resolve() for path in excluded_roots)

    def included(path: Path) -> bool:
        relative_parts = path.relative_to(root).parts[:-1]
        if any(part.casefold() in DERIVED_DIRECTORY_NAMES for part in relative_parts):
            return False
        resolved = path.resolve()
        return not any(
            resolved == excluded or resolved.is_relative_to(excluded)
            for excluded in resolved_exclusions
        )

    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and included(path)
        ),
        key=lambda path: str(path).casefold(),
    )


def inspect_image(path: Path, root: Path) -> ImageRecord:
    width, height, error = image_dimensions(path)
    relative_path = path.name if root.is_file() else str(path.relative_to(root))
    return ImageRecord(
        path=path,
        relative_path=relative_path,
        sha256=sha256_file(path),
        width=width,
        height=height,
        image_error=error,
    )


def profile_for(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if "hqchip" in {part.casefold() for part in path.parts}:
        return "generic"
    name = path.name.casefold()
    profile_tokens = (
        (("xianyu", "闲鱼", "閒魚"), "xianyu_order_detail"),
        (("alipay", "支付宝", "支付寶"), "alipay_payment_detail"),
        (("taobao", "淘宝"), "taobao_order_detail"),
        (("receipt", "invoice", "收据", "收據", "小票"), "vendor_receipt"),
        (("travel", "approval", "出差", "审批", "審批"), "travel_approval"),
        (("didi", "ride", "滴滴", "打车", "打車"), "ride_payment"),
        (
            ("mtr", "transit", "metro", "港铁", "港鐵", "地铁", "地鐵"),
            "transit_payment",
        ),
    )
    for tokens, profile in profile_tokens:
        if any(token in name for token in tokens):
            return profile
    return "generic"


def cache_key(
    record: ImageRecord,
    *,
    profile: str,
    min_score: float,
    evidence_id: str | None = None,
    business_context: dict[str, Any] | None = None,
    expected_fields: tuple[str, ...] = (),
) -> str:
    normalized_profile = _adapters.normalize_profile(profile)
    adapter_versions = (
        {name: adapter.version for name, adapter in _adapters.PROFILE_ADAPTERS.items()}
        if normalized_profile in {"auto", "generic"}
        else {
            normalized_profile: _adapters.PROFILE_ADAPTERS[normalized_profile].version
        }
    )
    payload = {
        "source_sha256": record.sha256,
        "schemas": {
            "adapter": ADAPTER_SCHEMA,
            "configuration": CONFIG_SCHEMA,
            "job": JOB_SCHEMA,
            "result": RESULT_SCHEMA,
        },
        "adapter_registry_version": ADAPTER_REGISTRY_VERSION,
        "adapter_versions": adapter_versions,
        "evidence_id": evidence_id,
        "requested_profile": profile,
        "minimum_score": round(min_score, 4),
        "business_context": business_context or {},
        "expected_fields": list(expected_fields),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
