"""Image discovery, normalized OCR results, caching, and field extraction."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from PIL import Image, UnidentifiedImageError

ADAPTER_VERSION = "rapidocr-normalized-v1"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pgm"}
TRADE_NO_RE = re.compile(r"(?<!\d)20\d{26}(?!\d)")
AMOUNT_RE = re.compile(
    r"(?:(?:RMB|CNY|HKD|USD|MOP|JPY|NTD)\s*)?[¥￥$]?[ ]*([0-9]{1,8}(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)",
    flags=re.IGNORECASE,
)

PROFILE_KEYWORDS = {
    "taobao": ("交易成功", "实付款", "支付方式", "支付宝交易号"),
    "alipay": ("交易成功", "流水号", "订单金额", "实付金额"),
    "generic": (),
}


class OCREngine(Protocol):
    name: str
    model: str

    def recognize(self, path: Path, *, visualize_path: Path | None = None) -> dict[str, Any]: ...


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


def discover_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_SUFFIXES else []
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
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


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text).replace("：", ":")


def profile_for(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if "hqchip" in {part.casefold() for part in path.parts}:
        return "generic"
    name = path.name.casefold()
    if any(token in name for token in ("alipay", "payment", "付款", "支付")):
        return "alipay"
    if any(token in name for token in ("taobao", "order", "淘宝", "訂單", "订单")):
        return "taobao"
    return "generic"


def extract_fields(lines: Iterable[dict[str, Any]], profile: str) -> dict[str, Any]:
    line_list = list(lines)
    full_text = "\n".join(str(line.get("text", "")) for line in line_list)
    joined = compact_text(full_text)
    trade_numbers = list(dict.fromkeys(TRADE_NO_RE.findall(joined)))

    amounts: list[str] = []
    for line in line_list:
        text = str(line.get("text", ""))
        if any(marker in text for marker in ("金额", "金額", "实付", "實付", "付款", "合计", "合計", "¥", "￥", "RMB", "HKD")):
            amounts.extend(match.replace(",", "") for match in AMOUNT_RE.findall(text))

    required = PROFILE_KEYWORDS[profile]
    found = [keyword for keyword in required if compact_text(keyword) in joined]
    missing = [keyword for keyword in required if keyword not in found]
    return {
        "profile": profile,
        "alipay_trade_no_candidates": trade_numbers,
        "amount_candidates": list(dict.fromkeys(amounts)),
        "required_keywords_found": found,
        "required_keywords_missing": missing,
        "profile_check": "pass" if required and not missing else ("review" if required else "not_applicable"),
    }


def cache_key(record: ImageRecord, *, profile: str, min_score: float) -> str:
    payload = f"{record.sha256}|{ADAPTER_VERSION}|{profile}|{min_score:.4f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
