"""Lazy RapidOCR adapter so discovery and tests do not require an inference runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class RapidOCREngine:
    name = "rapidocr"
    model = "rapidocr-default"

    def __init__(self) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as error:
            raise RuntimeError("RapidOCR is not installed. Run: uv sync") from error

        try:
            self._engine = RapidOCR()
        except Exception as error:
            raise RuntimeError(
                "RapidOCR could not initialize an inference runtime. "
                "For CPU inference run: uv sync --extra onnx"
            ) from error

    def recognize(self, path: Path, *, visualize_path: Path | None = None) -> dict[str, Any]:
        result = self._engine(str(path))
        boxes = result.boxes if result.boxes is not None else []
        texts = result.txts if result.txts is not None else []
        scores = result.scores if result.scores is not None else []
        lines = [
            {
                "text": str(text),
                "score": float(score),
                "box": box.astype(int).tolist() if hasattr(box, "astype") else box,
            }
            for box, text, score in zip(boxes, texts, scores, strict=True)
        ]

        if visualize_path is not None:
            visualize_path.parent.mkdir(parents=True, exist_ok=True)
            result.vis(str(visualize_path))

        return {
            "lines": lines,
            "elapsed_seconds": float(result.elapse) if result.elapse is not None else None,
            "stage_elapsed_seconds": [float(value) for value in (result.elapse_list or [])],
            "language": getattr(result, "lang_rec", None),
        }

