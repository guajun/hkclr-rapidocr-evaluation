from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from hkclr_rapidocr_eval.jobs import load_job_manifest
from hkclr_rapidocr_eval.runner import run_manifest, run_scan
from hkclr_rapidocr_eval.schemas import (
    CONFIG_SCHEMA,
    JOB_MANIFEST_SCHEMA,
    JOB_SCHEMA,
    RESULT_SCHEMA,
    SUMMARY_SCHEMA,
)

FIXTURES = Path(__file__).parent / "fixtures"


class FakeEngine:
    name = "synthetic-ocr"
    model = "fixture-v1"

    def __init__(self, lines: list[dict[str, Any]]) -> None:
        self.lines = lines

    def recognize(
        self, path: Path, *, visualize_path: Path | None = None
    ) -> dict[str, Any]:
        del path, visualize_path
        return {
            "lines": self.lines,
            "elapsed_seconds": 0.01,
            "stage_elapsed_seconds": [0.01],
            "language": "synthetic",
        }


def travel_lines() -> list[dict[str, Any]]:
    payload = json.loads(
        (FIXTURES / "ocr" / "travel_approval.json").read_text(encoding="utf-8")
    )
    return payload["lines"]


class JobAndRunnerTests(unittest.TestCase):
    def test_manifest_requires_absolute_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "jobs.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": JOB_MANIFEST_SCHEMA,
                        "jobs": [
                            {
                                "schema": JOB_SCHEMA,
                                "evidence_id": "synthetic-relative",
                                "source_path": "relative/image.png",
                                "requested_profile": "travel_approval",
                                "expected_fields": ["destination"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must be absolute"):
                load_job_manifest(manifest_path)

    def test_manifest_run_writes_the_typed_result_contract_and_uses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "synthetic-travel.pgm"
            shutil.copyfile(FIXTURES / "a.PGM", image)
            manifest_path = root / "jobs.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": JOB_MANIFEST_SCHEMA,
                        "configuration": {
                            "schema": CONFIG_SCHEMA,
                            "minimum_score": 0.7,
                        },
                        "jobs": [
                            {
                                "schema": JOB_SCHEMA,
                                "evidence_id": "synthetic-travel-001",
                                "source_path": str(image.resolve()),
                                "requested_profile": "travel_approval",
                                "business_context": {"purpose": "synthetic test"},
                                "expected_fields": [
                                    "destination",
                                    "start_date",
                                    "end_date",
                                    "approval_status",
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "private-output"
            engine = FakeEngine(travel_lines())

            first = run_manifest(
                manifest_path=manifest_path,
                output=output,
                min_score=None,
                limit=None,
                force=False,
                visualize=False,
                dry_run=False,
                engine=engine,
            )
            second = run_manifest(
                manifest_path=manifest_path,
                output=output,
                min_score=None,
                limit=None,
                force=False,
                visualize=False,
                dry_run=False,
                engine=engine,
            )

            self.assertEqual(first["schema"], SUMMARY_SCHEMA)
            self.assertEqual(first["configuration"]["minimum_score"], 0.7)
            self.assertEqual(first["image_path_count"], 1)
            self.assertEqual(first["unique_image_count"], 1)
            self.assertEqual(second["cache_hits"], 1)
            result_path = next((output / "objects").glob("*.ocr.json"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["schema"], RESULT_SCHEMA)
            self.assertEqual(result["schemas"]["job"], JOB_SCHEMA)
            self.assertEqual(result["job"]["evidence_id"], "synthetic-travel-001")
            self.assertEqual(result["source"]["sha256"], first_manifest_sha(output))
            self.assertEqual(result["ocr"]["status"], "ok")
            self.assertEqual(result["profile_support"]["status"], "supported")
            self.assertEqual(result["profile_support"]["check"], "pass")
            self.assertEqual(
                result["extracted_fields"]["destination"]["value"], "Shenzhen"
            )

    def test_scan_excludes_generated_folders_and_counts_unique_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            shutil.copyfile(FIXTURES / "a.PGM", source / "one.pgm")
            shutil.copyfile(FIXTURES / "a.PGM", source / "duplicate.pgm")
            generated = source / "runs"
            generated.mkdir()
            shutil.copyfile(FIXTURES / "a.PGM", generated / "generated-copy.pgm")

            summary = run_scan(
                source=source,
                output=generated / "evaluation",
                requested_profile="auto",
                min_score=0.5,
                limit=None,
                force=False,
                visualize=False,
                dry_run=True,
                engine=None,
            )

            self.assertEqual(summary["image_path_count"], 2)
            self.assertEqual(summary["unique_image_count"], 1)

    def test_distinct_evidence_ids_do_not_share_result_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "synthetic-travel.pgm"
            shutil.copyfile(FIXTURES / "a.PGM", image)
            manifest_path = root / "jobs.json"
            jobs = [
                {
                    "schema": JOB_SCHEMA,
                    "evidence_id": evidence_id,
                    "source_path": str(image.resolve()),
                    "requested_profile": "travel_approval",
                    "business_context": {},
                    "expected_fields": ["destination"],
                }
                for evidence_id in ("synthetic-a", "synthetic-b")
            ]
            manifest_path.write_text(
                json.dumps({"schema": JOB_MANIFEST_SCHEMA, "jobs": jobs}),
                encoding="utf-8",
            )
            output = root / "private-output"

            run_manifest(
                manifest_path=manifest_path,
                output=output,
                min_score=0.5,
                limit=None,
                force=False,
                visualize=False,
                dry_run=False,
                engine=FakeEngine(travel_lines()),
            )

            results = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in (output / "objects").glob("*.json")
            ]
            self.assertEqual(len(results), 2)
            self.assertEqual(
                {result["job"]["evidence_id"] for result in results},
                {"synthetic-a", "synthetic-b"},
            )


def first_manifest_sha(output: Path) -> str:
    row = json.loads(
        (output / "ocr-manifest.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    return row["source_sha256"]


if __name__ == "__main__":
    unittest.main()
