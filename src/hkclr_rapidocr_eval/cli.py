"""Command-line interface for local RapidOCR evaluation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

from .adapters import PROFILE_ALIASES, available_profiles
from .engine import RapidOCREngine
from .runner import run_manifest, run_scan

PROFILE_CHOICES = tuple(
    dict.fromkeys(("auto", "generic", *available_profiles(), *PROFILE_ALIASES))
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor", help="Check packages and optional engine initialization"
    )
    doctor.add_argument(
        "--initialize",
        action="store_true",
        help="Instantiate RapidOCR and load its runtime",
    )

    scan = subparsers.add_parser(
        "scan", help="Recursively evaluate images without changing source files"
    )
    scan.add_argument("source", type=Path, help="Image file or folder to scan")
    scan.add_argument(
        "--output", type=Path, required=True, help="Directory for private OCR results"
    )
    scan.add_argument("--profile", choices=PROFILE_CHOICES, default="auto")
    scan.add_argument("--min-score", type=float, default=0.50)
    scan.add_argument("--limit", type=int)
    scan.add_argument("--force", action="store_true")
    scan.add_argument("--visualize", action="store_true")
    scan.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and hash images without loading OCR",
    )

    run = subparsers.add_parser("run", help="Evaluate a versioned OCR job manifest")
    run.add_argument(
        "manifest", type=Path, help="Path to a hkclr.rapidocr.job-manifest.v1 JSON file"
    )
    run.add_argument(
        "--output", type=Path, required=True, help="Directory for private OCR results"
    )
    run.add_argument(
        "--min-score", type=float, help="Override configuration.minimum_score"
    )
    run.add_argument("--limit", type=int)
    run.add_argument("--force", action="store_true")
    run.add_argument("--visualize", action="store_true")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and hash jobs without loading OCR",
    )
    return parser


def run_doctor(initialize: bool) -> int:
    report = {
        "python": sys.version.split()[0],
        "rapidocr": package_version("rapidocr"),
        "onnxruntime": package_version("onnxruntime"),
        "runtime_required_now": initialize,
    }
    if initialize:
        try:
            RapidOCREngine()
            report["initialization"] = "ok"
        except Exception as error:  # noqa: BLE001 - doctor must report runtime initialization failures
            report["initialization"] = "error"
            report["error"] = str(error)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 1
    else:
        report["initialization"] = "not_requested"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args()
    if args.command == "doctor":
        return run_doctor(args.initialize)

    if args.min_score is not None and not 0 <= args.min_score <= 1:
        raise SystemExit("--min-score must be between 0 and 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    engine = None if args.dry_run else RapidOCREngine()
    if args.command == "scan":
        summary = run_scan(
            source=args.source,
            output=args.output,
            requested_profile=args.profile,
            min_score=args.min_score,
            limit=args.limit,
            force=args.force,
            visualize=args.visualize,
            dry_run=args.dry_run,
            engine=engine,
        )
    else:
        summary = run_manifest(
            manifest_path=args.manifest,
            output=args.output,
            min_score=args.min_score,
            limit=args.limit,
            force=args.force,
            visualize=args.visualize,
            dry_run=args.dry_run,
            engine=engine,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
