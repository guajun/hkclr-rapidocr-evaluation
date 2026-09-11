"""Typed OCR jobs and versioned job-manifest loading."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters import normalize_profile
from .schemas import CONFIG_SCHEMA, JOB_MANIFEST_SCHEMA, JOB_SCHEMA

EVIDENCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True)
class OCRJob:
    evidence_id: str
    source_path: Path
    requested_profile: str
    expected_fields: tuple[str, ...]
    business_context: dict[str, Any] = field(default_factory=dict)
    schema: str = JOB_SCHEMA

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OCRJob:
        if payload.get("schema") != JOB_SCHEMA:
            raise ValueError(f"Each job must use schema {JOB_SCHEMA!r}")

        evidence_id = payload.get("evidence_id")
        if not isinstance(evidence_id, str) or not EVIDENCE_ID_RE.fullmatch(
            evidence_id
        ):
            raise ValueError("evidence_id must be a stable 1-128 character identifier")

        raw_path = payload.get("source_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"Job {evidence_id!r} must include source_path")
        source_path = Path(raw_path)
        if not source_path.is_absolute():
            raise ValueError(f"Job {evidence_id!r} source_path must be absolute")

        requested_profile = payload.get("requested_profile")
        if not isinstance(requested_profile, str) or not requested_profile:
            raise ValueError(f"Job {evidence_id!r} must include requested_profile")
        normalize_profile(requested_profile)

        expected_fields = payload.get("expected_fields")
        if (
            not isinstance(expected_fields, list)
            or any(not isinstance(name, str) or not name for name in expected_fields)
            or len(set(expected_fields)) != len(expected_fields)
        ):
            raise ValueError(
                f"Job {evidence_id!r} expected_fields must be a unique string list"
            )

        business_context = payload.get("business_context", {})
        if not isinstance(business_context, dict):
            raise TypeError(f"Job {evidence_id!r} business_context must be an object")

        return cls(
            evidence_id=evidence_id,
            source_path=source_path.resolve(),
            requested_profile=requested_profile,
            expected_fields=tuple(expected_fields),
            business_context=business_context,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_id": self.evidence_id,
            "source_path": str(self.source_path),
            "requested_profile": self.requested_profile,
            "business_context": self.business_context,
            "expected_fields": list(self.expected_fields),
        }


@dataclass(frozen=True)
class JobManifest:
    jobs: tuple[OCRJob, ...]
    configuration: dict[str, Any]
    schema: str = JOB_MANIFEST_SCHEMA


def load_job_manifest(path: Path) -> JobManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON job manifest: {error}") from error

    if not isinstance(payload, dict) or payload.get("schema") != JOB_MANIFEST_SCHEMA:
        raise ValueError(f"Job manifest must use schema {JOB_MANIFEST_SCHEMA!r}")

    raw_configuration = payload.get("configuration", {"schema": CONFIG_SCHEMA})
    if (
        not isinstance(raw_configuration, dict)
        or raw_configuration.get("schema") != CONFIG_SCHEMA
    ):
        raise ValueError(f"Manifest configuration must use schema {CONFIG_SCHEMA!r}")
    configuration = dict(raw_configuration)

    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list):
        raise TypeError("Job manifest jobs must be a list")
    jobs = tuple(OCRJob.from_dict(job) for job in raw_jobs if isinstance(job, dict))
    if len(jobs) != len(raw_jobs):
        raise ValueError("Every job manifest entry must be an object")
    if len({job.evidence_id for job in jobs}) != len(jobs):
        raise ValueError("Job manifest evidence_id values must be unique")

    return JobManifest(jobs=jobs, configuration=configuration)
