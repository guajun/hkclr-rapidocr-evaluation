# HKCLR RapidOCR Evaluation

Local, read-only OCR evaluation harness for reimbursement evidence. It accepts
directory scans or typed OCR job manifests, runs RapidOCR, stores normalized
JSON results by versioned cache key, and applies business-aware evidence adapters.

The tool never edits, renames, or moves source images. OCR output can contain
sensitive reimbursement and payment data, so `runs/` is ignored by Git.
Local source copies under `input/` are also ignored by Git.

## Setup

RapidOCR is a project dependency. ONNX Runtime is deliberately optional so the
repository and non-inference checks can be used before a runtime is ready.

```powershell
uv sync
uv run hkclr-ocr doctor
uv run hkclr-ocr scan "C:\path\to\screenshots" --output .\runs\first --dry-run
```

Install the CPU runtime when ready:

```powershell
uv sync --extra onnx
uv run hkclr-ocr doctor --initialize
```

## Run OCR

```powershell
uv run hkclr-ocr scan "C:\path\to\screenshots" `
  --output .\runs\first `
  --profile auto
```

The adapter registry supports `taobao_order_detail`, `xianyu_order_detail`,
`alipay_payment_detail`, `vendor_receipt`, `travel_approval`, `ride_payment`,
and `transit_payment`. The legacy `taobao` and `alipay` names remain aliases.
Automatic selection uses both the filename and recognized layout markers.

Generated `runs/`, `objects/`, and `visualizations/` directories are excluded
from recursive discovery. The summary reports `image_path_count` separately
from content-addressed `unique_image_count`, so generated copies and duplicate
source files cannot inflate unique-image coverage.

## Typed job manifests

Use a manifest when filenames are not meaningful or extraction needs business
context and explicit expected fields:

```json
{
  "schema": "hkclr.rapidocr.job-manifest.v1",
  "configuration": {
    "schema": "hkclr.rapidocr.config.v1",
    "minimum_score": 0.5
  },
  "jobs": [
    {
      "schema": "hkclr.rapidocr.job.v1",
      "evidence_id": "example-receipt-001",
      "source_path": "C:\\path\\to\\receipt.png",
      "requested_profile": "vendor_receipt",
      "business_context": {"currency": "HKD"},
      "expected_fields": [
        "receipt_id",
        "paid_date",
        "amount",
        "currency",
        "payment_method"
      ]
    }
  ]
}
```

`source_path` must be absolute and each `evidence_id` must be unique and stable.
Run the manifest with:

```powershell
uv run hkclr-ocr run .\jobs.json --output .\runs\manifest-run
```

For a self-contained local evaluation, copy a batch into `input/<batch-name>`
and point `scan` at that directory. Do not commit the copied evidence.

Useful options:

- `--limit 20`: evaluate a small sample first.
- `--profile`: select `auto`, a registered profile, or a legacy alias.
- `--force`: ignore cached results and rerun OCR.
- `--visualize`: save RapidOCR result overlays for manual review.
- `--min-score 0.50`: omit very low-confidence lines from normalized output.

Each run writes:

- `objects/<cache-key>.ocr.json`: cached result for one evidence job and configuration.
- `ocr-manifest.jsonl`: one record per source image in the current run.
- `ocr-summary.json`: counts, timing, field coverage, and errors.
- `visualizations/`: optional OCR overlays.

Each v2 result distinguishes OCR engine status from profile support, and contains
typed fields (`value`, `confidence`, and `source_box`), transaction rows,
deterministic per-currency candidate totals, adapter warnings, source SHA-256,
and job/configuration/adapter/result schema versions. Unsupported layouts are
reported as `unsupported`; they are never counted as passes.

The cache key includes the source hash, evidence id, all schema versions, adapter
versions, requested profile, expected fields, business context, and minimum score.

## Tests

The tests do not require ONNX Runtime or model initialization.

```powershell
uv run python -m unittest discover -s tests -v
```

## Interpretation

`extracted_fields`, `transactions`, and `candidate_totals` are evidence
extraction, not accounting truth. Values and transaction identifiers should
still be cross-checked before reimbursement automation accepts evidence.
