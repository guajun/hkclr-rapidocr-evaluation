# HKCLR RapidOCR Evaluation

Local, read-only OCR evaluation harness for reimbursement screenshots. It scans
images recursively, runs RapidOCR, stores normalized JSON results by image hash,
and extracts a small set of deterministic reimbursement field candidates.

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

For a self-contained local evaluation, copy a batch into `input/<batch-name>`
and point `scan` at that directory. Do not commit the copied evidence.

Useful options:

- `--limit 20`: evaluate a small sample first.
- `--profile auto|taobao|alipay|generic`: choose deterministic field checks.
- `--force`: ignore cached results and rerun OCR.
- `--visualize`: save RapidOCR result overlays for manual review.
- `--min-score 0.50`: omit very low-confidence lines from normalized output.

Each run writes:

- `objects/<sha256>.ocr.json`: cached normalized result for one unique image.
- `ocr-manifest.jsonl`: one record per source image in the current run.
- `ocr-summary.json`: counts, timing, field coverage, and errors.
- `visualizations/`: optional OCR overlays.

The cache key includes the image hash, OCR adapter version, profile, and minimum
score. Moving an unchanged image does not require another inference run.

## Tests

The tests do not require ONNX Runtime or model initialization.

```powershell
uv run python -m unittest discover -s tests -v
```

## Interpretation

`extracted_fields` contains candidates, not accounting truth. A result only
passes a profile check when its required labels were recognized. Amounts and
transaction identifiers should still be cross-checked against the reimbursement
manifest before automation accepts evidence.
