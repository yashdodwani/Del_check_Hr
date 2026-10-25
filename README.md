# Ghost Data Still Exists — deletion-audit agent (solution)

A mostly deterministic audit agent: it reads every participant-facing evidence file, reconstructs each
record's state at `audit_at`, applies `deletion_rules.csv`, and emits the ordered
FINDING → REMEDIATION → VERIFICATION plan. An LLM is used only where language understanding is
required (ambiguous request intent) and as a selective OCR fallback for screenshots/scanned pages. Both use
Google `gemini-3.5-flash-lite` by default (chosen over NVIDIA Nemotron by `code/evaluation/provider_benchmark.md`);
set `GHOST_INTENT_PROVIDER=nvidia` / `GHOST_VISION_PROVIDER=nvidia` to use `nvidia/nemotron-3.5-lightning-30b-a3b` /
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` instead.

## Setup

```bash
# system dependency: Tesseract OCR with English data
sudo apt install tesseract-ocr        # macOS: brew install tesseract
python3 -m venv .venv
.venv/bin/pip install -r code/requirements.txt
cp code/.env.example .env              # optional: add GEMINI_API_KEY (and/or NVIDIA_API_KEY); never commit .env
```

## Run

```bash
.venv/bin/python code/main.py                 # all 518 requests → ./output.csv (+ code/evaluation/run_summary.json)
.venv/bin/python code/main.py --no-llm        # deterministic baseline only, no API calls
.venv/bin/python code/evaluation/main.py      # samples + synthetic + intent benchmark → evaluation reports
.venv/bin/python code/evaluation/provider_benchmark.py  # Gemini vs NVIDIA on intent + vision
.venv/bin/python -m pytest -q code/tests      # regression tests
.venv/bin/python code/synthetic/generate.py   # rebuild the synthetic dataset (deterministic)
.venv/bin/python code/package.py              # build code.zip (README.md + code/, no dataset/caches/secrets)
```

`main.py` validates the written `output.csv` against the contract (header, 518 sorted rows, step
grammar/order, evidence rules, no evidence captured after `audit_at`) and exits non-zero on any violation.

## Architecture

| Stage | Module | What it does |
|---|---|---|
| Evidence parsing | `ghostaudit/evidence/` | 6 log line formats (trace-merged), snapshot bundles, service exports, operational-record JSON, SQLite dumps (primary + every replica table), handover text, PDFs and screenshots → `Observation(resource, system, category, present, identifiers, flow/restore flags, evidence_id)` |
| Extraction | `ghostaudit/extraction/` | PDF: PyMuPDF native text → per-page quality gate → Tesseract on failing pages → optional vision model. Images: Tesseract on preprocessing variants (2x/3x upscale, threshold) → quality gate (confidence, malformed ids, broken emails, unknown emails) → vision model (`gemini-3.5-flash-lite` by default; only with a key and when the provider lists that exact id). Vision text is merged into Tesseract's rows (ids re-spelled from the vision text) because vision models emit one cell per line. Content-hash cache keyed by pipeline/model version; OCR'd emails are snapped to known identifiers. |
| Identity | `ghostaudit/identity.py` | Request email (current or historical) → `user_id` → all emails, phones, hashes (`sha256(lower(email))`), including old values from `user_audit_log.csv`. |
| Linking | `ghostaudit/audit.py`, `ghostaudit/attribution.py` | Records linked by exact identifier matches in any evidence, data-catalog activity, and — for records only ever observed without identifiers — attribution of deletion-worker activity (time window, worker result ↔ reported status, learned timestamp-phase consistency). Records are partitioned between multiple requests of the same person by time. |
| State at `audit_at` | `ghostaudit/audit.py` | Only evidence captured ≤ `audit_at`; latest file wins; copies in one snapshot are merged (a ghost in any replica keeps the record present); full snapshots that omit a record prove absence (never for retained categories). |
| Policy & steps | `ghostaudit/policy.py` | Rule per category (DELETE / REDACT / ANONYMIZE / RETAIN, tombstone keeps only the stable reference, flows must be disabled, backups must be restore-blocked), contract ordering (phase → operation table → system_id → resource_id), evidence ids only on findings and on COMPLETE verifications. |
| Intent | `ghostaudit/intent.py` | Multilingual deterministic baseline; only AMBIGUOUS texts go to `gemini-3.5-flash-lite` (or Nemotron) (strict JSON schema, validation, bounded retries, cache, exact-model availability check). Failures stay `UNRESOLVED` and never authorise deletion. |

Prompts live in `code/ghostaudit/intent.py` (`SYSTEM_PROMPT`) and `code/ghostaudit/extraction/vision.py` (`PROMPT`).
Evaluation results: `code/evaluation/evaluation_report.md`; token/cost report: `code/evaluation/usage_report.md`.

---

# HackerRank Orchestrate - October 2026 (challenge reference)

## Ghost Data Still Exists

Build an AI agent that audits whether a user's personal data was actually deleted.

A deletion workflow may report **Complete** while data still exists in databases, logs, support attachments, exports, search indexes, archives, or backups. Sync and restore processes may also bring deleted data back.

For each request, the agent must first determine whether deletion was actually requested. It then returns `COMPLETE`, `INCOMPLETE`, or `NOT_APPLICABLE`, reports any remaining data or recreation risk, and produces the ordered remediation and verification steps required to finish an authorized deletion safely.

Read [`problem_statement.md`](./problem_statement.md) for the complete challenge contract.

---

## Quick Start

Clone the repository and move into the project directory:

```bash
git clone https://github.com/interviewstreet/hackerrank-orchestrate-october26.git
cd hackerrank-orchestrate-october26
```

Start your AI coding tool from the **repository root** so it automatically reads [`AGENTS.md`](./AGENTS.md) and logs your session turns to `log.txt` in the root directory.

Build your solution in `code/main.py`, or use another language and document its setup and run command clearly in your README.

Your solution must:

- Read the input files from `dataset/`
- Generate one prediction row for every request in `dataset/deletion_requests.csv`
- Write the final predictions to `output.csv` in the repository root

Run the starter Python entry point with:

```bash
python3 code/main.py
```

The blank template at `dataset/output.csv` is provided as a reference. Your final generated file for submission must be the root-level `output.csv`.


## Dataset

All participant-facing files are in `dataset/`:

| File | Contents |
| --- | --- |
| `deletion_requests.csv` | 518 requests |
| `sample_deletion_requests.csv` | 32 sample requests |
| `sample_output.csv` | Correct sample answers |
| `user_identities.csv` | Current user identities |
| `user_audit_log.csv` | Historical identity changes |
| `system_activity.csv` | Activity across services and data processes |
| `deletion_rules.csv` | Required treatment for each data category |
| `evidence_files.csv` | Manifest of supplied evidence files |
| `output.csv` | Blank output template |

Evidence is stored under `dataset/evidence/` and may include PDFs, images, logs, JSON, plain text, or SQL snapshot exports. SQL captures may contain a primary database, regional replicas, a reporting replica, or a failover replica so replication drift can be investigated directly. The `.sql` files are readable as text and executable with SQLite.

Only `deletion_requests.csv` requires answers. The other files provide evidence and context.

---

## Output

Create `output.csv` with these columns in this exact order:

```text
case_id,deletion_status,steps_json
```

Produce exactly one row per request. `deletion_status` is `COMPLETE`, `INCOMPLETE`, or `NOT_APPLICABLE`. `steps_json` is an ordered JSON array containing findings, remediation actions, and verification checks. Use an empty array (`[]`) for `NOT_APPLICABLE`. Evidence references may be a `file_id`, `activity_id`, or `audit_id`. Determine the resource state at `audit_at` from the available evidence; reported workflow statuses and operation results are claims. Use `dataset/sample_output.csv` as the format reference.

The final file must contain 518 data rows sorted by `case_id` (plus the header row). The grader parses `steps_json` as JSON, so object-key order and whitespace do not matter; array order does. Order findings, remediation, and verification in that phase order. Within a phase, follow the operation order in the problem statement, then sort by `system_id` and `resource_id`.

---

## Evaluation

`output.csv` will be compared with hidden canonical answers. Scoring considers:

- deletion assessment accuracy;
- current and historical identity resolution;
- discovery of remaining data and recreation risks;
- correct policy application and evidence references; and
- completeness and ordering of remediation and verification steps.

Your `code.zip` must include:

```text
code/evaluation/evaluation_report.md
code/evaluation/usage_report.md
```

- `evaluation_report.md` must cover case, row, field, and sequence accuracy, at least three additional evaluation cases created by you, failures found, and improvements made.
- `usage_report.md` must cover the final full-dataset run, including model providers and names, model calls, input and output tokens, total and average tokens per request, and estimated total and per-request cost. Include per-model and overall totals when using multiple models.

**Important:** Place `evaluation_report.md` and `usage_report.md` inside `code/evaluation/` (matching `code/evaluation/main.py` in this starter layout), write your complete evaluation metrics, custom evaluation test cases, and token/cost breakdowns directly inside those Markdown files, and keep prompts in source code or Markdown files. Automated code evaluation ignores raw data files (`.csv`, `.tsv`, `.json`, `.jsonl`, `.sql`, `.log`), so content stored only in those formats will not be visible to the code judge or AI interviewer.

---

## Chat Transcript

This repository includes [`AGENTS.md`](./AGENTS.md), which instructs compatible AI coding tools to append conversation summaries to `log.txt` in the repository root:

| Platform | Path |
| --- | --- |
| macOS / Linux | `<repo root>/log.txt` |
| Windows | `<repo root>\log.txt` |

- Always launch your AI coding agent from the repository root (or explicitly ask it to read `./AGENTS.md` first) so every turn is logged to `log.txt` with the required `User Prompt`, `Agent Response Summary`, `Actions`, and `Context` sections.
- Upload `log.txt` directly as your `chat_transcript` submission on HackerRank.
- Do not include API keys, tokens, credentials, or secrets in prompts, code, configuration, or `log.txt`.

---

## Packaging `code.zip` and Submission

| Item | Description |
| --- | --- |
| `code.zip` | Runnable solution, prompts, configuration, `README.md`, and `evaluation/` reports |
| `output.csv` | One answer row for each of the 518 requests in `dataset/deletion_requests.csv` |
| `chat_transcript` | The root `log.txt` file generated by your coding agent |
| AI interview | Live explanation of your architecture, approach, and evaluation |

You may use any programming language, model, framework, or retrieval approach. Include clear setup and execution instructions in `README.md` inside `code.zip`.

To package `code.zip` from the repository root without bundling the dataset:

**macOS / Linux:**

```bash
zip -r code.zip README.md code -x "*/__pycache__/*" "*/.venv/*" "*/venv/*" "*/node_modules/*" "*/.git/*" "*/.DS_Store"
```

**Windows (PowerShell):**

```powershell
Compress-Archive -Path README.md, code -DestinationPath code.zip -Force
```

When packaging `code.zip`:

- Keep your evaluation reports in `code/evaluation/` so they are bundled automatically with `code` (or add `evaluation` to the archive command above if you placed them in a top-level `evaluation/` directory).
- Do **not** include `dataset/`, virtual environments (`.venv`, `venv`, `node_modules`), or build caches (`__pycache__`). On Windows, keep `.venv` at the repository root (outside `code/`) and remove any `code/__pycache__` folders before running `Compress-Archive`.
- Do **not** include nested archives (`.zip`, `.tar`, `.gz`, `.tgz`, `.7z`, `.rar`) inside `code.zip` — submissions containing nested archive files will be rejected.

Before submitting, confirm that:

- `output.csv` has the exact header `case_id,deletion_status,steps_json` and contains all 518 data rows sorted by `case_id`;
- every `steps_json` value is a valid JSON array with consecutive step numbers starting at `1`;
- findings cite valid evidence IDs, and `COMPLETE` verification steps cite evidence proving the current state;
- proposed remediation and future verification steps use an empty `evidence_id` (`""`);
- `code.zip` includes your runnable code, `README.md`, `code/evaluation/evaluation_report.md`, and `code/evaluation/usage_report.md`;
- `log.txt` follows the `AGENTS.md` turn format and is uploaded to the `chat_transcript` slot; and
- no API keys, credentials, real personal data, or sensitive configuration values are included.
