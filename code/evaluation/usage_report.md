# Usage report — final full-dataset run

Source: `code/evaluation/run_summary.json`, written by the run of `python code/main.py` that produced the submitted `output.csv`.

* Requests processed: 518
* Intent model configured: `nvidia/nemotron-3.5-lightning-30b-a3b` (provider: NVIDIA API, `integrate.api.nvidia.com`)
* API key present during the run: False
* Intent routing: {'baseline': 518, 'cache_hits': 0, 'llm': 0, 'llm_failures': 0, 'unresolved': 0}
* OCR: Tesseract (local, no tokens); vision fallback model: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (NVIDIA API) — configured but not called in this run (requires NVIDIA_API_KEY and the model being served)

| Provider | Model | Calls | Failed calls | Input tokens | Output tokens | Total tokens | Avg tokens / request | Est. cost (USD) | Est. cost / request (USD) |
|---|---|---|---|---|---|---|---|---|---|
| **All** | **all models** | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.0000 | 0.000000 |

Cost assumptions: USD 0.1/1M input and USD 0.4/1M output tokens (override with `GHOST_PRICE_IN` / `GHOST_PRICE_OUT`).

**No model calls were made in the final run.** `NVIDIA_API_KEY` was not available in the environment, and the deterministic baseline classified every request with high confidence (no AMBIGUOUS texts), so no request required the LLM and none was left unresolved. Token counts and cost are therefore zero; they are recorded from the API's `usage` field automatically when the key is configured.
