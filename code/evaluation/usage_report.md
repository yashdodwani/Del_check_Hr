# Usage report — final full-dataset run

Source: `code/evaluation/run_summary.json`, written by the run of `python code/main.py` that produced the submitted `output.csv`.

* Requests processed: 518
* Intent model configured: `gemini-3.5-flash-lite` (provider: Google Gemini API)
* API key present during the run: True
* Intent routing: {'baseline': 518, 'cache_hits': 0, 'llm': 0, 'llm_failures': 0, 'unresolved': 0}
* OCR: Tesseract (local, no tokens); vision fallback model: `gemini-3.5-flash-lite` — called (see table)

| Provider | Model | Calls | Failed calls | Input tokens | Output tokens | Total tokens | Avg tokens / request | Est. cost (USD) | Est. cost / request (USD) |
|---|---|---|---|---|---|---|---|---|---|
| Google Gemini API | gemini-3.5-flash-lite | 3 | 0 | 3442 | 1373 | 4815 | 9.30 | 0.0009 | 0.000002 |
| **All** | **all models** | 3 | 0 | 3442 | 1373 | 4815 | 9.30 | 0.0009 | 0.000002 |

Cost assumptions: USD 0.1/1M input and USD 0.4/1M output tokens for every model (override with `GHOST_PRICE_IN` / `GHOST_PRICE_OUT`); output tokens include any thinking tokens.
Calls and tokens come from each API's own usage fields (`usage` / `usageMetadata`), recorded in-process.

