# Truth Layer: PDF Fact-Checker

Upload a PDF; the app extracts checkable claims (stats, dates, financial and technical figures), verifies each
against live web search, and labels it **Verified**, **Inaccurate** (outdated or wrong number, with the correct
current fact) or **False** (no credible evidence).

## How it works
1. `pypdf` extracts page text.
2. Gemini (free tier) extracts up to 25 specific claims as JSON.
3. Each claim is searched on the live web (DuckDuckGo, no key) and judged by Gemini against that evidence; it returns verdict, confidence,
   the corrected fact, explanation and source URLs.
4. Streamlit renders a reliability score, per-claim cards (worst first) and a CSV export.

Extras: document reliability score, confidence per claim, source links, verdict filter, CSV report,
per-claim error isolation so one failed lookup never kills a run.

## Run locally
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=AIza...
streamlit run app.py
```

## Deploy (Streamlit Community Cloud, free)
1. Push this folder to a public GitHub repo.
2. share.streamlit.io → New app → select repo, main file `app.py`.
3. Advanced settings → Secrets: `GEMINI_API_KEY = "AIza..."`.
4. Deploy and share the resulting URL.

## Notes and limits
- Free key: aistudio.google.com/apikey (no card). Free tier is rate-limited, so runs take 1-3 minutes.
- Model is set by `MODEL` in `checker.py`.
- Verdicts are model judgements backed by cited sources; review high-stakes claims manually.
