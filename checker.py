"""Faster Truth Layer pipeline: parallel evidence gathering + batched Gemini verification."""
import json
import os
import re
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


from google import genai
from ddgs import DDGS
from pypdf import PdfReader

CHECKER_BUILD = "OPTIMIZED CHECKER v3 — PARALLEL SEARCH + BATCH VERIFICATION"
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
FALLBACK_MODEL = "gemini-2.5-flash"
TODAY = datetime.date.today().isoformat()
MAX_SEARCH_WORKERS = 5
SEARCH_RESULTS_PER_CLAIM = 4



EXTRACT_PROMPT = """Extract independently checkable factual claims from this document: statistics, dates,
rankings, prices, financial figures, technical specifications, and concrete factual assertions.
Skip opinions, vague statements, and duplicates. Prefer claims that matter most.
Return ONLY a JSON array, maximum {n} items, each:
{{"claim":"verbatim claim or faithful standalone sentence","category":"stat|date|financial|technical|other","page":1}}

DOCUMENT:
{doc}"""
VERIFY_BATCH_PROMPT = """You are a careful fact-checking analyst. Today's date is {today}.
Check each numbered claim against the supplied web evidence. Treat evidence quality and
source credibility critically. Do not invent sources or URLs. Distinguish outdated facts
from false claims and insufficient evidence.

For EACH claim, return an object with exactly these fields:
{{"id": 1, "verdict":"Verified|Inaccurate|False|Unverifiable", "confidence":0,
"real_fact":"one-sentence current/correct fact or best available finding",
"explanation":"brief reason for the verdict", "sources":["URL from supplied evidence"]}}
Use Verified only when evidence supports the claim; Inaccurate when it is partly correct,
outdated, or materially incorrect; False when reliable evidence contradicts it; Unverifiable
when evidence is insufficient. Return ONLY a JSON array, preserving each claim's id.

CLAIMS AND EVIDENCE:
{items}"""


def read_pdf(file):
    reader = PdfReader(file)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]



def _json(text):
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", text, re.S)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
    return None



def _ask(client, prompt):
    """Single model request; errors are surfaced instead of silently mislabeling claims."""
    last_error = None
    for model in dict.fromkeys([MODEL, FALLBACK_MODEL]):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            parsed = _json(response.text)
            if parsed is None:
                raise ValueError(f"{model} returned invalid JSON")
            return parsed
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            # Fall back only for model availability errors, not invalid keys, quotas, or other failures.
            if not ("404" in message or "not_found" in message or "not found" in message):
                raise
    raise RuntimeError(
        f"No configured Gemini model was available. Tried {MODEL} and {FALLBACK_MODEL}. "
        f"Last error: {last_error}. Check your available models using client.models.list()."
    )


