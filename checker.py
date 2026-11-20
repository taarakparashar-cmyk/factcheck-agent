"""Faster Truth Layer pipeline: parallel evidence gathering + batched Gemini verification."""
import json
import re
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


from google import genai
from ddgs import DDGS
from pypdf import PdfReader

CHECKER_BUILD = "OPTIMIZED CHECKER v3 — PARALLEL SEARCH + BATCH VERIFICATION"
MODEL = "gemini-flash-latest"
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
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    return _json(response.text)



def search_claim(item, n=SEARCH_RESULTS_PER_CLAIM):
    """One web query per claim; creates its own DDGS instance for thread safety."""
    claim = re.sub(r"\s+", " ", item.get("claim", "")).strip()[:200]
    if not claim:
        return []
    try:
        hits = list(DDGS().text(claim, max_results=n))
        # Keep only the fields the verifier needs and trim snippets to control prompt size.
        cleaned = []
        for hit in hits[:n]:
            cleaned.append({
                "title": hit.get("title", ""),
                "url": hit.get("href", ""),
                "snippet": (hit.get("body", "") or "")[:350],
            })
        return cleaned
    except Exception:
        return []
