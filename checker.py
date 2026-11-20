"""Faster Truth Layer pipeline: parallel evidence gathering + batched Gemini verification."""
import json
import re
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


from google import genai
from ddgs import DDGS
from pypdf import PdfReader

CHECKER_BUILD = "OPTIMIZED CHECKER v3 — PARALLEL SEARCH + BATCH VERIFICATION"
MODEL = "gemini-3.5-flash-lite"
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



def run(api_key, file, max_claims=15, progress=lambda d, t: None):
    client = genai.Client(api_key=api_key)


    # 1) PDF text extraction
    pages = read_pdf(file)
    doc = "\n\n".join(f"[Page {page_no}]\n{text}" for page_no, text in pages)
    doc = doc[:45000]  # reduce extraction prompt size versus the previous 60k-character cap


    # 2) Extract claims in one Gemini request
    claims = _ask(client, EXTRACT_PROMPT.format(n=max_claims, doc=doc)) or []
    claims = claims[:max_claims]
    if not claims:
        return []


    # 3) Gather web evidence concurrently; don't make Streamlit UI calls from worker threads.
    evidence_by_index = {}
    with ThreadPoolExecutor(max_workers=min(MAX_SEARCH_WORKERS, len(claims))) as pool:
        futures = {pool.submit(search_claim, claim): i for i, claim in enumerate(claims)}
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                evidence_by_index[idx] = future.result()
            except Exception:
                evidence_by_index[idx] = []
            completed += 1
            progress(completed, len(claims))


    # 4) One batched Gemini call to judge all claims against their own evidence.
    batch = []
    for i, claim in enumerate(claims, start=1):
        hits = evidence_by_index.get(i - 1, [])
        evidence_text = "\n".join(
            f"{h['title']} | {h['url']} | {h['snippet']}" for h in hits
        ) or "NO SEARCH EVIDENCE FOUND"
        batch.append({"id": i, "claim": claim, "evidence": evidence_text})

    judged = _ask(client, VERIFY_BATCH_PROMPT.format(today=TODAY, items=json.dumps(batch, ensure_ascii=False)))
    judged = judged if isinstance(judged, list) else []
    judged_by_id = {item.get("id"): item for item in judged if isinstance(item, dict)}


    # 5) Merge results with stable claim order and safe defaults.
    results = []
    for i, claim in enumerate(claims, start=1):
        verdict = judged_by_id.get(i, {})
        hits = evidence_by_index.get(i - 1, [])
        sources = verdict.get("sources") or [h["url"] for h in hits if h.get("url")][:3]
        results.append({
            **claim,
            "verdict": verdict.get("verdict", "Unverifiable"),
            "confidence": verdict.get("confidence", 0),
            "real_fact": verdict.get("real_fact", ""),
            "explanation": verdict.get("explanation", "Verification response missing or invalid."),
            "sources": sources,
        })
    return results
