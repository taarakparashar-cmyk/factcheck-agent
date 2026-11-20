"""Faster Truth Layer pipeline: parallel evidence gathering + batched Gemini verification."""
import json
import os
import re
import datetime
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


from google import genai
from ddgs import DDGS
from pypdf import PdfReader

CHECKER_BUILD = "QUALITY CHECKER v4 — ADAPTIVE EVIDENCE RETRIEVAL"
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
FALLBACK_MODEL = "gemini-2.5-flash"
TODAY = datetime.date.today().isoformat()
MAX_SEARCH_WORKERS = 3
SEARCH_RESULTS_PER_CLAIM = 5
LOG = logging.getLogger(__name__)



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
def _clean_hits(raw, limit=5):
    seen, cleaned = set(), []
    for hit in raw or []:
        url = hit.get("href") or hit.get("url") or ""
        if not url.startswith(("https://", "http://")) or url in seen:
            continue
        seen.add(url)
        cleaned.append({"title": hit.get("title", ""), "url": url,
                        "snippet": (hit.get("body", "") or "")[:700]})
        if len(cleaned) >= limit:
            break
    return cleaned


def _search(query, n=5):
    last = None
    for attempt in range(2):
        try:
            return _clean_hits(list(DDGS(timeout=12).text(query, max_results=n)), n)
        except Exception as exc:
            last = exc
            LOG.warning("Search attempt %s failed for %r: %s", attempt + 1, query[:90], exc)
            if attempt == 0:
                time.sleep(0.7)
    return []


def search_claim(item, n=SEARCH_RESULTS_PER_CLAIM, deep=False):
    claim = re.sub(r"\s+", " ", item.get("claim", "")).strip()[:250]
    if not claim:
        return []
    # The first query is fast. A second, broader query is used only if needed.
    hits = _search(claim, n)
    if deep or not hits:
        simplified = re.sub(r"[^\w\s.-]", " ", claim)
        simplified = re.sub(r"\s+", " ", simplified).strip()[:130]
        if simplified and simplified != claim:
            extra = _search(simplified, n)
        else:
            extra = _search('"' + claim[:110] + '"', n)
        hits = _clean_hits([{"href": h["url"], "title": h["title"], "body": h["snippet"]}
                            for h in hits + extra], n + 3)
    return hits


def _judge(client, claims, evidence, today, ids):
    batch = []
    for i in ids:
        hits = evidence.get(i - 1, [])
        ev = "\n".join(f"{h['title']} | {h['url']} | {h['snippet']}" for h in hits)
        batch.append({"id": i, "claim": claims[i - 1],
                      "evidence": ev or "NO SEARCH EVIDENCE FOUND"})
    judged = _ask(client, VERIFY_BATCH_PROMPT.format(
        today=today, items=json.dumps(batch, ensure_ascii=False)))
    if not isinstance(judged, list):
        return {}
    return {int(v["id"]): v for v in judged
            if isinstance(v, dict) and str(v.get("id", "")).isdigit()}


def run(api_key, file, max_claims=15, progress=lambda d, t: None):
    client = genai.Client(api_key=api_key)
    pages = read_pdf(file)
    doc = "\n\n".join(f"[Page {page_no}]\n{text}" for page_no, text in pages)[:45000]
    if not doc.strip():
        raise ValueError("No selectable PDF text found. A scanned PDF requires OCR.")
    claims = _ask(client, EXTRACT_PROMPT.format(n=max_claims, doc=doc)) or []
    if not isinstance(claims, list):
        raise ValueError("Claim extraction did not return a JSON array")
    claims = [c for c in claims if isinstance(c, dict) and c.get("claim")][:max_claims]
    if not claims:
        return []

    evidence = {}
    with ThreadPoolExecutor(max_workers=min(MAX_SEARCH_WORKERS, len(claims))) as pool:
        futures = {pool.submit(search_claim, c): i for i, c in enumerate(claims)}
        for completed, future in enumerate(as_completed(futures), 1):
            idx = futures[future]
            try:
                evidence[idx] = future.result()
            except Exception as exc:
                LOG.exception("Search worker failed for claim %s: %s", idx + 1, exc)
                evidence[idx] = []
            progress(completed, len(claims))

    # Small batches reduce omitted/mismatched claim IDs in the model response.
    verdicts = {}
    for start in range(1, len(claims) + 1, 5):
        ids = list(range(start, min(start + 5, len(claims) + 1)))
        verdicts.update(_judge(client, claims, evidence, TODAY, ids))

    # Retry only missing, unsupported or inconclusive claims; do not repeat good work.
    retry_ids = [i for i in range(1, len(claims) + 1)
                 if not evidence.get(i - 1) or
                 i not in verdicts or
                 verdicts[i].get("verdict") == "Unverifiable"]
    for i in retry_ids:
        additional = search_claim(claims[i - 1], deep=True)
        old_count = len(evidence.get(i - 1, []))
        combined = evidence.get(i - 1, []) + additional
        evidence[i - 1] = _clean_hits([
            {"href": h["url"], "title": h["title"], "body": h["snippet"]}
            for h in combined], 8)
        # Only reassess if new evidence was actually found or the response was missing.
        if evidence[i - 1] and (len(evidence[i - 1]) > old_count
                                or i not in verdicts):
            try:
                revised = _judge(client, claims, evidence, TODAY, [i]).get(i)
                if revised:
                    verdicts[i] = revised
            except Exception as exc:
                LOG.warning("Second-pass verification failed for claim %s: %s", i, exc)

    results = []
    for i, claim in enumerate(claims, 1):
        verdict = verdicts.get(i, {})
        hits = evidence.get(i - 1, [])
        allowed = {h["url"] for h in hits}
        cited = [u for u in (verdict.get("sources") or []) if isinstance(u, str) and u in allowed]
        sources = cited or [h["url"] for h in hits[:3]]
        label = verdict.get("verdict", "Unverifiable")
        if label not in ("Verified", "Inaccurate", "False", "Unverifiable"):
            label = "Unverifiable"
        # A source-less claim cannot receive an evidence-backed decisive verdict.
        if not hits:
