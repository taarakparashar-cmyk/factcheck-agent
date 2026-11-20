"""Truth Layer v5: coverage-audited claim extraction and evidence-grounded checking."""
import datetime
import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from ddgs import DDGS
from google import genai
from pypdf import PdfReader

CHECKER_BUILD = "V5 · COVERAGE AUDIT + ADAPTIVE EVIDENCE"
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")
LOG = logging.getLogger(__name__)
VERDICTS = {"Verified", "Inaccurate", "False", "Unverifiable"}


def read_pdf(file):
    file.seek(0)
    reader = PdfReader(file)
    return [(i + 1, p.extract_text(extraction_mode="plain") or "") for i, p in enumerate(reader.pages)]


def _json(text):
    if not text:
        raise ValueError("Gemini returned an empty response")
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"(\[.*\]|\{.*\})", cleaned, re.S)
        if not m:
            raise ValueError("Gemini returned non-JSON output")
        return json.loads(m.group(1))


def _code(exc):
    return getattr(exc, "code", None) or getattr(exc, "status_code", None)


def _model_list(client):
    names = []
    try:
        for m in client.models.list():
            name = (getattr(m, "name", "") or "").removeprefix("models/")
            if name.startswith("gemini-") and "flash" in name and not any(x in name for x in ("image", "tts", "live")):
                names.append(name)
    except Exception as exc:
        LOG.warning("Model discovery unavailable: %s", exc)
    return sorted(set(names), key=lambda x: ("lite" not in x, "preview" in x, x))


def _ask(client, prompt):
    models = list(dict.fromkeys([PRIMARY_MODEL, FALLBACK_MODEL]))
    last_error = None
    discovered = False
    for model in models:
        for attempt in range(3):
            try:
                r = client.models.generate_content(model=model, contents=prompt,
                    config={"response_mime_type": "application/json", "temperature": 0})
                return _json(r.text)
            except Exception as exc:
                last_error = exc
                code = str(_code(exc) or "")
                message = str(exc).lower()
                if code == "404" or "not_found" in message or "not found" in message:
                    if not discovered:
                        discovered = True
                        models.extend(x for x in _model_list(client) if x not in models)
                    break
                if code in ("429", "503", "500", "502", "504") or any(x in message for x in ("resource_exhausted", "unavailable", "high demand")):
                    if attempt < 2:
                        time.sleep(min(2 * 2**attempt + random.random(), 7))
                        continue
                    break
                if isinstance(exc, ValueError) and attempt < 2:
                    time.sleep(0.5)
                    continue
                raise
    raise RuntimeError(f"Gemini request failed after bounded retries and fallback: {last_error}")


def _sentences(pages):
    """Identify factual candidate lines for coverage auditing, not as a verdict engine."""
    out = []
    for page, text in pages:
        for line in text.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if len(line) < 24:
                continue
            if not (re.search(r"\d", line) or re.search(r"\b(is|was|are|were|has|have|can|founded|released|visible|overtook|capped)\b", line, re.I)):
                continue
            out.append({"page": page, "text": line[:600]})
    return out


EXTRACT_PROMPT = """You are extracting factual claims for independent verification.
Extract EVERY independently testable factual assertion from the document, including assertions without numbers.
Split compound statements into ATOMIC claims: e.g. 'Apple founded in 1976 and Google in 2004' => TWO claims.
Do not silently skip false-sounding claims, myths, or claims with unclear evidence.
Do not infer verdicts. Do not invent claims. Ignore opinions and promotional rhetoric.
Return a JSON array of objects {{\"claim\":\"standalone factual statement\",\"page\":1,\"category\":\"stat|date|technical|historical|other\"}}.
The user's limit is {limit}; return up to {limit} distinct claims, prioritizing document order.
DOCUMENT:\n{document}"""

AUDIT_PROMPT = """You are auditing extraction completeness. Compare the document and the extracted claims.
Find factual statements that are missing, including factual clauses within compound sentences.
Only add statements actually asserted in the document, with page number. Do not paraphrase into a different claim.
Return JSON array of {{\"claim\":\"standalone factual statement\",\"page\":1,\"category\":\"stat|date|technical|historical|other\"}}.
Return [] if all checkable claims are covered. Maximum additional claims: {limit}.
DOCUMENT:\n{document}\nEXTRACTED:\n{claims}"""


def _normalize_claims(items, max_claims, pages):
    result, seen = [], set()
    for c in items if isinstance(items, list) else []:
        if not isinstance(c, dict):
            continue
        claim = re.sub(r"\s+", " ", str(c.get("claim") or "")).strip(" .")
        key = re.sub(r"[^a-z0-9]", "", claim.lower())
        if len(claim) < 12 or key in seen:
            continue
        seen.add(key)
        try:
            page = int(c.get("page") or 1)
        except (TypeError, ValueError):
            page = 1
        result.append({"claim": claim + ".", "page": min(max(page, 1), pages),
                       "category": str(c.get("category") or "other")})
        if len(result) >= max_claims:
            break
    return result


def _hits(raw, n=6):
    result, seen = [], set()
    for h in raw or []:
        url = str(h.get("url") or h.get("href") or "")
        if urlparse(url).scheme not in ("http", "https") or url in seen:
            continue
        seen.add(url)
        result.append({"url": url, "title": str(h.get("title") or "")[:180],
                       "snippet": str(h.get("snippet") or h.get("body") or "")[:650]})
        if len(result) >= n:
            break
    return result


def _search(query):
    try:
        return _hits(DDGS(timeout=12).text(query, max_results=6))
    except Exception as exc:
        LOG.warning("Search failed for %r: %s", query, exc)
        return []


def search_claim(c, deep=False):
    text = re.sub(r"\s+", " ", c["claim"]).strip()
    query = text[:180] if not deep else re.sub(r"[^\w\s.-]", " ", text)[:115]
    hits = _search(query)
    if deep or not hits:
        short = " ".join(text.split()[:12])
        extra = _search(short + " fact check" if deep else short)
        hits = _hits(hits + extra, 9)
    return hits


VERIFY_PROMPT = """You are a skeptical evidence-based fact checker. Date: {today}.
Use ONLY supplied search evidence for evidence-backed verdicts. Search snippets are leads, not guaranteed truth.
A claim is Verified if credible evidence supports its meaning, even if the source supplies a more precise date.
Inaccurate means partially correct, outdated, or materially wrong in a detail.
False means directly contradicted by credible evidence.
Unverifiable means evidence is missing, weak, conflicting, or the search failed. Absence of search results does NOT prove falsehood.
Do not invent URLs or studies. Prefer primary sources. Confidence is 0-100, not 0-5.
For every id return JSON array of objects with id, verdict, confidence, real_fact, explanation, sources (URLs exclusively from evidence).
DATA:\n{data}"""


def _judge(client, claims, evidence, ids):
    payload = []
    for i in ids:
        payload.append({"id": i, "claim": claims[i-1]["claim"], "evidence": evidence.get(i, [])})
    data = _ask(client, VERIFY_PROMPT.format(today=datetime.date.today().isoformat(),
                                            data=json.dumps(payload, ensure_ascii=False)))
    out = {}
    for v in data if isinstance(data, list) else []:
        if not isinstance(v, dict):
            continue
        try:
            i = int(v.get("id"))
        except (ValueError, TypeError):
            continue
        if i in ids:
            out[i] = v
    return out


def run(api_key, file, max_claims=15, progress=lambda done, total: None, return_meta=False):
    """Existing app compatibility: defaults to list[dict]. return_meta=True -> (rows, metrics)."""
    client = genai.Client(api_key=api_key)
    pages = read_pdf(file)
    text_pages = [(p, re.sub(r"\s+", " ", t).strip()) for p, t in pages]
    usable = sum(bool(t) for _, t in text_pages)
    chars = sum(len(t) for _, t in text_pages)
    metrics = {"pages_total": len(pages), "pages_with_text": usable,
               "characters_extracted": chars, "pages_without_text": [p for p, t in text_pages if not t],
               "text_extraction_coverage": round(100 * usable / max(1, len(pages))),
               "claims_detected": 0, "claims_checked": 0, "claim_limit": max_claims,
               "coverage_audit": "completed"}
    if not chars:
        raise ValueError("No extractable text found. This PDF likely needs OCR.")
    if chars > 85000:
        metrics["coverage_audit"] = "partial: document truncated at 85,000 characters"
    doc = "\n".join(f"[Page {p}] {t}" for p, t in pages)[:85000]
    initial = _ask(client, EXTRACT_PROMPT.format(limit=max_claims, document=doc))
    claims = _normalize_claims(initial, max_claims, len(pages))
    if len(claims) < max_claims:
        missing = _ask(client, AUDIT_PROMPT.format(limit=max_claims - len(claims), document=doc,
                       claims=json.dumps(claims, ensure_ascii=False)))
        claims = _normalize_claims(claims + (missing if isinstance(missing, list) else []), max_claims, len(pages))
    metrics["claims_detected"] = len(claims)
    metrics["candidate_lines"] = len(_sentences(pages))
    if not claims:
        return ([], metrics) if return_meta else []
    evidence = {}
    with ThreadPoolExecutor(max_workers=min(3, len(claims))) as pool:
        futures = {pool.submit(search_claim, c): i for i, c in enumerate(claims, 1)}
        for done, future in enumerate(as_completed(futures), 1):
            i = futures[future]
            try:
                evidence[i] = future.result()
            except Exception as exc:
                LOG.warning("Worker failed for claim %s: %s", i, exc)
                evidence[i] = []
            progress(done, len(claims))
    verdicts = {}
    for start in range(1, len(claims) + 1, 5):
        ids = list(range(start, min(start + 5, len(claims) + 1)))
        try:
            verdicts.update(_judge(client, claims, evidence, ids))
        except Exception as exc:
            LOG.warning("Verification batch failed, attempting individual claims: %s", exc)
            for i in ids:
                try:
                    verdicts.update(_judge(client, claims, evidence, [i]))
                except Exception as inner:
                    LOG.warning("Claim %s could not be verified: %s", i, inner)
    retry = [i for i in range(1, len(claims)+1) if not evidence.get(i) or
             i not in verdicts or verdicts[i].get("verdict") == "Unverifiable"]
    if retry:
        with ThreadPoolExecutor(max_workers=min(3, len(retry))) as pool:
            extra = {pool.submit(search_claim, claims[i-1], True): i for i in retry}
            for future in as_completed(extra):
                i = extra[future]
                try:
                    evidence[i] = _hits(evidence.get(i, []) + future.result(), 9)
                except Exception as exc:
                    LOG.warning("Deep search failed for claim %s: %s", i, exc)
        for i in retry:
            if evidence.get(i):
                try:
                    verdicts.update(_judge(client, claims, evidence, [i]))
                except Exception as exc:
                    LOG.warning("Re-verification failed for claim %s: %s", i, exc)
    rows = []
    for i, claim in enumerate(claims, 1):
        v = verdicts.get(i, {})
        hits = evidence.get(i, [])
        allowed = {h["url"] for h in hits}
        cited = [u for u in (v.get("sources") or []) if isinstance(u, str) and u in allowed]
        label = v.get("verdict", "Unverifiable")
        if label not in VERDICTS or not cited:
            label = "Unverifiable"
        try:
            confidence = min(100, max(0, int(v.get("confidence") or 0)))
        except (TypeError, ValueError):
            confidence = 0
        rows.append({**claim, "verdict": label, "confidence": confidence if cited else 0,
                     "real_fact": str(v.get("real_fact") or ""),
                     "explanation": str(v.get("explanation") or ("No supported evidence-based verdict available." if not cited else "")),
                     "sources": cited, "evidence_results": len(hits)})
    metrics["claims_checked"] = len(rows)
    metrics["unverifiable"] = sum(r["verdict"] == "Unverifiable" for r in rows)
    return (rows, metrics) if return_meta else rows
