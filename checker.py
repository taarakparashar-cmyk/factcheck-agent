"""Free pipeline: Gemini (free tier) for reasoning + DuckDuckGo for live web evidence."""
import json, re, time, datetime
from concurrent.futures import ThreadPoolExecutor
from google import genai
from ddgs import DDGS
from pypdf import PdfReader

MODEL = "gemini-3.8-flash" # alias to the current free Flash model; change if Google renames it
TODAY = datetime.date.today().isoformat()

EXTRACT_PROMPT = """You are a meticulous fact-checking analyst. From the document, extract checkable factual
claims: statistics, percentages, dates, rankings, prices, financial figures, technical specs.
Skip opinions and vague statements. Return ONLY a JSON array of objects:
{{"claim": "<verbatim sentence>", "category": "stat|date|financial|technical|other", "page": <int or null>}}
Most specific claims first. Maximum {n} claims.

DOCUMENT:
{doc}"""

VERIFY_PROMPT = """Today is {today}. Fact-check the CLAIM using ONLY the web search EVIDENCE below plus your
own knowledge of well-established facts. Figures that were true in the past but have since changed are Inaccurate.

CLAIM: {claim}

EVIDENCE (title | url | snippet):
{evidence}

Return ONLY a JSON object:
{{"verdict": "Verified|Inaccurate|False",
 "confidence": <0-100>,
 "real_fact": "<the correct, current fact in one sentence; if Verified, restate the confirmed fact>",
 "explanation": "<one or two sentences on what differs and why>",
 "sources": ["<urls taken from the evidence above only>"]}}
Verified = matches current data. Inaccurate = partly right, outdated, or wrong number/date (give the right one).
False = no credible evidence it is true, or it is contradicted outright. Never invent URLs."""


def read_pdf(file):
    return [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(file).pages)]


def _json(text):
    m = re.search(r"(\[.*\]|\{.*\})", text or "", re.S)
    return json.loads(m.group(1)) if m else None


MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite"
]


def _ask(client, prompt, retries=3):
    """Try each free model in turn; thinking off for speed; short backoff on rate limits."""
    last = None
    for model in MODELS:
        cfg = {"response_mime_type": "application/json", "thinking_config": {"thinking_budget": 0}}
        for i in range(retries):
            try:
                r = client.models.generate_content(model=model, contents=prompt, config=cfg)
                return _json(r.text)
            except Exception as e:
                last = e
                code = getattr(e, "code", None)
                if code == 400 and "thinking" in str(e).lower() and "thinking_config" in cfg:
                    cfg = {"response_mime_type": "application/json"}; continue  # model has no thinking switch
                if code in (429, 503) and i < retries - 1:
                    time.sleep(4 * (i + 1)); continue
                break
    raise RuntimeError(f"Gemini request failed on all models. Last error: {last}")


def search(claim, n=6):
    seen, out = set(), []
    q = re.sub(r"\s+", " ", claim)[:200]
    for query in (q,):
        try:
            for h in DDGS(timeout=5).text(query, max_results=n):
                if h["href"] not in seen:
                    seen.add(h["href"]); out.append(h)
        except Exception:
            pass
    return out[: n * 2]


def verify_claim(client, item):
    hits = search(item["claim"])
    ev = "\n".join(f"{h['title']} | {h['href']} | {h['body'][:300]}" for h in hits) or "NO RESULTS FOUND"
    try:
        out = _ask(client, VERIFY_PROMPT.format(today=TODAY, claim=item["claim"], evidence=ev)) or {}
    except Exception as e:
        out = {"verdict": "False", "confidence": 0, "explanation": f"Verification error: {e}"}
    return {**item, "verdict": out.get("verdict", "False"), "confidence": out.get("confidence", 0),
            "real_fact": out.get("real_fact", ""), "explanation": out.get("explanation", ""),
            "sources": out.get("sources") or [h["href"] for h in hits[:2]]}


BATCH_PROMPT = """Today is {today}. Fact-check EACH claim below (a claim may be a whole sentence; if ANY part of it is wrong, flag it and state the correct fact for that part) using ONLY its web EVIDENCE plus well-established
knowledge. Figures true in the past but since changed are Inaccurate.

{items}

Return ONLY a JSON array with one object per claim, in the same order:
{{"id": <int>, "verdict": "Verified|Inaccurate|False", "confidence": <0-100>,
 "real_fact": "<correct current fact, one sentence>", "explanation": "<one short sentence>",
 "sources": ["<urls from that claim's evidence only>"]}}
Verified = matches current data. Inaccurate = outdated or wrong number/date (give the right one).
False = no credible evidence, or contradicted outright. Never invent URLs."""


def verify_batch(client, batch):
    """batch: list of (id, claim_item, hits). One model call judges the whole batch."""
    items = "\n\n".join(
        f"[{i}] CLAIM: {c['claim']}\nEVIDENCE:\n" +
        ("\n".join(f"- {h['title']} | {h['href']} | {h['body'][:220]}" for h in hits[:6]) or "NO RESULTS FOUND")
        for i, c, hits in batch)
    try:
        out = _ask(client, BATCH_PROMPT.format(today=TODAY, items=items)) or []
    except Exception as e:
        out = [{"id": i, "verdict": "False", "confidence": 0, "explanation": f"Verification error: {e}"} for i, _, _ in batch]
    by_id = {o.get("id"): o for o in out if isinstance(o, dict)}
    res = []
    for i, c, hits in batch:
        o = by_id.get(i, {})
        res.append({**c, "verdict": o.get("verdict", "False"), "confidence": o.get("confidence", 0),
                    "real_fact": o.get("real_fact", ""), "explanation": o.get("explanation", ""),
                    "sources": o.get("sources") or [h["href"] for h in hits[:2]]})
    return res


NUM = re.compile(r"\d|\b(million|billion|trillion|percent|majority|half|doubled|tripled)\b", re.I)
CAT = [("financial", r"[$€£₹]|revenue|profit|valuation|billion|million|price"), ("date", r"\b(1[5-9]|20)\d{2}\b|founded|released|launched"),
       ("stat", r"%|percent|\d+x|\bstudy|survey|report")]


def extract_fast(pages, n):
    """Instant, no-AI claim finder: sentences carrying numbers, dates or figures, most numeric first."""
    seen, found = set(), []
    for pg, text in pages:
        flat = re.sub(r"\s+", " ", text)
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", flat):
            s = s.strip()
            if 25 <= len(s) <= 400 and NUM.search(s) and s.lower() not in seen:
                seen.add(s.lower())
                cat = next((c for c, rx in CAT if re.search(rx, s, re.I)), "other")
                found.append({"claim": s, "category": cat, "page": pg, "_w": len(re.findall(r"\d+", s))})
    found.sort(key=lambda c: -c["_w"])
    for c in found: c.pop("_w")
    found = found[:n]
    if len(found) < n:  # fill spare slots with number-free factual statements (e.g. myths, "is visible from...")
        VERB = re.compile(r"\b(is|are|was|were|has|have|had|can|cannot|became|visible|causes?|invented|discovered)\b", re.I)
        for pg, text in pages:
            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", re.sub(r"\s+", " ", text)):
                s = s.strip()
                if 40 <= len(s) <= 300 and VERB.search(s) and s.lower() not in seen and len(found) < n:
                    seen.add(s.lower()); found.append({"claim": s, "category": "other", "page": pg})
    return found


def run(api_key, file, max_claims=10, progress=lambda d, t: None, log=lambda m: None):
    t0 = time.time()
    client = genai.Client(api_key=api_key)
    pages = read_pdf(file)
    claims = extract_fast(pages, max_claims)
    if len(claims) < 3:  # sparse or unusual document: fall back to AI extraction
        doc = "\n\n".join(f"[Page {n}]\n{t}" for n, t in pages)[:40000]
        claims = (_ask(client, EXTRACT_PROMPT.format(n=max_claims, doc=doc)) or [])[:max_claims]
    total = len(claims)
    log(f"1/3 Found {total} claims in {time.time()-t0:.1f}s"); t1 = time.time()
    ex = ThreadPoolExecutor(max_workers=10)                 # all searches at once, hard 12s cap
    futs = [ex.submit(search, c["claim"]) for c in claims]
    hits = []
    for f in futs:
        try: hits.append(f.result(timeout=max(0.5, 12 - (time.time() - t1))))
        except Exception: hits.append([])
    ex.shutdown(wait=False)
    log(f"2/3 Web search done in {time.time()-t1:.1f}s"); t2 = time.time()
    jobs = [[(i, claims[i], hits[i]) for i in range(s, min(s + 10, total))] for s in range(0, total, 10)]
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=2) as ex2:
        for r in ex2.map(lambda b: verify_batch(client, b), jobs):
            results += r; done += len(r); progress(done, total)
    log(f"3/3 Verdicts in {time.time()-t2:.1f}s · total {time.time()-t0:.1f}s")
    return results
