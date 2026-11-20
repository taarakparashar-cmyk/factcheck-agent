"""Free pipeline: Gemini (free tier) for reasoning + DuckDuckGo for live web evidence."""
import json, re, time, datetime
from google import genai
from ddgs import DDGS
from pypdf import PdfReader

MODEL = "gemini-3.8-flash"  # alias to the current free Flash model; change if Google renames it
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


MODELS = [MODEL, "gemini-3.8-flash","gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]


def _ask(client, prompt, retries=3):
    """Try each free model in turn; back off on rate limits; surface readable errors."""
    last = None
    for model in MODELS:
        for i in range(retries):
            try:
                r = client.models.generate_content(model=model, contents=prompt,
                                                   config={"response_mime_type": "application/json"})
                return _json(r.text)
            except Exception as e:
                last = e
                code = getattr(e, "code", None)
                if code in (429, 503) and i < retries - 1:
                    time.sleep(10 * (i + 1)); continue  # free-tier rate limit: wait, retry
                break  # 400/401/403/404 or retries used up: move on to the next model
    raise RuntimeError(f"Gemini request failed on all models. Last error: {last}")


def search(claim, n=5):
    seen, out = set(), []
    q = re.sub(r"\s+", " ", claim)[:200]
    for query in (q, q + " latest " + TODAY[:4]):
        try:
            for h in DDGS().text(query, max_results=n):
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


def run(api_key, file, max_claims=15, progress=lambda d, t: None):
    client = genai.Client(api_key=api_key)
    pages = read_pdf(file)
    doc = "\n\n".join(f"[Page {n}]\n{t}" for n, t in pages)[:60000]
    claims = (_ask(client, EXTRACT_PROMPT.format(n=max_claims, doc=doc)) or [])[:max_claims]
    results = []
    for i, c in enumerate(claims, 1):
        results.append(verify_claim(client, c)); progress(i, len(claims))
        time.sleep(4)  # stay under free-tier requests-per-minute
    return results
