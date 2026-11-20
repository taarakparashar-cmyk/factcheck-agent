import html
import os
from urllib.parse import urlparse

def urlparse_host(url):
    return urlparse(url).netloc or "source"
import pandas as pd
import streamlit as st
from checker import run, CHECKER_BUILD

st.set_page_config(page_title="Truth Layer | Fact Check", page_icon="🔎", layout="wide")
st.markdown("""<style>.block-container{max-width:1120px;padding-top:1.8rem}.tl-head{padding:24px;border-radius:16px;background:linear-gradient(120deg,#123b46,#176e69);color:#fff}.tl-small{opacity:.8;font-size:14px}.tl-card{border:1px solid #a3a3a344;border-radius:12px;padding:16px;margin:12px 0}.tl-claim{font-weight:650;font-size:17px}</style>""", unsafe_allow_html=True)
st.markdown('<div class="tl-head"><div class="tl-small">TRUTH LAYER</div><h1 style="color:white;margin:8px 0">Fact checker</h1><div class="tl-small">Atomic claims · Coverage audit · Adaptive search · Source-linked verdicts</div></div>', unsafe_allow_html=True)
st.caption(f"Backend: {CHECKER_BUILD}")
try:
    key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
except Exception:
    key = os.getenv("GEMINI_API_KEY", "")
if not key:
    key = st.sidebar.text_input("Gemini API key", type="password")
f = st.file_uploader("Upload a PDF", type="pdf")
maxc = st.slider("Maximum atomic claims to verify", 5, 50, 20, help="An upper limit, not a guaranteed count. Each independent factual assertion counts separately.")
if f and st.button("Analyze document", type="primary", disabled=not bool(key)):
    progress_bar = st.progress(0, text="Extracting document and auditing claim coverage...")
    def update(done, total):
        progress_bar.progress(min(0.9, done/max(total,1)*0.9), text=f"Searching evidence: {done}/{total} claims")
    try:
        with st.spinner("Extracting, checking coverage, retrieving evidence and verifying..."):
            rows, metrics = run(key, f, maxc, update, return_meta=True)
        st.session_state["tl_result"] = (rows, metrics, f.name)
        progress_bar.progress(1.0, text="Complete")
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
    finally:
        progress_bar.empty()

if "tl_result" in st.session_state:
    rows, metrics, filename = st.session_state["tl_result"]
    if f is not None and f.name != filename:
        st.info("Upload changed. Click Analyze document to generate results for this file.")
    else:
        st.subheader("Document extraction")
        a,b,c,d = st.columns(4)
        a.metric("Pages processed", metrics["pages_total"])
        b.metric("Pages with text", metrics["pages_with_text"])
        c.metric("Characters extracted", f'{metrics["characters_extracted"]:,}')
        d.metric("Pages with extractable text", f'{metrics["text_extraction_coverage"]}%')
        st.caption("This percentage measures pages containing extractable text, NOT reading ease or the percentage of words recovered. It cannot prove full extraction completeness.")
        if metrics["pages_without_text"]:
            st.warning("Pages with no selectable text: " + ", ".join(map(str, metrics["pages_without_text"])) + ". OCR may be required.")
        if metrics["coverage_audit"] != "completed":
            st.warning(metrics["coverage_audit"])
        st.subheader("Claim coverage and verdicts")
        counts = pd.Series([r["verdict"] for r in rows]).value_counts() if rows else pd.Series(dtype=int)
        x,y,z,w = st.columns(4)
        x.metric("Atomic claims checked", metrics["claims_checked"])
        y.metric("Verified", int(counts.get("Verified",0)))
        z.metric("Inaccurate / False", int(counts.get("Inaccurate",0)+counts.get("False",0)))
        w.metric("Unverifiable", int(counts.get("Unverifiable",0)))
        st.caption(f'Maximum requested: {metrics["claim_limit"]} · Detected: {metrics["claims_detected"]} · Checked: {metrics["claims_checked"]}. The limit is a ceiling, not a promise of that many claims.')
        st.info("Unverifiable means insufficient evidence or a verification failure; it does not automatically mean the claim is false.")
        verdict_order = {"False":0,"Inaccurate":1,"Unverifiable":2,"Verified":3}
        choices = st.multiselect("Show verdicts", ["False","Inaccurate","Unverifiable","Verified"], default=["False","Inaccurate","Unverifiable","Verified"])
        for r in sorted(rows, key=lambda v: verdict_order.get(v["verdict"], 9)):
            if r["verdict"] not in choices:
                continue
            st.markdown(f'<div class="tl-card"><div class="tl-small">{html.escape(r["verdict"])} · Confidence {r["confidence"]}% · Page {r["page"]} · {html.escape(r["category"])}</div><div class="tl-claim">{html.escape(r["claim"])}</div><p><b>Finding:</b> {html.escape(r["real_fact"] or "Not established")}</p><div>{html.escape(r["explanation"])}</div></div>', unsafe_allow_html=True)
            if r["sources"]:
                st.markdown("Sources: " + " · ".join(f"[{urlparse_host(u)}]({u})" for u in r["sources"][:4]))
            else:
                st.caption("No sufficiently supported source citations available")
        export = pd.DataFrame(rows)
        if not export.empty:
            export["sources"] = export["sources"].map(lambda x: " | ".join(x))
            st.download_button("Download CSV report", export.to_csv(index=False), "factcheck_report.csv", "text/csv")
