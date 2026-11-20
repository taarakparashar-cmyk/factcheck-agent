import os
import html
import pandas as pd
import streamlit as st
from checker import run

st.set_page_config(page_title="Truth Layer · PDF Fact-Checker", page_icon="🔎", layout="wide")
st.markdown("""<style>
.block-container{max-width:1100px;padding-top:2rem}
.card{border:1px solid #e4e1d9;border-radius:10px;padding:14px 18px;margin-bottom:12px;background:#fff}
.tag{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.78rem;font-weight:600;color:#fff}
.Verified{background:#1f6f6b}.Inaccurate{background:#c98a0b}.False{background:#c0392b}.Unverifiable{background:#777}
.small{color:#6b7078;font-size:.85rem}
</style>""", unsafe_allow_html=True)
st.markdown("""
<div style="padding:24px 28px;border-radius:18px;background:linear-gradient(110deg,#123b46,#126e69);color:white;margin-bottom:20px">
  <div style="font-size:13px;letter-spacing:2px;font-weight:700;color:#b7f4e7">TRUTH LAYER · OPTIMIZED BUILD v2</div>
  <div style="font-size:38px;font-weight:800;line-height:1.25;margin:8px 0">⚡ Faster PDF Fact-Checking</div>
  <div style="font-size:15px;color:#d4f3ed">Parallel evidence search · Batch AI verification · No artificial claim delays</div>
</div>
""", unsafe_allow_html=True)
st.success("✅ Optimized version loaded — v2 interface is active.")
st.caption("Upload a PDF, choose the number of claims, then select Run fact-check.")

key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")) if hasattr(st, "secrets") else os.getenv("GEMINI_API_KEY", "")
if not key:
    key = st.sidebar.text_input("Gemini API key", type="password", help="Used only for this session.")
f = st.file_uploader("PDF document", type="pdf")
maxc = st.slider("Max claims to check", 5, 25, 10, help="More claims take longer and require more web evidence.")

if f and key and st.button("Run fact-check", type="primary"):
    bar = st.progress(0.0, text="Reading PDF and extracting claims…")
    try:
        def update(done, total):
            bar.progress(done / max(total, 1), text=f"Gathering web evidence for {done} of {total} claims…")
        with st.spinner("Fact-checking claims in one batch…"):
            res = run(key, f, maxc, update)
        st.session_state["res"] = res
        st.session_state["res_file"] = f.name
        bar.progress(1.0, text="Fact-check complete")
    except Exception as e:
        st.error(f"Fact-check failed: {e}")
    finally:
        bar.empty()
elif f and not key:
    st.info("Add a Gemini API key in the sidebar to continue.")

res = st.session_state.get("res")
if res:
    df = pd.DataFrame(res)
    counts = df["verdict"].value_counts()
    n = len(df)
    bad = int(counts.get("Inaccurate", 0) + counts.get("False", 0))
    score = round(100 * counts.get("Verified", 0) / n) if n else 0
    c = st.columns(4)
    c[0].metric("Claims checked", n)
    c[1].metric("Verified", int(counts.get("Verified", 0)))
    c[2].metric("Inaccurate", int(counts.get("Inaccurate", 0)))
    c[3].metric("False", int(counts.get("False", 0)))
    st.subheader(f"Document reliability: {score}%")
    st.progress(score / 100)
    if bad:
        st.warning(f"{bad} claim(s) need correcting before this goes out.")
    verdicts = ["Verified", "Inaccurate", "False", "Unverifiable"]
    pick = st.multiselect("Filter", verdicts, default=verdicts)
    order = {"False": 0, "Inaccurate": 1, "Unverifiable": 2, "Verified": 3}
    for r in sorted(res, key=lambda row: order.get(row.get("verdict", "Unverifiable"), 4)):
        verdict = r.get("verdict", "Unverifiable")
        if verdict not in pick:
            continue
        src = " · ".join(f"[{html.escape(u.split('//')[-1].split('/')[0])}]({u})" for u in r.get("sources", [])[:3])
        st.markdown(f"""<div class="card"><span class="tag {html.escape(verdict)}">{html.escape(verdict)}</span>
<span class="small"> &nbsp;confidence {html.escape(str(r.get('confidence', 0)))}% · page {html.escape(str(r.get('page') or '?'))} · {html.escape(str(r.get('category', '')))}</span>
<p style="margin:.6rem 0 .3rem"><b>Claim:</b> {html.escape(str(r.get('claim', '')))}</p>
<p style="margin:.2rem 0"><b>Real fact:</b> {html.escape(str(r.get('real_fact') or '—'))}</p>
<p class="small" style="margin:.2rem 0">{html.escape(str(r.get('explanation', '')))}</p></div>""", unsafe_allow_html=True)
        if src:
            st.markdown(f"Sources: {src}")
    export_df = df.copy()
    export_df["sources"] = export_df["sources"].map(lambda values: " | ".join(values) if isinstance(values, list) else str(values))
    st.download_button("Download report (CSV)", export_df.to_csv(index=False), "factcheck_report.csv", "text/csv")
