import os
import pandas as pd
import streamlit as st
from checker import run

st.set_page_config(page_title="Truth Layer · PDF Fact-Checker", page_icon="🔎", layout="wide")
st.markdown("""<style>
.block-container{max-width:1100px;padding-top:2rem}
.card{border:1px solid #e4e1d9;border-radius:10px;padding:14px 18px;margin-bottom:12px;background:#fff}
.tag{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.78rem;font-weight:600;color:#fff}
.Verified{background:#1f6f6b}.Inaccurate{background:#c98a0b}.False{background:#c0392b}
.small{color:#6b7078;font-size:.85rem}
</style>""", unsafe_allow_html=True)

st.title("Truth Layer")
st.caption("Upload a PDF. Every stat, date and figure is pulled out and checked against the live web.")

key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")) if hasattr(st, "secrets") else os.getenv("GEMINI_API_KEY", "")
if not key:
    key = st.sidebar.text_input("Gemini API key (free)", type="password", help="Only used for this session.")

f = st.file_uploader("PDF document", type="pdf")
maxc = st.slider("Max claims to check", 5, 25, 15, help="Free tier is rate-limited; fewer claims = faster.")
if f and key and st.button("Run fact-check", type="primary"):
    bar = st.progress(0.0, text="Extracting claims…")
    res = run(key, f, maxc, lambda d, t: bar.progress(d / max(t, 1), text=f"Verified {d} of {t} claims"))
    bar.empty()
    st.session_state["res"] = res
elif f and not key:
    st.info("Add an API key in the sidebar to continue.")

res = st.session_state.get("res")
if res:
    df = pd.DataFrame(res)
    counts = df["verdict"].value_counts()
    n = len(df); bad = int(counts.get("Inaccurate", 0) + counts.get("False", 0))
    score = round(100 * counts.get("Verified", 0) / n) if n else 0
    c = st.columns(4)
    c[0].metric("Claims checked", n); c[1].metric("Verified", int(counts.get("Verified", 0)))
    c[2].metric("Inaccurate", int(counts.get("Inaccurate", 0))); c[3].metric("False", int(counts.get("False", 0)))
    st.subheader(f"Document reliability: {score}%")
    st.progress(score / 100)
    if bad: st.warning(f"{bad} claim(s) need correcting before this goes out.")

    pick = st.multiselect("Filter", ["Verified", "Inaccurate", "False"], default=["Inaccurate", "False", "Verified"])
    order = {"False": 0, "Inaccurate": 1, "Verified": 2}
    for r in sorted(res, key=lambda r: order.get(r["verdict"], 3)):
        if r["verdict"] not in pick: continue
        src = " · ".join(f"[{u.split('//')[-1].split('/')[0]}]({u})" for u in r["sources"][:3])
        st.markdown(f"""<div class="card"><span class="tag {r['verdict']}">{r['verdict']}</span>
<span class="small"> &nbsp;confidence {r['confidence']}% · page {r.get('page') or '?'} · {r.get('category','')}</span>
<p style="margin:.6rem 0 .3rem"><b>Claim:</b> {r['claim']}</p>
<p style="margin:.2rem 0"><b>Real fact:</b> {r['real_fact'] or '—'}</p>
<p class="small" style="margin:.2rem 0">{r['explanation']}</p></div>""", unsafe_allow_html=True)
        if src: st.markdown(f"Sources: {src}")

    st.download_button("Download report (CSV)", df.assign(sources=df["sources"].map(" | ".join)).to_csv(index=False),
                       "factcheck_report.csv", "text/csv")
