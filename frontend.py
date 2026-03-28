"""
Two panels:
  Left PDF upload + ingestion status
  RightChat interface with streaming feel

Design: dark space theme, monospaced accents, NASA aesthetic.
"""
from __future__ import annotations
import os
import time
import requests
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DOCNAVI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")
API = f"{BACKEND}/api/v1"

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');

/* ── FIXED READABLE DARK THEME ── */
:root {
    --bg-deep:    #0b0f17;
    --bg-panel:   #111827;
    --bg-card:    #1f2937;

    --accent-1:   #3b82f6;   /* blue */
    --accent-2:   #f59e0b;   /* amber */
    --accent-3:   #22c55e;   /* green */

    --text-main:  #f9fafb;   /* strong white */
    --text-muted: #cbd5e1;   /* readable gray */

    --border:     #374151;

    --user-bg:    #1e3a8a;
    --bot-bg:     #111827;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: var(--bg-deep) !important;
    color: var(--text-main) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border);
}

/* Headings */
h1, h2, h3 {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--accent-1) !important;
}

/* File uploader */
[data-testid="stFileUploadDropzone"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--accent-1) !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Share Tech Mono', monospace !important;
    background: transparent !important;
    border: 1px solid var(--accent-1) !important;
    color: var(--accent-1) !important;
}
.stButton > button:hover {
    background: var(--accent-1) !important;
    color: #ffffff !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: #111827 !important;
    border: 1px solid var(--border) !important;
    color: var(--text-main) !important;
    caret-color: var(--accent-1) !important;  /* blue blinking cursor */
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent-1) !important;
    caret-color: var(--accent-1) !important;
    outline: none !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
}

/* Force all text inside chat messages to be visible */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] ol,
[data-testid="stChatMessage"] ul,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em,
[data-testid="stChatMessage"] code {
    color: var(--text-main) !important;
}

/* Markdown rendered content */
.stMarkdown p,
.stMarkdown span,
.stMarkdown li {
    color: var(--text-main) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent-3) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
}

/* Divider */
hr {
    border-color: var(--border) !important;
}

/* Progress */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2)) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: var(--border); }

/* Citation pills */
.citation-pill {
    background: rgba(59,130,246,0.15);
    border: 1px solid var(--accent-1);
    color: var(--accent-1);
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 20px;
}

.page-pill {
    background: rgba(245,158,11,0.15);
    border: 1px solid var(--accent-2);
    color: var(--accent-2);
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 20px;
}

/* Confidence bar */
.confidence-bar {
    height: 4px;
    border-radius: 2px;
}

/* Banner */
.top-banner {
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    padding: 0.6rem 1.2rem;
}
.banner-title {
    color: var(--accent-1);
}
.banner-sub {
    color: var(--text-muted);
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state init ─────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "messages": [],
        "ingestion_done": False,
        "ingestion_stats": {},
        "collection_stats": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Helper functions ───────────────────────────────────────────────────────────

def _check_health() -> dict:
    try:
        r = requests.get(f"{API}/health", timeout=4)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def _fetch_collections() -> list[dict]:
    try:
        r = requests.get(f"{API}/collections", timeout=4)
        if r.status_code == 200:
            return r.json().get("collections", [])
    except Exception:
        pass
    return []


def _ingest_pdf(uploaded_file) -> dict:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    r = requests.post(f"{API}/ingest", files=files, timeout=300)
    r.raise_for_status()
    return r.json()


def _query(question: str) -> dict:
    payload = {
        "query": question,
    }
    r = requests.post(f"{API}/query", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def _render_citation_pills(sections: list[str], pages: list[int]) -> str:
    parts = []
    for s in sections:
        parts.append(f'<span class="citation-pill">§ {s}</span>')
    for p in pages:
        parts.append(f'<span class="page-pill">p. {p}</span>')
    return " ".join(parts)


# ── Sidebar – Upload panel ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛰 DOCUMENT UPLINK")
    st.markdown(
        "<p style='color:#6a84b0;font-size:0.82rem;'>Upload a PDF to index into Qdrant.<br>"
        "Supports multi-section, images & tables.</p>",
        unsafe_allow_html=True,
    )

    # Health indicator
    health = _check_health()
    if health.get("status") == "ok":
        qdrant_ok = health.get("qdrant") == "ok"
        bm25_ok = health.get("bm25_ready", False)
        st.markdown(
            f"""
            <div style='display:flex;gap:8px;align-items:center;margin-bottom:12px;'>
                <span style='width:8px;height:8px;border-radius:50%;
                    background:{"#7fff7f" if qdrant_ok else "#ff4444"};
                    box-shadow:0 0 6px {"#7fff7f" if qdrant_ok else "#ff4444"};
                    display:inline-block;'></span>
                <span style='font-size:0.78rem;color:#6a84b0;font-family:monospace;'>
                    QDRANT {"ONLINE" if qdrant_ok else "OFFLINE"}
                </span>
                <span style='width:8px;height:8px;border-radius:50%;
                    background:{"#7fff7f" if bm25_ok else "#ff9900"};
                    box-shadow:0 0 6px {"#7fff7f" if bm25_ok else "#ff9900"};
                    display:inline-block;margin-left:8px;'></span>
                <span style='font-size:0.78rem;color:#6a84b0;font-family:monospace;'>
                    BM25 {"READY" if bm25_ok else "NOT BUILT"}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("⚠️ Backend unreachable. Start the FastAPI server first.")

    st.divider()

    # File uploader
    uploaded = st.file_uploader(
        "Drop PDF here",
        type=["pdf"],
        help="NASA SP-2016-6105 or any PDF. Max recommended: ~300 pages.",
        label_visibility="collapsed",
    )

    if uploaded:
        st.markdown(
            f"<div style='font-size:0.8rem;color:#0af0ff;font-family:monospace;"
            f"margin:6px 0;'>📄 {uploaded.name}<br>"
            f"<span style='color:#6a84b0;'>{len(uploaded.getvalue()):,} bytes</span></div>",
            unsafe_allow_html=True,
        )

        if st.button("⚡ INGEST DOCUMENT", use_container_width=True):
            with st.spinner("Running ingestion pipeline…"):
                progress = st.progress(0, text="Parsing PDF…")
                try:
                    # Animate progress bar while waiting
                    for pct, label in [
                        (15, "Classifying blocks…"),
                        (30, "Chunking text…"),
                        (50, "Describing images with GPT-4o…"),
                        (65, "Resolving acronyms…"),
                        (80, "Embedding & upserting…"),
                    ]:
                        time.sleep(0.3)
                        progress.progress(pct, text=label)

                    result = _ingest_pdf(uploaded)
                    progress.progress(100, text="✅ Complete!")
                    time.sleep(0.5)
                    progress.empty()

                    st.session_state.ingestion_done = True
                    st.session_state.ingestion_stats = result
                    st.success(f"✅ {result.get('message', 'Done!')}")
                except Exception as exc:
                    progress.empty()
                    st.error(f"❌ Ingestion failed: {exc}")

    # Show ingestion stats
    if st.session_state.ingestion_done:
        stats = st.session_state.ingestion_stats
        st.divider()
        st.markdown("#### 📊 INDEX STATS")
        c1, c2 = st.columns(2)
        c1.metric("Pages", stats.get("num_pages", 0))
        c2.metric("Chunks", stats.get("num_text_chunks", 0))
        c1.metric("Images", stats.get("num_images", 0))
        c2.metric("Tables", stats.get("num_tables", 0))

    # Collections info
    st.divider()
    st.markdown("#### 🗄 COLLECTIONS")
    if st.button("↻ Refresh", use_container_width=False):
        st.session_state.collection_stats = {
            c["name"]: c["vectors_count"] for c in _fetch_collections()
        }

    if st.session_state.collection_stats:
        for name, count in st.session_state.collection_stats.items():
            icon = {"text_chunks": "📝", "image_store": "🖼", "table_store": "📋"}.get(name, "•")
            st.markdown(
                f"<div style='font-family:monospace;font-size:0.78rem;"
                f"color:#6a84b0;padding:2px 0;'>{icon} {name}: "
                f"<span style='color:#0af0ff;'>{count:,}</span></div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # Clear chat
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<p style='font-size:0.7rem;color:#2a3a5a;text-align:center;margin-top:24px;'>"
        "NASA RAG · Qdrant + LangGraph + GPT-4o</p>",
        unsafe_allow_html=True,
    )


# ── Main panel – Chat ──────────────────────────────────────────────────────────

# Top banner
st.markdown(
    """
    <div class="top-banner">
        <span style="font-size:1.6rem;">🚀</span>
        <div>
            <div class="banner-title">DOCUMENT NAVIGATOR</div>
            <div class="banner-sub"> STRUCTURED RAG INTERFACE</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Welcome message
if not st.session_state.messages:
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0d1525,#111d33);
            border:1px solid #1e3158;border-radius:10px;
            padding:24px 28px;margin-bottom:20px;">
            <h3 style="color:#0af0ff;font-family:'Share Tech Mono',monospace;
                margin:0 0 12px;">MISSION BRIEFING</h3>
            <p style="color:#8aa0c8;line-height:1.7;margin:0;">
                Upload a NASA handbook PDF using the sidebar uplink, then ask anything about it.<br>
                The system uses <b style="color:#0af0ff;">hybrid retrieval</b> (dense vector + BM25),
                <b style="color:#ff6b35;">GPT-4o vision</b> for diagrams, and
                <b style="color:#7fff7f;">LangGraph</b> for orchestration.
            </p>
            <div style="margin-top:16px;display:flex;gap:10px;flex-wrap:wrap;">
                <span style="background:rgba(10,240,255,0.08);border:1px solid #1e3158;
                    border-radius:6px;padding:6px 14px;font-size:0.82rem;color:#6a84b0;">
                    💡 What are the SE lifecycle phases?
                </span>
                <span style="background:rgba(10,240,255,0.08);border:1px solid #1e3158;
                    border-radius:6px;padding:6px 14px;font-size:0.82rem;color:#6a84b0;">
                    💡 Explain the technical review process
                </span>
                <span style="background:rgba(10,240,255,0.08);border:1px solid #1e3158;
                    border-radius:6px;padding:6px 14px;font-size:0.82rem;color:#6a84b0;">
                    💡 What does ICD stand for?
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render existing messages
for msg in st.session_state.messages:
    avatar = "🧑‍🚀" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            sections = meta.get("cited_sections", [])
            pages = meta.get("page_refs", [])
            confidence = meta.get("confidence_score")

            if sections or pages:
                st.markdown(
                    _render_citation_pills(sections, pages),
                    unsafe_allow_html=True,
                )

            conf_pct = int(confidence * 100)
            conf_color = "#7fff7f" if conf_pct > 60 else "#ff9900" if conf_pct > 30 else "#ff4444"
            st.markdown(
                f"<div style='margin-top:8px;'>"
                f"<span style='font-family:monospace;font-size:0.72rem;color:{conf_color};'>"
                f"▮ CONFIDENCE {conf_pct}%</span>"
                f"<div class='confidence-bar' style='width:{conf_pct}%;background:{conf_color};'></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

# Chat input
if prompt := st.chat_input("Ask about the NASA handbook…"):
    # User bubble
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🚀"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Querying knowledge base…"):
            try:
                result = _query(prompt)
                answer = result.get("answer", "No answer returned.")
                meta = {
                    "cited_sections": result.get("cited_sections", []),
                    "page_refs": result.get("page_refs", []),
                    "confidence": result.get("confidence_score"),
                }
            except requests.exceptions.ConnectionError:
                answer = (
                    "⚠️ **Backend not reachable.** "
                    "Please start the FastAPI server:\n```\npython main.py\n```"
                )
                meta = {}
            except Exception as exc:
                answer = f"❌ Error: {exc}"
                meta = {}

        st.markdown(answer)

        if meta:
            sections = meta.get("cited_sections", [])
            pages = meta.get("page_refs", [])
            confidence = meta.get("confidence")

            if sections or pages:
                st.markdown(
                    _render_citation_pills(sections, pages),
                    unsafe_allow_html=True,
                )

            conf_pct = int(confidence * 100)
            conf_color = "#7fff7f" if conf_pct > 60 else "#ff9900" if conf_pct > 30 else "#ff4444"
            st.markdown(
                f"<div style='margin-top:8px;'>"
                f"<span style='font-family:monospace;font-size:0.72rem;color:{conf_color};'>"
                f"▮ CONFIDENCE {conf_pct}%</span>"
                f"<div class='confidence-bar' style='width:{conf_pct}%;background:{conf_color};'></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "meta": meta}
    )
