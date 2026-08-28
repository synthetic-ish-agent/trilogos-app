from __future__ import annotations

import os, time, uuid
from pathlib import Path
from io import BytesIO
import streamlit as st

from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import HRFlowable
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from google import genai
from google.genai.errors import APIError
from lexicore.store import EvidenceStore, DEFAULT_COLLECTION
from lexicore.loaders import load_all
from lexicore.llm import answer, assess

st.set_page_config(
    page_title="THE ARMOR", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <link rel="manifest" href="static/manifest.json">
    <meta name="theme-color" content="#1e293b">
    """,
    unsafe_allow_html=True
)
DB = os.getenv("LEXICORE_DB_PATH", "./chroma_db")
COLLECTION = os.getenv("LEXICORE_COLLECTION", "lexicore_evidence_v3")

@st.cache_resource(show_spinner="Initializing canonical evidence index...")
def get_store():
    db_path = Path(DB)
    db_path.mkdir(parents=True, exist_ok=True)

    # Safely get or create the store
    try:
        store = EvidenceStore.open_or_create(DB, COLLECTION)
    except Exception:
        # If collection/db creation fails due to missing state, initialize a fresh one
        store = EvidenceStore(DB, COLLECTION)

    # If the collection is empty and data folder is present, populate it automatically
    if store.count() == 0:
        data_path = Path("./data")
        if data_path.exists():
            records = load_all(data_path, include_poc=True)
            if records:
                store.add_records(records)
    return store

def generate_pdf(history: list, weakness_data) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter, 
        rightMargin=40, 
        leftMargin=40, 
        topMargin=40, 
        bottomMargin=40
    )
    styles = getSampleStyleSheet()

    # Polished Styles
    title_style = ParagraphStyle(
        'ReportTitle', 
        parent=styles['Heading1'], 
        fontSize=20, 
        leading=24, 
        textColor="#1e293b", 
        alignment=TA_CENTER,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'ReportSub', 
        parent=styles['Normal'], 
        fontSize=10, 
        textColor="#64748b", 
        alignment=TA_CENTER,
        spaceAfter=15
    )

    qa_query_style = ParagraphStyle(
        'QAQuery', 
        parent=styles['Heading2'], 
        fontSize=11, 
        leading=15, 
        textColor="#1d4ed8", 
        spaceBefore=12, 
        spaceAfter=4
    )

    qa_answer_style = ParagraphStyle(
        'QAAnswer', 
        parent=styles['Normal'], 
        fontSize=9.5, 
        leading=14, 
        textColor="#334155", 
        spaceAfter=10
    )

    weakness_heading = ParagraphStyle(
        'WeaknessHeading', 
        parent=styles['Heading2'], 
        fontSize=11, 
        leading=15, 
        textColor="#b91c1c", 
        spaceBefore=12, 
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'ReportBody', 
        parent=styles['Normal'], 
        fontSize=9.5, 
        leading=14, 
        textColor="#334155", 
        spaceAfter=6
    )

    story = [
        Paragraph("THE ARMOR Research Report", title_style),
        Paragraph("Evidence-Grounded Theological Research & Apologetics", subtitle_style),
        HRFlowable(width="100%", thickness=1, color="#cbd5e1", spaceAfter=15)
    ]

    for turn in reversed(history):
        story.append(Paragraph(f"<b>Query:</b> {turn['query']}", qa_query_style))
        story.append(Paragraph(f"<b>Answer:</b><br/>{turn['answer']}", qa_answer_style))

    if weakness_data:
        story.append(HRFlowable(width="100%", thickness=0.5, color="#f1f5f9", spaceBefore=10, spaceAfter=10))
        story.append(Paragraph("Adversarial Review / Weaknesses", weakness_heading))
        for wp in weakness_data.weakest_points:
            story.append(Paragraph(f"• {wp}", body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def translate_text(text: str, target_lang: str) -> str:
    """Helper to translate existing output text when language changes."""
    if not text or target_lang == "English":
        return text
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=f"Translate the following theological research response accurately into {target_lang}. Preserve markdown formatting and structure:\n\n{text}"
        )
        return response.text
    except Exception as e:
        return text

def on_language_change():
    """Callback triggered instantly when the language selectbox changes."""
    new_lang = st.session_state.get("setting_lang", "English")
    old_lang = st.session_state.get("last_lang", "English")
    if new_lang != old_lang:
        st.session_state.last_lang = new_lang
        if st.session_state.get("history"):
            for turn in st.session_state.history:
                turn["answer"] = translate_text(turn["answer"], new_lang)
            if st.session_state.get("answer") and hasattr(st.session_state.answer, "answer"):
                st.session_state.answer.answer = translate_text(st.session_state.answer.answer, new_lang)
    
def init():
    defaults = {
        "hits": [], 
        "query": "", 
        "answer": None, 
        "weakness": None, 
        "run_id": str(uuid.uuid4().hex),
        "history": [],  # Current active conversation thread
        "sessions": {}, # Saved history sessions dictionary
        "active_session_id": None,
        # Persistent setting defaults (Source filters set to empty list by default)
        "setting_stance": "Didactic/Explanatory",
        "setting_categories": [],
        "setting_n": 15,
        "setting_temp": 0.2,
        "setting_lang": "English",
        "last_lang": "English"
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def main():
    init()
    
    # Centered Header Layout using columns and HTML/CSS
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="display: flex; align-items: center; justify-content: center; gap: 10px; font-size: 2.5rem; margin-bottom: 0px;">
                    🛡️ THE ARMOR
                </h1>
                <p style="color: #94a3b8; font-size: 1rem; margin-top: 5px;">
                    Evidence-grounded theological research, apologetics, and cross-examination
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    try:
        store = get_store()
    except Exception as e:
        st.error(f"Could not open the canonical evidence index: {e}")
        st.info(f"Set LEXICORE_DB_PATH and build the index with: python ingest.py --data ./data --db \"{DB}\"")
        st.stop()

    with st.sidebar:
        st.header("Research controls")

        # Persistent widgets tied to session state keys
        stance = st.selectbox(
            "Mode", 
            ["Didactic/Explanatory", "Scholarly (Debate)", "Skeptical/Contrarian"],
            key="setting_stance"
        )

        languages = ["English", "French", "Arabic", "Hausa", "Igbo", "Yoruba"]
        current_lang_idx = languages.index(st.session_state.get("setting_lang", "English"))

        target_lang = st.selectbox(
            "Display / Translation Language",
            languages,
            index=current_lang_idx,
            key="setting_lang",
            on_change=on_language_change
        )

        selected_categories = st.multiselect(
            "Source filters",
            [
                "Christian Scripture",
                "Islamic Scripture",
                "Islamic Hadith",
                "Islamic History",
                "Christian Creed",
                "Christian Patristic",
                "Jewish Scripture / Commentary",
                "Historical / Other",
            ],
            key="setting_categories"
        )

        n = st.slider("Evidence segments", 5, 30, key="setting_n")
        temp = st.slider("Generation temperature", 0.0, 1.0, key="setting_temp", step=0.05)

        st.divider()
        st.subheader("Saved Conversations")

        if st.button("➕ New Research Thread", use_container_width=True):
            if st.session_state.history:
                if not st.session_state.active_session_id:
                    st.session_state.active_session_id = str(uuid.uuid4().hex)
                st.session_state.sessions[st.session_state.active_session_id] = {
                    "title": st.session_state.history[0]["query"][:30] + "...",
                    "history": list(st.session_state.history)
                }
            st.session_state.history = []
            st.session_state.answer = None
            st.session_state.weakness = None
            st.session_state.active_session_id = None
            st.rerun()

        if st.session_state.sessions:
            for sess_id, sess_data in list(st.session_state.sessions.items()):
                if st.button(f"💬 {sess_data['title']}", key=f"load_{sess_id}", use_container_width=True):
                    st.session_state.history = list(sess_data["history"])
                    st.session_state.active_session_id = sess_id
                    st.session_state.answer = None
                    st.session_state.weakness = None
                    st.rerun()

        if st.button("🗑️ Clear All History", use_container_width=True):
            st.session_state.history = []
            st.session_state.sessions = {}
            st.session_state.active_session_id = None
            st.session_state.answer = None
            st.session_state.weakness = None
            st.success("History cleared.")
            st.rerun()

    q = st.text_area(
        "Question / claim / counter-question",
        value=st.session_state.query,
        height=120,
        placeholder="Ask a theological question or follow up with a counter-question…",
        max_chars=2000,
    )

    c1, c2 = st.columns(2)
    retrieve = c1.button("🔎 Retrieve evidence", use_container_width=True)
    generate = c2.button("🛡️ Generate answer", type="primary", use_container_width=True)

    if retrieve or generate:
        if not q.strip():
            st.warning("Enter a question first.")
            st.stop()
        st.session_state.query = q

        cats = selected_categories if selected_categories else None

        with st.spinner("Retrieving evidence…"):
            st.session_state.hits = store.search(q, n=n, categories=cats)
            st.session_state.run_id = uuid.uuid4().hex

    if not st.session_state.hits and (retrieve or generate):
        st.warning("No matching evidence was found.")
        st.stop()

    hits = st.session_state.hits
    if hits:
        selected = [h.to_record() for h in hits]

        if generate:
            if not selected:
                st.error("Select at least one evidence segment.")
                st.stop()
            
            try:
                effective_query = q
                if target_lang and target_lang != "English":
                    effective_query = f"{q}\n\n[Instruction: Provide the final response translated entirely into {target_lang}.]"

                # Using st.status keeps all background retry logs hidden inside a clean expandable box
                with st.status("Generating evidence-grounded response...", expanded=False) as status:
                    started = time.perf_counter()

                    result, used = answer(
                        query=effective_query, 
                        records=selected, 
                        stance=stance, 
                        temperature=temp, 
                        history=st.session_state.history
                    )

                    elapsed = time.perf_counter() - started
                    st.session_state.answer = result
                    st.session_state.weakness = None

                    if stance != "Didactic/Explanatory":
                        st.session_state.weakness, _ = assess(result.answer, used, stance=stance, temperature=temp)

                    st.session_state.elapsed = elapsed

                    st.session_state.history.append({
                        "query": q,
                        "answer": result.answer
                    })

                    if not st.session_state.active_session_id:
                        st.session_state.active_session_id = str(uuid.uuid4().hex)
                    st.session_state.sessions[st.session_state.active_session_id] = {
                        "title": st.session_state.history[0]["query"][:30] + "...",
                        "history": list(st.session_state.history)
                    }

                    status.update(label="Response generated successfully!", state="complete", expanded=False)

                st.session_state.query = ""
                st.rerun()

            except APIError as api_err:
                st.error(f"API Error encountered: {api_err.message}")
                st.info("The model might be experiencing temporary high traffic spikes (503 Service Unavailable). Please try clicking 'Generate answer' again in a few moments.")
            except Exception as e:
                st.error(f"An unexpected error occurred during generation: {e}")

    # Display conversation history if available
    if st.session_state.get("history"):
        st.divider()
        st.subheader("Conversation thread")

        pdf_bytes = generate_pdf(st.session_state.history, st.session_state.get("weakness"))
        st.download_button(
            label="📥 Download Research Thread as PDF",
            data=pdf_bytes,
            file_name="armor_research_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        for turn in reversed(st.session_state.history):
            with st.chat_message("user"):
                st.write(turn["query"])
            with st.chat_message("assistant"):
                st.write(turn["answer"])

    st.caption(f"Generation time: {st.session_state.get('elapsed', 0):.2f}s. Similarity is a ranking signal, not a truth probablity.")

    if st.session_state.get("weakness"):
        w = st.session_state.weakness
        with st.expander("Adversarial review"):
            st.markdown("**Weakest points**")
            for x in w.weakest_points:
                st.write(f"- {x}")
            st.markdown("**Defense / qualification strategy**")
            for x in w.defense_strategy:
                st.write(f"- {x}")
            if w.unsupported_claims:
                st.markdown("**Unsupported claims**")
                for x in w.unsupported_claims:
                    st.write(f"- {x}")
                           
if __name__ == "__main__":
    main()