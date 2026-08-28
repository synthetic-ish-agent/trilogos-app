from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from io import BytesIO
from copy import deepcopy

import streamlit as st

from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    HRFlowable,
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from xml.sax.saxutils import escape

from google import genai
from google.genai.errors import APIError

from lexicore.store import EvidenceStore
from lexicore.store import DEFAULT_COLLECTION
from lexicore.loaders import load_all
from lexicore.llm import answer, assess, client, MODEL


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="THE ARMOR",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <link rel="manifest" href="static/manifest.json">
    <meta name="theme-color" content="#1e293b">
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

DB = os.getenv("LEXICORE_DB_PATH", "./chroma_db")

COLLECTION = os.getenv(
    "LEXICORE_COLLECTION",
    "lexicore_evidence_v3",
)

TRANSLATION_MODEL = "gemini-3.6-flash"

SUPPORTED_LANGUAGES = [
    "English",
    "French",
    "Arabic",
    "Hausa",
    "Igbo",
    "Yoruba",
]


# ============================================================
# EVIDENCE STORE
# ============================================================

@st.cache_resource(
    show_spinner="Initializing canonical evidence index..."
)
def get_store():
    """
    Open or create the canonical evidence store.

    The store is cached because rebuilding/opening the database
    on every Streamlit rerun would be unnecessarily expensive.
    """

    db_path = Path(DB)
    db_path.mkdir(parents=True, exist_ok=True)

    try:
        store = EvidenceStore.open_or_create(
            DB,
            COLLECTION,
        )
    except Exception:
        store = EvidenceStore(
            DB,
            COLLECTION,
        )

    # Automatically ingest evidence if the collection is empty.
    if store.count() == 0:
        data_path = Path("./data")

        if data_path.exists():
            records = load_all(
                data_path,
                include_poc=True,
            )

            if records:
                store.add_records(records)

    return store


# ============================================================
# SESSION STATE
# ============================================================

def init():
    """
    Initialize all Streamlit session state values.

    Important:
    - history stores the ORIGINAL English responses.
    - translation_cache stores translated display versions.
    """

    defaults = {
        "hits": [],
        "query": "",
        "answer": None,
        "weakness": None,
        "run_id": str(uuid.uuid4().hex),
        "history": [],
        "sessions": {},
        "active_session_id": None,

        "setting_stance": "Didactic/Explanatory",
        "setting_categories": [],
        "setting_n": 15,
        "setting_temp": 0.2,

        "setting_lang": "English",
        "last_lang": "English",

        # ----------------------------------------------------
        # Translation cache
        #
        # {
        #     "French": [
        #         {
        #             "query": "...",
        #             "answer": "..."
        #         }
        #     ],
        #     "Hausa": [...]
        # }
        # ----------------------------------------------------
        "translation_cache": {},

        "elapsed": 0.0,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ============================================================
# TRANSLATION
# ============================================================

def translate_text(
    text: str,
    target_lang: str,
) -> str:
    """
    Translate an ORIGINAL English response into the selected
    display language.

    IMPORTANT:
    This uses LexiCore's existing Gemini client, which means
    translation uses the exact same API-key management,
    Streamlit secrets, key rotation, timeout, and configuration
    already used by answer().
    """

    if not text:
        return text

    if target_lang == "English":
        return text

    try:
        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Use LexiCore's existing Gemini client.
        #
        # DO NOT create another genai.Client() here.
        # ----------------------------------------------------

        ai_client = client()

        prompt = f"""
Translate the following theological research response
from English into {target_lang}.

IMPORTANT RULES:

- Translate the ENTIRE response.
- Do NOT summarize it.
- Do NOT shorten it.
- Do NOT remove information.
- Do NOT add information.
- Preserve the exact meaning.
- Preserve Markdown formatting.
- Preserve headings.
- Preserve paragraphs.
- Preserve numbered lists if present.
- Preserve bullet points if present.
- Preserve Bible references.
- Preserve Quran references.
- Preserve citations and evidence IDs.
- Preserve names of people and places.
- Preserve historical events.
- Preserve theological terminology accurately.
- Do not change the argument.
- Do not add theological commentary.
- Do not explain the translation.
- Return ONLY the translated response.

TARGET LANGUAGE:

{target_lang}

ORIGINAL ENGLISH RESPONSE:

{text}
"""

        response = ai_client.models.generate_content(
            model=MODEL,
            contents=prompt,
        )

        translated = getattr(
            response,
            "text",
            None,
        )

        if not translated:
            raise RuntimeError(
                "Gemini returned an empty translation."
            )

        return translated.strip()

    except Exception as exc:

        st.error(
            f"Translation to {target_lang} failed: {exc}"
        )

        return text

def translate_history(
    history: list,
    target_lang: str,
) -> list:
    """
    Translate an entire conversation from the ORIGINAL English
    responses.

    The returned list is for display only.

    The original history is never modified.
    """

    if target_lang == "English":
        return deepcopy(history)

    translated_history = []

    for turn in history:

        # New format stores original_answer.
        #
        # The fallback is useful for conversations created by
        # older versions of the application.
        original_answer = turn.get(
            "original_answer",
            turn.get("answer", ""),
        )

        translated_answer = translate_text(
            original_answer,
            target_lang,
        )

        translated_history.append(
            {
                "query": turn.get("query", ""),
                "answer": translated_answer,
                "original_answer": original_answer,
            }
        )

    return translated_history


def get_display_history(
    target_lang: str,
) -> list:
    """
    Return the conversation in the currently selected language.

    English uses the original history.

    Other languages use the translation cache.
    """

    history = st.session_state.get(
        "history",
        [],
    )

    if not history:
        return []

    if target_lang == "English":
        return history

    cached = st.session_state.translation_cache.get(
        target_lang
    )

    if cached is not None:
        return cached

    # Translation is performed only when this language has not
    # already been translated.
    translated = translate_history(
        history,
        target_lang,
    )

    st.session_state.translation_cache[target_lang] = (
        translated
    )

    return translated


def on_language_change():
    """
    Called when the language selectbox changes.

    IMPORTANT:
    This does NOT modify the original English conversation.

    It creates/loads a translation cache instead.
    """

    new_lang = st.session_state.setting_lang
    old_lang = st.session_state.last_lang

    if new_lang == old_lang:
        return

    st.session_state.last_lang = new_lang

    # Nothing to translate if there is no conversation yet.
    if not st.session_state.history:
        return

    # English is already the canonical language.
    if new_lang == "English":
        return

    # If this language was already translated, do nothing.
    if new_lang in st.session_state.translation_cache:
        return

    with st.spinner(
        f"Translating conversation to {new_lang}..."
    ):
        translated = translate_history(
            st.session_state.history,
            new_lang,
        )

        st.session_state.translation_cache[new_lang] = (
            translated
        )


# ============================================================
# SESSION / CONVERSATION MANAGEMENT
# ============================================================

def clear_translation_cache():
    """
    Remove all cached translations.

    This should happen whenever the underlying conversation
    changes.
    """

    st.session_state.translation_cache = {}


def save_current_session():
    """
    Save the current conversation thread.

    Sessions always store the ORIGINAL English history.
    """

    history = st.session_state.get(
        "history",
        [],
    )

    if not history:
        return

    if not st.session_state.active_session_id:
        st.session_state.active_session_id = (
            str(uuid.uuid4().hex)
        )

    session_id = st.session_state.active_session_id

    title = (
        history[0]
        .get("query", "Research")
        .strip()
    )

    if len(title) > 30:
        title = title[:30] + "..."

    st.session_state.sessions[session_id] = {
        "title": title,
        "history": deepcopy(history),
    }


def start_new_thread():
    """
    Start a completely new research thread.
    """

    save_current_session()

    st.session_state.history = []
    st.session_state.answer = None
    st.session_state.weakness = None
    st.session_state.hits = []
    st.session_state.query = ""
    st.session_state.active_session_id = None

    clear_translation_cache()


def load_session(session_id: str):
    """
    Load an existing saved conversation.
    """

    session = st.session_state.sessions.get(
        session_id
    )

    if not session:
        return

    st.session_state.history = deepcopy(
        session.get("history", [])
    )

    st.session_state.active_session_id = session_id

    st.session_state.answer = None
    st.session_state.weakness = None
    st.session_state.hits = []
    st.session_state.query = ""

    clear_translation_cache()


# ============================================================
# PDF GENERATION
# ============================================================

def generate_pdf(
    history: list,
    weakness_data=None,
    language: str = "English",
) -> bytes:
    """
    Generate a Unicode-compatible PDF.

    Uses DejaVu Sans so characters from:
    - English
    - French
    - Hausa
    - Igbo
    - Yoruba
    - many Arabic/Unicode characters

    are much less likely to appear as black dots or boxes.
    """

    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        HRFlowable,
    )
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import (
        getSampleStyleSheet,
        ParagraphStyle,
    )

    # ========================================================
    # FIND A UNICODE FONT
    # ========================================================

    font_candidates = [
        # Streamlit Cloud / Debian / Ubuntu
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",

        # Common Linux locations
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",

        # Local project font
        "./fonts/DejaVuSans.ttf",

        # Windows
        "C:/Windows/Fonts/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]

    font_path = None

    for candidate in font_candidates:
        if os.path.exists(candidate):
            font_path = candidate
            break

    if font_path is None:
        raise RuntimeError(
            "No Unicode-compatible font was found. "
            "Please add DejaVuSans.ttf to the project's "
            "fonts/ directory."
        )

    # ========================================================
    # REGISTER FONT
    # ========================================================

    if "ArmorUnicode" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(
            TTFont(
                "ArmorUnicode",
                font_path,
            )
        )

    # ========================================================
    # PDF DOCUMENT
    # ========================================================

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
        title="THE ARMOR Research Report",
        author="THE ARMOR",
    )

    styles = getSampleStyleSheet()

    # ========================================================
    # STYLES
    # ========================================================

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="ArmorUnicode",
        fontSize=20,
        leading=24,
        textColor="#1e293b",
        alignment=TA_CENTER,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "ReportSub",
        parent=styles["Normal"],
        fontName="ArmorUnicode",
        fontSize=10,
        leading=14,
        textColor="#64748b",
        alignment=TA_CENTER,
        spaceAfter=15,
    )

    qa_query_style = ParagraphStyle(
        "QAQuery",
        parent=styles["Heading2"],
        fontName="ArmorUnicode",
        fontSize=11,
        leading=15,
        textColor="#1d4ed8",
        spaceBefore=12,
        spaceAfter=4,
    )

    qa_answer_style = ParagraphStyle(
        "QAAnswer",
        parent=styles["Normal"],
        fontName="ArmorUnicode",
        fontSize=9.5,
        leading=14,
        textColor="#334155",
        spaceAfter=10,
    )

    weakness_heading = ParagraphStyle(
        "WeaknessHeading",
        parent=styles["Heading2"],
        fontName="ArmorUnicode",
        fontSize=11,
        leading=15,
        textColor="#b91c1c",
        spaceBefore=12,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="ArmorUnicode",
        fontSize=9.5,
        leading=14,
        textColor="#334155",
        spaceAfter=6,
    )

    # ========================================================
    # BUILD STORY
    # ========================================================

    story = [
        Paragraph(
            "THE ARMOR Research Report",
            title_style,
        ),

        Paragraph(
            (
                "Evidence-Grounded Theological Research & "
                f"Apologetics — {language}"
            ),
            subtitle_style,
        ),

        HRFlowable(
            width="100%",
            thickness=1,
            color="#cbd5e1",
            spaceAfter=15,
        ),
    ]

    # ========================================================
    # CONVERSATION
    # ========================================================

    for turn in history:

        query = str(
            turn.get("query", "")
        )

        answer_text = str(
            turn.get("answer", "")
        )

        # ----------------------------------------------------
        # Escape HTML-sensitive characters.
        # ----------------------------------------------------

        query = escape(query)

        answer_text = escape(
            answer_text
        )

        # ----------------------------------------------------
        # Preserve line breaks.
        # ----------------------------------------------------

        answer_text = answer_text.replace(
            "\n",
            "<br/>",
        )

        story.append(
            Paragraph(
                f"<b>Query:</b> {query}",
                qa_query_style,
            )
        )

        story.append(
            Paragraph(
                f"<b>Answer:</b><br/>{answer_text}",
                qa_answer_style,
            )
        )

        story.append(
            Spacer(
                1,
                5,
            )
        )

    # ========================================================
    # ADVERSARIAL REVIEW
    # ========================================================

    if weakness_data:

        story.append(
            HRFlowable(
                width="100%",
                thickness=0.5,
                color="#f1f5f9",
                spaceBefore=10,
                spaceAfter=10,
            )
        )

        story.append(
            Paragraph(
                "Adversarial Review / Weaknesses",
                weakness_heading,
            )
        )

        for wp in weakness_data.weakest_points:

            story.append(
                Paragraph(
                    f"• {escape(str(wp))}",
                    body_style,
                )
            )

    # ========================================================
    # BUILD PDF
    # ========================================================

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()

# ============================================================
# HEADER
# ============================================================

def render_header():
    _, center_col, _ = st.columns([1, 2, 1])

    with center_col:
        st.markdown(
            "<h1 style='text-align:center;'>🛡️ THE ARMOR</h1>",
            unsafe_allow_html=True,
        )

        st.caption(
            "Evidence-grounded theological research, "
            "apologetics, and cross-examination"
        )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    with st.sidebar:

        st.header("Research controls")

        # ----------------------------------------------------
        # Mode
        # ----------------------------------------------------

        stance = st.selectbox(
            "Mode",
            [
                "Didactic/Explanatory",
                "Scholarly (Debate)",
                "Skeptical/Contrarian",
            ],
            key="setting_stance",
        )

        # ----------------------------------------------------
        # Language
        # ----------------------------------------------------

        target_lang = st.selectbox(
            "Display / Translation Language",
            SUPPORTED_LANGUAGES,
            key="setting_lang",
        )

        # ----------------------------------------------------
        # Source categories
        # ----------------------------------------------------

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
            key="setting_categories",
        )

        # ----------------------------------------------------
        # Evidence count
        # ----------------------------------------------------

        n = st.slider(
            "Evidence segments",
            5,
            30,
            key="setting_n",
        )

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        temp = st.slider(
            "Generation temperature",
            0.0,
            1.0,
            key="setting_temp",
            step=0.05,
        )

        st.divider()

        # ----------------------------------------------------
        # Saved conversations
        # ----------------------------------------------------

        st.subheader("Saved Conversations")

        if st.button(
            "➕ New Research Thread",
            use_container_width=True,
        ):
            start_new_thread()
            st.rerun()

        if st.session_state.sessions:

            for sess_id, sess_data in list(
                st.session_state.sessions.items()
            ):

                title = sess_data.get(
                    "title",
                    "Research",
                )

                if st.button(
                    f"💬 {title}",
                    key=f"load_{sess_id}",
                    use_container_width=True,
                ):
                    load_session(sess_id)
                    st.rerun()

        if st.button(
            "🗑️ Clear All History",
            use_container_width=True,
        ):

            st.session_state.history = []
            st.session_state.sessions = {}
            st.session_state.active_session_id = None
            st.session_state.answer = None
            st.session_state.weakness = None
            st.session_state.hits = []
            st.session_state.query = ""

            clear_translation_cache()

            st.success("History cleared.")

            st.rerun()

    return (
        stance,
        target_lang,
        selected_categories,
        n,
        temp,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    init()

    render_header()

    # --------------------------------------------------------
    # Evidence store
    # --------------------------------------------------------

    try:
        store = get_store()

    except Exception as exc:

        st.error(
            f"Could not open the canonical evidence index: {exc}"
        )

        st.info(
            "Set LEXICORE_DB_PATH and build the index with: "
            f'python ingest.py --data ./data --db "{DB}"'
        )

        st.stop()

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    (
        stance,
        target_lang,
        selected_categories,
        n,
        temp,
    ) = render_sidebar()

    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------

    q = st.text_area(
        "Question / claim / counter-question",
        value=st.session_state.query,
        height=120,
        placeholder=(
            "Ask a theological question or follow up "
            "with a counter-question…"
        ),
        max_chars=2000,
    )

    # --------------------------------------------------------
    # Action buttons
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    retrieve = c1.button(
        "🔎 Retrieve evidence",
        use_container_width=True,
    )

    generate = c2.button(
        "🛡️ Generate answer",
        type="primary",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    if retrieve or generate:

        if not q.strip():
            st.warning(
                "Enter a question first."
            )
            st.stop()

        st.session_state.query = q

        cats = (
            selected_categories
            if selected_categories
            else None
        )

        with st.spinner(
            "Retrieving evidence…"
        ):

            st.session_state.hits = store.search(
                q,
                n=n,
                categories=cats,
            )

            st.session_state.run_id = (
                uuid.uuid4().hex
            )

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if (
        not st.session_state.hits
        and (retrieve or generate)
    ):

        st.warning(
            "No matching evidence was found."
        )

        st.stop()

    hits = st.session_state.hits

    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    if hits:

        selected = [
            h.to_record()
            for h in hits
        ]

        if generate:

            if not selected:

                st.error(
                    "Select at least one evidence segment."
                )

                st.stop()

            try:

                # ------------------------------------------------
                # IMPORTANT:
                #
                # Always generate the canonical answer in English.
                #
                # The selected language is a DISPLAY language.
                #
                # This gives us one stable source from which we can
                # translate into any language.
                # ------------------------------------------------

                effective_query = f"""
Answer the following theological research question
in ENGLISH.

The answer must be evidence-grounded and should use
the supplied evidence records.

Question:

{q}

IMPORTANT:
The canonical response must be written in English.
It will be translated separately for display if the user
selects another language.
"""

                with st.status(
                    "Generating evidence-grounded response...",
                    expanded=False,
                ) as status:

                    started = time.perf_counter()

                    result, used = answer(
                        query=effective_query,
                        records=selected,
                        stance=stance,
                        temperature=temp,
                        history=st.session_state.history,
                    )

                    elapsed = (
                        time.perf_counter()
                        - started
                    )

                    # ------------------------------------------------
                    # Store the canonical English answer.
                    # ------------------------------------------------

                    canonical_answer = result.answer

                    st.session_state.answer = result

                    st.session_state.weakness = None

                    # ------------------------------------------------
                    # Adversarial review
                    # ------------------------------------------------

                    if stance != "Didactic/Explanatory":

                        (
                            st.session_state.weakness,
                            _,
                        ) = assess(
                            canonical_answer,
                            used,
                            stance=stance,
                            temperature=temp,
                        )

                    st.session_state.elapsed = elapsed

                    # ------------------------------------------------
                    # Add the turn.
                    #
                    # BOTH fields intentionally contain the same
                    # canonical English response.
                    #
                    # "original_answer" is the permanent source.
                    # "answer" is kept for compatibility.
                    # ------------------------------------------------

                    st.session_state.history.append(
                        {
                            "query": q,
                            "answer": canonical_answer,
                            "original_answer": canonical_answer,
                        }
                    )

                    # ------------------------------------------------
                    # VERY IMPORTANT:
                    #
                    # A new answer means all previous translation
                    # caches are invalid.
                    # ------------------------------------------------

                    clear_translation_cache()

                    # ------------------------------------------------
                    # Save conversation.
                    # ------------------------------------------------

                    save_current_session()

                    status.update(
                        label=(
                            "Response generated successfully!"
                        ),
                        state="complete",
                        expanded=False,
                    )

                # Clear question field.
                st.session_state.query = ""

                # Rerun so the generated response appears.
                st.rerun()

            except APIError as api_err:

                message = getattr(
                    api_err,
                    "message",
                    str(api_err),
                )

                st.error(
                    f"API Error encountered: {message}"
                )

                st.info(
                    "The model might be experiencing "
                    "temporary high traffic or service "
                    "availability problems. Please try "
                    "generating the answer again."
                )

            except Exception as exc:

                st.error(
                    "An unexpected error occurred "
                    f"during generation: {exc}"
                )

    # ========================================================
    # DISPLAY CONVERSATION
    # ========================================================

    if st.session_state.get("history"):

        st.divider()

        st.subheader(
            "Conversation thread"
        )

        # ----------------------------------------------------
        # Get conversation in selected language.
        #
        # This does NOT modify the original English history.
        # ----------------------------------------------------

        display_history = get_display_history(target_lang)

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        pdf_bytes = generate_pdf(
            display_history,
            st.session_state.get("weakness"),
            language=target_lang,
        )

        st.download_button(
            label="📥 Download Research Thread as PDF",
            data=pdf_bytes,
            file_name=(
                "armor_research_report.pdf"
            ),
            mime="application/pdf",
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Conversation messages
        # ----------------------------------------------------

        for turn in reversed(display_history):

            with st.chat_message("user"):

                st.write(
                    turn.get(
                        "query",
                        "",
                    )
                )

            with st.chat_message("assistant"):

                st.write(
                    turn.get(
                        "answer",
                        "",
                    )
                )

    # ========================================================
    # GENERATION TIME
    # ========================================================

    st.caption(
        "Generation time: "
        f"{st.session_state.get('elapsed', 0):.2f}s. "
        "Similarity is a ranking signal, not a truth probability."
    )

    # ========================================================
    # ADVERSARIAL REVIEW
    # ========================================================

    if st.session_state.get("weakness"):

        w = st.session_state.weakness

        with st.expander(
            "Adversarial review"
        ):

            st.markdown(
                "**Weakest points**"
            )

            for x in w.weakest_points:

                st.write(
                    f"- {x}"
                )

            st.markdown(
                "**Defense / qualification strategy**"
            )

            for x in w.defense_strategy:

                st.write(
                    f"- {x}"
                )

            if w.unsupported_claims:

                st.markdown(
                    "**Unsupported claims**"
                )

                for x in w.unsupported_claims:

                    st.write(
                        f"- {x}"
                    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()