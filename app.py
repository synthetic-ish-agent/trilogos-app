from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from io import BytesIO
from copy import deepcopy

import streamlit as st

from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.platypus import (
    HRFlowable,
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xml.sax.saxutils import escape

from lexicore.store import EvidenceStore
from lexicore.store import DEFAULT_COLLECTION
from lexicore.loaders import load_all
from lexicore.llm import (
    answer,
    assess,
    get_available_keys,
    get_next_client,
    rotate_key_on_error,
    MODEL,
)


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

DB = os.getenv(
    "LEXICORE_DB_PATH",
    "./chroma_db",
)

COLLECTION = os.getenv(
    "LEXICORE_COLLECTION",
    DEFAULT_COLLECTION,
)

TRANSLATION_MODEL = MODEL


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

    db_path.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    try:
        count = store.count()
    except Exception:
        count = 0

    if count == 0:

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

    The application maintains the canonical conversation in
    English. Translations are cached separately and never
    overwrite the original English conversation.
    """

    defaults = {
        "hits": [],
        "query": "",
        "answer": None,
        "weakness": None,
        "run_id": uuid.uuid4().hex,

        "history": [],
        "sessions": {},
        "active_session_id": None,

        "setting_stance": "Didactic/Explanatory",
        "setting_categories": [],
        "setting_n": 15,
        "setting_temp": 0.2,
        "setting_lang": "English",
        "last_lang": "English",

        "translation_cache": {},

        "elapsed": 0.0,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# TRANSLATION
# ============================================================

def translate_text(
    text: str,
    target_lang: str,
) -> str:
    """
    Translate an existing English response into target_lang.

    Translation failures are intentionally silent at the UI
    level. The original English response is returned if
    translation fails.
    """

    if not text:
        return text

    target_lang = str(target_lang).strip()

    if not target_lang or target_lang == "English":
        return text

    try:
        keys = get_available_keys()

    except Exception:
        return text

    if not keys:
        return text

    prompt = f"""
Translate the following theological research response from
English into {target_lang}.

IMPORTANT RULES:

Translate the ENTIRE response.

Do NOT summarize it.

Do NOT shorten it.

Do NOT omit information.

Do NOT add information.

Do NOT change the argument.

Preserve the exact meaning.

Preserve theological terminology accurately.

Preserve Bible references.

Preserve Quran references.

Preserve historical names and places.

Preserve citations.

Preserve evidence IDs.

Preserve Markdown structure where applicable.

Preserve paragraphs.

Preserve headings.

Preserve numbered lists.

Preserve bullet points.

Do not introduce new theological claims.

Do not explain the translation.

Return ONLY the translated response.

TARGET LANGUAGE:
{target_lang}

ORIGINAL ENGLISH RESPONSE:
{text}
"""

    attempts = max(
        1,
        len(keys),
    )

    last_error = None

    for attempt in range(attempts):

        try:

            ai_client = get_next_client()

            response = ai_client.models.generate_content(
                model=TRANSLATION_MODEL,
                contents=prompt,
            )

            translated = getattr(
                response,
                "text",
                None,
            )

            if translated:

                translated = str(
                    translated
                ).strip()

                if translated:
                    return translated

            last_error = RuntimeError(
                "Empty translation response."
            )

        except Exception as exc:

            last_error = exc

            error_text = (
                f"{type(exc).__name__}: {exc}"
            ).lower()

            retryable = any(
                marker in error_text
                for marker in (
                    "429",
                    "resource_exhausted",
                    "rate limit",
                    "quota",
                    "403",
                    "permission_denied",
                    "leaked",
                    "500",
                    "502",
                    "503",
                    "504",
                    "service unavailable",
                    "timeout",
                    "timed out",
                    "connecterror",
                    "connection reset",
                    "connection aborted",
                    "temporary failure",
                    "name or service not known",
                    "getaddrinfo failed",
                    "nodename nor servname",
                    "unexpected eof",
                )
            )

            if not retryable:
                break

            if attempt < attempts - 1:

                # Rotate silently.
                try:
                    rotate_key_on_error()
                except Exception:
                    pass

                time.sleep(0.5)

    # Never expose the underlying API error to the user.
    return text


# ============================================================
# TRANSLATE HISTORY
# ============================================================

def translate_history(
    history: list,
    target_lang: str,
) -> list:
    """
    Translate an entire conversation from the ORIGINAL
    English responses.

    The original history is never modified.
    """

    if target_lang == "English":
        return deepcopy(history)

    translated_history = []

    for turn in history:

        original_answer = turn.get(
            "original_answer",
            turn.get(
                "answer",
                "",
            ),
        )

        translated_answer = translate_text(
            original_answer,
            target_lang,
        )

        translated_history.append(
            {
                "query": turn.get(
                    "query",
                    "",
                ),
                "answer": translated_answer,
                "original_answer": original_answer,
            }
        )

    return translated_history


# ============================================================
# DISPLAY HISTORY
# ============================================================

def get_display_history(
    target_lang: str,
) -> list:
    """
    Return the conversation in the currently selected language.
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

    translated = translate_history(
        history,
        target_lang,
    )

    st.session_state.translation_cache[target_lang] = (
        translated
    )

    return translated


# ============================================================
# LANGUAGE CHANGE
# ============================================================

def on_language_change():
    """
    Handle language changes without modifying the canonical
    English conversation.
    """

    new_lang = st.session_state.setting_lang
    old_lang = st.session_state.last_lang

    if new_lang == old_lang:
        return

    st.session_state.last_lang = new_lang

    if not st.session_state.history:
        return

    if new_lang == "English":
        return

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
# CLEAR TRANSLATION CACHE
# ============================================================

def clear_translation_cache():
    """
    Remove all cached translations.

    Translation caches must be invalidated whenever the
    underlying English conversation changes.
    """

    st.session_state.translation_cache = {}


# ============================================================
# SAVE CURRENT SESSION
# ============================================================

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
            uuid.uuid4().hex
        )

    session_id = (
        st.session_state.active_session_id
    )

    title = (
        history[0]
        .get(
            "query",
            "Research",
        )
        .strip()
    )

    if not title:
        title = "Research"

    if len(title) > 30:
        title = title[:30] + "..."

    st.session_state.sessions[session_id] = {
        "title": title,
        "history": deepcopy(history),
    }


# ============================================================
# START NEW THREAD
# ============================================================

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


# ============================================================
# LOAD SESSION
# ============================================================

def load_session(
    session_id: str,
):
    """
    Load an existing saved conversation.
    """

    session = st.session_state.sessions.get(
        session_id
    )

    if not session:
        return

    st.session_state.history = deepcopy(
        session.get(
            "history",
            [],
        )
    )

    st.session_state.active_session_id = (
        session_id
    )

    st.session_state.answer = None
    st.session_state.weakness = None
    st.session_state.hits = []
    st.session_state.query = ""

    clear_translation_cache()


# ============================================================
# PDF HELPER
# ============================================================

def safe_paragraph_text(
    value,
) -> str:
    """
    Convert arbitrary text into safe ReportLab Paragraph XML.

    ReportLab Paragraph uses XML-like markup, so user/model text
    must be escaped before being inserted into a Paragraph.

    Newlines are converted into <br/> elements.
    """

    if value is None:
        return ""

    text = str(value)

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    text = escape(text)

    text = text.replace(
        "\n",
        "<br/>",
    )

    return text


# ============================================================
# PDF GENERATION
# ============================================================

def generate_pdf(
    history: list,
    weakness_data=None,
    language: str = "English",
) -> bytes:
    """
    Generate a Unicode-aware THE ARMOR research PDF.

    Supported languages:

        English
        French
        Hausa
        Igbo
        Yoruba
        Arabic

    Font strategy:

        Latin languages:
            DejaVu Sans when available.

        Arabic:
            Noto Naskh Arabic when available.

    Expected font files:

        fonts/DejaVuSans.ttf
        fonts/NotoNaskhArabic-Regular.ttf
    """

    # ========================================================
    # FONT PATHS
    # ========================================================

    base_dir = Path(
        __file__
    ).resolve().parent

    fonts_dir = (
        base_dir / "fonts"
    )

    latin_font_path = (
        fonts_dir / "DejaVuSans.ttf"
    )

    arabic_font_path = (
        fonts_dir / "NotoNaskhArabic-Regular.ttf"
    )

    # ========================================================
    # REGISTER LATIN FONT
    # ========================================================

    latin_font = "Helvetica"

    if latin_font_path.exists():

        try:

            if (
                "ArmorDejaVu"
                not in pdfmetrics.getRegisteredFontNames()
            ):

                pdfmetrics.registerFont(
                    TTFont(
                        "ArmorDejaVu",
                        str(latin_font_path),
                    )
                )

            latin_font = "ArmorDejaVu"

        except Exception:
            latin_font = "Helvetica"

    # ========================================================
    # REGISTER ARABIC FONT
    # ========================================================

    arabic_font = latin_font

    if arabic_font_path.exists():

        try:

            if (
                "ArmorArabic"
                not in pdfmetrics.getRegisteredFontNames()
            ):

                pdfmetrics.registerFont(
                    TTFont(
                        "ArmorArabic",
                        str(arabic_font_path),
                    )
                )

            arabic_font = "ArmorArabic"

        except Exception:
            arabic_font = latin_font

    # ========================================================
    # LANGUAGE
    # ========================================================

    language_normalized = (
        str(language)
        .strip()
        .lower()
    )

    is_arabic = (
        language_normalized
        in {
            "arabic",
            "العربية",
            "عربي",
        }
    )

    active_font = (
        arabic_font
        if is_arabic
        else latin_font
    )

    # ReportLab's basic Paragraph engine does not provide full
    # Arabic shaping/bidi support by itself. We therefore use
    # right alignment and an Arabic-compatible font safely.
    rtl_supported = False

    # ========================================================
    # DOCUMENT
    # ========================================================

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="THE ARMOR Research Report",
        author="THE ARMOR",
        subject=(
            "Evidence-Grounded Theological "
            "Research & Apologetics"
        ),
    )

    # ========================================================
    # STYLES
    # ========================================================

    styles = getSampleStyleSheet()

    if is_arabic:

        title_alignment = TA_RIGHT
        subtitle_alignment = TA_RIGHT
        query_alignment = TA_RIGHT
        answer_alignment = TA_RIGHT
        body_alignment = TA_RIGHT

    else:

        title_alignment = TA_CENTER
        subtitle_alignment = TA_CENTER
        query_alignment = TA_LEFT
        answer_alignment = TA_LEFT
        body_alignment = TA_LEFT

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    title_style = ParagraphStyle(
        "ArmorReportTitle",
        parent=styles["Heading1"],
        fontName=active_font,
        fontSize=20,
        leading=26,
        alignment=title_alignment,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=5,
    )

    # --------------------------------------------------------
    # Subtitle
    # --------------------------------------------------------

    subtitle_style = ParagraphStyle(
        "ArmorReportSubtitle",
        parent=styles["Normal"],
        fontName=active_font,
        fontSize=9.5,
        leading=14,
        alignment=subtitle_alignment,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=14,
    )

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    query_style = ParagraphStyle(
        "ArmorReportQuery",
        parent=styles["Heading2"],
        fontName=active_font,
        fontSize=11,
        leading=16,
        alignment=query_alignment,
        textColor=colors.HexColor("#1d4ed8"),
        spaceBefore=12,
        spaceAfter=5,
    )

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    answer_style_kwargs = {
        "name": "ArmorReportAnswer",
        "parent": styles["Normal"],
        "fontName": active_font,
        "fontSize": 9.5,
        "leading": 15,
        "alignment": answer_alignment,
        "textColor": colors.HexColor("#334155"),
        "spaceAfter": 10,
    }

    if is_arabic and rtl_supported:
        answer_style_kwargs["wordWrap"] = "RTL"

    answer_style = ParagraphStyle(
        **answer_style_kwargs
    )

    # --------------------------------------------------------
    # Weakness heading
    # --------------------------------------------------------

    weakness_heading_style = ParagraphStyle(
        "ArmorWeaknessHeading",
        parent=styles["Heading2"],
        fontName=active_font,
        fontSize=11,
        leading=16,
        alignment=query_alignment,
        textColor=colors.HexColor("#b91c1c"),
        spaceBefore=12,
        spaceAfter=5,
    )

    # --------------------------------------------------------
    # Body
    # --------------------------------------------------------

    body_style_kwargs = {
        "name": "ArmorReportBody",
        "parent": styles["Normal"],
        "fontName": active_font,
        "fontSize": 9.5,
        "leading": 15,
        "alignment": body_alignment,
        "textColor": colors.HexColor("#334155"),
        "spaceAfter": 7,
    }

    if is_arabic and rtl_supported:
        body_style_kwargs["wordWrap"] = "RTL"

    body_style = ParagraphStyle(
        **body_style_kwargs
    )

    # ========================================================
    # PARAGRAPH FACTORY
    # ========================================================

    def make_paragraph(
        text: str,
        style,
    ):
        """
        Create a ReportLab Paragraph.

        Kept inside generate_pdf() so it has access to the
        current language/font configuration.
        """

        return Paragraph(
            text,
            style,
        )

    # ========================================================
    # STORY
    # ========================================================

    story = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    if is_arabic:

        title_text = safe_paragraph_text(
            "THE ARMOR — تقرير البحث"
        )

        subtitle_text = safe_paragraph_text(
            "البحث اللاهوتي المدعوم بالأدلة "
            f"والدفاعيات المسيحية — {language}"
        )

    else:

        title_text = safe_paragraph_text(
            "THE ARMOR Research Report"
        )

        subtitle_text = safe_paragraph_text(
            "Evidence-Grounded Theological Research & "
            f"Apologetics — {language}"
        )

    story.append(
        make_paragraph(
            title_text,
            title_style,
        )
    )

    story.append(
        make_paragraph(
            subtitle_text,
            subtitle_style,
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor("#cbd5e1"),
            spaceAfter=12,
        )
    )

    # ========================================================
    # CONVERSATION HISTORY
    # ========================================================

    # Keep the existing application's newest-first display.
    for turn in reversed(history):

        query_value = turn.get(
            "query",
            "",
        )

        answer_value = turn.get(
            "answer",
            "",
        )

        query = safe_paragraph_text(
            query_value
        )

        answer_text = safe_paragraph_text(
            answer_value
        )

        if is_arabic:

            query_label = "السؤال:"
            answer_label = "الإجابة:"

        else:

            query_label = "Query:"
            answer_label = "Answer:"

        query_label = escape(
            query_label
        )

        answer_label = escape(
            answer_label
        )

        story.append(
            make_paragraph(
                f"<b>{query_label}</b> {query}",
                query_style,
            )
        )

        story.append(
            make_paragraph(
                f"<b>{answer_label}</b><br/>"
                f"{answer_text}",
                answer_style,
            )
        )

        story.append(
            Spacer(
                1,
                4,
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
                color=colors.HexColor("#e2e8f0"),
                spaceBefore=8,
                spaceAfter=10,
            )
        )

        if is_arabic:

            weakness_title = (
                "المراجعة الجدلية / نقاط الضعف"
            )

        else:

            weakness_title = (
                "Adversarial Review / Weaknesses"
            )

        story.append(
            make_paragraph(
                safe_paragraph_text(
                    weakness_title
                ),
                weakness_heading_style,
            )
        )

        # ----------------------------------------------------
        # Weakest points
        # ----------------------------------------------------

        weakest_points = getattr(
            weakness_data,
            "weakest_points",
            [],
        )

        for wp in weakest_points:

            bullet = safe_paragraph_text(
                wp
            )

            story.append(
                make_paragraph(
                    f"• {bullet}",
                    body_style,
                )
            )

        # ----------------------------------------------------
        # Defense strategy
        # ----------------------------------------------------

        defense_strategy = getattr(
            weakness_data,
            "defense_strategy",
            [],
        )

        if defense_strategy:

            if is_arabic:

                defense_title = (
                    "استراتيجية الدفاع"
                )

            else:

                defense_title = (
                    "Defense / Qualification Strategy"
                )

            story.append(
                make_paragraph(
                    f"<b>"
                    f"{safe_paragraph_text(defense_title)}"
                    f"</b>",
                    body_style,
                )
            )

            for item in defense_strategy:

                story.append(
                    make_paragraph(
                        f"• "
                        f"{safe_paragraph_text(item)}",
                        body_style,
                    )
                )

        # ----------------------------------------------------
        # Unsupported claims
        # ----------------------------------------------------

        unsupported_claims = getattr(
            weakness_data,
            "unsupported_claims",
            [],
        )

        if unsupported_claims:

            if is_arabic:

                unsupported_title = (
                    "الادعاءات غير المدعومة"
                )

            else:

                unsupported_title = (
                    "Unsupported Claims"
                )

            story.append(
                make_paragraph(
                    f"<b>"
                    f"{safe_paragraph_text(unsupported_title)}"
                    f"</b>",
                    body_style,
                )
            )

            for item in unsupported_claims:

                story.append(
                    make_paragraph(
                        f"• "
                        f"{safe_paragraph_text(item)}",
                        body_style,
                    )
                )

    # ========================================================
    # BUILD PDF
    # ========================================================

    doc.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# HEADER
# ============================================================

def render_header():

    _, center_col, _ = st.columns(
        [1, 2, 1]
    )

    with center_col:

        st.markdown(
            "<h1 style='text-align:center;'>"
            "🛡️ THE ARMOR"
            "</h1>",
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

        st.header(
            "Research controls"
        )

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
            on_change=on_language_change,
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

        st.subheader(
            "Saved Conversations"
        )

        if st.button(
            "➕ New Research Thread",
            use_container_width=True,
        ):

            start_new_thread()

            st.rerun()

        if st.session_state.sessions:

            for (
                sess_id,
                sess_data,
            ) in list(
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

                    load_session(
                        sess_id
                    )

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

            st.success(
                "History cleared."
            )

            st.rerun()

    return (
        stance,
        target_lang,
        selected_categories,
        n,
        temp,
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    init()

    render_header()

    # ========================================================
    # EVIDENCE STORE
    # ========================================================

    try:

        store = get_store()

    except Exception:

        st.error(
            "Could not open the canonical evidence index."
        )

        st.info(
            "Check LEXICORE_DB_PATH and make sure the "
            "evidence database has been created."
        )

        st.stop()

    # ========================================================
    # SIDEBAR
    # ========================================================

    (
        stance,
        target_lang,
        selected_categories,
        n,
        temp,
    ) = render_sidebar()

    # ========================================================
    # QUESTION
    # ========================================================

    q = st.text_area(
        "Question / claim / counter-question",
        value=st.session_state.query,
        height=120,
        placeholder=(
            "Ask a theological question or follow up "
            "with a counter-question…"
        ),
        max_chars=5000,
    )

    # ========================================================
    # ACTION BUTTONS
    # ========================================================

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

    # ========================================================
    # RETRIEVAL
    # ========================================================

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

        try:

            with st.spinner(
                "Retrieving evidence…"
            ):

                st.session_state.hits = store.search(
                    q,
                    n=n,
                    categories=cats,
                )

        except Exception:

            st.error(
                "Evidence retrieval failed."
            )

            st.info(
                "Please check the evidence database and "
                "try again."
            )

            st.stop()

        st.session_state.run_id = (
            uuid.uuid4().hex
        )

    # ========================================================
    # NO EVIDENCE
    # ========================================================

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
                # Always generate canonical answer in English.
                # ------------------------------------------------

                effective_query = f"""
Answer the following theological research question in ENGLISH.

The answer must be evidence-grounded and should use the
supplied evidence records.

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

                    canonical_answer = (
                        result.answer
                    )

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

                    st.session_state.elapsed = (
                        elapsed
                    )

                    # ------------------------------------------------
                    # Store canonical English turn.
                    # ------------------------------------------------

                    st.session_state.history.append(
                        {
                            "query": q,
                            "answer": canonical_answer,
                            "original_answer": canonical_answer,
                        }
                    )

                    # ------------------------------------------------
                    # New answer invalidates translations.
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

                st.session_state.query = ""

                st.rerun()

            except Exception:

                # IMPORTANT:
                #
                # The LLM layer already handles key rotation and
                # internal Gemini failures.
                #
                # Never expose the raw exception here because it
                # may reveal:
                #
                # - API details
                # - quota information
                # - key indexes
                # - provider internals
                # - network details
                #
                st.error(
                    "The response could not be generated."
                )

                st.info(
                    "Please check your Google API key "
                    "configuration and try again."
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
        # Display selected language.
        # ----------------------------------------------------

        display_history = get_display_history(
            target_lang
        )

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        try:

            pdf_bytes = generate_pdf(
                display_history,
                st.session_state.get(
                    "weakness"
                ),
                language=target_lang,
            )

            st.download_button(
                label=(
                    "📥 Download Research Thread as PDF"
                ),
                data=pdf_bytes,
                file_name=(
                    "armor_research_report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True,
            )

        except Exception:

            st.warning(
                "The conversation was generated, but the "
                "PDF could not be created."
            )

        # ----------------------------------------------------
        # Conversation messages
        # ----------------------------------------------------

        for turn in reversed(
            display_history
        ):

            with st.chat_message(
                "user"
            ):

                st.write(
                    turn.get(
                        "query",
                        "",
                    )
                )

            with st.chat_message(
                "assistant"
            ):

                st.write(
                    turn.get(
                        "answer",
                        "",
                    )
                )

        # ====================================================
        # GENERATION TIME
        # ====================================================

        st.caption(
            "Generation time: "
            f"{st.session_state.get('elapsed', 0):.2f}s. "
            "Similarity is a ranking signal, not a truth probability."
        )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

