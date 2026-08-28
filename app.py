from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from io import BytesIO
from copy import deepcopy

import streamlit as st

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
)
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.enums import (
    TA_CENTER,
    TA_LEFT,
    TA_RIGHT,
)
from reportlab.lib import colors
from xml.sax.saxutils import escape

from google.genai.errors import APIError

from lexicore.store import (
    EvidenceStore,
    DEFAULT_COLLECTION,
)
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

TRANSLATION_MODEL = os.getenv(
    "LEXICORE_TRANSLATION_MODEL",
    MODEL,
)

SUPPORTED_LANGUAGES = [
    "English",
    "French",
    "Arabic",
    "Hausa",
    "Igbo",
    "Yoruba",
]

MAX_TRANSLATION_CHARS = int(
    os.getenv(
        "LEXICORE_MAX_TRANSLATION_CHARS",
        "24000",
    )
)


# ============================================================
# EVIDENCE STORE
# ============================================================

@st.cache_resource(
    show_spinner="Initializing canonical evidence index..."
)
def get_store():

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

    defaults = {

        "hits": [],

        "query": "",

        "answer": None,

        "weakness": None,

        "run_id": uuid.uuid4().hex,

        "history": [],

        "sessions": {},

        "active_session_id": None,

        "setting_stance":
            "Didactic/Explanatory",

        "setting_categories": [],

        "setting_n": 15,

        "setting_temp": 0.2,

        "setting_lang": "English",

        "last_lang": "English",

        # language -> translated history
        "translation_cache": {},

        # Tracks how many canonical turns were already
        # translated for each language.
        #
        # {
        #   "Igbo": 3
        # }
        "translation_counts": {},

        "elapsed": 0.0,

        # Cache PDF bytes so Streamlit reruns don't rebuild it.
        "pdf_cache": None,

        "pdf_cache_key": None,
    }

    for key, value in defaults.items():

        st.session_state.setdefault(
            key,
            value,
        )


# ============================================================
# TRANSLATION
# ============================================================

def _translation_keys():

    try:
        return get_available_keys()

    except Exception:
        return []


def _translation_error_is_retryable(
    exc: Exception,
) -> bool:

    text = (
        f"{type(exc).__name__}: {exc}"
    ).lower()

    return any(
        marker in text
        for marker in (
            "429",
            "resource_exhausted",
            "rate limit",
            "quota",
            "403",
            "permission_denied",
            "500",
            "502",
            "503",
            "504",
            "service unavailable",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "connecterror",
            "unexpected eof",
        )
    )


def _clip_translation_input(
    text: str,
) -> str:

    if len(text) <= MAX_TRANSLATION_CHARS:
        return text

    return (
        text[:MAX_TRANSLATION_CHARS]
        + "\n\n[Remaining text omitted for translation efficiency.]"
    )


def _translate_batch(
    turns: list,
    target_lang: str,
) -> list:

    """
    Translate ALL supplied turns in ONE Gemini request.

    This is the major speed improvement over the old implementation.
    """

    if not turns:
        return []

    if target_lang == "English":

        return deepcopy(turns)

    keys = _translation_keys()

    if not keys:

        return deepcopy(turns)

    blocks = []

    for index, turn in enumerate(
        turns,
        start=1,
    ):

        original = turn.get(
            "original_answer",
            turn.get(
                "answer",
                "",
            ),
        )

        blocks.append(
            f"""
TURN {index}

QUERY:
{turn.get("query", "")}

ANSWER:
{original}
""".strip()
        )

    conversation_text = "\n\n==========\n\n".join(
        blocks
    )

    conversation_text = _clip_translation_input(
        conversation_text
    )

    prompt = f"""
Translate the following THE ARMOR theological research
conversation from English into {target_lang}.

This is a TRANSLATION task, not a summarization task.

Rules:

- Translate every supplied turn.
- Do not omit information.
- Do not summarize.
- Do not add information.
- Preserve the exact meaning.
- Preserve Bible references.
- Preserve Quran references.
- Preserve historical names.
- Preserve citations.
- Preserve evidence IDs.
- Preserve Markdown.
- Preserve headings.
- Preserve numbered lists.
- Preserve bullet points.
- Preserve paragraph structure.
- Do not alter the argument.
- Do not explain the translation.

Return ONLY the translated conversation.

Use exactly this structure:

TURN 1
QUERY:
...
ANSWER:
...

TURN 2
QUERY:
...
ANSWER:
...

and so on.

TARGET LANGUAGE:
{target_lang}

CONVERSATION:

{conversation_text}
"""

    last_error = None

    for _ in range(
        len(keys)
    ):

        try:

            client = get_next_client()

            response = client.models.generate_content(
                model=TRANSLATION_MODEL,
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 7000,
                },
            )

            raw = getattr(
                response,
                "text",
                None,
            )

            if not raw:

                raise RuntimeError(
                    "Empty translation response."
                )

            return _parse_translated_turns(
                raw,
                turns,
            )

        except Exception as exc:

            last_error = exc

            if not _translation_error_is_retryable(
                exc
            ):
                break

            rotate_key_on_error()

            # IMPORTANT:
            # No artificial sleep.
            # The llm layer handles key cooldown.
            continue

    # Silent failure:
    # Return original English instead of exposing API
    # errors or key numbers to the user.
    return deepcopy(turns)


def _parse_translated_turns(
    text: str,
    original_turns: list,
) -> list:

    """
    Parse the structured translation response.

    If parsing is imperfect, retain the original query and
    fall back to the original answer for the affected turn.
    """

    result = []

    blocks = []

    current = []

    for line in text.splitlines():

        stripped = line.strip()

        if stripped.upper().startswith(
            "TURN "
        ) and current:

            blocks.append(
                "\n".join(current)
            )

            current = [
                line
            ]

        else:

            current.append(line)

    if current:
        blocks.append(
            "\n".join(current)
        )

    # Remove accidental preamble.
    cleaned_blocks = []

    for block in blocks:

        if "QUERY:" in block.upper():

            cleaned_blocks.append(
                block
            )

    if not cleaned_blocks:

        return deepcopy(
            original_turns
        )

    for index, original in enumerate(
        original_turns
    ):

        if index >= len(
            cleaned_blocks
        ):

            result.append(
                {
                    "query":
                        original.get(
                            "query",
                            "",
                        ),
                    "answer":
                        original.get(
                            "original_answer",
                            original.get(
                                "answer",
                                "",
                            ),
                        ),
                    "original_answer":
                        original.get(
                            "original_answer",
                            original.get(
                                "answer",
                                "",
                            ),
                        ),
                }
            )

            continue

        block = cleaned_blocks[index]

        query_pos = block.upper().find(
            "QUERY:"
        )

        answer_pos = block.upper().find(
            "ANSWER:"
        )

        if (
            query_pos == -1
            or answer_pos == -1
            or answer_pos <= query_pos
        ):

            result.append(
                {
                    "query":
                        original.get(
                            "query",
                            "",
                        ),
                    "answer":
                        original.get(
                            "original_answer",
                            original.get(
                                "answer",
                                "",
                            ),
                        ),
                    "original_answer":
                        original.get(
                            "original_answer",
                            original.get(
                                "answer",
                                "",
                            ),
                        ),
                }
            )

            continue

        translated_query = block[
            query_pos
            + len("QUERY:"):
            answer_pos
        ].strip()

        translated_answer = block[
            answer_pos
            + len("ANSWER:"):
        ].strip()

        if not translated_answer:

            translated_answer = original.get(
                "original_answer",
                original.get(
                    "answer",
                    "",
                ),
            )

        result.append(
            {
                "query":
                    translated_query
                    or original.get(
                        "query",
                        "",
                    ),

                "answer":
                    translated_answer,

                "original_answer":
                    original.get(
                        "original_answer",
                        original.get(
                            "answer",
                            "",
                        ),
                    ),
            }
        )

    return result


def translate_text(
    text: str,
    target_lang: str,
) -> str:

    """
    Compatibility wrapper.

    For a single response this still makes one request,
    but normal application flow uses _translate_batch().
    """

    if not text:
        return text

    if target_lang == "English":
        return text

    turns = [
        {
            "query": "",
            "answer": text,
            "original_answer": text,
        }
    ]

    translated = _translate_batch(
        turns,
        target_lang,
    )

    if translated:

        return translated[0].get(
            "answer",
            text,
        )

    return text


def translate_history(
    history: list,
    target_lang: str,
) -> list:

    if not history:
        return []

    if target_lang == "English":
        return deepcopy(history)

    return _translate_batch(
        history,
        target_lang,
    )


# ============================================================
# DISPLAY TRANSLATION CACHE
# ============================================================

def get_display_history(
    target_lang: str,
) -> list:

    history = st.session_state.get(
        "history",
        [],
    )

    if not history:
        return []

    if target_lang == "English":
        return history

    cache = st.session_state.translation_cache

    cached = cache.get(
        target_lang
    )

    translated_count = st.session_state.translation_counts.get(
        target_lang,
        0,
    )

    current_count = len(history)

    # Nothing changed.
    if (
        cached is not None
        and translated_count
        == current_count
    ):

        return cached

    # First translation for this language.
    if cached is None:

        translated = translate_history(
            history,
            target_lang,
        )

        cache[target_lang] = translated

        st.session_state.translation_counts[
            target_lang
        ] = len(history)

        return translated

    # --------------------------------------------------------
    # A new canonical answer was added.
    #
    # Translate ONLY the new turns.
    # --------------------------------------------------------

    if current_count > translated_count:

        new_turns = history[
            translated_count:
        ]

        new_translations = _translate_batch(
            new_turns,
            target_lang,
        )

        cache[target_lang] = (
            cached
            + new_translations
        )

        st.session_state.translation_counts[
            target_lang
        ] = current_count

        return cache[target_lang]

    # Defensive fallback.
    translated = translate_history(
        history,
        target_lang,
    )

    cache[target_lang] = translated

    st.session_state.translation_counts[
        target_lang
    ] = current_count

    return translated


def on_language_change():

    new_lang = (
        st.session_state.setting_lang
    )

    old_lang = (
        st.session_state.last_lang
    )

    if new_lang == old_lang:
        return

    st.session_state.last_lang = (
        new_lang
    )

    # English requires no API call.
    if new_lang == "English":
        return

    if not st.session_state.history:
        return

    # Translation is intentionally lazy.
    #
    # We do NOT translate here.
    #
    # get_display_history() performs it only when the
    # conversation is actually rendered.
    return


# ============================================================
# SESSION MANAGEMENT
# ============================================================

def clear_translation_cache():

    st.session_state.translation_cache = {}

    st.session_state.translation_counts = {}

    st.session_state.pdf_cache = None

    st.session_state.pdf_cache_key = None


def save_current_session():

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

    if len(title) > 30:

        title = (
            title[:30]
            + "..."
        )

    st.session_state.sessions[
        session_id
    ] = {
        "title": title,
        "history": deepcopy(
            history
        ),
    }


def start_new_thread():

    save_current_session()

    st.session_state.history = []

    st.session_state.answer = None

    st.session_state.weakness = None

    st.session_state.hits = []

    st.session_state.query = ""

    st.session_state.active_session_id = None

    clear_translation_cache()


def load_session(
    session_id: str,
):

    session = (
        st.session_state.sessions.get(
            session_id
        )
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
# PDF
# ============================================================

def generate_pdf(
    history: list,
    weakness_data=None,
    language: str = "English",
) -> bytes:

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

    is_arabic = (
        str(language).lower()
        == "arabic"
    )

    alignment = (
        TA_RIGHT
        if is_arabic
        else TA_LEFT
    )

    title_alignment = (
        TA_RIGHT
        if is_arabic
        else TA_CENTER
    )

    title_style = ParagraphStyle(
        "ArmorTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=23,
        alignment=title_alignment,
        textColor="#1e293b",
        spaceAfter=5,
    )

    subtitle_style = ParagraphStyle(
        "ArmorSubtitle",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
        alignment=title_alignment,
        textColor="#64748b",
        spaceAfter=15,
    )

    query_style = ParagraphStyle(
        "ArmorQuery",
        parent=styles["Heading2"],
        fontSize=11,
        leading=15,
        alignment=alignment,
        textColor="#1d4ed8",
        spaceBefore=10,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "ArmorBody",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
        alignment=alignment,
        textColor="#334155",
        spaceAfter=8,
    )

    weakness_style = ParagraphStyle(
        "ArmorWeakness",
        parent=styles["Heading2"],
        fontSize=11,
        leading=15,
        alignment=alignment,
        textColor="#b91c1c",
        spaceBefore=12,
        spaceAfter=5,
    )

    def safe(value):

        text = (
            ""
            if value is None
            else str(value)
        )

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
            "\n\n",
            "<br/><br/>",
        )

        text = text.replace(
            "\n",
            "<br/>",
        )

        return text

    story = []

    story.append(
        Paragraph(
            safe(
                "THE ARMOR Research Report"
            ),
            title_style,
        )
    )

    story.append(
        Paragraph(
            safe(
                "Evidence-Grounded Theological Research & "
                f"Apologetics — {language}"
            ),
            subtitle_style,
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color="#cbd5e1",
            spaceAfter=12,
        )
    )

    for turn in reversed(history):

        query = safe(
            turn.get(
                "query",
                "",
            )
        )

        answer_text = safe(
            turn.get(
                "answer",
                "",
            )
        )

        story.append(
            Paragraph(
                f"<b>{'السؤال:' if is_arabic else 'Query:'}</b> "
                f"{query}",
                query_style,
            )
        )

        story.append(
            Paragraph(
                f"<b>{'الإجابة:' if is_arabic else 'Answer:'}</b>"
                f"<br/>{answer_text}",
                body_style,
            )
        )

        story.append(
            Spacer(
                1,
                4,
            )
        )

    if weakness_data:

        story.append(
            HRFlowable(
                width="100%",
                thickness=0.5,
                color="#e2e8f0",
                spaceBefore=8,
                spaceAfter=10,
            )
        )

        story.append(
            Paragraph(
                safe(
                    "Adversarial Review / Weaknesses"
                ),
                weakness_style,
            )
        )

        for item in getattr(
            weakness_data,
            "weakest_points",
            [],
        ):

            story.append(
                Paragraph(
                    "• "
                    + safe(item),
                    body_style,
                )
            )

        defense = getattr(
            weakness_data,
            "defense_strategy",
            [],
        )

        if defense:

            story.append(
                Paragraph(
                    "<b>Defense / Qualification Strategy</b>",
                    body_style,
                )
            )

            for item in defense:

                story.append(
                    Paragraph(
                        "• "
                        + safe(item),
                        body_style,
                    )
                )

        unsupported = getattr(
            weakness_data,
            "unsupported_claims",
            [],
        )

        if unsupported:

            story.append(
                Paragraph(
                    "<b>Unsupported Claims</b>",
                    body_style,
                )
            )

            for item in unsupported:

                story.append(
                    Paragraph(
                        "• "
                        + safe(item),
                        body_style,
                    )
                )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


def get_cached_pdf(
    history: list,
    weakness_data,
    language: str,
) -> bytes:

    # Hashable/simple cache key.
    history_key = tuple(
        (
            str(
                turn.get(
                    "query",
                    "",
                )
            ),
            str(
                turn.get(
                    "answer",
                    "",
                )
            ),
        )
        for turn in history
    )

    weakness_key = str(
        weakness_data
    )

    key = (
        history_key,
        weakness_key,
        language,
    )

    if (
        st.session_state.pdf_cache
        is not None
        and st.session_state.pdf_cache_key
        == key
    ):

        return st.session_state.pdf_cache

    pdf = generate_pdf(
        history,
        weakness_data,
        language,
    )

    st.session_state.pdf_cache = pdf

    st.session_state.pdf_cache_key = key

    return pdf


# ============================================================
# HEADER
# ============================================================

def render_header():

    _, center_col, _ = st.columns(
        [1, 2, 1]
    )

    with center_col:

        st.markdown(
            """
            <h1 style="text-align:center;">
                🛡️ THE ARMOR
            </h1>
            """,
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

        stance = st.selectbox(
            "Mode",
            [
                "Didactic/Explanatory",
                "Scholarly (Debate)",
                "Skeptical/Contrarian",
            ],
            key="setting_stance",
        )

        target_lang = st.selectbox(
            "Display / Translation Language",
            SUPPORTED_LANGUAGES,
            key="setting_lang",
        )

        selected_categories = (
            st.multiselect(
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
        )

        n = st.slider(
            "Evidence segments",
            5,
            30,
            key="setting_n",
        )

        temp = st.slider(
            "Generation temperature",
            0.0,
            1.0,
            key="setting_temp",
            step=0.05,
        )

        st.divider()

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
            "Could not open the canonical evidence index."
        )

        st.info(
            "Set LEXICORE_DB_PATH and build the index with "
            "the ingest command."
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
    # Language change
    # --------------------------------------------------------

    on_language_change()

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
    # Buttons
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
    # Evidence retrieval
    # --------------------------------------------------------

    if retrieve or generate:

        if not q.strip():

            st.warning(
                "Enter a question first."
            )

            st.stop()

        st.session_state.query = q

        categories = (
            selected_categories
            if selected_categories
            else None
        )

        with st.spinner(
            "Retrieving evidence…"
        ):

            st.session_state.hits = (
                store.search(
                    q,
                    n=n,
                    categories=categories,
                )
            )

            st.session_state.run_id = (
                uuid.uuid4().hex
            )

    if (
        not st.session_state.hits
        and (
            retrieve
            or generate
        )
    ):

        st.warning(
            "No matching evidence was found."
        )

        st.stop()

    hits = st.session_state.hits

    # ========================================================
    # GENERATION
    # ========================================================

    if hits:

        selected = [
            h.to_record()
            for h in hits
        ]

        if generate:

            if not selected:

                st.error(
                    "No evidence segments are available."
                )

                st.stop()

            try:

                with st.status(
                    "Generating evidence-grounded response...",
                    expanded=False,
                ) as status:

                    started = time.perf_counter()

                    result, used = answer(
                        query=q,
                        records=selected,
                        stance=stance,
                        temperature=temp,
                        history=st.session_state.history,
                    )

                    elapsed = (
                        time.perf_counter()
                        - started
                    )

                    st.session_state.answer = (
                        result
                    )

                    st.session_state.weakness = (
                        None
                    )

                    # ------------------------------------------------
                    # Optional adversarial review.
                    # ------------------------------------------------

                    if stance != (
                        "Didactic/Explanatory"
                    ):

                        (
                            st.session_state.weakness,
                            _,
                        ) = assess(
                            result.answer,
                            used,
                            stance=stance,
                            temperature=temp,
                        )

                    st.session_state.elapsed = (
                        elapsed
                    )

                    canonical_answer = (
                        result.answer
                    )

                    # ------------------------------------------------
                    # Store canonical English answer.
                    # ------------------------------------------------

                    st.session_state.history.append(
                        {
                            "query": q,
                            "answer": canonical_answer,
                            "original_answer":
                                canonical_answer,
                        }
                    )

                    # ------------------------------------------------
                    # IMPORTANT:
                    #
                    # DO NOT destroy existing translation caches.
                    #
                    # This is one of the biggest speed improvements.
                    #
                    # Existing translated turns remain valid.
                    # get_display_history() translates only the
                    # newly-added turn.
                    # ------------------------------------------------

                    st.session_state.pdf_cache = None

                    st.session_state.pdf_cache_key = None

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

            except APIError:

                st.error(
                    "The research request could not be completed. "
                    "Please try again."
                )

            except Exception:

                # Do not expose API keys, model internals,
                # quotas or raw exception details.
                st.error(
                    "The research request could not be completed. "
                    "Please try again."
                )

    # ========================================================
    # DISPLAY CONVERSATION
    # ========================================================

    if st.session_state.history:

        st.divider()

        st.subheader(
            "Conversation thread"
        )

        # ----------------------------------------------------
        # Translation happens here only when needed.
        # ----------------------------------------------------

        display_history = (
            get_display_history(
                target_lang
            )
        )

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        pdf_bytes = get_cached_pdf(
            display_history,
            st.session_state.get(
                "weakness"
            ),
            target_lang,
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

        # ----------------------------------------------------
        # Conversation
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

    if st.session_state.get(
        "weakness"
    ):

        w = st.session_state.weakness

        with st.expander(
            "Adversarial review"
        ):

            st.markdown(
                "**Weakest points**"
            )

            for item in (
                w.weakest_points
            ):

                st.write(
                    f"- {item}"
                )

            st.markdown(
                "**Defense / qualification strategy**"
            )

            for item in (
                w.defense_strategy
            ):

                st.write(
                    f"- {item}"
                )

            if w.unsupported_claims:

                st.markdown(
                    "**Unsupported claims**"
                )

                for item in (
                    w.unsupported_claims
                ):

                    st.write(
                        f"- {item}"
                    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
