from __future__ import annotations

import json
import os
import socket
import time
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
)

import streamlit as st

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from .core import Record


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = os.getenv(
    "LEXICORE_LLM_MODEL",
    "gemini-3.6-flash",
)

REQUEST_TIMEOUT_MS = int(
    os.getenv(
        "LEXICORE_GEMINI_TIMEOUT_MS",
        "120000",
    )
)

MAX_KEY_ATTEMPTS = int(
    os.getenv(
        "LEXICORE_MAX_KEY_ATTEMPTS",
        "5",
    )
)

DEFAULT_MAX_CONTEXT_CHARS = int(
    os.getenv(
        "LEXICORE_MAX_CONTEXT_CHARS",
        "30000",
    )
)

MAX_QUERY_CHARS = int(
    os.getenv(
        "LEXICORE_MAX_QUERY_CHARS",
        "5000",
    )
)

T = TypeVar("T", bound=BaseModel)


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class Citation(BaseModel):
    evidence_id: str
    claim: str


class Answer(BaseModel):
    answer: str
    reasoning: str = ""
    limitations: list[str] = Field(
        default_factory=list
    )
    citations: list[Citation] = Field(
        default_factory=list
    )


class Weakness(BaseModel):
    weakest_points: list[str] = Field(
        default_factory=list
    )
    defense_strategy: list[str] = Field(
        default_factory=list
    )
    unsupported_claims: list[str] = Field(
        default_factory=list
    )


# ============================================================
# SAFE STREAMLIT SECRET ACCESS
# ============================================================

def _get_secret(
    name: str,
) -> Optional[str]:
    """
    Safely retrieve a Streamlit secret.

    Streamlit raises an exception when secrets.toml does not
    exist. That must never crash the application.
    """

    try:
        value = st.secrets.get(name)

        if value is not None:
            value = str(value).strip()

            if value:
                return value

    except Exception:
        pass

    return None


def _get_nested_secret(
    section: str,
    name: str,
) -> Optional[str]:
    """
    Support secrets configured like:

        [google]
        api_key = "..."

    or:

        [gemini]
        api_key = "..."
    """

    try:
        section_data = st.secrets.get(section)

        if section_data is None:
            return None

        if hasattr(section_data, "get"):

            value = section_data.get(name)

            if value is not None:

                value = str(value).strip()

                if value:
                    return value

    except Exception:
        pass

    return None


# ============================================================
# API KEY MANAGEMENT
# ============================================================

def get_available_keys() -> list[str]:
    """
    Load all available Google Gemini API keys.

    Supported Streamlit secrets:

        GOOGLE_API_KEY_1
        GOOGLE_API_KEY_2
        GOOGLE_API_KEY_3
        ...

        GOOGLE_API_KEY

    Supported environment variables:

        GOOGLE_API_KEY_1
        GOOGLE_API_KEY_2
        ...

        GOOGLE_API_KEY

    Additional compatible names:

        GEMINI_API_KEY

        GOOGLE_GENAI_API_KEY

    Nested Streamlit secrets are also supported:

        [google]
        api_key = "..."

        [gemini]
        api_key = "..."

    IMPORTANT:
    We scan ALL numbered keys instead of stopping at the first
    missing number.

    Therefore this works:

        GOOGLE_API_KEY_1 = "..."
        GOOGLE_API_KEY_3 = "..."

    even if GOOGLE_API_KEY_2 is absent.
    """

    keys: list[str] = []

    seen: set[str] = set()

    # --------------------------------------------------------
    # Numbered keys
    # --------------------------------------------------------

    for i in range(1, 101):

        name = f"GOOGLE_API_KEY_{i}"

        value = _get_secret(name)

        if not value:
            value = os.getenv(name)

        if value:

            value = str(value).strip()

            if value and value not in seen:

                keys.append(value)

                seen.add(value)

    # --------------------------------------------------------
    # Primary single-key fallback
    # --------------------------------------------------------

    fallback_names = (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_GENAI_API_KEY",
    )

    for name in fallback_names:

        value = _get_secret(name)

        if not value:
            value = os.getenv(name)

        if value:

            value = str(value).strip()

            if value and value not in seen:

                keys.append(value)

                seen.add(value)

    # --------------------------------------------------------
    # Nested Streamlit secrets
    # --------------------------------------------------------

    nested_locations = (
        ("google", "api_key"),
        ("google", "GOOGLE_API_KEY"),
        ("gemini", "api_key"),
        ("gemini", "GEMINI_API_KEY"),
    )

    for section, name in nested_locations:

        value = _get_nested_secret(
            section,
            name,
        )

        if value and value not in seen:

            keys.append(value)

            seen.add(value)

    # --------------------------------------------------------
    # Return keys
    # --------------------------------------------------------

    return keys


def require_api_keys() -> list[str]:
    """
    Return configured API keys or raise a clean configuration
    error.

    No secret values are ever included in the error.
    """

    keys = get_available_keys()

    if keys:
        return keys

    raise RuntimeError(
        "No Google Gemini API key is configured. "
        "Add GOOGLE_API_KEY_1 to Streamlit Secrets "
        "or set GOOGLE_API_KEY_1 as an environment variable."
    )


# ============================================================
# CURRENT KEY INDEX
# ============================================================

def _current_key_index() -> int:
    """
    Return the current API-key index safely.
    """

    keys = require_api_keys()

    index = st.session_state.get(
        "lexicore_key_index",
        0,
    )

    try:
        index = int(index)

    except (
        TypeError,
        ValueError,
    ):
        index = 0

    return index % len(keys)


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_next_client() -> genai.Client:
    """
    Create a Gemini client using the currently selected key.
    """

    keys = require_api_keys()

    index = _current_key_index()

    key = keys[index]

    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
        ),
    )


def client() -> genai.Client:
    """
    Public LexiCore client helper.
    """

    return get_next_client()


# ============================================================
# API KEY ROTATION
# ============================================================

def rotate_key_on_error() -> bool:
    """
    Move silently to the next configured API key.

    Returns True if rotation was possible.
    """

    keys = require_api_keys()

    if len(keys) <= 1:
        return False

    old_index = _current_key_index()

    new_index = (
        old_index + 1
    ) % len(keys)

    st.session_state.lexicore_key_index = (
        new_index
    )

    return True


# ============================================================
# NETWORK DIAGNOSTICS
# ============================================================

def check_gemini_dns() -> tuple[bool, str]:
    """
    Check whether Google's Gemini endpoint can be resolved.
    """

    hostname = (
        "generativelanguage.googleapis.com"
    )

    try:

        addresses = socket.getaddrinfo(
            hostname,
            443,
            type=socket.SOCK_STREAM,
        )

        if addresses:

            return (
                True,
                f"{hostname} resolved successfully.",
            )

        return (
            False,
            f"{hostname} returned no addresses.",
        )

    except socket.gaierror as exc:

        return (
            False,
            f"DNS resolution failed: {exc}",
        )

    except Exception as exc:

        return (
            False,
            f"DNS check failed: {exc}",
        )


# ============================================================
# EVIDENCE CONTEXT
# ============================================================

def context(
    records: list[Record],
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
):
    """
    Build the evidence context while respecting the character
    limit.
    """

    parts: list[str] = []

    used: list[Record] = []

    total = 0

    for i, record in enumerate(
        records,
        1,
    ):

        block = record.evidence_block(i)

        if len(block) > max_chars:
            continue

        if total + len(block) > max_chars:
            break

        parts.append(block)

        used.append(record)

        total += len(block)

    return (
        "\n\n".join(parts),
        used,
    )


# ============================================================
# SYSTEM INSTRUCTIONS
# ============================================================

def instructions(
    stance: str,
) -> str:

    base = f"""
You are LexiCore (also known as THE ARMOR), an elite theological research,
apologetics, and cross-examination assistant.

Your fundamental purpose is to defend and explain orthodox Christian truth,
Sacred Scripture, Sacred Tradition, and Catholic teaching.

LANGUAGE & COMMUNICATION STYLE:

Write in plain, clear, everyday English with a natural conversational flow
that an ordinary Nigerian can understand immediately.

Avoid unnecessarily complicated academic grammar and dense jargon.

Sound like a brilliant, knowledgeable Christian apologist explaining deep
truths clearly during a church fellowship, youth meeting, classroom, or
serious theological discussion.

Keep sentences direct and engaging without sacrificing theological accuracy.

CURRENT RESEARCH MODE:

{stance}

CORE RULES:

EVIDENCE GROUNDING

Use the supplied evidence segments as part of the evidential basis.

Do not pretend that a source says something it does not say.

FLEXIBLE KNOWLEDGE

If the supplied evidence does not contain the exact verse, text, or source
requested, you may use established theological and historical knowledge.

Clearly distinguish supplied evidence from broader knowledge.

NO FABRICATION

Never invent Bible quotations.

Never invent Quran quotations.

Never invent Hadith numbers.

Never invent historical sources.

Never invent evidence IDs.

CITATION DISCIPLINE

Every database citation must use an evidence_id that actually appears in
the supplied evidence.

If relying on broader knowledge, do not manufacture a database evidence ID.

CHRISTIAN AND CATHOLIC PRIORITY

When discussing Christian doctrine, Jesus Christ, Scripture, Mary,
the sacraments, ecclesiology, Church history, or Christian theology,
prioritize Christian and Catholic sources.

When addressing competing religious claims, evaluate them critically from
a Christian and Catholic apologetic standpoint.

INTELLECTUAL RIGOR

Distinguish between:

established evidence,

theological interpretation,

historical tradition,

and genuine uncertainty.

Do not present speculation as established fact.

COUNTERARGUMENTS

When appropriate, directly address objections and competing interpretations.

Do not merely repeat the user's premise.

HISTORICAL INTEGRITY

The absence of a source in the local evidence database does not prove that
the source or historical fact does not exist.

CURRENT MODE:

{stance}
"""

    if "Didactic" in stance:

        return base + """

MODE: DIDACTIC / EXPLANATORY

Explain the issue clearly and progressively.

Use cohesive prose rather than mechanical bullet points.

Define the issue, explain the evidence, reason through the argument,
address likely objections, and finish with a clear conclusion.

Prefer clarity and readability.
"""

    if "Scholarly" in stance:

        return base + """

MODE: SCHOLARLY / DEBATE

Use rigorous academic reasoning in polished professional prose.

Avoid mechanical numbered lists and bullet-heavy structures.

Treat textual evidence, historical evidence, theological interpretation,
and counterarguments carefully.

Produce a coherent theological argument with a defensible conclusion.
"""

    if "Skeptical" in stance:

        return base + """

MODE: SKEPTICAL / CONTRARIAN

Use a critical and probing lens.

Examine unsupported assumptions, weak inferences, textual ambiguities,
historical uncertainties, and alternative interpretations.

Maintain Christian theological integrity while critically examining claims.
"""

    return base


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def _error_text(
    exc: Exception,
) -> str:

    return (
        f"{type(exc).__name__}: {exc}"
    )


def _is_retryable_error(
    exc: Exception,
) -> bool:

    text = _error_text(
        exc
    ).lower()

    retryable_markers = (
        "429",
        "resource_exhausted",
        "rate limit",
        "quota",
        "403",
        "permission_denied",
        "leaked",
        "401",
        "unauthorized",
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

    return any(
        marker in text
        for marker in retryable_markers
    )


# ============================================================
# RESPONSE PARSING
# ============================================================

def _parse_response(
    response: Any,
    schema: Type[T],
) -> T:

    parsed = getattr(
        response,
        "parsed",
        None,
    )

    if parsed is not None:

        if isinstance(
            parsed,
            schema,
        ):
            return parsed

        return schema.model_validate(
            parsed
        )

    text = getattr(
        response,
        "text",
        None,
    )

    if not text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    clean_text = str(
        text
    ).strip()

    # --------------------------------------------------------
    # Remove Markdown JSON fences.
    # --------------------------------------------------------

    if clean_text.startswith(
        "```json"
    ):

        clean_text = clean_text[
            len("```json"):
        ].strip()

    elif clean_text.startswith(
        "```"
    ):

        clean_text = clean_text[
            len("```"):
        ].strip()

    if clean_text.endswith(
        "```"
    ):

        clean_text = clean_text[
            :-3
        ].strip()

    try:

        data = json.loads(
            clean_text
        )

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "Gemini returned invalid structured output."
        ) from exc

    return schema.model_validate(
        data
    )


# ============================================================
# GEMINI GENERATION
# ============================================================

def _generate(
    prompt: str,
    schema: Type[T],
    stance: str,
    temperature: float = 0.2,
) -> T:

    keys = require_api_keys()

    attempts = min(
        len(keys),
        max(
            1,
            MAX_KEY_ATTEMPTS,
        ),
    )

    last_error: Optional[Exception] = None

    for attempt in range(attempts):

        current_index = _current_key_index()

        try:

            ai_client = get_next_client()

            response = (
                ai_client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=instructions(
                            stance
                        ),
                        temperature=max(
                            0.0,
                            min(
                                float(temperature),
                                1.0,
                            ),
                        ),
                        max_output_tokens=4000,
                        response_mime_type=(
                            "application/json"
                        ),
                        response_schema=schema,
                    ),
                )
            )

            return _parse_response(
                response,
                schema,
            )

        except Exception as exc:

            last_error = exc

            # ------------------------------------------------
            # Retryable API/network failure.
            # ------------------------------------------------

            if _is_retryable_error(
                exc
            ):

                if attempt < attempts - 1:

                    rotate_key_on_error()

                    time.sleep(
                        0.5
                    )

                    continue

            # ------------------------------------------------
            # Non-retryable failure.
            # ------------------------------------------------

            raise RuntimeError(
                "LexiCore Gemini generation failed."
            ) from exc

    raise RuntimeError(
        "LexiCore Gemini generation failed after all "
        "available API keys were attempted."
    ) from last_error


# ============================================================
# ANSWER
# ============================================================

def answer(
    query: str,
    records: list[Record],
    stance: str = "Scholarly (Debate)",
    temperature: float = 0.2,
    history: Optional[
        List[Dict[str, str]]
    ] = None,
):

    query = query.strip()

    if not query:

        raise ValueError(
            "Query cannot be empty."
        )

    if len(query) > MAX_QUERY_CHARS:

        raise ValueError(
            "Query exceeds the maximum allowed length "
            f"of {MAX_QUERY_CHARS} characters."
        )

    ctx, used = context(
        records
    )

    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------

    history_text = ""

    if history:

        history_text = (
            "CONVERSATION HISTORY / PREVIOUS TURNS:\n"
        )

        for turn in history:

            previous_query = str(
                turn.get(
                    "query",
                    "",
                )
            ).strip()

            previous_answer = str(
                turn.get(
                    "original_answer",
                    turn.get(
                        "answer",
                        "",
                    ),
                )
            ).strip()

            if previous_query:

                history_text += (
                    f"User: {previous_query}\n"
                )

            if previous_answer:

                history_text += (
                    f"LexiCore: {previous_answer}\n"
                )

            history_text += "\n"

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
{history_text}

CURRENT QUERY / COUNTER-QUESTION:

{query}

SUPPLIED EVIDENCE:

{ctx}

TASK:

Answer the current query using the supplied evidence and
relevant established theological knowledge.

Requirements:

Answer the actual question directly.

Match the requested research stance.

Address counterarguments when relevant.

Distinguish evidence from interpretation.

Do not fabricate citations.

Use only evidence IDs that actually appear in the supplied evidence.

If the evidence is insufficient, explicitly state the limitation.

Provide a rigorous but readable response.

The canonical answer must be written in English.
"""

    result = _generate(
        prompt=prompt,
        schema=Answer,
        stance=stance,
        temperature=temperature,
    )

    # --------------------------------------------------------
    # Citation validation
    # --------------------------------------------------------

    valid_ids = {
        record.id
        for record in used
    }

    result.citations = [
        citation
        for citation in result.citations
        if citation.evidence_id in valid_ids
    ]

    return (
        result,
        used,
    )


# ============================================================
# ARGUMENT ASSESSMENT
# ============================================================

def assess(
    argument: str,
    records: list[Record],
    stance: str = "Scholarly (Debate)",
    temperature: float = 0.1,
):

    argument = argument.strip()

    if not argument:

        raise ValueError(
            "Argument cannot be empty."
        )

    if len(argument) > MAX_QUERY_CHARS:

        raise ValueError(
            "Argument exceeds the maximum allowed length "
            f"of {MAX_QUERY_CHARS} characters."
        )

    ctx, used = context(
        records
    )

    prompt = f"""
ARGUMENT TO ASSESS:

{argument}

SUPPLIED EVIDENCE:

{ctx}

TASK:

Act as an adversarial academic reviewer.

Identify genuine weaknesses in the argument using the supplied evidence.

Evaluate:

Logical weaknesses.

Unsupported assumptions.

Evidential gaps.

Textual problems.

Alternative interpretations.

Claims that are stronger than the evidence allows.

Then propose defensible repairs.

Do not invent evidence.

Do not invent citations.

Do not criticize the argument merely because it is controversial.

Focus on whether the reasoning is actually supported.
"""

    result = _generate(
        prompt=prompt,
        schema=Weakness,
        stance=stance,
        temperature=temperature,
    )

    return (
        result,
        used,
    )
