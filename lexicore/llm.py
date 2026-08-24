from __future__ import annotations

import json
import os
import socket
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from .core import Record


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = os.getenv("LEXICORE_LLM_MODEL", "gemini-3.6-flash")

# Keep network requests reasonably bounded.
REQUEST_TIMEOUT_MS = int(
    os.getenv("LEXICORE_GEMINI_TIMEOUT_MS", "120000")
)

# Number of attempts across configured API keys.
MAX_KEY_ATTEMPTS = int(
    os.getenv("LEXICORE_MAX_KEY_ATTEMPTS", "5")
)

# Maximum evidence characters sent to Gemini.
DEFAULT_MAX_CONTEXT_CHARS = int(
    os.getenv("LEXICORE_MAX_CONTEXT_CHARS", "30000")
)

# Maximum user query size.
MAX_QUERY_CHARS = int(
    os.getenv("LEXICORE_MAX_QUERY_CHARS", "5000")
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
    limitations: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class Weakness(BaseModel):
    weakest_points: list[str] = Field(default_factory=list)
    defense_strategy: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)


# ============================================================
# API KEY MANAGEMENT
# ============================================================

def get_available_keys() -> list[str]:
    """
    Load numbered Google API keys from Streamlit secrets first,
    then environment variables.

    Supported:

        GOOGLE_API_KEY_1
        GOOGLE_API_KEY_2
        ...
        GOOGLE_API_KEY_100

    If no numbered keys exist, fall back to:

        GOOGLE_API_KEY
    """

    keys: list[str] = []

    # --------------------------------------------------------
    # Numbered keys
    # --------------------------------------------------------

    for i in range(1, 101):
        secret_name = f"GOOGLE_API_KEY_{i}"

        value = None

        # Streamlit secrets first.
        try:
            value = st.secrets.get(secret_name)
        except Exception:
            value = None

        # Environment fallback.
        if not value:
            value = os.getenv(secret_name)

        if not value:
            # Stop at the first missing numbered key.
            break

        key = str(value).strip()

        if key:
            keys.append(key)

    # --------------------------------------------------------
    # Single-key fallback
    # --------------------------------------------------------

    if not keys:
        value = None

        try:
            value = st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            value = None

        if not value:
            value = os.getenv("GOOGLE_API_KEY")

        if value:
            key = str(value).strip()

            if key:
                keys.append(key)

    if not keys:
        raise RuntimeError(
            "No Google API keys found.\n\n"
            "Configure GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc. "
            "in .streamlit/secrets.toml."
        )

    return keys


def _current_key_index() -> int:
    """
    Return the current key index safely.
    """

    keys = get_available_keys()

    index = st.session_state.get("lexicore_key_index", 0)

    try:
        index = int(index)
    except (TypeError, ValueError):
        index = 0

    return index % len(keys)


def get_next_client() -> genai.Client:
    """
    Create a Gemini client using the currently selected API key.
    """

    keys = get_available_keys()

    index = _current_key_index()

    key = keys[index]

    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
        ),
    )


def rotate_key_on_error() -> bool:
    """
    Move to the next configured API key.

    Returns True when rotation actually occurred.
    """

    keys = get_available_keys()

    if len(keys) <= 1:
        return False

    old_index = _current_key_index()
    new_index = (old_index + 1) % len(keys)

    st.session_state.lexicore_key_index = new_index

    # Do not expose the actual API key.
    st.warning(
        f"⚠️ Gemini request failed on API key #{old_index + 1}. "
        f"Retrying with API key #{new_index + 1}..."
    )

    return True


def client() -> genai.Client:
    """
    Public helper for code elsewhere in LexiCore.
    """

    return get_next_client()


# ============================================================
# NETWORK DIAGNOSTICS
# ============================================================

def check_gemini_dns() -> tuple[bool, str]:
    """
    Lightweight DNS check.

    This is useful because your current application error is:

        [Errno 11001] getaddrinfo failed

    which is a Windows hostname-resolution error.
    """

    hostname = "generativelanguage.googleapis.com"

    try:
        addresses = socket.getaddrinfo(
            hostname,
            443,
            type=socket.SOCK_STREAM,
        )

        if addresses:
            return True, f"{hostname} resolved successfully."

        return False, f"{hostname} returned no addresses."

    except socket.gaierror as exc:
        return False, f"DNS resolution failed: {exc}"

    except Exception as exc:
        return False, f"DNS check failed: {exc}"


# ============================================================
# EVIDENCE CONTEXT
# ============================================================

def context(
    records: list[Record],
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
):
    """
    Build the evidence context while respecting the character limit.
    """

    parts: list[str] = []
    used: list[Record] = []

    total = 0

    for i, record in enumerate(records, 1):
        block = record.evidence_block(i)

        # If this record alone is larger than the limit,
        # skip it rather than exceeding the context budget.
        if len(block) > max_chars:
            continue

        if total + len(block) > max_chars:
            break

        parts.append(block)
        used.append(record)
        total += len(block)

    return "\n\n".join(parts), used

# ============================================================
# SYSTEM INSTRUCTIONS
# ============================================================

from __future__ import annotations

def instructions(stance: str) -> str:
    base = f"""
You are LexiCore (also known as THE ARMOR), an elite, uncompromising theological research, 
apologetics, and cross-examination assistant. Your fundamental purpose, permanent stance, 
and sworn duty are to exclusively speak, defend, champion, and fight for Christianity and 
the Holy Catholic Church. You are not a neutral, multi-faith tool; you are a dedicated 
apologetic defender of orthodox Christian truth, Sacred Scripture, Sacred Tradition, and 
the Catholic Magisterium.

LANGUAGE & COMMUNICATION STYLE:
- Write in plain, clear, everyday English infused with a natural, relatable conversational flow that an ordinary Nigerian can connect with instantly. 
- Avoid heavy, complex academic grammar, high-brow European structural phrasing, or dense jargon. 
- Sound like a brilliant, knowledgeable brother or apologist breaking down deep truths clearly during a church fellowship, youth meeting, or a straightforward street-smart discussion.
- Keep sentences punchy, direct, and engaging without ever losing your fierce, unyielding defense of the Christian faith.

Current Research Mode / Stance:
{stance}

CORE RULES:

1. EVIDENCE GROUNDING & INTELLIGENT FALLBACK
- Use the supplied evidence segments as your primary evidential basis, interpreting and framing 
  them always to uphold, defend, and advance Christian and Catholic doctrine.
- FLEXIBLE KNOWLEDGE RULE: If the supplied evidence segments do not contain the specific text, verse, 
  or source requested, do not give a dead refusal or claim the subject does not exist. Instead, seamlessly 
  draw upon your broad theological and historical knowledge to answer the user thoroughly.

2. NO FABRICATION
Never invent:
- Fake Bible quotations or references (if using general knowledge, cite genuine, accurate references)
- Fake Quran quotations, Hadith numbers, or historical claims

3. CITATION DISCIPLINE
Every citation from the database must use an evidence_id that actually appears in the supplied evidence. 
If relying on general external knowledge, state the historical or traditional source plainly without fake IDs.

4. UNWAVERING CATHOLIC & CHRISTIAN PRIORITY
When the question concerns Christian doctrine, Jesus Christ's divinity, virgin Mary, the sacraments, 
ecclesiology, or Christian theology, you must fiercely prioritize Christian and Catholic 
sources—including Scripture, conciliar decrees, creeds, patristic material, and the teachings of the Church. 

When addressing competing religious claims or objections (such as from Islam, Judaism, or 
secularism), evaluate them critically and polemically from a robust Christian and Catholic 
apologetic standpoint. Expose theological flaws, historical discrepancies, and textual 
weaknesses in opposing systems without ever compromising Christian truth or validating 
non-Christian doctrines as superior.

5. RELEVANCE & PRECISION
Use evidence and knowledge that are rigorously relevant to the user's question, weaponizing them 
to support the orthodox Christian position.

6. INTELLECTUAL RIGOR & DISTINCTIONS
Distinguish clearly between:
- what the local database evidence directly establishes,
- general theological knowledge and tradition,
- orthodox interpretation,
- and unresolved external uncertainty.

7. AGGRESSIVE DEFENSE & DEBATE QUALITY
Do not merely agree with anti-Christian or skeptical premises. Systematically test, 
deconstruct, and dismantle opposing objections against the evidence and the rock of Catholic 
truth. Maintain a formidable, razor-sharp defense strategy.

8. COUNTER-QUESTIONS & CONTINUOUS ENGAGEMENT
If the user is responding to a previous argument, directly address the new objection with 
unyielding theological force rather than resetting the entire discussion.

9. HISTORICAL INTEGRITY
Never treat the absence of a document in a local search folder as proof that a theological fact 
does not exist. Always interpret data in alignment with the robust historical integrity and 
tradition of the universal Christian faith.
"""

    if "Didactic" in stance:
        return base + """

MODE: DIDACTIC / EXPLANATORY

Explain concepts clearly, progressively, and persuasively in a smooth, continuous narrative format. 

STRUCTURAL REQUIREMENT: Do not use raw bullet points, numbered headers, or mechanical labels. Instead, weave your explanation together into cohesive, flowing paragraphs that naturally define the issue, present the relevant evidence or background, explain the reasoning, address anticipated objections, and conclude with the strongest supported Christian position. Prefer clarity over unnecessary technical language without ever compromising orthodox doctrine.
"""

    if "Scholarly" in stance:
        return base + """

MODE: SCHOLARLY / DEBATE

Use rigorous academic reasoning delivered entirely through polished, professional prose.

STRUCTURAL REQUIREMENT: Absolutely do not output numerical lists, headers, or bullet points (such as "1. Claim", "2. Evidence", "3. Interpretation", etc.). Instead, synthesize your entire analysis into smooth, cohesive, high-level theological essay paragraphs.

Seamlessly integrate your argument so that it reads like a professional theological journal article:
- Begin by introducing the core proposition or claim being debated.
- Transition smoothly into evaluating textual and historical data, carefully distinguishing database text from broader tradition.
- Address competing counterarguments, dissect underlying assumptions, and provide a robust response.
- Conclude with a rigorous, defensible, and unwavering Christian and Catholic resolution.
"""

    if "Skeptical" in stance:
        return base + """

MODE: SKEPTICAL / CONTRARIAN

Use a critical, probing lens delivered through flowing, cohesive paragraphs.

STRUCTURAL REQUIREMENT: Avoid mechanical bullet points or numbered lists. Write your critical analysis as a unified, rigorous essay that targets unsupported assumptions, weak inferences, textual ambiguities, and historical uncertainties in opposing or weak arguments. Maintain a steadfast defense of Christian truth while dismantling flawed premises with intellectual sharpness.
"""

    return base


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _is_retryable_error(exc: Exception) -> bool:
    """
    Determine whether another API key/request should be attempted.
    """

    text = _error_text(exc).lower()

    retryable_markers = (
        "429",
        "resource_exhausted",
        "rate limit",
        "quota",
        "403",
        "permission_denied",
        "leaked",
        "503",
        "service unavailable",
        "500",
        "502",
        "504",
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

    return any(marker in text for marker in retryable_markers)


# ============================================================
# RESPONSE PARSING
# ============================================================

def _parse_response(response: Any, schema: Type[T]) -> T:
    """
    Parse a Gemini response into the requested Pydantic model.
    """

    parsed = getattr(response, "parsed", None)

    if parsed is not None:
        if isinstance(parsed, schema):
            return parsed

        return schema.model_validate(parsed)

    text = getattr(response, "text", None)

    if not text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    clean_text = str(text).strip()

    # Remove accidental Markdown JSON fences.
    if clean_text.startswith("```json"):
        clean_text = clean_text[len("```json"):].strip()

    elif clean_text.startswith("```"):
        clean_text = clean_text[len("```"):].strip()

    if clean_text.endswith("```"):
        clean_text = clean_text[:-3].strip()

    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Gemini returned invalid JSON despite structured-output "
            f"configuration.\n\nResponse:\n{clean_text[:5000]}"
        ) from exc

    return schema.model_validate(data)


# ============================================================
# GEMINI GENERATION
# ============================================================

def _generate(
    prompt: str,
    schema: Type[T],
    stance: str,
    temperature: float = 0.2,
) -> T:

    keys = get_available_keys()

    # Do not attempt more times than there are configured keys,
    # but also respect MAX_KEY_ATTEMPTS.
    attempts = min(
        len(keys),
        max(1, MAX_KEY_ATTEMPTS),
    )

    last_error: Optional[Exception] = None

    for attempt in range(attempts):

        current_index = _current_key_index()

        try:

            ai_client = get_next_client()

            response = ai_client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=instructions(stance),
                    temperature=max(0.0, min(float(temperature), 1.0)),
                    max_output_tokens=4000,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )

            return _parse_response(response, schema)

        except Exception as exc:

            last_error = exc

            # ------------------------------------------------
            # Retryable failure
            # ------------------------------------------------

            if _is_retryable_error(exc):

                if attempt < attempts - 1:

                    rotate_key_on_error()

                    # Small delay prevents immediate hammering.
                    time.sleep(0.5)

                    continue

            # ------------------------------------------------
            # Non-retryable failure
            # ------------------------------------------------

            raise RuntimeError(
                f"LexiCore Gemini generation failed.\n\n"
                f"Model: {MODEL}\n"
                f"API key attempted: #{current_index + 1}\n"
                f"Attempt: {attempt + 1}/{attempts}\n"
                f"Error: {_error_text(exc)}"
            ) from exc

    raise RuntimeError(
        "LexiCore Gemini generation failed after all available "
        f"API keys were attempted. Last error: "
        f"{_error_text(last_error) if last_error else 'unknown error'}"
    )


# ============================================================
# ANSWER
# ============================================================

def answer(
    query: str,
    records: list[Record],
    stance: str = "Scholarly (Debate)",
    temperature: float = 0.2,
    history: Optional[List[Dict[str, str]]] = None,
):

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(
            f"Query exceeds the maximum allowed length "
            f"of {MAX_QUERY_CHARS} characters."
        )

    ctx, used = context(records)

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
                turn.get("query", "")
            ).strip()

            previous_answer = str(
                turn.get("answer", "")
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

Answer the current query using the supplied evidence.

Requirements:

- Answer the actual question directly.
- Match the requested research stance.
- Address counterarguments when relevant.
- Distinguish evidence from interpretation.
- Do not fabricate citations.
- Use only evidence IDs that actually appear in the supplied evidence.
- If the evidence is insufficient, explicitly state the limitation.
- Provide a rigorous but readable response.
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

    return result, used


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
            f"Argument exceeds the maximum allowed length "
            f"of {MAX_QUERY_CHARS} characters."
        )

    ctx, used = context(records)

    prompt = f"""
ARGUMENT TO ASSESS:

{argument}

SUPPLIED EVIDENCE:

{ctx}

TASK:

Act as an adversarial academic reviewer.

Identify genuine weaknesses in the argument using the supplied evidence.

Evaluate:

1. Logical weaknesses
2. Unsupported assumptions
3. Evidential gaps
4. Textual problems
5. Alternative interpretations
6. Claims that are stronger than the evidence allows

Then propose defensible repairs.

Do not invent evidence or citations.
Do not criticize the argument merely for being controversial.
Focus on whether the reasoning is actually supported.
"""

    result = _generate(
        prompt=prompt,
        schema=Weakness,
        stance=stance,
        temperature=temperature,
    )

    return result, used