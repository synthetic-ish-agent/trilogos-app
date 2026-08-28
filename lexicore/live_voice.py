from __future__ import annotations

import datetime as dt
import os
from typing import Optional

import streamlit as st
from google import genai


LIVE_MODEL = os.getenv(
    "LEXICORE_LIVE_MODEL",
    "gemini-3.1-flash-live-preview",
)

LIVE_TOKEN_MINUTES = int(
    os.getenv(
        "LEXICORE_LIVE_TOKEN_MINUTES",
        "30",
    )
)


def _get_api_key() -> str:
    """
    Use the same API-key configuration already used by
    LexiCore.

    The permanent key never gets placed in the browser.
    """

    keys = []

    for i in range(1, 101):

        name = f"GOOGLE_API_KEY_{i}"

        value = None

        try:
            value = st.secrets.get(name)
        except Exception:
            value = None

        if not value:
            value = os.getenv(name)

        if not value:
            break

        value = str(value).strip()

        if value:
            keys.append(value)

    if not keys:

        value = None

        try:
            value = st.secrets.get(
                "GOOGLE_API_KEY"
            )
        except Exception:
            value = None

        if not value:
            value = os.getenv(
                "GOOGLE_API_KEY"
            )

        if value:
            value = str(value).strip()

            if value:
                keys.append(value)

    if not keys:
        raise RuntimeError(
            "No Google API key is configured."
        )

    # Use the same first key for token creation.
    #
    # The token itself is short-lived and restricted to
    # the Live API model.
    return keys[0]


@st.cache_resource(
    show_spinner=False
)
def _token_client(
    api_key: str,
) -> genai.Client:

    return genai.Client(
        api_key=api_key
    )


def create_live_token(
    system_instruction: str,
) -> str:
    """
    Create a short-lived Gemini Live ephemeral token.

    The permanent API key remains server-side.
    """

    api_key = _get_api_key()

    client = _token_client(
        api_key
    )

    now = dt.datetime.now(
        dt.timezone.utc
    )

    expire_time = (
        now
        + dt.timedelta(
            minutes=LIVE_TOKEN_MINUTES
        )
    )

    new_session_expire_time = (
        now
        + dt.timedelta(
            minutes=2
        )
    )

    config = {
        "uses": 1,
        "expire_time": expire_time,
        "new_session_expire_time": (
            new_session_expire_time
        ),
        "live_connect_constraints": {
            "model": LIVE_MODEL,
            "config": {
                "response_modalities": [
                    "AUDIO"
                ],
                "input_audio_transcription": {},
                "output_audio_transcription": {},
                "session_resumption": {},
                "system_instruction": {
                    "parts": [
                        {
                            "text": system_instruction
                        }
                    ]
                },
            },
        },
    }

    token = client.auth_tokens.create(
        config=config
    )

    token_name = getattr(
        token,
        "name",
        None,
    )

    if not token_name:
        raise RuntimeError(
            "Gemini did not return a Live API token."
        )

    return token_name
