from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional

from google.genai import types

from .llm import (
    get_next_client,
    rotate_key_on_error,
    instructions,
)


LIVE_MODEL = "gemini-3.1-flash-live-preview"

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000

AUDIO_MIME_TYPE = "audio/pcm;rate=16000"


@dataclass
class LiveVoiceState:
    """
    Thread-safe state shared between Streamlit/WebRTC and
    the Gemini Live worker.
    """

    input_queue: queue.Queue = field(
        default_factory=lambda: queue.Queue(
            maxsize=100
        )
    )

    output_queue: queue.Queue = field(
        default_factory=lambda: queue.Queue(
            maxsize=200
        )
    )

    transcript_queue: queue.Queue = field(
        default_factory=lambda: queue.Queue(
            maxsize=200
        )
    )

    error_queue: queue.Queue = field(
        default_factory=lambda: queue.Queue(
            maxsize=20
        )
    )

    stop_event: threading.Event = field(
        default_factory=threading.Event
    )

    worker: Optional[threading.Thread] = None

    connected: bool = False


def _put_nowait_bounded(
    q: queue.Queue,
    item,
):
    """
    Put data into a queue without allowing a slow consumer
    to block realtime audio processing.
    """

    try:
        q.put_nowait(item)

    except queue.Full:

        try:
            q.get_nowait()
        except queue.Empty:
            pass

        try:
            q.put_nowait(item)
        except queue.Full:
            pass


def _safe_error_text(exc: Exception) -> str:

    text = str(exc)

    # Never expose API keys.
    for attr in (
        "GOOGLE_API_KEY",
        "GOOGLE_API_KEY_1",
        "GOOGLE_API_KEY_2",
    ):
        text = text.replace(
            attr,
            "[REDACTED]",
        )

    return text[:500]


async def _send_audio_loop(
    session,
    state: LiveVoiceState,
):
    """
    Continuously forwards microphone PCM audio to Gemini.
    """

    while not state.stop_event.is_set():

        try:

            audio = await asyncio.to_thread(
                state.input_queue.get,
                True,
                0.1,
            )

        except queue.Empty:
            continue

        if audio is None:
            continue

        try:

            await session.send_realtime_input(
                audio=types.Blob(
                    data=audio,
                    mime_type=AUDIO_MIME_TYPE,
                )
            )

        except Exception:

            if state.stop_event.is_set():
                break

            raise


async def _receive_loop(
    session,
    state: LiveVoiceState,
):
    """
    Receive Gemini's streamed audio and transcription.
    """

    async for response in session.receive():

        if state.stop_event.is_set():
            break

        # ----------------------------------------------------
        # User speech transcription
        # ----------------------------------------------------

        server_content = getattr(
            response,
            "server_content",
            None,
        )

        if server_content:

            input_transcription = getattr(
                server_content,
                "input_transcription",
                None,
            )

            if input_transcription:

                text = getattr(
                    input_transcription,
                    "text",
                    None,
                )

                if text:
                    _put_nowait_bounded(
                        state.transcript_queue,
                        (
                            "user",
                            text,
                        ),
                    )

            # ------------------------------------------------
            # Model output
            # ------------------------------------------------

            model_turn = getattr(
                server_content,
                "model_turn",
                None,
            )

            if model_turn:

                parts = getattr(
                    model_turn,
                    "parts",
                    None,
                )

                if parts:

                    for part in parts:

                        inline_data = getattr(
                            part,
                            "inline_data",
                            None,
                        )

                        if not inline_data:
                            continue

                        audio_data = getattr(
                            inline_data,
                            "data",
                            None,
                        )

                        if audio_data:

                            _put_nowait_bounded(
                                state.output_queue,
                                audio_data,
                            )

        # ----------------------------------------------------
        # Text response / transcription where available
        # ----------------------------------------------------

        response_text = getattr(
            response,
            "text",
            None,
        )

        if response_text:

            _put_nowait_bounded(
                state.transcript_queue,
                (
                    "assistant",
                    response_text,
                ),
            )


async def _live_session(
    state: LiveVoiceState,
    stance: str,
    history_text: str,
    language: str,
    temperature: float,
):
    """
    Maintain one Gemini Live session.
    """

    client = get_next_client()

    system_prompt = instructions(
        stance
    )

    voice_instruction = f"""
You are now operating in LIVE VOICE CONVERSATION mode.

You are THE ARMOR / LexiCore.

{system_prompt}

VOICE CONVERSATION RULES:

Speak naturally and conversationally.

Do not sound like you are reading an academic paper.

Give direct answers first, then explain.

Keep normal spoken responses reasonably concise.

When the user asks a difficult theological question, reason carefully
before answering.

Maintain the same Christian and Catholic apologetic position required
by the main THE ARMOR system.

Do not invent Bible references, Quran references, Hadith numbers,
historical claims, or evidence IDs.

The user's selected display language is:

{language}

If the user speaks another supported language, respond naturally
in that language when appropriate.

Current conversation context:

{history_text}
"""

    config = types.LiveConnectConfig(
        response_modalities=[
            "AUDIO"
        ],
        system_instruction=voice_instruction,
        temperature=max(
            0.0,
            min(
                float(temperature),
                1.0,
            ),
        ),
        input_audio_transcription=(
            types.AudioTranscriptionConfig()
        ),
        speech_config={
            "voice_config": {
                "prebuilt_voice_config": {
                    "voice_name": "Kore"
                }
            }
        },
        realtime_input_config={
            "automatic_activity_detection": {
                "disabled": False,
            }
        },
    )

    async with client.aio.live.connect(
        model=LIVE_MODEL,
        config=config,
    ) as session:

        state.connected = True

        sender = asyncio.create_task(
            _send_audio_loop(
                session,
                state,
            )
        )

        receiver = asyncio.create_task(
            _receive_loop(
                session,
                state,
            )
        )

        try:

            await asyncio.wait(
                [
                    sender,
                    receiver,
                ],
                return_when=asyncio.FIRST_EXCEPTION,
            )

            for task in (
                sender,
                receiver,
            ):

                if not task.done():
                    task.cancel()

            for task in (
                sender,
                receiver,
            ):

                try:
                    await task
                except asyncio.CancelledError:
                    pass

        finally:

            state.connected = False


def _worker_main(
    state: LiveVoiceState,
    stance: str,
    history_text: str,
    language: str,
    temperature: float,
):
    """
    Dedicated thread containing its own asyncio event loop.

    This keeps the Gemini Live WebSocket away from Streamlit's
    synchronous execution thread.
    """

    attempts = 2

    for attempt in range(
        attempts
    ):

        if state.stop_event.is_set():
            return

        try:

            asyncio.run(
                _live_session(
                    state=state,
                    stance=stance,
                    history_text=history_text,
                    language=language,
                    temperature=temperature,
                )
            )

            return

        except Exception as exc:

            state.connected = False

            if (
                attempt
                >= attempts - 1
            ):

                _put_nowait_bounded(
                    state.error_queue,
                    _safe_error_text(exc),
                )

                return

            rotate_key_on_error()


def start_live_voice(
    state: LiveVoiceState,
    stance: str,
    history_text: str = "",
    language: str = "English",
    temperature: float = 0.2,
):
    """
    Start Gemini Live in a background thread.
    """

    if (
        state.worker
        and state.worker.is_alive()
    ):
        return

    state.stop_event.clear()

    state.worker = threading.Thread(
        target=_worker_main,
        kwargs={
            "state": state,
            "stance": stance,
            "history_text": history_text,
            "language": language,
            "temperature": temperature,
        },
        daemon=True,
        name="armor-gemini-live",
    )

    state.worker.start()


def stop_live_voice(
    state: LiveVoiceState,
):
    """
    Stop the Live session.
    """

    state.stop_event.set()

    _put_nowait_bounded(
        state.input_queue,
        None,
    )

    state.connected = False


def push_microphone_audio(
    state: LiveVoiceState,
    pcm_bytes: bytes,
):
    """
    Push browser microphone PCM into Gemini's input queue.
    """

    if not pcm_bytes:
        return

    if state.stop_event.is_set():
        return

    _put_nowait_bounded(
        state.input_queue,
        pcm_bytes,
    )


def read_output_audio(
    state: LiveVoiceState,
):
    """
    Return one available Gemini output audio chunk.
    """

    try:
        return state.output_queue.get_nowait()

    except queue.Empty:
        return None


def read_transcript(
    state: LiveVoiceState,
):
    """
    Return one available transcript event.
    """

    try:
        return state.transcript_queue.get_nowait()

    except queue.Empty:
        return None


def read_error(
    state: LiveVoiceState,
):
    """
    Return one available error.
    """

    try:
        return state.error_queue.get_nowait()

    except queue.Empty:
        return None


def build_voice_history(
    history: list,
    max_turns: int = 6,
) -> str:
    """
    Compact recent conversation for the Live session.
    """

    if not history:
        return ""

    recent = history[
        -max_turns:
    ]

    parts = []

    for turn in recent:

        query = str(
            turn.get(
                "query",
                "",
            )
        ).strip()

        answer = str(
            turn.get(
                "original_answer",
                turn.get(
                    "answer",
                    "",
                ),
            )
        ).strip()

        if query:
            parts.append(
                f"User: {query}"
            )

        if answer:
            parts.append(
                f"THE ARMOR: {answer[:5000]}"
            )

    return "\n".join(parts)
