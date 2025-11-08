from __future__ import annotations

import asyncio
import os
from base64 import b64encode
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import random

from openai import OpenAI

try:
    from openai import (
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
        APIStatusError,
    )
except ImportError:  # Backwards compatibility for older client versions
    RateLimitError = APIConnectionError = APITimeoutError = APIStatusError = Exception

from py.config import settings
from py.utils.logging import get_logger

logger = get_logger(__name__, category="video")
description_logger = get_logger(__name__, category="video_description")

PROMPT_MAX_SUMMARY_CHARS = 600
PROMPT_MAX_TRANSCRIPT_CHARS = 400
PROMPT_MAX_CHAT_MESSAGES = 5
PROMPT_MAX_CHAT_CHARS = 200
PROMPT_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "reason"
    / "prompts"
    / "video_description_prompt.txt"
)
SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "reason"
    / "prompts"
    / "video_description_system_prompt.txt"
)

MAX_RETRIES = 3
BASE_DELAY_SECONDS = 0.6
MAX_DELAY_SECONDS = 6.0
LOG_BASE_PATH = Path(__file__).resolve().parents[2] / "logs" / "video_description"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


@lru_cache()
def _load_prompt_template() -> str:
    if not PROMPT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Video description prompt template not found: {PROMPT_TEMPLATE_PATH}"
        )
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()


@lru_cache()
def _load_system_prompt() -> str:
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Video description system prompt not found: {SYSTEM_PROMPT_PATH}"
        )
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _write_response_log(
    *,
    channel_id: str,
    captured_at: datetime,
    response_payload: Dict[str, Any],
    suffix: str,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> None:
    try:
        timestamp = captured_at.strftime("%Y%m%dT%H%M%S%fZ")
        channel_dir = LOG_BASE_PATH / channel_id
        channel_dir.mkdir(parents=True, exist_ok=True)
        filename = channel_dir / f"{timestamp}_{suffix}.json"
        log_data: Dict[str, Any] = {
            "channel_id": channel_id,
            "captured_at": captured_at.isoformat(),
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response": response_payload,
        }
        with filename.open("w", encoding="utf-8") as fh:
            json.dump(log_data, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        description_logger.warning(
            "Failed to write video description log for channel %s: %s",
            channel_id,
            exc,
        )


def _build_description_prompt(
    *,
    previous_frame_description: Optional[str] = None,
    recent_summary: Optional[str] = None,
    transcript: Optional[Dict[str, str]] = None,
    chat: Optional[List[Dict[str, str]]] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    context_parts: List[str] = []

    if previous_frame_description:
        context_parts.append(
            f"Previous frame context: {_truncate(previous_frame_description, PROMPT_MAX_SUMMARY_CHARS)}"
        )

    if recent_summary:
        context_parts.append(
            f"Recent stream summary: {_truncate(recent_summary, PROMPT_MAX_SUMMARY_CHARS)}"
        )

    if transcript and transcript.get("text"):
        context_parts.append(
            f'Streamer is saying: "{_truncate(transcript["text"], PROMPT_MAX_TRANSCRIPT_CHARS)}"'
        )

    if chat:
        limited_chat = chat[:PROMPT_MAX_CHAT_MESSAGES]
        chat_lines = [
            f'{c.get("username", "viewer")}: {_truncate(c.get("message", ""), PROMPT_MAX_CHAT_CHARS)}'
            for c in limited_chat
        ]
        if chat_lines:
            context_parts.append(f"Chat discussion: {' | '.join(chat_lines)}")

    if metadata:
        metadata_tokens = []
        game_name = metadata.get("game_name")
        if game_name:
            metadata_tokens.append(f"Game: {game_name}")
        category = metadata.get("category")
        if category and category != game_name:
            metadata_tokens.append(f"Category: {category}")
        title = metadata.get("title")
        if title:
            metadata_tokens.append(f'Title: "{title}"')
        tags = metadata.get("tags")
        if isinstance(tags, (list, tuple)):
            tag_list = [str(tag) for tag in tags if tag]
            if tag_list:
                metadata_tokens.append("Tags: " + ", ".join(tag_list[:6]))
        if metadata_tokens:
            context_parts.append(" | ".join(metadata_tokens))

    context_block = f'CONTEXT: {" | ".join(context_parts)}' if context_parts else ""

    template = _load_prompt_template()
    return template.format(context_block=context_block)


async def generate_frame_description(
    *,
    image_path: str,
    channel_id: str,
    captured_at: datetime,
    previous_frame_description: Optional[str] = None,
    recent_summary: Optional[str] = None,
    transcript: Optional[Dict[str, str]] = None,
    chat: Optional[List[Dict[str, str]]] = None,
    metadata: Optional[Dict[str, str]] = None,
    openai_client: Optional[OpenAI] = None,
) -> str:
    """Generate a high-detail textual description for a video frame using GPT-4o-mini."""
    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for visual description generation."
        )

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found for description: {image_path}")

    if openai_client is not None:
        client = openai_client
    else:
        try:
            client = OpenAI(api_key=settings.openai_api_key, max_retries=0)
        except TypeError:
            client = OpenAI(api_key=settings.openai_api_key)

    with open(image_path, "rb") as image_file:
        base64_image = b64encode(image_file.read()).decode("utf-8")

    prompt = _build_description_prompt(
        previous_frame_description=previous_frame_description,
        recent_summary=recent_summary,
        transcript=transcript,
        chat=chat,
        metadata=metadata,
    )
    system_prompt = _load_system_prompt()

    def _call_openai() -> Any:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            max_tokens=500,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
        )
        return response

    def _extract_content(response: Any) -> str:
        choice = response.choices[0]
        if not choice.message or not choice.message.content:
            raise RuntimeError("OpenAI returned empty description content.")
        return choice.message.content

    async def _call_with_retry() -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return await asyncio.to_thread(_call_openai)
            except RateLimitError as exc:
                last_error = exc
                if attempt == MAX_RETRIES:
                    break
                delay = min(
                    MAX_DELAY_SECONDS, BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                )
                jitter = random.uniform(0, 0.3)
                wait_time = delay + jitter
                description_logger.warning(
                    "GPT-4o-mini rate limit (attempt %s/%s) for channel %s at %s. Retrying in %.2fs",
                    attempt,
                    MAX_RETRIES,
                    channel_id,
                    captured_at.isoformat(),
                    wait_time,
                )
                await asyncio.sleep(wait_time)
            except (APIConnectionError, APITimeoutError) as exc:
                last_error = exc
                if attempt == MAX_RETRIES:
                    break
                delay = min(
                    MAX_DELAY_SECONDS, BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                )
                jitter = random.uniform(0, 0.3)
                wait_time = delay + jitter
                description_logger.warning(
                    "GPT-4o-mini transient error (attempt %s/%s) for channel %s at %s: %s. Retrying in %.2fs",
                    attempt,
                    MAX_RETRIES,
                    channel_id,
                    captured_at.isoformat(),
                    exc,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
            except APIStatusError as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                if status == 429 and attempt < MAX_RETRIES:
                    delay = min(
                        MAX_DELAY_SECONDS, BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                    )
                    jitter = random.uniform(0, 0.3)
                    wait_time = delay + jitter
                    description_logger.warning(
                        "GPT-4o-mini status 429 (attempt %s/%s) for channel %s at %s. Retrying in %.2fs",
                        attempt,
                        MAX_RETRIES,
                        channel_id,
                        captured_at.isoformat(),
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                break
            except Exception as exc:
                last_error = exc
                break

        if last_error:
            raise last_error
        raise RuntimeError("Failed to generate description without specific error.")

    try:
        response = await _call_with_retry()
        content = _extract_content(response)
        try:
            payload = (
                response.model_dump()
                if hasattr(response, "model_dump")
                else json.loads(response.json())
            )
        except Exception:
            payload = {"raw": str(response)}
        _write_response_log(
            channel_id=channel_id,
            captured_at=captured_at,
            response_payload=payload,
            suffix="success",
            prompt=prompt,
            system_prompt=system_prompt,
        )
        logger.debug(
            "Generated visual description for channel %s at %s",
            channel_id,
            captured_at.isoformat(),
        )
        description_logger.info(
            "Generated visual description for channel %s at %s",
            channel_id,
            captured_at.isoformat(),
        )
        return content.strip()
    except Exception as exc:
        logger.error(
            "Failed to generate visual description for channel %s at %s: %s",
            channel_id,
            captured_at.isoformat(),
            exc,
        )
        description_logger.error(
            "Failed visual description for channel %s at %s: %s",
            channel_id,
            captured_at.isoformat(),
            exc,
        )
        _write_response_log(
            channel_id=channel_id,
            captured_at=captured_at,
            response_payload={"error": str(exc)},
            suffix="error",
            prompt=prompt,
            system_prompt=system_prompt,
        )
        raise
