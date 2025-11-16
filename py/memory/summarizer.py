from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI
from sqlalchemy import Integer, String, bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

try:
    from openai import (
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
        APIStatusError,
    )
except ImportError:
    RateLimitError = APIConnectionError = APITimeoutError = APIStatusError = Exception

from py.config import settings
from py.database.connection import SessionLocal
from py.database.models import Summary
from py.memory.chat_store import ChatStore
from py.memory.vector_store import VectorStore
from py.memory.video_store import VideoStore
from py.utils.embeddings import embed_text
from py.utils.logging import get_logger

logger = get_logger(__name__, category="summarizer")
summary_logger = get_logger(__name__, category="summary")

LOG_BASE_PATH = Path(__file__).resolve().parents[2] / "logs" / "summaries"
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 0.6
MAX_DELAY_SECONDS = 6.0

PROMPT_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1] / "reason" / "prompts" / "summary_prompt.txt"
)
SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "reason"
    / "prompts"
    / "summary_system_prompt.txt"
)


@lru_cache()
def _load_prompt_template() -> str:
    """Load summary prompt template from file."""
    if not PROMPT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Summary prompt template not found: {PROMPT_TEMPLATE_PATH}"
        )
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()


@lru_cache()
def _load_system_prompt() -> str:
    """Load summary system prompt from file."""
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Summary system prompt not found: {SYSTEM_PROMPT_PATH}"
        )
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _json_to_text_for_embedding(json_dict: Dict[str, Any]) -> str:
    """Convert JSON summary dict to text representation for embeddings.

    This preserves semantic markers (field names) while creating a readable
    text representation suitable for embedding generation.

    Args:
        json_dict: JSON dict with summary structure

    Returns:
        Text representation of the JSON structure
    """
    if not json_dict:
        return ""

    parts: List[str] = []

    # Segment info
    if segment := json_dict.get("segment"):
        start_time = segment.get("start_time", "")
        end_time = segment.get("end_time", "")
        if start_time and end_time:
            parts.append(f"Segment: {start_time} to {end_time}")

    # Key events
    if key_events := json_dict.get("key_events"):
        if isinstance(key_events, list) and key_events:
            event_parts = []
            for event in key_events:
                if isinstance(event, dict):
                    timestamp = event.get("timestamp", "")
                    event_name = event.get("event", "")
                    description = event.get("description", "")
                    if event_name or description:
                        event_str = f"{timestamp}: {event_name}"
                        if description:
                            event_str += f" - {description}"
                        event_parts.append(event_str)
            if event_parts:
                parts.append(f"Key events: {'; '.join(event_parts)}")

    # Visual context
    if visual_context := json_dict.get("visual_context"):
        visual_parts = []
        if primary_activities := visual_context.get("primary_activities"):
            if isinstance(primary_activities, list):
                visual_parts.append(f"Activities: {', '.join(primary_activities)}")
        if ui_elements := visual_context.get("notable_ui_elements"):
            if isinstance(ui_elements, list):
                visual_parts.append(f"UI elements: {', '.join(ui_elements)}")
        if scene_changes := visual_context.get("scene_changes"):
            if isinstance(scene_changes, list):
                visual_parts.append(f"Scene changes: {', '.join(scene_changes)}")
        if visual_parts:
            parts.append(f"Visual context: {'; '.join(visual_parts)}")

    # Chat highlights
    if chat_highlights := json_dict.get("chat_highlights"):
        if isinstance(chat_highlights, list) and chat_highlights:
            chat_parts = []
            for chat in chat_highlights:
                if isinstance(chat, dict):
                    timestamp = chat.get("timestamp", "")
                    username = chat.get("username", "")
                    message = chat.get("message", "")
                    significance = chat.get("significance", "")
                    if username or message:
                        chat_str = f"{timestamp} {username}: {message}"
                        if significance:
                            chat_str += f" ({significance})"
                        chat_parts.append(chat_str)
            if chat_parts:
                parts.append(f"Chat highlights: {'; '.join(chat_parts)}")

    # Streamer commentary
    if commentary := json_dict.get("streamer_commentary"):
        parts.append(f"Streamer commentary: {commentary}")

    # Tone
    if tone := json_dict.get("tone"):
        parts.append(f"Tone: {tone}")

    return " | ".join(parts)


class Summarizer:
    """Memory-propagated summarizer for long-term context retention.

    Generates 2-minute segment summaries that propagate forward, maintaining
    constant token budget while preserving long-term temporal dynamics.
    """

    def __init__(
        self,
        *,
        video_store: VideoStore,
        vector_store: VectorStore,
        chat_store: ChatStore,
        completion_model: Optional[str] = None,
        session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
    ) -> None:
        self.video_store = video_store
        self.vector_store = vector_store
        self.chat_store = chat_store
        self.completion_model = completion_model or settings.rag_completion_model
        self.session_factory: async_sessionmaker[AsyncSession] = (
            session_factory or SessionLocal
        )

    def _write_response_log(
        self,
        *,
        channel_id: str,
        start_time: datetime,
        end_time: datetime,
        response_payload: Dict[str, Any],
        suffix: str,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Log summarization API calls to logs/summaries/<channel_id>/<timestamp>_<suffix>.json"""
        try:
            timestamp = start_time.strftime("%Y%m%dT%H%M%S%fZ")
            channel_dir = LOG_BASE_PATH / channel_id
            channel_dir.mkdir(parents=True, exist_ok=True)
            filename = channel_dir / f"{timestamp}_{suffix}.json"
            log_data: Dict[str, Any] = {
                "channel_id": channel_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "prompt": prompt,
                "system_prompt": system_prompt,
                "response": response_payload,
            }
            with filename.open("w", encoding="utf-8") as fh:
                json.dump(log_data, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            summary_logger.warning(
                "Failed to write summary log for channel %s: %s",
                channel_id,
                exc,
            )

    def _build_context_json(
        self,
        *,
        transcripts: List[Dict[str, Any]],
        video_frames: List[Dict[str, Any]],
        chat_messages: List[Dict[str, Any]],
        previous_summary: Optional[str] = None,
        segment_start: datetime,
        segment_end: datetime,
    ) -> str:
        """Build JSON context structure from transcripts, video frames, chat, and previous summary.

        Groups data by video frames (10s windows) with overlapping transcripts and chat.

        Args:
            transcripts: List of transcript dicts with text, started_at, ended_at
            video_frames: List of video frame dicts with description_json, captured_at
            chat_messages: List of chat message dicts with username, message, sent_at
            previous_summary: Optional previous segment summary for continuity
            segment_start: Segment start time
            segment_end: Segment end time

        Returns:
            JSON string with structured context
        """
        from datetime import timedelta

        # Sort frames by captured_at
        sorted_frames = sorted(
            video_frames, key=lambda f: f.get("captured_at", segment_start)
        )

        windows: List[Dict[str, Any]] = []

        # Group data by video frames (10s windows)
        for frame in sorted_frames:
            captured_at = frame.get("captured_at")
            if not captured_at:
                continue

            # Create window around frame (±5 seconds)
            window_start = captured_at - timedelta(seconds=5)
            window_end = captured_at + timedelta(seconds=5)

            # Get video frame description (prefer JSON, fall back to text)
            description_json = frame.get("description_json")
            description_text = frame.get("description")

            video_frame_data: Dict[str, Any] = {
                "timestamp": (
                    captured_at.isoformat()
                    if isinstance(captured_at, datetime)
                    else str(captured_at)
                ),
            }

            if description_json:
                video_frame_data["description"] = description_json
            elif description_text:
                # Old format - use text description
                video_frame_data["description"] = description_text
            else:
                # Fallback to image path
                video_frame_data["description"] = frame.get("image_path", "")

            # Find overlapping transcripts
            window_transcripts: List[Dict[str, Any]] = []
            for transcript in transcripts:
                started_at = transcript.get("started_at")
                ended_at = transcript.get("ended_at")
                if not started_at or not ended_at:
                    continue

                # Check if transcript overlaps with window
                # Transcript overlaps if it starts before window_end and ends after window_start
                if started_at < window_end and ended_at > window_start:
                    window_transcripts.append(
                        {
                            "timestamp": (
                                started_at.isoformat()
                                if isinstance(started_at, datetime)
                                else str(started_at)
                            ),
                            "text": transcript.get("text", ""),
                        }
                    )

            # Find overlapping chat messages
            window_chat: List[Dict[str, Any]] = []
            for chat in chat_messages:
                sent_at = chat.get("sent_at")
                if not sent_at:
                    continue

                # Check if chat message is within window
                if window_start <= sent_at < window_end:
                    window_chat.append(
                        {
                            "timestamp": (
                                sent_at.isoformat()
                                if isinstance(sent_at, datetime)
                                else str(sent_at)
                            ),
                            "username": chat.get("username", ""),
                            "message": chat.get("message", ""),
                        }
                    )

            windows.append(
                {
                    "window_start": (
                        window_start.isoformat()
                        if isinstance(window_start, datetime)
                        else str(window_start)
                    ),
                    "window_end": (
                        window_end.isoformat()
                        if isinstance(window_end, datetime)
                        else str(window_end)
                    ),
                    "video_frame": video_frame_data,
                    "transcripts": window_transcripts,
                    "chat_reactions": window_chat,
                }
            )

        # Build JSON structure
        context_data: Dict[str, Any] = {
            "segment": {
                "start_time": (
                    segment_start.isoformat()
                    if isinstance(segment_start, datetime)
                    else str(segment_start)
                ),
                "end_time": (
                    segment_end.isoformat()
                    if isinstance(segment_end, datetime)
                    else str(segment_end)
                ),
                "windows": windows,
            }
        }

        # Add previous summary if provided
        if previous_summary:
            context_data["previous_summary"] = previous_summary

        # If no windows and no previous summary, return empty structure
        if not windows and not previous_summary:
            context_data["segment"]["windows"] = []
            context_data["note"] = (
                "No transcripts, video frames, or chat messages available for this segment"
            )

        return json.dumps(context_data, indent=2)

    async def _generate_summary(
        self,
        *,
        channel_id: str,
        start_time: datetime,
        end_time: datetime,
        context: str,
    ) -> Dict[str, Any]:
        """Generate summary using OpenAI Responses API.

        Args:
            channel_id: Broadcaster channel ID
            start_time: Segment start time
            end_time: Segment end time
            context: JSON context string

        Returns:
            Generated summary as JSON dict
        """
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for summary generation.")

        try:
            client = OpenAI(api_key=settings.openai_api_key, max_retries=0)
        except TypeError:
            client = OpenAI(api_key=settings.openai_api_key)

        # Load prompts from files
        system_prompt = _load_system_prompt()
        prompt_template = _load_prompt_template()
        prompt = prompt_template.format(context=context)

        def _call_openai() -> Dict[str, Any]:
            request_body = {
                "model": self.completion_model,
                "temperature": 0.4,
                "max_output_tokens": 800,
                "instructions": system_prompt,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                        ],
                    },
                ],
            }
            response = client.post(
                "/responses",
                body=request_body,
                cast_to=Dict[str, Any],
            )
            return response

        def _strip_markdown_code_blocks(text: str) -> str:
            """Strip markdown code blocks from text (e.g., ```json ... ```).
            
            Args:
                text: Text that may contain markdown code blocks
                
            Returns:
                Text with markdown code blocks removed
            """
            text = text.strip()
            # Remove ```json or ``` at the start
            if text.startswith("```"):
                # Find the first newline after ```
                first_newline = text.find("\n")
                if first_newline != -1:
                    text = text[first_newline + 1 :]
                else:
                    # No newline, just remove ```
                    text = text[3:]
            # Remove ``` at the end
            if text.endswith("```"):
                text = text[:-3]
            return text.strip()

        def _strip_trailing_commas(text: str) -> str:
            """Remove trailing commas before closing braces/brackets."""
            return re.sub(r',(\s*[}\]])', r'\1', text)

        def _parse_json_text(raw_text: str) -> Dict[str, Any]:
            """Parse cleaned JSON text, repairing trailing commas if needed."""
            cleaned_text = _strip_markdown_code_blocks(raw_text)
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                repaired_text = _strip_trailing_commas(cleaned_text)
                if repaired_text != cleaned_text:
                    try:
                        return json.loads(repaired_text)
                    except json.JSONDecodeError:
                        pass
                raise

        def _extract_content(response: Dict[str, Any]) -> Dict[str, Any]:
            """Extract JSON content from OpenAI response and parse it.
            
            Returns:
                Parsed JSON dict with summary structure
            """
            if not isinstance(response, dict):
                raise RuntimeError(
                    "Unexpected response type from OpenAI Responses API."
                )

            # First, check for the convenience field if present
            convenience_text = response.get("output_text")
            if isinstance(convenience_text, str) and convenience_text.strip():
                try:
                    return _parse_json_text(convenience_text)
                except json.JSONDecodeError as exc:
                    summary_logger.debug(
                        "Failed to parse output_text as JSON: %s. Raw text: %s",
                        exc,
                        convenience_text[:200] if convenience_text else "empty",
                    )
                    raise RuntimeError(f"Failed to parse JSON from output_text: {exc}")

            # Check output array
            output_items = response.get("output") or []
            if not output_items:
                summary_logger.error(
                    "No output items found in response. Response keys: %s. Full response: %s",
                    list(response.keys()) if isinstance(response, dict) else "not a dict",
                    json.dumps(response, indent=2, default=str)[:2000],
                )
                raise RuntimeError("OpenAI returned empty summary content (no output items).")
            
            # Try to extract from output items
            for item in output_items:
                if not isinstance(item, dict):
                    continue
                
                # Check if item has text directly (some response formats)
                if "text" in item:
                    text_value = item.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        try:
                            return _parse_json_text(text_value)
                        except json.JSONDecodeError:
                            pass
                
                # Check message type items
                if item.get("type") != "message":
                    continue

                content = item.get("content") or []
                text_parts: List[str] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type == "output_text":
                        text_value = part.get("text")
                        if isinstance(text_value, str) and text_value.strip():
                            text_parts.append(text_value)
                    elif part_type == "output_image_text":
                        text_value = part.get("text")
                        if isinstance(text_value, str) and text_value.strip():
                            text_parts.append(text_value)
                    # Also check for direct text field
                    elif "text" in part:
                        text_value = part.get("text")
                        if isinstance(text_value, str) and text_value.strip():
                            text_parts.append(text_value)

                if text_parts:
                    json_text = "".join(text_parts).strip()
                    if not json_text:
                        summary_logger.debug(
                            "Found text_parts but joined result is empty. Parts: %s",
                            text_parts,
                        )
                        continue
                    try:
                        return _parse_json_text(json_text)
                    except json.JSONDecodeError as exc:
                        summary_logger.debug(
                            "Failed to parse JSON. Error: %s. Text (first 500 chars): %s",
                            exc,
                            json_text[:500],
                        )
                        raise RuntimeError(f"Failed to parse JSON from response: {exc}")

            # If we get here, we didn't find any parseable content
            summary_logger.error(
                "Could not extract content from response. Response structure: %s",
                json.dumps(response, indent=2, default=str)[:2000],
            )
            raise RuntimeError("OpenAI returned empty summary content (no parseable text found).")

        async def _call_with_retry() -> Dict[str, Any]:
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
                    summary_logger.warning(
                        "Rate limit (attempt %s/%s) for channel %s. Retrying in %.2fs",
                        attempt,
                        MAX_RETRIES,
                        channel_id,
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
                    summary_logger.warning(
                        "Transient error (attempt %s/%s) for channel %s: %s. Retrying in %.2fs",
                        attempt,
                        MAX_RETRIES,
                        channel_id,
                        exc,
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                except APIStatusError as exc:
                    last_error = exc
                    status = getattr(exc, "status_code", None)
                    if status == 429 and attempt < MAX_RETRIES:
                        delay = min(
                            MAX_DELAY_SECONDS,
                            BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
                        )
                        jitter = random.uniform(0, 0.3)
                        wait_time = delay + jitter
                        summary_logger.warning(
                            "Status 429 (attempt %s/%s) for channel %s. Retrying in %.2fs",
                            attempt,
                            MAX_RETRIES,
                            channel_id,
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
            raise RuntimeError("Failed to generate summary without specific error.")

        try:
            response = await _call_with_retry()
            content = _extract_content(response)
            try:
                if isinstance(response, dict):
                    payload = response
                elif hasattr(response, "model_dump"):
                    payload = response.model_dump()
                elif hasattr(response, "json"):
                    payload = json.loads(response.json())
                else:
                    payload = {"raw": str(response)}
            except Exception:
                payload = {"raw": str(response)}
            self._write_response_log(
                channel_id=channel_id,
                start_time=start_time,
                end_time=end_time,
                response_payload=payload,
                suffix="success",
                prompt=prompt,
                system_prompt=system_prompt,
            )
            summary_logger.info(
                "Generated summary for channel %s between %s and %s",
                channel_id,
                start_time.isoformat(),
                end_time.isoformat(),
            )
            return content
        except Exception as exc:
            logger.error(
                "Failed to generate summary for channel %s between %s and %s: %s",
                channel_id,
                start_time.isoformat(),
                end_time.isoformat(),
                exc,
            )
            summary_logger.error(
                "Failed summary for channel %s between %s and %s: %s",
                channel_id,
                start_time.isoformat(),
                end_time.isoformat(),
                exc,
            )
            # Load prompts for error logging (may not be defined if error occurred early)
            try:
                error_system_prompt = _load_system_prompt()
                error_prompt = prompt if "prompt" in locals() else None
            except Exception:
                error_system_prompt = None
                error_prompt = None
            self._write_response_log(
                channel_id=channel_id,
                start_time=start_time,
                end_time=end_time,
                response_payload={"error": str(exc)},
                suffix="error",
                prompt=error_prompt,
                system_prompt=error_system_prompt,
            )
            raise

    async def summarize_segment(
        self,
        *,
        channel_id: str,
        start_time: datetime,
        end_time: datetime,
        previous_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate summary for a 2-minute segment.

        Args:
            channel_id: Broadcaster channel ID
            start_time: Segment start time
            end_time: Segment end time
            previous_summary: Optional previous segment summary for continuity

        Returns:
            Generated summary text
        """
        # 1. Generate descriptions for lazy frames first
        lazy_frames = await self.video_store.get_lazy_frames_in_range(
            channel_id=channel_id,
            start_time=start_time,
            end_time=end_time,
        )

        if lazy_frames:
            logger.debug(
                "Backfilling %d lazy frame descriptions for channel %s between %s and %s",
                len(lazy_frames),
                channel_id,
                start_time.isoformat(),
                end_time.isoformat(),
            )
            await asyncio.gather(
                *[
                    self.video_store.generate_description_for_frame(frame_id)
                    for frame_id in lazy_frames
                ]
            )

        # 2. Retrieve all data in segment
        transcripts = await self.vector_store.get_range(
            channel_id=channel_id, start_time=start_time, end_time=end_time
        )
        video_frames = await self.video_store.get_range(
            channel_id=channel_id, start_time=start_time, end_time=end_time
        )
        chat_messages = await self.chat_store.get_range(
            channel_id=channel_id, start_time=start_time, end_time=end_time
        )

        # 3. Build context
        context = self._build_context_json(
            transcripts=transcripts,
            video_frames=video_frames,
            chat_messages=chat_messages,
            previous_summary=previous_summary,
            segment_start=start_time,
            segment_end=end_time,
        )

        # 4. Generate summary via LLM
        summary = await self._generate_summary(
            channel_id=channel_id,
            start_time=start_time,
            end_time=end_time,
            context=context,
        )

        return summary

    async def insert_summary(
        self,
        *,
        channel_id: str,
        start_time: datetime,
        end_time: datetime,
        summary_json: Dict[str, Any],
        segment_number: int,
    ) -> str:
        """Store summary in database with embedding.

        Args:
            channel_id: Broadcaster channel ID
            start_time: Segment start time
            end_time: Segment end time
            summary_json: Generated summary as JSON dict
            segment_number: Segment number for ordering

        Returns:
            Summary ID as string
        """
        import uuid

        # Generate text representation from JSON for description field
        summary_text = _json_to_text_for_embedding(summary_json)

        # Generate embedding from JSON string for semantic search
        json_string = json.dumps(summary_json, sort_keys=True)
        embedding = await embed_text(json_string)

        new_id = uuid.uuid4()
        async with self.session_factory() as session:
            entity = Summary(
                id=new_id,
                channel_id=channel_id,
                start_time=start_time,
                end_time=end_time,
                summary_text=summary_text,
                summary_json=summary_json,
                embedding=embedding,
                segment_number=segment_number,
            )
            session.add(entity)
            await session.commit()

        logger.debug(
            "Stored summary %s for channel %s (segment %d)",
            new_id,
            channel_id,
            segment_number,
        )
        return str(new_id)

    async def get_previous_summary(
        self,
        *,
        channel_id: str,
        segment_number: int,
    ) -> Optional[str]:
        """Get the immediately previous summary (segment_number - 1).

        Args:
            channel_id: Broadcaster channel ID
            segment_number: Current segment number

        Returns:
            Previous summary text, or None if not found
        """
        if segment_number <= 1:
            return None

        sql = """
        SELECT summary_text
        FROM summaries
        WHERE channel_id = :channel_id
          AND segment_number = :prev_segment
        ORDER BY segment_number DESC
        LIMIT 1
        """

        async with self.session_factory() as session:
            result = await session.execute(
                text(sql),
                {
                    "channel_id": channel_id,
                    "prev_segment": segment_number - 1,
                },
            )
            row = result.mappings().first()
            if row:
                return row["summary_text"]
            return None

    async def get_next_segment_number(
        self,
        *,
        channel_id: str,
    ) -> int:
        """Get the next segment number for a channel.

        Args:
            channel_id: Broadcaster channel ID

        Returns:
            Next segment number (1 if no summaries exist)
        """
        sql = """
        SELECT MAX(segment_number) AS max_segment
        FROM summaries
        WHERE channel_id = :channel_id
        """

        async with self.session_factory() as session:
            result = await session.execute(
                text(sql),
                {"channel_id": channel_id},
            )
            row = result.mappings().first()
            if row and row["max_segment"] is not None:
                return int(row["max_segment"]) + 1
            return 1

    async def get_most_recent_summary(
        self,
        *,
        channel_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent summary (highest segment_number) for a channel.

        Args:
            channel_id: Broadcaster channel ID

        Returns:
            Summary dict with id, summary_text, start_time, end_time, segment_number,
            or None if no summaries exist
        """
        sql = """
        SELECT id, channel_id, summary_text, summary_json, start_time, end_time, segment_number
        FROM summaries
        WHERE channel_id = :channel_id
        ORDER BY segment_number DESC
        LIMIT 1
        """

        async with self.session_factory() as session:
            result = await session.execute(
                text(sql),
                {"channel_id": channel_id},
            )
            row = result.mappings().first()
            if row:
                return {
                    "id": str(row["id"]),
                    "channel_id": row["channel_id"],
                    "summary_text": row["summary_text"],
                    "summary_json": row.get("summary_json"),
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "segment_number": row["segment_number"],
                }
            return None

    async def summarize_with_propagation(
        self,
        *,
        channel_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> tuple[Dict[str, Any], str]:
        """Generate summary with memory propagation (includes previous summary).

        Args:
            channel_id: Broadcaster channel ID
            start_time: Segment start time
            end_time: Segment end time

        Returns:
            Tuple of (summary_json, summary_id)
        """
        # Get next segment number
        segment_number = await self.get_next_segment_number(channel_id=channel_id)

        # Get previous summary (N-1) for continuity
        previous_summary = await self.get_previous_summary(
            channel_id=channel_id, segment_number=segment_number
        )

        # Generate summary with propagated memory
        summary_json = await self.summarize_segment(
            channel_id=channel_id,
            start_time=start_time,
            end_time=end_time,
            previous_summary=previous_summary,
        )

        # Store summary with embedding
        summary_id = await self.insert_summary(
            channel_id=channel_id,
            start_time=start_time,
            end_time=end_time,
            summary_json=summary_json,
            segment_number=segment_number,
        )

        return (summary_json, summary_id)

    async def select_relevant_summaries(
        self,
        *,
        query_embedding: List[float],
        channel_id: str,
        limit: int = 5,
        half_life_minutes: int = 60,
        prefilter_limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Select relevant summaries using semantic search with time-decay.

        Args:
            query_embedding: Query vector embedding (1536 dimensions)
            channel_id: Broadcaster channel ID
            limit: Maximum number of results to return
            half_life_minutes: Half-life for time decay scoring
            prefilter_limit: Maximum candidates to consider before time decay

        Returns:
            List of summary dicts with id, summary_text, start_time, end_time, score
        """
        if prefilter_limit < limit:
            prefilter_limit = max(limit, 1)

        # Format vector literal
        vec_str = "[" + ",".join(f"{v:.8f}" for v in query_embedding) + "]"
        half_life_seconds = half_life_minutes * 60

        cosine_order = "embedding <=> (:vec)::vector"
        sql = f"""
            WITH pre AS (
            SELECT id, channel_id, summary_text, summary_json, start_time, end_time, segment_number,
                    {cosine_order} AS dist
            FROM summaries
            WHERE channel_id = :channel_id
            AND embedding IS NOT NULL
            ORDER BY {cosine_order}
            LIMIT :k
            )
            SELECT id, channel_id, summary_text, summary_json, start_time, end_time, segment_number, dist,
                dist / POWER(2, EXTRACT(EPOCH FROM (NOW() - end_time)) / :half_life_seconds) AS score
            FROM pre
            ORDER BY score ASC
            LIMIT :limit
            """

        async with self.session_factory() as session:
            stmt = text(sql).bindparams(
                bindparam("vec", type_=String()),
                bindparam("channel_id", type_=String()),
                bindparam("k", type_=Integer()),
                bindparam("half_life_seconds", type_=Integer()),
                bindparam("limit", type_=Integer()),
            )
            result = await session.execute(
                stmt,
                {
                    "vec": vec_str,
                    "channel_id": channel_id,
                    "k": prefilter_limit,
                    "half_life_seconds": half_life_seconds,
                    "limit": limit,
                },
            )
            rows = result.mappings().all()

        output: List[Dict[str, Any]] = []
        for row in rows:
            output.append(
                {
                    "id": str(row["id"]),
                    "channel_id": row["channel_id"],
                    "summary_text": row["summary_text"],
                    "summary_json": row.get("summary_json"),
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "segment_number": row["segment_number"],
                    "cosine_distance": float(row["dist"]),
                    "score": float(row["score"]),
                }
            )
        return output

    async def is_segment_summarized(
        self,
        *,
        channel_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> bool:
        """Check if a segment has already been summarized.

        Args:
            channel_id: Broadcaster channel ID
            start_time: Segment start time
            end_time: Segment end time

        Returns:
            True if segment is already summarized, False otherwise
        """
        sql = """
        SELECT COUNT(*) AS count
        FROM summaries
        WHERE channel_id = :channel_id
          AND start_time = :start_time
          AND end_time = :end_time
        """

        async with self.session_factory() as session:
            result = await session.execute(
                text(sql),
                {
                    "channel_id": channel_id,
                    "start_time": start_time,
                    "end_time": end_time,
                },
            )
            row = result.mappings().first()
            return int(row["count"]) > 0 if row else False

    async def get_active_channels(
        self,
        *,
        window_minutes: int = 10,
    ) -> List[str]:
        """Get list of channels with recent activity (transcripts or video frames).

        Args:
            window_minutes: Time window to check for activity (default: 10 minutes)

        Returns:
            List of channel IDs with recent activity
        """
        sql = """
        SELECT DISTINCT channel_id
        FROM (
            SELECT channel_id, ended_at AS activity_time
            FROM transcripts
            WHERE ended_at >= NOW() - (:window_minutes * INTERVAL '1 minute')
            UNION
            SELECT channel_id, captured_at AS activity_time
            FROM video_frames
            WHERE captured_at >= NOW() - (:window_minutes * INTERVAL '1 minute')
        ) AS active_channels
        ORDER BY channel_id
        """

        async with self.session_factory() as session:
            result = await session.execute(
                text(sql),
                {"window_minutes": window_minutes},
            )
            rows = result.mappings().all()

        return [row["channel_id"] for row in rows]

    async def _process_segment_for_channel(
        self,
        *,
        channel_id: str,
        segment_start: datetime,
        segment_end: datetime,
    ) -> Optional[str]:
        """Process a single segment for a channel.

        Args:
            channel_id: Broadcaster channel ID
            segment_start: Segment start time
            segment_end: Segment end time

        Returns:
            Summary ID if successful, None otherwise
        """
        try:
            # Check if already summarized
            if await self.is_segment_summarized(
                channel_id=channel_id,
                start_time=segment_start,
                end_time=segment_end,
            ):
                logger.debug(
                    "Segment already summarized for channel %s: %s to %s",
                    channel_id,
                    segment_start.isoformat(),
                    segment_end.isoformat(),
                )
                return None

            # Generate summary with propagation
            summary_text, summary_id = await self.summarize_with_propagation(
                channel_id=channel_id,
                start_time=segment_start,
                end_time=segment_end,
            )

            logger.info(
                "Generated summary %s for channel %s (segment %s to %s)",
                summary_id,
                channel_id,
                segment_start.isoformat(),
                segment_end.isoformat(),
            )

            return summary_id
        except Exception as exc:
            logger.error(
                "Failed to process segment for channel %s (%s to %s): %s",
                channel_id,
                segment_start.isoformat(),
                segment_end.isoformat(),
                exc,
                exc_info=True,
            )
            return None

    async def get_channel_id_from_data(self) -> Optional[str]:
        """Get channel_id (broadcaster ID) from existing data in database.

        For single-channel-per-instance architecture, queries the database
        for the channel_id that has recent activity. This avoids needing
        to convert channel name to broadcaster ID via API.

        Returns:
            Channel ID (broadcaster ID) if found, None otherwise
        """
        sql = """
        SELECT DISTINCT channel_id
        FROM (
            SELECT channel_id, ended_at AS activity_time
            FROM transcripts
            WHERE ended_at >= NOW() - (10 * INTERVAL '1 minute')
            UNION
            SELECT channel_id, captured_at AS activity_time
            FROM video_frames
            WHERE captured_at >= NOW() - (10 * INTERVAL '1 minute')
        ) AS active_channels
        ORDER BY channel_id
        LIMIT 1
        """

        async with self.session_factory() as session:
            result = await session.execute(text(sql))
            row = result.mappings().first()
            if row:
                return row["channel_id"]
            return None

    async def run_summarization_job(self) -> None:
        """Background job that runs every 2 minutes to generate summaries.

        Processes segments that ended 2+ minutes ago to ensure all data is captured.
        Assumes single-channel-per-instance architecture.
        Gets channel_id from existing data in database (avoids API lookup).
        """
        try:
            # Get channel ID from database (single-channel-per-instance)
            # Query for channel_id that has recent activity
            channel_id = await self.get_channel_id_from_data()

            if not channel_id:
                logger.debug(
                    "No channel with recent activity found, skipping summarization"
                )
                return

            # Calculate segment boundaries (2-minute windows)
            now = datetime.now(timezone.utc)
            # Process segments that ended 2+ minutes ago
            # Round down to nearest 2-minute boundary
            current_minute = now.minute
            segment_minute = (current_minute // 2) * 2
            latest_segment_end = now.replace(
                minute=segment_minute, second=0, microsecond=0
            )
            # Process segment that ended 2 minutes before the latest
            target_segment_end = latest_segment_end - timedelta(minutes=2)
            target_segment_start = target_segment_end - timedelta(minutes=2)

            logger.debug(
                "Processing summarization job for channel %s, target segment: %s to %s",
                channel_id,
                target_segment_start.isoformat(),
                target_segment_end.isoformat(),
            )

            # Process single channel
            summary_id = await self._process_segment_for_channel(
                channel_id=channel_id,
                segment_start=target_segment_start,
                segment_end=target_segment_end,
            )

            if summary_id:
                logger.info(
                    "Summarization job completed successfully for channel %s (summary: %s)",
                    channel_id,
                    summary_id,
                )
            else:
                logger.debug(
                    "Summarization job skipped for channel %s (segment already processed or no data)",
                    channel_id,
                )
        except Exception as exc:
            logger.error("Summarization job failed: %s", exc, exc_info=True)

    def start_background_job(self) -> None:
        """Start the background summarization job that runs every 2 minutes."""

        async def _job_loop() -> None:
            while True:
                try:
                    await self.run_summarization_job()
                except Exception as exc:
                    logger.error(
                        "Background summarization job error: %s", exc, exc_info=True
                    )
                # Wait 2 minutes before next run
                await asyncio.sleep(120)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("No running event loop to start background job")
            return

        logger.info("Starting background summarization job (runs every 2 minutes)")
        loop.create_task(_job_loop())
