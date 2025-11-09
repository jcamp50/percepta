from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

from py.config import settings
from py.memory.vector_store import VectorStore
from py.memory.video_store import VideoStore
from py.memory.chat_store import ChatStore
from py.utils.embeddings import embed_text
from .retriever import Retriever, RetrievalParams


@dataclass
class RetrievedChunk:
    id: str
    channel_id: str
    text: str
    started_at: datetime
    ended_at: datetime
    cosine_distance: float
    score: float

    @property
    def midpoint(self) -> datetime:
        return self.started_at + (self.ended_at - self.started_at) / 2


DEFAULT_CONTEXT_CHAR_LIMIT = 2000
DEFAULT_COMPLETION_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.4
MAX_LLM_ATTEMPTS = 4


def _format_timestamp(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S")


class RAGService:
    def __init__(
        self,
        *,
        vector_store: Optional[VectorStore] = None,
        video_store: Optional[VideoStore] = None,
        chat_store: Optional[ChatStore] = None,
        openai_client: Optional[OpenAI] = None,
        context_char_limit: Optional[int] = None,
        completion_model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        half_life_minutes: Optional[int] = None,
        prefilter_limit: Optional[int] = None,
        compressor: Optional[Callable[[Sequence[str], int], Sequence[str]]] = None,
        enable_critic: bool = False,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.video_store = video_store
        self.chat_store = chat_store
        self.retriever = Retriever(
            vector_store=self.vector_store,
            video_store=self.video_store,
            chat_store=self.chat_store,
        )
        self.context_char_limit = context_char_limit or getattr(
            settings, "rag_context_char_limit", DEFAULT_CONTEXT_CHAR_LIMIT
        )
        self.completion_model = completion_model or getattr(
            settings, "rag_completion_model", DEFAULT_COMPLETION_MODEL
        )
        self.temperature = (
            temperature
            if temperature is not None
            else getattr(settings, "rag_temperature", DEFAULT_TEMPERATURE)
        )
        self.top_k = top_k or getattr(settings, "rag_top_k", 5)
        self.half_life_minutes = half_life_minutes or getattr(
            settings, "rag_half_life_minutes", 60
        )
        inferred_prefilter = prefilter_limit or getattr(
            settings, "rag_prefilter_limit", max(self.top_k * 4, self.top_k)
        )
        self.prefilter_limit = max(inferred_prefilter, self.top_k)
        self._compressor = compressor
        self._enable_critic = enable_critic

        self._system_prompt = system_prompt or self._load_prompt_template(
            settings.rag_system_prompt_file
        )
        self._user_prompt_template = user_prompt_template or self._load_prompt_template(
            settings.rag_user_prompt_file
        )

        if openai_client is not None:
            self._client = openai_client
        else:
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for RAG responses")
            self._client = OpenAI(api_key=settings.openai_api_key)

    async def _retrieve_chunks(
        self,
        *,
        channel_id: str,
        question: str,
        top_k: int,
        half_life_minutes: int,
        prefilter_limit: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        t0 = time.monotonic()
        query_embedding = await embed_text(question)
        t_embed = time.monotonic()

        results = await self.retriever.retrieve(
            query_embedding=query_embedding,
            params=RetrievalParams(
                channel_id=channel_id,
                limit=top_k,
                half_life_minutes=half_life_minutes,
                prefilter_limit=prefilter_limit or max(top_k, 1),
            ),
        )
        t_search = time.monotonic()

        chunks: List[RetrievedChunk] = []
        for row in results:
            chunks.append(
                RetrievedChunk(
                    id=row.id,
                    channel_id=row.channel_id,
                    text=row.text,
                    started_at=row.started_at,
                    ended_at=row.ended_at,
                    cosine_distance=row.cosine_distance,
                    score=row.score,
                )
            )
        # Simple telemetry log
        try:
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(
                {
                    "telemetry": "rag.retrieve",
                    "embed_ms": int((t_embed - t0) * 1000),
                    "search_ms": int((t_search - t_embed) * 1000),
                    "hits": len(chunks),
                    "best_score": min((c.score for c in chunks), default=None),
                }
            )
        except Exception:
            pass

        return chunks

    def _select_context(
        self, chunks: Sequence[RetrievedChunk]
    ) -> Tuple[List[RetrievedChunk], List[str]]:
        selected: List[RetrievedChunk] = []
        formatted: List[str] = []
        budget = self.context_char_limit

        for chunk in chunks:
            text = chunk.text.strip()
            timestamp = _format_timestamp(chunk.midpoint)
            line = f"[{timestamp}] {text}"

            if not selected:
                selected.append(chunk)
                formatted.append(line)
                budget -= len(line) + 1
                continue

            if len(line) + 1 > budget:
                break

            selected.append(chunk)
            formatted.append(line)
            budget -= len(line) + 1

        # Allow optional compressor for agentic extensions
        if self._compressor is not None:
            formatted = list(self._compressor(formatted, self.context_char_limit))
        return selected, formatted

    def _build_citations(self, chunks: Sequence[RetrievedChunk]) -> List[dict]:
        citations: List[dict] = []
        for chunk in chunks:
            citations.append(
                {
                    "id": chunk.id,
                    "timestamp": f"(~{_format_timestamp(chunk.midpoint)})",
                }
            )
        return citations

    def _format_conversation_history(
        self, questions: List[str], answers: List[str]
    ) -> str:
        """
        Format conversation history for inclusion in prompt.

        Args:
            questions: List of previous questions
            answers: List of previous answers

        Returns:
            Formatted conversation history string
        """
        if not questions or not answers:
            return ""

        history_lines = ["Previous conversation:"]
        # Match up Q&A pairs (should be same length, but handle gracefully)
        pairs = list(zip(questions, answers))
        for i, (q, a) in enumerate(pairs, 1):
            history_lines.append(f"Q{i}: {q}")
            history_lines.append(f"A{i}: {a}")
            history_lines.append("")  # Empty line between pairs

        return "\n".join(history_lines)

    async def _build_messages(
        self,
        *,
        question: str,
        formatted_context: Sequence[str],
        channel_id: Optional[str] = None,
        conversation_history: Optional[str] = None,
    ) -> List[dict]:
        context_section = (
            "\n".join(formatted_context) if formatted_context else "(no context)"
        )
        
        # Include conversation history if provided (empty string if not provided)
        conversation_history_section = conversation_history if conversation_history else ""
        user_prompt = self._format_prompt(
            self._user_prompt_template,
            context_section=context_section,
            question=question,
            conversation_history=conversation_history_section,
        )

        # Fetch latest channel metadata and format system prompt
        system_prompt = await self._format_system_prompt_with_metadata(
            channel_id=channel_id
        )

        if formatted_context and any("[Video Frame]" in line for line in formatted_context):
            system_prompt = (
                f"{system_prompt}\n\n"
                "Visual context: Lines beginning with [Video Frame] are rich descriptions of on-screen content; "
                "treat them as authoritative when reasoning about visuals."
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def _format_system_prompt_with_metadata(
        self, channel_id: Optional[str] = None
    ) -> str:
        """Format system prompt with current channel metadata."""
        if not channel_id or not self.vector_store:
            # No metadata available, use base prompt
            return self._system_prompt

        try:
            snapshot = await self.vector_store.get_latest_channel_snapshot(channel_id)
            if not snapshot:
                return self._system_prompt

            # Extract metadata
            payload = snapshot.get("payload_json") or {}
            streamer_name = payload.get("channel_name", "the streamer")
            game_name = snapshot.get("game_name") or payload.get("game_name")
            viewer_count = snapshot.get("viewer_count") or payload.get("viewer_count")
            title = snapshot.get("title") or payload.get("title")
            is_live = payload.get("is_live", False)

            # Build stream context string
            context_parts = []
            if is_live:
                if game_name:
                    context_parts.append(
                        f"Currently livestreaming in the {game_name} category"
                    )
                if viewer_count is not None:
                    context_parts.append(f"with {viewer_count:,} viewers")
                if title:
                    context_parts.append(f'Stream title: "{title}"')
            else:
                context_parts.append("Currently offline")
                if game_name:
                    context_parts.append(f"Last category: {game_name}")
                if title:
                    context_parts.append(f'Last title: "{title}"')

            stream_context = ""
            if context_parts:
                stream_context = " " + ". ".join(context_parts) + "."

            # Format system prompt with metadata
            try:
                return self._system_prompt.format(
                    streamer_name=streamer_name,
                    stream_context=stream_context,
                )
            except KeyError:
                # Template doesn't have placeholders, append metadata
                return f"{self._system_prompt}\n\nStream context:{stream_context}"

        except Exception as e:
            # Log error but don't fail - fall back to base prompt
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to fetch metadata for system prompt: {e}")
            return self._system_prompt

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _resolve_prompt_path(cls, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return cls._project_root() / candidate

    @classmethod
    def _load_prompt_template(cls, path: str) -> str:
        resolved = cls._resolve_prompt_path(path)
        try:
            text = resolved.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Prompt template file not found: {resolved}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Failed to read prompt template file: {resolved}"
            ) from exc
        cleaned = text.strip()
        if not cleaned:
            raise ValueError(f"Prompt template file is empty: {resolved}")
        return cleaned

    @staticmethod
    def _format_prompt(template: str, **kwargs: str) -> str:
        try:
            return template.format(**kwargs)
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(
                f"Prompt template is missing placeholder '{{{missing}}}'"
            ) from exc

    async def _call_llm(self, *, messages: List[dict]) -> str:
        delay_seconds = 0.5

        for attempt in range(MAX_LLM_ATTEMPTS):
            try:
                response = await asyncio.to_thread(
                    self._client.chat.completions.create,
                    model=self.completion_model,
                    messages=messages,
                    temperature=self.temperature,
                )

                choice = response.choices[0].message
                return choice.content if choice and choice.content else ""

            except (RateLimitError, APITimeoutError, APIConnectionError, APIError):
                if attempt == MAX_LLM_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(delay_seconds + random.random() * 0.25)
                delay_seconds *= 2

        return ""

    async def answer(
        self,
        *,
        channel_id: str,
        question: str,
        top_k: Optional[int] = None,
        half_life_minutes: Optional[int] = None,
        prefilter_limit: Optional[int] = None,
        user_id: Optional[str] = None,
        session_manager: Optional[object] = None,
    ) -> dict:
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question must not be empty")

        limit = top_k or self.top_k
        half_life = half_life_minutes or self.half_life_minutes
        prefilter = max(prefilter_limit or self.prefilter_limit, limit)

        # Retrieve conversation history if user_id and session_manager provided
        conversation_history = None
        if user_id and session_manager and channel_id:
            try:
                session = await session_manager.get_session(user_id, channel_id)
                questions = session.get("last_questions", [])
                answers = session.get("last_answers", [])
                if questions and answers:
                    conversation_history = self._format_conversation_history(
                        questions, answers
                    )
            except Exception as e:
                # Log error but don't fail - graceful degradation
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to retrieve conversation history: {e}")

        chunks = await self._retrieve_chunks(
            channel_id=channel_id,
            question=cleaned_question,
            top_k=limit,
            half_life_minutes=half_life,
            prefilter_limit=prefilter,
        )

        if not chunks:
            # If we have conversation history, try to answer from that even without chunks
            if conversation_history:
                # Build prompts with conversation history but no chunks
                messages = await self._build_messages(
                    question=cleaned_question,
                    formatted_context=[],
                    channel_id=channel_id,
                    conversation_history=conversation_history,
                )
                system_prompt = messages[0]["content"]
                user_prompt_text = messages[1]["content"]
                
                # Try to generate answer from conversation history
                try:
                    answer_text = await self._call_llm(messages)
                    if answer_text and len(answer_text.strip()) > 20:  # Substantive answer
                        return {
                            "answer": answer_text,
                            "citations": [],
                            "chunks": [],
                            "context": [],
                            "prompts": {
                                "system": system_prompt,
                                "user": user_prompt_text,
                            },
                        }
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to generate answer from conversation history: {e}")
            
            # No chunks and no usable conversation history - fallback
            fallback = "I don't have enough context to answer that right now."
            # Build prompts for consistency with response schema
            messages = await self._build_messages(
                question=cleaned_question,
                formatted_context=[],
                channel_id=channel_id,
                conversation_history=conversation_history,
            )
            system_prompt = messages[0]["content"]
            user_prompt_text = messages[1]["content"]
            return {
                "answer": fallback,
                "citations": [],
                "chunks": [],
                "context": [],
                "prompts": {
                    "system": system_prompt,
                    "user": user_prompt_text,
                },
            }

        selected_chunks, formatted_context = self._select_context(chunks)

        if not selected_chunks:
            # If we have conversation history, try to answer from that even without selected chunks
            if conversation_history:
                # Build prompts with conversation history but no chunks
                messages = await self._build_messages(
                    question=cleaned_question,
                    formatted_context=[],
                    channel_id=channel_id,
                    conversation_history=conversation_history,
                )
                system_prompt = messages[0]["content"]
                user_prompt_text = messages[1]["content"]
                
                # Try to generate answer from conversation history
                try:
                    answer_text = await self._call_llm(messages)
                    if answer_text and len(answer_text.strip()) > 20:  # Substantive answer
                        return {
                            "answer": answer_text,
                            "citations": [],
                            "chunks": [],
                            "context": [],
                            "prompts": {
                                "system": system_prompt,
                                "user": user_prompt_text,
                            },
                        }
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to generate answer from conversation history: {e}")
            
            fallback = "I don't have enough context to answer that right now."
            # Build prompts even when no chunks selected
            messages = await self._build_messages(
                question=cleaned_question,
                formatted_context=[],
                channel_id=channel_id,
                conversation_history=conversation_history,
            )
            system_prompt = messages[0]["content"]
            user_prompt_text = messages[1]["content"]
            return {
                "answer": fallback,
                "citations": [],
                "chunks": [],
                "context": [],
                "prompts": {
                    "system": system_prompt,
                    "user": user_prompt_text,
                },
            }

        messages = await self._build_messages(
            question=cleaned_question,
            formatted_context=formatted_context,
            channel_id=channel_id,
            conversation_history=conversation_history,
        )
        system_prompt = messages[0]["content"]
        user_prompt_text = messages[1]["content"]

        t_llm0 = time.monotonic()
        answer_text = await self._call_llm(messages=messages)
        t_llm1 = time.monotonic()
        answer_text = answer_text.strip()
        if not answer_text:
            answer_text = "I don't have enough context to answer that right now."

        citations = self._build_citations(selected_chunks)

        # Optional critic: ensure a (~HH:MM:SS) appears in the output
        if self._enable_critic and "(~" not in answer_text:
            first_ts = citations[0]["timestamp"] if citations else ""
            if first_ts:
                answer_text = f"{answer_text} {first_ts}".strip()
        chunk_payload = [
            {
                "id": chunk.id,
                "channel_id": chunk.channel_id,
                "started_at": chunk.started_at,
                "ended_at": chunk.ended_at,
                "midpoint": chunk.midpoint,
                "timestamp": _format_timestamp(chunk.midpoint),
                "text": chunk.text,
                "score": chunk.score,
                "cosine_distance": chunk.cosine_distance,
            }
            for chunk in selected_chunks
        ]

        out = {
            "answer": answer_text,
            "citations": citations,
            "chunks": chunk_payload,
            "context": formatted_context,
            "prompts": {
                "system": system_prompt,
                "user": user_prompt_text,
            },
        }

        # Telemetry for LLM
        try:
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(
                {
                    "telemetry": "rag.llm",
                    "llm_ms": int((t_llm1 - t_llm0) * 1000),
                    "context_lines": len(formatted_context),
                    "answer_chars": len(answer_text),
                }
            )
        except Exception:
            pass

        return out
