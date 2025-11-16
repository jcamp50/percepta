"""
Percepta Python Service - FastAPI Application

This service:
- Receives chat messages from Node IRC service
- (Future) Processes with RAG/LLM
- Returns responses for Node to send to Twitch

Why FastAPI?
- Modern Python web framework
- Automatic API documentation
- Type validation with Pydantic
- Async support (for future AI processing)
- Fast and production-ready
"""

import json
import logging
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from py.config import settings
from schemas.messages import (
    ChatMessage,
    MessageReceived,
    SendRequest,
    SendResponse,
    ChatResponse,
    RAGQueryRequest,
    RAGAnswerResponse,
    TranscriptionResponse,
    AudioStreamUrlResponse,
)
from py.reason.rag import RAGService
from py.ingest.transcription import TranscriptionService
from py.ingest.twitch import EventSubWebSocketClient
from py.ingest.metadata import ChannelMetadataPoller
from py.ingest.event_handler import EventHandler
from py.memory.vector_store import VectorStore
from py.memory.video_store import VideoStore
from py.memory.chat_store import ChatStore
from py.ingest.video import determine_capture_interval
from py.memory.redis_session import RedisSessionManager
from py.utils.embeddings import embed_text
from py.utils.logging import get_logger
from py.utils.content_filter import is_safe_for_chat, filter_response_content
from py.utils.twitch_api import get_broadcaster_id_from_channel_name

RAG_ASK_LOG_ROOT = Path(__file__).resolve().parents[1] / "logs" / "rag_tests" / "ask"

# Configure logging
# TASK: Set up Python logging
# - Use logging.basicConfig()
# - Set level from settings.log_level
# - Format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#
# LEARNING NOTE: Python logging levels:
# - DEBUG: Detailed information, for diagnosing problems
# - INFO: General information about what's happening
# - WARNING: Something unexpected happened
# - ERROR: A serious problem occurred
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Use category-aware logger for system logs
logger = get_logger(__name__, category="system")

# Category-specific loggers for different endpoints
chat_logger = get_logger(f"{__name__}.chat", category="chat")
audio_logger = get_logger(f"{__name__}.audio", category="audio")
video_logger = get_logger(f"{__name__}.video", category="video")
stream_event_logger = get_logger(f"{__name__}.eventsub", category="stream_event_sub")
stream_metadata_logger = get_logger(f"{__name__}.metadata", category="stream_metadata")

# Configure uvicorn access logger to filter chat polling endpoints
# This reduces log noise from frequent chat polling
access_logger = logging.getLogger("uvicorn.access")


def filter_access_log(record):
    """Filter out verbose chat polling logs."""
    message = record.getMessage()
    # Suppress all /chat/send endpoint logs (polled every 500ms)
    if message.find("/chat/send") != -1:
        return False
    # Suppress all /chat/message endpoint logs (chat messages are too verbose)
    if message.find("/chat/message") != -1:
        return False
    return True


# Add filter to access logger
access_logger.addFilter(filter_access_log)


def _safe_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    return sanitized or "unknown"


def _json_default(value):  # type: ignore[no-untyped-def]
    if isinstance(value, datetime):
        return value.isoformat() + "Z"
    if isinstance(value, timedelta):
        return value.total_seconds()
    return str(value)


def log_rag_ask_event(
    *,
    channel_id: str,
    question: str,
    result: dict,
    system_prompt: str,
    user_prompt: str,
    metadata: Optional[dict] = None,
) -> None:
    event_time = datetime.utcnow()
    timestamp_str = event_time.replace(microsecond=0).isoformat() + "Z"

    payload = {
        "timestamp": timestamp_str,
        "channel_id": channel_id,
        "question": question,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "answer": result.get("answer"),
        "citations": result.get("citations", []),
        "context": result.get("context", []),
        "chunks": result.get("chunks", []),
        "metadata": metadata or {},
    }

    safe_channel = _safe_path_component(str(channel_id))
    channel_dir = RAG_ASK_LOG_ROOT / safe_channel

    try:
        channel_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{event_time.strftime('%Y%m%dT%H%M%S%f')}Z_success.json"
        (channel_dir / filename).write_text(
            json.dumps(payload, indent=2, default=_json_default),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to write RAG ask log for channel %s: %s",
            channel_id,
            exc,
            exc_info=True,
        )


# Create FastAPI app
# TASK: Create FastAPI instance
# - title: "Percepta Python Service"
# - description: "AI-powered Twitch chat bot backend"
# - version: "0.1.0"
#
# LEARNING NOTE: This metadata appears in auto-generated docs
# Access docs at http://localhost:8000/docs (Swagger UI)

# app = FastAPI(...)
app = FastAPI(
    title="Percepta Python Service",
    description="AI-powered Twitch chat bot backend",
    version="0.1.0",
)

# Add CORS middleware
# TASK: Add CORS middleware to app
# - allow_origins: ["*"] (allow all origins for development)
# - allow_credentials: True
# - allow_methods: ["*"]
# - allow_headers: ["*"]
#
# LEARNING NOTE: What is CORS?
# - Cross-Origin Resource Sharing
# - Browser security feature
# - Node service runs on different port (needs CORS)
# - In production, restrict to specific origins
#
# app.add_middleware(
#     CORSMiddleware,
#     ...
# )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MESSAGE QUEUE
# ============================================================================

# In-memory message queue for bot responses
# Each item: {"channel": str, "message": str, "reply_to": Optional[str]}
message_queue = []

# Bot state management
bot_paused = False  # Global bot paused state

# Parse admin users from config
admin_users_list: List[str] = []
if settings.admin_users:
    admin_users_list = [
        user.strip().lower() for user in settings.admin_users.split(",") if user.strip()
    ]

# Initialize transcription service (singleton)
try:
    transcription_service: Optional[TranscriptionService] = (
        TranscriptionService.get_instance()
    )
    logger.info("Transcription service initialized")
except Exception as exc:
    logger.warning("Transcription service unavailable: %s", exc)
    transcription_service = None

# Initialize vector store for transcript storage
try:
    vector_store: Optional[VectorStore] = VectorStore()
    logger.info("Vector store initialized")
except Exception as exc:
    logger.warning("Vector store unavailable: %s", exc)
    vector_store = None

# Initialize video store for video frame storage
try:
    video_store: Optional[VideoStore] = VideoStore()
    logger.info("Video store initialized")
except Exception as exc:
    logger.warning("Video store unavailable: %s", exc)
    video_store = None

# Initialize chat store for chat message storage
try:
    chat_store: Optional[ChatStore] = ChatStore()
    logger.info("Chat store initialized")
except Exception as exc:
    logger.warning("Chat store unavailable: %s", exc)
    chat_store = None

# Initialize summarizer for memory-propagated summarization
try:
    from py.memory.summarizer import Summarizer

    summarizer: Optional[Summarizer] = None
    if vector_store and video_store and chat_store:
        summarizer = Summarizer(
            video_store=video_store,
            vector_store=vector_store,
            chat_store=chat_store,
        )
        logger.info("Summarizer initialized")
    else:
        logger.warning(
            "Summarizer not initialized: missing required stores (vector=%s, video=%s, chat=%s)",
            vector_store is not None,
            video_store is not None,
            chat_store is not None,
        )
except Exception as exc:
    logger.warning("Summarizer unavailable: %s", exc)
    summarizer = None

# Initialize RAG service (after stores are initialized)
try:
    rag_service: Optional[RAGService] = RAGService(
        vector_store=vector_store,
        video_store=video_store,
        chat_store=chat_store,
        summarizer=summarizer,
    )
except ValueError as exc:
    logger.warning("RAG service unavailable: %s", exc)
    rag_service = None

# Initialize Redis session manager
session_manager: Optional[RedisSessionManager] = None
try:
    session_manager = RedisSessionManager(
        redis_url=f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
        session_expiry_minutes=settings.session_expiry_minutes,
        max_history=settings.max_session_history,
        global_rate_limit_msgs=settings.global_rate_limit_msgs,
        global_rate_limit_window=settings.global_rate_limit_window,
        repeated_question_cooldown=settings.repeated_question_cooldown,
    )
    logger.info("Redis session manager initialized")
except Exception as exc:
    logger.warning("Session manager unavailable: %s", exc)
    session_manager = None

# Initialize EventSub WebSocket client
eventsub_client: Optional[EventSubWebSocketClient] = None
if settings.eventsub_enabled:
    try:
        eventsub_client = EventSubWebSocketClient(
            client_id=settings.twitch_client_id,
            access_token=settings.twitch_bot_token,
            target_channel=settings.target_channel,
        )
        logger.info("EventSub client initialized")
    except Exception as exc:
        logger.warning("EventSub client unavailable: %s", exc)
        eventsub_client = None

# Initialize channel metadata poller
metadata_poller: Optional[ChannelMetadataPoller] = None
if settings.metadata_poll_enabled:
    try:
        metadata_poller = ChannelMetadataPoller(
            client_id=settings.twitch_client_id,
            access_token=settings.twitch_bot_token,
            target_channel=settings.target_channel,
            vector_store=vector_store,
        )
        logger.info("Metadata poller initialized")
    except Exception as exc:
        logger.warning("Metadata poller unavailable: %s", exc)
        metadata_poller = None

# ============================================================================
# ENDPOINTS
# ============================================================================


def is_admin_user(username: str) -> bool:
    """Check if user is an admin."""
    return username.lower() in admin_users_list


async def handle_admin_command(message: ChatMessage) -> Optional[str]:
    """
    Handle admin commands (!pause, !resume, !status).

    Args:
        message: Chat message containing command

    Returns:
        Response message if command was processed, None otherwise
    """
    global bot_paused

    if not is_admin_user(message.username):
        return None

    msg_lower = message.message.lower().strip()

    if msg_lower == "!pause":
        bot_paused = True
        logger.info(f"Bot paused by admin {message.username}")
        return "Bot paused. Use !resume to resume."

    elif msg_lower == "!resume":
        bot_paused = False
        logger.info(f"Bot resumed by admin {message.username}")
        return "Bot resumed."

    elif msg_lower == "!status":
        status = "paused" if bot_paused else "active"
        return f"Bot status: {status}"

    return None


@app.post("/chat/message", response_model=MessageReceived)
async def receive_message(message: ChatMessage):
    """
    Receive a chat message from Node IRC service

    This is called when someone sends a message in Twitch chat.
    Node captures it and forwards it here.

    Args:
        message: ChatMessage model (automatically validated by FastAPI)

    Returns:
        MessageReceived: Acknowledgment that we got the message

    TASK:
    1. Log the received message with INFO level
       Format: f"Received message from {username} in #{channel}: {message}"
    2. Generate a unique message_id using uuid.uuid4()
    3. Return MessageReceived with:
       - received: True
       - message_id: generated ID as string
       - timestamp: datetime.now()

    LEARNING NOTE: response_model
    - Tells FastAPI what model to return
    - FastAPI validates the response matches the model
    - Automatically generates API documentation
    - Catches bugs (can't return wrong shape)
    """
    # Chat messages - reduce logging verbosity (chat is working)
    # Only log if it's an @mention or important message
    if message.message.strip().startswith("@"):
        chat_logger.debug(
            f"Received @mention from {message.username} in #{message.channel}: {message.message}"
        )

    # Store chat message with embedding (if chat_store is available)
    if chat_store is not None:
        try:
            # Convert channel name to broadcaster ID (same pattern as transcripts)
            channel_broadcaster_id = message.channel
            broadcaster_id = await get_broadcaster_id_from_channel_name(message.channel)
            if broadcaster_id:
                channel_broadcaster_id = broadcaster_id

            # Generate embedding for message
            embedding = await embed_text(message.message)

            # Ensure timestamp is timezone-aware (UTC)
            sent_at = message.timestamp
            if sent_at.tzinfo is None:
                from datetime import timezone

                sent_at = sent_at.replace(tzinfo=timezone.utc)

            # Store message in chat_store
            await chat_store.insert_message(
                channel_id=channel_broadcaster_id,
                username=message.username,
                message=message.message,
                sent_at=sent_at,
                embedding=embedding,
            )
        except Exception as store_exc:
            # Log error but don't fail the request (graceful degradation)
            chat_logger.warning(
                f"Failed to store chat message from {message.username} in {message.channel}: {store_exc}",
                exc_info=True,
            )

    # Check for admin commands first (!pause, !resume, !status)
    admin_response = await handle_admin_command(message)
    if admin_response:
        message_queue.append(
            {
                "channel": message.channel,
                "message": admin_response,
                "reply_to": message.username,
            }
        )
        return MessageReceived(
            received=True,
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )

    # Check if bot is paused
    global bot_paused
    if bot_paused:
        # Skip processing if bot is paused (except admin commands)
        return MessageReceived(
            received=True,
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )

    # Check for !more command
    msg_stripped = message.message.strip().lower()
    if msg_stripped.startswith("!more"):
        if session_manager:
            # Parse answer ID from command (!more <answer_id>)
            parts = msg_stripped.split()
            if len(parts) >= 2:
                answer_id_prefix = parts[1]
                # Get full answer ID from prefix mapping
                full_answer_id = await session_manager.get_answer_id_from_prefix(
                    message.channel, message.username, answer_id_prefix
                )

                if full_answer_id:
                    # Retrieve full answer using full UUID
                    full_answer = await session_manager.get_long_answer(full_answer_id)
                    if full_answer:
                        # Found the answer, send it
                        message_queue.append(
                            {
                                "channel": message.channel,
                                "message": full_answer,
                                "reply_to": message.username,
                            }
                        )
                        chat_logger.info(
                            f"Sent full answer via !more for {message.username}"
                        )
                        return MessageReceived(
                            received=True,
                            message_id=str(uuid.uuid4()),
                            timestamp=datetime.now(),
                        )
                    else:
                        # Answer not found or expired
                        chat_logger.debug(
                            f"!more answer expired for {message.username}"
                        )
                else:
                    # Prefix mapping not found
                    chat_logger.debug(
                        f"!more answer ID not found for {message.username}"
                    )
            else:
                # No answer ID provided
                chat_logger.debug(
                    f"!more command without answer ID from {message.username}"
                )
        return MessageReceived(
            received=True,
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )

    # Check for @mention at start of message
    bot_name = settings.twitch_bot_name or "percepta"  # Fallback if not set
    tokens = message.message.lstrip().split()
    first_word = tokens[0].lower() if tokens else ""

    # If bot is mentioned and message contains "ping"
    if first_word == f"@{bot_name.lower()}" and "ping" in message.message.lower():
        response_text = "pong"
        message_queue.append(
            {
                "channel": message.channel,
                "message": response_text,
                "reply_to": message.username,
            }
        )
        chat_logger.info(f"Queued response: {response_text}")
        return MessageReceived(
            received=True,
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )

    # Check if bot is mentioned with a question (not just ping)
    elif first_word == f"@{bot_name.lower()}" and rag_service:
        # Extract question (remove @botname and any leading whitespace)
        question_tokens = tokens[1:] if len(tokens) > 1 else []
        question = " ".join(question_tokens).strip()

        # Skip if it was just "ping" (already handled above)
        if question and question.lower() != "ping":
            # Safety checks

            # 1. Check content filtering (toxic/PII)
            if not is_safe_for_chat(question):
                chat_logger.info(
                    f"Unsafe content detected from {message.username} in {message.channel}"
                )
                return MessageReceived(
                    received=True,
                    message_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                )

            # 2. Check global rate limit
            if session_manager:
                can_send_global = await session_manager.check_global_rate_limit(
                    message.channel
                )
                if not can_send_global:
                    chat_logger.info(
                        f"Global rate limit exceeded for {message.channel}"
                    )
                    return MessageReceived(
                        received=True,
                        message_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                    )

            # 3. Check per-user rate limit
            if session_manager:
                can_send = await session_manager.check_rate_limit(
                    message.username,
                    message.channel,
                    settings.rate_limit_seconds,
                )
                if not can_send:
                    chat_logger.info(
                        f"Rate limit hit for {message.username} in {message.channel}"
                    )
                    return MessageReceived(
                        received=True,
                        message_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                    )
                # Update last_message_time immediately to prevent race conditions
                # This ensures subsequent requests see the updated timestamp
                try:
                    session = await session_manager.get_session(
                        message.username, message.channel
                    )
                    session["last_message_time"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    key = session_manager._get_session_key(
                        message.username, message.channel
                    )
                    await session_manager._save_session(key, session)
                except Exception:
                    # Don't fail if session update fails
                    pass

            # 4. Check repeated question cooldown
            if session_manager:
                can_ask, cooldown_key = (
                    await session_manager.check_repeated_question_cooldown(
                        message.username,
                        message.channel,
                        question,
                    )
                )
                if not can_ask:
                    chat_logger.info(
                        f"Repeated question cooldown active for {message.username} in {message.channel}"
                    )
                    return MessageReceived(
                        received=True,
                        message_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                    )

            # Validate question has reasonable content
            if len(question) >= 3:  # Minimum meaningful question
                try:
                    # Convert channel name to broadcaster ID for RAG queries
                    channel_broadcaster_id = message.channel
                    # Try to get broadcaster ID using shared utility function
                    broadcaster_id = await get_broadcaster_id_from_channel_name(
                        message.channel
                    )
                    if broadcaster_id:
                        channel_broadcaster_id = broadcaster_id

                    # Trigger RAG query asynchronously with user session history
                    result = await rag_service.answer(
                        channel_id=channel_broadcaster_id,
                        question=question,
                        top_k=None,  # Use default from settings
                        half_life_minutes=None,  # Use default from settings
                        prefilter_limit=None,  # Use default from settings
                        user_id=message.username,
                        session_manager=session_manager,
                    )

                    # Extract answer text
                    answer_text = result.get(
                        "answer",
                        "I don't have enough context to answer that right now.",
                    )

                    # Filter response content for safety
                    answer_text = filter_response_content(answer_text)

                    # Handle response length limits
                    if len(answer_text) > settings.max_response_length:
                        # Truncate and store full answer
                        truncated_answer = answer_text[: settings.max_response_length]
                        answer_id = str(uuid.uuid4())
                        answer_id_prefix = answer_id[:8]

                        # Store full answer in Redis
                        if session_manager:
                            await session_manager.store_long_answer(
                                answer_id,
                                answer_text,
                                settings.long_answer_storage_hours,
                            )
                            # Store prefix mapping for !more command
                            await session_manager.store_answer_prefix_mapping(
                                message.channel,
                                message.username,
                                answer_id_prefix,
                                answer_id,
                                settings.long_answer_storage_hours,
                            )

                        # Append instruction for !more
                        truncated_answer += f" (Answer truncated. Use !more {answer_id_prefix} for full answer)"
                        answer_text = truncated_answer

                    # Queue response
                    message_queue.append(
                        {
                            "channel": message.channel,
                            "message": answer_text,
                            "reply_to": message.username,
                        }
                    )

                    # Note: Global rate limit counter is updated when messages are actually sent,
                    # not when queued, to accurately track messages sent to Twitch

                    chat_logger.info(
                        f"Queued RAG response for @{message.username} in #{message.channel}: "
                        f"{answer_text[:100]}{'...' if len(answer_text) > 100 else ''}"
                    )

                    # Update session with Q&A pair
                    if session_manager:
                        try:
                            await session_manager.update_session(
                                user_id=message.username,
                                channel=message.channel,
                                question=question,
                                answer=answer_text,
                            )
                        except Exception as session_exc:
                            # Log but don't fail if session update fails
                            chat_logger.warning(
                                f"Failed to update session for {message.username}: {session_exc}"
                            )
                except Exception as e:
                    # Log error but don't crash - gracefully handle RAG failures
                    chat_logger.error(
                        f"RAG query failed for @{message.username} in #{message.channel}: {e}",
                        exc_info=True,
                    )
            else:
                # Question too short/empty
                chat_logger.debug(
                    f"Received @mention with empty/short question from {message.username}: '{question}'"
                )

    return MessageReceived(
        received=True,
        message_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
    )


@app.post("/chat/send", response_model=SendResponse)
async def send_messages(request: SendRequest):
    """
    Get queued messages to send to Twitch

    Node polls this endpoint to check if there are messages to send.
    Respects global rate limits to prevent spam.

    Args:
        request: SendRequest with channel name

    Returns:
        SendResponse: List of messages to send (respecting rate limits)

    LEARNING NOTE: Why polling instead of push?
    - Simpler architecture for MVP
    - Node is in control of when it sends
    - Avoids need for webhooks or WebSockets
    - Good enough for low-traffic bot

    Future: Could use WebSockets for real-time push
    """
    # Chat polling endpoint - keep logs minimal since it's called frequently
    # logger.debug(f"Send request for channel: {request.channel}")  # Disabled: too verbose

    # Filter messages for this channel
    channel_messages = [
        msg for msg in message_queue if msg["channel"] == request.channel
    ]

    # Respect global rate limit - only send up to limit
    responses = []
    if session_manager:
        # Check how many messages we can send
        can_send_global = await session_manager.check_global_rate_limit(request.channel)
        if not can_send_global:
            # Rate limit exceeded, don't send any messages
            chat_logger.debug(
                f"Global rate limit exceeded for {request.channel}, skipping message send"
            )
            return SendResponse(messages=[])

        # Calculate how many messages we can send in this batch
        # Get current count
        key = session_manager._get_global_rate_limit_key(request.channel)
        if session_manager.redis_client and session_manager._connected:
            try:
                current_count_str = await session_manager.redis_client.get(key)
                current_count = int(current_count_str) if current_count_str else 0
                remaining = settings.global_rate_limit_msgs - current_count

                # Only send up to remaining limit
                if remaining > 0:
                    channel_messages = channel_messages[:remaining]
                else:
                    channel_messages = []
            except Exception:
                # On error, allow sending (graceful degradation)
                pass

    # Convert to ChatResponse objects
    responses = [
        ChatResponse(
            channel=msg["channel"], message=msg["message"], reply_to=msg.get("reply_to")
        )
        for msg in channel_messages
    ]

    # Remove sent messages from queue
    for msg in channel_messages:
        message_queue.remove(msg)

    # Update global rate limit counter for each message sent
    if session_manager and responses:
        for _ in responses:
            await session_manager.update_global_rate_limit(request.channel)

    if responses:
        # Only log when actually sending messages (not empty polls)
        chat_logger.debug(
            f"Returning {len(responses)} message(s) for channel {request.channel}"
        )

    return SendResponse(
        messages=responses,
    )


@app.post("/rag/answer", response_model=RAGAnswerResponse)
async def rag_answer(request: RAGQueryRequest):
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service unavailable")

    # Convert channel name to broadcaster ID if needed
    # If it's numeric, assume it's already a broadcaster ID
    # Otherwise, try to convert it using shared utility function
    channel_id = request.channel
    if not request.channel.isdigit():
        broadcaster_id = await get_broadcaster_id_from_channel_name(request.channel)
        if broadcaster_id:
            channel_id = broadcaster_id

    try:
        result = await rag_service.answer(
            channel_id=channel_id,
            question=request.question,
            top_k=request.top_k,
            half_life_minutes=request.half_life_minutes,
            prefilter_limit=request.prefilter_limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Failed to generate RAG answer")
        raise HTTPException(status_code=500, detail="Failed to generate answer")

    request_metadata = {
        "source": "api",
        "request_channel": request.channel,
        "top_k": request.top_k,
        "half_life_minutes": request.half_life_minutes,
        "prefilter_limit": request.prefilter_limit,
    }
    # Remove None values to keep metadata clean
    request_metadata = {
        key: value for key, value in request_metadata.items() if value is not None
    }
    prompts = result.get("prompts", {})
    log_rag_ask_event(
        channel_id=str(channel_id),
        question=request.question,
        result=result,
        system_prompt=prompts.get("system", ""),
        user_prompt=prompts.get("user", ""),
        metadata=request_metadata,
    )

    return RAGAnswerResponse(**result)


@app.get("/api/debug-credentials")
async def debug_credentials():
    """Debug endpoint to check what credentials are loaded"""
    client_id = settings.twitch_client_id or ""
    bot_token = settings.twitch_bot_token or ""

    return {
        "client_id_present": bool(client_id),
        "client_id_length": len(client_id),
        "client_id_prefix": client_id[:10] if client_id else None,
        "bot_token_present": bool(bot_token),
        "bot_token_length": len(bot_token),
        "bot_token_prefix": bot_token[:10] if bot_token else None,
        "bot_token_has_oauth_prefix": (
            bot_token.startswith("oauth:") if bot_token else False
        ),
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service is working correctly.

    Returns service status. Used by:
    - Docker health checks
    - Load balancers
    - Monitoring systems

    Tests broadcaster ID lookup to ensure the service can communicate with Twitch API.
    Uses TARGET_CHANNEL from .env for validation.
    """
    global eventsub_client
    eventsub_status = "disconnected"
    if eventsub_client:
        if eventsub_client.is_connected:
            eventsub_status = "connected"
        else:
            eventsub_status = "disconnected"

    # Test broadcaster ID lookup using configured target channel
    broadcaster_id_lookup_status = "skipped"
    broadcaster_id_lookup_error = None
    test_channel = None
    test_broadcaster_id = None

    if settings.target_channel:
        try:
            test_id = await get_broadcaster_id_from_channel_name(
                settings.target_channel
            )
            if test_id:
                broadcaster_id_lookup_status = "working"
                test_channel = settings.target_channel
                test_broadcaster_id = test_id
            else:
                broadcaster_id_lookup_status = "failed"
                broadcaster_id_lookup_error = "Broadcaster ID lookup returned None"
                test_channel = settings.target_channel
        except Exception as e:
            broadcaster_id_lookup_status = "error"
            broadcaster_id_lookup_error = str(e)
            test_channel = settings.target_channel
            logger.error(
                f"Health check broadcaster ID lookup failed: {e}", exc_info=True
            )

    return {
        "status": "healthy",
        "service": "percepta-python",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eventsub": {
            "enabled": settings.eventsub_enabled,
            "status": eventsub_status,
        },
        "broadcaster_id_lookup": {
            "status": broadcaster_id_lookup_status,
            "test_channel": test_channel,
            "test_broadcaster_id": test_broadcaster_id,
            "error": broadcaster_id_lookup_error,
        },
    }


@app.get("/api/get-broadcaster-id")
async def get_broadcaster_id(channel_name: str):
    """
    Get broadcaster user ID from channel name.

    This endpoint helps standardize channel identifiers across the system.
    All tables use broadcaster ID instead of channel names for consistency.

    Args:
        channel_name: Twitch channel name (e.g., "jynxzi")

    Returns:
        JSON with broadcaster_id
    """
    try:
        # Use shared utility function that matches metadata poller pattern
        broadcaster_id = await get_broadcaster_id_from_channel_name(channel_name)

        if broadcaster_id:
            return {"broadcaster_id": broadcaster_id, "channel_name": channel_name}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Channel not found: {channel_name}. Failed to get broadcaster ID from Twitch API.",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error in get_broadcaster_id endpoint: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get broadcaster ID: {str(e)}"
        )


@app.get("/api/get-audio-stream-url", response_model=AudioStreamUrlResponse)
async def get_audio_stream_url(channel_id: str):
    """
    Get authenticated Twitch audio-only stream URL using Streamlink.

    This endpoint uses Streamlink's Python API to obtain authenticated HLS URLs
    for Twitch audio-only streams. Streamlink handles all authentication internally,
    including OAuth tokens and client-integrity tokens.

    Returns an HLS URL that can be used directly with FFmpeg for audio capture.
    """
    try:
        import streamlink

        # Create Streamlink session
        session = streamlink.Streamlink()

        # Get streams for the Twitch channel
        twitch_url = f"https://www.twitch.tv/{channel_id}"
        streams = session.streams(twitch_url)

        # Get the audio_only stream
        audio_stream = streams.get("audio_only")

        if audio_stream is None:
            # Stream is not available (offline or no audio stream)
            audio_logger.warning(
                f"Audio-only stream not available for channel {channel_id}. "
                "Channel may be offline or stream not started."
            )
            return AudioStreamUrlResponse(
                channel_id=channel_id,
                stream_url=None,
                quality="audio_only",
                available=False,
            )

        # Extract the authenticated HLS URL
        stream_url = audio_stream.url

        audio_logger.info(
            f"Retrieved audio-only stream URL for channel {channel_id}: "
            f"{stream_url[:100]}..."  # Log first 100 chars for security
        )

        return AudioStreamUrlResponse(
            channel_id=channel_id,
            stream_url=stream_url,
            quality="audio_only",
            available=True,
        )

    except ImportError:
        logger.error(
            "Streamlink not installed. Please install with: pip install streamlink"
        )
        raise HTTPException(
            status_code=503,
            detail="Streamlink library not available. Please install streamlink.",
        )
    except streamlink.exceptions.NoPluginError:
        logger.error(f"No Streamlink plugin found for Twitch URL: {twitch_url}")
        raise HTTPException(
            status_code=500, detail="Streamlink Twitch plugin not available"
        )
    except streamlink.exceptions.PluginError as exc:
        logger.error(f"Streamlink plugin error for channel {channel_id}: {str(exc)}")
        raise HTTPException(status_code=502, detail=f"Failed to get stream: {str(exc)}")
    except Exception as exc:
        logger.exception(
            f"Unexpected error getting stream URL for channel {channel_id}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get stream URL: {str(exc)}"
        ) from exc


@app.get("/api/get-video-stream-url", response_model=AudioStreamUrlResponse)
async def get_video_stream_url(channel_id: str):
    """
    Get authenticated Twitch video stream URL using Streamlink.

    This endpoint uses Streamlink's Python API to obtain authenticated HLS URLs
    for Twitch video streams. Streamlink handles all authentication internally,
    including OAuth tokens and client-integrity tokens.

    Returns an HLS URL that can be used directly with FFmpeg for video capture.
    """
    try:
        import streamlink

        # Create Streamlink session
        session = streamlink.Streamlink()

        # Get streams for the Twitch channel
        twitch_url = f"https://www.twitch.tv/{channel_id}"
        streams = session.streams(twitch_url)

        # Get the best quality video stream (or worst for lower bandwidth)
        video_stream = streams.get("best")

        if video_stream is None:
            # Stream is not available (offline or restricted)
            return AudioStreamUrlResponse(
                channel_id=channel_id, available=False, stream_url=None
            )

        # Get the HLS URL
        stream_url = video_stream.url

        return AudioStreamUrlResponse(
            channel_id=channel_id, available=True, stream_url=stream_url
        )

    except ImportError:
        logger.error(
            "Streamlink not installed. Please install with: pip install streamlink"
        )
        raise HTTPException(
            status_code=503,
            detail="Streamlink library not available. Please install streamlink.",
        )
    except streamlink.exceptions.NoPluginError:
        logger.error(f"No Streamlink plugin found for Twitch URL: {twitch_url}")
        raise HTTPException(
            status_code=500, detail="Streamlink Twitch plugin not available"
        )
    except streamlink.exceptions.PluginError as exc:
        logger.error(f"Streamlink plugin error for channel {channel_id}: {str(exc)}")
        raise HTTPException(status_code=502, detail=f"Failed to get stream: {str(exc)}")
    except Exception as exc:
        logger.exception(
            f"Unexpected error getting video stream URL for channel {channel_id}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get stream URL: {str(exc)}"
        ) from exc


@app.post("/api/video-frame")
async def receive_video_frame(
    image_file: UploadFile = File(..., description="Video frame image (JPEG/PNG)"),
    channel_id: str = Form(..., description="Broadcaster channel ID"),
    captured_at: str = Form(..., description="ISO timestamp when frame was captured"),
    interesting_hint: Optional[str] = Form(
        None, description="Optional hint that frame may be interesting"
    ),
):
    """
    Receive video frame screenshot from Node service and store in vector DB.

    Complete pipeline: image → CLIP embedding → vector DB storage.
    Receives screenshot images from Twitch streams, generates CLIP embeddings,
    and stores frames in pgvector for RAG queries.
    """
    if video_store is None:
        raise HTTPException(status_code=503, detail="Video store unavailable")

    try:
        # Read image file content
        image_content = await image_file.read()
        file_size = len(image_content)

        video_logger.info(
            f"Received video frame: channel={channel_id}, "
            f"size={file_size} bytes, captured_at={captured_at}"
        )

        # Validate image file
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Image file is empty")

        # Save uploaded file temporarily
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(image_content)
            tmp_path = tmp_file.name

        try:
            # Parse timestamp
            captured_at_str = (
                captured_at.replace("Z", "+00:00")
                if captured_at.endswith("Z")
                else captured_at
            )
            captured_at_obj = datetime.fromisoformat(captured_at_str)

            hint_bool: Optional[bool] = None
            if interesting_hint is not None:
                lower_hint = interesting_hint.lower()
                if lower_hint in {"true", "1", "yes", "on"}:
                    hint_bool = True
                elif lower_hint in {"false", "0", "no", "off"}:
                    hint_bool = False

            # Store frame in database (VideoStore will generate embedding and move file)
            insert_result = await video_store.insert_frame(
                channel_id=channel_id,
                image_path=tmp_path,
                captured_at=captured_at_obj,
                is_interesting=hint_bool,
            )
            frame_id = insert_result["frame_id"]

            if chat_store is not None and vector_store is not None:
                capture_decision = await determine_capture_interval(
                    channel_id=channel_id,
                    captured_at=captured_at_obj,
                    chat_store=chat_store,
                    vector_store=vector_store,
                    recent_chat_count=insert_result.get("chat_count"),
                    transcript_text=insert_result.get("transcript_text"),
                    was_interesting=insert_result.get("was_interesting", False),
                )
            else:
                capture_decision = {
                    "next_interval_seconds": 10,
                    "recent_chat_count": insert_result.get("chat_count", 0),
                    "keyword_trigger": False,
                }

            video_logger.info(
                f"Stored video frame: frame_id={frame_id}, channel={channel_id}"
            )

            return {
                "frame_id": frame_id,
                "status": "success",
                "description_source": insert_result.get("description_source"),
                "reused_description": insert_result.get("reused_description", False),
                "interesting_frame": insert_result.get("was_interesting", False),
                "next_interval_seconds": capture_decision["next_interval_seconds"],
                "activity": {
                    "recent_chat_count": capture_decision["recent_chat_count"],
                    "keyword_trigger": capture_decision["keyword_trigger"],
                },
            }

        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    except ValueError as e:
        # Invalid timestamp format
        video_logger.error(f"Invalid timestamp format: {captured_at}")
        raise HTTPException(
            status_code=400, detail=f"Invalid timestamp format: {str(e)}"
        )
    except Exception as exc:
        video_logger.exception(f"Failed to process video frame: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process video frame: {str(exc)}"
        ) from exc


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="Audio file (WAV format)"),
    channel_id: str = Form(..., description="Twitch channel identifier"),
    started_at: str = Form(..., description="ISO timestamp when chunk started"),
    ended_at: str = Form(..., description="ISO timestamp when chunk ended"),
):
    """
    Transcribe audio chunk from Node service and store in vector DB.

    Complete pipeline: audio → transcription → embedding → vector DB storage.
    Receives audio chunks from Twitch streams, transcribes them with faster-whisper,
    generates embeddings, and stores transcripts in pgvector for RAG queries.
    """
    if transcription_service is None:
        raise HTTPException(status_code=503, detail="Transcription service unavailable")

    try:
        # Read audio file content
        audio_content = await audio_file.read()
        file_size = len(audio_content)

        audio_logger.info(
            f"Received audio chunk: channel={channel_id}, "
            f"size={file_size} bytes, "
            f"started_at={started_at}, ended_at={ended_at}"
        )

        # Validate audio file
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")

        # Transcribe audio using faster-whisper
        result = await transcription_service.transcribe_async(audio_content)

        # Convert segments to response format using Pydantic models
        from schemas.messages import TranscriptionSegment, TranscriptionWord

        segments = [
            TranscriptionSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=[
                    TranscriptionWord(
                        word=word["word"],
                        start=word["start"],
                        end=word["end"],
                    )
                    for word in seg.words
                ],
            )
            for seg in result.segments
        ]

        # Initialize response fields for storage
        stored_in_db = False
        transcript_id = None
        embedding_latency_ms = None
        db_insert_latency_ms = None

        # Pipeline: Embedding + Vector DB Storage
        if vector_store is not None and result.transcript.strip():
            try:
                # Parse timestamps as UTC (database stores timezone-aware timestamps in UTC)
                # PostgreSQL will display them in session timezone (configured in connection.py)
                start_time_str = (
                    started_at.replace("Z", "+00:00")
                    if started_at.endswith("Z")
                    else started_at
                )
                end_time_str = (
                    ended_at.replace("Z", "+00:00")
                    if ended_at.endswith("Z")
                    else ended_at
                )
                start_time_obj = datetime.fromisoformat(start_time_str)
                end_time_obj = datetime.fromisoformat(end_time_str)

                # Generate embedding for transcript text
                embedding_start = time.perf_counter()
                try:
                    embedding = await embed_text(result.transcript)
                    embedding_elapsed = time.perf_counter() - embedding_start
                    embedding_latency_ms = int(embedding_elapsed * 1000)
                    audio_logger.info(
                        f"Embedding generated: channel={channel_id}, "
                        f"latency={embedding_latency_ms}ms"
                    )
                except Exception as embed_exc:
                    audio_logger.error(
                        f"Failed to generate embedding: channel={channel_id}, "
                        f"error={str(embed_exc)}"
                    )
                    raise

                # Insert transcript into vector store
                db_start = time.perf_counter()
                try:
                    inserted_id = await vector_store.insert_transcript(
                        channel_id=channel_id,
                        text_value=result.transcript,
                        start_time=start_time_obj,
                        end_time=end_time_obj,
                        embedding=embedding,
                    )
                    db_elapsed = time.perf_counter() - db_start
                    db_insert_latency_ms = int(db_elapsed * 1000)
                    stored_in_db = True
                    transcript_id = inserted_id
                    audio_logger.info(
                        f"Transcript stored in DB: channel={channel_id}, "
                        f"id={transcript_id}, latency={db_insert_latency_ms}ms"
                    )
                except Exception as db_exc:
                    audio_logger.error(
                        f"Failed to insert transcript into DB: channel={channel_id}, "
                        f"error={str(db_exc)}"
                    )
                    raise

            except Exception as pipeline_exc:
                # Log error but don't fail the entire request
                # Transcription was successful, so return partial success
                logger.warning(
                    f"Pipeline error (embedding/DB): channel={channel_id}, "
                    f"error={str(pipeline_exc)}. Returning transcription only."
                )

        # Clear audio content from memory after processing
        del audio_content

        response = TranscriptionResponse(
            transcript=result.transcript,
            segments=segments,
            language=result.language,
            duration=result.duration,
            model=result.model,
            processing_time_ms=result.processing_time_ms,
            channel_id=channel_id,
            started_at=started_at,
            ended_at=ended_at,
            stored_in_db=stored_in_db,
            transcript_id=transcript_id,
            embedding_latency_ms=embedding_latency_ms,
            db_insert_latency_ms=db_insert_latency_ms,
        )

        total_latency = result.processing_time_ms
        if embedding_latency_ms:
            total_latency += embedding_latency_ms
        if db_insert_latency_ms:
            total_latency += db_insert_latency_ms

        audio_logger.info(
            f"Pipeline complete: channel={channel_id}, "
            f"duration={result.duration:.2f}s, "
            f"transcription={result.processing_time_ms}ms, "
            f"embedding={embedding_latency_ms}ms, "
            f"db_insert={db_insert_latency_ms}ms, "
            f"total={total_latency}ms, "
            f"stored={stored_in_db}, "
            f"language={result.language}"
        )

        return response

    except HTTPException:
        raise
    except Exception as exc:
        audio_logger.exception("Failed to transcribe audio chunk")
        raise HTTPException(
            status_code=500, detail=f"Failed to transcribe audio: {str(exc)}"
        ) from exc


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """
    Called when FastAPI starts

    Use for:
    - Database connection pooling
    - Loading ML models
    - Initializing caches
    - Starting EventSub WebSocket connection
    """
    logger.info(f"Percepta Python service starting on {settings.host}:{settings.port}")

    # Connect Redis session manager
    global session_manager
    if session_manager:
        try:
            await session_manager.connect()
            logger.info("Redis session manager connected")
        except Exception as exc:
            logger.warning(f"Failed to connect session manager: {exc}")
            session_manager = None

    # Preload Whisper model to avoid first-request delay
    if transcription_service is not None:
        try:
            logger.info("Preloading Whisper model...")
            transcription_service.load_model()
            logger.info("Whisper model loaded and ready")
        except Exception as exc:
            logger.warning(f"Failed to preload Whisper model: {exc}")

    # Start EventSub WebSocket connection
    global eventsub_client
    if eventsub_client:
        try:
            await eventsub_client.connect()
            stream_event_logger.info("EventSub WebSocket connection started")
        except Exception as exc:
            stream_event_logger.error(f"Failed to start EventSub connection: {exc}")

    # Start metadata polling
    global metadata_poller
    if metadata_poller:
        try:
            await metadata_poller.start()
            stream_metadata_logger.info("Metadata polling started")
        except Exception as exc:
            stream_metadata_logger.error(f"Failed to start metadata polling: {exc}")

    # Validate broadcaster ID lookup on startup
    if settings.target_channel:
        try:
            test_id = await get_broadcaster_id_from_channel_name(
                settings.target_channel
            )
            if test_id:
                logger.info(
                    f"Broadcaster ID lookup validated on startup for channel: {settings.target_channel} (ID: {test_id})"
                )
            else:
                logger.warning(
                    f"Broadcaster ID lookup returned None for channel: {settings.target_channel}. "
                    "Service may have issues."
                )
        except Exception as exc:
            logger.error(
                f"CRITICAL: Broadcaster ID lookup validation failed on startup for channel {settings.target_channel}: {exc}. "
                "Service may not function correctly."
            )
    else:
        logger.warning(
            "TARGET_CHANNEL not set in .env, skipping broadcaster ID lookup validation"
        )

    # Start background summarization job
    if summarizer:
        try:
            summarizer.start_background_job()
            logger.info("Background summarization job started")
        except Exception as exc:
            logger.warning("Failed to start background summarization job: %s", exc)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Called when FastAPI shuts down

    Use for:
    - Closing database connections
    - Saving state
    - Cleanup
    - Closing EventSub WebSocket connection
    """
    logger.info("Percepta Python service shutting down")

    # Close Redis session manager
    global session_manager
    if session_manager:
        try:
            await session_manager.close()
            logger.info("Redis session manager closed")
        except Exception as exc:
            logger.error(f"Error closing session manager: {exc}")

    # Stop metadata polling
    global metadata_poller
    if metadata_poller:
        try:
            await metadata_poller.stop()
            stream_metadata_logger.info("Metadata polling stopped")
        except Exception as exc:
            stream_metadata_logger.error(f"Error stopping metadata polling: {exc}")

    # Close EventSub WebSocket connection
    global eventsub_client
    if eventsub_client:
        try:
            await eventsub_client.disconnect()
            stream_event_logger.info("EventSub WebSocket connection closed")
        except Exception as exc:
            stream_event_logger.error(f"Error closing EventSub connection: {exc}")


"""
LEARNING NOTES:

1. FastAPI Request Flow:
  Client → FastAPI → Pydantic validates → Your function → Pydantic validates response → Client
  
2. Type Hints = Validation:
  def func(message: ChatMessage) ← FastAPI validates this
  → return MessageReceived        ← FastAPI validates this
  
3. Automatic Documentation:
  - Go to http://localhost:8000/docs
  - See all endpoints, models, try them out
  - Generated from your code automatically!
  
4. Async vs Sync:
  - async def: Can use await, non-blocking
  - def: Regular Python function, blocking
  - FastAPI handles both, but async is more scalable
  
5. Error Handling:
  - Pydantic validation errors → 422 Unprocessable Entity
  - Raise HTTPException for custom errors
  - FastAPI handles it gracefully

6. Compared to Node/Express:
  Node: app.post('/route', (req, res) => { res.json({}) })
  FastAPI: @app.post('/route') def func(data: Model) -> Model: return Model()
  
  FastAPI: Type hints do the validation!
  Node: You manually validate request body

RUNNING THE SERVER:
    uvicorn py.main:app --reload --port 8000
    
    - py.main: Python module path (py/main.py)
    - app: The FastAPI instance in that file
    - --reload: Auto-restart on code changes (development)
    - --port 8000: Which port to listen on
"""
