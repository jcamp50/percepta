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

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from typing import Optional
import logging
import time
import uuid

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
from py.memory.vector_store import VectorStore
from py.utils.embeddings import embed_text

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

logger = logging.getLogger(__name__)

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

try:
    rag_service: Optional[RAGService] = RAGService()
except ValueError as exc:
    logger.warning("RAG service unavailable: %s", exc)
    rag_service = None

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


@app.get("/health")
async def health_check():
    """
    Health check endpoint

    Returns service status. Used by:
    - Docker health checks
    - Load balancers
    - Monitoring systems
    """
    global eventsub_client
    eventsub_status = "disconnected"
    if eventsub_client:
        if eventsub_client.is_connected:
            eventsub_status = "connected"
        else:
            eventsub_status = "disconnected"

    return {
        "status": "healthy",
        "service": "percepta-python",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eventsub": {
            "enabled": settings.eventsub_enabled,
            "status": eventsub_status,
        },
    }


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
        logger.debug(
            f"Received @mention from {message.username} in #{message.channel}: {message.message}"
        )
    # Otherwise, silently process (chat I/O is working, focus on audio transcription)

    # Check for @mention at start of message
    bot_name = settings.twitch_bot_name or "percepta"  # Fallback if not set
    tokens = message.message.lstrip().split()
    first_word = tokens[0].lower() if tokens else ""

    # If bot is mentioned and message contains "ping"
    if first_word == f"@{bot_name.lower()}" and "ping" in message.message.lower():
        response_text = f"pong"
        message_queue.append(
            {
                "channel": message.channel,
                "message": response_text,
                "reply_to": message.username,
            }
        )
        logger.info(f"Queued response: {response_text}")

    # Check if bot is mentioned with a question (not just ping)
    elif first_word == f"@{bot_name.lower()}" and rag_service:
        # Extract question (remove @botname and any leading whitespace)
        question_tokens = tokens[1:] if len(tokens) > 1 else []
        question = " ".join(question_tokens).strip()

        # Skip if it was just "ping" (already handled above)
        if question and question.lower() != "ping":
            # Validate question has reasonable content
            if len(question) >= 3:  # Minimum meaningful question
                try:
                    # Convert channel name to broadcaster ID for RAG queries
                    channel_broadcaster_id = message.channel
                    if eventsub_client and eventsub_client.http_client:
                        try:
                            broadcaster_id = await eventsub_client._get_broadcaster_id(
                                message.channel
                            )
                            if broadcaster_id:
                                channel_broadcaster_id = broadcaster_id
                        except Exception:
                            # Fallback to channel name if conversion fails
                            pass

                    # Trigger RAG query asynchronously
                    result = await rag_service.answer(
                        channel_id=channel_broadcaster_id,
                        question=question,
                        top_k=None,  # Use default from settings
                        half_life_minutes=None,  # Use default from settings
                        prefilter_limit=None,  # Use default from settings
                    )

                    # Extract answer text
                    answer_text = result.get(
                        "answer",
                        "I don't have enough context to answer that right now.",
                    )

                    # Queue response
                    message_queue.append(
                        {
                            "channel": message.channel,
                            "message": answer_text,
                            "reply_to": message.username,
                        }
                    )

                    logger.info(
                        f"Queued RAG response for @{message.username} in #{message.channel}: "
                        f"{answer_text[:100]}{'...' if len(answer_text) > 100 else ''}"
                    )
                except Exception as e:
                    # Log error but don't crash - gracefully handle RAG failures
                    logger.error(
                        f"RAG query failed for @{message.username} in #{message.channel}: {e}",
                        exc_info=True,
                    )
            else:
                # Question too short/empty
                logger.debug(
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

    Args:
        request: SendRequest with channel name

    Returns:
        SendResponse: List of messages to send (empty for now)

    TASK:
    1. Log the request with INFO level
       Format: f"Send request for channel: {channel}"
    2. Return SendResponse with empty messages list
       (Later phases will actually queue and return messages)

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

    if responses:
        # Only log when actually sending messages (not empty polls)
        logger.debug(
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
    # Otherwise, try to convert it
    channel_id = request.channel
    if (
        not request.channel.isdigit()
        and eventsub_client
        and eventsub_client.http_client
    ):
        try:
            broadcaster_id = await eventsub_client._get_broadcaster_id(request.channel)
            if broadcaster_id:
                channel_id = broadcaster_id
        except Exception:
            # Fallback to original value if conversion fails
            pass

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

    return RAGAnswerResponse(**result)


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
    if not eventsub_client or not eventsub_client.http_client:
        raise HTTPException(status_code=503, detail="EventSub client not available")

    try:
        broadcaster_id = await eventsub_client._get_broadcaster_id(channel_name)
        if not broadcaster_id:
            raise HTTPException(
                status_code=404, detail=f"Channel not found: {channel_name}"
            )
        return {"broadcaster_id": broadcaster_id, "channel_name": channel_name}
    except Exception as exc:
        logger.error(f"Failed to get broadcaster ID: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get broadcaster ID: {str(exc)}"
        ) from exc


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
            logger.warning(
                f"Audio-only stream not available for channel {channel_id}. "
                "Channel may be offline or stream not started."
            )
            return AudioStreamUrlResponse(
                channel_id=channel_id,
                stream_url="",
                quality="audio_only",
                available=False,
            )

        # Extract the authenticated HLS URL
        stream_url = audio_stream.url

        logger.info(
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

        logger.info(
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
                    logger.info(
                        f"Embedding generated: channel={channel_id}, "
                        f"latency={embedding_latency_ms}ms"
                    )
                except Exception as embed_exc:
                    logger.error(
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
                    logger.info(
                        f"Transcript stored in DB: channel={channel_id}, "
                        f"id={transcript_id}, latency={db_insert_latency_ms}ms"
                    )
                except Exception as db_exc:
                    logger.error(
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

        logger.info(
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
        logger.exception("Failed to transcribe audio chunk")
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
            logger.info("EventSub WebSocket connection started")
        except Exception as exc:
            logger.error(f"Failed to start EventSub connection: {exc}")

    # Start metadata polling
    global metadata_poller
    if metadata_poller:
        try:
            await metadata_poller.start()
            logger.info("Metadata polling started")
        except Exception as exc:
            logger.error(f"Failed to start metadata polling: {exc}")


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

    # Stop metadata polling
    global metadata_poller
    if metadata_poller:
        try:
            await metadata_poller.stop()
            logger.info("Metadata polling stopped")
        except Exception as exc:
            logger.error(f"Error stopping metadata polling: {exc}")

    # Close EventSub WebSocket connection
    global eventsub_client
    if eventsub_client:
        try:
            await eventsub_client.disconnect()
            logger.info("EventSub WebSocket connection closed")
        except Exception as exc:
            logger.error(f"Error closing EventSub connection: {exc}")


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
