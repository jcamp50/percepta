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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional
import logging
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
)
from py.reason.rag import RAGService

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

    TASK: Return a dict with:
    - status: "healthy"
    - service: "percepta-python"
    - timestamp: current datetime as ISO string

    LEARNING NOTE: Why async def?
    - FastAPI supports both sync and async
    - async is better for I/O operations (DB, API calls)
    - For simple returns, sync (def) works too
    - We use async for consistency (will need it later)
    """
    # TODO: Implement health check
    return {
        "status": "healthy",
        "service": "percepta-python",
        "timestamp": datetime.now().isoformat(),
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
    # TODO: Implement message reception
    logger.info(
        f"Received message from {message.username} in #{message.channel}: {message.message}"
    )

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
    # TODO: Implement send endpoint
    logger.debug(f"Send request for channel: {request.channel}")

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
        logger.info(
            f"Returning {len(responses)} message(s) for channel {request.channel}"
        )

    return SendResponse(
        messages=responses,
    )


@app.post("/rag/answer", response_model=RAGAnswerResponse)
async def rag_answer(request: RAGQueryRequest):
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service unavailable")

    try:
        result = await rag_service.answer(
            channel_id=request.channel,
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

    TASK: Log startup message
    - Log at INFO level
    - Message: f"Percepta Python service starting on {settings.host}:{settings.port}"

    LEARNING NOTE: Lifecycle events
    - startup: Runs once when server starts
    - shutdown: Runs once when server stops
    - Good for resource management
    """
    # TODO: Log startup message
    logger.info(f"Percepta Python service starting on {settings.host}:{settings.port}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Called when FastAPI shuts down

    Use for:
    - Closing database connections
    - Saving state
    - Cleanup

    TASK: Log shutdown message
    - Log at INFO level
    - Message: "Percepta Python service shutting down"

    LEARNING NOTE: Graceful shutdown
    - Clean up resources
    - Finish pending requests
    - Like the shutdown() function in Node index.js
    """
    # TODO: Log shutdown message
    logger.info("Percepta Python service shutting down")


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
