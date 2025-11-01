"""
Message Schemas

These Pydantic models define:
- What data we expect to receive
- What data we send back
- Type validation
- Documentation

Why Pydantic models?
- Automatic validation (FastAPI checks data matches schema)
- Auto-generated API documentation (Swagger/OpenAPI)
- Type safety
- Serialization (Python objects ↔ JSON)
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional


class ChatMessage(BaseModel):
    """
    Incoming chat message from Node IRC service

    This represents a message that a user sent in Twitch chat.
    Node IRC service captures it and sends it here.

    TASK: Define these fields:
    - channel: str (e.g., "mychannel")
    - username: str (e.g., "viewer123")
    - message: str (the actual message text)
    - timestamp: datetime (when message was sent)

    LEARNING NOTE: Field() allows you to add:
    - description: Shows in API docs
    - example: Shows in API docs
    - min_length, max_length: Validation rules
    """

    # TODO: Define fields
    # Example:
    # channel: str = Field(..., description="Twitch channel name", example="mychannel")
    # username: str = Field(..., description="Username who sent message")
    channel: str = Field(..., description="Twitch channel name", example="mychannel")
    username: str = Field(..., description="Username who sent message")
    message: str = Field(..., description="The actual message text")
    timestamp: datetime = Field(..., description="When message was sent")

    class Config:
        """
        Model configuration

        TASK: Add json_schema_extra with an example
        This shows in API documentation
        """

        json_schema_extra = {
            "example": {
                # TODO: Add example data
                # "channel": "mychannel",
                # "username": "viewer123",
                # ...
                "channel": "mychannel",
                "username": "viewer123",
                "message": "Hello, world!",
                "timestamp": "2025-01-01T00:00:00Z"
            }
        }


class MessageReceived(BaseModel):
    """
    Response when we receive a message

    Acknowledges that we got the message from Node.

    TASK: Define fields:
    - received: bool (always True if we got here)
    - message_id: str (unique ID for tracking)
    - timestamp: datetime (when we received it)

    LEARNING NOTE: Why acknowledge?
    - Node knows message was delivered
    - Can retry if acknowledgment fails
    - Good for debugging (track message flow)
    """

    # TODO: Define fields
    received: bool = Field(..., description="Always True if we got here")
    message_id: str = Field(..., description="Unique ID for tracking")
    timestamp: datetime = Field(..., description="When we received it")


class ChatResponse(BaseModel):
    """
    Response message to send back to Twitch

    This is what the bot should say in chat.
    Node IRC service will send this to Twitch.

    TASK: Define fields:
    - channel: str (which channel to send to)
    - message: str (what to say)
    - reply_to: Optional[str] (username to reply to, if any)

    LEARNING NOTE: Why reply_to?
    - Twitch supports @mentions
    - Helps users know who bot is responding to
    - Optional because not all messages are replies
    """

    # TODO: Define fields
    channel: str = Field(..., description="Which channel to send to")
    message: str = Field(..., description="What to say")
    reply_to: Optional[str] = Field(None, description="Username to reply to, if any")


class SendRequest(BaseModel):
    """
    Request from Node asking for messages to send

    Node polls this endpoint to get queued responses.

    TASK: Define field:
    - channel: str (which channel is asking for messages)

    LEARNING NOTE: Why just channel?
    - Node manages multiple channels (future)
    - Only get messages for the channel that's asking
    - Keeps messages isolated per channel
    """

    # TODO: Define field
    channel: str = Field(..., description="Which channel is asking for messages")


class SendResponse(BaseModel):
    """
    Response with messages to send

    List of messages for Node to send to Twitch.

    TASK: Define field:
    - messages: list[ChatResponse] (list of messages to send)

    LEARNING NOTE: Why a list?
    - Bot might have multiple things to say
    - Batch responses efficiently
    - Node can send them in order
    """

    # TODO: Define field
    messages: list[ChatResponse] = Field(..., description="List of messages to send")


class RAGQueryRequest(BaseModel):
    channel: str = Field(..., description="Twitch channel identifier", example="mychannel")
    question: str = Field(..., description="Viewer question to answer")
    top_k: Optional[int] = Field(None, description="Override number of chunks to retrieve")
    half_life_minutes: Optional[int] = Field(
        None, description="Override for time-decay half-life in minutes"
    )
    prefilter_limit: Optional[int] = Field(
        None,
        description="Override for IVFFLAT prefilter limit before rescoring",
    )


class RAGCitation(BaseModel):
    id: str = Field(..., description="Identifier of the cited transcript chunk")
    timestamp: str = Field(..., description="Timestamp marker like (~01:23:10)")


class RAGChunkResult(BaseModel):
    id: str
    channel_id: str
    started_at: datetime
    ended_at: datetime
    midpoint: datetime
    timestamp: str
    text: str
    score: float
    cosine_distance: float


class RAGAnswerResponse(BaseModel):
    answer: str
    citations: List[RAGCitation]
    chunks: List[RAGChunkResult]
    context: List[str]


"""
LEARNING NOTES:

1. Request vs Response Models:
  - Request: Data coming IN to our API
  - Response: Data going OUT from our API
  - FastAPI validates both automatically

2. Pydantic Validation:
  When someone sends invalid data:
  - Missing required field? → 422 error
  - Wrong type (string instead of int)? → 422 error
  - FastAPI handles this automatically!

3. Field(...):
  - ... means "required, no default"
  - Field("default") means "optional with default"
  - Field(..., description="...") adds documentation

4. Model Relationships:
  ChatMessage → MessageReceived (acknowledgment)
  SendRequest → SendResponse (contains ChatResponse list)

USAGE EXAMPLE:
    # In FastAPI endpoint:
    @app.post("/chat/message")
    async def receive_message(msg: ChatMessage):
        # msg is automatically validated!
        print(msg.username)  # Guaranteed to exist
        print(msg.timestamp) # Guaranteed to be datetime
        return MessageReceived(
            received=True,
            message_id="abc123",
            timestamp=datetime.now()
        )
"""
