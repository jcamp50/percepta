"""
Configuration Management

Why separate config file?
- Single source of truth for all settings
- Type validation (Pydantic catches errors early)
- Environment variable loading with defaults
- Easy to test (can inject different configs)

Why Pydantic Settings vs manual env loading?
- Automatic type conversion (string "8000" → int 8000)
- Validation (required fields, valid ranges)
- Default values
- Auto-generated documentation
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application Settings

    Pydantic automatically loads from environment variables.
    Variable names match field names (case-insensitive).

    TASK: Define these settings with appropriate types:
    - log_level: str with default "INFO"
    - port: int with default 8000
    - host: str with default "0.0.0.0"
    - openai_api_key: Optional[str] (not required yet, will use later)

    LEARNING NOTE: Why Optional[str]?
    - Optional means it can be None
    - We don't need OpenAI yet, so it's optional for now
    - Later phases will require it
    """

    # TODO: Define settings fields here
    # Example:
    # log_level: str = "INFO"
    # port: int = 8000
    log_level: str = "INFO"
    port: int = 8000
    host: str = "0.0.0.0"
    openai_api_key: Optional[str] = None
    twitch_bot_name: Optional[str] = None
    rag_top_k: int = 5
    rag_prefilter_limit: int = 40
    rag_half_life_minutes: int = 60
    rag_context_char_limit: int = 2000
    rag_temperature: float = 0.4
    rag_completion_model: str = "gpt-4o-mini"
    rag_system_prompt_file: str = "py/reason/prompts/system_prompt.txt"
    rag_user_prompt_file: str = "py/reason/prompts/user_prompt.txt"

    # Whisper Configuration
    whisper_model: str = "base"  # tiny, base, small, medium, large
    whisper_language: Optional[str] = "en"  # None for auto-detect
    use_gpu: bool = False  # Enable GPU if available

    # Twitch Configuration
    twitch_client_id: Optional[str] = None
    twitch_bot_token: Optional[str] = None
    target_channel: Optional[str] = None  # Channel to monitor (e.g., "channelname")
    
    # EventSub WebSocket Configuration
    eventsub_enabled: bool = True  # Enable/disable EventSub client
    eventsub_reconnect_delay: int = 5  # Initial reconnect delay in seconds
    eventsub_max_reconnect_delay: int = 60  # Maximum reconnect delay in seconds

    class Config:
        """
        Pydantic configuration class

        TASK: Set these:
        - env_file: ".env"  # Load from .env file
        - case_sensitive: False  # LOG_LEVEL matches log_level
        - extra: "ignore"  # Ignore extra env vars (like Twitch creds for Node service)

        LEARNING NOTE: This tells Pydantic WHERE to find variables
        """

        # TODO: Set env_file and case_sensitive
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Allow extra env vars without validation errors


# Create a global settings instance
# This is loaded once when the module is imported
# TODO: Create settings instance
# settings = Settings()
settings = Settings()

"""
LEARNING NOTES:

1. Pydantic Settings vs BaseModel:
  - BaseSettings: For configuration (loads from env vars)
  - BaseModel: For request/response data (validates JSON)

2. Type Hints:
    Python uses type hints for documentation and validation
  - str, int, bool are simple types
  - Optional[str] means "str or None"

3. Default Values:
  - field: str = "default"
  - If env var not found, uses default

4. Why global instance?
  - Load config once at startup
  - Reuse across all endpoints
  - Efficient and consistent

USAGE EXAMPLE:
    from py.config import settings
    
    print(settings.log_level)  # "INFO"
    print(settings.port)       # 8000
"""
