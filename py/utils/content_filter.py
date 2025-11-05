"""
Content Filtering Utilities

Provides safety checks for chat messages to ensure:
- No toxic/offensive content
- No personal information disclosure
- Twitch TOS compliance
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# Basic toxic keywords (can be expanded or replaced with ML-based filtering)
TOXIC_KEYWORDS = [
    # Add common toxic/offensive terms here
    # Keeping minimal for now - can be expanded with more sophisticated filtering
]

# Patterns for detecting personal information
PII_PATTERNS = [
    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN format
    r'\b\d{3}\.\d{2}\.\d{4}\b',  # SSN format with dots
    r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card (simplified)
    r'\b\d{10}\b',  # Phone number (10 digits)
    r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number format
    r'\b\d{3}\.\d{3}\.\d{4}\b',  # Phone number format with dots
    r'\b\d{3}\s\d{3}\s\d{4}\b',  # Phone number format with spaces
]

# Email pattern (basic)
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'


def is_toxic_content(message: str) -> bool:
    """
    Check if message contains toxic/offensive content.

    Args:
        message: Message text to check

    Returns:
        True if message contains toxic content, False otherwise
    """
    message_lower = message.lower()
    
    # Check against toxic keywords
    for keyword in TOXIC_KEYWORDS:
        if keyword.lower() in message_lower:
            logger.debug(f"Toxic content detected: {keyword}")
            return True
    
    # Additional checks can be added here (e.g., ML-based classification)
    return False


def contains_pii(message: str) -> bool:
    """
    Detect if message contains personal information.

    Args:
        message: Message text to check

    Returns:
        True if PII detected, False otherwise
    """
    # Check for email addresses
    if re.search(EMAIL_PATTERN, message, re.IGNORECASE):
        logger.debug("PII detected: Email address")
        return True
    
    # Check for PII patterns
    for pattern in PII_PATTERNS:
        if re.search(pattern, message):
            logger.debug(f"PII detected: Pattern {pattern}")
            return True
    
    return False


def is_safe_for_chat(message: str) -> bool:
    """
    Main safety check combining all filters.

    Args:
        message: Message text to check

    Returns:
        True if message is safe, False if unsafe content detected
    """
    if not message or not message.strip():
        return True  # Empty messages are safe
    
    # Check for toxic content
    if is_toxic_content(message):
        return False
    
    # Check for PII
    if contains_pii(message):
        return False
    
    return True


def filter_response_content(response: str) -> str:
    """
    Filter response content to ensure it's safe.

    Args:
        response: Response text to filter

    Returns:
        Filtered response text
    """
    # For now, just return the response as-is if it passes safety checks
    # In the future, this could sanitize or redact content
    if is_safe_for_chat(response):
        return response
    
    # If unsafe, return a safe fallback message
    logger.warning("Unsafe response content detected, returning safe fallback")
    return "I'm unable to provide that response due to safety guidelines."

