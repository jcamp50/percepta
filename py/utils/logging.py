"""
Category-aware logging utility for Percepta

Provides logging functionality with category filtering and log level control.
Logs can be filtered by category (audio, video, chat, stream_metadata, stream_event_sub, system)
and log level (DEBUG, INFO, WARN, ERROR).

Usage:
    from py.utils.logging import get_logger

    logger = get_logger(__name__, category='audio')
    logger.info('Audio chunk processed')

    video_logger = get_logger(__name__, category='video')
    video_logger.info('Video frame captured')
"""

import logging
from typing import Optional
from py.config import settings


# Log level hierarchy (lower number = more verbose)
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Parse allowed categories from settings
# If not set, show all categories (default behavior)
_allowed_categories = None
if settings.log_categories:
    _allowed_categories = [
        cat.strip().lower() for cat in settings.log_categories.split(",")
    ]


class CategoryFilter(logging.Filter):
    """Filter logs by category if LOG_CATEGORIES is set."""

    def __init__(self, category: Optional[str] = None):
        """
        Initialize category filter.

        Args:
            category: Category name for this logger (e.g., 'audio', 'video', 'chat', 'system')
        """
        super().__init__()
        self.category = category.lower() if category else "system"

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log record based on category.

        Args:
            record: Log record to filter

        Returns:
            True if record should be logged, False otherwise
        """
        # If no category filter is set, show all logs
        if _allowed_categories is None:
            return True

        # If category filter is set, check if this category is allowed
        return self.category in _allowed_categories


def get_logger(name: str, category: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with category filtering support.

    Args:
        name: Logger name (typically __name__)
        category: Category for filtering (e.g., 'audio', 'video', 'chat', 'system')
                  If None, defaults to 'system'

    Returns:
        Logger instance with category filter applied
    """
    logger = logging.getLogger(name)

    # Set log level from settings
    log_level_str = settings.log_level.upper()
    log_level = LOG_LEVELS.get(log_level_str, logging.INFO)
    logger.setLevel(log_level)

    # Remove existing category filters to avoid duplicates
    logger.filters = [f for f in logger.filters if not isinstance(f, CategoryFilter)]

    # Add category filter
    category_filter = CategoryFilter(category)
    logger.addFilter(category_filter)

    return logger
