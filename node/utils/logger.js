/**
 * Logging Utility
 *
 * Why separate logger?
 * - Consistent formatting across the application
 * - Easy to add features later (file logging, log levels)
 * - Single place to modify logging behavior
 *
 * Available log categories:
 * - 'audio': Audio capture and processing
 * - 'video': Video frame capture and processing
 * - 'video_description': Visual description generation pipeline
 * - 'stream': Stream management and monitoring
 * - 'chat': Chat message handling
 * - 'system': System-level logs (default)
 *
 * Filter logs by category using LOG_CATEGORIES environment variable:
 *   LOG_CATEGORIES=audio,video
 *   LOG_CATEGORIES=stream
 */

require('dotenv').config(); // Load .env file for environment variables

// ANSI color codes for terminal output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  dim: '\x1b[2m',
};

// Log level hierarchy (lower number = more verbose)
const LOG_LEVELS = {
  DEBUG: 0,
  INFO: 1,
  WARN: 2,
  ERROR: 3,
};

// Get log level from environment (default: INFO)
const currentLogLevel = LOG_LEVELS[process.env.LOG_LEVEL?.toUpperCase()] ?? LOG_LEVELS.INFO;

// Get allowed categories from environment (comma-separated)
// If not set, show all categories (default behavior)
const allowedCategories = process.env.LOG_CATEGORIES
  ? process.env.LOG_CATEGORIES.split(',').map(cat => cat.trim().toLowerCase())
  : null; // null means show all categories

/**
 * Check if a log should be displayed based on level and category
 * @param {number} level - Log level (from LOG_LEVELS)
 * @param {string|null} category - Category name (e.g., 'audio', 'video', 'stream', 'chat', or null for system/unclassified)
 * @returns {boolean} True if log should be displayed
 */
function shouldLog(level, category = null) {
  // Check log level first
  if (level < currentLogLevel) {
    return false;
  }

  // If no category filter is set, show all logs
  if (allowedCategories === null) {
    return true;
  }

  // If category filter is set, check if this category is allowed
  // If category is null/undefined, treat it as "system" category
  const categoryToCheck = category ? category.toLowerCase() : 'system';
  return allowedCategories.includes(categoryToCheck);
}

/**
 * Get current timestamp in readable format
 * @returns {string} Formatted timestamp
 *
 * TASK: Implement this to return something like "2024-10-26 10:30:45"
 * HINT: Use Date object and pad numbers with leading zeros
 */
function getTimestamp() {
  const timestamp = new Date().toISOString().replace('T', ' ').substring(0, 19);
  return timestamp;
}

/**
 * Log a debug message
 * @param {string} message - Message to log
 * @param {string|null} category - Optional category for filtering (e.g., 'audio', 'video', 'stream', 'chat', 'system')
 */
function debug(message, category = null) {
  if (!shouldLog(LOG_LEVELS.DEBUG, category)) {
    return;
  }
  console.log(`${colors.dim}[${getTimestamp()}] DEBUG: ${message}${colors.reset}`);
}

/**
 * Log an info message
 * @param {string} message - Message to log
 * @param {string|null} category - Optional category for filtering (e.g., 'audio', 'video', 'stream', 'chat', 'system')
 */
function info(message, category = null) {
  if (!shouldLog(LOG_LEVELS.INFO, category)) {
    return;
  }
  const logMessage = `${colors.dim}[${getTimestamp()}] INFO: ${message}${colors.reset}`;
  console.log(logMessage);
  // Force flush on Windows to ensure logs appear immediately
  if (process.platform === 'win32' && process.stdout.isTTY) {
    process.stdout.write('', () => {}); // Force flush
  }
}

/**
 * Log a success message
 * @param {string} message - Success message
 * @param {string|null} category - Optional category for filtering (e.g., 'audio', 'video', 'stream', 'chat', 'system')
 */
function success(message, category = null) {
  if (!shouldLog(LOG_LEVELS.INFO, category)) {
    return;
  }
  console.log(`${colors.green}[${getTimestamp()}] ✓ ${message}${colors.reset}`);
}

/**
 * Log an error message
 * @param {string} message - Error message
 * @param {string|null} category - Optional category for filtering (e.g., 'audio', 'video', 'stream', 'chat', 'system')
 */
function error(message, category = null) {
  if (!shouldLog(LOG_LEVELS.ERROR, category)) {
    return;
  }
  const logMessage = `${colors.red}[${getTimestamp()}] ✗ ERROR: ${message}${colors.reset}`;
  console.error(logMessage);
  // Force flush on Windows to ensure logs appear immediately
  if (process.platform === 'win32' && process.stderr.isTTY) {
    process.stderr.write('', () => {}); // Force flush
  }
}

/**
 * Log a warning message
 * @param {string} message - Warning message
 * @param {string|null} category - Optional category for filtering (e.g., 'audio', 'video', 'stream', 'chat', 'system')
 */
function warn(message, category = null) {
  if (!shouldLog(LOG_LEVELS.WARN, category)) {
    return;
  }
  const logMessage = `${colors.yellow}[${getTimestamp()}] ⚠ WARN: ${message}${colors.reset}`;
  console.log(logMessage);
  // Force flush on Windows to ensure logs appear immediately
  if (process.platform === 'win32' && process.stdout.isTTY) {
    process.stdout.write('', () => {}); // Force flush
  }
}

/**
 * Log a chat message with special formatting
 * @param {string} channel - Channel name
 * @param {string} username - User who sent message
 * @param {string} message - Message content
 * @param {string|null} category - Optional category for filtering (defaults to 'chat')
 *
 * TASK: Format as "[TIMESTAMP] #channel [username]: message" in cyan
 * This helps distinguish chat messages from system logs
 */
function chat(channel, username, message, category = 'chat') {
  if (!shouldLog(LOG_LEVELS.INFO, category)) {
    return;
  }
  console.log(`${colors.cyan}[${getTimestamp()}] #${channel} ${username}: ${message}${colors.reset}`);
}

// Export all logging functions
module.exports = {
  debug,
  info,
  success,
  error,
  warn,
  chat,
};

/**
 * LEARNING NOTES:
 *
 * 1. Module Exports: In Node.js, module.exports defines what other files
 *    can import from this file. Think of it as the "public API".
 *
 * 2. ANSI Colors: Terminal colors use escape codes. The pattern is:
 *    console.log('\x1b[32m' + 'green text' + '\x1b[0m')
 *    The \x1b[0m resets the color after your text.
 *
 * 3. Why functions instead of console.log directly?
 *    - Consistency: All logs look the same
 *    - Flexibility: Easy to add file logging later
 *    - Testing: You can mock these functions in tests
 */
