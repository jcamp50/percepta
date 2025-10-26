/**
 * Logging Utility
 *
 * Why separate logger?
 * - Consistent formatting across the application
 * - Easy to add features later (file logging, log levels)
 * - Single place to modify logging behavior
 */

// ANSI color codes for terminal output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  dim: '\x1b[2m',
};

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
 * Log an info message
 * @param {string} message - Message to log
 *
 * TASK: Format as "[TIMESTAMP] INFO: message" in default color
 */
function info(message) {
  console.log(`${colors.dim}[${getTimestamp()}] INFO: ${message}${colors.reset}`);
}

/**
 * Log a success message
 * @param {string} message - Success message
 *
 * TASK: Format as "[TIMESTAMP] ✓ message" in green
 */
function success(message) {
  console.log(`${colors.green}[${getTimestamp()}] ✓ ${message}${colors.reset}`);
}

/**
 * Log an error message
 * @param {string} message - Error message
 *
 * TASK: Format as "[TIMESTAMP] ✗ ERROR: message" in red
 */
function error(message) {
  console.log(`${colors.red}[${getTimestamp()}] ✗ ERROR: ${message}${colors.reset}`);
}

/**
 * Log a warning message
 * @param {string} message - Warning message
 *
 * TASK: Format as "[TIMESTAMP] ⚠ WARN: message" in yellow
 */
function warn(message) {
  console.log(`${colors.yellow}[${getTimestamp()}] ⚠ WARN: ${message}${colors.reset}`);
}

/**
 * Log a chat message with special formatting
 * @param {string} channel - Channel name
 * @param {string} username - User who sent message
 * @param {string} message - Message content
 *
 * TASK: Format as "[TIMESTAMP] #channel [username]: message" in cyan
 * This helps distinguish chat messages from system logs
 */
function chat(channel, username, message) {
  console.log(`${colors.cyan}[${getTimestamp()}] #${channel} ${username}: ${message}${colors.reset}`);
}

// Export all logging functions
module.exports = {
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
