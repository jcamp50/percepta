/**
 * Percepta Node Service - Entry Point
 *
 * This file:
 * - Loads configuration
 * - Creates and starts the chat client
 * - Handles graceful shutdown
 *
 * Why separate from chat.js?
 * - Single Responsibility: chat.js handles IRC, index.js handles app lifecycle
 * - Easier testing: Can import ChatClient without running the app
 * - Clear entry point: Anyone can see how the app starts
 */

require('dotenv').config(); // Load .env file
const ChatClient = require('./chat');
const logger = require('./utils/logger');

let chatClient = null;

/**
 * Validate required environment variables
 *
 * TASK: Check that these env vars exist:
 * - TWITCH_BOT_NAME
 * - TWITCH_BOT_TOKEN
 * - TARGET_CHANNEL
 *
 * If any are missing:
 * - Log error for each missing var
 * - Log instructions to check .env file
 * - Exit with process.exit(1)
 *
 * LEARNING NOTE: Why validate early?
 * - Fail fast with clear errors
 * - Better than cryptic errors later
 * - process.exit(1) indicates error to the shell
 */
function validateEnvironment() {
  // TODO: Implement validation
  const requiredVars = [
    'TWITCH_BOT_NAME',
    'TWITCH_BOT_TOKEN',
    'TARGET_CHANNEL',
  ];
  const missingVars = requiredVars.filter((varName) => !process.env[varName]);
  if (missingVars.length > 0) {
    logger.error(
      `Missing required environment variables: ${missingVars.join(', ')}`
    );
    logger.info('Please check your .env file and try again.');
    process.exit(1);
  }
  logger.success('Environment variables validated successfully');
}

/**
 * Main application logic
 *
 * TASK:
 * 1. Log startup message
 * 2. Validate environment
 * 3. Create ChatClient instance with config object:
 *    { botName, oauthToken, channel }
 * 4. Initialize the client
 * 5. Connect to IRC (await this)
 * 6. Log success message
 *
 * LEARNING NOTE: Why async function?
 * - We need to await the connection
 * - async functions automatically return Promises
 * - Allows cleaner error handling with try/catch
 */
async function main() {
  try {
    // TODO: Implement main logic
    logger.info('Starting Percepta Node Service...');
    validateEnvironment();
    chatClient = new ChatClient({
      botName: process.env.TWITCH_BOT_NAME,
      oauthToken: process.env.TWITCH_BOT_TOKEN,
      channel: process.env.TARGET_CHANNEL,
    });
    chatClient.initialize();
    await chatClient.connect();
    logger.success('Percepta bot is now online!');
  } catch (error) {
    logger.error(`Failed to start: ${error.message}`);
    process.exit(1);
  }
}

/**
 * Handle graceful shutdown
 * @param {ChatClient} chatClient - Client to disconnect
 *
 * TASK:
 * 1. Log shutdown message
 * 2. Disconnect the chat client (await this)
 * 3. Log goodbye message
 * 4. Exit with process.exit(0)
 *
 * LEARNING NOTE: Graceful shutdown
 * - Close connections properly
 * - Prevent data loss
 * - Clean up resources
 * - process.exit(0) indicates successful exit
 */
async function shutdown() {
  // Remove parameter
  if (!chatClient) {
    // Safety check
    logger.warn('No client to disconnect');
    process.exit(0);
    return;
  }

  logger.info('Shutting down Percepta Node Service...');
  await chatClient.disconnect();
  logger.success('Percepta Node Service shut down successfully');
  process.exit(0);
}

// Set up signal handlers for graceful shutdown
// TASK: After creating chatClient in main(), register these handlers:
// - process.on('SIGINT', ...) - Ctrl+C
// - process.on('SIGTERM', ...) - Kill signal
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// Start the application
main();

/**
 * LEARNING NOTES:
 *
 * 1. Module Loading Order:
 *    - require('dotenv') must come first to load .env
 *    - Then import our modules
 *    - Then run main()
 *
 * 2. Error Handling Strategy:
 *    - Catch errors in main()
 *    - Log them clearly
 *    - Exit with non-zero code (indicates failure)
 *
 * 3. Process Signals:
 *    - SIGINT: User pressed Ctrl+C
 *    - SIGTERM: System wants to stop the process
 *    - Handling these allows clean shutdown
 *
 * 4. Why no 'export' in this file?
 *    - This is an entry point, not a library
 *    - It's meant to be run directly: node index.js
 *    - Nothing should import this file
 */
