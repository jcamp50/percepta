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
const axios = require('axios');
const ChatClient = require('./chat');
const AudioCapture = require('./audio');
const VideoCapture = require('./video');
const StreamManager = require('./stream');
const logger = require('./utils/logger');

let chatClient = null;
let audioCapture = null;
let videoCapture = null;
let streamManager = null;

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
    'TWITCH_CLIENT_ID', // Required for audio capture API calls
  ];
  const missingVars = requiredVars.filter((varName) => !process.env[varName]);
  if (missingVars.length > 0) {
    logger.error(
      `Missing required environment variables: ${missingVars.join(', ')}`,
      'system'
    );
    logger.info('Please check your .env file and try again.', 'system');
    process.exit(1);
  }

  // Set default for Python service URL if not provided
  if (!process.env.PYTHON_SERVICE_URL) {
    process.env.PYTHON_SERVICE_URL = 'http://localhost:8000';
    logger.info('Using default PYTHON_SERVICE_URL: http://localhost:8000', 'system');
  }

  logger.success('Environment variables validated successfully', 'system');
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
    logger.info('Starting Percepta Node Service...', 'system');
    validateEnvironment();
    chatClient = new ChatClient({
      botName: process.env.TWITCH_BOT_NAME,
      oauthToken: process.env.TWITCH_BOT_TOKEN,
      channel: process.env.TARGET_CHANNEL,
      pythonServiceUrl: process.env.PYTHON_SERVICE_URL,
    });
    chatClient.initialize();
    await chatClient.connect();
    logger.success('Percepta bot is now online!', 'system');

    // Initialize stream manager (shared between audio and video)
    try {
      streamManager = new StreamManager({
        pythonServiceUrl: process.env.PYTHON_SERVICE_URL,
        twitchClientId: process.env.TWITCH_CLIENT_ID,
        twitchOAuthToken: process.env.TWITCH_BOT_TOKEN,
      });

      await streamManager.initialize();
      logger.info('Stream manager initialized', 'system');
    } catch (error) {
      logger.warn(`Stream manager initialization failed: ${error.message}`, 'system');
      logger.info('Continuing without stream manager (audio/video capture unavailable)', 'system');
    }

    // Initialize audio capture with stream manager
    if (streamManager) {
      try {
        audioCapture = new AudioCapture({
          channel: process.env.TARGET_CHANNEL,
          pythonServiceUrl: process.env.PYTHON_SERVICE_URL,
          streamManager: streamManager,
          chunkSeconds: parseInt(process.env.AUDIO_CHUNK_SECONDS || '15', 10),
          sampleRate: parseInt(process.env.AUDIO_SAMPLE_RATE || '16000', 10),
          channels: parseInt(process.env.AUDIO_CHANNELS || '1', 10),
        });

        await audioCapture.initialize();
        logger.info('Audio capture initialized', 'system');
      } catch (error) {
        logger.warn(`Audio capture initialization failed: ${error.message}`, 'system');
        logger.info('Continuing without audio capture (chat bot still functional)', 'system');
      }

      // Initialize video capture with stream manager
      try {
        const baselineInterval = parseInt(process.env.VIDEO_CAPTURE_BASELINE_INTERVAL || '10', 10);
        const activeInterval = parseInt(process.env.VIDEO_CAPTURE_ACTIVE_INTERVAL || '5', 10);
        const initialInterval = parseInt(
          process.env.VIDEO_FRAME_INTERVAL || String(baselineInterval),
          10
        );

        videoCapture = new VideoCapture({
          channel: process.env.TARGET_CHANNEL,
          pythonServiceUrl: process.env.PYTHON_SERVICE_URL,
          streamManager: streamManager,
          baselineInterval,
          activeInterval,
          frameInterval: initialInterval,
        });

        await videoCapture.initialize();
        logger.info('Video capture initialized', 'system');
      } catch (error) {
        logger.warn(`Video capture initialization failed: ${error.message}`, 'system');
        logger.info('Continuing without video capture (chat bot still functional)', 'system');
      }

      // Start stream monitoring (will notify audio/video when stream is available)
      if (streamManager) {
        try {
          streamManager.startMonitoring(process.env.TARGET_CHANNEL);
          // Start audio and video capture (they will get stream URL from StreamManager events)
          if (audioCapture) {
            await audioCapture.startCapture(process.env.TARGET_CHANNEL);
          }
          if (videoCapture) {
            await videoCapture.startCapture(process.env.TARGET_CHANNEL);
          }
        } catch (error) {
          logger.warn(`Failed to start stream monitoring: ${error.message}`, 'system');
        }
      }
    }
  } catch (error) {
    logger.error(`Failed to start: ${error.message}`, 'system');
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
  logger.info('Shutting down Percepta Node Service...', 'system');

  // Stop video capture if running
  if (videoCapture) {
    try {
      await videoCapture.stopCapture();
    } catch (error) {
      logger.warn(`Error stopping video capture: ${error.message}`, 'system');
    }
  }

  // Stop audio capture if running
  if (audioCapture) {
    try {
      await audioCapture.stopCapture();
    } catch (error) {
      logger.warn(`Error stopping audio capture: ${error.message}`, 'system');
    }
  }

  // Stop stream monitoring
  if (streamManager) {
    try {
      streamManager.stopMonitoring();
    } catch (error) {
      logger.warn(`Error stopping stream manager: ${error.message}`, 'system');
    }
  }

  // Disconnect chat client
  if (chatClient) {
    try {
      await chatClient.disconnect();
    } catch (error) {
      logger.warn(`Error disconnecting chat client: ${error.message}`, 'system');
    }
  }

  logger.success('Percepta Node Service shut down successfully', 'system');
  process.exit(0);
}

// Set up signal handlers for graceful shutdown
// TASK: After creating chatClient in main(), register these handlers:
// - process.on('SIGINT', ...) - Ctrl+C
// - process.on('SIGTERM', ...) - Kill signal
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// Handle unhandled promise rejections to prevent silent failures
process.on('unhandledRejection', (reason, promise) => {
  logger.error(`Unhandled Promise Rejection: ${reason}`, 'system');
  if (reason instanceof Error) {
    logger.error(`Stack: ${reason.stack}`, 'system');
  }
  // Don't exit - log and continue, but this helps debug issues
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  logger.error(`Uncaught Exception: ${error.message}`, 'system');
  logger.error(`Stack: ${error.stack}`, 'system');
  // Exit on uncaught exceptions as they indicate serious problems
  shutdown().then(() => {
    process.exit(1);
  });
});

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
