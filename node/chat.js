/**
 * Twitch IRC Chat Client
 *
 * This module wraps the tmi.js library to:
 * - Manage connection lifecycle
 * - Handle all IRC events
 * - Provide a clean API for sending messages
 * - Implement our specific chat logic
 *
 * Why wrap tmi.js instead of using it directly?
 * - Separation of concerns (IRC logic separate from app logic)
 * - Easier to test and mock
 * - Can switch IRC libraries later if needed
 * - Encapsulates all Twitch-specific behavior
 */

const tmi = require('tmi.js');
const axios = require('axios');
const logger = require('./utils/logger');

/**
 * ChatClient Class
 *
 * Manages the IRC connection and provides message handling
 */
class ChatClient {
  constructor(config) {
    /**
     * TASK: Store the configuration
     * config should contain:
     * - botName: string
     * - oauthToken: string
     * - channel: string
     * - pythonServiceUrl: string
     */
    this.config = config;
    this.client = null;
    this.isConnected = false;
    this.pollInterval = null;
    this.isPolling = false; // Lock flag to prevent overlapping poll executions
  }

  /**
   * Initialize the tmi.js client
   *
   * TASK: Create a new tmi.Client with these options:
   * - identity: { username, password (oauth token) }
   * - channels: [channel to join]
   * - options: { debug: false }
   * - connection: {
   *     reconnect: true,  // Auto-reconnect on disconnect
   *     secure: true      // Use secure WebSocket
   *   }
   *
   * Store it in this.client
   * Then call this._setupEventHandlers()
   *
   * LEARNING NOTE: Why not create client in constructor?
   * - Constructor should be lightweight (no I/O)
   * - Allows async initialization if needed later
   * - Better error handling (constructor errors are harder to catch)
   */
  initialize() {
    // TODO: Create tmi.Client
    this.client = new tmi.Client({
      options: { debug: false },
      identity: {
        username: this.config.botName,
        password: this.config.oauthToken,
      },
      channels: [this.config.channel],
      connection: {
        reconnect: true,
        secure: true,
      },
    });
    // TODO: Call this._setupEventHandlers()
    this._setupEventHandlers();
    logger.info('Chat client initialized');
  }

  /**
   * Set up all event handlers
   *
   * PRIVATE METHOD (indicated by _ prefix - Node.js convention)
   *
   * TASK: Register handlers for these events:
   * - 'connected': Call this._onConnected
   * - 'message': Call this._onMessage
   * - 'disconnected': Call this._onDisconnected
   * - 'reconnect': Call this._onReconnect
   *
   * Example: this.client.on('connected', this._onConnected.bind(this))
   *
   * LEARNING NOTE: Why .bind(this)?
   * Event handlers lose their 'this' context when called.
   * .bind(this) ensures 'this' refers to our ChatClient instance.
   */
  _setupEventHandlers() {
    // TODO: Register event handlers with .bind(this)
    this.client.on('connected', this._onConnected.bind(this));
    this.client.on('message', this._onMessage.bind(this));
    this.client.on('disconnected', this._onDisconnected.bind(this));
    this.client.on('reconnect', this._onReconnect.bind(this));
  }

  /**
   * Handler for successful connection
   * @param {string} address - Server address
   * @param {number} port - Server port
   */
  _onConnected(address, port) {
    this.isConnected = true;
    // TODO: Log success message with address and port
    logger.success(`Connected to ${address}:${port}`);
    // TODO: Log which channel was joined
    logger.info(`Joined channel ${this.config.channel}`);
  }

  /**
   * Handler for incoming messages
   * @param {string} channel - Channel name (starts with #)
   * @param {Object} userstate - User information from Twitch
   * @param {string} message - Message content
   * @param {boolean} self - True if message is from our bot
   *
   * TASK:
   * 1. If self is true, return early (ignore our own messages)
   * 2. Extract username from userstate['display-name'] or userstate.username
   * 3. Log the message using logger.chat()
   *
   * LEARNING NOTE: userstate contains lots of Twitch metadata:
   * - badges, color, emotes, user-id, etc.
   * - For now we just need the username
   */
  _onMessage(channel, userstate, message, self) {
    // TODO: Implement message handling
    if (self) return;

    const username = userstate['display-name'] || userstate.username;

    if (
      username &&
      username.toLowerCase() === this.config.botName.toLowerCase()
    )
      return;

    const channelName = channel.replace('#', '');
    logger.chat(channelName, username, message);

    // Forward message to Python service (fire and forget)
    this._forwardMessageToPython(channelName, username, message);
  }

  /**
   * Forward message to Python service
   * @param {string} channel - Channel name (without #)
   * @param {string} username - Username who sent the message
   * @param {string} message - Message content
   * @private
   */
  _forwardMessageToPython(channel, username, message) {
    // Don't await - fire and forget to avoid blocking chat
    axios
      .post(`${this.config.pythonServiceUrl}/chat/message`, {
        channel: channel,
        username: username,
        message: message,
        timestamp: new Date().toISOString(),
      })
      .catch((error) => {
        // Log error but don't crash (Python might be down)
        logger.warn(`Failed to forward message to Python: ${error.message}`);
      });
  }

  /**
   * Handler for disconnection
   * @param {string} reason - Reason for disconnection
   */
  _onDisconnected(reason) {
    this.isConnected = false;
    // TODO: Log warning with reason
    logger.warn(`Disconnected from ${this.config.channel}: ${reason}`);
    // The client will auto-reconnect due to our config
  }

  /**
   * Handler for reconnection attempts
   */
  _onReconnect() {
    // TODO: Log info message about reconnecting
    logger.info(`Attempting to reconnect to ${this.config.channel}...`);
  }

  /**
   * Connect to Twitch IRC
   * @returns {Promise} Resolves when connected
   *
   * TASK: Call this.client.connect()
   * This returns a Promise, so you can await it
   *
   * LEARNING NOTE: Promises in Node.js
   * - Async operations return Promises
   * - Use await in async functions or .then()/.catch()
   * - Errors should be caught and handled
   */
  async connect() {
    // TODO: Implement connection
    // HINT: try/catch for error handling
    try {
      await this.client.connect();
      logger.success('Connected to Twitch IRC');

      // Start polling for responses from Python
      this.startPolling();
    } catch (error) {
      logger.error(`Failed to connect to Twitch IRC: ${error.message}`);
      throw error;
    }
  }

  /**
   * Disconnect from Twitch IRC
   * @returns {Promise} Resolves when disconnected
   */
  async disconnect() {
    // TODO: Implement disconnection
    // Should also set this.isConnected = false

    // Stop polling
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }

    try {
      await this.client.disconnect();
      this.isConnected = false;
      logger.info('Disconnected from Twitch IRC');
    } catch (error) {
      logger.error(`Failed to disconnect from Twitch IRC: ${error.message}`);
      throw error;
    }
  }

  /**
   * Send a message to a channel
   * @param {string} channel - Channel to send to (without #)
   * @param {string} message - Message to send
   * @returns {Promise} Resolves when message is sent
   *
   * TASK:
   * 1. Check if connected, throw error if not
   * 2. Call this.client.say(channel, message)
   * 3. Log that we sent the message
   *
   * LEARNING NOTE: Why validate isConnected?
   * - Fail fast with clear error messages
   * - Better than cryptic tmi.js errors
   */
  async sendMessage(channel, message) {
    // TODO: Implement message sending
    if (!this.isConnected) {
      throw new Error('Not connected to Twitch IRC');
    }
    try {
      await this.client.say(channel, message);
      logger.success(`Sent message to ${channel}: ${message}`);
    } catch (error) {
      logger.error(`Failed to send message to ${channel}: ${error.message}`);
      throw error;
    }
  }

  /**
   * Start polling Python service for queued responses
   * Polls every 500ms with protection against overlapping executions
   */
  startPolling() {
    logger.info('Starting polling for Python responses (500ms interval)');

    this.pollInterval = setInterval(async () => {
      // Skip this iteration if previous poll is still running
      if (this.isPolling) {
        logger.debug(
          'Skipping poll iteration - previous poll still in progress'
        );
        return;
      }

      // Set lock to prevent concurrent executions
      this.isPolling = true;

      try {
        // Poll Python /chat/send endpoint
        const response = await axios.post(
          `${this.config.pythonServiceUrl}/chat/send`,
          { channel: this.config.channel }
        );

        // Send each queued message
        const messages = response.data.messages || [];
        for (const msg of messages) {
          // If reply_to is set, prefix message with @username
          const messageText = msg.reply_to
            ? `@${msg.reply_to} ${msg.message}`
            : msg.message;

          await this.sendMessage(msg.channel, messageText);
        }
      } catch (error) {
        // Log error but don't crash polling
        if (error.code !== 'ECONNREFUSED') {
          // Only log non-connection errors to reduce spam
          logger.warn(`Polling error: ${error.message}`);
        }
      } finally {
        // Always release the lock, even if an error occurred
        this.isPolling = false;
      }
    }, 500);
  }
}

// Export the class
module.exports = ChatClient;

/**
 * LEARNING NOTES:
 *
 * 1. Class vs Functions:
 *    We use a class here because:
 *    - Maintains state (client, isConnected)
 *    - Groups related functionality
 *    - Can create multiple instances if needed later
 *
 * 2. Event-Driven Architecture:
 *    tmi.js emits events, we listen and respond.
 *    This is common in Node.js (EventEmitter pattern).
 *
 * 3. Async/Await:
 *    Network operations are async. Using async/await makes
 *    the code read like synchronous code while staying non-blocking.
 *
 * 4. Error Handling Strategy:
 *    - Wrap async operations in try/catch
 *    - Log errors but let tmi.js handle reconnection
 *    - Validate inputs before making calls
 */
