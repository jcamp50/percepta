/**
 * Twitch Stream Manager
 *
 * This module manages a single shared video stream URL for both audio and video processors.
 * It eliminates duplicate Streamlink calls and provides centralized stream monitoring.
 *
 * Responsibilities:
 * - Fetch and cache video stream URL from Python service
 * - Monitor stream status (live/offline)
 * - Share stream URL with audio and video processors via events
 * - Handle stream URL refresh when stream goes offline/online
 * - Provide broadcaster ID resolution
 */

const EventEmitter = require('events');
const axios = require('axios');
const logger = require('./utils/logger');

/**
 * StreamManager Class
 *
 * Manages shared video stream URL and broadcaster ID for audio/video processors
 */
class StreamManager extends EventEmitter {
  constructor(config) {
    super();

    /**
     * config should contain:
     * - pythonServiceUrl: string (Python backend URL)
     * - twitchClientId: string (Twitch API client ID)
     * - twitchOAuthToken: string (Twitch OAuth token for API calls)
     */
    this.pythonServiceUrl = config.pythonServiceUrl || 'http://localhost:8000';
    this.twitchClientId = config.twitchClientId;
    this.twitchOAuthToken = config.twitchOAuthToken;

    // Cached stream data
    this.streamUrl = null;
    this.broadcasterId = null;
    this.channelId = null;

    // Monitoring state
    this.monitorInterval = null;
    this.isMonitoring = false;
    this.lastStreamStatus = null;
  }

  /**
   * Initialize the stream manager
   * Validates configuration
   */
  initialize() {
    if (!this.twitchClientId) {
      throw new Error('TWITCH_CLIENT_ID is required for stream manager');
    }

    logger.info('Stream manager initialized', 'stream');
    return Promise.resolve();
  }

  /**
   * Check if Twitch channel is currently live
   * @param {string} channelId - Channel name (without #)
   * @returns {Promise<boolean>} True if stream is live
   */
  async _checkStreamStatus(channelId) {
    try {
      const token = this.twitchOAuthToken?.replace(/^oauth:/, '') || '';

      if (!token) {
        logger.warn('No OAuth token provided for Twitch API calls', 'stream');
        return false;
      }

      const response = await axios.get(
        `https://api.twitch.tv/helix/streams?user_login=${channelId}`,
        {
          headers: {
            'Client-ID': this.twitchClientId,
            Authorization: `Bearer ${token}`,
          },
        }
      );

      return response.data.data && response.data.data.length > 0;
    } catch (error) {
      if (error.response?.status === 404) {
        return false; // Channel doesn't exist
      }
      if (error.response?.status === 401) {
        logger.error(
          'Twitch API authentication failed. Check TWITCH_BOT_TOKEN and TWITCH_CLIENT_ID.',
          'stream'
        );
        return false;
      }
      logger.warn(`Failed to check stream status: ${error.message}`, 'stream');
      return false;
    }
  }

  /**
   * Get broadcaster ID from channel name
   * @param {string} channelId - Channel name
   * @returns {Promise<string|null>} Broadcaster ID or null
   */
  async getBroadcasterId(channelId) {
    // Return cached broadcaster ID if available and same channel
    if (this.broadcasterId && this.channelId === channelId) {
      return this.broadcasterId;
    }

    try {
      const response = await axios.get(
        `${this.pythonServiceUrl}/api/get-broadcaster-id`,
        {
          params: { channel_name: channelId },
          timeout: 15000, // Increased from 5s to 15s for better reliability
          httpAgent: new (require('http').Agent)({
            keepAlive: true,
            keepAliveMsecs: 30000, // Keep connections alive for 30 seconds
            timeout: 15000,
          }),
          httpsAgent: new (require('https').Agent)({
            keepAlive: true,
            keepAliveMsecs: 30000,
            timeout: 15000,
          }),
        }
      );

      const broadcasterId = response.data.broadcaster_id || null;
      if (broadcasterId) {
        this.broadcasterId = broadcasterId;
        this.channelId = channelId;
      }

      return broadcasterId;
    } catch (error) {
      // Detailed error logging to see actual issues
      if (error.response) {
        const status = error.response.status;
        const detail =
          error.response.data?.detail ||
          error.response.data?.message ||
          'Unknown error';

        if (status === 503) {
          logger.error(
            `EventSub client not available for broadcaster ID lookup: ${detail}`,
            'stream'
          );
        } else if (status === 404) {
          logger.warn(
            `Channel not found: ${channelId}. Detail: ${detail}`,
            'stream'
          );
        } else if (status === 500) {
          logger.error(
            `Python service error getting broadcaster ID for ${channelId}: ${detail}`,
            'stream'
          );
        } else {
          logger.error(
            `Failed to get broadcaster ID for ${channelId}: HTTP ${status} - ${detail}`,
            'stream'
          );
        }
      } else if (error.code === 'ECONNREFUSED') {
        logger.error(
          `Cannot connect to Python service at ${this.pythonServiceUrl}. Is it running?`,
          'stream'
        );
      } else if (error.code === 'ETIMEDOUT') {
        logger.error(
          `Timeout waiting for broadcaster ID from Python service (${channelId})`,
          'stream'
        );
      } else {
        logger.error(
          `Failed to get broadcaster ID for ${channelId}: ${error.message}`,
          'stream'
        );
      }

      return null;
    }
  }

  /**
   * Fetch video stream URL from Python service
   * @param {string} channelId - Channel name (without #)
   * @returns {Promise<string|null>} Video stream URL or null if not available
   */
  async _fetchStreamUrl(channelId) {
    try {
      logger.info(
        `Fetching video stream URL from Python service for channel ${channelId}`,
        'stream'
      );

      const response = await axios.get(
        `${this.pythonServiceUrl}/api/get-video-stream-url`,
        {
          params: {
            channel_id: channelId,
          },
          timeout: 30000, // Increased from 10s to 30s for better reliability
          httpAgent: new (require('http').Agent)({
            keepAlive: true,
            keepAliveMsecs: 30000,
            timeout: 30000,
          }),
          httpsAgent: new (require('https').Agent)({
            keepAlive: true,
            keepAliveMsecs: 30000,
            timeout: 30000,
          }),
        }
      );

      logger.info(
        `Python service response for ${channelId}: status=${
          response.status
        }, data=${JSON.stringify(response.data)}`,
        'stream'
      );

      const { stream_url, available } = response.data;

      if (!available || !stream_url) {
        logger.warn(
          `Video stream not available for channel ${channelId} via Streamlink. ` +
            `Response: available=${available}, stream_url=${
              stream_url ? 'present' : 'missing'
            }`,
          'stream'
        );
        return null;
      }

      logger.info(
        `Retrieved authenticated video stream URL for channel ${channelId} via Streamlink. ` +
          `URL length: ${stream_url.length} chars`,
        'stream'
      );

      return stream_url;
    } catch (error) {
      logger.error(
        `Failed to get video stream URL via Streamlink: ${error.message}`,
        'stream'
      );

      if (error.response) {
        logger.error(
          `HTTP error details: status=${error.response.status}, ` +
            `data=${JSON.stringify(error.response.data)}`,
          'stream'
        );

        if (error.response.status === 503) {
          logger.error(
            'Python Streamlink service unavailable. Ensure streamlink is installed: pip install streamlink',
            'stream'
          );
        } else if (error.response.status === 502) {
          logger.error(
            `Streamlink plugin error for channel ${channelId}. Channel may be offline or restricted.`,
            'stream'
          );
        } else if (error.response.status === 500) {
          logger.error(
            `Python service internal error. Check Python service logs for details.`,
            'stream'
          );
        }
      } else if (error.code === 'ECONNREFUSED') {
        logger.error(
          `Cannot connect to Python service at ${this.pythonServiceUrl}. Is it running?`,
          'stream'
        );
      } else if (error.code === 'ETIMEDOUT') {
        logger.error(
          'Timeout waiting for Python service response. Streamlink may be taking too long.',
          'stream'
        );
      } else {
        logger.error(
          `Unexpected error: ${error.code || 'unknown'}, message: ${
            error.message
          }`,
          'stream'
        );
      }

      return null;
    }
  }

  /**
   * Get stream URL (cached or fetch new)
   * @param {string} channelId - Channel name (without #)
   * @returns {Promise<string|null>} Stream URL or null if not available
   */
  async getStreamUrl(channelId) {
    // Return cached URL if available and same channel
    if (this.streamUrl && this.channelId === channelId) {
      return this.streamUrl;
    }

    // Fetch new stream URL
    const streamUrl = await this._fetchStreamUrl(channelId);
    if (streamUrl) {
      this.streamUrl = streamUrl;
      this.channelId = channelId;

      // Get broadcaster ID if not cached (retry if needed)
      if (!this.broadcasterId || this.channelId !== channelId) {
        const broadcasterId = await this.getBroadcasterId(channelId);

        // If broadcaster ID lookup failed, retry once more after a short delay
        if (!broadcasterId) {
          logger.warn(
            `Broadcaster ID lookup failed for ${channelId}, retrying...`,
            'stream'
          );
          await new Promise((resolve) => setTimeout(resolve, 2000)); // Wait 2 seconds
          const retryBroadcasterId = await this.getBroadcasterId(channelId);

          if (!retryBroadcasterId) {
            logger.error(
              `Failed to get broadcaster ID for ${channelId} after retry. Capture will use channel name.`,
              'stream'
            );
          }
        }
      }

      // Emit stream URL event
      this.emit('streamUrl', {
        streamUrl: streamUrl,
        broadcasterId: this.broadcasterId,
        channelId: channelId,
      });
    }

    return streamUrl;
  }

  /**
   * Start monitoring stream status
   * @param {string} channelId - Channel name to monitor
   */
  startMonitoring(channelId) {
    if (this.isMonitoring) {
      logger.warn('Stream monitoring already active', 'stream');
      return;
    }

    this.isMonitoring = true;
    this.channelId = channelId;

    logger.info(`Starting stream monitoring for ${channelId}`, 'stream');

    // Check stream status immediately - FIX: Add error handling and make it non-blocking
    this._checkAndUpdateStream(channelId).catch((error) => {
      logger.error(
        `Error in initial stream status check: ${error.message}`,
        'stream'
      );
      logger.error(error.stack, 'stream');
    });

    // Check every 30 seconds if stream is live
    this.monitorInterval = setInterval(async () => {
      try {
        await this._checkAndUpdateStream(channelId);
      } catch (error) {
        logger.error(
          `Error in periodic stream status check: ${error.message}`,
          'stream'
        );
      }
    }, 30000); // Check every 30 seconds
  }

  /**
   * Check stream status and update accordingly
   * @param {string} channelId - Channel name
   */
  async _checkAndUpdateStream(channelId) {
    try {
      logger.info(`Checking stream status for ${channelId}...`, 'stream');
      const isLive = await this._checkStreamStatus(channelId);
      logger.info(
        `Stream status for ${channelId}: ${isLive ? 'LIVE' : 'OFFLINE'}`,
        'stream'
      );

      if (isLive && this.lastStreamStatus !== true) {
        // Stream just came online
        logger.info(`Stream is now live for ${channelId}`, 'stream');
        this.lastStreamStatus = true;
        this.emit('streamOnline', { channelId: channelId });

        // Wait a bit for Streamlink to be ready (streams can take a few seconds to be fully available)
        logger.info(
          `Waiting 3 seconds before fetching stream URL for ${channelId} (Streamlink may need time to process)`,
          'stream'
        );
        await new Promise((resolve) => setTimeout(resolve, 3000));

        // Try to fetch stream URL with retries
        const maxRetries = 3;
        const retryDelay = 2000; // 2 seconds between retries
        let streamUrl = null;

        for (let attempt = 1; attempt <= maxRetries; attempt++) {
          logger.info(
            `Attempting to fetch stream URL for ${channelId} (attempt ${attempt}/${maxRetries})`,
            'stream'
          );

          streamUrl = await this.getStreamUrl(channelId);

          if (streamUrl) {
            logger.info(
              `Successfully retrieved stream URL for ${channelId} on attempt ${attempt}`,
              'stream'
            );
            break; // Success, exit retry loop
          }

          if (attempt < maxRetries) {
            logger.warn(
              `Stream URL not available yet for ${channelId}, retrying in ${retryDelay}ms...`,
              'stream'
            );
            await new Promise((resolve) => setTimeout(resolve, retryDelay));
          } else {
            logger.error(
              `Failed to get stream URL for ${channelId} after ${maxRetries} attempts. ` +
                `Stream may not be fully ready yet. Will retry on next status check.`,
              'stream'
            );
          }
        }

        if (streamUrl) {
          // Stream URL event already emitted by getStreamUrl
          logger.info(
            `Stream URL available for ${channelId}, capture should start`,
            'stream'
          );
        }
      } else if (!isLive && this.lastStreamStatus !== false) {
        // Stream just went offline
        logger.info(`Stream went offline for ${channelId}`, 'stream');
        this.lastStreamStatus = false;
        this.streamUrl = null; // Clear cached URL
        this.emit('streamOffline', { channelId: channelId });
      } else {
        logger.debug(
          `Stream status unchanged for ${channelId} (${
            isLive ? 'live' : 'offline'
          })`,
          'stream'
        );
      }
    } catch (error) {
      logger.error(
        `Error in _checkAndUpdateStream for ${channelId}: ${error.message}`,
        'stream'
      );
      logger.error(error.stack, 'stream');
      // Don't throw - we want monitoring to continue even if one check fails
    }
  }

  /**
   * Stop monitoring stream status
   */
  stopMonitoring() {
    if (!this.isMonitoring) {
      return;
    }

    this.isMonitoring = false;
    logger.info('Stopping stream monitoring', 'stream');

    if (this.monitorInterval) {
      clearInterval(this.monitorInterval);
      this.monitorInterval = null;
    }

    // Clear cached data
    this.streamUrl = null;
    this.broadcasterId = null;
    this.channelId = null;
    this.lastStreamStatus = null;
  }
}

module.exports = StreamManager;
