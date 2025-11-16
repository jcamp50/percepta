/**
 * Twitch Stream Video Capture Service
 *
 * This module captures video frames from Twitch HLS streams, extracts
 * screenshots every 2 seconds, and forwards them to the Python backend
 * for embedding generation and storage.
 *
 * Why separate from audio.js?
 * - Separation of concerns (video vs audio)
 * - Different lifecycle management
 * - Can run independently or together
 */

const ffmpeg = require('fluent-ffmpeg');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const os = require('os');
const FormData = require('form-data');
const logger = require('./utils/logger');

/**
 * VideoCapture Class
 *
 * Manages video frame capture from Twitch HLS streams
 */
class VideoCapture {
  constructor(config) {
    /**
     * config should contain:
     * - channel: string (Twitch channel name)
     * - pythonServiceUrl: string (Python backend URL)
     * - streamManager: StreamManager instance (required)
     * - frameInterval: number (seconds between frames, default: 2)
     */
    this.config = config;
    this.channel = config.channel;
    this.pythonServiceUrl = config.pythonServiceUrl || 'http://localhost:8000';
    this.streamManager = config.streamManager; // Required
    
    if (!this.streamManager) {
      throw new Error('StreamManager is required for video capture');
    }
    
    this.baselineInterval = Math.max(2, Number(config.baselineInterval) || 10);
    this.activeInterval = Math.max(2, Number(config.activeInterval) || 5);
    this.frameInterval = Math.max(
      2,
      Number(config.frameInterval) || this.baselineInterval
    );
    if (this.frameInterval < this.activeInterval) {
      this.frameInterval = this.activeInterval;
    }

    this.isCapturing = false;
    this.isShuttingDown = false; // Flag to prevent new captures during shutdown
    this.ffmpegProcess = null;
    this.frameCaptureInterval = null;
    this.heartbeatInterval = null;
    this.broadcasterId = null; // Will be set when capture starts
    this.tmpDir = path.join(os.tmpdir(), 'percepta_video');
    this.framesCaptured = 0; // Counter for periodic summary logs
    this.framesFailed = 0; // Counter for failed frames
    this.currentStreamUrl = null;
    this.currentChannelId = null;
    this.lastInteresting = false;
    
    logger.info(
      `Video capture intervals configured (baseline=${this.baselineInterval}s, active=${this.activeInterval}s, start=${this.frameInterval}s)`,
      'video_description'
    );

    // Listen to StreamManager events
    this._setupStreamManagerListeners();
  }

  /**
   * Setup listeners for StreamManager events
   */
  _setupStreamManagerListeners() {
    // Listen for stream URL availability
    this.streamManager.on('streamUrl', async (data) => {
      const { streamUrl, broadcasterId, channelId } = data;
      
      // Only process if this is for our channel
      if (channelId !== this.channel) {
        return;
      }
      
      // Store broadcaster ID if provided
      if (broadcasterId) {
        this.broadcasterId = broadcasterId;
        logger.info(
          `Resolved channel ${channelId} to broadcaster ID: ${broadcasterId}`,
          'video'
        );
      } else {
        // Try to get broadcaster ID ourselves if StreamManager didn't provide it
        logger.warn(
          `StreamManager didn't provide broadcaster ID for ${channelId}, attempting lookup...`,
          'video'
        );
        this.broadcasterId = await this.streamManager.getBroadcasterId(channelId);
        if (this.broadcasterId) {
          logger.info(
            `Successfully obtained broadcaster ID: ${this.broadcasterId}`,
            'video'
          );
        } else {
          logger.error(
            `Failed to get broadcaster ID for ${channelId}. Video frames will be stored with channel name.`,
            'video'
          );
        }
      }
      
      // Start capture if not already capturing
      if (!this.isCapturing) {
        logger.info(`Stream URL available for ${channelId}, starting video capture`, 'video');
        await this._startCaptureWithUrl(streamUrl, channelId);
      }
    });
    
    // Listen for stream offline
    this.streamManager.on('streamOffline', (data) => {
      if (data.channelId === this.channel) {
        logger.info('Stream went offline, stopping video capture', 'video');
        this.stopCapture();
      }
    });
    
    // Listen for stream online (will trigger streamUrl event)
    this.streamManager.on('streamOnline', (data) => {
      if (data.channelId === this.channel) {
        logger.info('Stream came online, will start capture when URL is available', 'video');
      }
    });
  }

  /**
   * Initialize the video capture service
   * Validates configuration and checks ffmpeg availability
   */
  initialize() {
    // Ensure temp directory exists
    if (!fs.existsSync(this.tmpDir)) {
      fs.mkdirSync(this.tmpDir, { recursive: true });
    }

    // Check if ffmpeg is available
    return new Promise((resolve, reject) => {
      ffmpeg.getAvailableEncoders((err, encoders) => {
        if (err) {
          logger.error(`FFmpeg not found: ${err.message}`, 'video');
          logger.info(
            'Please install ffmpeg: https://ffmpeg.org/download.html',
            'video'
          );
          reject(new Error('FFmpeg is required but not found in PATH'));
        } else {
          logger.info('FFmpeg found, video capture ready', 'video');
          resolve();
        }
      });
    });
  }


  /**
   * Capture a single frame from video stream
   * @param {string} hlsUrl - Video stream HLS URL
   * @param {string} channelId - Channel name
   * @returns {Promise<void>}
   */
  async _captureFrame(hlsUrl, channelId) {
    return new Promise((resolve, reject) => {
      const frameId = Date.now();
      const outputPath = path.join(this.tmpDir, `frame_${channelId}_${frameId}.jpg`);

      // Use ffmpeg to extract a single frame
      const ffmpegCmd = ffmpeg(hlsUrl)
        .inputOptions([
          '-loglevel', 'error',
          '-reconnect', '1',
          '-reconnect_at_eof', '1',
          '-reconnect_streamed', '1',
          '-reconnect_delay_max', '2',
          '-ss', '0', // Start from beginning of stream buffer
        ])
        .outputOptions([
          '-vframes', '1', // Extract only 1 frame
          '-q:v', '2', // High quality JPEG
        ])
        .output(outputPath)
        .on('end', () => {
          // Use .then().catch() to ensure promise is properly handled
          this._sendFrameToPython(outputPath, channelId, new Date().toISOString())
            .then(() => {
            // Clean up temp file after sending
            if (fs.existsSync(outputPath)) {
                try {
              fs.unlinkSync(outputPath);
                } catch (unlinkError) {
                  logger.warn(`Failed to delete temp file: ${unlinkError.message}`, 'video');
                }
            }
            resolve();
            })
            .catch((error) => {
            logger.error(`Failed to send frame: ${error.message}`, 'video');
            logger.error(`Frame send error: ${error.message}`, 'video_description');
            // Clean up temp file on error
            if (fs.existsSync(outputPath)) {
                try {
              fs.unlinkSync(outputPath);
                } catch (unlinkError) {
                  logger.warn(`Failed to delete temp file: ${unlinkError.message}`, 'video');
                }
            }
            reject(error);
            });
        })
        .on('error', (err) => {
          // Don't log errors during shutdown as they're expected (SIGTERM, etc.)
          if (!this.isShuttingDown) {
            this.framesFailed++;
            // Log more details about FFmpeg errors
            const errorMsg = err.message || 'Unknown FFmpeg error';
            const exitCode = err.code || 'unknown';
            logger.error(`FFmpeg frame capture error (code ${exitCode}): ${errorMsg}`, 'video');
            
            // Log stderr if available for debugging
            if (err.stderr) {
              logger.debug(`FFmpeg stderr: ${err.stderr}`, 'video');
            }
            
            // Don't reject on transient errors - resolve instead to prevent unhandled rejection
            // FFmpeg code 69 (and other transient errors) might be network/stream issues
            // that will resolve on the next capture attempt
            logger.warn(`Skipping frame due to FFmpeg error, will retry on next interval (failed: ${this.framesFailed})`, 'video');
          }
          
          // Clean up temp file on error
          if (fs.existsSync(outputPath)) {
            try {
            fs.unlinkSync(outputPath);
            } catch (unlinkError) {
              // Ignore cleanup errors
            }
          }
          
          // Always resolve to prevent unhandled rejection, even during shutdown
          resolve();
        });

      ffmpegCmd.run();
    });
  }

  _createFrameInterval(streamUrl, channelId) {
    if (this.frameCaptureInterval) {
      clearInterval(this.frameCaptureInterval);
      this.frameCaptureInterval = null;
    }
    this.currentStreamUrl = null;
    this.currentChannelId = null;

    const intervalMs = Math.max(2, this.frameInterval) * 1000;
    this.frameCaptureInterval = setInterval(() => {
      if (!this.isCapturing || this.isShuttingDown) {
        return;
      }

      this._captureFrame(streamUrl, channelId)
        .then(() => {})
        .catch((error) => {
          if (!this.isShuttingDown) {
            this.framesFailed++;
            logger.error(`Frame capture error: ${error.message}`, 'video');
            if (error.stack) {
              logger.debug(`Frame capture error stack: ${error.stack}`, 'video');
            }
          }
        });
    }, intervalMs);
  }

  _updateCaptureInterval(nextIntervalSeconds) {
    if (!this.isCapturing || !this.currentStreamUrl || !this.currentChannelId) {
      return;
    }

    const parsed = Number(nextIntervalSeconds);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      return;
    }

    const sanitized = Math.max(2, Math.round(parsed));
    if (sanitized === this.frameInterval) {
      return;
    }

    this.frameInterval = sanitized;
    this._createFrameInterval(this.currentStreamUrl, this.currentChannelId);
    logger.info(
      `Adjusted video capture interval to ${sanitized}s for ${this.currentChannelId}`,
      'video'
    );
    logger.info(
      `Video capture interval now ${sanitized}s (channel=${this.currentChannelId})`,
      'video_description'
    );
  }

  /**
   * Send video frame to Python backend
   * @param {string} framePath - Path to frame image file
   * @param {string} channelId - Channel name
   * @param {string} capturedAt - ISO timestamp when frame was captured
   */
  async _sendFrameToPython(framePath, channelId, capturedAt) {
    const FormData = require('form-data');
    const form = new FormData();

    // Get broadcaster ID from StreamManager if not cached
    if (!this.broadcasterId) {
      this.broadcasterId = await this.streamManager.getBroadcasterId(channelId);
    }
    
    // Use broadcaster ID if available, otherwise use channel name
    const broadcasterId = this.broadcasterId || channelId;

    // Create readable stream from file
    const frameStream = fs.createReadStream(framePath);

    form.append('image_file', frameStream, {
      filename: `frame_${channelId}_${Date.now()}.jpg`,
      contentType: 'image/jpeg',
    });
    form.append('channel_id', broadcasterId);
    form.append('captured_at', capturedAt);
      const interestingHint = this.frameInterval <= this.activeInterval; 
    form.append('interesting_hint', interestingHint ? 'true' : 'false');

    try {
      const response = await axios.post(
        `${this.pythonServiceUrl}/api/video-frame`,
        form,
        {
          headers: form.getHeaders(),
          maxContentLength: Infinity,
          maxBodyLength: Infinity,
          timeout: 60000, // Increased to 60 seconds for CLIP embedding generation
          httpAgent: new (require('http').Agent)({ 
            keepAlive: true,
            keepAliveMsecs: 60000, // Keep connections alive for 60 seconds
            timeout: 60000,
          }),
          httpsAgent: new (require('https').Agent)({ 
            keepAlive: true,
            keepAliveMsecs: 60000,
            timeout: 60000,
          }),
        }
      );

      this.framesCaptured++;
      const data = response.data || {};
      if (typeof data.next_interval_seconds === 'number') {
        logger.info(
          `Python recommended capture interval ${data.next_interval_seconds}s (current ${this.frameInterval}s)`,
          'video_description'
        );
        this._updateCaptureInterval(data.next_interval_seconds);
      }
      if (typeof data.interesting_frame === 'boolean') {
        this.lastInteresting = data.interesting_frame;
      } else {
        this.lastInteresting = false;
      }

      const recentChat = data.activity?.recent_chat_count ?? 'n/a';
      const keywordTrigger = data.activity?.keyword_trigger ? 'yes' : 'no';
      logger.info(
        `Frame ${data.frame_id || 'unknown'} stored (source=${data.description_source || 'n/a'}, reused=${data.reused_description ? 'yes' : 'no'}, interesting=${data.interesting_frame ? 'yes' : 'no'}, chat=${recentChat}, keyword=${keywordTrigger})`,
        'video_description'
      );

      // Log every 10th frame to avoid spam, or always log if frames are failing
      if (this.framesCaptured % 10 === 0 || this.framesFailed > 0) {
        logger.info(
          `Video frame sent successfully: ${data.frame_id || 'no frame_id'} (${channelId}, total: ${this.framesCaptured}, interval: ${this.frameInterval}s)`,
          'video'
        );
      }
    } catch (error) {
      // Log but don't crash - Python service might be down or busy
      logger.warn(
        `Failed to send video frame to Python: ${error.message}`,
        'video'
      );
      throw error;
    }
  }

  /**
   * Start capturing frames from video stream
   * @param {string} channelId - Channel name (without #)
   * @returns {Promise<void>}
   */
  async startCapture(channelId) {
    if (this.isCapturing) {
      logger.warn('Video capture already in progress', 'video');
      return;
    }

    // Get broadcaster ID from StreamManager
    this.broadcasterId = await this.streamManager.getBroadcasterId(channelId);
    if (!this.broadcasterId) {
      logger.warn(
        `Failed to get broadcaster ID for ${channelId}, will try again when stream URL is available`,
        'video'
      );
    } else {
      logger.info(
        `Resolved channel ${channelId} to broadcaster ID: ${this.broadcasterId}`,
        'video'
      );
    }

    // Get stream URL from StreamManager
    const streamUrl = await this.streamManager.getStreamUrl(channelId);
    if (!streamUrl) {
      logger.info(`Stream URL not available yet, will start when stream comes online`, 'video');
      // StreamManager will emit streamUrl event when stream becomes available
      return;
    }

    // Start capture with the stream URL
    await this._startCaptureWithUrl(streamUrl, channelId);
  }

  /**
   * Internal method to start capture with a specific stream URL
   * @param {string} streamUrl - Video stream URL
   * @param {string} channelId - Channel name
   */
  async _startCaptureWithUrl(streamUrl, channelId) {
    if (this.isCapturing) {
      logger.debug('Video capture already in progress, skipping duplicate start', 'video');
      return;
    }

    // Get broadcaster ID if not already set
    if (!this.broadcasterId) {
      this.broadcasterId = await this.streamManager.getBroadcasterId(channelId);
    }

    this.isCapturing = true;
    this.framesCaptured = 0; // Reset counters
    this.framesFailed = 0;
    this.currentStreamUrl = streamUrl;
    this.currentChannelId = channelId;
    this.frameInterval = this.baselineInterval;
    logger.info(`Starting video frame capture for channel ${channelId}`, 'video');

    try {
      this._createFrameInterval(streamUrl, channelId);

      // Add heartbeat logger every 30 seconds to verify service is alive
      this.heartbeatInterval = setInterval(() => {
        if (this.isCapturing) {
          const successRate = this.framesCaptured + this.framesFailed > 0 
            ? ((this.framesCaptured / (this.framesCaptured + this.framesFailed)) * 100).toFixed(1)
            : 0;
          logger.info(
            `Video capture heartbeat: ${channelId} (active, current interval ${this.frameInterval}s, ` +
            `captured: ${this.framesCaptured}, failed: ${this.framesFailed}, success rate: ${successRate}%)`,
            'video'
          );
        }
      }, 30000);

      // Capture first frame immediately
      await this._captureFrame(streamUrl, channelId);
    } catch (error) {
      logger.error(`Video capture failed: ${error.message}`, 'video');
      this.isCapturing = false;
      throw error;
    }
  }

  /**
   * Stop capturing frames
   * @returns {Promise<void>}
   */
  async stopCapture() {
    if (!this.isCapturing) {
      return;
    }

    this.isShuttingDown = true; // Set flag to prevent new captures
    this.isCapturing = false;
    logger.info('Stopping video frame capture', 'video');

    // Clear frame capture interval
    if (this.frameCaptureInterval) {
      clearInterval(this.frameCaptureInterval);
      this.frameCaptureInterval = null;
    }
    this.currentStreamUrl = null;
    this.currentChannelId = null;

    // Clear heartbeat interval
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    // Stop FFmpeg process if running
    if (this.ffmpegProcess) {
      try {
      this.ffmpegProcess.kill('SIGTERM');
      } catch (error) {
        // Ignore errors during shutdown
      }
      this.ffmpegProcess = null;
    }

    // Wait a moment for any in-flight captures to complete
    await new Promise((resolve) => setTimeout(resolve, 1000));
    this.frameInterval = this.baselineInterval;
    this.isShuttingDown = false;
  }
}

module.exports = VideoCapture;

