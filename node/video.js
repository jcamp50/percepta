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
    
    this.frameInterval = config.frameInterval || 2; // Default: 2 seconds

    this.isCapturing = false;
    this.ffmpegProcess = null;
    this.frameCaptureInterval = null;
    this.broadcasterId = null; // Will be set when capture starts
    this.tmpDir = path.join(os.tmpdir(), 'percepta_video');
    
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
        .on('end', async () => {
          try {
            // Send frame to Python service
            await this._sendFrameToPython(outputPath, channelId, new Date().toISOString());
            // Clean up temp file after sending
            if (fs.existsSync(outputPath)) {
              fs.unlinkSync(outputPath);
            }
            resolve();
          } catch (error) {
            logger.error(`Failed to send frame: ${error.message}`, 'video');
            // Clean up temp file on error
            if (fs.existsSync(outputPath)) {
              fs.unlinkSync(outputPath);
            }
            reject(error);
          }
        })
        .on('error', (err) => {
          logger.error(`FFmpeg frame capture error: ${err.message}`, 'video');
          // Clean up temp file on error
          if (fs.existsSync(outputPath)) {
            fs.unlinkSync(outputPath);
          }
          reject(err);
        });

      ffmpegCmd.run();
    });
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

    try {
      const response = await axios.post(
        `${this.pythonServiceUrl}/api/video-frame`,
        form,
        {
          headers: form.getHeaders(),
          maxContentLength: Infinity,
          maxBodyLength: Infinity,
          timeout: 30000, // 30 second timeout for CLIP embedding generation
        }
      );

      logger.debug(
        `Video frame sent successfully: ${response.data.frame_id}`,
        'video'
      );
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
      logger.warn('Video capture already in progress', 'video');
      return;
    }

    // Get broadcaster ID if not already set
    if (!this.broadcasterId) {
      this.broadcasterId = await this.streamManager.getBroadcasterId(channelId);
    }

    this.isCapturing = true;
    logger.info(`Starting video frame capture for channel ${channelId}`, 'video');

    try {
      // Capture frames at regular intervals
      this.frameCaptureInterval = setInterval(async () => {
        if (!this.isCapturing) {
          return;
        }

        try {
          await this._captureFrame(streamUrl, channelId);
        } catch (error) {
          logger.error(`Frame capture error: ${error.message}`, 'video');
          // Continue capturing even if one frame fails
        }
      }, this.frameInterval * 1000); // Convert seconds to milliseconds

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

    this.isCapturing = false;
    logger.info('Stopping video frame capture', 'video');

    // Clear frame capture interval
    if (this.frameCaptureInterval) {
      clearInterval(this.frameCaptureInterval);
      this.frameCaptureInterval = null;
    }

    // Stop FFmpeg process if running
    if (this.ffmpegProcess) {
      this.ffmpegProcess.kill('SIGTERM');
      this.ffmpegProcess = null;
    }
  }
}

module.exports = VideoCapture;

