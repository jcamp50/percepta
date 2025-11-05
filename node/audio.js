/**
 * Twitch Stream Audio Capture Service
 *
 * This module captures audio from Twitch HLS streams, segments it into
 * configurable chunks (default 15 seconds), and forwards them to the Python
 * backend for transcription.
 *
 * Why separate from chat.js?
 * - Separation of concerns (audio vs chat)
 * - Different lifecycle management
 * - Can run independently or together
 */

const ffmpeg = require('fluent-ffmpeg');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const { Readable } = require('stream');
const logger = require('./utils/logger');

/**
 * AudioCapture Class
 *
 * Manages audio capture from Twitch HLS streams
 */
class AudioCapture {
  constructor(config) {
    /**
     * config should contain:
     * - channel: string (Twitch channel name)
     * - pythonServiceUrl: string (Python backend URL)
     * - twitchClientId: string (Twitch API client ID)
     * - twitchOAuthToken: string (Twitch OAuth token for API calls)
     * - chunkSeconds: number (default: 15)
     * - sampleRate: number (default: 16000)
     * - channels: number (default: 1 for mono)
     */
    this.config = config;
    this.channel = config.channel;
    this.pythonServiceUrl = config.pythonServiceUrl || 'http://localhost:8000';
    this.twitchClientId = config.twitchClientId;
    this.twitchOAuthToken = config.twitchOAuthToken;
    this.chunkSeconds = config.chunkSeconds || 15;
    this.sampleRate = config.sampleRate || 16000;
    this.audioChannels = config.channels || 1;

    this.isCapturing = false;
    this.ffmpegProcess = null;
    this.streamMonitorInterval = null;
    this.currentChunkStartTime = null;
    this.broadcasterId = null; // Will be set when capture starts
  }

  /**
   * Initialize the audio capture service
   * Validates configuration and checks ffmpeg availability
   */
  initialize() {
    if (!this.twitchClientId) {
      throw new Error('TWITCH_CLIENT_ID is required for audio capture');
    }

    // Check if ffmpeg is available
    return new Promise((resolve, reject) => {
      ffmpeg.getAvailableEncoders((err, encoders) => {
        if (err) {
          logger.error(`FFmpeg not found: ${err.message}`);
          logger.info(
            'Please install ffmpeg: https://ffmpeg.org/download.html'
          );
          reject(new Error('FFmpeg is required but not found in PATH'));
        } else {
          logger.info('FFmpeg found, audio capture ready');
          resolve();
        }
      });
    });
  }

  /**
   * Check if Twitch channel is currently live
   * @param {string} channelId - Channel name (without #)
   * @returns {Promise<boolean>} True if stream is live
   */
  async _checkStreamStatus(channelId) {
    try {
      // Extract token (remove 'oauth:' prefix if present, Twitch API expects just the token)
      const token = this.twitchOAuthToken?.replace(/^oauth:/, '') || '';

      if (!token) {
        logger.warn('No OAuth token provided for Twitch API calls');
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

      const streams = response.data.data || [];
      return streams.length > 0;
    } catch (error) {
      if (error.response?.status === 404) {
        return false; // Channel doesn't exist
      }
      if (error.response?.status === 401) {
        logger.error(
          'Twitch API authentication failed. Check TWITCH_BOT_TOKEN and TWITCH_CLIENT_ID.'
        );
        return false;
      }
      logger.warn(`Failed to check stream status: ${error.message}`);
      return false;
    }
  }

  /**
   * Get authenticated audio-only HLS stream URL via Python/Streamlink service
   *
   * Uses Streamlink Python API to get authenticated Twitch audio-only stream URLs.
   * Streamlink handles all Twitch authentication (OAuth, client-integrity tokens) internally.
   *
   * @param {string} channelId - Channel name (without #)
   * @returns {Promise<string|null>} HLS URL or null if not available
   */
  async _getHlsUrl(channelId) {
    // First check if stream is live (still useful for early detection)
    const isLive = await this._checkStreamStatus(channelId);
    if (!isLive) {
      logger.info(`Channel ${channelId} is not currently live`);
      return null;
    }

    try {
      // Call Python service endpoint that uses Streamlink to get authenticated URL
      const response = await axios.get(
        `${this.pythonServiceUrl}/api/get-audio-stream-url`,
        {
          params: {
            channel_id: channelId,
          },
          timeout: 10000, // 10 second timeout
        }
      );

      const { stream_url, available } = response.data;

      if (!available || !stream_url) {
        logger.warn(
          `Audio-only stream not available for channel ${channelId} via Streamlink`
        );
        return null;
      }

      logger.info(
        `Retrieved authenticated audio-only stream URL for channel ${channelId} via Streamlink`
      );
      return stream_url;
    } catch (error) {
      logger.error(`Failed to get HLS URL via Streamlink: ${error.message}`);

      if (error.response) {
        // HTTP error response
        if (error.response.status === 503) {
          logger.error(
            'Python Streamlink service unavailable. Ensure streamlink is installed: pip install streamlink'
          );
        } else if (error.response.status === 502) {
          logger.error(
            `Streamlink plugin error for channel ${channelId}. Channel may be offline or restricted.`
          );
        } else if (error.response.status === 500) {
          logger.error(
            'Python service error getting stream URL. Check Python service logs.'
          );
        }
      } else if (error.code === 'ECONNREFUSED') {
        logger.error(
          `Cannot connect to Python service at ${this.pythonServiceUrl}. Is it running?`
        );
      } else if (error.code === 'ETIMEDOUT') {
        logger.error(
          'Timeout waiting for Python service response. Streamlink may be taking too long.'
        );
      }

      return null;
    }
  }

  /**
   * Start monitoring stream status and auto-start capture when live
   * @param {string} channelId - Channel name to monitor
   */
  _monitorStream(channelId) {
    logger.info(`Starting stream monitor for ${channelId}`);

    // Check every 30 seconds if stream is live
    this.streamMonitorInterval = setInterval(async () => {
      if (this.isCapturing) {
        // Already capturing, just verify still live
        const isLive = await this._checkStreamStatus(channelId);
        if (!isLive) {
          logger.info('Stream went offline, stopping capture');
          await this.stopCapture();
        }
      } else {
        // Not capturing, check if stream went live
        const isLive = await this._checkStreamStatus(channelId);
        if (isLive) {
          logger.info('Stream is now live, starting capture');
          await this.startCapture(channelId);
        }
      }
    }, 30000); // Check every 30 seconds
  }

  /**
   * Send audio chunk to Python backend
   * @param {Buffer} chunkBuffer - Audio data buffer
   * @param {Object} metadata - Chunk metadata (channel, timestamps)
   */
  async _sendChunkToPython(chunkBuffer, metadata) {
    const FormData = require('form-data');
    const form = new FormData();

    // Create a readable stream from the buffer
    const audioStream = new Readable();
    audioStream.push(chunkBuffer);
    audioStream.push(null);

    form.append('audio_file', audioStream, {
      filename: `chunk_${metadata.channel_id}_${metadata.started_at}.wav`,
      contentType: 'audio/wav',
    });
    form.append('channel_id', metadata.channel_id);
    form.append('started_at', metadata.started_at);
    form.append('ended_at', metadata.ended_at);

    try {
      const response = await axios.post(
        `${this.pythonServiceUrl}/transcribe`,
        form,
        {
          headers: form.getHeaders(),
          maxContentLength: Infinity,
          maxBodyLength: Infinity,
        }
      );
      logger.info(`Audio chunk sent successfully: ${metadata.channel_id}`);
    } catch (error) {
      // Log but don't crash - Python service might be down
      logger.warn(`Failed to send audio chunk to Python: ${error.message}`);
    }
  }

  /**
   * Process audio stream into chunks
   * Uses ffmpeg segment muxer to create separate files for each duration
   */
  _chunkAudio(hlsUrl, channelId) {
    return new Promise((resolve, reject) => {
      const os = require('os');
      const tmpDir = path.join(os.tmpdir(), 'percepta_audio');

      // Ensure temp directory exists
      if (!fs.existsSync(tmpDir)) {
        fs.mkdirSync(tmpDir, { recursive: true });
      }

      const chunkPattern = path.join(tmpDir, `chunk_${channelId}_%03d.wav`);
      let chunkIndex = 0;
      const chunkStart = new Date();
      const processedChunks = new Set(); // Track processed chunks to avoid duplicates
      const processingLocks = new Set(); // Track chunks currently being processed (prevents race conditions)
      const filesToDelete = new Map(); // Track files pending deletion with timestamps (delay deletion)

      // Periodic cleanup of old chunk files (runs every 30 seconds)
      const cleanupInterval = setInterval(() => {
        const now = Date.now();
        const maxAge = 60000; // Delete files older than 60 seconds (safe buffer for FFmpeg)

        for (const [filePath, timestamp] of filesToDelete.entries()) {
          if (now - timestamp > maxAge) {
            try {
              if (fs.existsSync(filePath)) {
                fs.unlinkSync(filePath);
                filesToDelete.delete(filePath);
                // File cleaned up successfully (silent)
              } else {
                filesToDelete.delete(filePath);
              }
            } catch (deleteError) {
              if (deleteError.code === 'EBUSY') {
                // Still locked, will retry on next cleanup cycle
                // Don't remove from map yet
              } else {
                // Other error, give up on this file
                filesToDelete.delete(filePath);
                // Cleanup gave up on file (non-critical, silent)
              }
            }
          }
        }
      }, 30000); // Run cleanup every 30 seconds

      // Store cleanup interval reference
      this.chunkCleanupInterval = cleanupInterval;

      // Use ffmpeg segment muxer to automatically create chunks
      // For live streams, segment muxer will create files as it processes
      this.ffmpegProcess = ffmpeg(hlsUrl)
        .inputOptions([
          '-loglevel',
          'error',
          '-reconnect',
          '1',
          '-reconnect_at_eof',
          '1',
          '-reconnect_streamed',
          '1',
          '-reconnect_delay_max',
          '2',
        ])
        .outputOptions([
          '-vn', // No video
          '-acodec',
          'pcm_s16le', // PCM 16-bit little-endian
          '-ar',
          String(this.sampleRate), // Sample rate
          '-ac',
          String(this.audioChannels), // Mono
          '-f',
          'segment', // Use segment muxer
          '-segment_time',
          String(this.chunkSeconds), // Chunk duration in seconds
          '-segment_format',
          'wav', // Output format
          '-reset_timestamps',
          '1', // Reset timestamps for each segment
          '-strftime',
          '0', // Don't use strftime in filename pattern
          '-write_empty_segments',
          '0', // Don't write empty segments (prevents 0-byte files)
          '-segment_list_flags',
          'cache', // Cache segment list for better performance
        ])
        .output(chunkPattern)
        .on('start', (commandLine) => {
          logger.info(`FFmpeg started: ${channelId}`);
        })
        .on('end', () => {
          logger.info(`FFmpeg ended for ${channelId}`);
          this.isCapturing = false;
          resolve();
        })
        .on('error', (err) => {
          if (
            err.message.includes('ECONNRESET') ||
            err.message.includes('ETIMEDOUT')
          ) {
            logger.warn(`Stream connection error: ${err.message}`);
            this.isCapturing = false;
            resolve(); // Resolve to allow restart
          } else {
            logger.error(`FFmpeg error: ${err.message}`);
            // Log more details for debugging
            if (err.stderr) {
              logger.error(`FFmpeg stderr: ${err.stderr}`);
            }
            reject(err);
          }
        })
        .on('stderr', (stderrLine) => {
          // Log FFmpeg warnings/errors for monitoring
          if (stderrLine.includes('error') || stderrLine.includes('Error')) {
            logger.warn(`FFmpeg: ${stderrLine.trim()}`);
          }
        })
        .on('progress', async (progress) => {
          // Check for new chunks periodically
          // Note: We use a polling approach since segment muxer doesn't emit file-created events
        });

      // Poll for new chunk files with directory scanning (handles gaps)
      let lastDiagnosticLog = 0;
      const diagnosticInterval = 30000; // Log diagnostics every 30 seconds

      const chunkPollInterval = setInterval(async () => {
        // Scan directory for all available chunk files
        // This approach handles gaps and non-sequential chunk creation
        let chunkToProcess = null;
        let foundChunk = null;
        let lowestChunkNumber = Infinity;
        let allChunkFiles = [];
        let skippedFiles = [];

        try {
          // Read all files in temp directory
          const files = fs.readdirSync(tmpDir);

          // Filter for chunk files matching our pattern: chunk_{channelId}_{number}.wav
          const chunkFilePattern = new RegExp(
            `^chunk_${channelId}_(\\d+)\\.wav$`
          );

          // Collect all valid chunk files first, then sort by number
          const validChunks = [];

          for (const file of files) {
            const match = file.match(chunkFilePattern);
            if (!match) {
              continue; // Not a chunk file
            }

            const chunkNumber = parseInt(match[1], 10);
            const chunkKey = `chunk_${channelId}_${String(chunkNumber).padStart(
              3,
              '0'
            )}`;

            // Track all chunk files found
            allChunkFiles.push({
              number: chunkNumber,
              key: chunkKey,
              processed: processedChunks.has(chunkKey),
              locked: processingLocks.has(chunkKey),
            });

            // Skip if already processed
            if (processedChunks.has(chunkKey)) {
              skippedFiles.push({ number: chunkNumber, reason: 'processed' });
              continue;
            }

            // Skip if currently being processed
            if (processingLocks.has(chunkKey)) {
              skippedFiles.push({ number: chunkNumber, reason: 'locked' });
              continue;
            }

            // Check if file exists and collect valid chunks
            const chunkPath = path.join(tmpDir, file);
            if (fs.existsSync(chunkPath)) {
              try {
                const fileSize = fs.statSync(chunkPath).size;
                if (fileSize > 0) {
                  validChunks.push({
                    number: chunkNumber,
                    path: chunkPath,
                    key: chunkKey,
                    size: fileSize,
                  });
                } else {
                  skippedFiles.push({ number: chunkNumber, reason: 'empty' });
                }
              } catch (statError) {
                skippedFiles.push({
                  number: chunkNumber,
                  reason: `stat_error:${statError.code}`,
                });
              }
            }
          }

          // Process chunks in order (000, 001, 002...) to match FFmpeg creation order
          // This reduces interference with FFmpeg writes
          if (validChunks.length > 0) {
            validChunks.sort((a, b) => a.number - b.number); // Sort by chunk number
            const nextChunk = validChunks[0]; // Get the lowest-numbered unprocessed chunk
            chunkToProcess = nextChunk.number;
            foundChunk = nextChunk.path;
          }

          // Diagnostic logging every 30 seconds
          const now = Date.now();
          if (now - lastDiagnosticLog > diagnosticInterval) {
            const processedList = Array.from(processedChunks)
              .map((k) => {
                const match = k.match(/\d+/);
                return match ? parseInt(match[0]) : null;
              })
              .filter((n) => n !== null)
              .sort((a, b) => a - b);

            const lockedList = Array.from(processingLocks)
              .map((k) => {
                const match = k.match(/\d+/);
                return match ? parseInt(match[0]) : null;
              })
              .filter((n) => n !== null)
              .sort((a, b) => a - b);

            logger.info(
              `[Diagnostics] Total chunks found: ${allChunkFiles.length}, ` +
                `valid unprocessed: ${validChunks.length}, ` +
                `processed: ${processedChunks.size}, ` +
                `locked: ${processingLocks.size}, ` +
                `highest processed: ${
                  processedList.length > 0 ? Math.max(...processedList) : 'none'
                }, ` +
                `next to process: ${
                  chunkToProcess !== null ? chunkToProcess : 'none'
                }`
            );

            // Check FFmpeg process health
            if (this.ffmpegProcess) {
              // FFmpeg process exists - fluent-ffmpeg manages it internally
              // We infer it's healthy if chunks are being created
              logger.info(
                `[Diagnostics] FFmpeg process active (chunks being created)`
              );
            } else {
              logger.warn(`[Diagnostics] FFmpeg process not available`);
            }

            lastDiagnosticLog = now;
          }
        } catch (error) {
          // Directory might not exist yet or read failed
          logger.warn(`[Diagnostics] Directory scan error: ${error.message}`);
          return;
        }

        // No chunk found to process
        if (!foundChunk || chunkToProcess === null) {
          // Silently continue - chunks will be detected when FFmpeg creates them
          return;
        }

        const expectedChunk = foundChunk;
        const chunkKey = `chunk_${channelId}_${String(chunkToProcess).padStart(
          3,
          '0'
        )}`;

        // Acquire processing lock
        processingLocks.add(chunkKey);

        const chunkStartTime = new Date(
          chunkStart.getTime() + chunkToProcess * this.chunkSeconds * 1000
        );
        const chunkEndTime = new Date(
          chunkStartTime.getTime() + this.chunkSeconds * 1000
        );

        try {
          // Wait for file to be fully written and stable
          // Check file size multiple times to ensure it's complete
          // Also handle Windows file locking by retrying statSync
          let previousSize = -1;
          let stableCount = 0;
          const maxWaitTime = 5000; // Increased to 5 seconds for slow writes
          const startWait = Date.now();
          let fileAccessible = false;

          while (Date.now() - startWait < maxWaitTime) {
            if (!fs.existsSync(expectedChunk)) {
              // File was deleted, skip
              processingLocks.delete(chunkKey);
              return;
            }

            // Try to get file size with retry for Windows file locking
            let currentSize = -1;
            let retryCount = 0;
            const maxRetries = 3;

            while (retryCount < maxRetries && currentSize === -1) {
              try {
                currentSize = fs.statSync(expectedChunk).size;
                fileAccessible = true;
              } catch (statError) {
                if (statError.code === 'EBUSY' || statError.code === 'ENOENT') {
                  // File is locked or doesn't exist - wait and retry
                  retryCount++;
                  if (retryCount < maxRetries) {
                    await new Promise((resolve) =>
                      setTimeout(resolve, 100 * retryCount)
                    );
                  } else {
                    logger.warn(
                      `Cannot access chunk file (locked?): ${expectedChunk}, error: ${statError.code}`
                    );
                    processingLocks.delete(chunkKey);
                    return;
                  }
                } else {
                  // Other error
                  logger.warn(
                    `Error accessing chunk file: ${expectedChunk}, error: ${statError.message}`
                  );
                  processingLocks.delete(chunkKey);
                  return;
                }
              }
            }

            // Check if file size is stable (same size for 3 checks = 300ms)
            if (currentSize === previousSize && currentSize > 0) {
              stableCount++;
              if (stableCount >= 3) {
                // File is stable and has content
                break;
              }
            } else {
              stableCount = 0;
            }

            previousSize = currentSize;
            await new Promise((resolve) => setTimeout(resolve, 100));
          }

          // Final check: ensure file exists and has content
          if (!fs.existsSync(expectedChunk)) {
            logger.warn(
              `Chunk file disappeared before processing: ${expectedChunk}`
            );
            processingLocks.delete(chunkKey);
            return;
          }

          // Final size check with error handling
          let finalSize = 0;
          try {
            finalSize = fs.statSync(expectedChunk).size;
          } catch (statError) {
            logger.warn(
              `Cannot read chunk file size: ${expectedChunk}, error: ${
                statError.code || statError.message
              }`
            );
            processingLocks.delete(chunkKey);
            return;
          }
          if (finalSize === 0) {
            logger.warn(
              `Chunk file is empty (0 bytes), skipping: ${expectedChunk}`
            );
            // Mark as processed (even though skipped) to avoid reprocessing
            processedChunks.add(chunkKey);
            // Schedule empty file for delayed deletion
            filesToDelete.set(expectedChunk, Date.now());
            // Update chunkIndex - if we're processing a gap chunk, update accordingly
            chunkIndex = Math.max(chunkIndex, chunkToProcess + 1);
            processingLocks.delete(chunkKey);
            return;
          }

          // Read the file with error handling for Windows file locking
          let chunkBuffer = null;
          try {
            chunkBuffer = fs.readFileSync(expectedChunk);
          } catch (readError) {
            if (readError.code === 'EBUSY') {
              logger.warn(
                `Chunk file is locked, will retry later: ${expectedChunk}`
              );
            } else {
              logger.error(
                `Error reading chunk file: ${expectedChunk}, error: ${readError.message}`
              );
            }
            // Don't mark as processed - allow retry on next poll
            processingLocks.delete(chunkKey);
            return;
          }

          // Verify buffer has content
          if (!chunkBuffer || chunkBuffer.length === 0) {
            logger.warn(`Chunk buffer is empty, skipping: ${expectedChunk}`);
            processedChunks.add(chunkKey);
            chunkIndex = Math.max(chunkIndex, chunkToProcess + 1);
            processingLocks.delete(chunkKey);
            return;
          }

          logger.info(
            `Processing chunk ${chunkToProcess} (file: chunk_${channelId}_${String(
              chunkToProcess
            ).padStart(3, '0')}.wav) for ${channelId}: ${finalSize} bytes`
          );

          // Mark as processed BEFORE sending (prevents duplicate processing)
          processedChunks.add(chunkKey);

          // Send to Python service
          // Use broadcaster ID instead of channel name for consistency
          await this._sendChunkToPython(chunkBuffer, {
            channel_id: this.broadcasterId || channelId, // Fallback to channel name if broadcaster ID not available
            started_at: chunkStartTime.toISOString(),
            ended_at: chunkEndTime.toISOString(),
          });

          // Schedule file for delayed deletion (don't delete immediately)
          // This prevents interfering with FFmpeg writes and Windows file locking
          // Files will be deleted by periodic cleanup after a safe delay
          filesToDelete.set(expectedChunk, Date.now());

          // Update chunkIndex to continue from the processed chunk
          // If we processed a gap chunk (chunkToProcess > chunkIndex), update to that + 1
          // Otherwise, just increment normally
          chunkIndex = Math.max(chunkIndex, chunkToProcess + 1);

          // Release processing lock
          processingLocks.delete(chunkKey);
        } catch (error) {
          logger.error(
            `Error processing chunk ${chunkToProcess}: ${error.message}`
          );

          // Don't mark as processed if it's a file access error (allow retry)
          // Only mark as processed for actual processing errors (send failure, etc.)
          const isFileAccessError =
            error.code === 'EBUSY' ||
            error.code === 'ENOENT' ||
            error.message.includes('locked') ||
            error.message.includes('access');

          if (!isFileAccessError) {
            // Mark as processed for non-file-access errors to avoid infinite retries
            processedChunks.add(chunkKey);
            chunkIndex = Math.max(chunkIndex, chunkToProcess + 1);
          } else {
            logger.info(
              `Will retry chunk ${chunkToProcess} on next poll (file access error)`
            );
          }

          // Schedule file for delayed deletion (use delayed cleanup instead of immediate)
          filesToDelete.set(expectedChunk, Date.now());

          // Release lock
          processingLocks.delete(chunkKey);
        }

        // Clean up intervals if not capturing anymore
        if (!this.isCapturing) {
          clearInterval(chunkPollInterval);
          clearInterval(cleanupInterval);
        }

        // Periodic logging to show we're still polling (every 10 chunks)
        if (chunkIndex % 10 === 0 && chunkIndex > 0) {
          logger.info(
            `Chunk polling active: processed ${
              processedChunks.size
            } chunks, last processed: chunk ${chunkToProcess || 'none'}`
          );
        }
      }, 1000); // Check every second

      // Store interval reference for cleanup
      this.chunkPollInterval = chunkPollInterval;

      this.ffmpegProcess.run();
    });
  }

  /**
   * Get broadcaster ID from channel name
   * Retries with exponential backoff to handle Python service startup race condition
   * @param {string} channelName - Channel name (without #)
   * @param {number} maxRetries - Maximum number of retry attempts (default: 5)
   * @returns {Promise<string|null>} Broadcaster ID or null if not found
   */
  async _getBroadcasterId(channelName, maxRetries = 5) {
    const baseDelay = 1000; // Start with 1 second delay
    const maxDelay = 10000; // Cap at 10 seconds

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const response = await axios.get(
          `${this.pythonServiceUrl}/api/get-broadcaster-id`,
          {
            params: {
              channel_name: channelName,
            },
            timeout: 5000,
          }
        );
        if (attempt > 0) {
          logger.info(
            `Successfully got broadcaster ID for ${channelName} after ${attempt} retry(ies)`
          );
        }
        return response.data.broadcaster_id || null;
      } catch (error) {
        // Check if this is a connection error (service not ready) vs API error (channel not found)
        const isConnectionError =
          error.code === 'ECONNREFUSED' ||
          error.code === 'ETIMEDOUT' ||
          error.message.includes('Network Error') ||
          (error.response && error.response.status >= 500);

        // If it's not a connection error, don't retry (e.g., 404 = channel not found)
        if (!isConnectionError) {
          logger.error(
            `Failed to get broadcaster ID for ${channelName}: ${error.message}`
          );
          return null;
        }

        // If we've exhausted retries, give up
        if (attempt >= maxRetries) {
          logger.error(
            `Failed to get broadcaster ID for ${channelName} after ${maxRetries} retries: ${error.message}`
          );
          return null;
        }

        // Calculate exponential backoff delay with jitter
        const delay = Math.min(
          baseDelay * Math.pow(2, attempt) + Math.random() * 1000,
          maxDelay
        );

        logger.warn(
          `Python service not ready yet, retrying broadcaster ID lookup for ${channelName} in ${Math.round(
            delay
          )}ms (attempt ${attempt + 1}/${maxRetries})...`
        );

        // Wait before retrying
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }

    return null;
  }

  /**
   * Start capturing audio from Twitch stream
   * @param {string} channelId - Channel name (without #)
   */
  async startCapture(channelId) {
    if (this.isCapturing) {
      logger.warn(`Audio capture already running for ${channelId}`);
      return;
    }

    // Get broadcaster ID for this channel
    this.broadcasterId = await this._getBroadcasterId(channelId);
    if (!this.broadcasterId) {
      logger.error(
        `Failed to get broadcaster ID for ${channelId}, cannot start capture`
      );
      return;
    }
    logger.info(
      `Resolved channel ${channelId} to broadcaster ID: ${this.broadcasterId}`
    );

    // Get HLS URL
    const hlsUrl = await this._getHlsUrl(channelId);
    if (!hlsUrl) {
      logger.info(`Cannot start capture: stream is not live`);
      // Start monitoring to auto-start when live
      this._monitorStream(channelId);
      return;
    }

    this.isCapturing = true;
    this.currentChunkStartTime = new Date();
    logger.info(`Starting audio capture for ${channelId}`);

    try {
      await this._chunkAudio(hlsUrl, channelId);
    } catch (error) {
      logger.error(`Audio capture failed: ${error.message}`);
      this.isCapturing = false;
      throw error;
    }
  }

  /**
   * Stop audio capture
   */
  async stopCapture() {
    if (!this.isCapturing) {
      return;
    }

    logger.info('Stopping audio capture...');

    // Stop stream monitoring
    if (this.streamMonitorInterval) {
      clearInterval(this.streamMonitorInterval);
      this.streamMonitorInterval = null;
    }

    // Stop chunk polling
    if (this.chunkPollInterval) {
      clearInterval(this.chunkPollInterval);
      this.chunkPollInterval = null;
    }

    // Stop cleanup interval
    if (this.chunkCleanupInterval) {
      clearInterval(this.chunkCleanupInterval);
      this.chunkCleanupInterval = null;
    }

    // Kill ffmpeg process
    if (this.ffmpegProcess) {
      try {
        this.ffmpegProcess.kill('SIGTERM');
      } catch (error) {
        logger.warn(`Error stopping ffmpeg: ${error.message}`);
      }
      this.ffmpegProcess = null;
    }

    this.isCapturing = false;
    logger.info('Audio capture stopped');
  }
}

// Export the class
module.exports = AudioCapture;
