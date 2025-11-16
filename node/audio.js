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
     * - streamManager: StreamManager instance (required)
     * - chunkSeconds: number (default: 15)
     * - sampleRate: number (default: 16000)
     * - channels: number (default: 1 for mono)
     */
    this.config = config;
    this.channel = config.channel;
    this.pythonServiceUrl = config.pythonServiceUrl || 'http://localhost:8000';
    this.streamManager = config.streamManager; // Required

    if (!this.streamManager) {
      throw new Error('StreamManager is required for audio capture');
    }

    this.chunkSeconds = config.chunkSeconds || 15;
    this.sampleRate = config.sampleRate || 16000;
    this.audioChannels = config.channels || 1;

    this.isCapturing = false;
    this.ffmpegProcess = null;
    this.currentChunkStartTime = null;
    this.broadcasterId = null; // Will be set when capture starts

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
          'audio'
        );
      } else {
        // Try to get broadcaster ID ourselves if StreamManager didn't provide it
        logger.warn(
          `StreamManager didn't provide broadcaster ID for ${channelId}, attempting lookup...`,
          'audio'
        );
        this.broadcasterId = await this.streamManager.getBroadcasterId(
          channelId
        );
        if (this.broadcasterId) {
          logger.info(
            `Successfully obtained broadcaster ID: ${this.broadcasterId}`,
            'audio'
          );
        } else {
          logger.error(
            `Failed to get broadcaster ID for ${channelId}. Audio chunks will be stored with channel name.`,
            'audio'
          );
        }
      }

      // Start capture if not already capturing
      if (!this.isCapturing) {
        logger.info(
          `Stream URL available for ${channelId}, starting audio capture`,
          'audio'
        );
        await this._startCaptureWithUrl(streamUrl, channelId);
      }
    });

    // Listen for stream offline
    this.streamManager.on('streamOffline', (data) => {
      if (data.channelId === this.channel) {
        logger.info('Stream went offline, stopping audio capture', 'audio');
        this.stopCapture();
      }
    });

    // Listen for stream online (will trigger streamUrl event)
    this.streamManager.on('streamOnline', (data) => {
      if (data.channelId === this.channel) {
        logger.info(
          'Stream came online, will start capture when URL is available',
          'audio'
        );
      }
    });
  }

  /**
   * Initialize the audio capture service
   * Validates configuration and checks ffmpeg availability
   */
  initialize() {
    // Check if ffmpeg is available
    return new Promise((resolve, reject) => {
      ffmpeg.getAvailableEncoders((err, encoders) => {
        if (err) {
          logger.error(`FFmpeg not found: ${err.message}`, 'audio');
          logger.info(
            'Please install ffmpeg: https://ffmpeg.org/download.html',
            'audio'
          );
          reject(new Error('FFmpeg is required but not found in PATH'));
        } else {
          logger.info('FFmpeg found, audio capture ready', 'audio');
          resolve();
        }
      });
    });
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
      // Set generous timeout for transcription: 15s audio + 30s transcription + 5s embedding + 5s buffer = 55s
      // Round up to 90 seconds to be safe for slower processing
      const response = await axios.post(
        `${this.pythonServiceUrl}/transcribe`,
        form,
        {
          headers: form.getHeaders(),
          maxContentLength: Infinity,
          maxBodyLength: Infinity,
          timeout: 90000, // 90 seconds timeout for transcription processing
          // Keep connection alive for long-running requests
          httpAgent: new (require('http').Agent)({ keepAlive: true }),
          httpsAgent: new (require('https').Agent)({ keepAlive: true }),
        }
      );
      logger.info(
        `Audio chunk sent successfully: ${metadata.channel_id}`,
        'audio'
      );
    } catch (error) {
      // Log detailed error information for debugging
      if (
        error.code === 'ECONNABORTED' ||
        (error.message && error.message.includes('timeout'))
      ) {
        logger.error(
          `Timeout sending audio chunk to Python (request took >90s). This may indicate slow transcription processing.`,
          'audio'
        );
      } else if (
        error.code === 'ECONNRESET' ||
        (error.message && error.message.includes('socket hang up'))
      ) {
        logger.error(
          `Connection reset while sending audio chunk to Python. Python service may have closed the connection due to timeout or error. Error: ${error.message}`,
          'audio'
        );
      } else if (error.response) {
        // Server responded with error status
        logger.error(
          `Python service returned error ${error.response.status}: ${
            error.response.data?.detail || error.response.statusText
          }`,
          'audio'
        );
      } else {
        // Other network/connection errors
        logger.warn(
          `Failed to send audio chunk to Python: ${error.message} (code: ${
            error.code || 'unknown'
          })`,
          'audio'
        );
      }
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
          logger.info(`FFmpeg started: ${channelId}`, 'audio');
        })
        .on('end', () => {
          logger.info(`FFmpeg ended for ${channelId}`, 'audio');
          this.isCapturing = false;
          resolve();
        })
        .on('error', (err) => {
          if (
            err.message.includes('ECONNRESET') ||
            err.message.includes('ETIMEDOUT')
          ) {
            logger.warn(`Stream connection error: ${err.message}`, 'audio');
            this.isCapturing = false;
            resolve(); // Resolve to allow restart
          } else {
            logger.error(`FFmpeg error: ${err.message}`, 'audio');
            // Log more details for debugging
            if (err.stderr) {
              logger.error(`FFmpeg stderr: ${err.stderr}`, 'audio');
            }
            reject(err);
          }
        })
        .on('stderr', (stderrLine) => {
          // Log FFmpeg warnings/errors for monitoring
          if (stderrLine.includes('error') || stderrLine.includes('Error')) {
            logger.warn(`FFmpeg: ${stderrLine.trim()}`, 'audio');
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
                }`,
              'audio'
            );

            // Check FFmpeg process health
            if (this.ffmpegProcess) {
              // FFmpeg process exists - fluent-ffmpeg manages it internally
              // We infer it's healthy if chunks are being created
              logger.info(
                `[Diagnostics] FFmpeg process active (chunks being created)`,
                'audio'
              );
            } else {
              logger.warn(
                `[Diagnostics] FFmpeg process not available`,
                'audio'
              );
            }

            lastDiagnosticLog = now;
          }
        } catch (error) {
          // Directory might not exist yet or read failed
          logger.warn(
            `[Diagnostics] Directory scan error: ${error.message}`,
            'audio'
          );
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
              `Chunk file disappeared before processing: ${expectedChunk}`,
              'audio'
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
              }`,
              'audio'
            );
            processingLocks.delete(chunkKey);
            return;
          }
          if (finalSize === 0) {
            logger.warn(
              `Chunk file is empty (0 bytes), skipping: ${expectedChunk}`,
              'audio'
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
                `Chunk file is locked, will retry later: ${expectedChunk}`,
                'audio'
              );
            } else {
              logger.error(
                `Error reading chunk file: ${expectedChunk}, error: ${readError.message}`,
                'audio'
              );
            }
            // Don't mark as processed - allow retry on next poll
            processingLocks.delete(chunkKey);
            return;
          }

          // Verify buffer has content
          if (!chunkBuffer || chunkBuffer.length === 0) {
            logger.warn(
              `Chunk buffer is empty, skipping: ${expectedChunk}`,
              'audio'
            );
            processedChunks.add(chunkKey);
            chunkIndex = Math.max(chunkIndex, chunkToProcess + 1);
            processingLocks.delete(chunkKey);
            return;
          }

          logger.info(
            `Processing chunk ${chunkToProcess} (file: chunk_${channelId}_${String(
              chunkToProcess
            ).padStart(3, '0')}.wav) for ${channelId}: ${finalSize} bytes`,
            'audio'
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
            `Error processing chunk ${chunkToProcess}: ${error.message}`,
            'audio'
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
              `Will retry chunk ${chunkToProcess} on next poll (file access error)`,
              'audio'
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
            } chunks, last processed: chunk ${chunkToProcess || 'none'}`,
            'audio'
          );
        }
      }, 1000); // Check every second

      // Store interval reference for cleanup
      this.chunkPollInterval = chunkPollInterval;

      this.ffmpegProcess.run();
    });
  }

  /**
   * Start capturing audio from Twitch stream
   * @param {string} channelId - Channel name (without #)
   */
  async startCapture(channelId) {
    if (this.isCapturing) {
      logger.warn(`Audio capture already running for ${channelId}`, 'audio');
      return;
    }

    // Get broadcaster ID from StreamManager
    this.broadcasterId = await this.streamManager.getBroadcasterId(channelId);
    if (!this.broadcasterId) {
      logger.warn(
        `Failed to get broadcaster ID for ${channelId}, will try again when stream URL is available`,
        'audio'
      );
    } else {
      logger.info(
        `Resolved channel ${channelId} to broadcaster ID: ${this.broadcasterId}`,
        'audio'
      );
    }

    // Get stream URL from StreamManager
    const streamUrl = await this.streamManager.getStreamUrl(channelId);
    if (!streamUrl) {
      logger.info(
        `Stream URL not available yet, will start when stream comes online`,
        'audio'
      );
      // StreamManager will emit streamUrl event when stream becomes available
      return;
    }

    // Start capture with the stream URL
    await this._startCaptureWithUrl(streamUrl, channelId);
  }

  /**
   * Internal method to start capture with a specific stream URL
   * @param {string} streamUrl - Video stream URL (full video, will extract audio)
   * @param {string} channelId - Channel name
   */
  async _startCaptureWithUrl(streamUrl, channelId) {
    if (this.isCapturing) {
      logger.warn(`Audio capture already running`, 'audio');
      return;
    }

    // Get broadcaster ID if not already set
    if (!this.broadcasterId) {
      this.broadcasterId = await this.streamManager.getBroadcasterId(channelId);
    }

    this.isCapturing = true;
    this.currentChunkStartTime = new Date();
    logger.info(`Starting audio capture for ${channelId}`, 'audio');

    try {
      // Use full video stream URL and extract audio with -vn flag
      await this._chunkAudio(streamUrl, channelId);
    } catch (error) {
      logger.error(`Audio capture failed: ${error.message}`, 'audio');
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

    logger.info('Stopping audio capture...', 'audio');

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
        logger.warn(`Error stopping ffmpeg: ${error.message}`, 'audio');
      }
      this.ffmpegProcess = null;
    }

    this.isCapturing = false;
    logger.info('Audio capture stopped', 'audio');
  }
}

// Export the class
module.exports = AudioCapture;
