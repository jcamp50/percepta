const ffmpeg = require('fluent-ffmpeg');
const axios = require('axios');
const fs = require('fs');

jest.mock('fluent-ffmpeg');
jest.mock('axios');
jest.mock('fs');

// Mock form-data properly - it's a constructor
const mockFormDataInstance = {
  append: jest.fn(),
  getHeaders: jest.fn().mockReturnValue({ 'Content-Type': 'multipart/form-data' }),
};

jest.mock('form-data', () => {
  const MockFormData = function() {
    return mockFormDataInstance;
  };
  return MockFormData;
});

const AudioCapture = require('../../audio');
// Silence expected FFmpeg error logs in tests
const logger = require('../../utils/logger');

describe('AudioCapture', () => {
  let audioCapture;
  let mockStreamManager;
  let mockAxiosPost;
  let mockFfmpeg;
  let errorSpy;

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
    // Reset form-data mock instance methods
    mockFormDataInstance.append.mockClear();
    mockFormDataInstance.getHeaders.mockClear();

    // Silence logger.error to avoid noisy console for expected FFmpeg errors
    errorSpy = jest.spyOn(logger, 'error').mockImplementation(() => {});

    // Mock StreamManager
    mockStreamManager = {
      on: jest.fn(),
      emit: jest.fn(),
      getBroadcasterId: jest.fn().mockResolvedValue('123456'),
      getStreamUrl: jest.fn().mockResolvedValue('https://example.com/stream.m3u8'),
    };

    // Mock axios
    mockAxiosPost = jest.fn().mockResolvedValue({ data: {} });
    axios.post = mockAxiosPost;

    // Mock ffmpeg
    mockFfmpeg = {
      getAvailableEncoders: jest.fn((callback) => callback(null, {})),
      inputOptions: jest.fn().mockReturnThis(),
      outputOptions: jest.fn().mockReturnThis(),
      output: jest.fn().mockReturnThis(),
      on: jest.fn().mockReturnThis(),
      run: jest.fn(),
    };
    ffmpeg.getAvailableEncoders = mockFfmpeg.getAvailableEncoders;
    ffmpeg.mockReturnValue(mockFfmpeg);

    // Mock fs
    fs.existsSync = jest.fn().mockReturnValue(true);
    fs.mkdirSync = jest.fn();
    fs.readdirSync = jest.fn().mockReturnValue([]);
    fs.statSync = jest.fn().mockReturnValue({ size: 1024 });
    fs.readFileSync = jest.fn().mockReturnValue(Buffer.from('audio data'));
    fs.unlinkSync = jest.fn();

    audioCapture = new AudioCapture({
      channel: 'testchannel',
      pythonServiceUrl: 'http://localhost:8000',
      streamManager: mockStreamManager,
      chunkSeconds: 15,
      sampleRate: 16000,
      channels: 1,
    });
  });

  afterEach(() => {
    jest.useRealTimers();
    if (audioCapture.chunkPollInterval) {
      clearInterval(audioCapture.chunkPollInterval);
    }
    if (audioCapture.chunkCleanupInterval) {
      clearInterval(audioCapture.chunkCleanupInterval);
    }
    if (errorSpy) {
      errorSpy.mockRestore();
    }
  });

  describe('constructor', () => {
    test('requires StreamManager', () => {
      expect(() => {
        new AudioCapture({
          channel: 'test',
          pythonServiceUrl: 'http://localhost:8000',
        });
      }).toThrow('StreamManager is required');
    });

    test('sets up stream manager listeners', () => {
      expect(mockStreamManager.on).toHaveBeenCalledWith('streamUrl', expect.any(Function));
      expect(mockStreamManager.on).toHaveBeenCalledWith('streamOffline', expect.any(Function));
      expect(mockStreamManager.on).toHaveBeenCalledWith('streamOnline', expect.any(Function));
    });
  });

  describe('initialize', () => {
    let loggerErrorSpy;

    beforeEach(() => {
      const logger = require('../../utils/logger');
      loggerErrorSpy = jest.spyOn(logger, 'error').mockImplementation(() => {});
    });

    afterEach(() => {
      loggerErrorSpy && loggerErrorSpy.mockRestore();
    });

    test('validates ffmpeg availability', async () => {
      await expect(audioCapture.initialize()).resolves.not.toThrow();
      expect(mockFfmpeg.getAvailableEncoders).toHaveBeenCalled();
    });

    test('rejects when ffmpeg not found', async () => {
      mockFfmpeg.getAvailableEncoders.mockImplementation((callback) => {
        callback(new Error('FFmpeg not found'), null);
      });

      await expect(audioCapture.initialize()).rejects.toThrow('FFmpeg is required');
    });
  });

  describe('_sendChunkToPython', () => {
    test('sends chunk with FormData', async () => {
      const chunkBuffer = Buffer.from('audio chunk data');
      const metadata = {
        channel_id: '123456',
        started_at: '2024-01-01T00:00:00Z',
        ended_at: '2024-01-01T00:00:15Z',
      };

      await audioCapture._sendChunkToPython(chunkBuffer, metadata);

      expect(mockFormDataInstance.append).toHaveBeenCalledWith('audio_file', expect.anything(), expect.anything());
      expect(mockFormDataInstance.append).toHaveBeenCalledWith('channel_id', '123456');
      expect(mockFormDataInstance.append).toHaveBeenCalledWith('started_at', '2024-01-01T00:00:00Z');
      expect(mockFormDataInstance.append).toHaveBeenCalledWith('ended_at', '2024-01-01T00:00:15Z');
      expect(mockAxiosPost).toHaveBeenCalledWith(
        'http://localhost:8000/transcribe',
        mockFormDataInstance,
        expect.objectContaining({
          timeout: 90000,
        })
      );
    });

    test('handles timeout errors', async () => {
      // Mock logger.error - it's called from audio.js
      const logger = require('../../utils/logger');
      const errorSpy = jest.spyOn(logger, 'error').mockImplementation(() => {});

      mockAxiosPost.mockRejectedValue({
        code: 'ECONNABORTED',
        message: 'timeout',
      });

      await audioCapture._sendChunkToPython(Buffer.from('data'), {
        channel_id: '123',
        started_at: '2024-01-01T00:00:00Z',
        ended_at: '2024-01-01T00:00:15Z',
      });

      expect(errorSpy).toHaveBeenCalled();
      errorSpy.mockRestore();
    });

    test('handles connection reset errors', async () => {
      const logger = require('../../utils/logger');
      const errorSpy = jest.spyOn(logger, 'error').mockImplementation(() => {});

      mockAxiosPost.mockRejectedValue({
        code: 'ECONNRESET',
        message: 'socket hang up',
      });

      await audioCapture._sendChunkToPython(Buffer.from('data'), {
        channel_id: '123',
        started_at: '2024-01-01T00:00:00Z',
        ended_at: '2024-01-01T00:00:15Z',
      });

      expect(errorSpy).toHaveBeenCalled();
      errorSpy.mockRestore();
    });

    test('handles HTTP error responses', async () => {
      const logger = require('../../utils/logger');
      const errorSpy = jest.spyOn(logger, 'error').mockImplementation(() => {});

      mockAxiosPost.mockRejectedValue({
        response: {
          status: 500,
          data: { detail: 'Internal server error' },
        },
      });

      await audioCapture._sendChunkToPython(Buffer.from('data'), {
        channel_id: '123',
        started_at: '2024-01-01T00:00:00Z',
        ended_at: '2024-01-01T00:00:15Z',
      });

      expect(errorSpy).toHaveBeenCalled();
      errorSpy.mockRestore();
    });
  });

  describe('_chunkAudio', () => {
    test('processes chunks in order', async () => {
      audioCapture.isCapturing = true;
      const sendSpy = jest.spyOn(audioCapture, '_sendChunkToPython').mockResolvedValue();

      // Mock chunk files appearing
      const chunkFiles = [`chunk_testchannel_000.wav`];
      fs.readdirSync.mockReturnValue(chunkFiles);
      fs.statSync.mockReturnValue({ size: 1024 });
      fs.readFileSync.mockReturnValue(Buffer.from('chunk0 data'));

      // Start chunking (this starts ffmpeg and polling)
      const chunkPromise = audioCapture._chunkAudio('https://example.com/stream.m3u8', 'testchannel');

      // Force the polling interval to execute
      jest.runOnlyPendingTimers();
      await Promise.resolve();
      await Promise.resolve();

      // Also advance time to allow any internal waits to resolve
      jest.advanceTimersByTime(2000);
      await Promise.resolve();
      await Promise.resolve();

      expect(sendSpy).toHaveBeenCalled();

      // Cleanup
      audioCapture.isCapturing = false;
      if (audioCapture.chunkPollInterval) {
        clearInterval(audioCapture.chunkPollInterval);
      }
      if (audioCapture.chunkCleanupInterval) {
        clearInterval(audioCapture.chunkCleanupInterval);
      }
    }, 20000);

    test('skips empty chunks', async () => {
      audioCapture.isCapturing = true;
      const sendSpy = jest.spyOn(audioCapture, '_sendChunkToPython').mockResolvedValue();

      fs.readdirSync.mockReturnValue(['chunk_testchannel_000.wav']);
      fs.statSync.mockReturnValue({ size: 0 }); // Empty file

      const chunkPromise = audioCapture._chunkAudio('https://example.com/stream.m3u8', 'testchannel');

      jest.advanceTimersByTime(1000);
      await Promise.resolve();

      // Should not send empty chunks
      expect(sendSpy).not.toHaveBeenCalled();

      audioCapture.isCapturing = false;
      if (audioCapture.chunkPollInterval) {
        clearInterval(audioCapture.chunkPollInterval);
      }
    });

    test('handles file locking errors gracefully', async () => {
      audioCapture.isCapturing = true;
      const sendSpy = jest.spyOn(audioCapture, '_sendChunkToPython').mockResolvedValue();

      fs.readdirSync.mockReturnValue(['chunk_testchannel_000.wav']);
      fs.statSync.mockImplementation(() => {
        const error = new Error('EBUSY');
        error.code = 'EBUSY';
        throw error;
      });

      const chunkPromise = audioCapture._chunkAudio('https://example.com/stream.m3u8', 'testchannel');

      jest.advanceTimersByTime(1000);
      await Promise.resolve();

      // Should handle gracefully, not crash
      expect(sendSpy).not.toHaveBeenCalled();

      audioCapture.isCapturing = false;
      if (audioCapture.chunkPollInterval) {
        clearInterval(audioCapture.chunkPollInterval);
      }
    });
  });

  describe('stopCapture', () => {
    test('stops capturing and clears intervals', async () => {
      audioCapture.isCapturing = true;
      audioCapture.chunkPollInterval = setInterval(() => {}, 1000);
      audioCapture.chunkCleanupInterval = setInterval(() => {}, 30000);

      await audioCapture.stopCapture();

      expect(audioCapture.isCapturing).toBe(false);
      expect(audioCapture.chunkPollInterval).toBeNull();
      expect(audioCapture.chunkCleanupInterval).toBeNull();
    });

    test('kills ffmpeg process', async () => {
      audioCapture.isCapturing = true;
      const mockKill = jest.fn();
      audioCapture.ffmpegProcess = {
        kill: mockKill,
      };

      await audioCapture.stopCapture();

      expect(mockKill).toHaveBeenCalledWith('SIGTERM');
      expect(audioCapture.ffmpegProcess).toBeNull();
    });

    test('handles already stopped gracefully', async () => {
      audioCapture.isCapturing = false;

      await expect(audioCapture.stopCapture()).resolves.not.toThrow();
    });
  });

  describe('startCapture', () => {
    test('gets broadcaster ID and stream URL', async () => {
      // Mock stream URL to be available
      mockStreamManager.getBroadcasterId.mockResolvedValue('123456');
      mockStreamManager.getStreamUrl.mockResolvedValue('https://example.com/stream.m3u8');
      
      // Mock _chunkAudio to resolve immediately (it normally runs indefinitely)
      const chunkAudioSpy = jest.spyOn(audioCapture, '_chunkAudio').mockResolvedValue();
      
      const startPromise = audioCapture.startCapture('testchannel');
      // Advance timers to allow any internal timeouts to complete
      jest.advanceTimersByTime(100);
      await startPromise;

      expect(mockStreamManager.getBroadcasterId).toHaveBeenCalledWith('testchannel');
      expect(mockStreamManager.getStreamUrl).toHaveBeenCalledWith('testchannel');
      
      chunkAudioSpy.mockRestore();
    }, 20000);

    test('waits for stream URL if not available', async () => {
      mockStreamManager.getStreamUrl.mockResolvedValue(null);

      await audioCapture.startCapture('testchannel');

      // Should not start capture yet
      expect(audioCapture.isCapturing).toBe(false);
    });
  });
});

