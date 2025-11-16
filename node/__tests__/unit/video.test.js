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

const VideoCapture = require('../../video');
// Silence expected FFmpeg error logs in tests
const logger = require('../../utils/logger');

describe('VideoCapture', () => {
  let videoCapture;
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
    mockAxiosPost = jest.fn().mockResolvedValue({
      data: {
        frame_id: 'frame-123',
        next_interval_seconds: 7,
        interesting_frame: true,
        activity: { recent_chat_count: 15, keyword_trigger: false },
      },
    });
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
    fs.createReadStream = jest.fn().mockReturnValue({
      pipe: jest.fn(),
    });
    fs.unlinkSync = jest.fn();

    videoCapture = new VideoCapture({
      channel: 'testchannel',
      pythonServiceUrl: 'http://localhost:8000',
      streamManager: mockStreamManager,
      baselineInterval: 10,
      activeInterval: 5,
      frameInterval: 10,
    });
  });

  afterEach(() => {
    jest.useRealTimers();
    if (videoCapture.frameCaptureInterval) {
      clearInterval(videoCapture.frameCaptureInterval);
    }
    if (videoCapture.heartbeatInterval) {
      clearInterval(videoCapture.heartbeatInterval);
    }
    if (errorSpy) {
      errorSpy.mockRestore();
    }
  });

  describe('constructor', () => {
    test('requires StreamManager', () => {
      expect(() => {
        new VideoCapture({
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
      mockFfmpeg.getAvailableEncoders.mockImplementation((callback) => {
        callback(null, {});
      });

      await expect(videoCapture.initialize()).resolves.not.toThrow();
      expect(mockFfmpeg.getAvailableEncoders).toHaveBeenCalled();
    });

    test('rejects when ffmpeg not found', async () => {
      mockFfmpeg.getAvailableEncoders.mockImplementation((callback) => {
        callback(new Error('FFmpeg not found'), null);
      });

      await expect(videoCapture.initialize()).rejects.toThrow('FFmpeg is required');
    });

    test('creates temp directory', async () => {
      fs.existsSync.mockReturnValue(false);
      mockFfmpeg.getAvailableEncoders.mockImplementation((callback) => callback(null, {}));

      await videoCapture.initialize();

      expect(fs.mkdirSync).toHaveBeenCalled();
    });
  });

  describe('_sendFrameToPython', () => {
    beforeEach(() => {
      videoCapture.broadcasterId = '123456';
    });

    test('sends frame with FormData', async () => {
      await videoCapture._sendFrameToPython('/tmp/frame.jpg', 'testchannel', '2024-01-01T00:00:00Z');

      expect(mockFormDataInstance).toBeTruthy();
      expect(mockFormDataInstance.append).toHaveBeenCalledWith('image_file', expect.anything(), expect.anything());
      expect(mockFormDataInstance.append).toHaveBeenCalledWith('channel_id', '123456');
      expect(mockFormDataInstance.append).toHaveBeenCalledWith('captured_at', '2024-01-01T00:00:00Z');
      expect(mockAxiosPost).toHaveBeenCalledWith(
        'http://localhost:8000/api/video-frame',
        mockFormDataInstance,
        expect.objectContaining({
          timeout: 60000,
        })
      );
    });

    test('uses broadcaster ID when available', async () => {
      videoCapture.broadcasterId = '123456';

      await videoCapture._sendFrameToPython('/tmp/frame.jpg', 'testchannel', '2024-01-01T00:00:00Z');

      expect(mockFormDataInstance).toBeTruthy();
      expect(mockFormDataInstance.append).toHaveBeenCalledWith('channel_id', '123456');
    });

    test('falls back to channel name when broadcaster ID unavailable', async () => {
      videoCapture.broadcasterId = null;
      mockStreamManager.getBroadcasterId.mockResolvedValue(null);

      await videoCapture._sendFrameToPython('/tmp/frame.jpg', 'testchannel', '2024-01-01T00:00:00Z');

      expect(mockFormDataInstance).toBeTruthy();
      expect(mockFormDataInstance.append).toHaveBeenCalledWith('channel_id', 'testchannel');
    });

    test('updates interval from Python response', async () => {
      videoCapture.isCapturing = true;
      videoCapture.currentStreamUrl = 'https://example.com/stream.m3u8';
      videoCapture.currentChannelId = 'testchannel';
      videoCapture.frameInterval = 10;

      const updateSpy = jest.spyOn(videoCapture, '_updateCaptureInterval');

      mockAxiosPost.mockResolvedValue({
        data: { next_interval_seconds: 7 },
      });

      await videoCapture._sendFrameToPython('/tmp/frame.jpg', 'testchannel', '2024-01-01T00:00:00Z');

      expect(updateSpy).toHaveBeenCalledWith(7);
    });

    test('sets interesting_frame flag', async () => {
      mockAxiosPost.mockResolvedValue({
        data: { interesting_frame: true },
      });

      await videoCapture._sendFrameToPython('/tmp/frame.jpg', 'testchannel', '2024-01-01T00:00:00Z');

      expect(videoCapture.lastInteresting).toBe(true);
    });
  });

  describe('_updateCaptureInterval', () => {
    test('updates interval when valid', () => {
      videoCapture.isCapturing = true;
      videoCapture.currentStreamUrl = 'https://example.com/stream.m3u8';
      videoCapture.currentChannelId = 'testchannel';
      videoCapture.frameInterval = 10;

      const createSpy = jest.spyOn(videoCapture, '_createFrameInterval');

      videoCapture._updateCaptureInterval(7);

      expect(videoCapture.frameInterval).toBe(7);
      expect(createSpy).toHaveBeenCalled();
    });

    test('does not update when not capturing', () => {
      videoCapture.isCapturing = false;
      videoCapture.frameInterval = 10;

      videoCapture._updateCaptureInterval(7);

      expect(videoCapture.frameInterval).toBe(10); // Unchanged
    });

    test('clamps interval to minimum 2 seconds', () => {
      videoCapture.isCapturing = true;
      videoCapture.currentStreamUrl = 'https://example.com/stream.m3u8';
      videoCapture.currentChannelId = 'testchannel';

      videoCapture._updateCaptureInterval(1);

      expect(videoCapture.frameInterval).toBe(2); // Clamped
    });

    test('does not update if interval unchanged', () => {
      videoCapture.isCapturing = true;
      videoCapture.currentStreamUrl = 'https://example.com/stream.m3u8';
      videoCapture.currentChannelId = 'testchannel';
      videoCapture.frameInterval = 7;

      const createSpy = jest.spyOn(videoCapture, '_createFrameInterval');

      videoCapture._updateCaptureInterval(7);

      expect(createSpy).not.toHaveBeenCalled(); // No change
    });
  });

  describe('stopCapture', () => {
    test('stops capturing and clears intervals', async () => {
      videoCapture.isCapturing = true;
      videoCapture.frameCaptureInterval = setInterval(() => {}, 1000);
      videoCapture.heartbeatInterval = setInterval(() => {}, 30000);

      const stopPromise = videoCapture.stopCapture();
      // Fast-forward past the 1 second wait
      jest.advanceTimersByTime(1100);
      await stopPromise;

      expect(videoCapture.isCapturing).toBe(false);
      expect(videoCapture.frameCaptureInterval).toBeNull();
      expect(videoCapture.heartbeatInterval).toBeNull();
    }, 10000);

    test('resets frame interval to baseline', async () => {
      videoCapture.isCapturing = true;
      videoCapture.frameInterval = 5;

      const stopPromise = videoCapture.stopCapture();
      jest.advanceTimersByTime(1100);
      await stopPromise;

      expect(videoCapture.frameInterval).toBe(10); // baselineInterval
    }, 10000);

    test('handles already stopped gracefully', async () => {
      videoCapture.isCapturing = false;

      await expect(videoCapture.stopCapture()).resolves.not.toThrow();
    });
  });

  describe('startCapture', () => {
    test('gets broadcaster ID and stream URL', async () => {
      mockFfmpeg.getAvailableEncoders.mockImplementation((callback) => callback(null, {}));
      await videoCapture.initialize();
      
      // Mock stream URL to be available
      mockStreamManager.getBroadcasterId.mockResolvedValue('123456');
      mockStreamManager.getStreamUrl.mockResolvedValue('https://example.com/stream.m3u8');

      // Mock _createFrameInterval to avoid setting up real intervals
      const createFrameIntervalSpy = jest.spyOn(videoCapture, '_createFrameInterval').mockImplementation(() => {});
      // Mock _startCaptureWithUrl to resolve immediately
      const startWithUrlSpy = jest.spyOn(videoCapture, '_startCaptureWithUrl').mockResolvedValue();

      const startPromise = videoCapture.startCapture('testchannel');
      // Advance timers to allow any internal timeouts to complete
      jest.advanceTimersByTime(100);
      await startPromise;

      expect(mockStreamManager.getBroadcasterId).toHaveBeenCalledWith('testchannel');
      expect(mockStreamManager.getStreamUrl).toHaveBeenCalledWith('testchannel');
      
      createFrameIntervalSpy.mockRestore();
      startWithUrlSpy.mockRestore();
    }, 20000);

    test('waits for stream URL if not available', async () => {
      mockStreamManager.getStreamUrl.mockResolvedValue(null);
      mockFfmpeg.getAvailableEncoders.mockImplementation((callback) => callback(null, {}));
      await videoCapture.initialize();

      await videoCapture.startCapture('testchannel');

      // Should not start capture yet
      expect(videoCapture.isCapturing).toBe(false);
    });
  });
});

