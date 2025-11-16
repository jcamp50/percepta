const StreamManager = require('../../stream');
const axios = require('axios');
const EventEmitter = require('events');

jest.mock('axios');

describe('StreamManager', () => {
  let streamManager;
  let mockAxiosGet;

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();

    mockAxiosGet = jest.fn();
    axios.get = mockAxiosGet;

    streamManager = new StreamManager({
      pythonServiceUrl: 'http://localhost:8000',
      twitchClientId: 'test_client_id',
      twitchOAuthToken: 'oauth:test_token',
    });
  });

  afterEach(() => {
    jest.useRealTimers();
    if (streamManager.monitorInterval) {
      clearInterval(streamManager.monitorInterval);
    }
  });

  describe('initialize', () => {
    test('validates configuration', () => {
      const manager = new StreamManager({
        pythonServiceUrl: 'http://localhost:8000',
        twitchClientId: null,
        twitchOAuthToken: 'token',
      });
      expect(() => {
        manager.initialize();
      }).toThrow('TWITCH_CLIENT_ID is required');
    });

    test('initializes successfully with valid config', () => {
      expect(() => {
        streamManager.initialize();
      }).not.toThrow();
    });
  });

  describe('_checkStreamStatus', () => {
    test('returns true when stream is live', async () => {
      mockAxiosGet.mockResolvedValue({
        data: { data: [{ id: '123' }] },
      });

      const isLive = await streamManager._checkStreamStatus('testchannel');

      expect(isLive).toBe(true);
      expect(mockAxiosGet).toHaveBeenCalledWith(
        'https://api.twitch.tv/helix/streams?user_login=testchannel',
        expect.objectContaining({
          headers: expect.objectContaining({
            'Client-ID': 'test_client_id',
            Authorization: expect.stringContaining('Bearer'),
          }),
        })
      );
    });

    test('returns false when stream is offline', async () => {
      mockAxiosGet.mockResolvedValue({
        data: { data: [] },
      });

      const isLive = await streamManager._checkStreamStatus('testchannel');

      expect(isLive).toBe(false);
    });

    test('handles 404 (channel not found)', async () => {
      mockAxiosGet.mockRejectedValue({
        response: { status: 404 },
      });

      const isLive = await streamManager._checkStreamStatus('nonexistent');

      expect(isLive).toBe(false);
    });

    test('handles 401 (auth error)', async () => {
      const logger = require('../../utils/logger');
      const errorSpy = jest.spyOn(logger, 'error').mockImplementation(() => {});
      mockAxiosGet.mockRejectedValue({
        response: { status: 401 },
      });

      const isLive = await streamManager._checkStreamStatus('testchannel');

      expect(isLive).toBe(false);
      expect(errorSpy).toHaveBeenCalled();
      
      errorSpy.mockRestore();
    });

    test('strips oauth: prefix from token', async () => {
      mockAxiosGet.mockResolvedValue({ data: { data: [] } });

      await streamManager._checkStreamStatus('testchannel');

      const authHeader = mockAxiosGet.mock.calls[0][1].headers.Authorization;
      expect(authHeader).not.toContain('oauth:');
      expect(authHeader).toContain('Bearer');
    });
  });

  describe('getBroadcasterId', () => {
    test('fetches broadcaster ID from Python service', async () => {
      mockAxiosGet.mockResolvedValue({
        data: { broadcaster_id: '123456' },
      });

      const id = await streamManager.getBroadcasterId('testchannel');

      expect(id).toBe('123456');
      expect(mockAxiosGet).toHaveBeenCalledWith(
        'http://localhost:8000/api/get-broadcaster-id',
        expect.objectContaining({
          params: { channel_name: 'testchannel' },
        })
      );
    });

    test('caches broadcaster ID', async () => {
      mockAxiosGet.mockResolvedValue({
        data: { broadcaster_id: '123456' },
      });

      const id1 = await streamManager.getBroadcasterId('testchannel');
      const id2 = await streamManager.getBroadcasterId('testchannel');

      expect(id1).toBe('123456');
      expect(id2).toBe('123456');
      expect(mockAxiosGet).toHaveBeenCalledTimes(1); // Cached
    });

    test('returns null on error', async () => {
      mockAxiosGet.mockRejectedValue(new Error('Network error'));

      const id = await streamManager.getBroadcasterId('testchannel');

      expect(id).toBeNull();
    });

    test('handles 503 (service unavailable)', async () => {
      const logger = require('../../utils/logger');
      const errorSpy = jest.spyOn(logger, 'error').mockImplementation(() => {});
      mockAxiosGet.mockRejectedValue({
        response: { status: 503, data: { detail: 'Service unavailable' } },
      });

      const id = await streamManager.getBroadcasterId('testchannel');

      expect(id).toBeNull();
      expect(errorSpy).toHaveBeenCalled();
      
      errorSpy.mockRestore();
    });
  });

  describe('_fetchStreamUrl', () => {
    test('fetches stream URL from Python service', async () => {
      mockAxiosGet.mockResolvedValue({
        data: {
          available: true,
          stream_url: 'https://example.com/stream.m3u8',
        },
      });

      const url = await streamManager._fetchStreamUrl('testchannel');

      expect(url).toBe('https://example.com/stream.m3u8');
      expect(mockAxiosGet).toHaveBeenCalledWith(
        'http://localhost:8000/api/get-video-stream-url',
        expect.objectContaining({
          params: { channel_id: 'testchannel' },
        })
      );
    });

    test('returns null when stream not available', async () => {
      mockAxiosGet.mockResolvedValue({
        data: {
          available: false,
          stream_url: null,
        },
      });

      const url = await streamManager._fetchStreamUrl('testchannel');

      expect(url).toBeNull();
    });

    test('handles errors gracefully', async () => {
      mockAxiosGet.mockRejectedValue(new Error('Network error'));

      const url = await streamManager._fetchStreamUrl('testchannel');

      expect(url).toBeNull();
    });
  });

  describe('getStreamUrl', () => {
    test('caches stream URL', async () => {
      streamManager.initialize();
      // Reset state
      streamManager.streamUrl = null;
      streamManager.broadcasterId = null;
      streamManager.channelId = null;
      // Reset mock call count
      mockAxiosGet.mockClear();
      // Order: first _fetchStreamUrl (stream URL), then getBroadcasterId (broadcaster ID)
      mockAxiosGet
        .mockResolvedValueOnce({
          data: {
            available: true,
            stream_url: 'https://example.com/stream.m3u8',
          },
        })
        .mockResolvedValueOnce({
          data: { broadcaster_id: '123456' },
        });

      const url1 = await streamManager.getStreamUrl('testchannel');
      // Second call should use cache, so no new axios calls
      const url2 = await streamManager.getStreamUrl('testchannel');

      expect(url1).toBe('https://example.com/stream.m3u8');
      expect(url2).toBe('https://example.com/stream.m3u8');
      // Should call for stream URL once, broadcaster ID once (total 2 calls)
      expect(mockAxiosGet).toHaveBeenCalledTimes(2);
    }, 15000);

    test('emits streamUrl event', async () => {
      streamManager.initialize();
      const eventSpy = jest.fn();
      streamManager.on('streamUrl', eventSpy);

      // Reset to ensure clean state
      streamManager.streamUrl = null;
      streamManager.broadcasterId = null;
      streamManager.channelId = null;
      mockAxiosGet.mockClear();

      // Order: first _fetchStreamUrl (stream URL), then getBroadcasterId (broadcaster ID)
      mockAxiosGet
        .mockResolvedValueOnce({
          data: {
            available: true,
            stream_url: 'https://example.com/stream.m3u8',
          },
        })
        .mockResolvedValueOnce({
          data: { broadcaster_id: '123456' },
        });

      await streamManager.getStreamUrl('testchannel');

      expect(eventSpy).toHaveBeenCalledWith({
        streamUrl: 'https://example.com/stream.m3u8',
        broadcasterId: '123456',
        channelId: 'testchannel',
      });
    }, 15000);
  });

  describe('startMonitoring', () => {
    test('starts monitoring interval', () => {
      streamManager.initialize();
      mockAxiosGet.mockResolvedValue({ data: { data: [] } });

      streamManager.startMonitoring('testchannel');

      expect(streamManager.isMonitoring).toBe(true);
      expect(streamManager.monitorInterval).toBeTruthy();
    });

    test('emits streamOnline when stream comes online', async () => {
      streamManager.initialize();
      const eventSpy = jest.fn();
      streamManager.on('streamOnline', eventSpy);

      // Reset state
      streamManager.lastStreamStatus = null;

      // First check: offline
      mockAxiosGet.mockResolvedValueOnce({ data: { data: [] } });
      // Second check: online
      mockAxiosGet.mockResolvedValueOnce({ data: { data: [{ id: '123' }] } });
      // Stream URL fetch (called after delay)
      mockAxiosGet.mockResolvedValueOnce({
        data: { available: true, stream_url: 'https://example.com/stream.m3u8' },
      });
      // Broadcaster ID fetch
      mockAxiosGet.mockResolvedValueOnce({
        data: { broadcaster_id: '123456' },
      });

      streamManager.startMonitoring('testchannel');

      // Fast-forward past initial check and delay
      jest.advanceTimersByTime(30000 + 3000); // 30s check + 3s delay
      // Wait for async operations
      await Promise.resolve();
      await Promise.resolve();

      expect(eventSpy).toHaveBeenCalledWith({ channelId: 'testchannel' });
    }, 20000);

    test('emits streamOffline when stream goes offline', async () => {
      streamManager.initialize();
      const eventSpy = jest.fn();
      streamManager.on('streamOffline', eventSpy);
      streamManager.lastStreamStatus = true; // Was online

      // Reset mock
      mockAxiosGet.mockReset();
      // Check: offline (status change)
      mockAxiosGet.mockResolvedValue({ data: { data: [] } });

      streamManager.startMonitoring('testchannel');

      // Fast-forward to trigger check
      jest.advanceTimersByTime(30000);
      // Wait for async operations
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();

      expect(eventSpy).toHaveBeenCalledWith({ channelId: 'testchannel' });
    }, 20000);
  });

  describe('stopMonitoring', () => {
    test('stops monitoring and clears interval', () => {
      streamManager.initialize();
      streamManager.startMonitoring('testchannel');
      streamManager.stopMonitoring();

      expect(streamManager.isMonitoring).toBe(false);
      expect(streamManager.monitorInterval).toBeNull();
    });

    test('clears cached data', () => {
      streamManager.initialize();
      streamManager.startMonitoring('testchannel');
      // Set cached values directly
      streamManager.streamUrl = 'https://example.com/stream.m3u8';
      streamManager.broadcasterId = '123456';
      streamManager.channelId = 'testchannel';

      streamManager.stopMonitoring();

      // stopMonitoring should clear these
      expect(streamManager.streamUrl).toBeNull();
      expect(streamManager.broadcasterId).toBeNull();
      expect(streamManager.channelId).toBeNull();
    });
  });
});

