const ChatClient = require('../../chat');
const axios = require('axios');
const tmi = require('tmi.js');

jest.mock('axios');
jest.mock('tmi.js');

describe('ChatClient', () => {
  let chatClient;
  let mockTmiClient;
  let mockAxiosPost;

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();

    // Mock tmi.js client
    mockTmiClient = {
      on: jest.fn(),
      connect: jest.fn().mockResolvedValue(),
      disconnect: jest.fn().mockResolvedValue(),
      say: jest.fn().mockResolvedValue(),
    };
    tmi.Client = jest.fn().mockReturnValue(mockTmiClient);

    // Mock axios
    mockAxiosPost = jest.fn().mockResolvedValue({ data: { messages: [] } });
    axios.post = mockAxiosPost;

    chatClient = new ChatClient({
      botName: 'test_bot',
      oauthToken: 'oauth:test_token',
      channel: '#testchannel',
      pythonServiceUrl: 'http://localhost:8000',
    });
  });

  afterEach(() => {
    jest.useRealTimers();
    if (chatClient.pollInterval) {
      clearInterval(chatClient.pollInterval);
    }
  });

  describe('initialize', () => {
    test('creates tmi client with correct config', () => {
      chatClient.initialize();
      
      expect(tmi.Client).toHaveBeenCalledWith({
        options: { debug: false },
        identity: {
          username: 'test_bot',
          password: 'oauth:test_token',
        },
        channels: ['#testchannel'],
        connection: expect.objectContaining({
          reconnect: true,
          secure: true,
        }),
      });
    });

    test('sets up event handlers', () => {
      chatClient.initialize();
      
      expect(mockTmiClient.on).toHaveBeenCalledWith('connected', expect.any(Function));
      expect(mockTmiClient.on).toHaveBeenCalledWith('message', expect.any(Function));
      expect(mockTmiClient.on).toHaveBeenCalledWith('disconnected', expect.any(Function));
      expect(mockTmiClient.on).toHaveBeenCalledWith('reconnect', expect.any(Function));
    });
  });

  describe('_onMessage', () => {
    beforeEach(() => {
      chatClient.initialize();
    });

    test('ignores messages from self', () => {
      const forwardSpy = jest.spyOn(chatClient, '_forwardMessageToPython');
      
      chatClient._onMessage('#testchannel', { username: 'test_bot' }, 'Hello', true);
      
      expect(forwardSpy).not.toHaveBeenCalled();
    });

    test('ignores messages from bot name (case insensitive)', () => {
      const forwardSpy = jest.spyOn(chatClient, '_forwardMessageToPython');
      
      chatClient._onMessage('#testchannel', { 'display-name': 'Test_Bot' }, 'Hello', false);
      
      expect(forwardSpy).not.toHaveBeenCalled();
    });

    test('forwards valid messages to Python', () => {
      const forwardSpy = jest.spyOn(chatClient, '_forwardMessageToPython');
      
      chatClient._onMessage('#testchannel', { username: 'other_user' }, 'Hello world', false);
      
      expect(forwardSpy).toHaveBeenCalledWith('testchannel', 'other_user', 'Hello world');
    });

    test('uses display-name if available', () => {
      const forwardSpy = jest.spyOn(chatClient, '_forwardMessageToPython');
      
      chatClient._onMessage('#testchannel', { 'display-name': 'DisplayName', username: 'other_user' }, 'Hello', false);
      
      expect(forwardSpy).toHaveBeenCalledWith('testchannel', 'DisplayName', 'Hello');
    });
  });

  describe('_forwardMessageToPython', () => {
    test('sends POST request to Python service', async () => {
      mockAxiosPost.mockResolvedValue({});
      
      chatClient._forwardMessageToPython('testchannel', 'testuser', 'Hello');
      
      // Wait for promise (fire and forget, but we can check the call)
      // Advance timers and wait for async operations
      jest.advanceTimersByTime(100);
      await Promise.resolve();
      await Promise.resolve();
      
      expect(mockAxiosPost).toHaveBeenCalledWith(
        'http://localhost:8000/chat/message',
        expect.objectContaining({
          channel: 'testchannel',
          username: 'testuser',
          message: 'Hello',
          timestamp: expect.any(String),
        })
      );
    }, 20000);

    test('handles errors gracefully', async () => {
      const logger = require('../../utils/logger');
      const warnSpy = jest.spyOn(logger, 'warn').mockImplementation(() => {});
      mockAxiosPost.mockRejectedValue(new Error('Network error'));
      
      chatClient._forwardMessageToPython('testchannel', 'testuser', 'Hello');
      
      // Wait for promise chain to complete (axios.post().catch())
      jest.advanceTimersByTime(100);
      await Promise.resolve();
      await Promise.resolve();
      
      // Should not throw, just log warning
      expect(mockAxiosPost).toHaveBeenCalled();
      expect(warnSpy).toHaveBeenCalled();
      
      warnSpy.mockRestore();
    }, 20000);
  });

  describe('connect', () => {
    test('connects to Twitch IRC', async () => {
      chatClient.initialize();
      // Mock the connected event handler to set isConnected
      const connectedHandler = mockTmiClient.on.mock.calls.find(
        call => call[0] === 'connected'
      )[1];
      
      await chatClient.connect();
      
      expect(mockTmiClient.connect).toHaveBeenCalled();
      // Manually trigger connected handler to set isConnected
      connectedHandler('address', 6667);
      expect(chatClient.isConnected).toBe(true);
    });

    test('starts polling after connection', async () => {
      chatClient.initialize();
      const startPollingSpy = jest.spyOn(chatClient, 'startPolling');
      
      await chatClient.connect();
      
      expect(startPollingSpy).toHaveBeenCalled();
    });

    test('handles connection errors', async () => {
      chatClient.initialize();
      mockTmiClient.connect.mockRejectedValue(new Error('Connection failed'));
      
      await expect(chatClient.connect()).rejects.toThrow('Connection failed');
    });
  });

  describe('disconnect', () => {
    test('disconnects from Twitch IRC', async () => {
      chatClient.initialize();
      chatClient.isConnected = true;
      chatClient.pollInterval = setInterval(() => {}, 500);
      
      await chatClient.disconnect();
      
      expect(mockTmiClient.disconnect).toHaveBeenCalled();
      expect(chatClient.isConnected).toBe(false);
      expect(chatClient.pollInterval).toBeNull();
    });

    test('clears polling interval', async () => {
      chatClient.initialize();
      chatClient.pollInterval = setInterval(() => {}, 500);
      
      await chatClient.disconnect();
      
      expect(chatClient.pollInterval).toBeNull();
    });
  });

  describe('sendMessage', () => {
    test('sends message when connected', async () => {
      chatClient.initialize();
      chatClient.isConnected = true;
      
      await chatClient.sendMessage('testchannel', 'Hello world');
      
      expect(mockTmiClient.say).toHaveBeenCalledWith('testchannel', 'Hello world');
    });

    test('throws error when not connected', async () => {
      chatClient.isConnected = false;
      
      await expect(chatClient.sendMessage('testchannel', 'Hello')).rejects.toThrow('Not connected');
    });
  });

  describe('startPolling', () => {
    test('polls Python service at interval', async () => {
      chatClient.initialize();
      mockAxiosPost.mockResolvedValue({ data: { messages: [] } });
      
      chatClient.startPolling();
      
      // Fast-forward time
      jest.advanceTimersByTime(500);
      await Promise.resolve(); // Let async operations complete
      
      expect(mockAxiosPost).toHaveBeenCalledWith(
        'http://localhost:8000/chat/send',
        { channel: '#testchannel' }
      );
    });

    test('sends queued messages', async () => {
      chatClient.initialize();
      chatClient.isConnected = true;
      let callCount = 0;
      mockAxiosPost.mockImplementation(() => {
        callCount++;
        return Promise.resolve({
          data: {
            messages: callCount === 1 ? [
              { channel: 'testchannel', message: 'Hello', reply_to: null },
              { channel: 'testchannel', message: 'World', reply_to: 'user1' },
            ] : [],
          },
        });
      });
      
      chatClient.startPolling();
      
      jest.advanceTimersByTime(500);
      // Wait for async operations to complete - flush promises
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      
      // Check both messages were sent
      const sayCalls = mockTmiClient.say.mock.calls;
      expect(sayCalls.length).toBeGreaterThanOrEqual(1);
      // Find the calls - messages are sent sequentially
      const helloCall = sayCalls.find(call => call[1] === 'Hello');
      const worldCall = sayCalls.find(call => call[1] === '@user1 World');
      expect(helloCall).toBeTruthy();
      expect(worldCall).toBeTruthy();
    }, 20000);

    test('prevents overlapping poll executions', async () => {
      chatClient.initialize();
      let resolvePoll;
      const pollPromise = new Promise(resolve => { resolvePoll = resolve; });
      mockAxiosPost.mockReturnValue(pollPromise);
      
      chatClient.startPolling();
      
      // Trigger multiple polls quickly
      jest.advanceTimersByTime(500);
      jest.advanceTimersByTime(500);
      jest.advanceTimersByTime(500);
      
      resolvePoll({ data: { messages: [] } });
      await Promise.resolve();
      
      // Should only have been called once (others skipped due to isPolling lock)
      expect(mockAxiosPost).toHaveBeenCalledTimes(1);
    });
  });
});

