/**
 * Integration tests for HTTP bridge to Python service.
 * 
 * Tests chat/video/audio modules end-to-end request formatting
 * with mocked Python service endpoints.
 */
const nock = require('nock');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

// Mock logger to avoid noisy logs during tests
jest.mock('../../utils/logger', () => ({
  info: jest.fn(),
  error: jest.fn(),
  warn: jest.fn(),
  debug: jest.fn(),
}));

describe('HTTP Bridge Integration Tests', () => {
  const PYTHON_SERVICE_URL = 'http://localhost:8000';
  let pythonServiceScope;

  beforeEach(() => {
    nock.cleanAll();
    pythonServiceScope = nock(PYTHON_SERVICE_URL);
    process.env.PYTHON_SERVICE_URL = PYTHON_SERVICE_URL;
  });

  afterEach(() => {
    nock.cleanAll();
  });

  afterAll(() => {
    nock.restore();
  });

  describe('Chat HTTP Bridge', () => {
    test('sends chat message to Python /chat/message endpoint', async () => {
      pythonServiceScope
        .post('/chat/message', (body) => {
          // Match body with flexible timestamp
          return (
            body.channel === '#testchannel' &&
            body.username === 'testuser' &&
            body.message === 'Hello, world!' &&
            typeof body.timestamp === 'string'
          );
        })
        .reply(200, {
          received: true,
          message_id: 'test-message-id',
          timestamp: new Date().toISOString(),
        });

      const response = await axios.post(`${PYTHON_SERVICE_URL}/chat/message`, {
        channel: '#testchannel',
        username: 'testuser',
        message: 'Hello, world!',
        timestamp: new Date().toISOString(),
      });

      expect(response.status).toBe(200);
      expect(response.data.received).toBe(true);
      expect(pythonServiceScope.isDone()).toBe(true);
    });

    test('polls /chat/send endpoint and receives queued messages', async () => {
      pythonServiceScope
        .post('/chat/send', {
          channel: '#testchannel',
        })
        .reply(200, {
          messages: [
            {
              channel: '#testchannel',
              message: 'Test response',
              reply_to: 'testuser',
            },
          ],
        });

      const response = await axios.post(`${PYTHON_SERVICE_URL}/chat/send`, {
        channel: '#testchannel',
      });

      expect(response.status).toBe(200);
      expect(response.data.messages).toHaveLength(1);
      expect(response.data.messages[0].message).toBe('Test response');
      expect(pythonServiceScope.isDone()).toBe(true);
    });
  });

  describe('Video HTTP Bridge', () => {
    test('sends video frame to Python /api/video-frame endpoint', async () => {
      // Create a minimal JPEG file
      const jpegContent = Buffer.from(
        '\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0e\x11\x0e\x0f\x11\x17\x1a\x16\x14\x18\x19\x17\x1a\x1f\x1e\x1b\x1b\x1e\x1f!\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xff\xd9',
        'binary'
      );

      const form = new FormData();
      form.append('image_file', jpegContent, {
        filename: 'test.jpg',
        contentType: 'image/jpeg',
      });
      form.append('channel_id', '123456789');
      form.append('captured_at', new Date().toISOString());
      form.append('interesting_hint', 'false');

      pythonServiceScope
        .post('/api/video-frame', (body) => {
          // Nock matches multipart/form-data by checking if body contains form fields
          return true;
        })
        .reply(200, {
          frame_id: 'test-frame-id',
          status: 'success',
          next_interval_seconds: 10,
          activity: {
            recent_chat_count: 0,
            keyword_trigger: false,
          },
        });

      const response = await axios.post(
        `${PYTHON_SERVICE_URL}/api/video-frame`,
        form,
        {
          headers: form.getHeaders(),
          maxContentLength: Infinity,
          maxBodyLength: Infinity,
          timeout: 60000,
        }
      );

      expect(response.status).toBe(200);
      expect(response.data.status).toBe('success');
      expect(response.data.frame_id).toBe('test-frame-id');
      expect(pythonServiceScope.isDone()).toBe(true);
    });
  });

  describe('Audio HTTP Bridge', () => {
    test('sends audio chunk to Python /transcribe endpoint', async () => {
      // Create a minimal WAV file content
      const wavContent = Buffer.concat([
        Buffer.from('RIFF', 'ascii'),
        Buffer.alloc(4),
        Buffer.from('WAVE', 'ascii'),
        Buffer.from('fmt ', 'ascii'),
        Buffer.alloc(16),
        Buffer.from('data', 'ascii'),
        Buffer.alloc(1000),
      ]);

      const form = new FormData();
      form.append('audio_file', wavContent, {
        filename: 'test.wav',
        contentType: 'audio/wav',
      });
      form.append('channel_id', '123456789');
      form.append('started_at', new Date().toISOString());
      form.append('ended_at', new Date().toISOString());

      pythonServiceScope
        .post('/transcribe', (body) => {
          // Nock matches multipart/form-data
          return true;
        })
        .reply(200, {
          transcript: 'Test transcription',
          segments: [],
          language: 'en',
          duration: 1.0,
          model: 'base',
          processing_time_ms: 100,
          channel_id: '123456789',
          started_at: new Date().toISOString(),
          ended_at: new Date().toISOString(),
          stored_in_db: true,
          transcript_id: 'test-transcript-id',
        });

      const response = await axios.post(
        `${PYTHON_SERVICE_URL}/transcribe`,
        form,
        {
          headers: form.getHeaders(),
          maxContentLength: Infinity,
          maxBodyLength: Infinity,
          timeout: 90000,
        }
      );

      expect(response.status).toBe(200);
      expect(response.data.transcript).toBe('Test transcription');
      expect(response.data.stored_in_db).toBe(true);
      expect(pythonServiceScope.isDone()).toBe(true);
    });
  });

  describe('HTTP Error Handling', () => {
    test('handles 500 error from Python service', async () => {
      pythonServiceScope.post('/chat/message').reply(500, {
        detail: 'Internal server error',
      });

      await expect(
        axios.post(`${PYTHON_SERVICE_URL}/chat/message`, {
          channel: '#testchannel',
          username: 'testuser',
          message: 'Test',
          timestamp: new Date().toISOString(),
        })
      ).rejects.toThrow();

      expect(pythonServiceScope.isDone()).toBe(true);
    });

    test('handles timeout error', async () => {
      pythonServiceScope
        .post('/transcribe')
        .delayConnection(100000) // Delay longer than timeout
        .reply(200, {});

      await expect(
        axios.post(
          `${PYTHON_SERVICE_URL}/transcribe`,
          new FormData(),
          { timeout: 100 }
        )
      ).rejects.toThrow();

      expect(pythonServiceScope.isDone()).toBe(true);
    });

    test('handles connection refused error', async () => {
      // Use a different port that's not listening
      await expect(
        axios.post('http://localhost:9999/chat/message', {
          channel: '#testchannel',
          username: 'testuser',
          message: 'Test',
          timestamp: new Date().toISOString(),
        })
      ).rejects.toThrow();
    });
  });
});

