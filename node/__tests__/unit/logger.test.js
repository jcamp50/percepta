// Mock dotenv to prevent loading from .env file in tests
jest.mock('dotenv', () => ({
  config: jest.fn(),
}));

// Set env vars before requiring logger (logger reads them at module load time)
delete require.cache[require.resolve('../../utils/logger')];

describe('Logger', () => {
  let logger;
  let originalEnv;
  let consoleLogSpy;
  let consoleErrorSpy;

  beforeEach(() => {
    // Save original env
    originalEnv = { ...process.env };
    // Clear any category filters and set level
    delete process.env.LOG_CATEGORIES;
    process.env.LOG_LEVEL = 'INFO';
    
    // Reload logger module to pick up new env vars
    delete require.cache[require.resolve('../../utils/logger')];
    logger = require('../../utils/logger');
    
    // Spy on console methods (must be after logger reload)
    consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    // Restore original env
    process.env = originalEnv;
    consoleLogSpy.mockRestore();
    consoleErrorSpy.mockRestore();
    // Clear cache for next test
    delete require.cache[require.resolve('../../utils/logger')];
  });

  describe('Log levels', () => {
    test('INFO level logs info messages', () => {
      process.env.LOG_LEVEL = 'INFO';
      delete require.cache[require.resolve('../../utils/logger')];
      logger = require('../../utils/logger');
      logger.info('Test info message');
      expect(consoleLogSpy).toHaveBeenCalled();
    });

    test('DEBUG level does not log when level is INFO', () => {
      process.env.LOG_LEVEL = 'INFO';
      delete require.cache[require.resolve('../../utils/logger')];
      logger = require('../../utils/logger');
      logger.debug('Test debug message');
      expect(consoleLogSpy).not.toHaveBeenCalled();
    });

    test('DEBUG level logs when level is DEBUG', () => {
      // Clear existing spy
      if (consoleLogSpy) {
        consoleLogSpy.mockRestore();
      }
      // Set LOG_LEVEL before requiring logger
      process.env.LOG_LEVEL = 'DEBUG';
      delete require.cache[require.resolve('../../utils/logger')];
      // Clear module cache to force reload
      jest.resetModules();
      logger = require('../../utils/logger');
      // Re-spy after reload
      consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      // DEBUG level is 0, which equals DEBUG level (0), so should log
      logger.debug('Test debug message');
      expect(consoleLogSpy).toHaveBeenCalled();
    });

    test('ERROR level logs error messages', () => {
      process.env.LOG_LEVEL = 'ERROR';
      logger.error('Test error message');
      expect(consoleErrorSpy).toHaveBeenCalled();
    });

    test('WARN level logs warning messages', () => {
      process.env.LOG_LEVEL = 'WARN';
      logger.warn('Test warning message');
      expect(consoleLogSpy).toHaveBeenCalled();
    });
  });

  describe('Category filtering', () => {
    test('Shows all categories when LOG_CATEGORIES not set', () => {
      // Clear existing spy
      if (consoleLogSpy) {
        consoleLogSpy.mockRestore();
      }
      // Delete env var to show all categories
      delete process.env.LOG_CATEGORIES;
      delete require.cache[require.resolve('../../utils/logger')];
      jest.resetModules();
      logger = require('../../utils/logger');
      // Re-spy after reload
      consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      consoleLogSpy.mockClear(); // Clear any calls from module load
      logger.info('Test message', 'audio');
      logger.info('Test message', 'video');
      // Both should be logged when no filter
      expect(consoleLogSpy).toHaveBeenCalledTimes(2);
    });

    test('Filters by category when LOG_CATEGORIES is set', () => {
      // Clear existing spy
      if (consoleLogSpy) {
        consoleLogSpy.mockRestore();
      }
      process.env.LOG_CATEGORIES = 'audio';
      delete require.cache[require.resolve('../../utils/logger')];
      jest.resetModules();
      logger = require('../../utils/logger');
      consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      consoleLogSpy.mockClear(); // Clear any calls from module load
      logger.info('Audio message', 'audio');
      logger.info('Video message', 'video');
      // Only audio should be logged
      expect(consoleLogSpy).toHaveBeenCalledTimes(1);
      // Check which one was logged (should be audio)
      const loggedMessage = consoleLogSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('Audio message');
    });

    test('Shows multiple categories when comma-separated', () => {
      // Clear existing spy
      if (consoleLogSpy) {
        consoleLogSpy.mockRestore();
      }
      process.env.LOG_CATEGORIES = 'audio,video';
      delete require.cache[require.resolve('../../utils/logger')];
      jest.resetModules();
      logger = require('../../utils/logger');
      consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      consoleLogSpy.mockClear(); // Clear any calls from module load
      logger.info('Audio message', 'audio');
      logger.info('Video message', 'video');
      logger.info('Chat message', 'chat');
      // audio and video should be logged, chat should not
      expect(consoleLogSpy).toHaveBeenCalledTimes(2);
    });

    test('System category (null) is shown when no filter', () => {
      // Clear existing spy
      if (consoleLogSpy) {
        consoleLogSpy.mockRestore();
      }
      // Delete env var to show all categories
      delete process.env.LOG_CATEGORIES;
      delete require.cache[require.resolve('../../utils/logger')];
      jest.resetModules();
      logger = require('../../utils/logger');
      // Re-spy after reload
      consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      consoleLogSpy.mockClear(); // Clear any calls from module load
      logger.info('System message'); // No category = system
      expect(consoleLogSpy).toHaveBeenCalled();
    });
  });

  describe('Timestamp format', () => {
    test('Logs include timestamp', () => {
      consoleLogSpy.mockClear(); // Clear any previous calls
      logger.info('Test message');
      expect(consoleLogSpy).toHaveBeenCalled();
      const logCall = consoleLogSpy.mock.calls[0][0];
      // Should contain timestamp pattern [YYYY-MM-DD HH:MM:SS]
      // Note: ANSI codes may be present, so we check for the pattern anywhere
      expect(logCall).toMatch(/\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]/);
    });

    test('Timestamp is in ISO-like format', () => {
      consoleLogSpy.mockClear(); // Clear any previous calls
      logger.info('Test message');
      expect(consoleLogSpy).toHaveBeenCalled();
      const logCall = consoleLogSpy.mock.calls[0][0];
      // Should match pattern [YYYY-MM-DD HH:MM:SS]
      // Extract timestamp ignoring ANSI codes
      const timestampMatch = logCall.match(/\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]/);
      expect(timestampMatch).toBeTruthy();
      const timestamp = timestampMatch[1];
      expect(timestamp).toMatch(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/);
    });
  });

  describe('Chat logging', () => {
    test('chat() formats message correctly', () => {
      // Clear existing spy
      if (consoleLogSpy) {
        consoleLogSpy.mockRestore();
      }
      // Delete env var to show all categories
      delete process.env.LOG_CATEGORIES;
      delete require.cache[require.resolve('../../utils/logger')];
      jest.resetModules();
      logger = require('../../utils/logger');
      consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      consoleLogSpy.mockClear(); // Clear any calls from module load
      logger.chat('testchannel', 'testuser', 'Hello world', 'chat');
      expect(consoleLogSpy).toHaveBeenCalled();
      const logCall = consoleLogSpy.mock.calls[0][0];
      // Should contain channel, username, and message
      expect(logCall).toContain('#testchannel');
      expect(logCall).toContain('testuser');
      expect(logCall).toContain('Hello world');
    });

    test('chat() includes timestamp', () => {
      // Clear existing spy
      if (consoleLogSpy) {
        consoleLogSpy.mockRestore();
      }
      // Delete env var to show all categories
      delete process.env.LOG_CATEGORIES;
      delete require.cache[require.resolve('../../utils/logger')];
      jest.resetModules();
      logger = require('../../utils/logger');
      consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      consoleLogSpy.mockClear(); // Clear any calls from module load
      logger.chat('testchannel', 'testuser', 'Hello', 'chat');
      expect(consoleLogSpy).toHaveBeenCalled();
      const logCall = consoleLogSpy.mock.calls[0][0];
      expect(logCall).toMatch(/\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]/);
    });

    test('chat() respects category filter', () => {
      // Clear existing spy
      if (consoleLogSpy) {
        consoleLogSpy.mockRestore();
      }
      process.env.LOG_CATEGORIES = 'chat';
      delete require.cache[require.resolve('../../utils/logger')];
      jest.resetModules();
      logger = require('../../utils/logger');
      consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      consoleLogSpy.mockClear(); // Clear any calls from module load
      logger.chat('testchannel', 'testuser', 'Hello', 'chat');
      logger.info('Other message', 'audio');
      // Only chat message should be logged
      expect(consoleLogSpy).toHaveBeenCalledTimes(1);
      expect(consoleLogSpy.mock.calls[0][0]).toContain('#testchannel');
    });
  });

  describe('Success logging', () => {
    test('success() logs with checkmark', () => {
      // Ensure no category filter blocks logging
      if (consoleLogSpy) {
        consoleLogSpy.mockRestore();
      }
      delete process.env.LOG_CATEGORIES;
      delete require.cache[require.resolve('../../utils/logger')];
      jest.resetModules();
      logger = require('../../utils/logger');
      consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      
      consoleLogSpy.mockClear(); // Clear any previous calls
      logger.success('Operation completed');
      expect(consoleLogSpy).toHaveBeenCalled();
      const logCall = consoleLogSpy.mock.calls[0][0];
      expect(logCall).toContain('✓');
      expect(logCall).toContain('Operation completed');
    });
  });

  describe('Error logging', () => {
    test('error() logs to stderr', () => {
      // Ensure no category filter blocks logging
      if (consoleErrorSpy) {
        consoleErrorSpy.mockRestore();
      }
      delete process.env.LOG_CATEGORIES;
      delete require.cache[require.resolve('../../utils/logger')];
      jest.resetModules();
      logger = require('../../utils/logger');
      consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

      consoleErrorSpy.mockClear(); // Clear any previous calls
      logger.error('Something went wrong');
      expect(consoleErrorSpy).toHaveBeenCalled();
      const logCall = consoleErrorSpy.mock.calls[0][0];
      expect(logCall).toContain('✗');
      expect(logCall).toContain('ERROR');
      expect(logCall).toContain('Something went wrong');
    });
  });
});

