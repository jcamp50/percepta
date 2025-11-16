module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/__tests__/**/*.test.js'],
  collectCoverage: true,
  collectCoverageFrom: ['node/**/*.js', '!node/**/index.js'],
  coverageThreshold: {
    global: {
      lines: 80,
      statements: 80,
      functions: 80,
      branches: 70,
    },
  },
  setupFiles: ['<rootDir>/node/__tests__/setupEnv.js'],
  resetMocks: true,
  restoreMocks: true,
};

