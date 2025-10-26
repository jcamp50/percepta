#!/usr/bin/env node

/**
 * Twitch Authentication Test Script
 *
 * Verifies that Twitch credentials are valid and have the correct scopes.
 */

require('dotenv').config();
const axios = require('axios');
const tmi = require('tmi.js');

const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  bold: '\x1b[1m',
};

function log(message, color = colors.reset) {
  console.log(`${color}${message}${colors.reset}`);
}

async function validateEnvironmentVariables() {
  log('\n==============================================', colors.cyan);
  log('   Twitch Authentication Test', colors.cyan);
  log('==============================================\n', colors.cyan);

  const requiredVars = [
    'TWITCH_CLIENT_ID',
    'TWITCH_CLIENT_SECRET',
    'TWITCH_BOT_TOKEN',
    'TWITCH_BOT_NAME',
  ];

  let allPresent = true;

  log('1. Checking environment variables...', colors.bold);

  for (const varName of requiredVars) {
    if (process.env[varName]) {
      log(`   ✓ ${varName} is set`, colors.green);
    } else {
      log(`   ✗ ${varName} is missing`, colors.red);
      allPresent = false;
    }
  }

  if (!allPresent) {
    log('\n❌ Some required environment variables are missing.', colors.red);
    log('Please check your .env file.\n', colors.yellow);
    return false;
  }

  log('   ✓ All required variables present\n', colors.green);
  return true;
}

async function validateOAuthToken() {
  log('2. Validating OAuth token...', colors.bold);

  const token = process.env.TWITCH_BOT_TOKEN.replace('oauth:', '');
  const clientId = process.env.TWITCH_CLIENT_ID;

  try {
    const response = await axios.get('https://id.twitch.tv/oauth2/validate', {
      headers: {
        Authorization: `OAuth ${token}`,
      },
    });

    const data = response.data;

    log(`   ✓ Token is valid`, colors.green);
    log(`   ✓ Client ID: ${data.client_id}`, colors.green);
    log(`   ✓ User ID: ${data.user_id}`, colors.green);
    log(`   ✓ Login: ${data.login}`, colors.green);
    log(
      `   ✓ Expires in: ${Math.floor(data.expires_in / 3600)} hours`,
      colors.green
    );
    log(`   ✓ Scopes: ${data.scopes.join(', ')}\n`, colors.green);

    // Check required scopes
    const requiredScopes = ['chat:read', 'chat:edit'];
    const hasAllScopes = requiredScopes.every((scope) =>
      data.scopes.includes(scope)
    );

    if (hasAllScopes) {
      log('   ✓ All required scopes present\n', colors.green);
    } else {
      log('   ⚠️  Warning: Missing required scopes', colors.yellow);
      const missing = requiredScopes.filter(
        (scope) => !data.scopes.includes(scope)
      );
      log(`   Missing: ${missing.join(', ')}\n`, colors.yellow);
    }

    // Verify client ID matches
    if (data.client_id !== clientId) {
      log(
        '   ⚠️  Warning: Token client ID does not match TWITCH_CLIENT_ID',
        colors.yellow
      );
      log(`   Token client ID: ${data.client_id}`, colors.yellow);
      log(`   Expected: ${clientId}\n`, colors.yellow);
    }

    return true;
  } catch (error) {
    if (error.response?.status === 401) {
      log('   ✗ Token is invalid or expired', colors.red);
      log('   Please regenerate your OAuth token.\n', colors.yellow);
    } else {
      log(`   ✗ Error validating token: ${error.message}`, colors.red);
    }
    return false;
  }
}

async function testIRCConnection() {
  log('3. Testing IRC connection...', colors.bold);

  return new Promise((resolve) => {
    const client = new tmi.Client({
      identity: {
        username: process.env.TWITCH_BOT_NAME,
        password: process.env.TWITCH_BOT_TOKEN,
      },
      connection: {
        reconnect: false,
        timeout: 10000,
      },
    });

    const timeout = setTimeout(() => {
      log('   ✗ Connection timeout', colors.red);
      client.disconnect();
      resolve(false);
    }, 15000);

    client.on('connected', () => {
      clearTimeout(timeout);
      log('   ✓ Successfully connected to Twitch IRC', colors.green);
      log(`   ✓ Bot username: ${process.env.TWITCH_BOT_NAME}\n`, colors.green);
      client.disconnect();
      resolve(true);
    });

    client.on('disconnected', (reason) => {
      if (reason !== 'Connection closed.') {
        clearTimeout(timeout);
        log(`   ✗ Disconnected: ${reason}`, colors.red);
        resolve(false);
      }
    });

    client.connect().catch((error) => {
      clearTimeout(timeout);
      log(`   ✗ Connection failed: ${error.message}`, colors.red);

      if (error.message.includes('Login authentication failed')) {
        log(
          '   This usually means the OAuth token is invalid.\n',
          colors.yellow
        );
      }

      resolve(false);
    });
  });
}

async function main() {
  try {
    const envValid = await validateEnvironmentVariables();
    if (!envValid) {
      process.exit(1);
    }

    const tokenValid = await validateOAuthToken();
    if (!tokenValid) {
      log('==============================================', colors.cyan);
      log('❌ Authentication test failed', colors.red);
      log('==============================================\n', colors.cyan);
      process.exit(1);
    }

    const ircConnected = await testIRCConnection();

    log('==============================================', colors.cyan);
    if (ircConnected) {
      log('✓ All tests passed!', colors.green);
      log('Your Twitch authentication is configured correctly.', colors.green);
    } else {
      log('⚠️  Some tests failed', colors.yellow);
      log('Please check the errors above.', colors.yellow);
    }
    log('==============================================\n', colors.cyan);

    process.exit(ircConnected ? 0 : 1);
  } catch (error) {
    log(`\n❌ Unexpected error: ${error.message}`, colors.red);
    console.error(error);
    process.exit(1);
  }
}

main();
