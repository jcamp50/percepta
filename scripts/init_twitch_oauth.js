#!/usr/bin/env node

/**
 * Twitch OAuth Token Generator
 *
 * Interactive CLI tool to generate OAuth tokens for Twitch bot authentication.
 * Opens browser for OAuth flow and captures the access token.
 */

const express = require('express');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function question(query) {
  return new Promise((resolve) => rl.question(query, resolve));
}

async function main() {
  console.log('\n==============================================');
  console.log('   Twitch OAuth Token Generator');
  console.log('   Percepta Bot Authentication Setup');
  console.log('==============================================\n');

  console.log(
    'This tool will help you generate an OAuth token for your Twitch bot.\n'
  );
  console.log('Required scopes:');
  console.log('  • chat:read  - Read chat messages');
  console.log('  • chat:edit  - Send chat messages\n');

  // Get credentials
  const clientId = await question('Enter your Twitch Client ID: ');
  const clientSecret = await question('Enter your Twitch Client Secret: ');

  if (!clientId || !clientSecret) {
    console.error('\n❌ Error: Client ID and Client Secret are required.');
    rl.close();
    process.exit(1);
  }

  console.log('\n📋 Setting up local server to handle OAuth callback...');

  // Set up Express server to handle OAuth callback
  const app = express();
  const PORT = 3000;
  const REDIRECT_URI = `http://localhost:${PORT}/callback`;

  let server;
  let accessToken = null;

  app.get('/callback', async (req, res) => {
    const code = req.query.code;
    const error = req.query.error;

    if (error) {
      res.send(`
        <html>
          <body style="font-family: Arial, sans-serif; padding: 40px; text-align: center;">
            <h1 style="color: #dc2626;">❌ Authorization Failed</h1>
            <p>Error: ${error}</p>
            <p>${req.query.error_description || ''}</p>
            <p>You can close this window.</p>
          </body>
        </html>
      `);
      console.error('\n❌ Authorization failed:', error);
      server.close();
      rl.close();
      process.exit(1);
      return;
    }

    if (!code) {
      res.send(`
        <html>
          <body style="font-family: Arial, sans-serif; padding: 40px; text-align: center;">
            <h1 style="color: #dc2626;">❌ No Authorization Code</h1>
            <p>No authorization code received from Twitch.</p>
            <p>You can close this window.</p>
          </body>
        </html>
      `);
      console.error('\n❌ No authorization code received.');
      server.close();
      rl.close();
      process.exit(1);
      return;
    }

    console.log(
      '\n✓ Authorization code received, exchanging for access token...'
    );

    // Exchange authorization code for access token
    try {
      const axios = require('axios');
      const tokenResponse = await axios.post(
        'https://id.twitch.tv/oauth2/token',
        null,
        {
          params: {
            client_id: clientId,
            client_secret: clientSecret,
            code: code,
            grant_type: 'authorization_code',
            redirect_uri: REDIRECT_URI,
          },
        }
      );

      accessToken = tokenResponse.data.access_token;
      const scopes = tokenResponse.data.scope;

      res.send(`
        <html>
          <body style="font-family: Arial, sans-serif; padding: 40px; text-align: center;">
            <h1 style="color: #16a34a;">✓ Success!</h1>
            <p style="font-size: 18px;">Your OAuth token has been generated.</p>
            <p>Scopes granted: <code>${scopes.join(', ')}</code></p>
            <p style="margin-top: 30px; color: #666;">Return to your terminal to complete setup.</p>
            <p style="color: #666;">You can close this window now.</p>
          </body>
        </html>
      `);

      console.log('\n✓ Access token received!');
      console.log(`✓ Scopes granted: ${scopes.join(', ')}`);
      console.log('\n==============================================');
      console.log('   Copy this to your .env file:');
      console.log('==============================================\n');
      console.log(`TWITCH_BOT_TOKEN=oauth:${accessToken}\n`);
      console.log('==============================================\n');

      // Give the response time to send before closing
      setTimeout(() => {
        server.close();
        rl.close();
        process.exit(0);
      }, 1000);
    } catch (error) {
      console.error(
        '\n❌ Error exchanging code for token:',
        error.response?.data || error.message
      );
      res.send(`
        <html>
          <body style="font-family: Arial, sans-serif; padding: 40px; text-align: center;">
            <h1 style="color: #dc2626;">❌ Token Exchange Failed</h1>
            <p>Failed to exchange authorization code for access token.</p>
            <p>Error: ${error.response?.data?.message || error.message}</p>
            <p>You can close this window.</p>
          </body>
        </html>
      `);
      server.close();
      rl.close();
      process.exit(1);
    }
  });

  // Start server
  server = app.listen(PORT, () => {
    console.log(`✓ Server running on http://localhost:${PORT}`);
  });

  // Build authorization URL
  const scopes = ['chat:read', 'chat:edit'];
  const authUrl = `https://id.twitch.tv/oauth2/authorize?client_id=${clientId}&redirect_uri=${encodeURIComponent(
    REDIRECT_URI
  )}&response_type=code&scope=${encodeURIComponent(scopes.join(' '))}`;

  console.log('\n🌐 Opening browser for authorization...');
  console.log(
    '\nIMPORTANT: Make sure you are logged into your BOT account in your browser!'
  );
  console.log('If you are logged into your personal account, log out first.\n');

  await question('Press ENTER when you are ready to open the browser...');

  try {
    const open = (await import('open')).default;
    await open(authUrl);
    console.log(
      '\n✓ Browser opened. Please authorize the application in your browser.'
    );
    console.log('Waiting for authorization...\n');
  } catch (error) {
    console.error('\n⚠️  Could not open browser automatically.');
    console.log('\nPlease manually open this URL in your browser:\n');
    console.log(authUrl);
    console.log('\n');
  }

  // Handle timeout
  setTimeout(() => {
    if (!accessToken) {
      console.error('\n❌ Timeout: No authorization received after 5 minutes.');
      console.log('Please try again.\n');
      server.close();
      rl.close();
      process.exit(1);
    }
  }, 5 * 60 * 1000); // 5 minutes
}

// Handle Ctrl+C
process.on('SIGINT', () => {
  console.log('\n\n❌ Process interrupted. Exiting...\n');
  rl.close();
  process.exit(0);
});

main().catch((error) => {
  console.error('\n❌ Unexpected error:', error.message);
  rl.close();
  process.exit(1);
});
