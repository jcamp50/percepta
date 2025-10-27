# Twitch Bot Setup Guide

Complete guide for setting up Twitch authentication for the Percepta bot.

---

## Overview

To connect your bot to Twitch chat, you need three pieces of information:

1. **Client ID** - Identifies your application
2. **Client Secret** - Authenticates your application
3. **OAuth Token** - Authorizes your bot to read/send chat messages

---

## Step 1: Create a Twitch Bot Account

1. Go to [Twitch Signup](https://www.twitch.tv/signup)
2. Create a new account for your bot
   - Suggested username: `perceptabot` or similar
   - Use a dedicated email address
3. Verify your email if prompted
4. **Important**: Keep the login credentials safe

---

## Step 2: Register Your Application

1. **Log into the bot account** (not your personal account!)
2. Go to [Twitch Developer Console](https://dev.twitch.tv/console/apps)
3. Click **"Register Your Application"**
4. Fill in the application details:
   - **Name**: `Percepta Bot` (or your chosen name)
   - **OAuth Redirect URLs**: `http://localhost:3000/callback`
   - **Category**: `Chat Bot`
5. Click **"Create"**
6. Click **"Manage"** on your newly created application
7. Copy and save:
   - **Client ID** (shown directly)
   - **Client Secret** (click "New Secret" if not visible)

### Add to `.env` File

Open your `.env` file and add the credentials:

```env
TWITCH_CLIENT_ID=your_client_id_here
TWITCH_CLIENT_SECRET=your_client_secret_here
TWITCH_BOT_NAME=perceptabot  # or your bot's username
```

---

## Step 3: Generate OAuth Token

Now you need to generate an OAuth token with the proper scopes.

### Install Node Dependencies

First, make sure you have the required dependencies:

```bash
npm install
```

### Run the OAuth Helper Script

```bash
node scripts/init_twitch_oauth.js
```

### Follow the Interactive Prompts

1. The script will ask for your **Client ID** and **Client Secret**
2. It will open your browser to the Twitch authorization page
   - **IMPORTANT**: Make sure you're logged into your **BOT account**, not your personal account
   - If you're logged into the wrong account, log out and log into the bot account
3. Click **"Authorize"** to grant permissions
4. The browser will redirect to a success page
5. Return to your terminal - the script will display your OAuth token

### Copy the Token to `.env`

The script will output something like:

```
==============================================
   Copy this to your .env file:
==============================================

TWITCH_BOT_TOKEN=oauth:krjgn3k2jn4k2j3n4kj2n34kj

==============================================
```

Copy this line and paste it into your `.env` file.

---

## Step 4: Set Target Channel

In your `.env` file, specify which channel the bot should monitor:

```env
TARGET_CHANNEL=your_twitch_username
```

Replace `your_twitch_username` with the channel you want to test with (usually your personal channel).

---

## Step 5: Test Authentication

Verify everything is set up correctly:

```bash
node scripts/test_twitch_auth.js
```

This script will:

- ✓ Check all required environment variables are set
- ✓ Validate your OAuth token with Twitch
- ✓ Verify the token has correct scopes
- ✓ Test connection to Twitch IRC

### Expected Output

```
==============================================
   Twitch Authentication Test
==============================================

1. Checking environment variables...
   ✓ TWITCH_CLIENT_ID is set
   ✓ TWITCH_CLIENT_SECRET is set
   ✓ TWITCH_BOT_TOKEN is set
   ✓ TWITCH_BOT_NAME is set
   ✓ All required variables present

2. Validating OAuth token...
   ✓ Token is valid
   ✓ Client ID: abc123...
   ✓ User ID: 123456789
   ✓ Login: perceptabot
   ✓ Expires in: 1461 hours
   ✓ Scopes: chat:read, chat:edit
   ✓ All required scopes present

3. Testing IRC connection...
   ✓ Successfully connected to Twitch IRC
   ✓ Bot username: perceptabot

==============================================
✓ All tests passed!
Your Twitch authentication is configured correctly.
==============================================
```

---

## Complete `.env` Example

Your `.env` file should now look like this:

```env
# Twitch Bot Credentials
TWITCH_CLIENT_ID=a1b2c3d4e5f6g7h8i9j0
TWITCH_CLIENT_SECRET=x9y8z7w6v5u4t3s2r1q0
TWITCH_BOT_TOKEN=oauth:krjgn3k2jn4k2j3n4kj2n34kj
TWITCH_BOT_NAME=perceptabot
TARGET_CHANNEL=yourchannelname

# OpenAI
OPENAI_API_KEY=sk-...

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/percepta
REDIS_URL=redis://localhost:6379

# ngrok (development)
NGROK_AUTH_TOKEN=your_ngrok_token_here

# STT Configuration
WHISPER_MODEL=base
USE_GPU=false

# Memory Configuration
TRANSCRIPT_WINDOW_MINUTES=10
SUMMARY_INTERVAL_SECONDS=30

# Performance
MAX_CONCURRENT_REQUESTS=10
EMBEDDING_BATCH_SIZE=10

# Logging
LOG_LEVEL=INFO
```

---

## Troubleshooting

### "Login authentication failed"

**Cause**: OAuth token is invalid, expired, or malformed.

**Solution**:

1. Make sure the token starts with `oauth:` in your `.env` file
2. Regenerate the token using `node scripts/init_twitch_oauth.js`
3. Make sure you authorized with the bot account, not your personal account

### "Client ID does not match"

**Cause**: The OAuth token was generated with a different Client ID.

**Solution**: Regenerate the token with the correct Client ID.

### "Missing required scopes"

**Cause**: The token doesn't have `chat:read` and `chat:edit` permissions.

**Solution**: Regenerate the token using the OAuth helper script (it automatically requests the correct scopes).

### "Connection timeout"

**Cause**: Network issues or Twitch IRC is down.

**Solution**:

1. Check your internet connection
2. Verify Twitch is not experiencing outages: [Twitch Status](https://status.twitch.tv)
3. Try again in a few minutes

### Wrong Account Authorized

**Cause**: You authorized with your personal account instead of the bot account.

**Solution**:

1. Log out of all Twitch accounts in your browser
2. Log into the bot account only
3. Run `node scripts/init_twitch_oauth.js` again

### Token Expired

OAuth tokens from Twitch typically last for about 60 days.

**Solution**: When your token expires, simply run the OAuth script again to generate a new one:

```bash
node scripts/init_twitch_oauth.js
```

---

## Security Best Practices

1. **Never commit your `.env` file** - It's already in `.gitignore`
2. **Keep Client Secret private** - Don't share it or commit it to version control
3. **Regenerate secrets if exposed** - If you accidentally expose credentials, regenerate them in the Twitch Developer Console
4. **Use environment-specific credentials** - Use different bot accounts for development and production

---

## Next Steps

Once authentication is working:

- ✓ JCB-6 Complete: Twitch OAuth Setup
- ⏭️ JCB-7: Node IRC Chat Service - Build the actual chat bot

---

## Additional Resources

- [Twitch Authentication Docs](https://dev.twitch.tv/docs/authentication)
- [Twitch IRC Documentation](https://dev.twitch.tv/docs/irc)
- [tmi.js Library](https://tmijs.com/)
- [OAuth 2.0 Guide](https://dev.twitch.tv/docs/authentication/getting-tokens-oauth)

---

**Need Help?**

If you encounter issues not covered here, check:

1. The test script output for specific error messages
2. Twitch Developer Console for app status
3. Twitch API status page for service disruptions
