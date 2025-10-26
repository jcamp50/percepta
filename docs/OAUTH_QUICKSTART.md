# OAuth Setup Quickstart

Fast-track guide to get your Twitch OAuth token.

## Prerequisites

✅ You have already:

- Created a Twitch bot account
- Registered your application in Twitch Developer Console
- Added `TWITCH_CLIENT_ID` and `TWITCH_CLIENT_SECRET` to your `.env` file

## Generate Your OAuth Token

### Option 1: Using npm script (recommended)

```bash
npm run setup:oauth
```

### Option 2: Direct node command

```bash
node scripts/init_twitch_oauth.js
```

## Steps

1. **Run the script** - It will prompt for your Client ID and Client Secret
2. **Press ENTER** - When ready to open browser
3. **Authorize** - Browser opens to Twitch authorization page
   - ⚠️ Make sure you're logged into your **BOT account**
   - Click "Authorize"
4. **Copy token** - Terminal displays your OAuth token
5. **Paste to .env** - Add the token to your `.env` file

## Test Your Setup

```bash
npm run test:auth
```

If all tests pass ✅, you're ready for the next phase!

## Troubleshooting

| Issue                         | Solution                                                                    |
| ----------------------------- | --------------------------------------------------------------------------- |
| Wrong account authorized      | Log out of all Twitch accounts, log into bot account only, run script again |
| "Login authentication failed" | Token is invalid, regenerate with `npm run setup:oauth`                     |
| Browser doesn't open          | Manually copy the URL from terminal and paste in browser                    |
| Connection timeout            | Check internet connection, try again                                        |

## Need More Help?

See the complete guide: [docs/TWITCH_SETUP.md](./TWITCH_SETUP.md)
