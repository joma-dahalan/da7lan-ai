Overview
This is a Telegram bot for gold (XAU/USD) trading analysis. The bot provides technical analysis, pattern recognition, session sentiment, and AI-powered trading recommendations using real-time market data from the TwelveData API.
===================================================================================================
===================================================================================================
Key features:

Real-time gold price fetching
Multiple analysis reports (sniper, scalp, titan, oracle, patterns, divergence, supply/demand, session sentiment)
Subscription-based access with activation codes
Trial system for new users
Admin management capabilities

==================================================================================================================
User Preferences
Preferred communication style: Simple, everyday language.

==================================================================================================================
System Architecture
Bot Framework
Technology: Python with python-telegram-bot library
Pattern: Command and callback-based async handlers
Rationale: Native Telegram bot framework provides robust message handling and inline keyboard support

==================================================================================================================
Data Layer
Market Data: TwelveData API for real-time XAU/USD prices and OHLCV candles
User Storage: JSON files (users.json, codes.json) for persistence
Rationale: Simple file-based storage suits the low-volume user management needs; external API provides reliable market data
===========================================================================================================================
Analysis Engine
Location: analysis.py module
Capabilities: Technical indicators, chart patterns, wave analysis, session timing, meta AI reports
Libraries: pandas, numpy for data processing
Rationale: Centralized analysis module allows modular report generation with consistent data fetching
=============================================================================================================================
Access Control
Subscription Model: Time-based expiry stored per user
Activation Codes: Pre-generated codes in codes.json with single-use enforcement
Trial System: One-time trial per user tracked via trial_used flag
Admin: Single admin ID from environment for management commands
================================================================================================================================
Code Generation
Utility: generate_codes.py creates random GOLD_XXXXXX format codes
Purpose: Offline code generation for distribution to paying users

================================================================================================================================

                                            External Dependencies
================================================================================================================================

APIs
================================================================================================================================
TwelveData API: Market data provider for gold prices and candlestick data
Requires TWELVEDATA_KEY environment variable
Has daily quota limits
Endpoints: /price, /time_series
=================================================================================================================================
Environment Variables
BOT_TOKEN: Telegram Bot API token (required)
TWELVEDATA_KEY: TwelveData API key (required)
ADMIN_ID: Telegram user ID for admin commands
SYMBOL: Trading symbol (defaults to XAU/USD)
==================================================================================================================================
Python Dependencies
python-telegram-bot: Telegram bot framework
requests: HTTP client for API calls
pandas: Data manipulation
numpy: Numerical operations
python-dotenv: Environment variable loading
nest_asyncio: Async compatibility
