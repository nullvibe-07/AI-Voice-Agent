# Real-Time Voice AI Agent Backend

A production-ready backend for a real-time conversational AI assistant built with FastAPI, LiveKit, and Groq LLM.

## Features

### Core Voice Pipeline
- **Voice Activity Detection (VAD)** - Detects when user starts/stops speaking
- **Speech-to-Text (STT)** - Real-time speech transcription using Deepgram
- **Turn Detection** - Intelligent pause detection to know when user finishes speaking
- **Language Model** - Groq LLM (Mixtral-8x7b) for intelligent, context-aware responses
- **Text-to-Speech (TTS)** - Converts responses to natural-sounding audio
- **Sub-second latency** - Optimized for responsive real-time interaction

### Dual-Mode Interaction
- **Voice Input** - Open-mic with VAD or push-to-talk
- **Text Input** - Chat-style text input
- **Seamless Switching** - Switch between voice and text mid-conversation
- **Streaming Output** - Text and audio stream simultaneously

### 4 Intelligent Tools

**Client-Side Tools** (Execute in Browser):
1. `change_theme` - Switch UI between light and dark mode
2. `show_notification` - Display toast notifications
3. `play_sound` - Play sound effects
4. `open_url` - Open links in new tabs

**Server-Side Tools** (Execute on Backend):
1. `get_weather` - Fetch live weather using Open-Meteo API
2. `search_wikipedia` - Look up topics on Wikipedia
3. `get_news` - Fetch news headlines (requires NewsAPI key)
4. `calculate` - Evaluate mathematical expressions

### Emotional Presence
- Adaptive tone matching user's energy
- Natural conversational rhythm with filler phrases
- Distinct, friendly personality
- Empathetic responses

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Voice Infrastructure** | LiveKit Cloud + LiveKit Agents |
| **Backend Framework** | FastAPI + Uvicorn |
| **Agent Framework** | LiveKit Agents Python SDK |
| **LLM** | Groq (Mixtral-8x7b or Llama2-70b) |
| **STT** | Deepgram |
| **TTS** | Deepgram or ElevenLabs |
| **Real-time Communication** | WebSockets |
| **Containerization** | Docker & Docker Compose |

## Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional, for containerized deployment)
- API Keys:
  - Groq API Key (free from https://console.groq.com)
  - Deepgram API Key (optional, for STT)
  - News API Key (optional, for news tool)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI-Voice-Agent
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the backend**
   ```bash
   python main.py
   ```

   Server will start at `http://localhost:8000`

### Docker Setup

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

   Services:
   - Backend: `http://localhost:8000`
   - LiveKit: `ws://localhost:7880`
   - Redis: `localhost:6379`

## API Endpoints

### REST Endpoints

#### Health & Status
- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /tools` - Available tools

#### Session Management
- `POST /session/create?room_name=<room>&participant_name=<name>` - Create session
- `GET /session/{conversation_id}` - Get session details
- `GET /session/{conversation_id}/messages` - Get conversation history
- `POST /text-input?conversation_id=<id>&text=<input>` - Submit text input

### WebSocket Endpoints

#### Connect
```
ws://localhost:8000/ws/{conversation_id}
```

#### Message Types

**User Input**
```json
{
  "type": "user_input",
  "data": {
    "text": "What's the weather in New York?",
    "input_type": "text"
  }
}
```

**Status Updates**
```json
{
  "type": "status_update",
  "data": {
    "is_listening": true,
    "transcript": "What's the..."
  }
}
```

**Server Responses**

- `streaming_text` - Text chunks from LLM
- `streaming_audio` - Audio chunks (base64)
- `tool_call` - Tool invocation notification
- `tool_result` - Tool execution result
- `status` - Status updates

## File Structure

```
AI-Voice-Agent/
├── main.py                 # FastAPI application
├── config.py              # Configuration management
├── models.py              # Pydantic models
├── voice_pipeline.py      # Voice processing pipeline
├── session_manager.py     # Session management
├── tools/
│   ├── __init__.py
│   ├── client_tools.py    # Browser-executed tools
│   └── server_tools.py    # Backend-executed tools
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── livekit.yaml         # LiveKit config
└── README.md            # This file
```

## Usage Examples

### Create a Session

```bash
curl -X POST "http://localhost:8000/session/create?room_name=room1&participant_name=John"
```

Response:
```json
{
  "success": true,
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "room_name": "room1",
  "participant_name": "John",
  "created_at": "2024-01-10T12:00:00"
}
```

### Send Text Input

```bash
curl -X POST "http://localhost:8000/text-input?conversation_id=550e8400-e29b-41d4-a716-446655440000&text=What's%20the%20weather%20today?"
```

### Connect WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/550e8400-e29b-41d4-a716-446655440000');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'user_input',
    data: {
      text: 'Hello, how are you?',
      input_type: 'text'
    }
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log(message.type, message.data);
};
```

## Configuration

### Environment Variables

```bash
# LiveKit
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret

# Groq LLM
GROQ_API_KEY=your_groq_api_key

# Deepgram (STT)
DEEPGRAM_API_KEY=your_deepgram_key

# News API (optional)
NEWS_API_KEY=your_news_api_key

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
DEBUG=False
```

## Getting Groq API Key

1. Visit [Groq Console](https://console.groq.com)
2. Sign up for a free account
3. Create an API key
4. Add it to your `.env` file

**Groq Advantages:**
- **Free tier** - No credit card required for development
- **Fast inference** - Extremely low latency responses
- **Multiple models** - Mixtral-8x7b, Llama2-70b, etc.
- **Competitive performance** - High-quality responses

## Performance Optimization

### Latency Targets
- **User speaks → ASR output**: < 200ms
- **ASR → LLM processing**: < 300ms
- **LLM → First TTS chunk**: < 200ms
- **Total**: < 1 second

### Optimization Strategies
1. **Streaming at every step** - Don't wait for full responses
2. **Parallel processing** - Process while user is still speaking
3. **Connection pooling** - Reuse HTTP/WebSocket connections
4. **Caching** - Cache frequently requested data
5. **Async operations** - Non-blocking I/O throughout

## Monitoring & Logging

Logging is configured with `loguru` for detailed debugging:

```python
from loguru import logger

logger.info("User input received")
logger.debug("Processing details")
logger.error("Error occurred")
```

Logs are printed to console with timestamps.

## Tool Development

### Adding a New Server Tool

1. Add method to `ServerTools` class in `tools/server_tools.py`:

```python
async def my_tool(self, param1: str) -> ToolResult:
    try:
        # Implementation
        result = await self._do_something(param1)
        return ToolResult(
            tool_name="my_tool",
            success=True,
            result=result
        )
    except Exception as e:
        return ToolResult(
            tool_name="my_tool",
            success=False,
            error=str(e)
        )
```

2. Update `get_available_tools()` method with tool schema

3. Add execution logic in `voice_pipeline.py`'s `_execute_server_tool()`

### Adding a New Client Tool

1. Add method to `ClientTools` class in `tools/client_tools.py`
2. Update `get_available_tools()` with schema
3. Add execution logic in `voice_pipeline.py`'s `_execute_client_tool()`
4. Implement browser-side handler in frontend

## Troubleshooting

### Common Issues

**WebSocket connection fails**
- Check LiveKit server is running
- Verify conversation_id is valid
- Check CORS settings

**STT not working**
- Verify Deepgram API key
- Check audio input permissions
- Test with HTTP endpoint first

**LLM rate limiting**
- Groq has generous free tier limits
- Check your API key quota
- Consider rate limiting on client side

**High latency**
- Check network connectivity
- Monitor server CPU/memory
- Review streaming implementation
- Reduce model size if needed

## Migration from OpenAI

This project has been migrated from OpenAI to Groq LLM. Key changes:

1. **LLM Client**: Changed from `AsyncOpenAI` to `Groq`
2. **Dependencies**: Replaced `openai` package with `groq`
3. **API Key**: Changed from `OPENAI_API_KEY` to `GROQ_API_KEY`
4. **TTS**: Migrated to Deepgram TTS (previously used OpenAI TTS)
5. **STT**: Using Deepgram (no OpenAI Whisper API dependency)

### Benefits of Groq:
- ✅ Free API tier (no credit card needed)
- ✅ Ultra-fast inference (10x faster than cloud LLMs)
- ✅ Cost-effective for production
- ✅ Multiple model options
- ✅ Excellent for real-time applications

## Deployment

### Production Deployment

1. **Environment setup**
   - Use environment variables for all secrets
   - Set DEBUG=False
   - Configure proper CORS origins

2. **Use production ASGI server**
   ```bash
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
   ```

3. **Database setup**
   - Configure Redis for session management
   - Set up PostgreSQL for persistent storage (if needed)

4. **Monitoring**
   - Set up logging aggregation (e.g., ELK, Datadog)
   - Configure error tracking (e.g., Sentry)
   - Set up performance monitoring

5. **Scaling**
   - Use load balancer (nginx, HAProxy)
   - Run multiple backend instances
   - Use Redis for session sharing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Submit pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review LiveKit documentation: https://docs.livekit.io
- Check Groq documentation: https://console.groq.com/docs

## Acknowledgments

- LiveKit for voice infrastructure
- Groq for LLM inference
- Deepgram for STT/TTS
- FastAPI for the web framework
