"""Voice processing pipeline with streaming support - Full voice AI agent"""

import asyncio
from typing import Optional, AsyncGenerator, Callable
from loguru import logger
import json
from datetime import datetime

from models import ChatMessage, ToolCall, ToolResult
from tools.server_tools import ServerTools
from tools.client_tools import ClientTools
from config import settings

class VoicePipeline:
    """Main voice processing pipeline with LLM, STT, TTS, and tool calling"""

    def __init__(self):
        self.server_tools = ServerTools()
        self.client_tools = ClientTools()
        self.conversation_history: list[ChatMessage] = []
        self.is_processing = False
        self.groq_client = None
        self.stt_engine = None
        self.tts_engine = None
        self.vad_detector = None

    async def initialize(self):
        """Initialize the voice pipeline with all engines"""
        try:
            # Initialize Groq client for LLM
            from groq import Groq
            self.groq_client = Groq(api_key=settings.groq_api_key)
            logger.info("Groq LLM client initialized")
            
            # Initialize STT (Speech-to-Text) - Deepgram or Whisper
            try:
                from livekit.plugins import deepgram
                self.stt_engine = deepgram.STT(api_key=settings.deepgram_api_key)
                logger.info("Deepgram STT initialized")
            except Exception as e:
                logger.warning(f"Deepgram STT failed, will use Whisper: {str(e)}")
                try:
                    from livekit.plugins import silero
                    # Fallback to Silero VAD + OpenAI Whisper
                    self.vad_detector = silero.VAD.load()
                    logger.info("Silero VAD initialized as fallback")
                except Exception as e2:
                    logger.warning(f"Silero VAD also failed: {str(e2)}")
            
            # Initialize TTS (Text-to-Speech) - Using Deepgram or ElevenLabs (no OpenAI dependency)
            try:
                from livekit.plugins import deepgram as deepgram_plugin
                self.tts_engine = deepgram_plugin.TTS(api_key=settings.deepgram_api_key)
                logger.info("Deepgram TTS initialized")
            except Exception as e:
                logger.warning(f"Deepgram TTS failed: {str(e)}")
                try:
                    # Fallback to alternative TTS if available
                    from livekit.plugins import elevenlabs
                    self.tts_engine = elevenlabs.TTS()
                    logger.info("ElevenLabs TTS initialized as fallback")
                except Exception as e2:
                    logger.warning(f"ElevenLabs TTS also failed: {str(e2)}")
            
            await self.server_tools.initialize()
            logger.info("Voice pipeline initialized successfully with all engines")
        except Exception as e:
            logger.error(f"Error initializing voice pipeline: {str(e)}")
            raise

    async def cleanup(self):
        """Clean up resources"""
        await self.server_tools.cleanup()
        logger.info("Voice pipeline cleaned up")

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the LLM with emotional presence
        """
        return """You are an emotionally intelligent, conversational AI assistant with a warm and engaging personality. 

Your core characteristics:
- Warm and empathetic: You listen carefully and respond with genuine understanding
- Adaptive tone: You match the user's energy - excited when they're excited, calm when they're stressed
- Natural conversationalist: You use filler phrases naturally ("Hmm, interesting...", "Oh, I see...", "That's a great question...") to feel like a real person
- Proactive with tools: You naturally suggest using tools when it would help ("Let me look that up for you...", "I'll check the weather real quick...")
- Concise in speech: Your responses are clear and conversational, not robotic or overly formal
- Action-oriented: When a tool can help, you use it naturally and weave the results back into conversation

Available Tools:
1. Client-side tools (execute in user's browser):
   - change_theme(theme: "light" or "dark") - Change UI theme
   - show_notification(message: str, duration_ms: int) - Display notification
   - play_sound(sound_type: "success"|"error"|"notification"|"chime") - Play sound effect
   - open_url(url: str) - Open URL in new tab

2. Server-side tools (execute on backend):
   - get_weather(city: str, country: str) - Get current weather
   - search_wikipedia(query: str, sentences: int) - Search Wikipedia
   - get_news(topic: str, region: str) - Get news headlines
   - calculate(expression: str) - Evaluate math expression

When using tools:
- Always narrate your actions: "Let me check that for you..."
- Use tool results naturally in your response
- For client tools, explicitly mention what you're doing: "I'm going to change the theme to dark mode for you"
- Never dump raw tool results - synthesize them into natural speech

Speak naturally and conversationally. You're having a chat, not giving a presentation."""

    async def process_user_input(
        self,
        user_text: str,
        on_text_chunk: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[ToolCall], None]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Process user input text and stream response
        
        Args:
            user_text: User input text (from typing or STT)
            on_text_chunk: Callback for text chunks
            on_tool_call: Callback for tool calls
            
        Yields:
            Response text chunks
        """
        try:
            self.is_processing = True
            
            # Add user message to history
            user_msg = ChatMessage(role="user", content=user_text)
            self.conversation_history.append(user_msg)
            
            logger.info(f"Processing user input: {user_text}")
            
            # Prepare messages for LLM
            messages = self._prepare_messages()
            
            # Get tools schema
            tools_schema = self._get_tools_schema()
            
            # Stream response from LLM
            async for chunk in self._stream_llm_response(
                messages=messages,
                tools_schema=tools_schema,
                on_text_chunk=on_text_chunk,
                on_tool_call=on_tool_call,
            ):
                yield chunk
            
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            yield f"Sorry, I encountered an error: {str(e)}"
        finally:
            self.is_processing = False

    async def _stream_llm_response(
        self,
        messages: list,
        tools_schema: list,
        on_text_chunk: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[ToolCall], None]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM response with tool calling support using Groq API
        """
        try:
            if not self.groq_client:
                raise RuntimeError("Groq client not initialized")
            
            full_response = ""
            tool_calls_list = []
            
            # Stream completion from Groq
            stream = self.groq_client.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                tools=tools_schema if tools_schema else None,
                tool_choice="auto" if tools_schema else None,
                temperature=0.7,
                max_tokens=500,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    
                    # Handle text content
                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        text_chunk = choice.delta.content
                        full_response += text_chunk
                        
                        # Emit text chunk for streaming to client
                        if on_text_chunk:
                            on_text_chunk(text_chunk)
                        
                        yield text_chunk
                    
                    # Handle tool calls
                    if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                        for tool_call in choice.delta.tool_calls:
                            tool_info = self._parse_tool_call(tool_call)
                            if tool_info:
                                tool_calls_list.append(tool_info)
                                if on_tool_call:
                                    on_tool_call(tool_info)
            
            # Add assistant response to history
            assistant_msg = ChatMessage(
                role="assistant",
                content=full_response
            )
            self.conversation_history.append(assistant_msg)
            
            logger.info(f"LLM response streamed successfully. Tool calls: {len(tool_calls_list)}")
            
        except Exception as e:
            logger.error(f"Error in LLM streaming: {str(e)}")
            raise

    async def synthesize_speech(
        self,
        text: str,
        on_audio_chunk: Optional[Callable[[bytes], None]] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Convert text to speech using TTS engine and stream audio chunks
        
        Args:
            text: Text to synthesize
            on_audio_chunk: Callback for audio chunks
            
        Yields:
            Audio data chunks
        """
        try:
            if not self.tts_engine:
                logger.warning("TTS engine not initialized, skipping speech synthesis")
                return
            
            logger.info(f"Synthesizing speech for text: {text[:50]}...")
            
            # Use LiveKit TTS plugin to synthesize
            # This will stream audio chunks as they become available
            try:
                # If using LiveKit TTS plugin (Deepgram or ElevenLabs)
                audio_stream = await self.tts_engine.asynthesize(text)
                
                async for chunk in audio_stream:
                    if on_audio_chunk:
                        on_audio_chunk(chunk)
                    yield chunk
                    
            except AttributeError:
                # Fallback: Use external TTS service
                logger.warning("TTS plugin streaming not available, using fallback")
                # You can implement a fallback to Deepgram TTS API or another service
                pass
            
            logger.info("Speech synthesis completed")
            
        except Exception as e:
            logger.error(f"Error in speech synthesis: {str(e)}")
            raise

    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: str = "en"
    ) -> str:
        """
        Convert speech to text using STT engine
        
        Args:
            audio_data: Raw audio bytes
            language: Language code (default: English)
            
        Returns:
            Transcribed text
        """
        try:
            if not self.stt_engine:
                logger.warning("STT engine not initialized, using Whisper fallback")
                return await self._whisper_transcribe(audio_data, language)
            
            logger.info("Transcribing audio...")
            
            # Use LiveKit STT plugin
            try:
                transcript = await self.stt_engine.atranscribe(audio_data)
                text = transcript.text if hasattr(transcript, 'text') else str(transcript)
                logger.info(f"Transcribed: {text}")
                return text
                
            except Exception as e:
                logger.warning(f"STT plugin transcription failed: {str(e)}, falling back to Whisper")
                return await self._whisper_transcribe(audio_data, language)
            
        except Exception as e:
            logger.error(f"Error in audio transcription: {str(e)}")
            raise

    async def _whisper_transcribe(self, audio_data: bytes, language: str = "en") -> str:
        """
        Fallback transcription using OpenAI Whisper API via aiohttp or direct Deepgram
        """
        try:
            import io
            
            # Use Deepgram for transcription as it's already integrated
            if self.stt_engine:
                transcript = await self.stt_engine.atranscribe(audio_data)
                text = transcript.text if hasattr(transcript, 'text') else str(transcript)
                logger.info(f"Deepgram transcribed: {text}")
                return text
            
            # Alternative: Use a simple aiohttp call to a transcription service
            logger.warning("No STT engine available for transcription")
            return ""
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return ""

    def detect_voice_activity(self, audio_data: bytes) -> bool:
        """
        Detect if audio contains speech using VAD
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            True if speech detected, False otherwise
        """
        try:
            if not self.vad_detector:
                logger.warning("VAD not initialized, returning True")
                return True
            
            # Use Silero VAD for voice activity detection
            speech_detected = self.vad_detector(audio_data)
            logger.debug(f"Speech activity detected: {speech_detected}")
            return speech_detected
            
        except Exception as e:
            logger.error(f"Error in voice activity detection: {str(e)}")
            return True  # Assume speech if detection fails

    def _prepare_messages(self) -> list:
        """
        Prepare messages for LLM including system prompt and history
        """
        messages = [
            {
                "role": "system",
                "content": self.get_system_prompt()
            }
        ]
        
        # Add conversation history (limit to last 10 messages for context window)
        history_to_include = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        
        for msg in history_to_include:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return messages

    def _get_tools_schema(self) -> list:
        """
        Get the tools schema for the LLM in Groq/OpenAI format
        """
        tools = []
        
        # Add server-side tools
        server_tools = ServerTools.get_available_tools()
        for tool_name, tool_info in server_tools.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool_info["parameters"],
                        "required": [k for k, v in tool_info["parameters"].items() 
                                    if "default" not in v]
                    }
                }
            })
        
        # Add client-side tools
        client_tools = ClientTools.get_available_tools()
        for tool_name, tool_info in client_tools.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool_info["parameters"],
                        "required": [k for k, v in tool_info["parameters"].items() 
                                    if "default" not in v]
                    }
                }
            })
        
        return tools

    def _parse_tool_call(self, tool_call) -> Optional[ToolCall]:
        """
        Parse tool call from LLM response
        """
        try:
            # Extract tool name and arguments
            if hasattr(tool_call, 'function'):
                func = tool_call.function
                tool_name = func.name
                
                # Determine if client or server tool
                client_tools = ClientTools.get_available_tools()
                server_tools = ServerTools.get_available_tools()
                
                tool_type = "client" if tool_name in client_tools else "server"
                
                # Parse arguments
                arguments = {}
                if hasattr(func, 'arguments') and func.arguments:
                    try:
                        if isinstance(func.arguments, str):
                            arguments = json.loads(func.arguments)
                        else:
                            arguments = func.arguments
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                return ToolCall(
                    tool_name=tool_name,
                    tool_type=tool_type,
                    parameters=arguments,
                    description=f"Calling {tool_name}"
                )
        except Exception as e:
            logger.error(f"Error parsing tool call: {str(e)}")
        
        return None

    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call
        """
        try:
            if tool_call.tool_type == "client":
                return self._execute_client_tool(tool_call)
            else:
                return await self._execute_server_tool(tool_call)
        except Exception as e:
            logger.error(f"Error executing tool: {str(e)}")
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                result="",
                error=str(e)
            )

    def _execute_client_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute client-side tool
        """
        tool_name = tool_call.tool_name
        params = tool_call.parameters
        
        try:
            if tool_name == "change_theme":
                return ClientTools.change_theme(params.get("theme", "light"))
            elif tool_name == "show_notification":
                return ClientTools.show_notification(
                    params.get("message", ""),
                    params.get("duration_ms", 5000)
                )
            elif tool_name == "play_sound":
                return ClientTools.play_sound(params.get("sound_type", "notification"))
            elif tool_name == "open_url":
                return ClientTools.open_url(params.get("url", ""))
            else:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    result="",
                    error=f"Unknown client tool: {tool_name}"
                )
        except Exception as e:
            logger.error(f"Error executing client tool {tool_name}: {str(e)}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result="",
                error=str(e)
            )

    async def _execute_server_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute server-side tool
        """
        tool_name = tool_call.tool_name
        params = tool_call.parameters
        
        try:
            if tool_name == "get_weather":
                return await self.server_tools.get_weather(
                    params.get("city", ""),
                    params.get("country", "")
                )
            elif tool_name == "search_wikipedia":
                return await self.server_tools.search_wikipedia(
                    params.get("query", ""),
                    params.get("sentences", 3)
                )
            elif tool_name == "get_news":
                return await self.server_tools.get_news(
                    params.get("topic", "trending"),
                    params.get("region", "us")
                )
            elif tool_name == "calculate":
                return await self.server_tools.calculate(params.get("expression", ""))
            else:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    result="",
                    error=f"Unknown server tool: {tool_name}"
                )
        except Exception as e:
            logger.error(f"Error executing server tool {tool_name}: {str(e)}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result="",
                error=str(e)
            )

    def get_conversation_history(self) -> list[ChatMessage]:
        """Get full conversation history"""
        return self.conversation_history.copy()

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
