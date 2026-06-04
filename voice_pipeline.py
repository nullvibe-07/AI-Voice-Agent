"""Voice processing pipeline with streaming support"""

import asyncio
from typing import Optional, AsyncGenerator, Callable
from loguru import logger
from livekit.agents import (
    VoiceAssistantOptions,
    WorkerOptions,
    JobContext,
    llm,
    TurnEndedCallback,
)
from livekit.agents.openai import OpenAI, LLMOptions as OpenAILLMOptions
from livekit.agents.silero import VAD
from livekit.agents.deepgram import STT as DeepgramSTT
from livekit.agents.openai import TTS as OpenAITTS
from livekit import agents
import json
from models import ChatMessage, ToolCall, ToolResult
from tools.server_tools import ServerTools
from tools.client_tools import ClientTools
import re

class VoicePipeline:
    """Main voice processing pipeline"""

    def __init__(self):
        self.server_tools = ServerTools()
        self.client_tools = ClientTools()
        self.conversation_history: list[ChatMessage] = []
        self.is_processing = False

    async def initialize(self):
        """Initialize the voice pipeline"""
        await self.server_tools.initialize()
        logger.info("Voice pipeline initialized")

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
        ctx: Optional[JobContext] = None,
        on_text_chunk: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[ToolCall], None]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Process user input and stream response
        
        Args:
            user_text: User input text
            ctx: LiveKit job context
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
        Stream LLM response with tool calling support
        """
        from config import settings
        
        try:
            # Initialize OpenAI LLM
            llm_client = OpenAI(api_key=settings.openai_api_key)
            
            full_response = ""
            
            # Create message with tools
            messages_to_send = messages.copy()
            
            # Stream completion
            stream = await llm_client.astream_chat(
                model=settings.openai_model,
                messages=messages_to_send,
                tools=tools_schema if tools_schema else None,
                temperature=0.7,
                max_tokens=500,
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    
                    # Handle text content
                    if hasattr(delta, 'content') and delta.content:
                        text_chunk = delta.content
                        full_response += text_chunk
                        
                        # Emit text chunk
                        if on_text_chunk:
                            on_text_chunk(text_chunk)
                        
                        yield text_chunk
                    
                    # Handle tool calls
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            tool_info = self._parse_tool_call(tool_call, full_response)
                            if tool_info and on_tool_call:
                                on_tool_call(tool_info)
            
            # Add assistant response to history
            assistant_msg = ChatMessage(
                role="assistant",
                content=full_response
            )
            self.conversation_history.append(assistant_msg)
            
        except Exception as e:
            logger.error(f"Error in LLM streaming: {str(e)}")
            raise

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
        
        # Add conversation history
        for msg in self.conversation_history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return messages

    def _get_tools_schema(self) -> list:
        """
        Get the tools schema for the LLM
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

    def _parse_tool_call(self, tool_call, full_response: str) -> Optional[ToolCall]:
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
                        arguments = json.loads(func.arguments)
                    except json.JSONDecodeError:
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
