"""Main FastAPI application with WebSocket integration"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from typing import Dict, Set
from datetime import datetime
from loguru import logger

from config import settings
from models import (
    ChatMessage,
    ToolCall,
    ToolResult,
)
from voice_pipeline import VoicePipeline
from session_manager import SessionManager
from tools.client_tools import ClientTools
from tools.server_tools import ServerTools

# Initialize FastAPI app
app = FastAPI(
    title="Real-Time Voice AI Agent",
    description="A real-time conversational AI assistant with voice and text",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers and pipelines
session_manager = SessionManager()
voice_pipeline = VoicePipeline()

# WebSocket connections
active_connections: Dict[str, Set[WebSocket]] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    try:
        await voice_pipeline.initialize()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    try:
        await voice_pipeline.cleanup()
        logger.info("Application shutdown successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Real-Time Voice AI Agent Backend",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(session_manager.get_all_sessions())
    }

@app.get("/tools")
async def get_tools():
    """Get available tools"""
    return {
        "client_tools": ClientTools.get_available_tools(),
        "server_tools": ServerTools.get_available_tools()
    }

@app.post("/session/create")
async def create_session(
    room_name: str = Query(...),
    participant_name: str = Query(default="User")
):
    """Create a new conversation session"""
    try:
        state = session_manager.create_session(room_name, participant_name)
        return {
            "success": True,
            "conversation_id": state.conversation_id,
            "room_name": state.room_name,
            "participant_name": state.participant_name,
            "created_at": state.created_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{conversation_id}")
async def get_session(conversation_id: str):
    """Get session details"""
    session = session_manager.get_session(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "conversation_id": session.conversation_id,
        "room_name": session.room_name,
        "participant_name": session.participant_name,
        "connected": session.connected,
        "is_listening": session.is_listening,
        "is_speaking": session.is_speaking,
        "message_count": len(session.messages),
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat()
    }

@app.get("/session/{conversation_id}/messages")
async def get_session_messages(conversation_id: str):
    """Get all messages in a session"""
    session = session_manager.get_session(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "is_tool_call": msg.is_tool_call,
                "tool_name": msg.tool_name
            }
            for msg in session.messages
        ]
    }

@app.websocket("/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """
    WebSocket endpoint for real-time bidirectional communication
    
    Message types:
    - user_input: {type: "user_input", data: {text: str, input_type: "text"|"speech"}}
    - streaming_text: {type: "streaming_text", data: {chunk: str}}
    - streaming_audio: {type: "streaming_audio", data: {audio_data: base64}}
    - tool_call: {type: "tool_call", data: {tool_name: str, tool_type: str, parameters: {}}}
    - status: {type: "status", data: {status: str, message: str}}
    """
    
    # Verify session exists
    session = session_manager.get_session(conversation_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    # Accept connection
    await websocket.accept()
    logger.info(f"WebSocket connected for session {conversation_id}")
    
    # Add to active connections
    if conversation_id not in active_connections:
        active_connections[conversation_id] = set()
    active_connections[conversation_id].add(websocket)
    
    # Update session state
    session_manager.set_connected_state(conversation_id, True)
    
    # Send connection confirmation
    await websocket.send_json({
        "type": "status",
        "data": {
            "status": "connected",
            "message": "Connected to voice AI agent",
            "conversation_id": conversation_id
        }
    })
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            message_data = message.get("data", {})
            
            logger.debug(f"Received message: {message_type}")
            
            if message_type == "user_input":
                await handle_user_input(
                    websocket,
                    conversation_id,
                    message_data
                )
            elif message_type == "status_update":
                # Update session state
                if "is_listening" in message_data:
                    session_manager.set_listening_state(
                        conversation_id,
                        message_data["is_listening"]
                    )
                if "is_speaking" in message_data:
                    session_manager.set_speaking_state(
                        conversation_id,
                        message_data["is_speaking"]
                    )
                if "transcript" in message_data:
                    session_manager.update_transcript(
                        conversation_id,
                        message_data["transcript"]
                    )
            else:
                logger.warning(f"Unknown message type: {message_type}")
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {conversation_id}")
        if conversation_id in active_connections:
            active_connections[conversation_id].discard(websocket)
            if not active_connections[conversation_id]:
                del active_connections[conversation_id]
        session_manager.set_connected_state(conversation_id, False)
    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "status",
                "data": {
                    "status": "error",
                    "message": str(e)
                }
            })
        except:
            pass
    
    finally:
        # Clean up
        if conversation_id in active_connections:
            active_connections[conversation_id].discard(websocket)

async def handle_user_input(
    websocket: WebSocket,
    conversation_id: str,
    data: dict
):
    """
    Handle user input (text or transcribed speech)
    """
    try:
        user_text = data.get("text", "").strip()
        input_type = data.get("input_type", "text")
        
        if not user_text:
            await websocket.send_json({
                "type": "status",
                "data": {
                    "status": "error",
                    "message": "Empty input"
                }
            })
            return
        
        logger.info(f"Processing {input_type} input: {user_text}")
        
        # Add user message to session
        user_msg = ChatMessage(role="user", content=user_text)
        session_manager.add_message(conversation_id, user_msg)
        
        # Send user message to client
        await websocket.send_json({
            "type": "user_message",
            "data": {
                "content": user_text,
                "timestamp": user_msg.timestamp.isoformat()
            }
        })
        
        # Update session to indicate thinking
        await websocket.send_json({
            "type": "status",
            "data": {
                "status": "thinking",
                "message": "Processing your request..."
            }
        })
        
        # Collect full response
        full_response = ""
        text_chunks = []
        tool_calls = []
        
        # Process through voice pipeline
        async def on_text_chunk(chunk: str):
            nonlocal full_response
            full_response += chunk
            text_chunks.append(chunk)
            
            # Send text chunk to client immediately
            try:
                asyncio.create_task(
                    websocket.send_json({
                        "type": "streaming_text",
                        "data": {
                            "chunk": chunk,
                            "timestamp": datetime.now().isoformat()
                        }
                    })
                )
            except Exception as e:
                logger.error(f"Error sending text chunk: {str(e)}")
        
        async def on_tool_call(tool_call: ToolCall):
            tool_calls.append(tool_call)
            
            # Send tool call notification
            try:
                asyncio.create_task(
                    websocket.send_json({
                        "type": "tool_call",
                        "data": {
                            "tool_name": tool_call.tool_name,
                            "tool_type": tool_call.tool_type,
                            "description": tool_call.description,
                            "parameters": tool_call.parameters
                        }
                    })
                )
            except Exception as e:
                logger.error(f"Error sending tool call: {str(e)}")
        
        # Stream response from LLM
        async for chunk in voice_pipeline.process_user_input(
            user_text=user_text,
            on_text_chunk=on_text_chunk,
            on_tool_call=on_tool_call
        ):
            pass  # Text chunks are handled via callback
        
        # Execute tool calls if any
        for tool_call in tool_calls:
            try:
                logger.info(f"Executing tool: {tool_call.tool_name}")
                tool_result = await voice_pipeline.execute_tool_call(tool_call)
                
                # Send tool result
                await websocket.send_json({
                    "type": "tool_result",
                    "data": {
                        "tool_name": tool_result.tool_name,
                        "success": tool_result.success,
                        "result": tool_result.result,
                        "error": tool_result.error
                    }
                })
                
                logger.info(f"Tool executed: {tool_call.tool_name}")
            except Exception as e:
                logger.error(f"Error executing tool: {str(e)}")
                await websocket.send_json({
                    "type": "tool_result",
                    "data": {
                        "tool_name": tool_call.tool_name,
                        "success": False,
                        "error": str(e)
                    }
                })
        
        # Send completion signal
        await websocket.send_json({
            "type": "status",
            "data": {
                "status": "complete",
                "message": "Response complete"
            }
        })
        
        # Add assistant message to session
        assistant_msg = ChatMessage(role="assistant", content=full_response)
        session_manager.add_message(conversation_id, assistant_msg)
        
        logger.info(f"Response processed for session {conversation_id}")
    
    except Exception as e:
        logger.error(f"Error handling user input: {str(e)}")
        try:
            await websocket.send_json({
                "type": "status",
                "data": {
                    "status": "error",
                    "message": f"Error: {str(e)}"
                }
            })
        except:
            pass

@app.post("/text-input")
async def text_input(
    conversation_id: str = Query(...),
    text: str = Query(...)
):
    """
    HTTP endpoint for text input (alternative to WebSocket)
    Returns streaming response
    """
    session = session_manager.get_session(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Process input
        user_msg = ChatMessage(role="user", content=text)
        session_manager.add_message(conversation_id, user_msg)
        
        full_response = ""
        
        async def on_text_chunk(chunk: str):
            nonlocal full_response
            full_response += chunk
        
        # Stream response
        async for chunk in voice_pipeline.process_user_input(
            user_text=text,
            on_text_chunk=on_text_chunk
        ):
            pass
        
        # Add assistant message
        assistant_msg = ChatMessage(role="assistant", content=full_response)
        session_manager.add_message(conversation_id, assistant_msg)
        
        return {
            "success": True,
            "response": full_response,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error processing text input: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.debug,
        log_level="info"
    )
