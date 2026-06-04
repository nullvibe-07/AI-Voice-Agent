from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from datetime import datetime

class ChatMessage(BaseModel):
    """Chat message model"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = None
    is_tool_call: bool = False
    tool_name: Optional[str] = None
    tool_result: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.timestamp is None:
            self.timestamp = datetime.now()

class StreamingTextChunk(BaseModel):
    """Streaming text chunk"""
    chunk: str
    timestamp: datetime = None
    token_count: int = 1

    def __init__(self, **data):
        super().__init__(**data)
        if self.timestamp is None:
            self.timestamp = datetime.now()

class StreamingAudioChunk(BaseModel):
    """Streaming audio chunk (base64 encoded)"""
    audio_data: str  # Base64 encoded audio
    timestamp: datetime = None
    duration_ms: int = 0

    def __init__(self, **data):
        super().__init__(**data)
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ToolCall(BaseModel):
    """Tool call request"""
    tool_name: str
    tool_type: str  # "client" or "server"
    parameters: Dict[str, Any]
    description: str

class ToolResult(BaseModel):
    """Tool call result"""
    tool_name: str
    success: bool
    result: str
    error: Optional[str] = None

class ConversationState(BaseModel):
    """Conversation state"""
    conversation_id: str
    room_name: str
    participant_name: str
    messages: List[ChatMessage] = []
    is_listening: bool = False
    is_speaking: bool = False
    current_transcript: str = ""
    connected: bool = False
    created_at: datetime = None
    updated_at: datetime = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

class UserInput(BaseModel):
    """User input (text or transcribed speech)"""
    text: str
    input_type: str  # "text" or "speech"
    conversation_id: str

class SessionMessage(BaseModel):
    """WebSocket message format"""
    type: str  # "user_input", "streaming_text", "streaming_audio", "tool_call", "status"
    data: Dict[str, Any]
    timestamp: datetime = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.timestamp is None:
            self.timestamp = datetime.now()