"""Session management for user conversations"""

import uuid
from datetime import datetime
from typing import Dict, Optional
from models import ConversationState, ChatMessage
from loguru import logger

class SessionManager:
    """Manages user sessions and conversation states"""

    def __init__(self):
        self.sessions: Dict[str, ConversationState] = {}

    def create_session(
        self,
        room_name: str,
        participant_name: str
    ) -> ConversationState:
        """
        Create a new conversation session
        
        Args:
            room_name: LiveKit room name
            participant_name: Participant name
            
        Returns:
            New ConversationState
        """
        conversation_id = str(uuid.uuid4())
        
        state = ConversationState(
            conversation_id=conversation_id,
            room_name=room_name,
            participant_name=participant_name,
            messages=[],
            is_listening=False,
            is_speaking=False,
            current_transcript="",
            connected=False
        )
        
        self.sessions[conversation_id] = state
        logger.info(f"Session created: {conversation_id} for {participant_name}")
        
        return state

    def get_session(self, conversation_id: str) -> Optional[ConversationState]:
        """
        Get a conversation session by ID
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            ConversationState or None
        """
        return self.sessions.get(conversation_id)

    def update_session(self, conversation_id: str, state: ConversationState):
        """
        Update a conversation session
        
        Args:
            conversation_id: Conversation ID
            state: Updated ConversationState
        """
        if conversation_id in self.sessions:
            state.updated_at = datetime.now()
            self.sessions[conversation_id] = state
            logger.debug(f"Session updated: {conversation_id}")

    def add_message(
        self,
        conversation_id: str,
        message: ChatMessage
    ):
        """
        Add a message to a conversation
        
        Args:
            conversation_id: Conversation ID
            message: ChatMessage to add
        """
        session = self.get_session(conversation_id)
        if session:
            session.messages.append(message)
            session.updated_at = datetime.now()
            logger.debug(f"Message added to session {conversation_id}")

    def set_listening_state(self, conversation_id: str, is_listening: bool):
        """
        Update listening state
        
        Args:
            conversation_id: Conversation ID
            is_listening: Whether currently listening
        """
        session = self.get_session(conversation_id)
        if session:
            session.is_listening = is_listening
            session.updated_at = datetime.now()

    def set_speaking_state(self, conversation_id: str, is_speaking: bool):
        """
        Update speaking state
        
        Args:
            conversation_id: Conversation ID
            is_speaking: Whether currently speaking
        """
        session = self.get_session(conversation_id)
        if session:
            session.is_speaking = is_speaking
            session.updated_at = datetime.now()

    def update_transcript(self, conversation_id: str, transcript: str):
        """
        Update current transcript
        
        Args:
            conversation_id: Conversation ID
            transcript: Current transcript text
        """
        session = self.get_session(conversation_id)
        if session:
            session.current_transcript = transcript
            session.updated_at = datetime.now()

    def set_connected_state(self, conversation_id: str, connected: bool):
        """
        Update connection state
        
        Args:
            conversation_id: Conversation ID
            connected: Whether connected to LiveKit
        """
        session = self.get_session(conversation_id)
        if session:
            session.connected = connected
            session.updated_at = datetime.now()

    def end_session(self, conversation_id: str):
        """
        End a conversation session
        
        Args:
            conversation_id: Conversation ID
        """
        if conversation_id in self.sessions:
            del self.sessions[conversation_id]
            logger.info(f"Session ended: {conversation_id}")

    def get_all_sessions(self) -> Dict[str, ConversationState]:
        """
        Get all active sessions
        
        Returns:
            Dictionary of all sessions
        """
        return self.sessions.copy()
