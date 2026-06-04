"""Client-side tools executed in the browser via JavaScript"""

from typing import Any, Dict
from models import ToolResult

class ClientTools:
    """Client-side tool implementations"""

    @staticmethod
    def change_theme(theme: str) -> ToolResult:
        """
        Change UI theme between light and dark mode
        
        Args:
            theme: "light" or "dark"
            
        Returns:
            ToolResult indicating success
        """
        valid_themes = ["light", "dark"]
        if theme.lower() not in valid_themes:
            return ToolResult(
                tool_name="change_theme",
                success=False,
                result="",
                error=f"Invalid theme. Must be one of {valid_themes}"
            )
        
        return ToolResult(
            tool_name="change_theme",
            success=True,
            result=f"Theme changed to {theme} mode"
        )

    @staticmethod
    def show_notification(message: str, duration_ms: int = 5000) -> ToolResult:
        """
        Display a toast notification in the browser
        
        Args:
            message: Notification message text
            duration_ms: Duration to show notification in milliseconds
            
        Returns:
            ToolResult indicating success
        """
        if not message or len(message.strip()) == 0:
            return ToolResult(
                tool_name="show_notification",
                success=False,
                result="",
                error="Message cannot be empty"
            )
        
        return ToolResult(
            tool_name="show_notification",
            success=True,
            result=f"Notification displayed: {message}"
        )

    @staticmethod
    def play_sound(sound_type: str) -> ToolResult:
        """
        Play a sound effect in the browser
        
        Args:
            sound_type: Type of sound ("success", "error", "notification", "chime")
            
        Returns:
            ToolResult indicating success
        """
        valid_sounds = ["success", "error", "notification", "chime"]
        if sound_type.lower() not in valid_sounds:
            return ToolResult(
                tool_name="play_sound",
                success=False,
                result="",
                error=f"Invalid sound type. Must be one of {valid_sounds}"
            )
        
        return ToolResult(
            tool_name="play_sound",
            success=True,
            result=f"Playing {sound_type} sound"
        )

    @staticmethod
    def open_url(url: str) -> ToolResult:
        """
        Open a URL in a new browser tab
        
        Args:
            url: URL to open
            
        Returns:
            ToolResult indicating success
        """
        if not url or len(url.strip()) == 0:
            return ToolResult(
                tool_name="open_url",
                success=False,
                result="",
                error="URL cannot be empty"
            )
        
        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        return ToolResult(
            tool_name="open_url",
            success=True,
            result=f"Opening URL: {url}"
        )

    @staticmethod
    def get_available_tools() -> Dict[str, Any]:
        """Get list of available client tools"""
        return {
            "change_theme": {
                "description": "Change UI theme between light and dark mode",
                "parameters": {
                    "theme": {
                        "type": "string",
                        "description": "Theme name: 'light' or 'dark'",
                        "enum": ["light", "dark"]
                    }
                }
            },
            "show_notification": {
                "description": "Display a toast notification in the browser",
                "parameters": {
                    "message": {
                        "type": "string",
                        "description": "Notification message text"
                    },
                    "duration_ms": {
                        "type": "integer",
                        "description": "Duration to show notification in milliseconds",
                        "default": 5000
                    }
                }
            },
            "play_sound": {
                "description": "Play a sound effect in the browser",
                "parameters": {
                    "sound_type": {
                        "type": "string",
                        "description": "Type of sound to play",
                        "enum": ["success", "error", "notification", "chime"]
                    }
                }
            },
            "open_url": {
                "description": "Open a URL in a new browser tab",
                "parameters": {
                    "url": {
                        "type": "string",
                        "description": "URL to open"
                    }
                }
            }
        }
