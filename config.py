from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # LiveKit Configuration
    livekit_url: str = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    livekit_api_key: str = os.getenv("LIVEKIT_API_KEY", "")
    livekit_api_secret: str = os.getenv("LIVEKIT_API_SECRET", "")
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-4o"
    
    # Deepgram Configuration
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    
    # Server Configuration
    server_host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port: int = int(os.getenv("SERVER_PORT", 8000))
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Whisper Configuration
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")
    
    # Redis Configuration
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", 6379))
    redis_db: int = int(os.getenv("REDIS_DB", 0))
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()