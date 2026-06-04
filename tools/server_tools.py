"""Server-side tools executed on the agent backend"""

import aiohttp
import asyncio
from typing import Any, Dict, Optional
from models import ToolResult
import wikipedia
import os
from loguru import logger

class ServerTools:
    """Server-side tool implementations"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        """Initialize async session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def cleanup(self):
        """Cleanup async session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def get_weather(self, city: str, country: str = "") -> ToolResult:
        """
        Fetch current weather for a city using Open-Meteo API (free, no API key needed)
        
        Args:
            city: City name
            country: Country code (optional)
            
        Returns:
            ToolResult with weather information
        """
        try:
            if not self.session:
                await self.initialize()

            # Use Open-Meteo Geocoding API to get coordinates
            query = f"{city}"
            if country:
                query += f", {country}"

            geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=1&language=en&format=json"
            
            async with self.session.get(geocoding_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return ToolResult(
                        tool_name="get_weather",
                        success=False,
                        result="",
                        error=f"Failed to find location for {city}"
                    )
                
                geo_data = await resp.json()
                
            if not geo_data.get("results"):
                return ToolResult(
                    tool_name="get_weather",
                    success=False,
                    result="",
                    error=f"No location found for {city}"
                )

            location = geo_data["results"][0]
            latitude = location["latitude"]
            longitude = location["longitude"]
            location_name = location.get("name", city)
            country_name = location.get("country", "")

            # Fetch weather data
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&timezone=auto"
            
            async with self.session.get(weather_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return ToolResult(
                        tool_name="get_weather",
                        success=False,
                        result="",
                        error="Failed to fetch weather data"
                    )
                
                weather_data = await resp.json()

            current = weather_data.get("current", {})
            timezone = weather_data.get("timezone", "UTC")

            # Parse weather code to description
            weather_code = current.get("weather_code", 0)
            weather_description = self._get_weather_description(weather_code)

            result = f"In {location_name}, {country_name}: {current.get('temperature_2m', 'N/A')}°C with {weather_description}. Humidity is {current.get('relative_humidity_2m', 'N/A')}% and wind speed is {current.get('wind_speed_10m', 'N/A')} km/h."

            logger.info(f"Weather fetched for {location_name}: {result}")

            return ToolResult(
                tool_name="get_weather",
                success=True,
                result=result
            )

        except asyncio.TimeoutError:
            return ToolResult(
                tool_name="get_weather",
                success=False,
                result="",
                error="Request timed out"
            )
        except Exception as e:
            logger.error(f"Error fetching weather: {str(e)}")
            return ToolResult(
                tool_name="get_weather",
                success=False,
                result="",
                error=f"Error fetching weather: {str(e)}"
            )

    async def search_wikipedia(self, query: str, sentences: int = 3) -> ToolResult:
        """
        Search Wikipedia and return a summary
        
        Args:
            query: Search query
            sentences: Number of sentences to return
            
        Returns:
            ToolResult with Wikipedia summary
        """
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._wikipedia_search_sync,
                query,
                sentences
            )
            return result
        except Exception as e:
            logger.error(f"Error searching Wikipedia: {str(e)}")
            return ToolResult(
                tool_name="search_wikipedia",
                success=False,
                result="",
                error=f"Error searching Wikipedia: {str(e)}"
            )

    def _wikipedia_search_sync(self, query: str, sentences: int) -> ToolResult:
        """Synchronous Wikipedia search"""
        try:
            # Search for the query
            results = wikipedia.search(query, results=3)
            
            if not results:
                return ToolResult(
                    tool_name="search_wikipedia",
                    success=False,
                    result="",
                    error=f"No Wikipedia results found for '{query}'"
                )

            # Get the first result
            try:
                page = wikipedia.page(results[0])
                summary = wikipedia.summary(results[0], sentences=sentences)
                
                result = f"According to Wikipedia: {summary}\n\nFor more information, visit: {page.url}"
                
                logger.info(f"Wikipedia search for '{query}' successful")
                
                return ToolResult(
                    tool_name="search_wikipedia",
                    success=True,
                    result=result
                )
            except wikipedia.exceptions.DisambiguationError as e:
                # Handle disambiguation
                options = str(e).split('\n')[1:4]  # Get first 3 options
                result = f"Multiple results found for '{query}'. Did you mean: {', '.join(options)}?"
                return ToolResult(
                    tool_name="search_wikipedia",
                    success=True,
                    result=result
                )

        except Exception as e:
            logger.error(f"Error in Wikipedia search: {str(e)}")
            return ToolResult(
                tool_name="search_wikipedia",
                success=False,
                result="",
                error=f"Error searching Wikipedia: {str(e)}"
            )

    async def get_news(self, topic: str = "trending", region: str = "us") -> ToolResult:
        """
        Fetch top news headlines (using NewsAPI or alternative free service)
        
        Args:
            topic: Topic to search (e.g., "technology", "sports", "trending")
            region: Region code (e.g., "us", "gb", "in")
            
        Returns:
            ToolResult with news headlines
        """
        try:
            if not self.session:
                await self.initialize()

            # Using NewsAPI Free (https://newsapi.org)
            # Note: You'll need to get a free API key from newsapi.org
            api_key = os.getenv("NEWS_API_KEY", "")
            
            if not api_key:
                # Alternative: Use a free news endpoint or provide generic response
                return ToolResult(
                    tool_name="get_news",
                    success=False,
                    result="",
                    error="News API key not configured. Please set NEWS_API_KEY environment variable."
                )

            url = f"https://newsapi.org/v2/top-headlines?category={topic}&country={region}&apiKey={api_key}"
            
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return ToolResult(
                        tool_name="get_news",
                        success=False,
                        result="",
                        error="Failed to fetch news"
                    )
                
                news_data = await resp.json()

            articles = news_data.get("articles", [])[:5]  # Get top 5
            
            if not articles:
                return ToolResult(
                    tool_name="get_news",
                    success=False,
                    result="",
                    error=f"No news found for topic '{topic}'"
                )

            headlines = []
            for i, article in enumerate(articles, 1):
                headlines.append(
                    f"{i}. {article.get('title', 'N/A')} - {article.get('source', {}).get('name', 'N/A')}"
                )

            result = f"Top headlines for {topic}:\n" + "\n".join(headlines)
            
            logger.info(f"News fetched for topic: {topic}")
            
            return ToolResult(
                tool_name="get_news",
                success=True,
                result=result
            )

        except asyncio.TimeoutError:
            return ToolResult(
                tool_name="get_news",
                success=False,
                result="",
                error="Request timed out"
            )
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return ToolResult(
                tool_name="get_news",
                success=False,
                result="",
                error=f"Error fetching news: {str(e)}"
            )

    async def calculate(self, expression: str) -> ToolResult:
        """
        Evaluate a mathematical expression safely
        
        Args:
            expression: Mathematical expression (e.g., "2 + 2 * 5")
            
        Returns:
            ToolResult with calculation result
        """
        try:
            # Whitelist allowed characters
            import re
            allowed_chars = set('0123456789+-*/(). ')
            
            if not all(c in allowed_chars for c in expression):
                return ToolResult(
                    tool_name="calculate",
                    success=False,
                    result="",
                    error="Invalid characters in expression. Only numbers and basic operators (+, -, *, /, ()) are allowed."
                )

            # Use eval safely with restricted globals
            result = eval(expression, {"__builtins__": {}}, {})
            
            # Format result
            if isinstance(result, float):
                # Round to 2 decimal places
                result = round(result, 2)
            
            logger.info(f"Calculation: {expression} = {result}")
            
            return ToolResult(
                tool_name="calculate",
                success=True,
                result=f"{expression} = {result}"
            )

        except SyntaxError:
            return ToolResult(
                tool_name="calculate",
                success=False,
                result="",
                error="Invalid mathematical expression. Please use correct syntax."
            )
        except ZeroDivisionError:
            return ToolResult(
                tool_name="calculate",
                success=False,
                result="",
                error="Division by zero is not allowed."
            )
        except Exception as e:
            logger.error(f"Error in calculation: {str(e)}")
            return ToolResult(
                tool_name="calculate",
                success=False,
                result="",
                error=f"Error evaluating expression: {str(e)}"
            )

    @staticmethod
    def _get_weather_description(code: int) -> str:
        """Convert WMO weather code to description"""
        weather_codes = {
            0: "clear sky",
            1: "mainly clear",
            2: "partly cloudy",
            3: "overcast",
            45: "foggy",
            48: "foggy with rime",
            51: "light drizzle",
            53: "moderate drizzle",
            55: "dense drizzle",
            61: "slight rain",
            63: "moderate rain",
            65: "heavy rain",
            71: "slight snow",
            73: "moderate snow",
            75: "heavy snow",
            77: "snow grains",
            80: "slight rain showers",
            81: "moderate rain showers",
            82: "violent rain showers",
            85: "slight snow showers",
            86: "heavy snow showers",
            95: "thunderstorm",
            96: "thunderstorm with slight hail",
            99: "thunderstorm with heavy hail"
        }
        return weather_codes.get(code, "unknown weather")

    @staticmethod
    def get_available_tools() -> Dict[str, Any]:
        """Get list of available server tools"""
        return {
            "get_weather": {
                "description": "Fetch current weather for a city",
                "parameters": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code (optional)",
                        "default": ""
                    }
                }
            },
            "search_wikipedia": {
                "description": "Search Wikipedia and return a summary",
                "parameters": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "sentences": {
                        "type": "integer",
                        "description": "Number of sentences to return",
                        "default": 3
                    }
                }
            },
            "get_news": {
                "description": "Fetch top news headlines",
                "parameters": {
                    "topic": {
                        "type": "string",
                        "description": "News topic",
                        "default": "trending",
                        "enum": ["business", "entertainment", "health", "science", "sports", "technology", "trending"]
                    },
                    "region": {
                        "type": "string",
                        "description": "Region code",
                        "default": "us",
                        "enum": ["us", "gb", "ca", "in", "au", "de", "fr"]
                    }
                }
            },
            "calculate": {
                "description": "Evaluate a mathematical expression",
                "parameters": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2 * 5')"
                    }
                }
            }
        }
