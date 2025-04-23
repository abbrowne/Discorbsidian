#!/usr/bin/env python
"""
Text Analysis for Discord Logs

This module provides functionality to analyze Discord text logs using NLP techniques
to extract topics, generate summaries, and analyze sentiment.
"""

import logging
from typing import List, Dict, Any
import asyncio
import json
import re
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextAnalyzer:
    """
    Analyzes text content from Discord logs using OpenAI API.
    """
    
    def __init__(self, openai_client: OpenAI):
        """
        Initialize the text analyzer with OpenAI client.
        
        Args:
            openai_client: OpenAI client instance
        """
        self.openai_client = openai_client
    
    async def extract_topics(self, text: str, max_topics: int = 5) -> List[str]:
        """
        Extract main topics from the provided text.
        
        Args:
            text: The text to analyze for topics
            max_topics: Maximum number of topics to extract
            
        Returns:
            List of topics as strings
        """
        try:
            if not text or len(text.strip()) < 10:
                logger.warning("Text too short for topic extraction")
                return ["no content"]
                
            logger.info(f"Extracting topics from text ({len(text)} chars)")
            
            # Use OpenAI completion to extract topics
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"You are a topic extraction specialist. Extract the top {max_topics} topics or themes from the following text. Return ONLY a JSON array of topic strings, with no additional explanation. Keep each topic concise (1-4 words)."},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=150
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content
            topics_data = json.loads(content)
            
            # Extract topics from the response JSON
            if "topics" in topics_data and isinstance(topics_data["topics"], list):
                topics = topics_data["topics"]
            else:
                # Try to find any list in the response
                for key, value in topics_data.items():
                    if isinstance(value, list) and len(value) > 0:
                        topics = value
                        break
                else:
                    topics = []
            
            # Ensure we have topics and they're in the right format
            topics = [t for t in topics if isinstance(t, str)]
            topics = topics[:max_topics]  # Limit to max_topics
            
            if not topics:
                logger.warning("No topics extracted from text")
                return ["unknown"]
                
            logger.info(f"Extracted {len(topics)} topics: {', '.join(topics)}")
            return topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}", exc_info=True)
            return ["error"]
    
    async def extract_hierarchical_topics(self, text: str, max_topics: int = 5) -> Dict[str, List[str]]:
        """
        Extract hierarchical topics (main topics and subtopics) from the provided text.
        
        Args:
            text: The text to analyze for topics
            max_topics: Maximum number of main topics to extract
            
        Returns:
            Dictionary mapping main topics to lists of subtopics
        """
        try:
            if not text or len(text.strip()) < 50:
                logger.warning("Text too short for hierarchical topic extraction")
                return {"General": ["no content"]}
                
            logger.info(f"Extracting hierarchical topics from text ({len(text)} chars)")
            
            # Use OpenAI completion to extract hierarchical topics
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"""You are a topic extraction specialist. Extract hierarchical topics from the following text. 
                    Identify up to {max_topics} main topics and 2-4 subtopics for each main topic.
                    Return ONLY a JSON object with main topics as keys and arrays of subtopics as values.
                    Example format: {{"Technology": ["AI Ethics", "Software Development"], "Health": ["Nutrition", "Exercise"]}}
                    Keep topic names concise (1-4 words)."""},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=300
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content
            hierarchical_topics = json.loads(content)
            
            # Validate and clean up the hierarchical topics
            cleaned_topics = {}
            for main_topic, subtopics in hierarchical_topics.items():
                if isinstance(subtopics, list):
                    # Filter out non-string subtopics
                    valid_subtopics = [st for st in subtopics if isinstance(st, str)]
                    if valid_subtopics:
                        cleaned_topics[main_topic] = valid_subtopics
            
            if not cleaned_topics:
                logger.warning("No valid hierarchical topics extracted")
                return {"General": ["unknown"]}
                
            logger.info(f"Extracted {len(cleaned_topics)} main topics with subtopics")
            return cleaned_topics
            
        except Exception as e:
            logger.error(f"Error extracting hierarchical topics: {str(e)}", exc_info=True)
            return {"Error": ["failed to extract topics"]}
            
    async def generate_summary(self, text: str, max_length: int = 200) -> str:
        """
        Generate a concise summary of the provided text.
        
        Args:
            text: The text to summarize
            max_length: Maximum length of the summary in characters
            
        Returns:
            Summary as a string
        """
        try:
            if not text or len(text.strip()) < 50:
                logger.warning("Text too short for summarization")
                return "Content too brief for summary."
                
            logger.info(f"Generating summary for text ({len(text)} chars)")
            
            # Use OpenAI completion to generate summary
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"You are a summarization specialist. Create a concise summary of the following text in {max_length} characters or less. Be informative and capture the key points."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Ensure the summary isn't too long
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
                
            logger.info(f"Generated summary: {summary[:50]}...")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}", exc_info=True)
            return "Error generating summary."
    
    async def analyze_sentiment(self, text: str) -> str:
        """
        Analyze the sentiment of the provided text.
        
        Args:
            text: The text to analyze for sentiment
            
        Returns:
            Sentiment as a string (positive, negative, neutral)
        """
        try:
            if not text or len(text.strip()) < 10:
                logger.warning("Text too short for sentiment analysis")
                return "neutral"
                
            logger.info(f"Analyzing sentiment of text ({len(text)} chars)")
            
            # Use OpenAI completion to analyze sentiment
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis specialist. Analyze the sentiment of the following text and return ONLY one of these categories: 'positive', 'negative', 'neutral', or 'mixed'. Return just the single word, nothing else."},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            sentiment = response.choices[0].message.content.strip().lower()
            
            # Ensure we return one of our expected categories
            valid_sentiments = ["positive", "negative", "neutral", "mixed"]
            if sentiment not in valid_sentiments:
                # Match the closest sentiment if not an exact match
                for valid in valid_sentiments:
                    if valid in sentiment:
                        sentiment = valid
                        break
                else:
                    sentiment = "neutral"  # Default fallback
            
            logger.info(f"Analyzed sentiment: {sentiment}")
            return sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}", exc_info=True)
            return "neutral"

# Module-level convenience functions that will be imported elsewhere
async def extract_topics(text: str, max_topics: int = 5) -> List[str]:
    """
    Stand-alone function to extract topics from text.
    This is a placeholder that should be implemented with access to an OpenAI client.
    """
    logger.warning("Using unimplemented extract_topics function")
    return ["placeholder"]

async def generate_summary(text: str, max_length: int = 200) -> str:
    """
    Stand-alone function to generate a summary of text.
    This is a placeholder that should be implemented with access to an OpenAI client.
    """
    logger.warning("Using unimplemented generate_summary function")
    return "This is a placeholder summary."

async def analyze_sentiment(text: str) -> str:
    """
    Stand-alone function to analyze sentiment of text.
    This is a placeholder that should be implemented with access to an OpenAI client.
    """
    logger.warning("Using unimplemented analyze_sentiment function")
    return "neutral" 