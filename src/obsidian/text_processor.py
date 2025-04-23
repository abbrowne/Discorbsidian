#!/usr/bin/env python
"""
Obsidian Text Processor for Discord Logs

This module provides functionality to process Discord text logs that have been 
saved to an Obsidian vault. It enhances organization, adds metadata, creates 
indexes by channel and user, and improves searchability without requiring 
re-extraction from Discord.
"""

import os
import re
import logging
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ObsidianTextProcessor:
    """
    Processes Discord text logs in Obsidian vault to enhance organization,
    add metadata, and create indexes by channel and user.
    """
    
    def __init__(self, vault_path: str):
        """
        Initialize the text processor with the path to the Obsidian vault.
        
        Args:
            vault_path: The path to the Obsidian vault root directory
        """
        self.vault_path = Path(vault_path)
        self.daily_notes_dir = self.vault_path / "Daily Notes"
        self.channels_dir = self.vault_path / "Discord" / "Channels"
        self.users_dir = self.vault_path / "Discord" / "Users"
        self.voice_dir = self.vault_path / "Discord" / "Voice Transcriptions"
        
        # Stats tracking
        self.total_files_processed = 0
        self.total_messages_processed = 0
        
        # Patterns for extracting information from text logs
        self.message_pattern = re.compile(r'^\[(\d{2}:\d{2}:\d{2})\] \*\*(.+?)\*\*: (.+)$', re.MULTILINE)
        self.channel_header_pattern = re.compile(r'^## #(.+?)(?: \((\d+)\))?$', re.MULTILINE)
        
        # Data structures to store processed information
        self.channel_messages = defaultdict(lambda: defaultdict(list))  # {channel: {date: [messages]}}
        self.user_messages = defaultdict(lambda: defaultdict(list))     # {user: {date: [messages]}}
        
        # Ensure necessary directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in [
            self.channels_dir,
            self.users_dir,
            self.voice_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def process_all_logs(self) -> Tuple[int, int]:
        """
        Process all text logs in the Obsidian vault's Daily Notes directory.
        
        Returns:
            Tuple of (files_processed, messages_processed)
        """
        # Reset stats and data
        self.total_files_processed = 0
        self.total_messages_processed = 0
        self.channel_messages.clear()
        self.user_messages.clear()
        
        # Process daily logs
        daily_log_files = sorted(glob.glob(os.path.join(self.daily_notes_dir, "*.md")))
        
        if not daily_log_files:
            logger.warning(f"No log files found in {self.daily_notes_dir}")
            return 0, 0
        
        logger.info(f"Found {len(daily_log_files)} daily log files to process")
        
        # Process each daily log file
        for file_path in daily_log_files:
            self._process_file(file_path)
        
        # Process voice transcription files if they exist
        self._process_voice_transcriptions()
        
        # Generate channel-specific files
        logger.info("Generating channel files")
        for channel, date_messages in self.channel_messages.items():
            for date, messages in date_messages.items():
                self._create_channel_date_file(channel, date, messages)
        
        # Generate user-specific files
        logger.info("Generating user files")
        for user, date_messages in self.user_messages.items():
            self._create_user_file(user, date_messages)
        
        # Create indexes
        logger.info("Creating channel and user indexes")
        self._create_channel_index()
        self._create_user_index()
        
        return self.total_files_processed, self.total_messages_processed
    
    def _process_file(self, file_path: str):
        """
        Process a single daily log file, extracting messages and organizing them by channel and user.
        
        Args:
            file_path: Path to the daily log file
        """
        file_date = os.path.splitext(os.path.basename(file_path))[0]
        try:
            datetime.strptime(file_date, "%Y-%m-%d")  # Validate date format
        except ValueError:
            logger.warning(f"Skipping file with invalid date format: {file_path}")
            return
        
        logger.info(f"Processing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract channel sections
            current_channel = None
            current_channel_id = None
            channel_contents = {}
            
            for line in content.split('\n'):
                # Check for channel headers
                channel_match = self.channel_header_pattern.match(line)
                if channel_match:
                    current_channel = channel_match.group(1).strip()
                    current_channel_id = channel_match.group(2) if channel_match.group(2) else None
                    if current_channel not in channel_contents:
                        channel_contents[current_channel] = []
                    continue
                
                # Skip if we haven't found a channel yet
                if not current_channel:
                    continue
                
                # Add line to current channel's content
                channel_contents[current_channel].append(line)
            
            # Process messages in each channel
            for channel, lines in channel_contents.items():
                channel_content = '\n'.join(lines)
                message_matches = self.message_pattern.finditer(channel_content)
                
                for match in message_matches:
                    timestamp, username, message_text = match.groups()
                    
                    # Add to channel messages
                    self.channel_messages[channel][file_date].append({
                        'timestamp': timestamp,
                        'username': username,
                        'text': message_text
                    })
                    
                    # Add to user messages
                    self.user_messages[username][file_date].append({
                        'timestamp': timestamp,
                        'channel': channel,
                        'text': message_text
                    })
                    
                    self.total_messages_processed += 1
            
            self.total_files_processed += 1
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
    
    def _create_channel_date_file(self, channel: str, date: str, messages: List[Dict]):
        """
        Create a structured file for a specific channel and date.
        
        Args:
            channel: Channel name
            date: Date in YYYY-MM-DD format
            messages: List of message dictionaries
        """
        # Skip if no messages
        if not messages:
            return
        
        # Prepare path
        safe_channel = self._safe_filename(channel)
        channel_dir = self.channels_dir / safe_channel
        channel_dir.mkdir(exist_ok=True)
        
        file_path = channel_dir / f"{date}.md"
        
        # Calculate stats
        message_count = len(messages)
        word_count = sum(len(msg['text'].split()) for msg in messages)
        users = set(msg['username'] for msg in messages)
        
        # Create frontmatter
        frontmatter = {
            'date': date,
            'channel': channel,
            'message_count': message_count,
            'word_count': word_count,
            'users': sorted(list(users))
        }
        
        # Format messages
        formatted_messages = []
        for msg in messages:
            formatted_messages.append(f"[{msg['timestamp']}] **{msg['username']}**: {msg['text']}")
        
        # Write file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("---\n")
                f.write(yaml.dump(frontmatter, default_flow_style=False))
                f.write("---\n\n")
                f.write(f"# #{channel} - {date}\n\n")
                f.write('\n'.join(formatted_messages))
        except Exception as e:
            logger.error(f"Error writing channel file {file_path}: {str(e)}")
    
    def _create_user_file(self, username: str, date_messages: Dict[str, List[Dict]]):
        """
        Create a structured file for a specific user with all their messages.
        
        Args:
            username: Username
            date_messages: Dictionary of {date: messages}
        """
        # Skip if no messages
        if not date_messages:
            return
        
        # Prepare path
        safe_username = self._safe_filename(username)
        file_path = self.users_dir / f"{safe_username}.md"
        
        # Calculate stats
        total_messages = sum(len(msgs) for msgs in date_messages.values())
        total_words = sum(sum(len(msg['text'].split()) for msg in msgs) for msgs in date_messages.values())
        channels = set()
        for msgs in date_messages.values():
            for msg in msgs:
                channels.add(msg['channel'])
        
        # Create frontmatter
        frontmatter = {
            'username': username,
            'total_messages': total_messages,
            'total_words': total_words,
            'channels': sorted(list(channels)),
            'dates': sorted(date_messages.keys())
        }
        
        # Write file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("---\n")
                f.write(yaml.dump(frontmatter, default_flow_style=False))
                f.write("---\n\n")
                f.write(f"# User: {username}\n\n")
                
                # Write summary
                f.write(f"## Summary\n\n")
                f.write(f"- Total Messages: {total_messages}\n")
                f.write(f"- Total Words: {total_words}\n")
                f.write(f"- Active in {len(channels)} channels\n")
                f.write(f"- Messages on {len(date_messages)} days\n\n")
                
                # Write messages by date
                for date in sorted(date_messages.keys(), reverse=True):
                    f.write(f"## {date}\n\n")
                    for msg in date_messages[date]:
                        f.write(f"[{msg['timestamp']}] **#{msg['channel']}**: {msg['text']}\n")
                    f.write("\n")
        except Exception as e:
            logger.error(f"Error writing user file {file_path}: {str(e)}")
    
    def _create_channel_index(self):
        """Create an index file for all channels"""
        file_path = self.channels_dir / "00-Channel-Index.md"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# Discord Channel Index\n\n")
                
                # Summary stats
                total_channels = len(self.channel_messages)
                total_messages = sum(sum(len(msgs) for msgs in dates.values()) for dates in self.channel_messages.values())
                
                f.write(f"## Summary\n\n")
                f.write(f"- Total Channels: {total_channels}\n")
                f.write(f"- Total Messages: {total_messages}\n\n")
                
                f.write("## Channels\n\n")
                
                # List all channels with stats
                for channel in sorted(self.channel_messages.keys()):
                    channel_messages = sum(len(msgs) for msgs in self.channel_messages[channel].values())
                    dates = sorted(self.channel_messages[channel].keys())
                    
                    safe_channel = self._safe_filename(channel)
                    channel_link = f"[[{safe_channel}]]"
                    
                    f.write(f"- {channel_link} - {channel_messages} messages, active from {dates[0]} to {dates[-1]}\n")
        except Exception as e:
            logger.error(f"Error writing channel index: {str(e)}")
    
    def _create_user_index(self):
        """Create an index file for all users"""
        file_path = self.users_dir / "00-User-Index.md"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# Discord User Index\n\n")
                
                # Summary stats
                total_users = len(self.user_messages)
                total_messages = sum(sum(len(msgs) for msgs in dates.values()) for dates in self.user_messages.values())
                
                f.write(f"## Summary\n\n")
                f.write(f"- Total Users: {total_users}\n")
                f.write(f"- Total Messages: {total_messages}\n\n")
                
                f.write("## Users\n\n")
                
                # List all users with stats
                for username in sorted(self.user_messages.keys()):
                    user_messages = sum(len(msgs) for msgs in self.user_messages[username].values())
                    
                    safe_username = self._safe_filename(username)
                    user_link = f"[[{safe_username}]]"
                    
                    f.write(f"- {user_link} - {user_messages} messages\n")
        except Exception as e:
            logger.error(f"Error writing user index: {str(e)}")
    
    def _process_voice_transcriptions(self):
        """Process voice transcription files if they exist"""
        voice_files = glob.glob(os.path.join(self.voice_dir, "*.md"))
        
        if not voice_files:
            logger.info("No voice transcription files found")
            return
        
        logger.info(f"Processing {len(voice_files)} voice transcription files")
        
        # Process each voice file
        # This is a placeholder - you would implement specific processing for voice files here
        for file_path in voice_files:
            try:
                # Example: Extract metadata from voice files
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add processing logic specific to voice transcription files here
                self.total_files_processed += 1
            except Exception as e:
                logger.error(f"Error processing voice file {file_path}: {str(e)}")
    
    @staticmethod
    def _safe_filename(name: str) -> str:
        """
        Convert a name to a safe filename.
        
        Args:
            name: Original name that might contain invalid characters
            
        Returns:
            Safe filename
        """
        # Replace invalid filename characters
        safe = re.sub(r'[\\/*?:"<>|]', '_', name)
        # Replace spaces with dashes
        safe = safe.replace(' ', '-')
        return safe


def process_obsidian_logs(vault_path: str) -> Tuple[int, int]:
    """
    Process Discord logs in an Obsidian vault.
    
    Args:
        vault_path: Path to the Obsidian vault
        
    Returns:
        Tuple of (files_processed, messages_processed)
    """
    processor = ObsidianTextProcessor(vault_path)
    files_processed, messages_processed = processor.process_all_logs()
    logger.info(f"Processed {files_processed} files with {messages_processed} messages")
    return files_processed, messages_processed

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python text_processor.py <path_to_obsidian_vault>")
        sys.exit(1)
    
    vault_path = sys.argv[1]
    process_obsidian_logs(vault_path) 