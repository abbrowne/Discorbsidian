import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Set, Optional
import discord
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI
import logging

# Add project root to Python path before any imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

from src.obsidian.vault import ObsidianVault
from src.utils.helpers import format_discord_message, create_obsidian_note_content, get_daily_note_path
from src.rag.engine import RAGEngine
from src.bot.voice_handler import VoiceHandler

# Load environment variables
load_dotenv()
discord_token = os.getenv("DISCORD_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")
vault_path = os.getenv("OBSIDIAN_VAULT_PATH")

# Initialize Discord bot
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.voice_states = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Get allowed channel IDs from environment
# Format: DISCORD_CHANNEL_IDS=123456789,987654321
ALLOWED_CHANNEL_IDS = [int(id.strip()) for id in os.getenv('DISCORD_CHANNEL_IDS', '').split(',') if id.strip()]

# Load voice channel IDs from environment
voice_channel_ids_str = os.getenv("DISCORD_VOICE_CHANNEL_IDS", "")
MONITORED_VOICE_CHANNELS = set()
if voice_channel_ids_str:
    try:
        MONITORED_VOICE_CHANNELS = {int(channel_id.strip()) for channel_id in voice_channel_ids_str.split(",")}
        print(f"Loaded monitored voice channels: {MONITORED_VOICE_CHANNELS}")
    except ValueError as e:
        print(f"Error parsing voice channel IDs: {e}")

# Initialize OpenAI client
openai_client = OpenAI(api_key=openai_api_key)

# Initialize Obsidian vault
obsidian_vault = ObsidianVault(vault_path)

# Initialize voice handler with OpenAI integration
voice_handler = VoiceHandler(bot, obsidian_vault, openai_client)

# Store messages by date and channel
messages_by_date_and_channel: Dict[str, Dict[str, List[discord.Message]]] = defaultdict(lambda: defaultdict(list))

def create_concise_message_content(message):
    """Create a concise formatted message content"""
    timestamp = message.created_at.strftime('%H:%M:%S')
    author = str(message.author)
    
    # Start with timestamp and author
    content = f"**[{timestamp}] {author}**:\n"
    
    # Add message content
    if message.content:
        content += f"{message.content}\n"
    
    # Add attachments if any
    if message.attachments:
        content += "Attachments:\n"
        for attachment in message.attachments:
            content += f"- {attachment.url}\n"
    
    # Add embeds if any
    if message.embeds:
        content += "Embeds:\n"
        for embed in message.embeds:
            if embed.title:
                content += f"- {embed.title}\n"
            if embed.description:
                content += f"  {embed.description}\n"
    
    return content

async def save_messages_to_vault(date_str):
    """Save all messages for a specific date to the Obsidian vault"""
    if not messages_by_date_and_channel[date_str]:
        return
    
    # Create a folder for this date
    date_folder = f"Daily Notes/{date_str}"
    
    # Process each channel
    for channel_name, messages in messages_by_date_and_channel[date_str].items():
        if not messages:
            continue
            
        # Create content for new messages in this channel
        new_content = ""
        
        # Sort messages by timestamp
        sorted_messages = sorted(messages, key=lambda x: x.created_at)
        
        for message in sorted_messages:
            new_content += create_concise_message_content(message)
            new_content += "\n---\n\n"  # Add separator between messages
        
        # Sanitize channel name for use as filename
        safe_channel_name = channel_name.replace('/', '_').replace('\\', '_')
        note_path = obsidian_vault.vault_path / date_folder / f"{safe_channel_name}.md"
        
        try:
            # Check if the note already exists
            if note_path.exists():
                # Read existing content
                with open(note_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                
                # Append new content
                content = existing_content.rstrip() + "\n\n" + new_content
            else:
                # Create new note with header
                content = f"# {channel_name} - {date_str}\n\n## Messages\n\n" + new_content
            
            # Ensure the directory exists
            note_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the content
            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            print(f"Saved messages for {date_str} in channel {channel_name} to: {note_path}")
        except Exception as e:
            print(f"Error saving messages to vault: {e}")
    
    # Clear the messages for this date after saving
    messages_by_date_and_channel[date_str] = defaultdict(list)

async def process_message(message):
    """Process a single message and add it to the messages_by_date_and_channel dictionary"""
    # Log the message
    timestamp = message.created_at.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] {message.author}: {message.content}')
    
    # Get the date string and channel name for this message
    date_str = message.created_at.strftime('%Y-%m-%d')
    channel_name = str(message.channel.name)
    
    # Add message to the appropriate date and channel group
    messages_by_date_and_channel[date_str][channel_name].append(message)
    
    # If there are attachments, log them
    if message.attachments:
        print('Attachments:')
        for attachment in message.attachments:
            print(f'- {attachment.url}')

async def fetch_channel_history(channel, days=1):
    """Fetch message history from the channel for the specified number of days"""
    print(f"\nFetching message history for channel {channel.name} for the last {days} day(s)...")
    
    # Calculate the time threshold
    time_threshold = datetime.utcnow() - timedelta(days=days)
    
    # Fetch messages
    async for message in channel.history(limit=None, after=time_threshold):
        await process_message(message)
    
    print(f"Finished fetching message history for channel {channel.name}.")

async def fetch_all_channels_history(days=1):
    """Fetch message history from all allowed channels for the specified number of days"""
    print(f"\nFetching message history from all channels for the last {days} day(s)...")
    
    # Fetch messages from each channel
    for channel_id in ALLOWED_CHANNEL_IDS:
        channel = bot.get_channel(channel_id)
        if channel:
            await fetch_channel_history(channel, days)
        else:
            print(f"Warning: Could not find channel with ID {channel_id}")
    
    # Save messages for each day
    for date_str in sorted(messages_by_date_and_channel.keys()):
        await save_messages_to_vault(date_str)
    
    print("Finished fetching and processing message history from all channels.")

@bot.event
async def on_ready():
    print(f"{bot.user} is now running!")
    print(f"Connected to {len(bot.guilds)} guilds")
    print(f"Available voice channels: {MONITORED_VOICE_CHANNELS}")
    
    # Initialize voice recorder
    global voice_handler
    voice_handler = voice_handler
    print("Voice recorder initialized")
    print("To start recording, use !record or !monitor_voice in a text channel")
    
    # Fetch message history when bot starts
    await fetch_all_channels_history()

@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Only process messages from the allowed channels
    if message.channel.id in ALLOWED_CHANNEL_IDS:
        await process_message(message)
        
        # Save messages to vault after processing
        date_str = message.created_at.strftime('%Y-%m-%d')
        await save_messages_to_vault(date_str)

@bot.event
async def on_voice_state_update(member, before, after):
    """Handle voice state changes, such as users joining or leaving voice channels"""
    # Ignore bot's own voice state changes
    if member.id == bot.user.id:
        return
    
    try:
        # Forward the event to the voice handler
        await voice_handler.on_voice_state_update(member, before, after)
        
        # Check if the user joined a monitored channel
        if after and after.channel and after.channel.id in MONITORED_VOICE_CHANNELS:
            print(f"Member {member.name} joined monitored voice channel {after.channel.name}")
            # If the channel is not already being monitored, start monitoring
            if after.channel.id not in voice_handler.monitored_channels:
                print(f"Starting monitoring for channel {after.channel.name} as member joined")
                await voice_handler.monitor_voice_channel(after.channel.id)
            else:
                print(f"Channel {after.channel.name} already being monitored")
        
        # Check if the user left a monitored channel
        if before and before.channel and before.channel.id in MONITORED_VOICE_CHANNELS:
            print(f"Member {member.name} left monitored voice channel {before.channel.name}")
            # If there are no more members in the voice channel (except the bot), stop monitoring
            remaining_members = sum(1 for m in before.channel.members if m.id != bot.user.id)
            if remaining_members == 0:
                print(f"No members left in {before.channel.name}, stopping monitoring")
                await voice_handler.stop_monitoring(before.channel.id)
                print(f"Stopped monitoring voice channel: {before.channel.name}")
    except Exception as e:
        print(f"Error handling voice state update: {e}")
        logging.error(f"Error in on_voice_state_update: {e}", exc_info=True)

@bot.command(name='record')
async def record(ctx, channel_id: Optional[int] = None, chunk_duration: int = 300):
    """Start recording a voice channel
    
    Args:
        channel_id (int, optional): The ID of the voice channel to monitor. If not provided,
            attempts to use the channel the user is currently in.
        chunk_duration (int, optional): Duration in seconds between automatic audio chunk saves.
            Default is 300 seconds (5 minutes).
    """
    # Validate chunk duration
    if chunk_duration < 10:
        await ctx.send("Chunk duration must be at least 10 seconds.")
        return
        
    # If no channel ID specified, use the user's current voice channel
    if channel_id is None and ctx.author.voice:
        channel_id = ctx.author.voice.channel.id
    elif channel_id is None:
        await ctx.send("Please join a voice channel or specify a channel ID.")
        return
        
    # Check if the specified channel exists and is a voice channel
    channel = bot.get_channel(channel_id)
    if not channel or not isinstance(channel, discord.VoiceChannel):
        await ctx.send("Invalid voice channel ID.")
        return
    
    # Format duration for display
    if chunk_duration >= 3600:
        display_time = f"{chunk_duration / 3600:.1f} hours"
    elif chunk_duration >= 60:
        display_time = f"{chunk_duration / 60:.1f} minutes"
    else:
        display_time = f"{chunk_duration} seconds"
        
    # Start monitoring the channel
    success = await voice_handler.monitor_voice_channel(channel_id, chunk_duration=chunk_duration)
    if success:
        MONITORED_VOICE_CHANNELS.add(channel_id)
        await ctx.send(f"Started recording in {channel.name} with chunk duration of {display_time}. Audio will be transcribed when users leave the channel or chunks are saved.")
    else:
        await ctx.send(f"Failed to start recording in {channel.name}. Check logs for details.")

@bot.command(name='stop_record')
async def stop_record(ctx, channel_id: Optional[int] = None):
    """Stop recording a voice channel"""
    # If no channel ID specified, use currently monitored channel
    if channel_id is None:
        monitored = voice_handler.monitored_channels
        if not monitored:
            await ctx.send("Not currently monitoring any voice channels.")
            return
        channel_id = next(iter(monitored))
        
    # Stop the recording
    if channel_id in voice_handler.monitored_channels:
        user_files = await voice_handler.stop_monitoring(channel_id)
        channel = bot.get_channel(channel_id)
        channel_name = channel.name if channel else f"channel-{channel_id}"
        
        if channel_id in MONITORED_VOICE_CHANNELS:
            MONITORED_VOICE_CHANNELS.remove(channel_id)
            
        if user_files:
            await ctx.send(f"Stopped recording in {channel_name}. Processing {sum(len(files) for files in user_files.values())} audio files.")
        else:
            await ctx.send(f"Stopped recording in {channel_name}. No audio files were recorded.")
    else:
        await ctx.send(f"Not currently recording channel {channel_id}.")

@bot.command(name='recording_status')
async def recording_status(ctx):
    """Check the current recording status"""
    if hasattr(voice_handler, 'monitored_channels'):
        monitored = voice_handler.monitored_channels
        if not monitored:
            await ctx.send("Not currently recording any voice channels.")
        else:
            channel_list = []
            for channel_id in monitored:
                channel = bot.get_channel(channel_id)
                channel_name = channel.name if channel else f"channel-{channel_id}"
                channel_list.append(f"- {channel_name} (ID: {channel_id})")
            
            await ctx.send(f"Currently recording {len(monitored)} voice channel(s):\n" + "\n".join(channel_list))
    else:
        await ctx.send("Voice recording is not supported.")

@bot.command(name='history')
async def fetch_history(ctx, days: int = 1):
    """Command to fetch message history for a specific number of days"""
    if ctx.channel.id not in ALLOWED_CHANNEL_IDS:
        return
        
    await fetch_all_channels_history(days)

@bot.command(name='monitor_voice')
async def monitor_voice_command(ctx, channel_id=None, chunk_duration=300):
    """Monitor a voice channel for user presence.
    
    This command will have the bot join a voice channel and track when users join or leave.
    Note: This implementation tracks user presence only and does not capture actual audio.
    For audio capture functionality, a specialized library like discord-ext-voice-recv would be needed.
    
    Args:
        channel_id (str, optional): The ID of the voice channel to monitor. If not provided,
            attempts to use the channel the user is currently in.
        chunk_duration (int, optional): Duration in seconds between automatic audio chunk saves.
            Default is 300 seconds (5 minutes).
    """
    try:
        # Convert chunk_duration to int if provided as string
        if isinstance(chunk_duration, str):
            chunk_duration = int(chunk_duration)
            
        # Validate chunk duration
        if chunk_duration < 10:
            await ctx.send("Chunk duration must be at least 10 seconds.")
            return
            
        # If no channel ID provided, use the channel the message author is in
        if channel_id is None:
            if ctx.author.voice and ctx.author.voice.channel:
                channel = ctx.author.voice.channel
                channel_id = channel.id
            else:
                await ctx.send("You're not in a voice channel. Please join one or specify a channel ID.")
                return
        else:
            # Convert to int if a string ID was provided
            channel_id = int(channel_id)
            channel = bot.get_channel(channel_id)
            
            if not channel:
                await ctx.send(f"Could not find channel with ID {channel_id}")
                return
            
            if not isinstance(channel, discord.VoiceChannel):
                await ctx.send(f"Channel {channel.name} is not a voice channel.")
                return
        
        # Format duration for display
        if chunk_duration >= 3600:
            display_time = f"{chunk_duration / 3600:.1f} hours"
        elif chunk_duration >= 60:
            display_time = f"{chunk_duration / 60:.1f} minutes"
        else:
            display_time = f"{chunk_duration} seconds"
            
        # Start monitoring the channel
        success = await voice_handler.monitor_voice_channel(channel_id, chunk_duration=chunk_duration)
        if success:
            # Add to monitored channels set
            MONITORED_VOICE_CHANNELS.add(channel_id)
            await ctx.send(f"Now monitoring voice channel: {channel.name} with chunk duration of {display_time}. Audio will be transcribed when users leave the channel or chunks are saved.")
        else:
            await ctx.send(f"Failed to start monitoring voice channel: {channel.name}. Check logs for details.")
    except Exception as e:
        await ctx.send(f"Error monitoring voice channel: {str(e)}")
        logging.error(f"Error in monitor_voice command: {e}", exc_info=True)

@bot.command(name='stop_monitor_voice')
async def stop_monitor_voice_command(ctx, channel_id=None):
    """Stop monitoring a voice channel.
    
    Args:
        channel_id (str, optional): The ID of the voice channel to stop monitoring.
            If not provided, attempts to use the channel the user is currently in.
    """
    try:
        # If no channel ID provided, use the channel the message author is in
        if channel_id is None:
            if ctx.author.voice and ctx.author.voice.channel:
                channel = ctx.author.voice.channel
                channel_id = channel.id
            else:
                await ctx.send("You're not in a voice channel. Please join one or specify a channel ID.")
                return
        else:
            # Convert to int if a string ID was provided
            channel_id = int(channel_id)
            channel = bot.get_channel(channel_id)
            
            if not channel:
                await ctx.send(f"Could not find channel with ID {channel_id}")
                return
        
        # Check if the channel is being monitored
        if channel_id not in voice_handler.monitored_channels:
            await ctx.send(f"Channel {channel.name} is not being monitored.")
            return
        
        # Stop monitoring the channel
        user_files = await voice_handler.stop_monitoring(channel_id)
        if channel_id in MONITORED_VOICE_CHANNELS:
            MONITORED_VOICE_CHANNELS.remove(channel_id)
            
        if user_files:
            await ctx.send(f"Stopped monitoring voice channel: {channel.name}. Processing {sum(len(files) for files in user_files.values())} audio files.")
        else:
            await ctx.send(f"Stopped monitoring voice channel: {channel.name}. No audio files were recorded.")
    except Exception as e:
        await ctx.send(f"Error stopping voice channel monitoring: {str(e)}")
        logging.error(f"Error in stop_monitor_voice command: {e}", exc_info=True)

@bot.command(name='save_audio')
async def save_audio_command(ctx, channel_id=None):
    """Save current audio chunks for a voice channel without stopping monitoring.
    
    This command saves the current audio data for all users in the monitored voice channel,
    creating audio files that can be processed while continuing to record.
    
    Args:
        channel_id (str, optional): The ID of the voice channel to save audio from. If not provided,
            attempts to use the channel the user is currently in.
    """
    try:
        # If no channel ID provided, use the channel the message author is in
        if channel_id is None:
            if ctx.author.voice and ctx.author.voice.channel:
                channel = ctx.author.voice.channel
                channel_id = channel.id
            else:
                await ctx.send("You're not in a voice channel. Please specify a channel ID.")
                return
        else:
            # Convert to int if a string ID was provided
            channel_id = int(channel_id)
            channel = bot.get_channel(channel_id)
            
            if not channel:
                await ctx.send(f"Could not find channel with ID {channel_id}")
                return
            
            if not isinstance(channel, discord.VoiceChannel):
                await ctx.send(f"Channel {channel.name} is not a voice channel.")
                return
        
        # Check if the channel is being monitored
        if channel_id not in voice_handler.monitored_channels:
            await ctx.send(f"Channel {channel.name} is not being monitored.")
            return
        
        # Save current audio chunks
        user_files = await voice_handler.save_current_audio_chunks(channel_id)
        
        if user_files:
            file_count = sum(len(files) for files in user_files.values())
            user_count = len(user_files)
            await ctx.send(f"Saved audio chunks for {user_count} user(s) in {channel.name}. Processing {file_count} audio files.")
        else:
            await ctx.send(f"No audio chunks were saved for {channel.name}. Either no audio has been recorded or users have not spoken.")
    except Exception as e:
        await ctx.send(f"Error saving audio chunks: {str(e)}")
        logging.error(f"Error in save_audio command: {e}", exc_info=True)

@bot.command(name='set_chunk_duration')
async def set_chunk_duration_command(ctx, duration_seconds: int, channel_id=None):
    """Set the automatic chunk duration for a voice channel.
    
    This sets how often audio will be automatically saved into chunks.
    
    Args:
        duration_seconds (int): The duration in seconds between automatic chunk saves
        channel_id (str, optional): The ID of the voice channel to modify. If not provided,
            attempts to use the channel the user is currently in.
    """
    try:
        # Validate duration
        if duration_seconds < 10:
            await ctx.send("Chunk duration must be at least 10 seconds.")
            return
            
        # If no channel ID provided, use the channel the message author is in
        if channel_id is None:
            if ctx.author.voice and ctx.author.voice.channel:
                channel = ctx.author.voice.channel
                channel_id = channel.id
            else:
                await ctx.send("You're not in a voice channel. Please specify a channel ID.")
                return
        else:
            # Convert to int if a string ID was provided
            channel_id = int(channel_id)
            channel = bot.get_channel(channel_id)
            
            if not channel:
                await ctx.send(f"Could not find channel with ID {channel_id}")
                return
            
            if not isinstance(channel, discord.VoiceChannel):
                await ctx.send(f"Channel {channel.name} is not a voice channel.")
                return
        
        # Check if the channel is being monitored
        if channel_id not in voice_handler.monitored_channels:
            await ctx.send(f"Channel {channel.name} is not being monitored.")
            return
        
        # Update chunk duration for all recorders in this channel
        updated_count = 0
        for (recorder_channel_id, _), recorder in voice_handler.user_recorders.items():
            if recorder_channel_id == channel_id:
                recorder.chunk_duration = duration_seconds
                updated_count += 1
        
        # Format duration for display
        if duration_seconds >= 3600:
            display_time = f"{duration_seconds / 3600:.1f} hours"
        elif duration_seconds >= 60:
            display_time = f"{duration_seconds / 60:.1f} minutes"
        else:
            display_time = f"{duration_seconds} seconds"
            
        await ctx.send(f"Set chunk duration to {display_time} for {updated_count} active recorders in {channel.name}.")
        
    except Exception as e:
        await ctx.send(f"Error setting chunk duration: {str(e)}")
        logging.error(f"Error in set_chunk_duration command: {e}", exc_info=True)

@bot.command(name='debug_voice')
async def debug_voice_command(ctx, channel_id=None):
    """Display debug information about the current voice recording status.
    
    This command provides detailed diagnostic information about voice recording status,
    including active recorders, packet statistics, and voice client information.
    
    Args:
        channel_id (str, optional): The ID of the voice channel to check. If not provided,
            attempts to use the channel the user is currently in.
    """
    try:
        # If no channel ID provided, use the channel the message author is in
        if channel_id is None:
            if ctx.author.voice and ctx.author.voice.channel:
                channel = ctx.author.voice.channel
                channel_id = channel.id
            else:
                await ctx.send("You're not in a voice channel. Please specify a channel ID.")
                return
        else:
            # Convert to int if a string ID was provided
            channel_id = int(channel_id)
            channel = bot.get_channel(channel_id)
            
            if not channel:
                await ctx.send(f"Could not find channel with ID {channel_id}")
                return
        
        # Check voice_handler status
        status_lines = [f"Debug info for voice channel: {channel.name} ({channel_id})"]
        
        # Check if this channel is being monitored
        is_monitored = channel_id in voice_handler.monitored_channels
        status_lines.append(f"Channel monitored: {is_monitored}")
        
        # Check voice client
        voice_client = voice_handler.voice_clients.get(channel_id)
        if voice_client:
            status_lines.append(f"Voice client type: {type(voice_client).__name__}")
            status_lines.append(f"Voice client connected: {voice_client.is_connected()}")
            
            # Check if it's a VoiceRecvClient
            from discord.ext import voice_recv as vr
            if hasattr(vr, 'VoiceRecvClient') and isinstance(voice_client, vr.VoiceRecvClient):
                status_lines.append(f"VoiceRecvClient is_listening: {voice_client.is_listening()}")
        else:
            status_lines.append("No voice client for this channel")
        
        # Check active recorders
        active_recorders = {}
        for (recorder_channel_id, user_id), recorder in voice_handler.user_recorders.items():
            if recorder_channel_id == channel_id:
                active_recorders[user_id] = recorder
        
        status_lines.append(f"Active recorders: {len(active_recorders)}")
        for user_id, recorder in active_recorders.items():
            user = bot.get_user(user_id)
            username = user.name if user else f"User {user_id}"
            status_lines.append(f"- {username}: {recorder.packet_counter} packets, {recorder.total_bytes_received/1024:.2f} KB received, {len(recorder.saved_chunks)} chunks saved")
        
        # Check channel members
        status_lines.append(f"Current channel members: {len(channel.members)}")
        for member in channel.members:
            status_lines.append(f"- {member.name} (Bot: {member.bot})")
            
        # Add info about audio directory
        status_lines.append(f"Audio directory: {voice_handler.audio_dir}")
        channel_audio_dir = voice_handler.audio_dir / str(channel_id)
        if channel_audio_dir.exists():
            status_lines.append(f"Channel audio directory exists: Yes")
            
            # List files in the directory
            files = list(channel_audio_dir.glob("*"))
            status_lines.append(f"Files in directory: {len(files)}")
            for file in files[:5]:  # Show up to 5 files
                status_lines.append(f"- {file.name} ({file.stat().st_size/1024:.2f} KB)")
            if len(files) > 5:
                status_lines.append(f"... and {len(files) - 5} more files")
        else:
            status_lines.append(f"Channel audio directory exists: No")
        
        # Send the debug info
        debug_info = "\n".join(status_lines)
        if len(debug_info) > 1900:  # Discord message limit is 2000 chars
            # If it's too long, split it into multiple messages
            for i in range(0, len(debug_info), 1900):
                await ctx.send(debug_info[i:i+1900])
        else:
            await ctx.send(debug_info)
        
    except Exception as e:
        await ctx.send(f"Error getting debug info: {str(e)}")
        logging.error(f"Error in debug_voice command: {e}", exc_info=True)

def main():
    # Get the token from environment variables
    token = discord_token
    if not token:
        raise ValueError("No Discord token found in environment variables")
    
    # Run the bot
    bot.run(token)

if __name__ == "__main__":
    main() 