import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from datetime import datetime, timedelta
import asyncio
import sys
from pathlib import Path
from collections import defaultdict

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.obsidian.vault import ObsidianVault
from src.utils.helpers import format_discord_message, create_obsidian_note_content, get_daily_note_path
from src.rag.engine import RAGEngine

# Load environment variables
load_dotenv()

# Bot configuration
intents = discord.Intents.default()
intents.message_content = True  # We need this to read message content

bot = commands.Bot(command_prefix='!', intents=intents)

# Get allowed channel IDs from environment
# Format: DISCORD_CHANNEL_IDS=123456789,987654321
ALLOWED_CHANNEL_IDS = [int(id.strip()) for id in os.getenv('DISCORD_CHANNEL_IDS', '').split(',') if id.strip()]

# Initialize Obsidian vault and RAG engine
vault = ObsidianVault()
rag_engine = RAGEngine()

# Dictionary to store messages by date and channel
messages_by_date_and_channel = defaultdict(lambda: defaultdict(list))

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

async def generate_channel_summary(channel_name, messages):
    """Generate a summary for a specific channel's messages"""
    if not messages:
        return None
        
    # Prepare messages for summarization
    messages_text = ""
    
    # Sort messages by timestamp
    sorted_messages = sorted(messages, key=lambda x: x.created_at)
    
    for message in sorted_messages:
        timestamp = message.created_at.strftime('%H:%M:%S')
        author = str(message.author)
        content = message.content or ""
        
        # Format each message with clear structure
        messages_text += f"Message from {author} at {timestamp}:\n"
        if content:
            messages_text += f"{content}\n"
        
        # Add attachments if any
        if message.attachments:
            messages_text += "Attachments: " + ", ".join([a.url for a in message.attachments]) + "\n"
        
        # Add embeds if any
        if message.embeds:
            for embed in message.embeds:
                if embed.title:
                    messages_text += f"Embed Title: {embed.title}\n"
                if embed.description:
                    messages_text += f"Embed Description: {embed.description}\n"
        
        messages_text += "---\n"  # Separator between messages
    
    if not messages_text.strip():
        return None
    
    # Add the messages to the RAG engine
    rag_engine.add_documents([messages_text])
    
    # Generate summary using the RAG engine with a more specific prompt
    summary_prompt = f"""
    Based on the following Discord messages from the channel "{channel_name}", please provide a detailed summary:

    {messages_text}

    Please analyze these messages and provide a summary that includes:
    1. Main topics and discussions
    2. Key decisions or conclusions reached
    3. Important announcements or events mentioned
    4. Questions asked and their answers (if any)
    5. Action items or follow-ups needed

    Format the summary with clear sections and bullet points where appropriate.
    Focus on the actual content and context of the messages provided.
    """
    
    summary = rag_engine.query(summary_prompt)
    
    # Clear the RAG engine for next use
    rag_engine.clear()
    
    return summary

async def generate_channel_ten_day_summary(channel_name, daily_summaries):
    """Generate a 10-day summary for a specific channel"""
    if not daily_summaries:
        return None
        
    # Prepare the combined summaries for the RAG engine
    summaries_text = f"Daily Summaries for channel {channel_name}:\n\n"
    for date_str, summary in daily_summaries.items():
        summaries_text += f"=== {date_str} ===\n{summary}\n\n"
    
    # Add the summaries to the RAG engine
    rag_engine.add_documents([summaries_text])
    
    # Generate 10-day summary for this channel
    ten_day_prompt = f"""
    Based on the following daily summaries from the channel "{channel_name}" over the last 10 days, please provide a comprehensive overview:

    {summaries_text}

    Please analyze these daily summaries and provide a high-level overview that:
    1. Identifies major themes and trends in this channel
    2. Highlights significant developments and changes over time
    3. Notes recurring topics or patterns
    4. Summarizes key decisions and action items
    5. Provides insights into the evolution of discussions
    6. Identifies any emerging trends or new topics

    Format the summary with clear sections and bullet points where appropriate.
    Focus on the specific context and themes of this channel.
    """
    
    ten_day_summary = rag_engine.query(ten_day_prompt)
    
    # Clear the RAG engine for next use
    rag_engine.clear()
    
    return ten_day_summary

async def generate_cross_channel_summary(channel_ten_day_summaries):
    """Generate a summary combining all channels' 10-day summaries"""
    if not channel_ten_day_summaries:
        return None
        
    # Prepare the combined summaries for the RAG engine
    summaries_text = "10-Day Summaries by Channel:\n\n"
    for channel_name, summary in channel_ten_day_summaries.items():
        summaries_text += f"=== {channel_name} ===\n{summary}\n\n"
    
    # Add the summaries to the RAG engine
    rag_engine.add_documents([summaries_text])
    
    # Generate cross-channel summary
    cross_channel_prompt = f"""
    Based on the following 10-day summaries from different Discord channels, please provide a comprehensive overview:

    {summaries_text}

    Please analyze these channel summaries and provide a high-level overview that:
    1. Identifies common themes and patterns across all channels
    2. Highlights how different channels interact and relate to each other
    3. Notes any cross-channel dependencies or relationships
    4. Summarizes key decisions and action items across channels
    5. Provides insights into the overall project or community
    6. Identifies any emerging trends or topics that span multiple channels

    Format the summary with clear sections and bullet points where appropriate.
    Focus on connecting themes and developments across all channels.
    """
    
    cross_channel_summary = rag_engine.query(cross_channel_prompt)
    
    # Clear the RAG engine for next use
    rag_engine.clear()
    
    return cross_channel_summary

async def save_messages_to_vault(date_str):
    """Save all messages for a specific date to the Obsidian vault"""
    if not messages_by_date_and_channel[date_str]:
        return {}
    
    # Create a folder for this date
    date_folder = f"Daily Notes/{date_str}"
    
    # Dictionary to store channel summaries for this date
    channel_summaries = {}
    
    # Process each channel
    for channel_name, messages in messages_by_date_and_channel[date_str].items():
        if not messages:
            continue
            
        # Generate summary for this channel
        channel_summary = await generate_channel_summary(channel_name, messages)
        if channel_summary:
            channel_summaries[channel_name] = channel_summary
            
            # Create content for all messages in this channel
            content = f"# {channel_name} - {date_str}\n\n"
            content += "## Summary\n\n"
            content += f"{channel_summary}\n\n"
            content += "## Messages\n\n"
            
            # Sort messages by timestamp
            sorted_messages = sorted(messages, key=lambda x: x.created_at)
            
            for message in sorted_messages:
                content += create_concise_message_content(message)
                content += "\n---\n\n"  # Add separator between messages
            
            # Save to vault using the channel name as the filename
            try:
                # Sanitize channel name for use as filename
                safe_channel_name = channel_name.replace('/', '_').replace('\\', '_')
                note_path = vault.create_note(safe_channel_name, content, date_folder)
                print(f"Saved messages and summary for {date_str} in channel {channel_name} to: {note_path}")
            except Exception as e:
                print(f"Error saving messages to vault: {e}")
    
    # Clear the messages for this date after saving
    messages_by_date_and_channel[date_str] = defaultdict(list)
    
    return channel_summaries

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

async def fetch_all_channels_history(days=10):
    """Fetch message history from all allowed channels for the specified number of days"""
    print(f"\nFetching message history from all channels for the last {days} day(s)...")
    
    # Dictionary to store daily summaries by channel
    daily_summaries_by_channel = defaultdict(dict)
    
    # Fetch messages from each channel
    for channel_id in ALLOWED_CHANNEL_IDS:
        channel = bot.get_channel(channel_id)
        if channel:
            await fetch_channel_history(channel, days)
        else:
            print(f"Warning: Could not find channel with ID {channel_id}")
    
    # Save messages and generate summaries for each day
    for date_str in sorted(messages_by_date_and_channel.keys()):
        channel_summaries = await save_messages_to_vault(date_str)
        for channel_name, summary in channel_summaries.items():
            daily_summaries_by_channel[channel_name][date_str] = summary
    
    # Generate 10-day summaries for each channel
    channel_ten_day_summaries = {}
    for channel_name, daily_summaries in daily_summaries_by_channel.items():
        ten_day_summary = await generate_channel_ten_day_summary(channel_name, daily_summaries)
        if ten_day_summary:
            channel_ten_day_summaries[channel_name] = ten_day_summary
            try:
                # Save 10-day summary for this channel
                ten_day_folder = "Daily Notes/Ten Day Summaries"
                summary_path = vault.create_note(
                    f"{channel_name} - Ten Day Summary",
                    f"# {channel_name} - Ten Day Summary\n\n{ten_day_summary}",
                    ten_day_folder
                )
                print(f"Saved 10-day summary for {channel_name} to: {summary_path}")
            except Exception as e:
                print(f"Error saving 10-day summary for {channel_name} to vault: {e}")
    
    # Generate and save cross-channel summary
    if channel_ten_day_summaries:
        cross_channel_summary = await generate_cross_channel_summary(channel_ten_day_summaries)
        if cross_channel_summary:
            try:
                # Save cross-channel summary
                ten_day_folder = "Daily Notes/Ten Day Summaries"
                summary_path = vault.create_note(
                    "Cross Channel Summary",
                    f"# Cross Channel Summary - Last 10 Days\n\n{cross_channel_summary}",
                    ten_day_folder
                )
                print(f"Saved cross-channel summary to: {summary_path}")
            except Exception as e:
                print(f"Error saving cross-channel summary to vault: {e}")
    
    print("Finished fetching and processing message history from all channels.")

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    if ALLOWED_CHANNEL_IDS:
        print(f'Bot is listening to channels: {", ".join(str(id) for id in ALLOWED_CHANNEL_IDS)}')
        # Fetch message history when bot starts
        await fetch_all_channels_history()
    else:
        print('Warning: No channel IDs configured')

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

@bot.command(name='history')
async def fetch_history(ctx, days: int = 1):
    """Command to fetch message history for a specific number of days"""
    if ctx.channel.id not in ALLOWED_CHANNEL_IDS:
        return
        
    await fetch_all_channels_history(days)

@bot.command(name='summarize')
async def summarize(ctx, days: int = 1):
    """Command to generate a summary of messages for a specific number of days"""
    if ctx.channel.id not in ALLOWED_CHANNEL_IDS:
        return
    
    # Calculate the time threshold
    time_threshold = datetime.utcnow() - timedelta(days=days)
    
    # Clear existing messages
    messages_by_date_and_channel.clear()
    
    # Fetch messages from all channels
    await fetch_all_channels_history(days)
    
    await ctx.send(f"Summaries generated for the last {days} day(s) from all channels.")

def main():
    # Get the token from environment variables
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        raise ValueError("No Discord token found in environment variables")
    
    # Run the bot
    bot.run(token)

if __name__ == "__main__":
    main() 