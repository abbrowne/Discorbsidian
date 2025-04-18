import re
from typing import List, Dict
from datetime import datetime

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a string to be used as a filename
    
    Args:
        filename: The string to sanitize
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    return filename[:255]

def format_discord_message(message) -> Dict:
    """
    Format a Discord message for storage
    
    Args:
        message: Discord message object
        
    Returns:
        Dictionary with formatted message data
    """
    return {
        'content': message.content,
        'author': str(message.author),
        'channel': str(message.channel),
        'timestamp': message.created_at.isoformat(),
        'attachments': [a.url for a in message.attachments],
        'embeds': [e.to_dict() for e in message.embeds]
    }

def create_obsidian_note_content(message_data: Dict) -> str:
    """
    Create Obsidian note content from message data
    
    Args:
        message_data: Formatted message data
        
    Returns:
        Formatted note content
    """
    content = f"""# Discord Message from {message_data['author']}

**Channel:** {message_data['channel']}
**Timestamp:** {message_data['timestamp']}

## Content
{message_data['content']}

"""
    
    if message_data['attachments']:
        content += "\n## Attachments\n"
        for url in message_data['attachments']:
            content += f"- {url}\n"
            
    if message_data['embeds']:
        content += "\n## Embeds\n"
        for embed in message_data['embeds']:
            content += f"- {embed.get('title', 'Untitled')}\n"
            if 'description' in embed:
                content += f"  {embed['description']}\n"
                
    return content

def get_daily_note_path() -> str:
    """
    Get the path for today's daily note
    
    Returns:
        Path string for the daily note
    """
    today = datetime.now().strftime('%Y-%m-%d')
    return f"Daily Notes/{today}.md" 