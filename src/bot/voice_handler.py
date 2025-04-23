import os
import asyncio
import logging
import datetime
import tempfile
import uuid
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import discord
from discord.ext import commands
from openai import OpenAI
try:
    from discord.ext import voice_recv as vr
    VOICE_RECV_AVAILABLE = True
    print("discord-ext-voice-recv is available. Audio recording enabled.")
except ImportError:
    VOICE_RECV_AVAILABLE = False
    print("WARNING: discord-ext-voice-recv not found. Only presence tracking will be available.")

from src.obsidian.vault import ObsidianVault

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define CustomAudioSink if voice receiving is available
if VOICE_RECV_AVAILABLE:
    class CustomAudioSink(vr.AudioSink):
        """Audio sink for capturing and processing voice data from Discord users"""
        def __init__(self, voice_handler, channel_id):
            super().__init__()
            self.voice_handler = voice_handler
            self.channel_id = channel_id
            self.packet_counters = {}  # user_id -> packet count
            self.last_log_time = datetime.datetime.now()
            logger.info("CustomAudioSink initialized")
        
        def cleanup(self):
            """Cleanup method required by AudioSink interface"""
            logger.info(f"CustomAudioSink cleanup for channel {self.channel_id}")
            # No special cleanup needed for our simple implementation
        
        def wants_opus(self) -> bool:
            """Indicate whether we want Opus-encoded packets (True) or decoded PCM data (False)"""
            logger.debug(f"CustomAudioSink.wants_opus called, returning False to receive PCM data")
            return False  # We want decoded PCM data
        
        def write(self, user, data):
            """Process incoming audio data from a user"""
            now = datetime.datetime.now()
            
            # Log packet details
            if user:
                user_id = user.id if hasattr(user, 'id') else 'unknown'
                user_name = user.name if hasattr(user, 'name') else 'unknown'
                
                # Track packets per user
                if user_id not in self.packet_counters:
                    self.packet_counters[user_id] = 0
                self.packet_counters[user_id] += 1
                
                # Log detailed information about the packet
                has_pcm = hasattr(data, 'pcm') and data.pcm is not None
                pcm_size = len(data.pcm) if has_pcm else 0
                has_opus = hasattr(data, 'opus') and data.opus is not None
                opus_size = len(data.opus) if has_opus else 0
                
                # Lower log level for ignored bots to reduce noise
                if self.voice_handler.is_ignored_bot(user):
                    log_level = logging.DEBUG
                else:
                    log_level = logging.DEBUG
                
                logger.log(log_level, f"Received packet from {user_name} (ID: {user_id}): " 
                          f"PCM: {has_pcm} ({pcm_size} bytes), "
                          f"Opus: {has_opus} ({opus_size} bytes)")
                
                # Periodically log packet statistics
                if (now - self.last_log_time).total_seconds() > 5:
                    for uid, count in self.packet_counters.items():
                        u = self.voice_handler.bot.get_user(uid)
                        name = u.name if u else f"User {uid}"
                        # Only log non-ignored bots or users at INFO level
                        if u and not self.voice_handler.is_ignored_bot(u):
                            logger.info(f"User {name} has sent {count} audio packets so far")
                    self.last_log_time = now
            else:
                logger.warning("Received audio packet with no user information")
            
            # Skip processing for empty users, bots, or users in the ignore list
            if not user or self.voice_handler.is_ignored_bot(user):
                if user and user.name in self.voice_handler.ignored_bot_names:
                    logger.debug(f"Ignoring audio from bot in ignored list: {user.name}")
                return
                
            # Check data validity
            if not hasattr(data, 'pcm') or data.pcm is None or len(data.pcm) == 0:
                logger.warning(f"Received empty PCM data from {user.name if user else 'unknown user'}")
                return
                
            # Get or create recorder for this user
            recorder_key = (self.channel_id, user.id)
            if recorder_key not in self.voice_handler.user_recorders:
                logger.info(f"Creating new recorder for {user.name} (first audio packet)")
                recorder = AudioRecorder(
                    user.id,
                    user.name,
                    self.channel_id,
                    self.voice_handler.audio_dir / str(self.channel_id)
                )
                recorder.start()
                self.voice_handler.user_recorders[recorder_key] = recorder
                
            # Add audio to recorder (using PCM data)
            try:
                self.voice_handler.user_recorders[recorder_key].add_audio(data.pcm)
                logger.debug(f"Added {len(data.pcm)} bytes of audio data to recorder for {user.name}")
            except Exception as e:
                logger.error(f"Error adding audio data to recorder: {e}", exc_info=True)

class AudioRecorder:
    """
    Records audio for a specific user in a voice channel
    """
    def __init__(self, user_id: int, username: str, channel_id: int, output_dir: Path, chunk_duration: int = 300):
        self.user_id = user_id
        self.username = username
        self.channel_id = channel_id
        self.output_dir = output_dir
        self.recording = False
        self.audio_data = bytearray()
        self.start_time = None
        self.last_chunk_time = None
        self.packet_counter = 0
        self.total_bytes_received = 0
        self.last_log_time = datetime.datetime.now()
        self.presence_duration = 0
        self.chunk_duration = chunk_duration  # Duration in seconds before creating a new chunk (default: 5 minutes)
        self.chunk_counter = 0
        self.saved_chunks = []
        
        # Ensure output directory exists
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory set up at {self.output_dir}")
        except Exception as e:
            logger.error(f"Error creating output directory {self.output_dir}: {e}", exc_info=True)
    
    def start(self):
        """Start recording session for this user"""
        self.recording = True
        self.audio_data = bytearray()
        self.start_time = datetime.datetime.now()
        self.last_chunk_time = self.start_time
        self.packet_counter = 0
        self.total_bytes_received = 0
        self.last_log_time = datetime.datetime.now()
        self.chunk_counter = 0
        self.saved_chunks = []
        logger.info(f"Started recording session for user {self.username} in channel {self.channel_id}")
    
    def add_audio(self, audio_data: bytes):
        """Add audio data to the recording"""
        if not self.recording:
            logger.warning(f"Attempted to add audio for {self.username} but recording is not active")
            return
        
        if not audio_data:
            logger.debug(f"Received empty audio data from {self.username}")
            return
            
        try:
            # Log information about the audio data
            data_len = len(audio_data)
            logger.debug(f"Adding {data_len} bytes of audio data for {self.username}")
            
            # Extend our buffer with the new data
            self.audio_data.extend(audio_data)
            self.packet_counter += 1
            self.total_bytes_received += data_len
            
            # Log packet count every 5 seconds for diagnostic purposes
            now = datetime.datetime.now()
            if (now - self.last_log_time).total_seconds() > 5:
                logger.info(f"User {self.username} has received {self.packet_counter} audio packets " +
                            f"({self.total_bytes_received / 1024:.2f} KB) in the last 5 seconds")
                self.last_log_time = now
                
            # Check if it's time to save a chunk
            if self.last_chunk_time and (now - self.last_chunk_time).total_seconds() > self.chunk_duration:
                logger.info(f"Chunk duration of {self.chunk_duration}s exceeded for {self.username}, saving chunk")
                self.save_chunk()
        except Exception as e:
            logger.error(f"Error processing audio data for {self.username}: {e}", exc_info=True)
    
    def save_chunk(self) -> Optional[Path]:
        """
        Save the current audio buffer as a chunk and start a new one
        
        Returns:
            Path to the saved chunk file or None if no data to save
        """
        # Only check if recording state is active if we have no audio data
        if len(self.audio_data) == 0:
            logger.warning(f"No audio data to save for {self.username}")
            return None
        
        # Even if recording state has changed, we still want to save existing data
        if not self.recording:
            logger.warning(f"Recording flag is off, but saving final data for {self.username} anyway ({len(self.audio_data)/1024:.2f} KB)")
        
        try:
            # Create a filename with timestamp and user info
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.chunk_counter += 1
            
            if VOICE_RECV_AVAILABLE:
                # Ensure output directory exists
                self.output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save actual audio data
                filename = f"{timestamp}_{self.username}_{self.user_id}_chunk{self.chunk_counter}.pcm"
                file_path = self.output_dir / filename
                
                # Save the PCM audio data
                with open(file_path, 'wb') as f:
                    f.write(self.audio_data)
                    
                logger.info(f"Saved audio chunk {self.chunk_counter} for {self.username} to {file_path} " +
                             f"({len(self.audio_data)/1024:.2f} KB)")
                
                # Reset audio buffer for the next chunk, but only if still recording
                buffer_size = len(self.audio_data)
                self.audio_data = bytearray() if self.recording else self.audio_data
                
                # Add to saved chunks list
                self.saved_chunks.append(file_path)
                
                # If we're not recording anymore, clear the buffer to prevent duplicates
                if not self.recording:
                    self.audio_data = bytearray()
                
                self.last_chunk_time = datetime.datetime.now()
                
                return file_path
            else:
                logger.warning("VOICE_RECV_AVAILABLE is False, not saving actual audio data")
        except Exception as e:
            logger.error(f"Error saving audio chunk for {self.username}: {e}", exc_info=True)
            
        return None
    
    def stop(self) -> List[Path]:
        """
        Stop recording session and save any remaining audio data
        
        Returns:
            List of all audio file paths saved during this session
        """
        if not self.recording:
            return self.saved_chunks
            
        # Save the final chunk first, while recording is still active
        final_chunk = None
        try:
            if len(self.audio_data) > 0:
                logger.info(f"Saving final audio chunk for {self.username} before stopping recording (buffer size: {len(self.audio_data)/1024:.2f} KB)")
                final_chunk = self.save_chunk()
                if final_chunk:
                    logger.info(f"Successfully saved final audio chunk for {self.username}")
        except Exception as e:
            logger.error(f"Error saving final chunk for {self.username}: {e}", exc_info=True)
        
        # Now mark the recording as stopped
        self.recording = False
        
        # Calculate how long the user was in the channel
        end_time = datetime.datetime.now()
        self.presence_duration = (end_time - self.start_time).total_seconds()
        
        # Create a metadata file with session information
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_filename = f"{timestamp}_{self.username}_{self.user_id}_meta.txt"
        meta_file_path = self.output_dir / meta_filename
        
        with open(meta_file_path, 'w') as f:
            f.write(f"User: {self.username}\n")
            f.write(f"User ID: {self.user_id}\n")
            f.write(f"Channel ID: {self.channel_id}\n")
            f.write(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {self.presence_duration:.2f} seconds\n")
            f.write(f"Total chunks: {len(self.saved_chunks)}\n")
            f.write(f"Total audio data: {self.total_bytes_received/1024:.2f} KB\n")
            f.write("\nChunks:\n")
            for i, chunk_path in enumerate(self.saved_chunks):
                f.write(f"{i+1}. {chunk_path.name}\n")
            
            if not VOICE_RECV_AVAILABLE:
                f.write("\nNote: This is a placeholder file. Actual audio recording requires discord-ext-voice-recv.\n")
            elif len(self.saved_chunks) == 0:
                f.write("\nNote: No audio data was captured during this session.\n")
            
        logger.info(f"Saved recording metadata for {self.username} to {meta_file_path} (presence duration: {self.presence_duration:.2f}s)")
            
        return self.saved_chunks

class VoiceHandler:
    """
    Handles voice channel recording and transcription using Whisper API
    """
    def __init__(self, bot: commands.Bot, obsidian_vault: ObsidianVault, openai_client: OpenAI):
        self.bot = bot
        self.obsidian_vault = obsidian_vault
        self.openai_client = openai_client
        self.voice_clients: Dict[int, discord.VoiceClient] = {}
        self.monitored_channels: Set[int] = set()
        self.user_recorders: Dict[Tuple[int, int], AudioRecorder] = {}  # (channel_id, user_id) -> recorder
        self.connection_lock = asyncio.Lock()
        self.temp_dir = Path(tempfile.gettempdir()) / "discorbsidian_audio"
        self.audio_dir = Path("audio_recordings")
        
        # Config file for persisting settings
        self.config_file = Path("config/voice_handler_config.json")
        
        # Load or initialize the ignored bots list
        self.ignored_bot_names = self._load_ignored_bots()
        if not self.ignored_bot_names:
            self.ignored_bot_names = {"Lofi Radio"}
            self._save_ignored_bots()
            
        logger.info(f"Ignoring bots: {', '.join(self.ignored_bot_names)}")
        self.setup_audio_directory()
        
    def setup_audio_directory(self):
        """Set up directories for audio recordings"""
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Audio directory set up at {self.audio_dir}")
        logger.info(f"Temp directory set up at {self.temp_dir}")
    
    def check_voice_permissions(self, channel: discord.VoiceChannel) -> bool:
        """Check if the bot has necessary permissions in the voice channel"""
        permissions = channel.permissions_for(channel.guild.me)
        return permissions.connect and permissions.speak
        
    async def monitor_voice_channel(self, channel_id: int, chunk_duration: int = 300) -> bool:
        """
        Start monitoring a voice channel for audio
        
        Args:
            channel_id: Discord voice channel ID
            chunk_duration: How often to save audio chunks in seconds (default: 5 minutes)
            
        Returns:
            True if monitoring started, False otherwise
        """
        async with self.connection_lock:
            # Check if already monitoring
            if channel_id in self.monitored_channels:
                logger.info(f"Already monitoring voice channel {channel_id}")
                return True
                
            # Get channel
            channel = self.bot.get_channel(channel_id)
            if not channel:
                logger.error(f"Could not find channel with ID {channel_id}")
                return False
                
            # Verify it's a voice channel
            if not isinstance(channel, discord.VoiceChannel):
                logger.error(f"Channel {channel_id} is not a voice channel")
                return False
                
            # Check permissions
            if not self.check_voice_permissions(channel):
                logger.error(f"Bot lacks necessary permissions in channel {channel.name}")
                return False
                
            try:
                # Connect to voice channel
                logger.info(f"Connecting to voice channel {channel.name} ({channel_id})")
                
                if VOICE_RECV_AVAILABLE:
                    try:
                        # Use VoiceRecvClient when connecting to enable voice receiving capability
                        logger.info(f"Using VoiceRecvClient for voice channel connection")
                        voice_client = await channel.connect(cls=vr.VoiceRecvClient)
                        # Inspect the voice client details including available methods
                        logger.info(f"VoiceRecvClient methods: {[method for method in dir(voice_client) if not method.startswith('_')]}")
                        logger.info(f"VoiceRecvClient connected: {voice_client}")
                        logger.info(f"Connection mode: {getattr(voice_client, 'mode', 'unknown')}")
                        logger.info(f"Encryption key size: {len(getattr(voice_client, 'secret_key', b'')) if hasattr(voice_client, 'secret_key') else 'unknown'}")
                    except Exception as e:
                        logger.error(f"Error connecting with VoiceRecvClient: {e}", exc_info=True)
                        logger.info("Falling back to standard VoiceClient")
                        voice_client = await channel.connect()
                else:
                    # Fall back to regular VoiceClient if voice_recv is not available
                    voice_client = await channel.connect()
                    logger.info(f"Connected using standard VoiceClient: {voice_client}")
                
                self.voice_clients[channel_id] = voice_client
                self.monitored_channels.add(channel_id)
                
                # Log voice client details for diagnostics
                logger.info(f"Voice client connected: {voice_client}")
                logger.info(f"Voice client type: {type(voice_client).__name__}")
                
                # Setup voice packet handler if available
                if VOICE_RECV_AVAILABLE and isinstance(voice_client, vr.VoiceRecvClient):
                    logger.info(f"Setting up voice packet receiver for {channel.name}")
                    
                    try:
                        # Create and use the sink
                        audio_sink = CustomAudioSink(self, channel_id)
                        logger.info(f"Created CustomAudioSink for channel {channel.name}")
                        
                        # Try to listen for packets
                        voice_client.listen(audio_sink)
                        logger.info(f"Successfully started listening for voice packets in {channel.name}")
                        
                        # Check if we've successfully registered for packets
                        is_listening = getattr(voice_client, 'is_listening', lambda: False)()
                        logger.info(f"Voice client is_listening: {is_listening}")
                    except Exception as e:
                        logger.error(f"Error setting up voice packet receiver: {e}", exc_info=True)
                        logger.warning(f"Voice packet reception will not be available for {channel.name}")
                else:
                    logger.info(f"Voice packet receiver not available, using presence tracking only")
                    if VOICE_RECV_AVAILABLE:
                        logger.warning(f"VoiceRecvClient not properly instantiated despite being available")
                
                # Initialize recorders for users already in the channel
                for member in channel.members:
                    if not member.bot:
                        recorder_key = (channel_id, member.id)
                        logger.info(f"Setting up recorder for existing member {member.name} in channel {channel.name}")
                        recorder = AudioRecorder(
                            member.id,
                            member.name,
                            channel_id,
                            self.audio_dir / str(channel_id),
                            chunk_duration=chunk_duration
                        )
                        recorder.start()
                        self.user_recorders[recorder_key] = recorder
                        logger.info(f"Created recorder for existing channel member: {member.name}")
                
                logger.info(f"Started monitoring voice channel {channel.name} ({channel_id}) with chunk duration of {chunk_duration} seconds")
                if not VOICE_RECV_AVAILABLE:
                    logger.info(f"NOTE: This implementation can only track user presence.")
                    logger.info(f"For voice capture, install discord-ext-voice-recv: pip install discord-ext-voice-recv")
                
                return True
            except Exception as e:
                logger.error(f"Error monitoring voice channel: {e}", exc_info=True)
                return False
    
    async def stop_monitoring(self, channel_id: int) -> Dict[str, List[Path]]:
        """
        Stop monitoring a voice channel
        
        Args:
            channel_id: Discord voice channel ID
            
        Returns:
            Dictionary mapping usernames to lists of audio file paths
        """
        async with self.connection_lock:
            if channel_id not in self.monitored_channels:
                logger.warning(f"Not monitoring voice channel {channel_id}")
                return {}
                
            # Stop all recorders for this channel
            user_audio_files = {}
            for (recorder_channel_id, user_id), recorder in list(self.user_recorders.items()):
                if recorder_channel_id == channel_id:
                    audio_files = recorder.stop()
                    if audio_files:
                        if recorder.username not in user_audio_files:
                            user_audio_files[recorder.username] = []
                        user_audio_files[recorder.username].extend(audio_files)
                    del self.user_recorders[(recorder_channel_id, user_id)]
            
            # Disconnect voice client
            voice_client = self.voice_clients.get(channel_id)
            if voice_client:
                # The voice_client is already VoiceRecvClient when VOICE_RECV_AVAILABLE is True
                # No need to stop a separate voice_recv instance
                await voice_client.disconnect()
                del self.voice_clients[channel_id]
                
            # Remove from monitored channels
            self.monitored_channels.remove(channel_id)
            
            logger.info(f"Stopped monitoring voice channel {channel_id}")
            
            # Process audio files with Whisper
            for username, audio_files in user_audio_files.items():
                for audio_file in audio_files:
                    await self.transcribe_audio_file(channel_id, username, audio_file)
                    
            return user_audio_files
            
    async def on_voice_state_update(self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
        """
        Handle voice state updates for users joining/leaving channels
        """
        if member.bot:
            return
            
        # Handle user joining a monitored channel
        if after and after.channel and after.channel.id in self.monitored_channels:
            logger.info(f"Member {member.name} joined monitored voice channel {after.channel.name}")
            recorder_key = (after.channel.id, member.id)
            if recorder_key not in self.user_recorders:
                # Find the chunk duration used by other recorders in this channel
                chunk_duration = 300  # Default: 5 minutes
                for (ch_id, _), recorder in self.user_recorders.items():
                    if ch_id == after.channel.id:
                        chunk_duration = recorder.chunk_duration
                        break
                        
                recorder = AudioRecorder(
                    member.id,
                    member.name,
                    after.channel.id,
                    self.audio_dir / str(after.channel.id),
                    chunk_duration=chunk_duration
                )
                recorder.start()
                self.user_recorders[recorder_key] = recorder
                logger.info(f"Started recording for {member.name} who joined channel {after.channel.id}")
        
        # Handle user leaving a monitored channel
        if before and before.channel and before.channel.id in self.monitored_channels:
            logger.info(f"Member {member.name} left monitored voice channel {before.channel.name}")
            recorder_key = (before.channel.id, member.id)
            if recorder_key in self.user_recorders:
                recorder = self.user_recorders[recorder_key]
                
                # Manually force a save of any audio data before stopping recording
                try:
                    # Only attempt to save if we have data
                    if hasattr(recorder, 'audio_data') and len(recorder.audio_data) > 0:
                        logger.info(f"Force saving audio data for {member.name} before they leave the channel ({len(recorder.audio_data)/1024:.2f} KB)")
                        chunk_path = recorder.save_chunk()
                        if chunk_path:
                            logger.info(f"Successfully saved audio data to {chunk_path}")
                        else:
                            logger.warning(f"Failed to save audio data for {member.name}")
                except Exception as e:
                    logger.error(f"Error force-saving audio data: {e}", exc_info=True)
                
                # Now stop the recording
                audio_files = recorder.stop()
                del self.user_recorders[recorder_key]
                
                # Process audio files if there are any
                if audio_files:
                    for audio_file in audio_files:
                        await self.transcribe_audio_file(before.channel.id, member.name, audio_file)
                
                logger.info(f"Stopped recording for {member.name} who left channel {before.channel.id}")
    
    async def convert_pcm_to_wav(self, pcm_path: Path) -> Optional[Path]:
        """
        Convert PCM audio to WAV format for Whisper API
        
        Args:
            pcm_path: Path to PCM audio file
            
        Returns:
            Path to WAV file if successful, None otherwise
        """
        # Skip if the file isn't a PCM file
        if pcm_path.suffix.lower() != '.pcm':
            return None
            
        try:
            wav_path = pcm_path.with_suffix('.wav')
            
            # Use FFmpeg to convert PCM to WAV
            # PCM format: 16-bit, 48kHz, 2 channel
            cmd = [
                'ffmpeg',
                '-f', 's16le',      # Format: signed 16-bit little-endian
                '-ar', '48000',     # Sample rate: 48kHz (Discord standard)
                '-ac', '2',         # Channels: 2 (stereo)
                '-i', str(pcm_path),  # Input file
                str(wav_path)       # Output file
            ]
            
            logger.info(f"Converting PCM to WAV: {' '.join(cmd)}")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error converting PCM to WAV: {stderr.decode()}")
                return None
                
            logger.info(f"Successfully converted PCM to WAV: {wav_path}")
            return wav_path
        except Exception as e:
            logger.error(f"Error in PCM to WAV conversion: {e}", exc_info=True)
            return None
    
    async def transcribe_audio_file(self, channel_id: int, username: str, audio_path: Path) -> Optional[str]:
        """
        Transcribe an audio file using Whisper API or create a placeholder
        
        Args:
            channel_id: Discord channel ID
            username: Username of the speaker
            audio_path: Path to the audio file or placeholder
            
        Returns:
            Transcription text if successful, None otherwise
        """
        try:
            # Get the channel name
            channel = self.bot.get_channel(channel_id)
            channel_name = channel.name if channel else f"channel-{channel_id}"
            
            # Extract metadata from the filename
            filename = audio_path.name
            file_timestamp = filename.split('_')[0]
            
            # Try to convert file timestamp to a readable format
            try:
                timestamp_obj = datetime.datetime.strptime(file_timestamp, "%Y%m%d_%H%M%S")
                formatted_timestamp = timestamp_obj.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_timestamp = "Unknown time"
            
            # Find the chunk number if available
            chunk_info = ""
            if "chunk" in filename:
                try:
                    chunk_num = filename.split("chunk")[1].split(".")[0]
                    chunk_info = f" (Chunk {chunk_num})"
                except:
                    pass
                
            # Get audio duration information
            duration_info = ""
            if audio_path.suffix.lower() == '.pcm':
                try:
                    # PCM format: 16-bit, 48kHz, 2 channels
                    pcm_size = audio_path.stat().st_size
                    duration_seconds = pcm_size / (48000 * 2 * 2)  # size / (sample_rate * channels * bytes_per_sample)
                    
                    if duration_seconds < 60:
                        duration_info = f" - Duration: {duration_seconds:.1f} seconds"
                    else:
                        minutes = int(duration_seconds // 60)
                        seconds = int(duration_seconds % 60)
                        duration_info = f" - Duration: {minutes}m {seconds}s"
                except Exception as e:
                    logger.warning(f"Could not calculate audio duration: {e}")
                
            # Add a header to the transcript with the recording time
            transcript_header = f"Recording started at {formatted_timestamp}{chunk_info}{duration_info}\n\n"
            
            # Check if this is a PCM audio file or a placeholder
            if audio_path.suffix.lower() == '.pcm' and VOICE_RECV_AVAILABLE:
                # Convert PCM to WAV for Whisper API
                wav_path = await self.convert_pcm_to_wav(audio_path)
                if not wav_path:
                    logger.error(f"Failed to convert audio file {audio_path} to WAV")
                    return None
                    
                logger.info(f"Transcribing audio file {wav_path} with Whisper API")
                
                # Send audio file to Whisper API
                with open(wav_path, "rb") as audio_file:
                    transcript_response = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                    
                transcript = transcript_response.text
                
                if not transcript:
                    logger.warning(f"Whisper API returned empty transcript for {wav_path}")
                    transcript = f"[No speech detected for {username}]"
                    
                logger.info(f"Successfully transcribed audio: {transcript[:100]}...")
                
                # Clean up WAV file to save space
                try:
                    if wav_path and wav_path.exists():
                        wav_path.unlink()
                        logger.debug(f"Deleted temporary WAV file {wav_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary WAV file {wav_path}: {e}")
            else:
                # Create a placeholder transcript for text files
                logger.info(f"Creating placeholder for {username} in {channel_name}")
                
                if audio_path.suffix.lower() == '.txt':
                    # Read the placeholder file
                    with open(audio_path, "r") as f:
                        placeholder_info = f.read()
                    
                    # Create a placeholder transcript
                    transcript = f"[Placeholder] User {username} was present in the voice channel.\n\n"
                    if not VOICE_RECV_AVAILABLE:
                        transcript += "Note: Actual audio recording requires discord-ext-voice-recv.\n\n"
                    else:
                        transcript += "Note: No audio was captured during this session.\n\n"
                    transcript += "User presence information:\n"
                    transcript += placeholder_info
                else:
                    # Unknown file type
                    transcript = f"[Unknown file type] Unable to process {audio_path.name}"
            
            # Add the header to the transcript
            full_transcript = transcript_header + transcript
            
            # Save transcription to Obsidian
            await self.save_transcription(channel_id, username, full_transcript)
            
            return full_transcript
            
        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {e}", exc_info=True)
            return None
    
    async def save_transcription(self, channel_id: int, username: str, transcript: str):
        """
        Save a voice transcription to Obsidian vault, appending to existing transcriptions for the same user and date
        
        Args:
            channel_id: Discord channel ID
            username: Username of the speaker
            transcript: Transcription text
        """
        try:
            # Get channel name
            channel = self.bot.get_channel(channel_id)
            channel_name = channel.name if channel else f"channel-{channel_id}"
            
            # Get timestamp for this recording
            current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create a folder for voice transcriptions
            folder_path = "Voice Transcriptions"
            
            # Generate a filename with date and username (use date only, not time)
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            safe_username = username.replace('/', '_').replace('\\', '_')
            safe_channel = channel_name.replace('/', '_').replace('\\', '_')
            filename = f"{date_str} - {safe_username} in {safe_channel}.md"
            
            # Full path in vault
            note_path = self.obsidian_vault.vault_path / folder_path / filename
            
            # Ensure directory exists
            note_path.parent.mkdir(parents=True, exist_ok=True)
            
            # New section to add
            new_section = f"\n\n## Recording at {current_timestamp}\n\n"
            new_section += transcript + "\n\n---\n"
            
            # Check if the file already exists
            if note_path.exists():
                # Read existing content
                with open(note_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                    
                # If this is the first recording added to an existing file, add a separator
                if "## Recording at" not in existing_content:
                    # Convert the single transcription format to multiple recordings format
                    header_end_idx = existing_content.find("## Transcript")
                    if header_end_idx != -1:
                        # Extract header and add a recording timestamp
                        header = existing_content[:header_end_idx]
                        original_transcript = existing_content[header_end_idx + len("## Transcript"):]
                        
                        # Get timestamp from the header if available
                        original_timestamp = "Unknown time"
                        date_line = [line for line in header.split('\n') if "**Date:**" in line]
                        if date_line:
                            original_timestamp = date_line[0].replace("**Date:**", "").strip()
                        
                        # Reformat with timestamp sections
                        content = header
                        content += f"\n\n## Recording at {original_timestamp}\n"
                        content += original_transcript.strip()
                        content += "\n\n---\n"
                        content += new_section
                    else:
                        # Couldn't find expected format, just append
                        content = existing_content + new_section
                else:
                    # Already has multiple recordings, just append
                    content = existing_content + new_section
            else:
                # Create new file with header
                content = f"# Voice Transcriptions: {username} in {channel_name}\n\n"
                content += f"**Speaker:** {username}\n\n"
                content += f"**Channel:** {channel_name}\n\n"
                content += new_section
            
            # Write the file
            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            logger.info(f"Saved/updated transcription for {username} to {note_path}")
            
        except Exception as e:
            logger.error(f"Error saving transcription to Obsidian: {e}", exc_info=True)
    
    def create_transcription_content(self, channel_name: str, username: str, transcript: str) -> str:
        """Create formatted content for Obsidian note"""
        # This method is kept for backward compatibility but is no longer used directly
        # The formatting is now handled directly in save_transcription
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        content = f"# Voice Transcription: {username} in {channel_name}\n\n"
        content += f"**Date:** {timestamp}\n\n"
        content += f"**Speaker:** {username}\n\n"
        content += f"**Channel:** {channel_name}\n\n"
        content += "## Transcript\n\n"
        content += transcript + "\n\n"
        
        return content
    
    async def save_current_audio_chunks(self, channel_id: int) -> Dict[str, List[Path]]:
        """
        Save current audio chunks for all users in a channel without stopping recording
        
        Args:
            channel_id: Discord voice channel ID
            
        Returns:
            Dictionary mapping usernames to lists of audio file paths
        """
        async with self.connection_lock:
            if channel_id not in self.monitored_channels:
                logger.warning(f"Not monitoring voice channel {channel_id}")
                return {}
                
            # Force save chunks for all recorders in this channel
            user_audio_files = {}
            for (recorder_channel_id, user_id), recorder in list(self.user_recorders.items()):
                if recorder_channel_id == channel_id:
                    audio_file = recorder.save_chunk()
                    if audio_file:
                        if recorder.username not in user_audio_files:
                            user_audio_files[recorder.username] = []
                        user_audio_files[recorder.username].append(audio_file)
            
            logger.info(f"Manually saved audio chunks for {len(user_audio_files)} users in channel {channel_id}")
            
            # Process audio files with Whisper
            for username, audio_files in user_audio_files.items():
                for audio_file in audio_files:
                    await self.transcribe_audio_file(channel_id, username, audio_file)
                    
            return user_audio_files

    def _load_ignored_bots(self) -> Set[str]:
        """
        Load the ignored bots list from config file
        
        Returns:
            Set of bot names to ignore
        """
        try:
            # Ensure the config directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    return set(config.get('ignored_bots', []))
        except Exception as e:
            logger.error(f"Error loading ignored bots list: {e}", exc_info=True)
        
        return set()
    
    def _save_ignored_bots(self) -> bool:
        """
        Save the ignored bots list to config file
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure the config directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create config with the current ignored bots list
            config = {
                'ignored_bots': list(self.ignored_bot_names),
                'last_updated': datetime.datetime.now().isoformat()
            }
            
            # Write to file
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.debug(f"Saved ignored bots list to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving ignored bots list: {e}", exc_info=True)
            return False

    # Add methods to manage ignored bots
    def add_ignored_bot(self, bot_name: str) -> bool:
        """
        Add a bot name to the ignore list
        
        Args:
            bot_name: Name of the bot to ignore
            
        Returns:
            True if the bot was added, False if it was already in the list
        """
        if bot_name in self.ignored_bot_names:
            return False
        self.ignored_bot_names.add(bot_name)
        logger.info(f"Added '{bot_name}' to ignored bots list")
        self._save_ignored_bots()
        return True
        
    def remove_ignored_bot(self, bot_name: str) -> bool:
        """
        Remove a bot name from the ignore list
        
        Args:
            bot_name: Name of the bot to stop ignoring
            
        Returns:
            True if the bot was removed, False if it wasn't in the list
        """
        if bot_name not in self.ignored_bot_names:
            return False
        self.ignored_bot_names.remove(bot_name)
        logger.info(f"Removed '{bot_name}' from ignored bots list")
        self._save_ignored_bots()
        return True
        
    def get_ignored_bots(self) -> List[str]:
        """
        Get the list of bot names being ignored
        
        Returns:
            List of ignored bot names
        """
        return list(self.ignored_bot_names)
        
    def is_ignored_bot(self, user) -> bool:
        """
        Check if a user should be ignored (either a bot or on the ignored list)
        
        Args:
            user: Discord user object
            
        Returns:
            True if the user should be ignored
        """
        if not user:
            return True
            
        # Check if user is a bot or has a name in our ignored list
        if getattr(user, 'bot', False) or user.name in self.ignored_bot_names:
            return True
            
        return False

async def process_and_enhance_transcript(transcript, metadata):
    """Add Obsidian-friendly enhancements to transcripts"""
    summary = await generate_summary(transcript)
    topics = await extract_topics(transcript)
    sentiment = await analyze_sentiment(transcript)
    
    enhanced_content = f"""---
type: discord-voice
channel: {metadata['channel']}
speaker: {metadata['username']}
date: {metadata['date']}
duration: {metadata['duration_seconds']}
topics: {topics}
sentiment: {sentiment}
summary: "{summary}"
---

# Voice Recording: {metadata['username']} in {metadata['channel']}

## Summary
{summary}

## Transcript
{transcript}
"""
    return enhanced_content 