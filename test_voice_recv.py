#!/usr/bin/env python
"""
Test script to check if discord-ext-voice-recv is installed and working properly
"""
import sys
import os

try:
    from discord.ext import voice_recv
    print(f"Successfully imported discord.ext.voice_recv")
    print(f"Version: {getattr(voice_recv, '__version__', 'Not available')}")
    print(f"Module location: {voice_recv.__file__}")
    print(f"Available attributes: {dir(voice_recv)}")
    print("Voice recording is AVAILABLE")
except ImportError as e:
    print(f"Failed to import discord.ext.voice_recv: {e}")
    print("Voice recording is NOT AVAILABLE")

print("\nEnvironment info:")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")
print(f"Working directory: {os.getcwd()}") 