#!/usr/bin/env python
"""
Simple runner script for the Discorbsidian bot
Run this from the project root with: python run_bot.py
"""
import sys
import os
from pathlib import Path

# Ensure the current directory is in the path so imports work correctly
sys.path.insert(0, str(Path(__file__).parent.absolute()))

print("Starting Discorbsidian bot...")

# Import the main module and call its main function
if __name__ == "__main__":
    from src.bot import main as bot_main
    bot_main.main() 