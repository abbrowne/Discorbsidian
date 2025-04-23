#!/usr/bin/env python
"""
Command-line script to process Discord logs in Obsidian

This script provides a simple interface to run the text processor
on an Obsidian vault containing Discord logs. It organizes the logs
into a more structured format with proper metadata and indexes.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

from src.obsidian.text_processor import process_obsidian_logs

def main():
    """Main function to process Discord logs in Obsidian vault"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process Discord logs in Obsidian vault")
    parser.add_argument("--vault", "-v", help="Path to Obsidian vault", type=str)
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get vault path from arguments or environment
    vault_path = args.vault or os.environ.get("OBSIDIAN_VAULT_PATH")
    
    if not vault_path:
        print("Error: Obsidian vault path not provided")
        print("Please provide a path using --vault argument or set OBSIDIAN_VAULT_PATH environment variable")
        sys.exit(1)
    
    # Normalize path
    vault_path = os.path.abspath(vault_path)
    
    # Check if vault directory exists
    if not os.path.isdir(vault_path):
        print(f"Error: Vault directory not found at {vault_path}")
        sys.exit(1)
    
    print(f"Processing Discord logs in Obsidian vault: {vault_path}")
    
    # Process the logs
    files_processed, messages_processed = process_obsidian_logs(vault_path)
    
    print(f"Processing complete! Processed {files_processed} files containing {messages_processed} messages.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 