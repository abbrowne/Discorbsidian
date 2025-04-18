import os
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

class ObsidianVault:
    def __init__(self):
        load_dotenv()
        self.vault_path = Path(os.getenv('OBSIDIAN_VAULT_PATH'))
        if not self.vault_path.exists():
            raise ValueError(f"Obsidian vault path does not exist: {self.vault_path}")

    def create_note(self, title: str, content: str, folder: Optional[str] = None) -> Path:
        """
        Create a new note in the Obsidian vault
        
        Args:
            title: The title of the note
            content: The content of the note
            folder: Optional subfolder to create the note in
            
        Returns:
            Path to the created note
        """
        # Sanitize the title for use as a filename
        filename = f"{title}.md"
        
        # Determine the full path for the note
        if folder:
            note_path = self.vault_path / folder / filename
            note_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            note_path = self.vault_path / filename
            
        # Write the note
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return note_path

    def read_note(self, path: str) -> str:
        """
        Read the content of a note
        
        Args:
            path: Path to the note relative to the vault root
            
        Returns:
            Content of the note
        """
        note_path = self.vault_path / path
        if not note_path.exists():
            raise FileNotFoundError(f"Note not found: {path}")
            
        with open(note_path, 'r', encoding='utf-8') as f:
            return f.read()

    def list_notes(self, folder: Optional[str] = None) -> List[Path]:
        """
        List all notes in the vault or a specific folder
        
        Args:
            folder: Optional subfolder to list notes from
            
        Returns:
            List of paths to notes
        """
        if folder:
            search_path = self.vault_path / folder
        else:
            search_path = self.vault_path
            
        return list(search_path.glob('**/*.md')) 