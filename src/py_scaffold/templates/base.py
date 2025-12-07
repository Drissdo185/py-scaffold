"""Base template manager."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class TemplateManager(ABC):
    """Base class for template managers."""

    @abstractmethod
    def render(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Render the template to the output path."""
        pass

    def _create_file(self, path: Path, content: str) -> None:
        """Create a file with content."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def _create_dir(self, path: Path) -> None:
        """Create a directory."""
        path.mkdir(parents=True, exist_ok=True)
