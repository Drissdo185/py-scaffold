"""Project generator logic."""

import shutil
from pathlib import Path
from typing import Dict, Any

from .templates import get_template_manager


class ProjectGenerator:
    """Handles project generation from templates."""

    def __init__(
        self,
        project_name: str,
        template_name: str,
        output_dir: Path,
        force: bool = False,
    ):
        self.project_name = project_name
        self.template_name = template_name
        self.output_dir = output_dir
        self.force = force
        self.project_path = output_dir / project_name

    def generate(self) -> Path:
        """Generate the project from template."""
        self._validate()
        self._create_project_dir()

        template_manager = get_template_manager(self.template_name)
        context = self._build_context()

        template_manager.render(self.project_path, context)

        return self.project_path

    def _validate(self) -> None:
        """Validate project generation parameters."""
        if self.project_path.exists() and not self.force:
            raise ValueError(
                f"Directory '{self.project_name}' already exists. "
                "Use --force to overwrite."
            )

    def _create_project_dir(self) -> None:
        """Create the project directory."""
        if self.project_path.exists() and self.force:
            shutil.rmtree(self.project_path)

        self.project_path.mkdir(parents=True, exist_ok=True)

    def _build_context(self) -> Dict[str, Any]:
        """Build template context variables."""
        return {
            "project_name": self.project_name,
            "project_name_snake": self.project_name.replace("-", "_"),
            "project_name_class": self._to_class_name(self.project_name),
        }

    @staticmethod
    def _to_class_name(name: str) -> str:
        """Convert project name to PascalCase class name."""
        return "".join(word.capitalize() for word in name.replace("-", "_").split("_"))
