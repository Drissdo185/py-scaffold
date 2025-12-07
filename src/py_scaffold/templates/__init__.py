"""Template management."""

from .base import TemplateManager
from .backend_api import BackendAPITemplate
from .ai_project import AIProjectTemplate


def get_template_manager(template_name: str) -> TemplateManager:
    """Get template manager by name."""
    templates = {
        "backend-api": BackendAPITemplate,
        "ai-project": AIProjectTemplate,
    }

    template_class = templates.get(template_name)
    if not template_class:
        raise ValueError(f"Unknown template: {template_name}")

    return template_class()


__all__ = ["TemplateManager", "BackendAPITemplate", "AIProjectTemplate", "get_template_manager"]
