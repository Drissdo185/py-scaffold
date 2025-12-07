"""Tests for project generator."""

import tempfile
from pathlib import Path

import pytest

from py_scaffold.generator import ProjectGenerator


def test_project_generator_creates_directory():
    """Test that project generator creates the project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = ProjectGenerator(
            project_name="test-project",
            template_name="backend-api",
            output_dir=Path(tmpdir),
            force=False,
        )

        project_path = generator.generate()

        assert project_path.exists()
        assert project_path.is_dir()
        assert project_path.name == "test-project"


def test_project_generator_creates_structure():
    """Test that project generator creates the expected directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = ProjectGenerator(
            project_name="test-project",
            template_name="backend-api",
            output_dir=Path(tmpdir),
            force=False,
        )

        project_path = generator.generate()

        # Check main files
        assert (project_path / "README.md").exists()
        assert (project_path / "requirements.txt").exists()
        assert (project_path / ".gitignore").exists()
        assert (project_path / "src/config.yaml").exists()
        assert (project_path / "src/main.py").exists()

        # Check directories
        assert (project_path / "src/app/core").exists()
        assert (project_path / "src/app/model").exists()
        assert (project_path / "src/app/repository").exists()
        assert (project_path / "src/app/service").exists()
        assert (project_path / "src/app/controller").exists()
        assert (project_path / "src/app/dto").exists()
        assert (project_path / "tests").exists()


def test_project_generator_raises_error_if_exists():
    """Test that project generator raises error if directory exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create project first time
        generator = ProjectGenerator(
            project_name="test-project",
            template_name="backend-api",
            output_dir=Path(tmpdir),
            force=False,
        )
        generator.generate()

        # Try to create again without force
        generator2 = ProjectGenerator(
            project_name="test-project",
            template_name="backend-api",
            output_dir=Path(tmpdir),
            force=False,
        )

        with pytest.raises(ValueError, match="already exists"):
            generator2.generate()


def test_project_generator_force_overwrite():
    """Test that project generator can force overwrite existing directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create project first time
        generator = ProjectGenerator(
            project_name="test-project",
            template_name="backend-api",
            output_dir=Path(tmpdir),
            force=False,
        )
        generator.generate()

        # Create again with force
        generator2 = ProjectGenerator(
            project_name="test-project",
            template_name="backend-api",
            output_dir=Path(tmpdir),
            force=True,
        )
        project_path = generator2.generate()

        assert project_path.exists()
