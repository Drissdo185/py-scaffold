# Contributing to py-scaffold

Thank you for your interest in contributing to py-scaffold! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions with the community.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git

### Setting Up Development Environment

1. Fork the repository on GitHub

2. Clone your fork:
```bash
git clone https://github.com/yourusername/py-scaffold.git
cd py-scaffold
```

3. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install the package in editable mode with development dependencies:
```bash
pip install -e ".[dev]"
```

## Development Workflow

### 1. Create a Branch

Create a new branch for your feature or bugfix:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

### 2. Make Changes

- Write clean, readable code
- Follow the existing code style
- Add type hints to all functions
- Update documentation as needed

### 3. Code Quality

Before committing, ensure your code passes all quality checks:

#### Format Code
```bash
black src/ tests/
isort src/ tests/
```

#### Type Checking
```bash
mypy src/
```

#### Run Tests
```bash
pytest tests/
```

### 4. Commit Changes

Write clear, descriptive commit messages:
```bash
git add .
git commit -m "feat: add new template for Django projects"
# or
git commit -m "fix: handle edge case in project generation"
```

Commit message format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `refactor:` for code refactoring
- `test:` for test additions or modifications
- `chore:` for maintenance tasks

### 5. Push Changes

```bash
git push origin your-branch-name
```

### 6. Create Pull Request

1. Go to the repository on GitHub
2. Click "New Pull Request"
3. Select your branch
4. Fill in the PR template with:
   - Description of changes
   - Related issue numbers
   - Testing performed
   - Screenshots (if applicable)

## Adding New Templates

To add a new project template:

1. Create a new file in `src/py_scaffold/templates/` (e.g., `django_template.py`)

2. Implement the template class inheriting from `BaseTemplate`:
```python
from .base import BaseTemplate

class DjangoTemplate(BaseTemplate):
    def get_structure(self) -> dict:
        # Define your project structure
        pass

    def get_dependencies(self) -> list[str]:
        # Return list of dependencies
        pass
```

3. Register the template in `src/py_scaffold/templates/__init__.py`

4. Update the CLI in `src/py_scaffold/cli.py` to include the new template

5. Add tests for the new template in `tests/`

6. Update README.md with template documentation

## Testing

### Running Tests

Run all tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=py_scaffold --cov-report=html
```

### Writing Tests

- Add tests for all new features
- Ensure edge cases are covered
- Use descriptive test names
- Follow the existing test structure

## Documentation

- Update README.md for user-facing changes
- Update docstrings for code changes
- Add examples for new features
- Keep CHANGELOG.md updated

## Questions?

If you have questions or need help:
- Open an issue on GitHub
- Check existing issues and discussions
- Review the documentation

## License

By contributing to py-scaffold, you agree that your contributions will be licensed under the MIT License.
