"""Backend API template."""

from pathlib import Path
from typing import Dict, Any

from .base import TemplateManager


class BackendAPITemplate(TemplateManager):
    """Template for Backend API projects."""

    def render(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Render the Backend API template."""
        # Create directory structure
        self._create_structure(output_path)

        # Generate non-Python files only
        self._generate_config_yaml(output_path, context)
        self._generate_requirements(output_path, context)
        self._generate_readme(output_path, context)
        self._generate_gitignore(output_path)

        # Documentation for AI assistants
        self._generate_claude_md(output_path, context)
        self._generate_copilot_instruction(output_path, context)

    def _create_structure(self, output_path: Path) -> None:
        """Create directory structure."""
        dirs = [
            "src/app/core",
            "src/app/model",
            "src/app/repository",
            "src/app/service",
            "src/app/controller",
            "src/app/dto",
            "tests",
            ".github",
        ]

        for dir_path in dirs:
            self._create_dir(output_path / dir_path)

    def _generate_config_yaml(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate config.yaml."""
        content = """# Application Configuration
app:
  name: {project_name}
  version: "1.0.0"
  environment: development

server:
  host: "0.0.0.0"
  port: 8000

database:
  host: localhost
  port: 5432
  name: {project_name_snake}_db
  user: postgres
  password: postgres

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
""".format(**context)

        self._create_file(output_path / "src/config.yaml", content)

    def _generate_main_py(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate main.py."""
        content = '''"""Main application entry point."""

from app.core.config import settings
from app.controller.user_controller import UserController


def main() -> None:
    """Run the application."""
    print(f"Starting {settings.app.name} v{settings.app.version}")
    print(f"Environment: {settings.app.environment}")
    print(f"Server: {settings.server.host}:{settings.server.port}")

    # Example usage
    user_controller = UserController()

    # Create a user
    user = user_controller.create_user("John Doe", "john@example.com")
    print(f"\\nCreated user: {user}")

    # Get user by ID
    retrieved_user = user_controller.get_user(user["id"])
    print(f"Retrieved user: {retrieved_user}")

    # List all users
    all_users = user_controller.list_users()
    print(f"All users: {all_users}")


if __name__ == "__main__":
    main()
'''

        self._create_file(output_path / "src/main.py", content)

    def _generate_requirements(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate requirements.txt."""
        content = """# Core dependencies

# Database (optional - uncomment if needed)


# API Framework (optional - uncomment if needed)


# Development dependencies

"""

        self._create_file(output_path / "requirements.txt", content)

    def _generate_readme(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate README.md."""
        content = """# {project_name}

A production-ready Backend API project generated with py-scaffold.

## Project Structure

```
src/
├── config.yaml               # YAML configuration
├── main.py
├── app/
│   ├── __init__.py
│   ├── core/
│   │   ├── config.py         # Load YAML config into Settings object
│   ├── model/                # Domain Models / ORM Entities
│   │   ├── __init__.py
│   │   └── user.py
│   ├── repository/           # Data access layer
│   │   ├── __init__.py
│   │   └── user_repository.py
│   ├── controller/           # Request handlers / API controllers
│   │   ├── __init__.py
│   │   └── user_controller.py
│   ├── dto/                  # Data Transfer Objects
│   │   ├── __init__.py
│   │   └── user_dto.py
│   └── service/              # Business logic
│       ├── __init__.py
│       └── user_service.py
└── tests/
    ├── __init__.py
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your application:
Edit `src/config.yaml` to match your environment.

4. Run the application:
```bash
python src/main.py
```

## Architecture

This project follows a layered architecture:

- **Model Layer**: Domain entities and data models
- **Repository Layer**: Data access and persistence
- **Service Layer**: Business logic and orchestration
- **Controller Layer**: Request handling and response formatting
- **DTO Layer**: Data transfer objects for API contracts

## Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black src/ tests/
isort src/ tests/
```

Type checking:
```bash
mypy src/
```

## License

MIT
""".format(**context)

        self._create_file(output_path / "README.md", content)

    def _generate_gitignore(self, output_path: Path) -> None:
        """Generate .gitignore."""
        content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store

# Environment variables
.env
.env.*

# Database
*.db
*.sqlite3

# Logs
*.log
"""

        self._create_file(output_path / ".gitignore", content)

    def _generate_core_config(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate core/config.py."""
        content = '''"""Configuration management."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class AppConfig(BaseModel):
    """Application configuration."""

    name: str
    version: str
    environment: str


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str
    port: int


class DatabaseConfig(BaseModel):
    """Database configuration."""

    host: str
    port: int
    name: str
    user: str
    password: str


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str
    format: str


class Settings(BaseSettings):
    """Application settings loaded from YAML config."""

    app: AppConfig
    server: ServerConfig
    database: DatabaseConfig
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Settings":
        """Load settings from YAML file."""
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        return cls(
            app=AppConfig(**config_data["app"]),
            server=ServerConfig(**config_data["server"]),
            database=DatabaseConfig(**config_data["database"]),
            logging=LoggingConfig(**config_data["logging"]),
        )


# Load settings from config.yaml
config_path = Path(__file__).parent.parent.parent / "config.yaml"
settings = Settings.from_yaml(config_path)
'''

        self._create_file(output_path / "src/app/core/config.py", content)

    def _generate_core_init(self, output_path: Path) -> None:
        """Generate core/__init__.py."""
        content = '''"""Core module."""

from .config import settings

__all__ = ["settings"]
'''

        self._create_file(output_path / "src/app/core/__init__.py", content)

    def _generate_model_user(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate model/user.py."""
        content = '''"""User domain model."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class User:
    """User entity."""

    id: int
    name: str
    email: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    def __str__(self) -> str:
        return f"User(id={self.id}, name={self.name}, email={self.email})"
'''

        self._create_file(output_path / "src/app/model/user.py", content)

    def _generate_model_init(self, output_path: Path) -> None:
        """Generate model/__init__.py."""
        content = '''"""Domain models."""

from .user import User

__all__ = ["User"]
'''

        self._create_file(output_path / "src/app/model/__init__.py", content)

    def _generate_repository_user(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate repository/user_repository.py."""
        content = '''"""User repository for data access."""

from datetime import datetime
from typing import List, Optional

from app.model.user import User


class UserRepository:
    """Repository for User entity data access."""

    def __init__(self) -> None:
        """Initialize the repository with in-memory storage."""
        self._users: dict[int, User] = {}
        self._next_id: int = 1

    def create(self, name: str, email: str) -> User:
        """Create a new user."""
        user = User(
            id=self._next_id,
            name=name,
            email=email,
            created_at=datetime.now(),
        )
        self._users[user.id] = user
        self._next_id += 1
        return user

    def get_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)

    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        for user in self._users.values():
            if user.email == email:
                return user
        return None

    def list_all(self) -> List[User]:
        """List all users."""
        return list(self._users.values())

    def update(self, user_id: int, name: Optional[str] = None, email: Optional[str] = None) -> Optional[User]:
        """Update user."""
        user = self._users.get(user_id)
        if not user:
            return None

        if name:
            user.name = name
        if email:
            user.email = email
        user.updated_at = datetime.now()

        return user

    def delete(self, user_id: int) -> bool:
        """Delete user."""
        if user_id in self._users:
            del self._users[user_id]
            return True
        return False
'''

        self._create_file(output_path / "src/app/repository/user_repository.py", content)

    def _generate_repository_init(self, output_path: Path) -> None:
        """Generate repository/__init__.py."""
        content = '''"""Repository layer."""

from .user_repository import UserRepository

__all__ = ["UserRepository"]
'''

        self._create_file(output_path / "src/app/repository/__init__.py", content)

    def _generate_service_user(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate service/user_service.py."""
        content = '''"""User service for business logic."""

from typing import List, Optional

from app.model.user import User
from app.repository.user_repository import UserRepository


class UserService:
    """Service layer for user business logic."""

    def __init__(self, repository: UserRepository) -> None:
        """Initialize service with repository."""
        self.repository = repository

    def create_user(self, name: str, email: str) -> User:
        """Create a new user with business logic validation."""
        # Business logic: Check if email already exists
        existing_user = self.repository.get_by_email(email)
        if existing_user:
            raise ValueError(f"User with email {email} already exists")

        # Business logic: Validate email format (simplified)
        if "@" not in email:
            raise ValueError("Invalid email format")

        # Business logic: Validate name
        if not name or len(name.strip()) == 0:
            raise ValueError("Name cannot be empty")

        return self.repository.create(name.strip(), email.lower())

    def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return self.repository.get_by_id(user_id)

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.repository.get_by_email(email.lower())

    def list_users(self) -> List[User]:
        """List all users."""
        return self.repository.list_all()

    def update_user(
        self, user_id: int, name: Optional[str] = None, email: Optional[str] = None
    ) -> Optional[User]:
        """Update user with business logic validation."""
        # Business logic: Validate new email if provided
        if email:
            if "@" not in email:
                raise ValueError("Invalid email format")
            # Check if email is already taken by another user
            existing_user = self.repository.get_by_email(email)
            if existing_user and existing_user.id != user_id:
                raise ValueError(f"Email {email} is already taken")

        # Business logic: Validate name if provided
        if name is not None and len(name.strip()) == 0:
            raise ValueError("Name cannot be empty")

        return self.repository.update(
            user_id, name.strip() if name else None, email.lower() if email else None
        )

    def delete_user(self, user_id: int) -> bool:
        """Delete user."""
        return self.repository.delete(user_id)
'''

        self._create_file(output_path / "src/app/service/user_service.py", content)

    def _generate_service_init(self, output_path: Path) -> None:
        """Generate service/__init__.py."""
        content = '''"""Service layer."""

from .user_service import UserService

__all__ = ["UserService"]
'''

        self._create_file(output_path / "src/app/service/__init__.py", content)

    def _generate_controller_user(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate controller/user_controller.py."""
        content = '''"""User controller for handling requests."""

from typing import Dict, Any, List, Optional

from app.dto.user_dto import UserCreateDTO, UserResponseDTO, UserUpdateDTO
from app.repository.user_repository import UserRepository
from app.service.user_service import UserService


class UserController:
    """Controller for user-related requests."""

    def __init__(self) -> None:
        """Initialize controller with dependencies."""
        self.repository = UserRepository()
        self.service = UserService(self.repository)

    def create_user(self, name: str, email: str) -> Dict[str, Any]:
        """
        Create a new user.

        Args:
            name: User name
            email: User email

        Returns:
            User response DTO as dict
        """
        try:
            user = self.service.create_user(name, email)
            response_dto = UserResponseDTO.from_model(user)
            return response_dto.to_dict()
        except ValueError as e:
            return {"error": str(e)}

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get user by ID.

        Args:
            user_id: User ID

        Returns:
            User response DTO as dict or None
        """
        user = self.service.get_user(user_id)
        if user:
            response_dto = UserResponseDTO.from_model(user)
            return response_dto.to_dict()
        return None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get user by email.

        Args:
            email: User email

        Returns:
            User response DTO as dict or None
        """
        user = self.service.get_user_by_email(email)
        if user:
            response_dto = UserResponseDTO.from_model(user)
            return response_dto.to_dict()
        return None

    def list_users(self) -> List[Dict[str, Any]]:
        """
        List all users.

        Returns:
            List of user response DTOs as dicts
        """
        users = self.service.list_users()
        return [UserResponseDTO.from_model(user).to_dict() for user in users]

    def update_user(
        self, user_id: int, name: Optional[str] = None, email: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update user.

        Args:
            user_id: User ID
            name: New name (optional)
            email: New email (optional)

        Returns:
            Updated user response DTO as dict or None
        """
        try:
            user = self.service.update_user(user_id, name, email)
            if user:
                response_dto = UserResponseDTO.from_model(user)
                return response_dto.to_dict()
            return None
        except ValueError as e:
            return {"error": str(e)}

    def delete_user(self, user_id: int) -> Dict[str, Any]:
        """
        Delete user.

        Args:
            user_id: User ID

        Returns:
            Success status
        """
        success = self.service.delete_user(user_id)
        return {"success": success}
'''

        self._create_file(output_path / "src/app/controller/user_controller.py", content)

    def _generate_controller_init(self, output_path: Path) -> None:
        """Generate controller/__init__.py."""
        content = '''"""Controller layer."""

from .user_controller import UserController

__all__ = ["UserController"]
'''

        self._create_file(output_path / "src/app/controller/__init__.py", content)

    def _generate_dto_user(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate dto/user_dto.py."""
        content = '''"""User Data Transfer Objects."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

from app.model.user import User


@dataclass
class UserCreateDTO:
    """DTO for creating a user."""

    name: str
    email: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"name": self.name, "email": self.email}


@dataclass
class UserUpdateDTO:
    """DTO for updating a user."""

    name: Optional[str] = None
    email: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {}
        if self.name is not None:
            data["name"] = self.name
        if self.email is not None:
            data["email"] = self.email
        return data


@dataclass
class UserResponseDTO:
    """DTO for user response."""

    id: int
    name: str
    email: str
    created_at: str
    updated_at: Optional[str] = None

    @classmethod
    def from_model(cls, user: User) -> "UserResponseDTO":
        """Create DTO from User model."""
        return cls(
            id=user.id,
            name=user.name,
            email=user.email,
            created_at=user.created_at.isoformat(),
            updated_at=user.updated_at.isoformat() if user.updated_at else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
'''

        self._create_file(output_path / "src/app/dto/user_dto.py", content)

    def _generate_dto_init(self, output_path: Path) -> None:
        """Generate dto/__init__.py."""
        content = '''"""Data Transfer Objects."""

from .user_dto import UserCreateDTO, UserUpdateDTO, UserResponseDTO

__all__ = ["UserCreateDTO", "UserUpdateDTO", "UserResponseDTO"]
'''

        self._create_file(output_path / "src/app/dto/__init__.py", content)

    def _generate_app_init(self, output_path: Path) -> None:
        """Generate app/__init__.py."""
        content = '''"""Application package."""
'''

        self._create_file(output_path / "src/app/__init__.py", content)

    def _generate_tests_init(self, output_path: Path) -> None:
        """Generate tests/__init__.py."""
        content = '''"""Tests package."""
'''

        self._create_file(output_path / "tests/__init__.py", content)

    def _generate_claude_md(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate CLAUDE.md for Claude Code."""
        content = """# {project_name} - Project Documentation for Claude Code

This document provides an overview of the project structure and architecture for AI-assisted development with Claude Code.

## Project Overview

This is a production-ready Backend API project following a clean layered architecture pattern.

**Project Name**: {project_name}
**Architecture**: Layered (MVC-inspired with additional Service and Repository layers)
**Configuration**: YAML-based with Pydantic validation

## Project Structure

```
{project_name}/
├── src/
│   ├── config.yaml               # Application configuration (YAML format)
│   ├── main.py                   # Application entry point
│   └── app/
│       ├── core/                 # Core functionality
│       │   ├── __init__.py
│       │   └── config.py         # Configuration management (loads YAML into Pydantic models)
│       ├── model/                # Domain Models / Entities
│       │   ├── __init__.py
│       │   └── user.py           # User entity (dataclass)
│       ├── repository/           # Data Access Layer
│       │   ├── __init__.py
│       │   └── user_repository.py # User data persistence
│       ├── service/              # Business Logic Layer
│       │   ├── __init__.py
│       │   └── user_service.py   # User business logic & validation
│       ├── controller/           # Request Handling Layer
│       │   ├── __init__.py
│       │   └── user_controller.py # User request handlers
│       └── dto/                  # Data Transfer Objects
│           ├── __init__.py
│           └── user_dto.py       # User API contracts
├── tests/                        # Test files
│   └── __init__.py
├── docs/                         # Documentation
│   ├── Claude.md                 # This file
│   └── copilot_instruction.md    # GitHub Copilot instructions
├── requirements.txt              # Python dependencies
├── README.md                     # Project README
└── .gitignore

```

## Architectural Layers

### 1. Model Layer (`src/app/model/`)
**Purpose**: Define domain entities and data structures

**Responsibilities**:
- Define core business entities using dataclasses
- Represent the domain model
- No business logic or data access code

**Example**: `user.py` defines the User entity with id, name, email, and timestamps

**When to modify**:
- Adding new domain entities
- Adding/removing fields from existing entities
- Changing data types or relationships

### 2. Repository Layer (`src/app/repository/`)
**Purpose**: Abstract data access and persistence

**Responsibilities**:
- CRUD operations (Create, Read, Update, Delete)
- Database/storage interaction
- Query logic
- Data persistence abstraction

**Example**: `user_repository.py` provides methods like `create()`, `get_by_id()`, `get_by_email()`, `list_all()`, `update()`, `delete()`

**Current Implementation**: In-memory storage (dictionary-based)

**When to modify**:
- Switching to real database (PostgreSQL, MongoDB, etc.)
- Adding new query methods
- Optimizing data access patterns

### 3. Service Layer (`src/app/service/`)
**Purpose**: Implement business logic and orchestration

**Responsibilities**:
- Business rules and validation
- Orchestrate multiple repositories
- Transaction management
- Business logic workflows

**Example**: `user_service.py` validates email format, checks for duplicates, enforces business rules

**When to modify**:
- Adding new business rules
- Complex operations involving multiple entities
- Validation logic changes

### 4. Controller Layer (`src/app/controller/`)
**Purpose**: Handle requests and responses

**Responsibilities**:
- Request handling
- Input validation and parsing
- Response formatting
- Error handling
- Coordinate between Service and DTO layers

**Example**: `user_controller.py` handles user-related requests, converts between DTOs and models

**When to modify**:
- Adding new API endpoints
- Changing request/response formats
- Adding error handling

### 5. DTO Layer (`src/app/dto/`)
**Purpose**: Define API contracts and data transfer objects

**Responsibilities**:
- Define request/response schemas
- Data transformation between API and domain models
- API versioning

**Example**: `user_dto.py` defines `UserCreateDTO`, `UserUpdateDTO`, `UserResponseDTO`

**When to modify**:
- Changing API contracts
- Adding new endpoints
- API versioning

### 6. Core Layer (`src/app/core/`)
**Purpose**: Shared functionality and configuration

**Responsibilities**:
- Configuration management
- Shared utilities
- Cross-cutting concerns

**Example**: `config.py` loads YAML configuration into Pydantic Settings objects

## Configuration Management

**File**: `src/config.yaml`
**Parser**: `src/app/core/config.py`

The project uses YAML for configuration with Pydantic for type-safe validation:

1. Configuration is defined in `config.yaml`
2. Pydantic models in `core/config.py` define the structure
3. Settings are loaded at startup and available as `settings` singleton

**Configuration Sections**:
- `app`: Application metadata (name, version, environment)
- `server`: Server settings (host, port)
- `database`: Database connection details
- `logging`: Logging configuration

## Development Workflow

### Adding a New Entity

1. **Model**: Create entity in `app/model/`
2. **Repository**: Create repository in `app/repository/`
3. **Service**: Create service with business logic in `app/service/`
4. **DTO**: Create DTOs for API contracts in `app/dto/`
5. **Controller**: Create controller to handle requests in `app/controller/`
6. **Tests**: Add tests in `tests/`

### Common Tasks

**Adding a new field to User**:
1. Update `model/user.py`
2. Update `repository/user_repository.py` CRUD methods
3. Update `dto/user_dto.py` DTOs
4. Update service validation if needed
5. Update tests

**Adding business logic**:
- Add to appropriate Service class
- Keep business logic OUT of Repository and Controller layers

**Changing database**:
- Modify Repository layer only
- Service and Controller layers should remain unchanged

## Design Principles

1. **Separation of Concerns**: Each layer has a single, well-defined responsibility
2. **Dependency Direction**: Controller → Service → Repository → Model
3. **Dependency Injection**: Dependencies are passed through constructors
4. **Type Safety**: Full type hints throughout the codebase
5. **Testability**: Each layer can be tested in isolation with mocks

## Testing Strategy

- **Unit Tests**: Test each layer independently with mocks
- **Integration Tests**: Test layer interactions
- **E2E Tests**: Test complete request flows

## Notes for AI Assistants

- Always maintain layer separation
- Don't put business logic in Controllers or Repositories
- Keep DTOs separate from Models
- Use dependency injection
- Follow existing patterns when adding new features
- Validate input at Service layer
- Handle errors at Controller layer
""".format(**context)

        self._create_file(output_path / "CLAUDE.md", content)

    def _generate_copilot_instruction(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate .github/copilot-instructions.md for GitHub Copilot."""
        content = """# GitHub Copilot Instructions - {project_name}

## Project Overview

This is a production-ready Backend API project following a clean layered architecture pattern.

**Project Name**: {project_name}
**Architecture**: Layered (MVC-inspired with additional Service and Repository layers)
**Configuration**: YAML-based with Pydantic validation

## Project Structure

```
{project_name}/
├── src/
│   ├── config.yaml               # Application configuration (YAML format)
│   ├── main.py                   # Application entry point
│   └── app/
│       ├── core/                 # Core functionality
│       │   ├── __init__.py
│       │   └── config.py         # Configuration management (loads YAML into Pydantic models)
│       ├── model/                # Domain Models / Entities
│       │   ├── __init__.py
│       │   └── user.py           # User entity (dataclass)
│       ├── repository/           # Data Access Layer
│       │   ├── __init__.py
│       │   └── user_repository.py # User data persistence
│       ├── service/              # Business Logic Layer
│       │   ├── __init__.py
│       │   └── user_service.py   # User business logic & validation
│       ├── controller/           # Request Handling Layer
│       │   ├── __init__.py
│       │   └── user_controller.py # User request handlers
│       └── dto/                  # Data Transfer Objects
│           ├── __init__.py
│           └── user_dto.py       # User API contracts
├── tests/                        # Test files
│   └── __init__.py
├── docs/                         # Documentation
│   ├── Claude.md                 # This file
│   └── copilot_instruction.md    # GitHub Copilot instructions
├── requirements.txt              # Python dependencies
├── README.md                     # Project README
└── .gitignore

```

## Architectural Layers

### 1. Model Layer (`src/app/model/`)
**Purpose**: Define domain entities and data structures

**Responsibilities**:
- Define core business entities using dataclasses
- Represent the domain model
- No business logic or data access code

**Example**: `user.py` defines the User entity with id, name, email, and timestamps

**When to modify**:
- Adding new domain entities
- Adding/removing fields from existing entities
- Changing data types or relationships

### 2. Repository Layer (`src/app/repository/`)
**Purpose**: Abstract data access and persistence

**Responsibilities**:
- CRUD operations (Create, Read, Update, Delete)
- Database/storage interaction
- Query logic
- Data persistence abstraction

**Example**: `user_repository.py` provides methods like `create()`, `get_by_id()`, `get_by_email()`, `list_all()`, `update()`, `delete()`

**Current Implementation**: In-memory storage (dictionary-based)

**When to modify**:
- Switching to real database (PostgreSQL, MongoDB, etc.)
- Adding new query methods
- Optimizing data access patterns

### 3. Service Layer (`src/app/service/`)
**Purpose**: Implement business logic and orchestration

**Responsibilities**:
- Business rules and validation
- Orchestrate multiple repositories
- Transaction management
- Business logic workflows

**Example**: `user_service.py` validates email format, checks for duplicates, enforces business rules

**When to modify**:
- Adding new business rules
- Complex operations involving multiple entities
- Validation logic changes

### 4. Controller Layer (`src/app/controller/`)
**Purpose**: Handle requests and responses

**Responsibilities**:
- Request handling
- Input validation and parsing
- Response formatting
- Error handling
- Coordinate between Service and DTO layers

**Example**: `user_controller.py` handles user-related requests, converts between DTOs and models

**When to modify**:
- Adding new API endpoints
- Changing request/response formats
- Adding error handling

### 5. DTO Layer (`src/app/dto/`)
**Purpose**: Define API contracts and data transfer objects

**Responsibilities**:
- Define request/response schemas
- Data transformation between API and domain models
- API versioning

**Example**: `user_dto.py` defines `UserCreateDTO`, `UserUpdateDTO`, `UserResponseDTO`

**When to modify**:
- Changing API contracts
- Adding new endpoints
- API versioning

### 6. Core Layer (`src/app/core/`)
**Purpose**: Shared functionality and configuration

**Responsibilities**:
- Configuration management
- Shared utilities
- Cross-cutting concerns

**Example**: `config.py` loads YAML configuration into Pydantic Settings objects

## Configuration Management

**File**: `src/config.yaml`
**Parser**: `src/app/core/config.py`

The project uses YAML for configuration with Pydantic for type-safe validation:

1. Configuration is defined in `config.yaml`
2. Pydantic models in `core/config.py` define the structure
3. Settings are loaded at startup and available as `settings` singleton

**Configuration Sections**:
- `app`: Application metadata (name, version, environment)
- `server`: Server settings (host, port)
- `database`: Database connection details
- `logging`: Logging configuration

## Development Workflow

### Adding a New Entity

1. **Model**: Create entity in `app/model/`
2. **Repository**: Create repository in `app/repository/`
3. **Service**: Create service with business logic in `app/service/`
4. **DTO**: Create DTOs for API contracts in `app/dto/`
5. **Controller**: Create controller to handle requests in `app/controller/`
6. **Tests**: Add tests in `tests/`

### Common Tasks

**Adding a new field to User**:
1. Update `model/user.py`
2. Update `repository/user_repository.py` CRUD methods
3. Update `dto/user_dto.py` DTOs
4. Update service validation if needed
5. Update tests

**Adding business logic**:
- Add to appropriate Service class
- Keep business logic OUT of Repository and Controller layers

**Changing database**:
- Modify Repository layer only
- Service and Controller layers should remain unchanged

## Design Principles

1. **Separation of Concerns**: Each layer has a single, well-defined responsibility
2. **Dependency Direction**: Controller → Service → Repository → Model
3. **Dependency Injection**: Dependencies are passed through constructors
4. **Type Safety**: Full type hints throughout the codebase
5. **Testability**: Each layer can be tested in isolation with mocks

## Testing Strategy

- **Unit Tests**: Test each layer independently with mocks
- **Integration Tests**: Test layer interactions
- **E2E Tests**: Test complete request flows

## Notes for AI Assistants

- Always maintain layer separation
- Don't put business logic in Controllers or Repositories
- Keep DTOs separate from Models
- Use dependency injection
- Follow existing patterns when adding new features
- Validate input at Service layer
- Handle errors at Controller layer
""".format(**context)

        self._create_file(output_path / ".github/copilot-instructions.md", content)
