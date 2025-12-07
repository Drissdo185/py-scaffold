"""Backend API template."""

from pathlib import Path
from typing import Dict, Any

from .base import TemplateManager


class BackendAPITemplate(TemplateManager):
    """Template for Backend API projects."""

    def render(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Render the Backend API template."""
        project_name = context["project_name_snake"]

        # Create directory structure
        self._create_structure(output_path)

        # Generate files
        self._generate_config_yaml(output_path, context)
        self._generate_main_py(output_path, context)
        self._generate_requirements(output_path, context)
        self._generate_readme(output_path, context)
        self._generate_gitignore(output_path)

        # Core
        self._generate_core_config(output_path, context)
        self._generate_core_init(output_path)

        # Model
        self._generate_model_user(output_path, context)
        self._generate_model_init(output_path)

        # Repository
        self._generate_repository_user(output_path, context)
        self._generate_repository_init(output_path)

        # Service
        self._generate_service_user(output_path, context)
        self._generate_service_init(output_path)

        # Controller
        self._generate_controller_user(output_path, context)
        self._generate_controller_init(output_path)

        # DTO
        self._generate_dto_user(output_path, context)
        self._generate_dto_init(output_path)

        # App init
        self._generate_app_init(output_path)

        # Tests
        self._generate_tests_init(output_path)

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
