# py-scaffold

A Python CLI tool for generating production-ready project templates, similar to create-next-app.

## Features

- **Backend API Template**: Production-ready layered architecture with Models, Repositories, Services, Controllers, and DTOs
- **YAML Configuration**: Easy-to-manage configuration files
- **Type-Safe**: Full type hints support
- **Best Practices**: Industry-standard project structure and patterns

## Installation

### From Source

```bash
git clone https://github.com/yourusername/py-scaffold.git
cd py-scaffold
pip install -e .
```

### Using pip (once published)

```bash
pip install py-scaffold
```

## Usage

### Create a new project

```bash
py-scaffold my-api --template backend-api
```

This will create a new project with the following structure:

```
my-api/
├── src/
│   ├── config.yaml               # YAML configuration
│   ├── main.py
│   └── app/
│       ├── core/
│       │   └── config.py         # Load YAML config into Settings object
│       ├── model/                # Domain Models / ORM Entities
│       │   └── user.py
│       ├── repository/           # Data access layer
│       │   └── user_repository.py
│       ├── service/              # Business logic
│       │   └── user_service.py
│       ├── controller/           # Request handlers
│       │   └── user_controller.py
│       └── dto/                  # Data Transfer Objects
│           └── user_dto.py
└── tests/
```

### Command Options

```bash
py-scaffold <project-name> [OPTIONS]

Options:
  -t, --template TEXT     Template to use (default: backend-api)
  -o, --output PATH       Output directory (default: current directory)
  -f, --force             Force overwrite if directory exists
  --help                  Show help message
```

## Templates

### Backend API

A layered architecture template with:

- **Model Layer**: Domain entities and data models
- **Repository Layer**: Data access and persistence abstraction
- **Service Layer**: Business logic and orchestration
- **Controller Layer**: Request handling and response formatting
- **DTO Layer**: Data transfer objects for API contracts
- **YAML Configuration**: Type-safe configuration management with Pydantic

Perfect for building RESTful APIs, microservices, or any backend application.

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/py-scaffold.git
cd py-scaffold

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Project Architecture

The generated projects follow a clean layered architecture:

1. **Controller Layer**: Handles HTTP requests/responses and user interaction
2. **Service Layer**: Contains business logic and orchestrates operations
3. **Repository Layer**: Manages data access and persistence
4. **Model Layer**: Defines domain entities
5. **DTO Layer**: Defines data contracts for API communication

This separation ensures:
- Clear separation of concerns
- Easy testing and mocking
- Maintainable and scalable code
- Independent layer evolution

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Roadmap

- [ ] FastAPI template with REST endpoints
- [ ] Django template
- [ ] Flask template
- [ ] GraphQL API template
- [ ] CLI application template
- [ ] Async worker template
- [ ] Docker & Docker Compose support
- [ ] CI/CD configuration templates