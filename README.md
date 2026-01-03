# py-scaffold

[![PyPI version](https://badge.fury.io/py/py-scaffold.svg)](https://badge.fury.io/py/py-scaffold)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python CLI tool for generating production-ready project templates, similar to create-next-app.

## Features

- **Backend API Template**: Production-ready layered architecture with Models, Repositories, Services, Controllers, and DTOs
- **AI Project Template**: Modular pipeline architecture with RAG, NLP, and ML model training capabilities
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

**Backend API:**
```bash
py-scaffold my-api --template backend-api
```

**AI Project:**
```bash
py-scaffold my-ai-app --template ai-project
```

**Interactive mode** (no template specified):
```bash
py-scaffold my-project
```

## Project Structures

### Backend API Template

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

### AI Project Template

A comprehensive AI/ML project structure with modular pipelines:

```
my-ai-app/
├── app/
│   ├── main.py                   # Application entry point
│   ├── pipelines/                # Processing pipelines
│   │   ├── rag/                  # Retrieval-Augmented Generation
│   │   │   ├── embedder.py       # Text embedding component
│   │   │   ├── retriever.py      # Document retrieval
│   │   │   └── generator.py      # Response generation
│   │   └── nlp/                  # Natural Language Processing
│   │       └── processor.py      # Text processing utilities
│   ├── models/                   # ML models
│   │   ├── embedding/            # Embedding models
│   │   │   └── embedder.py
│   │   ├── finetune/             # Fine-tuning utilities
│   │   │   └── trainer.py
│   │   └── inference.py          # Inference engine
│   ├── data/                     # Data management
│   │   ├── raw/                  # Raw data storage
│   │   ├── processed/            # Processed data
│   │   └── loader.py             # Data loading utilities
│   └── utils/                    # Utility functions
│       ├── logger.py             # Logging configuration
│       └── helpers.py            # Helper functions
├── notebooks/                    # Jupyter notebooks
│   ├── preprocessing.ipynb       # Data preprocessing
│   ├── training.ipynb            # Model training
│   └── evaluation.ipynb          # Model evaluation
├── configs/
│   └── default.yml               # YAML configuration
├── tests/                        # Unit tests
├── .github/
│   └── copilot-instructions.md   # GitHub Copilot instructions
├── CLAUDE.md                     # Claude Code documentation
├── requirements.txt
└── README.md
```

### Command Options

```bash
py-scaffold <project-name> [OPTIONS]

Options:
  -t, --template TEXT     Template to use (backend-api or ai-project)
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

### AI Project

A modular pipeline architecture for AI/ML applications with:

- **RAG Pipeline**: Retrieval-Augmented Generation with embedder, retriever, and generator components
- **NLP Pipeline**: Natural language processing and text analysis
- **Model Components**: Embedding models, fine-tuning utilities, and inference engine
- **Data Management**: Raw and processed data storage with data loaders
- **Jupyter Notebooks**: Pre-configured notebooks for preprocessing, training, and evaluation
- **YAML Configuration**: Configurable hyperparameters for models, training, and pipelines
- **AI Assistant Documentation**: Built-in CLAUDE.md and GitHub Copilot instructions

Perfect for building RAG applications, NLP pipelines, ML model training, or any AI/ML project.

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

## Project Architecture

### Backend API Architecture

The Backend API template follows a clean layered architecture:

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

### AI Project Architecture

The AI Project template follows a modular pipeline architecture:

1. **Pipeline Layer**: High-level workflows that orchestrate components
   - RAG Pipeline: Embedding → Retrieval → Generation
   - NLP Pipeline: Text preprocessing and analysis

2. **Models Layer**: ML model wrappers and training utilities
   - Embedding models for vector representations
   - Fine-tuning utilities for model customization
   - Inference engine for predictions

3. **Data Layer**: Data loading and preprocessing
   - Raw data ingestion
   - Data transformation and cleaning
   - Dataset management

4. **Utils Layer**: Shared utilities and helpers
   - Logging and monitoring
   - Common helper functions

This architecture ensures:
- Modular and composable components
- Easy experimentation with Jupyter notebooks
- Reproducible training and evaluation
- Production-ready deployment patterns
- Clear data and model versioning

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Setting up the development environment
- Code style and quality standards
- Testing requirements
- Pull request process

## License

MIT License - see LICENSE file for details