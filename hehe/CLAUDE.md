# hehe - AI/ML Project Documentation for Claude Code

This document provides an overview of the project structure and architecture for AI-assisted development with Claude Code.

## Project Overview

This is an **AI/ML project** designed for building intelligent applications with modular pipelines, including RAG (Retrieval-Augmented Generation) and NLP capabilities.

## Project Structure

```
hehe/
├── app/                          # Main application code
│   ├── main.py                   # Application entry point
│   ├── pipelines/                # Processing pipelines
│   │   ├── rag/                  # Retrieval-Augmented Generation
│   │   │   ├── embedder.py       # Text embedding component
│   │   │   ├── retriever.py      # Document retrieval component
│   │   │   └── generator.py      # Response generation component
│   │   └── nlp/                  # Natural Language Processing
│   │       └── processor.py      # Text processing utilities
│   ├── models/                   # ML models
│   │   ├── embedding/            # Embedding models
│   │   │   └── embedder.py       # Embedding model wrapper
│   │   ├── finetune/             # Fine-tuning utilities
│   │   │   └── trainer.py        # Model training logic
│   │   └── inference.py          # Inference engine
│   ├── data/                     # Data management
│   │   ├── raw/                  # Raw data storage
│   │   ├── processed/            # Processed data storage
│   │   └── loader.py             # Data loading utilities
│   └── utils/                    # Utility functions
│       ├── logger.py             # Logging configuration
│       └── helpers.py            # Helper functions
├── notebooks/                    # Jupyter notebooks
│   ├── preprocessing.ipynb       # Data preprocessing
│   ├── training.ipynb            # Model training
│   └── evaluation.ipynb          # Model evaluation
├── configs/                      # Configuration files
│   └── default.yml               # Default configuration
├── tests/                        # Test files
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Architecture Pattern

### Pipeline-Based Architecture

The project follows a **modular pipeline architecture** where different components can be composed together:

1. **Pipelines**: High-level workflows that orchestrate multiple components
   - RAG Pipeline: Embedding → Retrieval → Generation
   - NLP Pipeline: Text preprocessing and analysis

2. **Models**: ML model wrappers and training utilities
   - Embedding models for vector representations
   - Fine-tuning utilities for model customization
   - Inference engine for predictions

3. **Data Layer**: Data loading and preprocessing
   - Raw data ingestion
   - Data transformation and cleaning
   - Dataset management

4. **Utils**: Shared utilities and helpers
   - Logging and monitoring
   - Common helper functions

## Key Components

### RAG Pipeline

**Purpose**: Implement retrieval-augmented generation for context-aware responses

**Components**:
- `embedder.py`: Converts text to vector embeddings
- `retriever.py`: Finds relevant documents based on query
- `generator.py`: Generates responses using retrieved context

**Flow**:
```
Query → Embedder → Retriever → Generator → Response
```

### NLP Pipeline

**Purpose**: Text processing and analysis

**Components**:
- `processor.py`: Text preprocessing, tokenization, and analysis

### Models Module

**Purpose**: ML model management and training

**Components**:
- `embedding/embedder.py`: Embedding model wrapper (e.g., sentence-transformers)
- `finetune/trainer.py`: Training loop for fine-tuning models
- `inference.py`: Inference engine for predictions

### Data Module

**Purpose**: Data management and loading

**Components**:
- `loader.py`: Data loading from various sources
- `raw/`: Raw data storage
- `processed/`: Processed and cleaned data

## Development Workflow

### 1. Data Preparation
```python
# Use notebooks/preprocessing.ipynb
from app.data.loader import DataLoader

loader = DataLoader()
data = loader.load_raw_data("path/to/data")
processed = loader.preprocess(data)
loader.save_processed(processed)
```

### 2. Model Training
```python
# Use notebooks/training.ipynb
from app.models.finetune.trainer import Trainer

trainer = Trainer(config)
model = trainer.train(training_data)
trainer.save_model(model, "path/to/save")
```

### 3. Inference
```python
# Use app/main.py or notebooks/evaluation.ipynb
from app.models.inference import InferenceEngine

engine = InferenceEngine(model_path)
predictions = engine.predict(input_data)
```

### 4. RAG Pipeline
```python
from app.pipelines.rag.embedder import Embedder
from app.pipelines.rag.retriever import Retriever
from app.pipelines.rag.generator import Generator

# Initialize components
embedder = Embedder()
retriever = Retriever(embedder)
generator = Generator()

# Process query
query = "What is machine learning?"
docs = retriever.retrieve(query, top_k=5)
response = generator.generate(query, docs)
```

## Configuration

Configuration is managed through YAML files in `configs/`:

```yaml
app:
  name: hehe
  version: "1.0.0"

data:
  raw_path: "app/data/raw"
  processed_path: "app/data/processed"

rag:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 512
  top_k: 5

model:
  name: "default-model"
  batch_size: 32
  learning_rate: 0.001
```

## Best Practices

### Code Organization
- Keep pipelines modular and composable
- Separate data loading from processing
- Use configuration files for hyperparameters
- Write comprehensive tests for each component

### ML Development
- Use notebooks for experimentation
- Version control your models and data
- Log experiments and metrics
- Document model performance and limitations

### Data Management
- Keep raw data immutable
- Version processed datasets
- Document data preprocessing steps
- Use data loaders for consistent access

## Testing

Run tests using pytest:
```bash
pytest tests/
```

Write tests for:
- Data loading and preprocessing
- Model inference
- Pipeline components
- Utility functions

## Common Tasks

### Adding a New Pipeline Component

1. Create new file in `app/pipelines/<pipeline_name>/`
2. Implement the component class
3. Add initialization in `__init__.py`
4. Write tests in `tests/`
5. Update configuration if needed

### Fine-tuning a Model

1. Prepare training data in `app/data/processed/`
2. Configure training parameters in `configs/`
3. Use `app/models/finetune/trainer.py`
4. Evaluate in `notebooks/evaluation.ipynb`
5. Save model checkpoints

### Adding New Data Sources

1. Update `app/data/loader.py` with new loader method
2. Document data format and schema
3. Add preprocessing logic
4. Write data validation tests

## Tips for AI Assistants

When working with this codebase:

1. **Understand the Pipeline**: Know the flow of data through RAG or NLP pipelines
2. **Check Configuration**: Many behaviors are configured in YAML files
3. **Use Notebooks**: Experiment and prototype in Jupyter notebooks first
4. **Test Components**: Each pipeline component should be testable independently
5. **Document Models**: Always document model architecture, training data, and performance
6. **Version Data**: Keep track of data versions and preprocessing steps

## Common Patterns

### Loading Configuration
```python
from app.core.config import settings

# Access config values
model_name = settings.model.name
batch_size = settings.model.batch_size
```

### Error Handling
- Validate inputs at pipeline entry points
- Handle model loading errors gracefully
- Log errors with context
- Provide meaningful error messages

### Logging
```python
from app.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Processing started")
```

## Resources

- Project documentation: See README.md
- Model documentation: See model cards in `models/`
- Data documentation: See data sheets in `data/`
- API documentation: Generate with sphinx or similar tools

---

**Note**: This project is designed for AI/ML development with focus on modularity, experimentation, and production deployment.
