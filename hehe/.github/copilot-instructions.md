# GitHub Copilot Instructions - hehe

## Project Context

This is an **AI/ML project** using a **pipeline-based architecture** for building intelligent applications.

## Architecture Overview

### Pipeline-Based Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Application Layer                     │
│                          (app/main.py)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Pipeline Layer                        │
│  ┌──────────────┐              ┌──────────────┐            │
│  │  RAG Pipeline │              │ NLP Pipeline  │            │
│  │  - Embedder   │              │  - Processor  │            │
│  │  - Retriever  │              │               │            │
│  │  - Generator  │              │               │            │
│  └──────────────┘              └──────────────┘            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         Models Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Embedding   │  │  Fine-tune   │  │  Inference   │      │
│  │   Models     │  │   Trainer    │  │   Engine     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                          Data Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Raw Data    │  │  Processed   │  │ Data Loader  │      │
│  │   Storage    │  │    Data      │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Code Generation Guidelines

### 1. RAG Pipeline Components

When generating RAG pipeline code:

**Embedder** (`app/pipelines/rag/embedder.py`):
```python
class Embedder:
    def __init__(self, model_name: str):
        # Initialize embedding model (e.g., sentence-transformers)
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        # Convert texts to embeddings
        pass
```

**Retriever** (`app/pipelines/rag/retriever.py`):
```python
class Retriever:
    def __init__(self, embedder: Embedder):
        # Initialize with embedder and vector store
        pass

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        # Retrieve relevant documents
        pass
```

**Generator** (`app/pipelines/rag/generator.py`):
```python
class Generator:
    def __init__(self, model_name: str):
        # Initialize generation model
        pass

    def generate(self, query: str, context: list[Document]) -> str:
        # Generate response using context
        pass
```

### 2. Model Components

**Embedding Model** (`app/models/embedding/embedder.py`):
```python
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts)
```

**Trainer** (`app/models/finetune/trainer.py`):
```python
class Trainer:
    def __init__(self, config: dict):
        # Initialize training configuration
        pass

    def train(self, train_data, val_data):
        # Training loop with logging and checkpointing
        pass

    def evaluate(self, test_data):
        # Model evaluation
        pass
```

**Inference Engine** (`app/models/inference.py`):
```python
class InferenceEngine:
    def __init__(self, model_path: str):
        # Load trained model
        pass

    def predict(self, inputs):
        # Run inference
        pass
```

### 3. Data Components

**Data Loader** (`app/data/loader.py`):
```python
class DataLoader:
    def __init__(self, config: dict):
        self.raw_path = config.get("raw_path")
        self.processed_path = config.get("processed_path")

    def load_raw_data(self, source: str):
        # Load raw data from various sources
        pass

    def preprocess(self, data):
        # Data preprocessing and cleaning
        pass

    def save_processed(self, data, filename: str):
        # Save processed data
        pass
```

### 4. Utility Components

**Logger** (`app/utils/logger.py`):
```python
import logging

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    # Configure logger
    return logger
```

**Helpers** (`app/utils/helpers.py`):
```python
def load_config(config_path: str) -> dict:
    # Load YAML configuration
    pass

def save_checkpoint(model, path: str):
    # Save model checkpoint
    pass

def load_checkpoint(path: str):
    # Load model checkpoint
    pass
```

## Naming Conventions

### Files and Directories
- Snake_case for Python files: `data_loader.py`, `embedding_model.py`
- Lowercase for directories: `pipelines/`, `models/`, `data/`
- Configuration files: `default.yml`, `production.yml`

### Python Code
- PascalCase for classes: `Embedder`, `Retriever`, `InferenceEngine`
- snake_case for functions and variables: `embed_text()`, `model_path`
- UPPERCASE for constants: `MAX_SEQUENCE_LENGTH`, `DEFAULT_BATCH_SIZE`

### Models and Experiments
- Descriptive names: `bert-base-finetuned-v1`, `rag-embedder-v2`
- Include version numbers: `model-v1.0.0`
- Date stamps for experiments: `experiment-2024-01-15`

## Common Patterns

### 1. Configuration Management
```python
from app.utils.helpers import load_config

config = load_config("configs/default.yml")
model_config = config["model"]
```

### 2. Pipeline Composition
```python
# Compose pipeline from components
embedder = Embedder(config["rag"]["embedding_model"])
retriever = Retriever(embedder)
generator = Generator(config["rag"]["generation_model"])

# Run pipeline
docs = retriever.retrieve(query)
response = generator.generate(query, docs)
```

### 3. Model Training
```python
from app.models.finetune.trainer import Trainer
from app.data.loader import DataLoader

# Load data
loader = DataLoader(config["data"])
train_data = loader.load_processed("train.pkl")

# Train model
trainer = Trainer(config["model"])
model = trainer.train(train_data)
trainer.save_checkpoint(model, "checkpoints/model-v1")
```

### 4. Experiment Tracking
```python
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Log experiments
logger.info(f"Starting training with config: {config}")
logger.info(f"Training metrics: {metrics}")
```

## Error Handling

### Pipeline Errors
```python
try:
    docs = retriever.retrieve(query)
except Exception as e:
    logger.error(f"Retrieval failed: {e}")
    # Fallback or re-raise
```

### Model Loading Errors
```python
try:
    model = load_checkpoint(model_path)
except FileNotFoundError:
    logger.warning("Checkpoint not found, using default model")
    model = load_default_model()
```

### Data Validation
```python
def validate_data(data):
    if data is None or len(data) == 0:
        raise ValueError("Empty dataset")
    # Additional validation
```

## Testing Patterns

### Unit Tests
```python
import pytest
from app.pipelines.rag.embedder import Embedder

def test_embedder():
    embedder = Embedder("test-model")
    texts = ["hello world"]
    embeddings = embedder.embed(texts)
    assert embeddings.shape[0] == len(texts)
```

### Integration Tests
```python
def test_rag_pipeline():
    # Test full pipeline
    embedder = Embedder(config)
    retriever = Retriever(embedder)
    generator = Generator(config)

    query = "test query"
    docs = retriever.retrieve(query)
    response = generator.generate(query, docs)

    assert response is not None
    assert len(docs) > 0
```

## Documentation Standards

### Docstrings
```python
def train_model(data, config):
    """Train ML model with given data.

    Args:
        data: Training dataset
        config: Training configuration dictionary

    Returns:
        Trained model instance

    Raises:
        ValueError: If data is empty or invalid
    """
    pass
```

### Model Documentation
- Document model architecture
- Include training data information
- Record hyperparameters
- Note performance metrics
- Document limitations and biases

## Performance Considerations

### Batch Processing
```python
# Process data in batches
def batch_inference(data, batch_size=32):
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        yield model.predict(batch)
```

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text: str):
    return embedder.embed([text])[0]
```

### Memory Management
```python
# Clear cache when needed
import gc
import torch

del model
gc.collect()
torch.cuda.empty_cache()  # If using GPU
```

## Deployment Patterns

### Model Serving
```python
from app.models.inference import InferenceEngine

class ModelServer:
    def __init__(self, model_path: str):
        self.engine = InferenceEngine(model_path)

    def predict(self, request):
        inputs = self.preprocess(request)
        outputs = self.engine.predict(inputs)
        return self.postprocess(outputs)
```

### API Endpoints (if applicable)
```python
@app.post("/predict")
async def predict(request: PredictRequest):
    result = model_server.predict(request.data)
    return {"prediction": result}
```

## Jupyter Notebook Guidelines

### Notebook Structure
1. **Setup**: Imports and configuration
2. **Data Loading**: Load and explore data
3. **Preprocessing**: Clean and transform data
4. **Modeling**: Train and evaluate models
5. **Analysis**: Analyze results and visualizations
6. **Conclusion**: Summary and next steps

### Code Organization
- Keep notebooks focused on one task
- Extract reusable code to modules
- Document assumptions and decisions
- Include visualizations and metrics

## Remember

When generating code for this AI/ML project:

1. **Use type hints** for better code clarity
2. **Log important events** and metrics
3. **Handle errors gracefully** with proper exception handling
4. **Write modular code** that can be tested independently
5. **Document complex logic** and model decisions
6. **Use configuration files** for hyperparameters
7. **Version models and data** for reproducibility
8. **Optimize for batch processing** when possible
9. **Consider memory usage** for large datasets
10. **Write comprehensive tests** for critical components

**Follow these patterns and the generated code will integrate seamlessly with this AI/ML project!**
