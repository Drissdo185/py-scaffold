"""AI Project template."""

from pathlib import Path
from typing import Dict, Any

from .base import TemplateManager


class AIProjectTemplate(TemplateManager):
    """Template for AI/ML projects."""

    def render(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Render the AI Project template."""
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
            "app/pipelines/rag",
            "app/pipelines/nlp",
            "app/models/embedding",
            "app/models/finetune",
            "app/data/raw",
            "app/data/processed",
            "app/utils",
            "notebooks",
            "configs",
            "tests",
            ".github",
        ]

        for dir_path in dirs:
            self._create_dir(output_path / dir_path)

    def _generate_config_yaml(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate config.yaml."""
        content = """# AI Project Configuration
app:
  name: {project_name}
  version: "1.0.0"
  environment: development

data:
  raw_path: "app/data/raw"
  processed_path: "app/data/processed"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

rag:
  embedding_model: 
  embedding_dim: 384
  chunk_size: 512
  chunk_overlap: 50
  top_k: 5
  llm_model: 
  temperature: 0.7

nlp:
  max_length: 512
  batch_size: 32
  model_name: 

model:
  embedding:
    model_name: 
    pooling: "mean"
    normalize: true
  finetune:
    batch_size: 16
    epochs: 10
    learning_rate: 2e-5
    warmup_steps: 500
    save_steps: 1000

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 10
  checkpoint_path: "app/models/checkpoints"

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
""".format(**context)

        self._create_file(output_path / "configs/default.yml", content)

    def _generate_main_py(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate main.py."""
        content = '''"""Main application entry point."""

import logging
from pathlib import Path

from app.utils.logger import setup_logging
from app.pipelines.rag import RAGPipeline
from app.models.inference import InferenceEngine


def main() -> None:
    """Run the AI application."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting AI Application")
    logger.info("=" * 50)

    # Example 1: RAG Pipeline
    logger.info("\\nExample 1: RAG Pipeline")
    logger.info("-" * 50)

    try:
        rag_pipeline = RAGPipeline()

        # Example query
        query = "What is machine learning?"
        logger.info(f"Query: {{query}}")

        # Run RAG pipeline (uncomment when implemented)
        # response = rag_pipeline.query(query)
        # logger.info(f"Response: {{response}}")

        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"RAG pipeline error: {{e}}")

    # Example 2: Inference
    logger.info("\\nExample 2: Model Inference")
    logger.info("-" * 50)

    try:
        inference_engine = InferenceEngine()
        logger.info("Inference engine initialized successfully")

        # Example inference (uncomment when implemented)
        # result = inference_engine.predict("Your input text here")
        # logger.info(f"Prediction: {{result}}")
    except Exception as e:
        logger.error(f"Inference error: {{e}}")

    logger.info("\\n" + "=" * 50)
    logger.info("Application setup complete!")
    logger.info("Modify app/main.py to implement your AI workflow")


if __name__ == "__main__":
    main()
'''

        self._create_file(output_path / "app/main.py", content)

    def _generate_requirements(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate requirements.txt."""
        content = """# Core dependencies


# NLP & Embeddings


# Vector databases (choose one or more)


# LLM Integration


# Text processing


# Machine Learning


# Data visualization


# Jupyter notebooks


# Development dependencies


# Utilities

"""

        self._create_file(output_path / "requirements.txt", content)

    def _generate_readme(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate README.md."""
        content = """# {project_name}

A production-ready AI/ML project with RAG pipeline and NLP capabilities, generated with py-scaffold.

## Project Structure

```
.
├── app/
│   ├── main.py                          # Main entry point
│   ├── pipelines/
│   │   ├── rag/                         # RAG Pipeline
│   │   │   ├── embedder.py              # Text embedding
│   │   │   ├── retriever.py             # Vector retrieval
│   │   │   └── generator.py             # Response generation
│   │   └── nlp/                         # NLP Pipeline
│   │       └── processor.py             # Text processing
│   ├── models/
│   │   ├── embedding/                   # Embedding models
│   │   │   └── embedder.py
│   │   ├── finetune/                    # Fine-tuning
│   │   │   └── trainer.py
│   │   └── inference.py                 # Model inference
│   ├── data/
│   │   ├── raw/                         # Raw data
│   │   ├── processed/                   # Processed data
│   │   └── loader.py                    # Data loading
│   └── utils/
│       ├── logger.py                    # Logging utilities
│       └── helpers.py                   # Helper functions
├── notebooks/
│   ├── preprocessing.ipynb              # Data preprocessing
│   ├── training.ipynb                   # Model training
│   └── evaluation.ipynb                 # Model evaluation
├── configs/
│   └── default.yml                      # Configuration
├── tests/                               # Unit tests
├── requirements.txt
└── README.md
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

3. Download NLP models (optional):
```bash
python -m spacy download en_core_web_sm
```

4. Configure your application:
Edit `configs/default.yml` to match your requirements.

5. Run the application:
```bash
python app/main.py
```

## Features

### RAG Pipeline

The RAG (Retrieval-Augmented Generation) pipeline includes:

- **Embedder**: Convert text to dense vectors using sentence transformers
- **Retriever**: Find relevant documents using vector similarity search
- **Generator**: Generate responses using retrieved context

Example usage:
```python
from app.pipelines.rag import RAGPipeline

rag = RAGPipeline()
response = rag.query("Your question here")
```

### NLP Pipeline

Text processing capabilities:

- Tokenization
- Named Entity Recognition (NER)
- Text classification
- Sentiment analysis

Example usage:
```python
from app.pipelines.nlp import NLPProcessor

nlp = NLPProcessor()
result = nlp.process("Your text here")
```

### Model Components

- **Embedding Models**: Generate text embeddings
- **Fine-tuning**: Train and fine-tune models
- **Inference Engine**: Run predictions on trained models

## Development Workflow

1. **Data Preparation**:
   - Place raw data in `app/data/raw/`
   - Use `DataLoader` to load and preprocess data
   - Processed data is saved to `app/data/processed/`

2. **Experimentation**:
   - Use Jupyter notebooks in `notebooks/` for exploration
   - Run: `jupyter notebook`

3. **Model Development**:
   - Define models in `app/models/`
   - Configure model parameters in `configs/default.yml`
   - Use fine-tuning scripts in `app/models/finetune/`

4. **Pipeline Integration**:
   - Implement RAG pipeline in `app/pipelines/rag/`
   - Add NLP processing in `app/pipelines/nlp/`
   - Integrate with main application in `app/main.py`

## Running Tests

```bash
pytest tests/
```

## Code Quality

Format code:
```bash
black app/ tests/
isort app/ tests/
```

Type checking:
```bash
mypy app/
```

Linting:
```bash
flake8 app/
```

## Configuration

Edit `configs/default.yml` to customize:

- Data paths and split ratios
- RAG pipeline settings (embedding model, chunk size, top-k)
- NLP model configuration
- Training hyperparameters
- Logging settings

## Architecture

This project follows a modular pipeline architecture:

- **Pipelines**: RAG and NLP processing pipelines
- **Models**: Embedding, fine-tuning, and inference components
- **Data**: Data loading and preprocessing
- **Utils**: Logging, helpers, and utilities

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

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/checkpoints/*
!models/checkpoints/.gitkeep
*.h5
*.pkl
*.pth
*.pt

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Logs
*.log
logs/

# ML specific
wandb/
mlruns/
"""

        self._create_file(output_path / ".gitignore", content)

    def _generate_core_config(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate core/config.py."""
        content = '''"""Configuration management."""

from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class AppConfig(BaseModel):
    """Application configuration."""

    name: str
    version: str
    environment: str


class DataConfig(BaseModel):
    """Data configuration."""

    raw_path: str
    processed_path: str
    train_split: float
    val_split: float
    test_split: float


class ModelConfig(BaseModel):
    """Model configuration."""

    type: str
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    dropout: float


class TrainingConfig(BaseModel):
    """Training configuration."""

    batch_size: int
    epochs: int
    learning_rate: float
    early_stopping_patience: int
    checkpoint_path: str


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str
    format: str


class Settings(BaseSettings):
    """Application settings loaded from YAML config."""

    app: AppConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Settings":
        """Load settings from YAML file."""
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        return cls(
            app=AppConfig(**config_data["app"]),
            data=DataConfig(**config_data["data"]),
            model=ModelConfig(**config_data["model"]),
            training=TrainingConfig(**config_data["training"]),
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

    def _generate_data_loader(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate data/data_loader.py."""
        content = '''"""Data loading utilities."""

import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np


logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading from various sources."""

    def __init__(self, raw_path: str, processed_path: str) -> None:
        """
        Initialize data loader.

        Args:
            raw_path: Path to raw data directory
            processed_path: Path to processed data directory
        """
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)

        # Create directories if they don't exist
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"DataLoader initialized with raw_path={self.raw_path}, processed_path={self.processed_path}")

    def load_raw_data(self, filename: str) -> Optional[np.ndarray]:
        """
        Load raw data from file.

        Args:
            filename: Name of the file to load

        Returns:
            Loaded data as numpy array or None if file doesn't exist
        """
        filepath = self.raw_path / filename

        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return None

        logger.info(f"Loading raw data from {filepath}")
        # Implement your data loading logic here
        # Example: return np.load(filepath)
        return None

    def save_processed_data(self, data: np.ndarray, filename: str) -> None:
        """
        Save processed data to file.

        Args:
            data: Data to save
            filename: Name of the file
        """
        filepath = self.processed_path / filename
        logger.info(f"Saving processed data to {filepath}")
        # Implement your data saving logic here
        # Example: np.save(filepath, data)

    def load_processed_data(self, filename: str) -> Optional[np.ndarray]:
        """
        Load processed data from file.

        Args:
            filename: Name of the file to load

        Returns:
            Loaded data as numpy array or None if file doesn't exist
        """
        filepath = self.processed_path / filename

        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return None

        logger.info(f"Loading processed data from {filepath}")
        # Implement your data loading logic here
        # Example: return np.load(filepath)
        return None
'''

        self._create_file(output_path / "src/app/data/data_loader.py", content)

    def _generate_data_preprocessor(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate data/preprocessor.py."""
        content = '''"""Data preprocessing utilities."""

import logging
from typing import Tuple
import numpy as np


logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing and transformation."""

    def __init__(self) -> None:
        """Initialize preprocessor."""
        self.mean = None
        self.std = None
        logger.info("DataPreprocessor initialized")

    def normalize(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize data using mean and standard deviation.

        Args:
            data: Input data
            fit: Whether to fit normalization parameters

        Returns:
            Normalized data
        """
        if fit:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            logger.info("Fitted normalization parameters")

        if self.mean is None or self.std is None:
            raise ValueError("Preprocessor not fitted. Call with fit=True first.")

        normalized = (data - self.mean) / (self.std + 1e-8)
        logger.info(f"Normalized data with shape {normalized.shape}")
        return normalized

    def split_data(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.

        Args:
            data: Input features
            labels: Target labels
            train_ratio: Proportion of training data
            val_ratio: Proportion of validation data
            test_ratio: Proportion of test data

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        n_samples = len(data)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        # Shuffle data
        indices = np.random.permutation(n_samples)
        data = data[indices]
        labels = labels[indices]

        # Split
        X_train = data[:n_train]
        y_train = labels[:n_train]

        X_val = data[n_train:n_train + n_val]
        y_val = labels[n_train:n_train + n_val]

        X_test = data[n_train + n_val:]
        y_test = labels[n_train + n_val:]

        logger.info(f"Split data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test
'''

        self._create_file(output_path / "src/app/data/preprocessor.py", content)

    def _generate_data_init(self, output_path: Path) -> None:
        """Generate data/__init__.py."""
        content = '''"""Data module."""

from .data_loader import DataLoader
from .preprocessor import DataPreprocessor

__all__ = ["DataLoader", "DataPreprocessor"]
'''

        self._create_file(output_path / "src/app/data/__init__.py", content)

    def _generate_model_base(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate models/model.py."""
        content = '''"""Model definitions."""

import logging
from typing import List


logger = logging.getLogger(__name__)


class NeuralNetworkModel:
    """Simple neural network model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2
    ) -> None:
        """
        Initialize neural network model.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout

        logger.info(
            f"Model initialized: input_dim={input_dim}, "
            f"hidden_dims={hidden_dims}, output_dim={output_dim}, dropout={dropout}"
        )

        # TODO: Implement your model architecture here
        # Example for PyTorch:
        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dims[0]),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     ...
        # )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # TODO: Implement forward pass
        logger.debug(f"Forward pass with input shape: {x.shape}")
        pass

    def get_num_parameters(self) -> int:
        """Get number of trainable parameters."""
        # TODO: Implement parameter counting
        return 0
'''

        self._create_file(output_path / "src/app/models/model.py", content)

    def _generate_model_init(self, output_path: Path) -> None:
        """Generate models/__init__.py."""
        content = '''"""Models module."""

from .model import NeuralNetworkModel

__all__ = ["NeuralNetworkModel"]
'''

        self._create_file(output_path / "src/app/models/__init__.py", content)

    def _generate_trainer(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate training/trainer.py."""
        content = '''"""Training utilities."""

import logging
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


class Trainer:
    """Handles model training and evaluation."""

    def __init__(
        self,
        model,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        epochs: int = 100,
        checkpoint_path: Optional[str] = None
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Model to train
            batch_size: Batch size for training
            learning_rate: Learning rate
            epochs: Number of training epochs
            checkpoint_path: Path to save checkpoints
        """
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        if self.checkpoint_path:
            self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Trainer initialized: batch_size={batch_size}, "
            f"lr={learning_rate}, epochs={epochs}"
        )

        # TODO: Initialize optimizer, loss function, etc.

    def train(self, train_data, val_data=None):
        """
        Train the model.

        Args:
            train_data: Training dataset
            val_data: Validation dataset (optional)
        """
        logger.info("Starting training...")

        for epoch in range(self.epochs):
            # TODO: Implement training loop
            train_loss = self._train_epoch(train_data)

            if val_data is not None:
                val_loss = self._validate(val_data)
                logger.info(f"Epoch {epoch+1}/{self.epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{self.epochs} - train_loss: {train_loss:.4f}")

            # Save checkpoint
            if self.checkpoint_path and (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)

        logger.info("Training complete!")

    def _train_epoch(self, train_data):
        """Train for one epoch."""
        # TODO: Implement epoch training
        logger.debug("Training epoch...")
        return 0.0

    def _validate(self, val_data):
        """Validate the model."""
        # TODO: Implement validation
        logger.debug("Validating...")
        return 0.0

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        if self.checkpoint_path:
            checkpoint_file = self.checkpoint_path / f"checkpoint_epoch_{epoch+1}.pth"
            logger.info(f"Saving checkpoint to {checkpoint_file}")
            # TODO: Implement checkpoint saving

    def evaluate(self, test_data):
        """
        Evaluate the model.

        Args:
            test_data: Test dataset

        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model...")
        # TODO: Implement evaluation
        return {}
'''

        self._create_file(output_path / "src/app/training/trainer.py", content)

    def _generate_training_init(self, output_path: Path) -> None:
        """Generate training/__init__.py."""
        content = '''"""Training module."""

from .trainer import Trainer

__all__ = ["Trainer"]
'''

        self._create_file(output_path / "src/app/training/__init__.py", content)

    def _generate_utils_logger(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate utils/logger.py."""
        content = '''"""Logging utilities."""

import logging
import sys
from pathlib import Path

from app.core.config import settings


def setup_logging(log_file: str = None) -> None:
    """
    Setup logging configuration.

    Args:
        log_file: Optional log file path
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, settings.logging.level))

    # Create formatter
    formatter = logging.Formatter(settings.logging.format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.logging.level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, settings.logging.level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info("Logging setup complete")
'''

        self._create_file(output_path / "src/app/utils/logger.py", content)

    def _generate_utils_init(self, output_path: Path) -> None:
        """Generate utils/__init__.py."""
        content = '''"""Utilities module."""

from .logger import setup_logging

__all__ = ["setup_logging"]
'''

        self._create_file(output_path / "src/app/utils/__init__.py", content)

    def _generate_app_init(self, output_path: Path) -> None:
        """Generate app/__init__.py."""
        content = '''"""Application package."""
'''

        self._create_file(output_path / "src/app/__init__.py", content)

    def _generate_notebook_example(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate example notebook."""
        content = '''{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# {project_name} - Data Exploration\\n",
    "\\n",
    "This notebook provides a starting point for data exploration and experimentation."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "import sys\\n",
    "sys.path.append('../src')\\n",
    "\\n",
    "import numpy as np\\n",
    "import pandas as pd\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "\\n",
    "from app.core.config import settings\\n",
    "from app.data.data_loader import DataLoader\\n",
    "from app.data.preprocessor import DataPreprocessor\\n",
    "\\n",
    "# Set style\\n",
    "sns.set_style('whitegrid')\\n",
    "plt.rcParams['figure.figsize'] = (12, 6)"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 1. Load Configuration"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "print(f\\"Project: {{settings.app.name}}\\")\\n",
    "print(f\\"Version: {{settings.app.version}}\\")\\n",
    "print(f\\"Environment: {{settings.app.environment}}\\")"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 2. Load Data"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "data_loader = DataLoader(\\n",
    "    raw_path=settings.data.raw_path,\\n",
    "    processed_path=settings.data.processed_path\\n",
    ")\\n",
    "\\n",
    "# TODO: Load your data\\n",
    "# data = data_loader.load_raw_data('your_data.csv')"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 3. Explore Data"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "# TODO: Add your data exploration code here"
   ],
   "outputs": []
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "codemirror_mode": {{
    "name": "ipython",
    "version": 3
   }},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}
'''.format(**context)

        self._create_file(output_path / "notebooks/exploration.ipynb", content)

    def _generate_tests_init(self, output_path: Path) -> None:
        """Generate tests/__init__.py."""
        content = '''"""Tests package."""
'''

        self._create_file(output_path / "tests/__init__.py", content)

    def _generate_claude_md(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate CLAUDE.md for Claude Code."""
        content = """# {project_name} - AI/ML Project Documentation for Claude Code

This document provides an overview of the project structure and architecture for AI-assisted development with Claude Code.

## Project Overview

This is an **AI/ML project** designed for building intelligent applications with modular pipelines, including RAG (Retrieval-Augmented Generation) and NLP capabilities.

## Project Structure

```
{project_name}/
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
  name: {project_name}
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
""".format(**context)

        self._create_file(output_path / "CLAUDE.md", content)

    def _generate_preprocessing_notebook(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate preprocessing.ipynb notebook."""
        content = '''{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# {project_name} - Data Preprocessing\\n",
    "\\n",
    "This notebook handles data preprocessing and preparation."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "import sys\\n",
    "sys.path.append('..')\\n",
    "\\n",
    "import numpy as np\\n",
    "import pandas as pd\\n",
    "import yaml\\n",
    "from pathlib import Path\\n",
    "\\n",
    "from app.data.loader import DataLoader\\n",
    "from app.utils.logger import setup_logging\\n",
    "\\n",
    "# Setup logging\\n",
    "setup_logging()"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 1. Load Configuration"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "# Load config\\n",
    "config_path = Path('../configs/default.yml')\\n",
    "with open(config_path, 'r') as f:\\n",
    "    config = yaml.safe_load(f)\\n",
    "\\n",
    "print('Configuration loaded successfully')\\n",
    "print(f\\"Data paths:\\")\\n",
    "print(f\\"  Raw: {{config['data']['raw_path']}}\\")\\n",
    "print(f\\"  Processed: {{config['data']['processed_path']}}\\")"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 2. Initialize Data Loader"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "data_loader = DataLoader(\\n",
    "    raw_path=config['data']['raw_path'],\\n",
    "    processed_path=config['data']['processed_path']\\n",
    ")"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 3. Load Raw Data\\n",
    "\\n",
    "TODO: Implement your data loading logic"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "# Example: Load your raw data\\n",
    "# raw_data = data_loader.load_raw_data('your_data_file.csv')\\n",
    "\\n",
    "print('Data loading completed')"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 4. Data Preprocessing\\n",
    "\\n",
    "TODO: Implement preprocessing steps"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "# Add your preprocessing logic here\\n",
    "pass"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 5. Save Processed Data"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "# Save processed data\\n",
    "# data_loader.save_processed_data(processed_data, 'processed_data.pkl')\\n",
    "\\n",
    "print('Preprocessing completed!')"
   ],
   "outputs": []
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "codemirror_mode": {{
    "name": "ipython",
    "version": 3
   }},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}
'''.format(**context)

        self._create_file(output_path / "notebooks/preprocessing.ipynb", content)

    def _generate_training_notebook(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate training.ipynb notebook."""
        content = '''{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# {project_name} - Model Training\\n",
    "\\n",
    "This notebook handles model training and fine-tuning."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "import sys\\n",
    "sys.path.append('..')\\n",
    "\\n",
    "import numpy as np\\n",
    "import yaml\\n",
    "from pathlib import Path\\n",
    "\\n",
    "from app.data.loader import DataLoader\\n",
    "from app.models.finetune.trainer import Trainer\\n",
    "from app.utils.logger import setup_logging\\n",
    "\\n",
    "# Setup logging\\n",
    "setup_logging()"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 1. Load Configuration"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "# Load config\\n",
    "config_path = Path('../configs/default.yml')\\n",
    "with open(config_path, 'r') as f:\\n",
    "    config = yaml.safe_load(f)\\n",
    "\\n",
    "print('Configuration loaded')\\n",
    "print(f\\"Training config: {{config['model']['finetune']}}\\")"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 2. Load Processed Data"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "data_loader = DataLoader(\\n",
    "    raw_path=config['data']['raw_path'],\\n",
    "    processed_path=config['data']['processed_path']\\n",
    ")\\n",
    "\\n",
    "# TODO: Load your processed data\\n",
    "# train_data = data_loader.load_processed_data('train.pkl')\\n",
    "# val_data = data_loader.load_processed_data('val.pkl')"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 3. Initialize Trainer"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "# TODO: Initialize your trainer\\n",
    "# trainer = Trainer(config['model']['finetune'])\\n",
    "\\n",
    "print('Trainer initialized')"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 4. Train Model"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "# TODO: Start training\\n",
    "# trainer.train(train_data, val_data)\\n",
    "\\n",
    "print('Training started...')"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 5. Save Model"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "# TODO: Save the trained model\\n",
    "# trainer.save_model('model_checkpoint.pth')\\n",
    "\\n",
    "print('Model saved!')"
   ],
   "outputs": []
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "codemirror_mode": {{
    "name": "ipython",
    "version": 3
   }},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}
'''.format(**context)

        self._create_file(output_path / "notebooks/training.ipynb", content)

    def _generate_evaluation_notebook(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate evaluation.ipynb notebook."""
        content = '''{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# {project_name} - Model Evaluation\\n",
    "\\n",
    "This notebook handles model evaluation and analysis."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "import sys\\n",
    "sys.path.append('..')\\n",
    "\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "import yaml\\n",
    "from pathlib import Path\\n",
    "\\n",
    "from app.data.loader import DataLoader\\n",
    "from app.models.inference import InferenceEngine\\n",
    "from app.utils.logger import setup_logging\\n",
    "\\n",
    "# Setup logging\\n",
    "setup_logging()\\n",
    "\\n",
    "# Set visualization style\\n",
    "sns.set_style('whitegrid')\\n",
    "plt.rcParams['figure.figsize'] = (12, 6)"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 1. Load Configuration"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "# Load config\\n",
    "config_path = Path('../configs/default.yml')\\n",
    "with open(config_path, 'r') as f:\\n",
    "    config = yaml.safe_load(f)\\n",
    "\\n",
    "print('Configuration loaded')"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 2. Load Test Data"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "data_loader = DataLoader(\\n",
    "    raw_path=config['data']['raw_path'],\\n",
    "    processed_path=config['data']['processed_path']\\n",
    ")\\n",
    "\\n",
    "# TODO: Load your test data\\n",
    "# test_data = data_loader.load_processed_data('test.pkl')"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 3. Load Model"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "# TODO: Load your trained model\\n",
    "# inference_engine = InferenceEngine(model_path='model_checkpoint.pth')\\n",
    "\\n",
    "print('Model loaded')"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 4. Run Evaluation"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "# TODO: Run evaluation\\n",
    "# predictions = inference_engine.predict(test_data)\\n",
    "\\n",
    "print('Evaluation completed')"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 5. Calculate Metrics"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "# TODO: Calculate evaluation metrics\\n",
    "# from sklearn.metrics import accuracy_score, f1_score\\n",
    "# accuracy = accuracy_score(y_true, predictions)\\n",
    "# f1 = f1_score(y_true, predictions, average='weighted')\\n",
    "\\n",
    "# print(f'Accuracy: {{accuracy:.4f}}')\\n",
    "# print(f'F1 Score: {{f1:.4f}}')"
   ],
   "outputs": []
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## 6. Visualizations"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "source": [
    "# TODO: Add visualizations (confusion matrix, ROC curve, etc.)\\n",
    "pass"
   ],
   "outputs": []
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "codemirror_mode": {{
    "name": "ipython",
    "version": 3
   }},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}
'''.format(**context)

        self._create_file(output_path / "notebooks/evaluation.ipynb", content)

    def _generate_copilot_instruction(self, output_path: Path, context: Dict[str, Any]) -> None:
        """Generate .github/copilot-instructions.md for GitHub Copilot."""
        content = """# GitHub Copilot Instructions - {project_name}

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
logger.info(f"Starting training with config: {{config}}")
logger.info(f"Training metrics: {{metrics}}")
```

## Error Handling

### Pipeline Errors
```python
try:
    docs = retriever.retrieve(query)
except Exception as e:
    logger.error(f"Retrieval failed: {{e}}")
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
    \"\"\"Train ML model with given data.

    Args:
        data: Training dataset
        config: Training configuration dictionary

    Returns:
        Trained model instance

    Raises:
        ValueError: If data is empty or invalid
    \"\"\"
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
    return {{"prediction": result}}
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
""".format(**context)

        self._create_file(output_path / ".github/copilot-instructions.md", content)
