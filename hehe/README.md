# hehe

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
source venv/bin/activate  # On Windows: venv\Scripts\activate
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
