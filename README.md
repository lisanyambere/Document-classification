# Enhanced Document Classification System

A hybrid document classification system combining traditional ML models with Large Language Models (LLMs) and intelligent routing for optimal performance and cost efficiency.

## System Architecture

This system features a **modular architecture** with intelligent routing that automatically selects between ML (fast, cheap) and LLM (accurate, expensive) classifiers based on document complexity and confidence levels.

### Key Components

- **Hybrid Classification**: ML + LLM with intelligent routing
- **Supervisory Agent**: Orchestrates classification decisions
- **Adaptive Routing**: Learns optimal routing strategies over time
- **Resilience Framework**: Error handling and retry mechanisms
- **Performance Monitoring**: Real-time metrics and feedback collection

## Features

- **Intelligent Routing**: Automatically chooses ML or LLM based on document characteristics
- **Cost Optimization**: Balances accuracy vs. processing costs
- **Real-time Learning**: Routing system improves with feedback
- **Modular Design**: Clean separation of concerns
- **Enterprise Ready**: Configuration management, logging, error handling

## Installation

1. **Clone repository**:
```bash
git clone <repository-url>
cd Document-classification
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp env_template.txt .env
# Edit .env with your GROQ_API_KEY and other settings
```

## Quick Start

### Single Document Classification

```bash
# Classify a single document
python main.py sample_documents/invoice.txt

# Detailed output with reasoning
python main.py sample_documents/technical_spec.txt --format detailed
```

### Batch Classification

```bash
# Process multiple files
python main.py sample_documents/*.txt --batch

# Process directory with JSON output
python main.py sample_documents/ --batch --format json --output results.json
```

## System Architecture

```
src/
├── core/                   # Foundation & Infrastructure
│   ├── base_classifier.py  # Base classes and interfaces
│   ├── config_manager.py   # Configuration management
│   └── resilience.py       # Error handling and resilience
├── routing/                # Intelligent Document Routing  
│   ├── routing_models.py   # Data models and ML predictor
│   ├── feature_extractor.py # Fast feature extraction
│   ├── threshold_manager.py # Adaptive threshold management
│   └── learned_routing.py  # Main routing agent
├── classifiers/            # Document Classification
│   ├── document_classifier.py # ML-based classifier
│   └── llm_classifier.py   # LLM-based classifier
├── agents/                 # Orchestration Layer
│   └── agent.py           # Supervisory agent
└── utils/                 # Utilities & Tools
    ├── training_bootstrap.py # Training data generation
    └── visualizations.py   # Data visualization tools
```

## Classification Categories

The system classifies documents into 15 business categories:

- **Administrative**: Letter, Memo, Form, Email
- **Financial**: Invoice, Budget
- **Technical**: Specification, Scientific Report, Scientific Publication
- **Marketing**: Advertisement, News Article, Presentation
- **Personal**: Resume, Handwritten
- **Organizational**: File Folder, Questionnaire

## Configuration

### Environment Variables

Configure via `.env` file (copy from `env_template.txt`):

```bash
# LLM Configuration
GROQ_API_KEY=your_api_key_here
LLM_MODEL=llama3-8b-8192
MAX_TOKENS=1024
TEMPERATURE=0.1

# Routing Parameters
ML_THRESHOLD=0.6
COST_BUDGET_PER_HOUR=1.0
MAX_LLM_RATIO=0.3
LEARNING_RATE=0.1
```

## API Usage

### Python API

```python
from src.classifiers import MLDocumentClassifier, LLMDocumentClassifier
from src.agents import SupervisoryAgent
from src.core import DocumentMetadata

# Initialize classifiers
ml_classifier = MLDocumentClassifier()
llm_classifier = LLMDocumentClassifier()

# Create supervisory agent with intelligent routing
agent = SupervisoryAgent(ml_classifier, llm_classifier)

# Classify document
metadata = DocumentMetadata(file_extension="pdf", sender_email="user@company.com")
result = agent.classify_document("Document content here", metadata)

print(f"Category: {result.category}")
print(f"Confidence: {result.confidence}")
print(f"Classifier used: {result.features_used.get('routing_decision')}")
```

### Routing System

The intelligent routing system:

1. **Extracts features** from document (word count, complexity, etc.)
2. **Predicts ML confidence** using lightweight ML model
3. **Routes to appropriate classifier**:
   - **ML**: Simple documents, high confidence, cost-sensitive
   - **LLM**: Complex documents, low confidence, accuracy-critical
4. **Learns from feedback** to improve routing decisions

## Training and Feedback

### Generate Training Data

```python
# Generate training data comparing ML vs LLM performance
python -m src.utils.training_bootstrap
```

This runs documents through both classifiers to learn optimal routing strategies.

### Performance Monitoring

```python
from src.agents import SupervisoryAgent

agent = SupervisoryAgent(ml_classifier, llm_classifier)

# Get system statistics  
stats = agent.get_system_statistics()
print(f"Total classifications: {stats['system']['total_classifications']}")
print(f"Average confidence: {stats['ml_classifier']['avg_confidence']}")
print(f"Routing accuracy: {stats['routing_agent']['routing_accuracy']}")
```

## Advanced Features

### Cost Management

The system automatically manages costs by:
- **Budget tracking**: Monitors hourly LLM usage costs
- **Ratio limits**: Caps percentage of requests routed to LLM
- **Adaptive thresholds**: Adjusts routing based on budget constraints

### Performance Optimization

- **Feature caching**: Fast document feature extraction (~2-3ms)
- **Model reuse**: Efficient classifier initialization
- **Batch processing**: Optimized for multiple document processing
- **Resilience**: Automatic retry and fallback mechanisms

### Extensibility

Add new classifiers by implementing the `DocumentClassifier` interface:

```python
from src.core.base_classifier import DocumentClassifier

class CustomClassifier(DocumentClassifier):
    def predict(self, content: str, metadata: DocumentMetadata) -> ClassificationResult:
        # Implementation here
        pass
```

## Monitoring and Debugging

### System Statistics

```bash
python main.py --stats
```

### Detailed Logging

Configure logging level in `.env`:
```bash
LOG_LEVEL=DEBUG  # INFO, WARNING, ERROR
```

### Performance Analysis

```python
# View routing decisions and performance
python -m src.utils.visualizations
```

## Deployment

### Production Setup

1. **Environment**: Set production environment variables
2. **Models**: Ensure all ML models are trained and available
3. **API Keys**: Configure LLM API access
4. **Monitoring**: Set up logging and metrics collection
5. **Scaling**: Consider load balancing for high-volume usage

### Docker Deployment

```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
```

## Troubleshooting

### Common Issues

1. **LLM API Errors**: Check GROQ_API_KEY in `.env`
2. **ML Models Missing**: Run training: `python src/classifiers/document_classifier.py`
3. **Import Errors**: Ensure proper Python path and package structure
4. **Slow Performance**: Check routing thresholds and system load

### Debug Mode

```bash
DEBUG=True python main.py sample_documents/invoice.txt --format detailed
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Follow the modular architecture patterns
4. Add tests for new functionality
5. Update documentation
6. Submit pull request

## Architecture Benefits

### Separation of Concerns
- **Core**: Fundamental interfaces and utilities
- **Routing**: Intelligent decision making
- **Classifiers**: Specialized classification logic
- **Agents**: High-level orchestration
- **Utils**: Supporting tools and analysis

### Scalability
- **Modular design**: Easy to extend and modify
- **Clean interfaces**: Well-defined component boundaries
- **Configuration management**: Environment-specific settings
- **Performance monitoring**: Real-time system insights

### Maintainability
- **Single responsibility**: Each module has one clear purpose
- **Dependency injection**: Loose coupling between components
- **Error handling**: Comprehensive resilience framework
- **Documentation**: Clear code structure and comments

## License

MIT License - see LICENSE file for details.