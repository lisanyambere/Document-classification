# How to Run Enhanced Document Classification System

## Quick Start

### 1. Classify Documents (Recommended)
```bash
# Single document
python main.py sample_documents/invoice.txt

# Multiple documents  
python main.py sample_documents/*.txt

# Directory of documents
python main.py sample_documents/ --batch

# Detailed output
python main.py sample_documents/technical_spec.txt --format detailed

# JSON output
python main.py sample_documents/*.txt --format json --output results.json
```

### 2. Test the Complete System
```bash
python test_enhanced_agent.py
```
This runs comprehensive tests of the ML + LLM routing system.

### 3. Use the Agent Directly
```python
from document_classifier import MLDocumentClassifier
from llm_classifier import LLMDocumentClassifier
from agent import SupervisoryAgent
from base_classifier import DocumentMetadata

# Initialize
ml_classifier = MLDocumentClassifier()
llm_classifier = LLMDocumentClassifier()
agent = SupervisoryAgent(ml_classifier, llm_classifier)

# Classify a document
content = "Technical specification for API architecture..."
metadata = DocumentMetadata(file_extension="pdf", sender_email="dev@company.com")

result = agent.classify_document(content, metadata)
print(f"Category: {result.category.value}")
print(f"Confidence: {result.confidence}")
print(f"Routed to: {result.features_used['routing_decision']}")
```

### 3. Run Original Demo (Fixed)
```bash
python run_demo.py
```

## System Components

**Core Files:**
- `agent.py` - Main supervisory agent with intelligent routing
- `document_classifier.py` - ML classifier (98.4% accuracy)
- `llm_classifier.py` - LLM classifier with GROQ API
- `learned_routing.py` - Enhanced routing with LLM reasoning
- `models/` - Trained ML models (auto-loaded)

**Test Files:**
- `test_enhanced_agent.py` - Complete system test
- `run_demo.py` - Original demo (fixed for Windows)

## Routing Logic

1. **Simple documents** (invoices, letters) → ML classifier (fast)
2. **Complex technical content** → LLM classifier (detailed analysis)  
3. **Uncertain cases** → LLM reasoning decides routing
4. **75% of decisions** now use LLM reasoning for better accuracy

## Requirements

- Python 3.8+
- GROQ API key in `.env` file
- All dependencies: `pip install -r requirements.txt`
- Models are already trained and saved in `models/`

## Performance

- **ML Classification**: <50ms for simple documents
- **LLM Classification**: ~500-1000ms for complex documents
- **Overall Accuracy**: 98.4% (ML) + enhanced LLM analysis
- **Intelligent Routing**: 75% LLM reasoning, 25% heuristic