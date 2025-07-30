# LLM Document Classification with Groq API

This module provides LLM-based document classification using LangChain and the Groq API. It's designed to work alongside the traditional ML classifier for a hybrid approach.

## Features

- **LangChain Integration**: Uses LangChain for structured LLM interactions
- **Groq API Support**: Leverages Groq's fast inference service for document classification
- **Document Sampling**: Only sends a portion of documents to LLM (configurable)
- **Environment Configuration**: All settings via environment variables
- **Structured Output**: Parses LLM responses into structured classification results
- **Performance Tracking**: Monitors classification performance and timing
- **Error Handling**: Robust error handling with fallback mechanisms

## Setup

### 1. Install Dependencies

The LLM dependencies are already included in the main requirements.txt file:

```bash
cd Document-classification
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the Document-classification folder:

```bash
# Copy the template
cp env_template.txt .env

# Edit the .env file with your actual values
nano .env
```

Required configuration:

```bash
# Required: Your Groq API key
GROQ_API_KEY=your-actual-groq-api-key-here

# Optional: Customize these settings
GROQ_MODEL_NAME=llama3-8b-8192
GROQ_TEMPERATURE=0.1
GROQ_MAX_TOKENS=1000
GROQ_TIMEOUT=30
GROQ_BASE_URL=https://api.groq.com/openai/v1

# Document sampling limits
LLM_MAX_CHARS=2000
LLM_MAX_WORDS=300
```

### 3. Test the Integration

Run the test script to verify everything works:

```bash
python test_llm.py
```

## Usage

### Basic Classification

```python
from llm_classifier import LLMDocumentClassifier

# Initialize classifier (uses environment variables)
classifier = LLMDocumentClassifier()

# Classify a document (only portion sent to LLM)
document_content = """
Dear Mr. Johnson,
I am writing to follow up on our meeting last week regarding the quarterly budget review.
As discussed, we need to allocate additional funds for the marketing campaign.
...
"""

metadata = {
    "sender": "sarah.smith@company.com",
    "file_extension": "pdf"
}

result = classifier.classify_document(document_content, metadata)

print(f"Category: {result.category.value}")
print(f"Confidence: {result.confidence}")
print(f"Processing Time: {result.processing_time_ms}ms")
print(f"Reasoning: {result.reasoning}")
```

### Performance Monitoring

```python
# Get performance statistics
stats = classifier.get_performance_stats()
print(f"Total classifications: {stats['total_classifications']}")
print(f"Average confidence: {stats['avg_confidence']:.2f}")
print(f"Average processing time: {stats['avg_processing_time_ms']:.2f}ms")
print(f"Average sample ratio: {stats['avg_sample_ratio']:.2f}")
```

## Document Sampling

The LLM classifier automatically samples documents to reduce token usage and costs:

- **Default Limit**: 2000 characters (configurable via `LLM_MAX_CHARS`)
- **Smart Truncation**: Breaks at word boundaries when possible
- **Ellipsis Indication**: Adds "..." to show truncation
- **Performance Tracking**: Logs sampling ratios for analysis

### Sampling Configuration

```bash
# In your .env file
LLM_MAX_CHARS=2000    # Maximum characters to send to LLM
LLM_MAX_WORDS=300     # Alternative word-based limit
```

### Sampling Examples

```python
# Short document (no sampling needed)
short_doc = "Hello world"  # 11 chars -> sent as-is

# Medium document (partial sampling)
medium_doc = "This is a medium length document..."  # 500 chars -> sent as-is

# Long document (sampled)
long_doc = "Very long document..." * 100  # 2000+ chars -> sampled to 2000 chars + "..."
```

## Document Categories

The LLM classifier supports the same 16 categories as the ML classifier:

- **letter**: Business letters, correspondence
- **form**: Applications, registration forms, surveys
- **email**: Email communications, messages
- **handwritten**: Handwritten notes, manuscripts
- **advertisement**: Marketing materials, ads, promotions
- **scientific_report**: Research reports, study reports, analysis
- **scientific_publication**: Academic papers, journal articles, conference papers
- **specification**: Technical specifications, product specs
- **file_folder**: Document folders, directories, organizers
- **news_article**: News articles, press releases
- **budget**: Financial budgets, budget plans
- **invoice**: Billing documents, receipts, payment records
- **presentation**: Slides, decks, PowerPoint files
- **questionnaire**: Surveys, polls, feedback forms
- **resume**: CVs, job applications, curriculum vitae
- **memo**: Business memos, memorandums

## Configuration Options

### Groq API Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | Required | Your Groq API key |
| `GROQ_MODEL_NAME` | `llama3-8b-8192` | Model to use |
| `GROQ_TEMPERATURE` | `0.1` | Response randomness (0.0-1.0) |
| `GROQ_MAX_TOKENS` | `1000` | Maximum tokens in response |
| `GROQ_TIMEOUT` | `30` | API timeout in seconds |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | API base URL |

### Document Sampling Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MAX_CHARS` | `2000` | Maximum characters to send to LLM |
| `LLM_MAX_WORDS` | `300` | Maximum words to send to LLM |

## Architecture

### Components

1. **GroqAPI**: Custom LangChain LLM wrapper for Groq API
2. **DocumentCategoryParser**: Parses LLM responses into structured results
3. **LLMDocumentClassifier**: Main classifier with document sampling and performance tracking

### Response Format

The LLM is prompted to return JSON responses:

```json
{
    "category": "letter",
    "confidence": 0.85,
    "reasoning": "This document appears to be a business letter because..."
}
```

### Error Handling

- **API Failures**: Graceful fallback with error logging
- **Parsing Errors**: Fallback parsing with regex extraction
- **Unknown Categories**: Default to 'letter' category
- **Timeout Handling**: Configurable timeout for API calls

## Performance Characteristics

- **Processing Time**: Typically 1-3 seconds per document (Groq is very fast)
- **API Latency**: Very low latency due to Groq's optimized infrastructure
- **Token Usage**: Optimized through document sampling
- **Confidence Scores**: 0.0-1.0 scale for routing decisions
- **Sampling Efficiency**: Reduces token usage by 50-80% for long documents

## Integration with Agent

This LLM classifier is designed to work with the `agent.py` supervisory system:

1. **Routing Decision**: Agent decides whether to use ML or LLM
2. **LLM Classification**: This module handles complex documents efficiently
3. **Performance Feedback**: Results feed back into agent's learning system

## Testing

The test script (`test_llm.py`) includes:

- **API Connection Test**: Verifies Groq API access
- **Document Sampling Test**: Tests truncation logic
- **Classification Test**: Tests with various document types
- **Performance Monitoring**: Shows statistics and configuration

Run tests:

```bash
python test_llm.py
```

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   ERROR: GROQ_API_KEY not found or not set properly
   ```
   Solution: Set the environment variable in your `.env` file

2. **API Connection Failed**
   ```
   ERROR: API connection failed with status 401
   ```
   Solution: Check your API key and ensure it's valid

3. **Parsing Errors**
   ```
   ERROR: Error parsing LLM response
   ```
   Solution: The system will fallback to regex parsing

4. **Timeout Errors**
   ```
   ERROR: API connection error: timeout
   ```
   Solution: Increase `GROQ_TIMEOUT` in your `.env` file

### Getting Help

- Check the test script output for detailed error messages
- Verify your API key is correct and has sufficient credits
- Ensure you have a stable internet connection
- Review the configuration in your `.env` file

## Next Steps

After testing the LLM integration:

1. **Connect to Agent**: Integrate with the supervisory agent in `agent.py`
2. **Performance Optimization**: Fine-tune sampling limits and prompts
3. **Cost Monitoring**: Track API usage and costs with sampling
4. **Hybrid Testing**: Test the complete ML+LLM pipeline 