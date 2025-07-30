import os
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# LangChain imports
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentCategory(Enum):
    """Document categories matching the ML classifier"""
    LETTER = "letter"
    FORM = "form"
    EMAIL = "email"
    HANDWRITTEN = "handwritten"
    ADVERTISEMENT = "advertisement"
    SCIENTIFIC_REPORT = "scientific_report"
    SCIENTIFIC_PUBLICATION = "scientific_publication"
    SPECIFICATION = "specification"
    FILE_FOLDER = "file_folder"
    NEWS_ARTICLE = "news_article"
    BUDGET = "budget"
    INVOICE = "invoice"
    PRESENTATION = "presentation"
    QUESTIONNAIRE = "questionnaire"
    RESUME = "resume"
    MEMO = "memo"

@dataclass
class ClassificationResult:
    """Result of document classification"""
    category: DocumentCategory
    confidence: float
    reasoning: str
    processing_time_ms: float
    tokens_used: Optional[int] = None

class GroqAPI(LLM):
    """Custom LangChain LLM wrapper for Groq API"""
    
    api_key: str
    model_name: str
    temperature: float
    base_url: str
    
    def __init__(self, api_key: str = None, model_name: str = None, temperature: float = None):
        # Get configuration from environment variables
        api_key = api_key or os.getenv("GROQ_API_KEY")
        model_name = model_name or os.getenv("GROQ_MODEL_NAME")
        temperature = temperature or float(os.getenv("GROQ_TEMPERATURE"))
        base_url = os.getenv("GROQ_BASE_URL")
        
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        if not model_name:
            raise ValueError("GROQ_MODEL_NAME not found in environment variables")
        if not temperature:
            raise ValueError("GROQ_TEMPERATURE not found in environment variables")
        if not base_url:
            raise ValueError("GROQ_BASE_URL not found in environment variables")
        
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            base_url=base_url
        )
        
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Make API call to Groq"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": int(os.getenv("GROQ_MAX_TOKENS"))
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=int(os.getenv("GROQ_TIMEOUT"))
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API error: {e}")
            raise Exception(f"Failed to call Groq API: {e}")

class DocumentCategoryParser(BaseOutputParser):
    """Parse LLM output to extract category and confidence"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse the LLM response to extract classification results"""
        try:
            # Look for JSON-like structure in the response
            import re
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # Validate required fields
                if 'category' not in result or 'confidence' not in result:
                    raise ValueError("Missing required fields in JSON response")
                
                return {
                    'category': result['category'],
                    'confidence': float(result['confidence']),
                    'reasoning': result.get('reasoning', 'No reasoning provided')
                }
            
            # Fallback: try to extract category from text
            category_match = re.search(r'category["\s]*:["\s]*([a-zA-Z_]+)', text, re.IGNORECASE)
            confidence_match = re.search(r'confidence["\s]*:["\s]*([0-9.]+)', text, re.IGNORECASE)
            
            if category_match and confidence_match:
                category = category_match.group(1).lower()
                confidence = float(confidence_match.group(1))
                
                return {
                    'category': category,
                    'confidence': confidence,
                    'reasoning': text
                }
            
            # Last resort: return the full text
            return {
                'category': 'unknown',
                'confidence': 0.0,
                'reasoning': text
            }
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {
                'category': 'unknown',
                'confidence': 0.0,
                'reasoning': f"Parsing error: {str(e)}"
            }

class LLMDocumentClassifier:
    """LLM-based document classifier using Groq API"""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        # Get configuration from environment
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name or os.getenv("GROQ_MODEL_NAME")
        
        # Document sampling configuration
        self.max_chars = int(os.getenv("LLM_MAX_CHARS"))
        self.max_words = int(os.getenv("LLM_MAX_WORDS"))
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        if not self.model_name:
            raise ValueError("GROQ_MODEL_NAME not found in environment variables")
        if not self.max_chars:
            raise ValueError("LLM_MAX_CHARS not found in environment variables")
        if not self.max_words:
            raise ValueError("LLM_MAX_WORDS not found in environment variables")
        
        # Initialize Groq LLM
        self.llm = GroqAPI(api_key=self.api_key, model_name=self.model_name)
        
        # Create classification prompt
        self.classification_prompt = PromptTemplate(
            input_variables=["document_sample", "document_metadata"],
            template="""You are an expert document classifier. Analyze the following document sample and classify it into one of the specified categories.

Document Sample (first portion):
{document_sample}

Document Metadata:
{document_metadata}

Available Categories:
- letter: Business letters, correspondence
- form: Applications, registration forms, surveys
- email: Email communications, messages
- handwritten: Handwritten notes, manuscripts
- advertisement: Marketing materials, ads, promotions
- scientific_report: Research reports, study reports, analysis
- scientific_publication: Academic papers, journal articles, conference papers
- specification: Technical specifications, product specs
- file_folder: Document folders, directories, organizers
- news_article: News articles, press releases
- budget: Financial budgets, budget plans
- invoice: Billing documents, receipts, payment records
- presentation: Slides, decks, PowerPoint files
- questionnaire: Surveys, polls, feedback forms
- resume: CVs, job applications, curriculum vitae
- memo: Business memos, memorandums

Please respond with a JSON object containing:
1. "category": The most appropriate category from the list above
2. "confidence": A confidence score between 0.0 and 1.0
3. "reasoning": A brief explanation of why you chose this category

Response format:
{{
    "category": "category_name",
    "confidence": 0.85,
    "reasoning": "This document appears to be a business letter because..."
}}"""
        )
        
        # Create modern LangChain chain using RunnableSequence
        self.chain = self.classification_prompt | self.llm
        self.parser = DocumentCategoryParser()
        
        # Performance tracking
        self.classification_history = []
        
    def _sample_document(self, document_content: str) -> str:
        """Extract a sample of the document for LLM processing"""
        # Remove extra whitespace and normalize
        content = ' '.join(document_content.split())
        
        # Sample by character limit first
        if len(content) <= self.max_chars:
            return content
        
        # Truncate to character limit
        sample = content[:self.max_chars]
        
        # Try to break at word boundary
        last_space = sample.rfind(' ')
        if last_space > self.max_chars * 0.8:  # If we can break at a reasonable point
            sample = sample[:last_space]
        
        # Add ellipsis to indicate truncation
        sample += "..."
        
        return sample
    
    def classify_document(self, document_content: str, metadata: Dict[str, Any] = None) -> ClassificationResult:
        """Classify a document using the LLM"""
        start_time = time.time()
        
        try:
            # Sample the document (only send portion to LLM)
            document_sample = self._sample_document(document_content)
            
            # Prepare metadata string
            metadata_str = json.dumps(metadata, indent=2) if metadata else "No metadata provided"
            
            # Log sampling info
            original_length = len(document_content)
            sample_length = len(document_sample)
            logger.info(f"Document sampled: {original_length} -> {sample_length} chars ({sample_length/original_length*100:.1f}%)")
            
            # Make LLM call using modern LangChain API
            response = self.chain.invoke({
                "document_sample": document_sample,
                "document_metadata": metadata_str
            })
            
            # Parse response
            parsed_result = self.parser.parse(response)
            
            # Convert category string to enum
            try:
                category = DocumentCategory(parsed_result['category'])
            except ValueError:
                # If category not found, default to letter
                category = DocumentCategory.LETTER
                logger.warning(f"Unknown category '{parsed_result['category']}', defaulting to LETTER")
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create result
            result = ClassificationResult(
                category=category,
                confidence=parsed_result['confidence'],
                reasoning=parsed_result['reasoning'],
                processing_time_ms=processing_time_ms
            )
            
            # Track classification
            self.classification_history.append({
                'timestamp': time.time(),
                'category': category.value,
                'confidence': result.confidence,
                'processing_time_ms': processing_time_ms,
                'original_length': original_length,
                'sample_length': sample_length
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            # Return fallback result
            return ClassificationResult(
                category=DocumentCategory.LETTER,
                confidence=0.0,
                reasoning=f"Classification failed: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.classification_history:
            return {}
        
        recent_classifications = self.classification_history[-100:]  # Last 100
        
        avg_confidence = sum(c['confidence'] for c in recent_classifications) / len(recent_classifications)
        avg_processing_time = sum(c['processing_time_ms'] for c in recent_classifications) / len(recent_classifications)
        avg_sample_ratio = sum(c['sample_length']/c['original_length'] for c in recent_classifications) / len(recent_classifications)
        
        category_counts = {}
        for c in recent_classifications:
            category_counts[c['category']] = category_counts.get(c['category'], 0) + 1
        
        return {
            'total_classifications': len(self.classification_history),
            'avg_confidence': avg_confidence,
            'avg_processing_time_ms': avg_processing_time,
            'avg_sample_ratio': avg_sample_ratio,
            'category_distribution': category_counts,
            'recent_classifications': len(recent_classifications),
            'config': {
                'max_chars': self.max_chars,
                'max_words': self.max_words,
                'model_name': self.model_name
            }
        }
    
    def save_classification_history(self, filepath: str):
        """Save classification history to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.classification_history, f)
    
    def load_classification_history(self, filepath: str):
        """Load classification history from file"""
        import pickle
        with open(filepath, 'rb') as f:
            self.classification_history = pickle.load(f)

# Example usage and testing
if __name__ == "__main__":
    # Initialize classifier (uses environment variables)
    try:
        classifier = LLMDocumentClassifier()
        print("LLM classifier initialized successfully")
        print(f"Model: {classifier.model_name}")
        print(f"Max chars: {classifier.max_chars}")
        print(f"Max words: {classifier.max_words}")
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please check your environment variables")
        exit(1) 