import os
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
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

from ..core.base_classifier import (
    DocumentClassifier as BaseDocumentClassifier,
    ClassificationResult,
    DocumentMetadata,
    DocumentCategory,
    create_classification_result,
    ClassifierNotReadyError,
    InvalidDocumentError,
    ClassificationTimeout
)

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
        
    # Return LLM type identifier
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    # Quick completion call optimized for routing decisions
    def quick_completion(self, prompt: str, max_tokens: int = 150) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1,
            }
            
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Quick completion failed: {e}")
            return "ROUTE: ML"
    
    # Make API call to Groq
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
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
    
    # Parse the LLM response to extract classification results
    def parse(self, text: str) -> Dict[str, Any]:
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

class LLMDocumentClassifier(BaseDocumentClassifier):
    
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
        
    # Extract a sample of the document for LLM processing
    def _sample_document(self, document_content: str) -> str:
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
    
    # Classify a document using the LLM
    def classify_document(self, document_content: str, metadata: Dict[str, Any] = None) -> ClassificationResult:
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
            
            # Create result using factory function
            result = create_classification_result(
                category=parsed_result['category'],
                confidence=parsed_result['confidence'],
                classifier_type=self.classifier_type,
                start_time=start_time,
                reasoning=parsed_result['reasoning'],
                cost_estimate=0.01  # Estimated LLM cost
            )
            
            # Track classification
            self.classification_history.append({
                'timestamp': time.time(),
                'category': category.value,
                'confidence': result.confidence,
                'processing_time_ms': result.processing_time_ms,
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
    
    # Get performance statistics
    def get_performance_stats(self) -> Dict[str, Any]:
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
    
    # Save classification history to file
    def save_classification_history(self, filepath: str):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.classification_history, f)
    
    # Load classification history from file
    def load_classification_history(self, filepath: str):
        import pickle
        with open(filepath, 'rb') as f:
            self.classification_history = pickle.load(f)
    
    # Interface methods for BaseDocumentClassifier
    # Predict document category using the new interface
    def predict(self, content: str, metadata: DocumentMetadata = None) -> ClassificationResult:
        if not self.is_ready():
            raise ClassifierNotReadyError("LLM classifier not configured properly")
        
        if not content or not content.strip():
            raise InvalidDocumentError("Empty document content")
        
        # Convert metadata to dict if provided
        metadata_dict = metadata.to_dict() if metadata else None
        
        # Use existing classify_document method
        return self.classify_document(content, metadata_dict)
    
    # Predict multiple documents efficiently
    def predict_batch(self, documents: List[Tuple[str, DocumentMetadata]]) -> List[ClassificationResult]:
        return [self.predict(content, metadata) for content, metadata in documents]
    
    # Get current performance statistics
    def get_performance_metrics(self) -> Dict[str, float]:
        stats = self.get_performance_stats()
        
        # Convert to match interface
        return {
            'avg_confidence': stats.get('avg_confidence', 0.0),
            'avg_processing_time_ms': stats.get('avg_processing_time_ms', 0.0),
            'total_predictions': stats.get('total_classifications', 0),
            'model_name': stats.get('config', {}).get('model_name', 'unknown')
        }
    
    # Check if classifier is ready to make predictions
    def is_ready(self) -> bool:
        return (self.api_key is not None and 
                self.model_name is not None and
                self.llm is not None)
    
    # Return classifier type identifier
    @property
    def classifier_type(self) -> str:
        return "LLM"
    
    # Quick LLM call for routing decisions (optimized for speed)
    def _quick_routing_decision(self, routing_prompt: str, metadata: DocumentMetadata) -> str:
        try:
            # Use a shorter, faster prompt for routing
            response = self.llm.quick_completion(routing_prompt, max_tokens=150)
            return response.strip()
        except Exception as e:
            logger.error(f"Quick routing decision failed: {e}")
            return "ROUTE: ML"  # Conservative fallback
    
    # Legacy method for backwards compatibility - use classify_document instead
    def classify_document_legacy(self, document_content: str, metadata: Dict[str, Any] = None) -> ClassificationResult:
        return self.classify_document(document_content, metadata)

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