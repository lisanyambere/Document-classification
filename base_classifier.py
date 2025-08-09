"""
Base interfaces and common types for document classification system
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import time


class DocumentCategory(Enum):
    """Standardized document categories"""
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
    """Standardized classification result"""
    category: DocumentCategory
    confidence: float
    processing_time_ms: float
    classifier_type: str  # 'ML' or 'LLM'
    reasoning: Optional[str] = None
    features_used: Optional[Dict[str, Any]] = None
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None


@dataclass
class DocumentMetadata:
    """Standardized document metadata"""
    sender_email: Optional[str] = None
    file_extension: Optional[str] = None
    file_size: Optional[int] = None
    timestamp: Optional[str] = None
    source_system: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class DocumentClassifier(ABC):
    """Abstract base class for all document classifiers"""
    
    @abstractmethod
    def predict(self, content: str, metadata: DocumentMetadata = None) -> ClassificationResult:
        """Predict document category with confidence score"""
        pass
    
    @abstractmethod
    def predict_batch(self, documents: List[Tuple[str, DocumentMetadata]]) -> List[ClassificationResult]:
        """Predict multiple documents efficiently"""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance statistics"""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if classifier is ready to make predictions"""
        pass
    
    @property
    @abstractmethod
    def classifier_type(self) -> str:
        """Return classifier type identifier"""
        pass


class ClassifierError(Exception):
    """Base exception for classifier errors"""
    pass


class ClassifierNotReadyError(ClassifierError):
    """Raised when classifier is not ready for predictions"""
    pass


class ClassificationTimeout(ClassifierError):
    """Raised when classification takes too long"""
    pass


class InvalidDocumentError(ClassifierError):
    """Raised when document format is invalid"""
    pass


def create_classification_result(
    category: str,
    confidence: float,
    classifier_type: str,
    start_time: float,
    reasoning: str = None,
    features_used: Dict[str, Any] = None,
    tokens_used: int = None,
    cost_estimate: float = None
) -> ClassificationResult:
    """Factory function to create standardized classification results"""
    
    # Convert string category to enum
    try:
        category_enum = DocumentCategory(category.lower())
    except ValueError:
        # Default to LETTER for unknown categories
        category_enum = DocumentCategory.LETTER
    
    processing_time_ms = (time.time() - start_time) * 1000
    
    return ClassificationResult(
        category=category_enum,
        confidence=max(0.0, min(1.0, confidence)),  # Clamp between 0-1
        processing_time_ms=processing_time_ms,
        classifier_type=classifier_type,
        reasoning=reasoning,
        features_used=features_used,
        tokens_used=tokens_used,
        cost_estimate=cost_estimate
    )