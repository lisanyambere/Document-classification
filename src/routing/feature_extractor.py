import numpy as np
import logging
from typing import Optional
from ..core.base_classifier import DocumentMetadata
from .routing_models import DocumentFeatures

# Extracts routing features in ~2-3ms
class FastFeatureExtractor:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    # Extract features optimized for speed (~2-3ms)
    def extract(self, document: str, metadata: DocumentMetadata = None) -> DocumentFeatures:
        import re
        
        if not document or len(document.strip()) < 10:
            return self._create_minimal_features(metadata)
        
        words = document.split()
        sentences = [s for s in document.split('.') if s.strip()]
        
        word_count = len(words)
        sentence_count = max(1, len(sentences))
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Technical density based on word complexity
        complex_words = [w for w in words if len(w) > 8]
        tech_term_density = len(complex_words) / word_count if word_count > 0 else 0
        
        formatting_complexity = (
            document.count('\n') + 
            document.count('\t') + 
            document.count('|') * 2 +
            document.count('-') * 0.1
        ) / len(document) if len(document) > 0 else 0
        
        has_tables = '|' in document or '\t' in document or 'table' in document.lower()
        
        # Mixed content detection based on structure variation
        line_lengths = [len(line.strip()) for line in document.split('\n') if line.strip()]
        length_variance = np.var(line_lengths) if len(line_lengths) > 1 else 0
        has_mixed_content = length_variance > 500
        semantic_similarity_flag = len(set(len(w) for w in words[:50])) > 10
        
        file_extension = (metadata.file_extension if metadata and metadata.file_extension 
                         else 'txt').lower()
        
        sender_email = metadata.sender_email if metadata and metadata.sender_email else ''
        sender_domain = sender_email.split('@')[-1] if '@' in sender_email else ''
        # Simple internal detection based on common patterns
        is_internal = any(pattern in sender_domain.lower() for pattern in ['company', 'corp', 'internal', '.local']) if sender_domain else False
        
        return DocumentFeatures(
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            tech_term_density=tech_term_density,
            formatting_complexity=formatting_complexity,
            has_tables=has_tables,
            has_mixed_content=has_mixed_content,
            file_extension=file_extension,
            sender_domain=sender_domain,
            is_internal=is_internal,
            semantic_similarity_flag=semantic_similarity_flag
        )
    
    # Create minimal features for empty/invalid documents
    def _create_minimal_features(self, metadata: DocumentMetadata = None) -> DocumentFeatures:
        file_extension = (metadata.file_extension if metadata and metadata.file_extension 
                         else 'txt').lower()
        sender_email = metadata.sender_email if metadata and metadata.sender_email else ''
        sender_domain = sender_email.split('@')[-1] if '@' in sender_email else ''
        
        return DocumentFeatures(
            word_count=0,
            sentence_count=1,
            avg_word_length=0.0,
            tech_term_density=0.0,
            formatting_complexity=0.0,
            has_tables=False,
            has_mixed_content=False,
            file_extension=file_extension,
            sender_domain=sender_domain,
            is_internal=any(pattern in sender_domain.lower() for pattern in ['company', 'corp', 'internal', '.local']) if sender_domain else False,
            semantic_similarity_flag=False
        )