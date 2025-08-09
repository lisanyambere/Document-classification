"""
Training data bootstrap and feedback collection system
Generates initial training data and manages ongoing feedback collection
"""

import time
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..routing.routing_models import RoutingExample, RoutingDecision
from ..routing.feature_extractor import FastFeatureExtractor, DocumentFeatures
from ..core.base_classifier import DocumentClassifier, DocumentMetadata, ClassificationResult
from ..core.config_manager import get_config
from ..core.resilience import resilient


@dataclass
class GroundTruthExample:
    """Ground truth example for training"""
    content: str
    metadata: DocumentMetadata
    true_category: str
    confidence_score: float = 1.0
    source: str = "manual"  # manual, automatic, user_correction
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class TrainingDataBootstrap:
    """Generates initial training data by running both classifiers"""
    
    def __init__(self, ml_classifier: DocumentClassifier, llm_classifier: DocumentClassifier):
        self.ml_classifier = ml_classifier
        self.llm_classifier = llm_classifier
        self.feature_extractor = FastFeatureExtractor()
        
        config = get_config()
        self.max_workers = min(config.system.max_workers, 4)  
        
        self.logger = logging.getLogger(__name__)
    
    def generate_training_data(self, documents: List[Tuple[str, DocumentMetadata]], 
                             max_samples: int = 1000) -> List[RoutingExample]:
        """Generate training data by comparing both classifiers"""
        
        self.logger.info(f"Generating training data from {len(documents)} documents (max {max_samples})")
        
        # Sample documents if we have too many
        if len(documents) > max_samples:
            documents = random.sample(documents, max_samples)
        
        training_examples = []
        failed_count = 0
        
        # Process in batches 
        batch_size = 50
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            batch_examples = self._process_batch_parallel(batch)
            training_examples.extend(batch_examples)
            
            # Count failures
            failed_count += len(batch) - len(batch_examples)
            
            # Small delay between batches
            time.sleep(0.5)
        
        success_rate = (len(training_examples) / len(documents)) * 100
        self.logger.info(f"Generated {len(training_examples)} training examples "
                        f"({success_rate:.1f}% success rate)")
        
        return training_examples
    
    def _process_batch_parallel(self, batch: List[Tuple[str, DocumentMetadata]]) -> List[RoutingExample]:
        """Process a batch of documents in parallel"""
        examples = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._process_single_document, content, metadata): (content, metadata)
                for content, metadata in batch
            }
            
            # Collect results
            for future in as_completed(futures):
                try:
                    example = future.result(timeout=30)  
                    if example:
                        examples.append(example)
                except Exception as e:
                    content, metadata = futures[future]
                    self.logger.warning(f"Failed to process document: {e}")
        
        return examples
    
    def _process_single_document(self, content: str, metadata: DocumentMetadata) -> Optional[RoutingExample]:
        """Process a single document with both classifiers"""
        try:
            start_time = time.time()
            
            # Run both classifiers
            ml_result = self.ml_classifier.predict(content, metadata)
            llm_result = self.llm_classifier.predict(content, metadata)
            
            # Extract features
            features = self.feature_extractor.extract(content, metadata)
            
            # Determine which classifier would be better
            better_classifier = RoutingDecision.LLM if llm_result.confidence > ml_result.confidence else RoutingDecision.ML
            
            # Create training example
            example = RoutingExample(
                features=features,
                routing_decision=better_classifier,
                ml_confidence=ml_result.confidence,
                llm_confidence=llm_result.confidence,
                ml_correct=ml_result.category == llm_result.category,  
                llm_correct=ml_result.category == llm_result.category,  
                actual_category=ml_result.category.value,  
                processing_time_ml=ml_result.processing_time_ms,
                processing_time_llm=llm_result.processing_time_ms,
                cost_ml=ml_result.cost_estimate or 0.001,
                cost_llm=llm_result.cost_estimate or 0.01,
                timestamp=time.time()
            )
            
            return example
            
        except Exception as e:
            self.logger.warning(f"Failed to process document: {e}")
            return None
    
    def save_training_data(self, examples: List[RoutingExample], filepath: str):
        """Save training examples to file"""
        
        # Convert to serializable format
        data = []
        for example in examples:
            record = {
                'features': asdict(example.features),
                'routing_decision': example.routing_decision.value,
                'ml_confidence': example.ml_confidence,
                'llm_confidence': example.llm_confidence,
                'ml_correct': example.ml_correct,
                'llm_correct': example.llm_correct,
                'actual_category': example.actual_category,
                'processing_time_ml': example.processing_time_ml,
                'processing_time_llm': example.processing_time_llm,
                'cost_ml': example.cost_ml,
                'cost_llm': example.cost_llm,
                'timestamp': example.timestamp
            }
            data.append(record)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved {len(examples)} training examples to {filepath}")
    
    def load_training_data(self, filepath: str) -> List[RoutingExample]:
        """Load training examples from file"""
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        examples = []
        for record in data:
            # Reconstruct features
            features = DocumentFeatures(**record['features'])
            
            # Reconstruct routing example
            example = RoutingExample(
                features=features,
                routing_decision=RoutingDecision(record['routing_decision']),
                ml_confidence=record['ml_confidence'],
                llm_confidence=record.get('llm_confidence'),
                ml_correct=record.get('ml_correct'),
                llm_correct=record.get('llm_correct'),
                actual_category=record.get('actual_category'),
                processing_time_ml=record.get('processing_time_ml'),
                processing_time_llm=record.get('processing_time_llm'),
                cost_ml=record.get('cost_ml', 0.001),
                cost_llm=record.get('cost_llm', 0.01),
                timestamp=record.get('timestamp', time.time())
            )
            examples.append(example)
        
        self.logger.info(f"Loaded {len(examples)} training examples from {filepath}")
        return examples


class FeedbackCollector:
    """Collects and manages user feedback for continuous learning"""
    
    def __init__(self, feedback_file: str = "feedback.jsonl"):
        self.feedback_file = Path(feedback_file)
        self.feature_extractor = FastFeatureExtractor()
        self.feedback_buffer = []
        
        self.logger = logging.getLogger(__name__)
    
    def add_user_correction(self, content: str, metadata: DocumentMetadata,
                          predicted_category: str, correct_category: str,
                          user_id: str = "anonymous") -> RoutingExample:
        """Add user correction feedback"""
        
        features = self.feature_extractor.extract(content, metadata)
        
        # Create feedback example
        feedback = RoutingExample(
            features=features,
            routing_decision=RoutingDecision.ML,  
            ml_confidence=0.5,  
            ml_correct=predicted_category.lower() == correct_category.lower(),
            actual_category=correct_category,
            timestamp=time.time()
        )
        
        # Add to buffer
        self.feedback_buffer.append(feedback)
        
        # Also save immediately to file
        self._append_feedback_to_file({
            'type': 'user_correction',
            'user_id': user_id,
            'content_hash': hash(content),  
            'metadata': asdict(metadata) if metadata else {},
            'predicted_category': predicted_category,
            'correct_category': correct_category,
            'features': asdict(features),
            'timestamp': time.time()
        })
        
        self.logger.info(f"Added user correction: {predicted_category} -> {correct_category}")
        return feedback
    
    def add_performance_feedback(self, content: str, metadata: DocumentMetadata,
                               routing_decision: RoutingDecision, 
                               predicted_confidence: float,
                               actual_result: ClassificationResult):
        """Add performance feedback from classification results"""
        
        features = self.feature_extractor.extract(content, metadata)
        
        feedback = RoutingExample(
            features=features,
            routing_decision=routing_decision,
            ml_confidence=predicted_confidence,
            actual_category=actual_result.category.value,
            processing_time_ml=actual_result.processing_time_ms if routing_decision == RoutingDecision.ML else None,
            processing_time_llm=actual_result.processing_time_ms if routing_decision == RoutingDecision.LLM else None,
            cost_ml=0.001 if routing_decision == RoutingDecision.ML else 0.001,
            cost_llm=0.01 if routing_decision == RoutingDecision.LLM else 0.001,
            timestamp=time.time()
        )
        
        self.feedback_buffer.append(feedback)
        
        # Periodically flush to file
        if len(self.feedback_buffer) >= 100:
            self.flush_feedback()
    
    def add_comparative_feedback(self, content: str, metadata: DocumentMetadata,
                               ml_result: ClassificationResult, 
                               llm_result: ClassificationResult,
                               ground_truth: str = None):
        """Add comparative feedback from running both classifiers"""
        
        features = self.feature_extractor.extract(content, metadata)
        
        # Determine correctness if ground truth is available
        ml_correct = None
        llm_correct = None
        if ground_truth:
            ml_correct = ml_result.category.value.lower() == ground_truth.lower()
            llm_correct = llm_result.category.value.lower() == ground_truth.lower()
        
        feedback = RoutingExample(
            features=features,
            routing_decision=RoutingDecision.ML,  
            ml_confidence=ml_result.confidence,
            llm_confidence=llm_result.confidence,
            ml_correct=ml_correct,
            llm_correct=llm_correct,
            actual_category=ground_truth or ml_result.category.value,
            processing_time_ml=ml_result.processing_time_ms,
            processing_time_llm=llm_result.processing_time_ms,
            cost_ml=ml_result.cost_estimate or 0.001,
            cost_llm=llm_result.cost_estimate or 0.01,
            timestamp=time.time()
        )
        
        self.feedback_buffer.append(feedback)
    
    def flush_feedback(self):
        """Flush feedback buffer to file"""
        if not self.feedback_buffer:
            return
        
        for feedback in self.feedback_buffer:
            self._append_feedback_to_file({
                'type': 'performance_feedback',
                'routing_decision': feedback.routing_decision.value,
                'features': asdict(feedback.features),
                'ml_confidence': feedback.ml_confidence,
                'llm_confidence': feedback.llm_confidence,
                'ml_correct': feedback.ml_correct,
                'llm_correct': feedback.llm_correct,
                'actual_category': feedback.actual_category,
                'processing_time_ml': feedback.processing_time_ml,
                'processing_time_llm': feedback.processing_time_llm,
                'cost_ml': feedback.cost_ml,
                'cost_llm': feedback.cost_llm,
                'timestamp': feedback.timestamp
            })
        
        self.logger.info(f"Flushed {len(self.feedback_buffer)} feedback examples to file")
        self.feedback_buffer.clear()
    
    def _append_feedback_to_file(self, feedback_record: Dict[str, Any]):
        """Append feedback record to JSONL file"""
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(feedback_record) + '\n')
    
    def load_feedback(self, limit: int = None) -> List[RoutingExample]:
        """Load feedback from file"""
        if not self.feedback_file.exists():
            return []
        
        examples = []
        
        with open(self.feedback_file, 'r') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                
                try:
                    record = json.loads(line.strip())
                    
                    if record.get('type') == 'performance_feedback':
                        features = DocumentFeatures(**record['features'])
                        
                        example = RoutingExample(
                            features=features,
                            routing_decision=RoutingDecision(record['routing_decision']),
                            ml_confidence=record.get('ml_confidence', 0.5),
                            llm_confidence=record.get('llm_confidence'),
                            ml_correct=record.get('ml_correct'),
                            llm_correct=record.get('llm_correct'),
                            actual_category=record.get('actual_category'),
                            processing_time_ml=record.get('processing_time_ml'),
                            processing_time_llm=record.get('processing_time_llm'),
                            cost_ml=record.get('cost_ml', 0.001),
                            cost_llm=record.get('cost_llm', 0.01),
                            timestamp=record.get('timestamp', time.time())
                        )
                        examples.append(example)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to parse feedback line {i}: {e}")
        
        self.logger.info(f"Loaded {len(examples)} feedback examples from file")
        return examples
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about collected feedback"""
        if not self.feedback_file.exists():
            return {'total_feedback': 0}
        
        stats = {
            'total_feedback': 0,
            'user_corrections': 0,
            'performance_feedback': 0,
            'feedback_by_category': {},
            'file_size_mb': self.feedback_file.stat().st_size / (1024 * 1024)
        }
        
        with open(self.feedback_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    stats['total_feedback'] += 1
                    
                    feedback_type = record.get('type', 'unknown')
                    stats[feedback_type] = stats.get(feedback_type, 0) + 1
                    
                    category = record.get('actual_category') or record.get('correct_category')
                    if category:
                        stats['feedback_by_category'][category] = stats['feedback_by_category'].get(category, 0) + 1
                        
                except Exception:
                    continue
        
        return stats


class GroundTruthManager:
    """Manages ground truth data for evaluation"""
    
    def __init__(self, ground_truth_file: str = "ground_truth.json"):
        self.ground_truth_file = Path(ground_truth_file)
        self.ground_truth_data = {}
        self.logger = logging.getLogger(__name__)
        
        # Load existing ground truth
        self.load_ground_truth()
    
    def add_ground_truth(self, content: str, metadata: DocumentMetadata, 
                        true_category: str, confidence: float = 1.0,
                        source: str = "manual"):
        """Add ground truth example"""
        
        content_hash = str(hash(content))
        
        example = GroundTruthExample(
            content=content[:200] + "..." if len(content) > 200 else content,  # Store sample
            metadata=metadata,
            true_category=true_category,
            confidence_score=confidence,
            source=source
        )
        
        self.ground_truth_data[content_hash] = example
        self.logger.info(f"Added ground truth: {true_category} (source: {source})")
    
    def get_ground_truth(self, content: str) -> Optional[GroundTruthExample]:
        """Get ground truth for content"""
        content_hash = str(hash(content))
        return self.ground_truth_data.get(content_hash)
    
    def save_ground_truth(self):
        """Save ground truth data to file"""
        data = {}
        for key, example in self.ground_truth_data.items():
            data[key] = asdict(example)
        
        with open(self.ground_truth_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved {len(data)} ground truth examples to {self.ground_truth_file}")
    
    def load_ground_truth(self):
        """Load ground truth data from file"""
        if not self.ground_truth_file.exists():
            return
        
        try:
            with open(self.ground_truth_file, 'r') as f:
                data = json.load(f)
            
            for key, record in data.items():
                metadata = DocumentMetadata(**record['metadata']) if record['metadata'] else None
                example = GroundTruthExample(
                    content=record['content'],
                    metadata=metadata,
                    true_category=record['true_category'],
                    confidence_score=record.get('confidence_score', 1.0),
                    source=record.get('source', 'unknown'),
                    timestamp=record.get('timestamp')
                )
                self.ground_truth_data[key] = example
            
            self.logger.info(f"Loaded {len(self.ground_truth_data)} ground truth examples")
            
        except Exception as e:
            self.logger.error(f"Failed to load ground truth: {e}")


# Example usage and testing
if __name__ == "__main__":
    from ..classifiers.document_classifier import MLDocumentClassifier
    from ..classifiers.llm_classifier import LLMDocumentClassifier
    
    print("Initializing REAL classifiers...")
    
    # Initialize real ML classifier
    ml_classifier = MLDocumentClassifier()
    if not ml_classifier.is_ready():
        print("Error: ML classifier not ready!")
        exit(1)
    print("ML classifier ready!")
    
    # Initialize real LLM classifier  
    llm_classifier = LLMDocumentClassifier()
    if not llm_classifier.is_ready():
        print("Error: LLM classifier not ready!")
        exit(1)  
    print("LLM classifier ready!")
    
    # Test bootstrap with REAL classifiers
    bootstrap = TrainingDataBootstrap(ml_classifier, llm_classifier)
    
    # Sample documents for training
    sample_docs = [
        ("TECHNICAL SPECIFICATION DOCUMENT\nVersion 2.1.0\nThis specification outlines the technical implementation of an advanced document classification pipeline.", 
         DocumentMetadata(sender_email="tech@company.com", file_extension="pdf")),
        ("Invoice #2024-001\nCompany: ABC Consulting Services\nDate: January 15, 2024\nServices Rendered:\n- Software Development: $1500\nTotal: $1500",
         DocumentMetadata(sender_email="billing@company.com", file_extension="pdf")),
        ("Johnson & Associates\n123 Corporate Drive\nDear Mr. Wilson,\nThank you for your inquiry about our consulting services. We would be happy to discuss your requirements.",
         DocumentMetadata(sender_email="sales@company.com", file_extension="txt"))
    ]
    
    # Generate training data using REAL ML and LLM classifiers
    print(f"\nGenerating training data with {len(sample_docs)} documents...")
    print("This will run each document through BOTH ML and LLM to compare results...")
    
    training_data = bootstrap.generate_training_data(sample_docs)
    print(f"\nGenerated {len(training_data)} training examples:")
    
    for i, example in enumerate(training_data, 1):
        print(f"\n  Example {i}:")
        print(f"    Category: {example.actual_category}")
        print(f"    Best route: {example.routing_decision.value}")
        print(f"    ML confidence: {example.ml_confidence:.3f}")
        print(f"    LLM confidence: {example.llm_confidence:.3f}")
        print(f"    ML correct: {example.ml_correct}")
        print(f"    LLM correct: {example.llm_correct}")
    
    # Test feedback collector
    print(f"\nTesting feedback collection...")
    feedback_collector = FeedbackCollector("real_training_feedback.jsonl")
    
    # Add real feedback from the generated examples
    if training_data:
        feedback_collector.add_user_correction(
            sample_docs[0][0], sample_docs[0][1],
            training_data[0].actual_category, "specification"
        )
        print("Added user correction feedback")
    
    # Get stats
    stats = feedback_collector.get_feedback_stats()
    print(f"Feedback stats: {stats}")