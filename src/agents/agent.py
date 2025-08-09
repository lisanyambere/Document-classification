"""
Clean implementation of the supervisory agent using the new architecture
This replaces the corrupted agent.py file
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple

from ..routing.learned_routing import LearnedRoutingAgent
from ..routing.routing_models import RoutingDecision, RoutingExample, DocumentFeatures
from ..core.base_classifier import DocumentClassifier, DocumentMetadata, ClassificationResult
from ..utils.training_bootstrap import TrainingDataBootstrap, FeedbackCollector
from ..core.config_manager import get_config


class SupervisoryAgent:
    
    # Initialize supervisory agent with ML and LLM classifiers
    def __init__(self, ml_classifier: DocumentClassifier, llm_classifier: DocumentClassifier):
        self.ml_classifier = ml_classifier
        self.llm_classifier = llm_classifier
        self.routing_agent = LearnedRoutingAgent(llm_classifier=llm_classifier)
        
        # Bootstrap and feedback components
        self.bootstrap = TrainingDataBootstrap(ml_classifier, llm_classifier)
        self.feedback_collector = FeedbackCollector()
        
        # System state
        self.classification_history = []
        self.system_load = 0.0
        
        # Configuration
        self.config = get_config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("SupervisoryAgent initialized successfully")
    
    # Classify document using intelligent routing
    def classify_document(self, content: str, metadata: DocumentMetadata = None) -> ClassificationResult:
        start_time = time.time()
        
        try:
            # Step 1: Get routing decision from learned agent
            routing_decision, predicted_confidence = self.routing_agent.route_document(
                content, metadata, self.system_load
            )
            
            self.logger.debug(f"Routing decision: {routing_decision.value}, "
                            f"predicted confidence: {predicted_confidence:.3f}")
            
            # Step 2: Route to appropriate classifier
            if routing_decision == RoutingDecision.ML:
                result = self.ml_classifier.predict(content, metadata)
                self.logger.debug("Routed to ML classifier")
            else:
                result = self.llm_classifier.predict(content, metadata)
                self.logger.debug("Routed to LLM classifier")
            
            # Step 3: Add routing metadata to result
            if not result.features_used:
                result.features_used = {}
            result.features_used.update({
                'routing_decision': routing_decision.value,
                'predicted_ml_confidence': predicted_confidence,
                'routing_threshold': self.routing_agent.threshold_manager.get_threshold(self.system_load)
            })
            
            # Step 4: Record for learning and analysis
            total_time_ms = (time.time() - start_time) * 1000
            self._record_classification(content, metadata, routing_decision, 
                                     predicted_confidence, result, total_time_ms)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            # Fallback to ML classifier
            try:
                return self.ml_classifier.predict(content, metadata)
            except Exception as fallback_error:
                self.logger.error(f"Fallback classification also failed: {fallback_error}")
                raise
    
    # Classify multiple documents efficiently
    def classify_batch(self, documents: List[Tuple[str, DocumentMetadata]]) -> List[ClassificationResult]:
        results = []
        for content, metadata in documents:
            try:
                result = self.classify_document(content, metadata)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to classify document: {e}")
                continue
        
        self.logger.info(f"Batch classification completed: {len(results)}/{len(documents)} successful")
        return results
    
    # Add user correction for continuous learning
    def add_user_correction(self, content: str, metadata: DocumentMetadata,
                          predicted_category: str, correct_category: str,
                          user_id: str = "anonymous"):
        # Add to feedback collector
        feedback = self.feedback_collector.add_user_correction(
            content, metadata, predicted_category, correct_category, user_id
        )
        
        # Also add to routing agent for learning
        self.routing_agent.add_feedback(feedback)
        
        self.logger.info(f"User correction added: {predicted_category} -> {correct_category}")
    
    # Run both classifiers for comparison and training data generation
    def compare_classifiers(self, content: str, metadata: DocumentMetadata = None) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # Run both classifiers
            ml_result = self.ml_classifier.predict(content, metadata)
            llm_result = self.llm_classifier.predict(content, metadata)
            
            # Create comparison analysis
            comparison = {
                'content_length': len(content),
                'metadata': metadata.to_dict() if metadata else {},
                'ml_result': {
                    'category': ml_result.category.value,
                    'confidence': ml_result.confidence,
                    'processing_time_ms': ml_result.processing_time_ms,
                    'classifier_type': ml_result.classifier_type
                },
                'llm_result': {
                    'category': llm_result.category.value,
                    'confidence': llm_result.confidence,
                    'processing_time_ms': llm_result.processing_time_ms,
                    'classifier_type': llm_result.classifier_type
                },
                'agreement': ml_result.category == llm_result.category,
                'confidence_diff': abs(ml_result.confidence - llm_result.confidence),
                'speed_ratio': llm_result.processing_time_ms / max(ml_result.processing_time_ms, 1),
                'cost_ratio': (llm_result.cost_estimate or 0.01) / (ml_result.cost_estimate or 0.001),
                'total_comparison_time_ms': (time.time() - start_time) * 1000
            }
            
            # Add comparative feedback for learning
            self.feedback_collector.add_comparative_feedback(
                content, metadata, ml_result, llm_result
            )
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Classifier comparison failed: {e}")
            return {'error': str(e)}
    
    # Generate initial training data by comparing classifiers
    def bootstrap_training_data(self, documents: List[Tuple[str, DocumentMetadata]], 
                              sample_size: int = 100) -> List[Dict[str, Any]]:
        self.logger.info(f"Bootstrapping training data with {min(len(documents), sample_size)} documents")
        
        bootstrap_data = []
        processed = 0
        
        for content, metadata in documents[:sample_size]:
            try:
                comparison = self.compare_classifiers(content, metadata)
                if 'error' not in comparison:
                    bootstrap_data.append(comparison)
                
                processed += 1
                if processed % 10 == 0:
                    self.logger.info(f"Processed {processed}/{min(len(documents), sample_size)} documents")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.warning(f"Failed to process document {processed}: {e}")
                continue
        
        self.logger.info(f"Bootstrap completed: {len(bootstrap_data)} successful comparisons")
        return bootstrap_data
    
    # Get comprehensive system performance statistics
    def get_system_statistics(self) -> Dict[str, Any]:
        # Routing agent stats
        routing_stats = self.routing_agent.get_performance_stats()
        
        # Individual classifier stats
        ml_stats = self.ml_classifier.get_performance_metrics()
        llm_stats = self.llm_classifier.get_performance_metrics()
        
        # Feedback stats
        feedback_stats = self.feedback_collector.get_feedback_stats()
        
        # System-level stats
        recent_classifications = self.classification_history[-100:] if self.classification_history else []
        
        system_stats = {
            'total_classifications': len(self.classification_history),
            'system_load': self.system_load,
            'avg_processing_time_ms': 0,
            'routing_accuracy_estimate': 0
        }
        
        if recent_classifications:
            system_stats['avg_processing_time_ms'] = sum(
                c.get('total_time_ms', 0) for c in recent_classifications
            ) / len(recent_classifications)
            
            # Rough estimate of routing accuracy (high confidence correlates with good routing)
            system_stats['routing_accuracy_estimate'] = sum(
                1 for c in recent_classifications if c.get('actual_confidence', 0) > 0.8
            ) / len(recent_classifications)
        
        return {
            'routing_agent': routing_stats,
            'ml_classifier': ml_stats,
            'llm_classifier': llm_stats,
            'feedback_system': feedback_stats,
            'system': system_stats,
            'configuration': {
                'ml_threshold': self.config.routing.ml_threshold,
                'cost_budget_per_hour': self.config.routing.cost_budget_per_hour,
                'max_llm_ratio': self.config.routing.max_llm_ratio
            }
        }
    
    # Update system load for adaptive routing
    def update_system_load(self, load: float):
        self.system_load = max(0.0, min(1.0, load))
        self.logger.debug(f"System load updated to {self.system_load:.2f}")
    
    # Save all learned models and data
    def save_models(self, directory: str):
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save routing agent models
        self.routing_agent.save_models(directory)
        
        # Save classification history
        import json
        history_file = os.path.join(directory, 'classification_history.json')
        with open(history_file, 'w') as f:
            # Convert to serializable format
            serializable_history = []
            for record in self.classification_history[-1000:]:  # Last 1000 records
                serializable_record = record.copy()
                # Convert any non-serializable objects
                if 'features' in serializable_record:
                    if hasattr(serializable_record['features'], '__dict__'):
                        serializable_record['features'] = serializable_record['features'].__dict__
                serializable_history.append(serializable_record)
            
            json.dump(serializable_history, f, indent=2)
        
        # Flush feedback collector
        self.feedback_collector.flush_feedback()
        
        self.logger.info(f"All models and data saved to {directory}")
    
    # Load previously saved models and data
    def load_models(self, directory: str):
        import os
        
        # Load routing agent models
        if os.path.exists(directory):
            self.routing_agent.load_models(directory)
            
            # Load classification history if available
            history_file = os.path.join(directory, 'classification_history.json')
            if os.path.exists(history_file):
                try:
                    import json
                    with open(history_file, 'r') as f:
                        self.classification_history = json.load(f)
                    self.logger.info(f"Loaded {len(self.classification_history)} historical records")
                except Exception as e:
                    self.logger.warning(f"Failed to load classification history: {e}")
            
            self.logger.info(f"Models loaded from {directory}")
        else:
            self.logger.warning(f"Directory {directory} does not exist")
    
    # Record classification for analysis and learning
    def _record_classification(self, content: str, metadata: DocumentMetadata,
                             routing_decision: RoutingDecision, predicted_confidence: float,
                             result: ClassificationResult, total_time_ms: float):
        
        # Extract features for feedback
        features = self.routing_agent.feature_extractor.extract(content, metadata)
        
        # Create record for history
        classification_record = {
            'timestamp': time.time(),
            'content_length': len(content),
            'routing_decision': routing_decision.value,
            'predicted_confidence': predicted_confidence,
            'actual_confidence': result.confidence,
            'category': result.category.value,
            'classifier_type': result.classifier_type,
            'total_time_ms': total_time_ms,
            'features': features  # This will be converted to dict when saving
        }
        
        self.classification_history.append(classification_record)
        
        # Create feedback example for learning
        feedback_example = RoutingExample(
            features=features,
            routing_decision=routing_decision,
            ml_confidence=predicted_confidence,
            actual_category=result.category.value,
            processing_time_ml=total_time_ms if routing_decision == RoutingDecision.ML else None,
            processing_time_llm=total_time_ms if routing_decision == RoutingDecision.LLM else None,
            cost_ml=result.cost_estimate if routing_decision == RoutingDecision.ML else 0.001,
            cost_llm=result.cost_estimate if routing_decision == RoutingDecision.LLM else 0.01
        )
        
        # Add to routing agent and feedback collector
        self.routing_agent.add_feedback(feedback_example)
        self.feedback_collector.add_performance_feedback(
            content, metadata, routing_decision, predicted_confidence, result
        )


# Example usage and testing
if __name__ == "__main__":
    # This would normally use the actual classifiers
    from document_classifier import MLDocumentClassifier
    from llm_classifier import LLMDocumentClassifier
    
    try:
        # Initialize classifiers
        print("Initializing classifiers...")
        ml_classifier = MLDocumentClassifier()
        llm_classifier = LLMDocumentClassifier()
        
        # Check if classifiers are ready
        if not ml_classifier.is_ready():
            print("Warning: ML classifier not ready (needs training)")
        
        if not llm_classifier.is_ready():
            print("Warning: LLM classifier not ready (check API key)")
        
        # Create supervisory agent
        print("Creating supervisory agent...")
        agent = SupervisoryAgent(ml_classifier, llm_classifier)
        
        # Test document classification
        sample_content = """
        Technical Implementation Report
        
        This document outlines the methodology for implementing a new classification system.
        The analysis includes performance metrics and architectural considerations.
        
        Key findings:
        - Algorithm performance: 94.5% accuracy
        - Processing speed: 1.2M documents/hour
        - Cost efficiency: $0.08 per 1000 documents
        """
        
        metadata = DocumentMetadata(
            sender_email="engineer@company.com",
            file_extension="pdf"
        )
        
        print("Classifying test document...")
        result = agent.classify_document(sample_content, metadata)
        
        print(f"Result: {result.category.value}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Classifier: {result.classifier_type}")
        print(f"Processing time: {result.processing_time_ms:.1f}ms")
        
        if result.features_used:
            routing_info = {k: v for k, v in result.features_used.items() 
                          if k.startswith('routing_')}
            print(f"Routing info: {routing_info}")
        
        # Get system stats
        print("\nSystem Statistics:")
        stats = agent.get_system_statistics()
        print(f"Total classifications: {stats['system']['total_classifications']}")
        print(f"Current system load: {stats['system']['system_load']}")
        
        routing_stats = stats['routing_agent']
        if routing_stats.get('total_decisions', 0) > 0:
            print(f"ML routing ratio: {routing_stats.get('recent_ml_ratio', 0):.2f}")
            print(f"Average decision time: {routing_stats.get('avg_decision_time_ms', 0):.1f}ms")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Trained ML models in models/ directory")
        print("2. Valid GROQ_API_KEY in .env file")
        print("3. All dependencies installed (pip install -r requirements.txt)")