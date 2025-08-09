import time
import logging
import pickle
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

from ..core.base_classifier import DocumentMetadata
from .routing_models import RoutingDecision, DocumentFeatures, RoutingExample, MLConfidencePredictor
from .feature_extractor import FastFeatureExtractor
from .threshold_manager import AdaptiveThresholdManager
from ..core.resilience import get_resilience_manager

# Main routing agent with learned decision making and LLM reasoning
class LearnedRoutingAgent:
    
    def __init__(self, llm_classifier=None):
        self.feature_extractor = FastFeatureExtractor()
        self.confidence_predictor = MLConfidencePredictor()
        self.threshold_manager = AdaptiveThresholdManager()
        
        self.llm_classifier = llm_classifier
        self.llm_reasoning_enabled = llm_classifier is not None
        
        self.routing_history = deque(maxlen=10000)
        self.feedback_buffer = deque(maxlen=5000)
        
        self.llm_routing_decisions = 0
        self.heuristic_routing_decisions = 0
        self.routing_accuracy_cache = {}
        
        self.resilience = get_resilience_manager().get_executor("learned_routing")
        
        self.logger = logging.getLogger(__name__)
    
    # Make intelligent routing decision with confidence score and LLM reasoning
    def route_document(self, content: str, metadata: DocumentMetadata = None, 
                      current_load: float = 0.0) -> Tuple[RoutingDecision, float]:
        start_time = time.time()
        
        try:
            features = self.feature_extractor.extract(content, metadata)
            
            predicted_confidence = self.confidence_predictor.predict(features)
            
            threshold = self.threshold_manager.get_threshold(current_load)
            
            if self.llm_reasoning_enabled and self._should_use_llm_reasoning(features, predicted_confidence, current_load):
                decision, confidence = self._llm_based_routing(content, features, predicted_confidence, metadata)
                self.llm_routing_decisions += 1
            else:
                decision = (RoutingDecision.LLM if predicted_confidence < threshold 
                           else RoutingDecision.ML)
                confidence = predicted_confidence
                self.heuristic_routing_decisions += 1
                
                self.logger.info(f"HEURISTIC ROUTING DECISION:")
                self.logger.info(f"Decision: {decision.value} (confidence: {confidence:.3f})")
                self.logger.info(f"Threshold: {threshold:.3f}, System Load: {current_load:.2f}")
                self.logger.info(f"Features - Words: {features.word_count}, Tech Density: {features.tech_term_density:.3f}")
                self.logger.info("-" * 80)
            
            decision_time = (time.time() - start_time) * 1000
            self._record_routing_decision(features, confidence, decision, 
                                        threshold, decision_time)
            
            return decision, confidence
            
        except Exception as e:
            self.logger.error(f"Routing decision failed: {e}")
            return RoutingDecision.ML, 0.5
    
    # Determine if we should use LLM reasoning for routing decision
    def _should_use_llm_reasoning(self, features: DocumentFeatures, predicted_confidence: float, current_load: float) -> bool:
        uncertain_confidence = 0.4 <= predicted_confidence <= 0.7
        complex_content = features.tech_term_density > 0.1 or features.formatting_complexity > 0.05
        mixed_content = features.has_mixed_content or features.semantic_similarity_flag
        reasonable_load = current_load < 0.8
        
        return uncertain_confidence and (complex_content or mixed_content) and reasonable_load
    
    # Use LLM to make sophisticated routing decision
    def _llm_based_routing(self, content: str, features: DocumentFeatures, 
                          predicted_confidence: float, metadata: DocumentMetadata) -> Tuple[RoutingDecision, float]:
        try:
            reasoning_content = content[:800] if len(content) > 800 else content
            
            routing_prompt = self._create_routing_prompt(reasoning_content, features, predicted_confidence)
            
            reasoning_metadata = DocumentMetadata(file_extension="txt", sender_email="routing@system")
            reasoning_result = self.llm_classifier._quick_routing_decision(routing_prompt, reasoning_metadata)
            
            decision, confidence = self._parse_routing_response(reasoning_result, predicted_confidence)
            
            self.logger.info(f"LLM ROUTING DECISION for document:")
            self.logger.info(f"Decision: {decision.value} (confidence: {confidence:.3f})")
            self.logger.info(f"Full LLM Reasoning: {reasoning_result}")
            self.logger.info("-" * 80)
            return decision, confidence
            
        except Exception as e:
            self.logger.warning(f"LLM routing failed, using heuristics: {e}")
            threshold = self.threshold_manager.get_threshold(0.0)
            decision = RoutingDecision.LLM if predicted_confidence < threshold else RoutingDecision.ML
            return decision, predicted_confidence
    
    # Create prompt for LLM routing decision
    def _create_routing_prompt(self, content: str, features: DocumentFeatures, predicted_confidence: float) -> str:
        return f"""ROUTING DECISION TASK:
You need to decide whether to route this document to ML (fast, traditional) or LLM (slower, more accurate) classification.

DOCUMENT PREVIEW:
{content}

DOCUMENT FEATURES:
- Words: {features.word_count}
- Technical density: {features.tech_term_density:.2f}
- Formatting complexity: {features.formatting_complexity:.2f}
- Has tables: {features.has_tables}
- Mixed content: {features.has_mixed_content}
- File type: {features.file_extension}
- Predicted ML confidence: {predicted_confidence:.2f}

ROUTING GUIDELINES:
- Route to ML if: Simple structure, clear category patterns, high predicted confidence (>0.7)
- Route to LLM if: Complex content, technical specifications, ambiguous categories, low confidence (<0.5)
- Consider: ML is 10x faster but less accurate on complex documents

Respond with: "ROUTE: ML" or "ROUTE: LLM" followed by your reasoning in 1-2 sentences."""
    
    # Parse LLM routing response
    def _parse_routing_response(self, response: str, fallback_confidence: float) -> Tuple[RoutingDecision, float]:
        try:
            response_upper = response.upper()
            
            if "ROUTE: ML" in response_upper:
                confidence = min(0.9, fallback_confidence + 0.1)
                return RoutingDecision.ML, confidence
            elif "ROUTE: LLM" in response_upper:
                confidence = max(0.1, fallback_confidence - 0.1)
                return RoutingDecision.LLM, confidence
            else:
                threshold = 0.6
                decision = RoutingDecision.LLM if fallback_confidence < threshold else RoutingDecision.ML
                return decision, fallback_confidence
                
        except Exception as e:
            self.logger.warning(f"Failed to parse routing response: {e}")
            return RoutingDecision.ML, fallback_confidence
    
    # Add feedback example for learning
    def add_feedback(self, example: RoutingExample):
        self.feedback_buffer.append(example)
        
        if len(self.feedback_buffer) >= 100 and len(self.feedback_buffer) % 50 == 0:
            self._update_from_feedback()
    
    # Update models based on accumulated feedback
    def _update_from_feedback(self):
        feedback_list = list(self.feedback_buffer)
        
        self.threshold_manager.update_from_feedback(feedback_list)
        
        if len(feedback_list) >= 200:
            training_success = self.confidence_predictor.train(feedback_list)
            if training_success:
                self.logger.info("Confidence predictor retrained successfully")
    
    # Record routing decision for analysis
    def _record_routing_decision(self, features: DocumentFeatures, confidence: float,
                               decision: RoutingDecision, threshold: float, 
                               decision_time_ms: float):
        record = {
            'timestamp': time.time(),
            'features': features,
            'predicted_confidence': confidence,
            'threshold': threshold,
            'decision': decision,
            'decision_time_ms': decision_time_ms
        }
        self.routing_history.append(record)
        
        cost = 0.01 if decision == RoutingDecision.LLM else 0.001
        self.threshold_manager.record_routing(decision, cost)
    
    # Get comprehensive performance statistics
    def get_performance_stats(self) -> Dict[str, Any]:
        if not self.routing_history:
            return {}
        
        recent_decisions = list(self.routing_history)[-1000:]
        
        ml_count = sum(1 for d in recent_decisions if d['decision'] == RoutingDecision.ML)
        llm_count = len(recent_decisions) - ml_count
        
        avg_confidence = sum(d['predicted_confidence'] for d in recent_decisions) / len(recent_decisions)
        avg_decision_time = sum(d['decision_time_ms'] for d in recent_decisions) / len(recent_decisions)
        avg_threshold = sum(d['threshold'] for d in recent_decisions) / len(recent_decisions)
        
        return {
            'total_decisions': len(self.routing_history),
            'recent_ml_ratio': ml_count / len(recent_decisions),
            'recent_llm_ratio': llm_count / len(recent_decisions),
            'avg_predicted_confidence': avg_confidence,
            'avg_decision_time_ms': avg_decision_time,
            'avg_threshold': avg_threshold,
            'confidence_predictor_trained': self.confidence_predictor.is_trained,
            'feedback_buffer_size': len(self.feedback_buffer),
            'threshold_manager_status': self.threshold_manager.get_status()
        }
    
    # Save all learned models
    def save_models(self, directory: str):
        import os
        os.makedirs(directory, exist_ok=True)
        
        self.confidence_predictor.save_model(f"{directory}/confidence_predictor.pkl")
        
        with open(f"{directory}/routing_history.pkl", 'wb') as f:
            pickle.dump(list(self.routing_history), f)
        
        with open(f"{directory}/feedback_buffer.pkl", 'wb') as f:
            pickle.dump(list(self.feedback_buffer), f)
        
        self.logger.info(f"Models saved to {directory}")
    
    # Load previously saved models
    def load_models(self, directory: str):
        import os
        
        confidence_path = f"{directory}/confidence_predictor.pkl"
        if os.path.exists(confidence_path):
            self.confidence_predictor.load_model(confidence_path)
        
        history_path = f"{directory}/routing_history.pkl"
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
                self.routing_history.extend(history[-5000:])
        
        feedback_path = f"{directory}/feedback_buffer.pkl"
        if os.path.exists(feedback_path):
            with open(feedback_path, 'rb') as f:
                feedback = pickle.load(f)
                self.feedback_buffer.extend(feedback[-1000:])
        
        self.logger.info(f"Models loaded from {directory}")


if __name__ == "__main__":
    agent = LearnedRoutingAgent()
    
    sample_document = """
    Technical Implementation Report
    
    This document outlines the methodology for implementing a new classification
    system. The analysis includes performance metrics and architectural considerations.
    """
    
    from ..core.base_classifier import DocumentMetadata
    sample_metadata = DocumentMetadata(
        sender_email="engineer@company.com",
        file_extension="pdf"
    )
    
    decision, confidence = agent.route_document(sample_document, sample_metadata)
    
    print(f"Routing Decision: {decision.value}")
    print(f"Predicted ML Confidence: {confidence:.3f}")
    print(f"Performance Stats: {agent.get_performance_stats()}")