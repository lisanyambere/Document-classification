"""
Learned routing system for document classification
Implements intelligent routing decisions based on document features and performance feedback
"""

import time
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import joblib

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    # Fallback to scikit-learn if LightGBM not available
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_LIGHTGBM = False

from base_classifier import DocumentClassifier, ClassificationResult, DocumentMetadata
from config_manager import get_config
from resilience import resilient, get_resilience_manager


class RoutingDecision(Enum):
    """Routing decisions"""
    ML = "ML"
    LLM = "LLM"


@dataclass
class DocumentFeatures:
    """Fast-extractable features for routing decisions"""
    word_count: int
    sentence_count: int
    avg_word_length: float
    tech_term_density: float
    formatting_complexity: float
    has_tables: bool
    has_mixed_content: bool
    file_extension: str
    sender_domain: str
    is_internal: bool
    semantic_similarity_flag: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ML model input"""
        return {
            'word_count': self.word_count,
            'sentence_count': self.sentence_count,
            'avg_word_length': self.avg_word_length,
            'tech_term_density': self.tech_term_density,
            'formatting_complexity': self.formatting_complexity,
            'has_tables': int(self.has_tables),
            'has_mixed_content': int(self.has_mixed_content),
            'file_extension_pdf': int(self.file_extension == 'pdf'),
            'file_extension_docx': int(self.file_extension == 'docx'),
            'file_extension_txt': int(self.file_extension == 'txt'),
            'is_internal': int(self.is_internal),
            'semantic_similarity_flag': int(self.semantic_similarity_flag)
        }


@dataclass
class RoutingExample:
    """Training example for routing decisions"""
    features: DocumentFeatures
    routing_decision: RoutingDecision
    ml_confidence: float
    llm_confidence: Optional[float] = None
    ml_correct: Optional[bool] = None
    llm_correct: Optional[bool] = None
    actual_category: Optional[str] = None
    processing_time_ml: Optional[float] = None
    processing_time_llm: Optional[float] = None
    cost_ml: float = 0.001
    cost_llm: float = 0.01
    timestamp: float = field(default_factory=time.time)


class FastFeatureExtractor:
    """Extracts routing features in ~2-3ms"""
    
    def __init__(self):
        # Technical terms that often confuse ML models
        self.tech_terms = {
            'algorithm', 'methodology', 'analysis', 'implementation', 
            'framework', 'architecture', 'specification', 'protocol',
            'compliance', 'regulation', 'policy', 'guideline',
            'technical', 'system', 'process', 'procedure',
            'research', 'study', 'evaluation', 'assessment'
        }
        
        # Domain patterns for sender classification
        self.internal_domains = {'company.com', 'internal.org', 'corp.com'}
        
        # Problematic content patterns (semantic similarity indicators)
        self.similarity_patterns = [
            r'\b(form|questionnaire|survey)\b',
            r'\b(news|article|publication|journal)\b', 
            r'\b(resume|cv|profile|curriculum)\b',
            r'\b(report|analysis|study|research)\b',
            r'\b(invoice|bill|receipt|payment)\b',
            r'\b(memo|memorandum|note)\b'
        ]
        
        self.logger = logging.getLogger(__name__)
    
    def extract(self, document: str, metadata: DocumentMetadata = None) -> DocumentFeatures:
        """Extract features optimized for speed (~2-3ms)"""
        import re
        
        # Handle empty or very short documents
        if not document or len(document.strip()) < 10:
            return self._create_minimal_features(metadata)
        
        # Basic text processing
        words = document.split()
        sentences = [s for s in document.split('.') if s.strip()]
        
        # Basic text metrics
        word_count = len(words)
        sentence_count = max(1, len(sentences))
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Technical density (fast approximation)
        tech_count = sum(1 for word in words if word.lower().strip('.,!?;') in self.tech_terms)
        tech_term_density = tech_count / word_count if word_count > 0 else 0
        
        # Formatting complexity (simple heuristics)
        formatting_complexity = (
            document.count('\n') + 
            document.count('\t') + 
            document.count('|') * 2 +  # Tables
            document.count('-') * 0.1   # Lists/bullet points
        ) / len(document) if len(document) > 0 else 0
        
        # Structure detection
        has_tables = '|' in document or '\t' in document or 'table' in document.lower()
        
        # Mixed content detection (semantic similarity flag)
        similarity_matches = sum(1 for pattern in self.similarity_patterns 
                               if re.search(pattern, document, re.IGNORECASE))
        has_mixed_content = similarity_matches >= 2
        semantic_similarity_flag = similarity_matches >= 1
        
        # Metadata extraction
        file_extension = (metadata.file_extension if metadata and metadata.file_extension 
                         else 'txt').lower()
        
        sender_email = metadata.sender_email if metadata and metadata.sender_email else ''
        sender_domain = sender_email.split('@')[-1] if '@' in sender_email else ''
        is_internal = sender_domain in self.internal_domains
        
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
    
    def _create_minimal_features(self, metadata: DocumentMetadata = None) -> DocumentFeatures:
        """Create minimal features for empty/invalid documents"""
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
            is_internal=sender_domain in self.internal_domains,
            semantic_similarity_flag=False
        )


class MLConfidencePredictor:
    """Predicts ML classifier confidence using lightweight ML model"""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.training_history = []
        self.logger = logging.getLogger(__name__)
    
    def train(self, examples: List[RoutingExample], validation_split: float = 0.2):
        """Train confidence prediction model"""
        if len(examples) < 50:
            self.logger.warning(f"Insufficient training data: {len(examples)} examples")
            return False
        
        # Prepare features and targets
        X_data = []
        y_data = []
        
        for example in examples:
            features_dict = example.features.to_dict()
            X_data.append(features_dict)
            y_data.append(example.ml_confidence)
        
        # Convert to DataFrame for consistent feature handling
        X_df = pd.DataFrame(X_data)
        y = np.array(y_data)
        
        # Store feature names for consistency
        self.feature_names = list(X_df.columns)
        
        # Split data
        n_train = int(len(X_df) * (1 - validation_split))
        X_train, X_val = X_df.iloc[:n_train], X_df.iloc[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        try:
            # Train model
            if HAS_LIGHTGBM:
                # LightGBM if available
                self.model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
            else:
                # Fallback to sklearn
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            
            self.model.fit(X_train, y_train)
            
            # Validate performance
            val_predictions = self.model.predict(X_val)
            mse = np.mean((val_predictions - y_val) ** 2)
            mae = np.mean(np.abs(val_predictions - y_val))
            
            self.logger.info(f"Confidence predictor trained: MSE={mse:.4f}, MAE={mae:.4f}")
            
            # Store training info
            self.training_history.append({
                'timestamp': time.time(),
                'n_examples': len(examples),
                'mse': mse,
                'mae': mae
            })
            
            self.is_trained = True
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return False
    
    def predict(self, features: DocumentFeatures) -> float:
        """Predict ML classifier confidence"""
        if not self.is_trained:
            # Return conservative estimate
            return 0.6
        
        try:
            # Prepare features
            features_dict = features.to_dict()
            
            # Ensure all features are present
            X = pd.DataFrame([features_dict])
            for feature in self.feature_names:
                if feature not in X.columns:
                    X[feature] = 0
            
            # Reorder columns to match training
            X = X[self.feature_names]
            
            # Make prediction
            confidence = self.model.predict(X)[0]
            
            # Clamp to valid range
            return max(0.1, min(0.95, confidence))
            
        except Exception as e:
            self.logger.warning(f"Prediction failed: {e}")
            return 0.6  # Conservative fallback
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if model supports it"""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'training_history': self.training_history,
                'is_trained': self.is_trained
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.training_history = model_data.get('training_history', [])
            self.is_trained = model_data.get('is_trained', True)
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False


class AdaptiveThresholdManager:
    """Manages adaptive routing thresholds based on system performance"""
    
    def __init__(self):
        config = get_config()
        self.base_threshold = config.routing.ml_threshold
        self.cost_budget_per_hour = config.routing.cost_budget_per_hour
        self.max_llm_ratio = config.routing.max_llm_ratio
        self.learning_rate = config.routing.learning_rate
        
        # State tracking
        self.current_threshold = self.base_threshold
        self.hourly_cost = 0.0
        self.hourly_llm_count = 0
        self.hourly_total_count = 0
        self.last_hour_reset = time.time()
        
        # Performance tracking
        self.performance_window = deque(maxlen=1000)
        
        self.logger = logging.getLogger(__name__)
    
    def get_threshold(self, current_load: float = 0.0) -> float:
        """Get current routing threshold with adaptive adjustments"""
        self._reset_hourly_counters()
        
        threshold = self.current_threshold
        
        # Cost-based adjustment
        if self.hourly_cost > self.cost_budget_per_hour * 0.8:
            threshold -= 0.1  # Route more to ML when approaching budget
        elif self.hourly_cost < self.cost_budget_per_hour * 0.5:
            threshold += 0.05  # Allow more LLM usage
        
        # Load-based adjustment
        if current_load > 0.8:
            threshold -= 0.1  # Prefer ML under high load
        
        # LLM ratio enforcement
        llm_ratio = self.hourly_llm_count / max(1, self.hourly_total_count)
        if llm_ratio > self.max_llm_ratio:
            threshold -= 0.15  # Force more ML routing
        
        return max(0.1, min(0.9, threshold))
    
    def update_from_feedback(self, feedback: List[RoutingExample]):
        """Update threshold based on routing performance feedback"""
        if len(feedback) < 10:
            return
        
        # Calculate routing accuracy (did we route to the better classifier?)
        correct_routings = 0
        total_routings = 0
        
        for example in feedback:
            if (example.ml_correct is not None and example.llm_correct is not None):
                total_routings += 1
                
                # Determine if routing was optimal
                if example.routing_decision == RoutingDecision.ML:
                    # Good if ML was correct or equally good but cheaper
                    if (example.ml_correct and not example.llm_correct) or \
                       (example.ml_correct == example.llm_correct):
                        correct_routings += 1
                else:  # LLM routing
                    # Good if LLM was correct and ML was wrong
                    if (example.llm_correct and not example.ml_correct):
                        correct_routings += 1
        
        if total_routings > 0:
            routing_accuracy = correct_routings / total_routings
            
            # Adjust threshold based on performance
            if routing_accuracy < 0.7:  # Poor routing performance
                # Analyze mistakes to adjust threshold
                adjustment = self._calculate_threshold_adjustment(feedback)
                self.current_threshold += adjustment * self.learning_rate
                self.current_threshold = max(0.1, min(0.9, self.current_threshold))
                
                self.logger.info(f"Adjusted threshold to {self.current_threshold:.3f} "
                               f"(accuracy: {routing_accuracy:.3f})")
    
    def _calculate_threshold_adjustment(self, feedback: List[RoutingExample]) -> float:
        """Calculate how to adjust threshold based on mistakes"""
        ml_mistakes = []  # Cases where we routed to ML but should have used LLM
        llm_mistakes = []  # Cases where we routed to LLM but should have used ML
        
        for example in feedback:
            if (example.ml_correct is not None and example.llm_correct is not None):
                if (example.routing_decision == RoutingDecision.ML and 
                    example.llm_correct and not example.ml_correct):
                    ml_mistakes.append(example.ml_confidence)
                elif (example.routing_decision == RoutingDecision.LLM and
                      example.ml_correct and not example.llm_correct):
                    llm_mistakes.append(example.ml_confidence)
        
        # If we're making too many ML mistakes with low confidence, raise threshold
        # If we're making too many LLM mistakes with high confidence, lower threshold
        
        adjustment = 0.0
        if len(ml_mistakes) > len(llm_mistakes):
            # Too many ML mistakes, raise threshold (send more to LLM)
            avg_mistake_confidence = np.mean(ml_mistakes)
            adjustment = 0.05 if avg_mistake_confidence < 0.8 else 0.02
        elif len(llm_mistakes) > len(ml_mistakes):
            # Too many LLM mistakes, lower threshold (send more to ML)
            adjustment = -0.03
        
        return adjustment
    
    def record_routing(self, decision: RoutingDecision, cost: float):
        """Record routing decision for cost tracking"""
        self._reset_hourly_counters()
        
        self.hourly_total_count += 1
        self.hourly_cost += cost
        
        if decision == RoutingDecision.LLM:
            self.hourly_llm_count += 1
    
    def _reset_hourly_counters(self):
        """Reset counters if hour has passed"""
        now = time.time()
        if now - self.last_hour_reset >= 3600:  # 1 hour
            self.hourly_cost = 0.0
            self.hourly_llm_count = 0
            self.hourly_total_count = 0
            self.last_hour_reset = now
    
    def get_status(self) -> Dict[str, Any]:
        """Get current threshold manager status"""
        self._reset_hourly_counters()
        
        llm_ratio = self.hourly_llm_count / max(1, self.hourly_total_count)
        
        return {
            'current_threshold': self.current_threshold,
            'base_threshold': self.base_threshold,
            'hourly_cost': self.hourly_cost,
            'cost_budget': self.cost_budget_per_hour,
            'hourly_llm_ratio': llm_ratio,
            'max_llm_ratio': self.max_llm_ratio,
            'hourly_total_requests': self.hourly_total_count
        }


class LearnedRoutingAgent:
    """Main routing agent with learned decision making and LLM reasoning"""
    
    def __init__(self, llm_classifier=None):
        self.feature_extractor = FastFeatureExtractor()
        self.confidence_predictor = MLConfidencePredictor()
        self.threshold_manager = AdaptiveThresholdManager()
        
        # LLM-based reasoning component
        self.llm_classifier = llm_classifier
        self.llm_reasoning_enabled = llm_classifier is not None
        
        # Performance tracking
        self.routing_history = deque(maxlen=10000)
        self.feedback_buffer = deque(maxlen=5000)
        
        # Enhanced routing statistics
        self.llm_routing_decisions = 0
        self.heuristic_routing_decisions = 0
        self.routing_accuracy_cache = {}
        
        # Resilience
        self.resilience = get_resilience_manager().get_executor("learned_routing")
        
        self.logger = logging.getLogger(__name__)
    
    def route_document(self, content: str, metadata: DocumentMetadata = None, 
                      current_load: float = 0.0) -> Tuple[RoutingDecision, float]:
        """Make intelligent routing decision with confidence score and LLM reasoning"""
        start_time = time.time()
        
        try:
            # Extract features (~2-3ms)
            features = self.feature_extractor.extract(content, metadata)
            
            # Predict ML confidence (~5-10ms)
            predicted_confidence = self.confidence_predictor.predict(features)
            
            # Get adaptive threshold (~1ms)
            threshold = self.threshold_manager.get_threshold(current_load)
            
            # Enhanced routing decision with LLM reasoning
            if self.llm_reasoning_enabled and self._should_use_llm_reasoning(features, predicted_confidence, current_load):
                decision, confidence = self._llm_based_routing(content, features, predicted_confidence, metadata)
                self.llm_routing_decisions += 1
            else:
                # Traditional heuristic-based routing
                decision = (RoutingDecision.LLM if predicted_confidence < threshold 
                           else RoutingDecision.ML)
                confidence = predicted_confidence
                self.heuristic_routing_decisions += 1
            
            # Record decision
            decision_time = (time.time() - start_time) * 1000
            self._record_routing_decision(features, confidence, decision, 
                                        threshold, decision_time)
            
            return decision, confidence
            
        except Exception as e:
            self.logger.error(f"Routing decision failed: {e}")
            # Conservative fallback
            return RoutingDecision.ML, 0.5
    
    def _should_use_llm_reasoning(self, features: DocumentFeatures, predicted_confidence: float, current_load: float) -> bool:
        """Determine if we should use LLM reasoning for routing decision"""
        # Use LLM reasoning in these cases:
        # 1. Confidence is in the uncertain range (0.4-0.7)
        # 2. Document has complex technical content
        # 3. Mixed content that might confuse heuristics
        # 4. System load is reasonable (< 0.8)
        
        uncertain_confidence = 0.4 <= predicted_confidence <= 0.7
        complex_content = features.tech_term_density > 0.1 or features.formatting_complexity > 0.05
        mixed_content = features.has_mixed_content or features.semantic_similarity_flag
        reasonable_load = current_load < 0.8
        
        return uncertain_confidence and (complex_content or mixed_content) and reasonable_load
    
    def _llm_based_routing(self, content: str, features: DocumentFeatures, 
                          predicted_confidence: float, metadata: DocumentMetadata) -> Tuple[RoutingDecision, float]:
        """Use LLM to make sophisticated routing decision"""
        try:
            # Truncate content for LLM reasoning (keep it fast)
            reasoning_content = content[:800] if len(content) > 800 else content
            
            # Create reasoning prompt
            routing_prompt = self._create_routing_prompt(reasoning_content, features, predicted_confidence)
            
            # Quick LLM call for routing decision
            reasoning_metadata = DocumentMetadata(file_extension="txt", sender_email="routing@system")
            reasoning_result = self.llm_classifier._quick_routing_decision(routing_prompt, reasoning_metadata)
            
            # Parse LLM reasoning response
            decision, confidence = self._parse_routing_response(reasoning_result, predicted_confidence)
            
            self.logger.debug(f"LLM routing: {decision.value} (confidence: {confidence:.3f}, reasoning: {reasoning_result[:100]})")
            return decision, confidence
            
        except Exception as e:
            self.logger.warning(f"LLM routing failed, using heuristics: {e}")
            # Fallback to heuristic decision
            threshold = self.threshold_manager.get_threshold(0.0)  # Use base threshold
            decision = RoutingDecision.LLM if predicted_confidence < threshold else RoutingDecision.ML
            return decision, predicted_confidence
    
    def _create_routing_prompt(self, content: str, features: DocumentFeatures, predicted_confidence: float) -> str:
        """Create prompt for LLM routing decision"""
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
    
    def _parse_routing_response(self, response: str, fallback_confidence: float) -> Tuple[RoutingDecision, float]:
        """Parse LLM routing response"""
        try:
            response_upper = response.upper()
            
            if "ROUTE: ML" in response_upper:
                # LLM recommends ML, adjust confidence upward
                confidence = min(0.9, fallback_confidence + 0.1)
                return RoutingDecision.ML, confidence
            elif "ROUTE: LLM" in response_upper:
                # LLM recommends LLM, adjust confidence downward  
                confidence = max(0.1, fallback_confidence - 0.1)
                return RoutingDecision.LLM, confidence
            else:
                # Unclear response, use heuristic with slight confidence adjustment
                threshold = 0.6  # Default threshold
                decision = RoutingDecision.LLM if fallback_confidence < threshold else RoutingDecision.ML
                return decision, fallback_confidence
                
        except Exception as e:
            self.logger.warning(f"Failed to parse routing response: {e}")
            # Conservative fallback
            return RoutingDecision.ML, fallback_confidence
    
    def add_feedback(self, example: RoutingExample):
        """Add feedback example for learning"""
        self.feedback_buffer.append(example)
        
        # Periodic learning updates
        if len(self.feedback_buffer) >= 100 and len(self.feedback_buffer) % 50 == 0:
            self._update_from_feedback()
    
    def _update_from_feedback(self):
        """Update models based on accumulated feedback"""
        feedback_list = list(self.feedback_buffer)
        
        # Update threshold manager
        self.threshold_manager.update_from_feedback(feedback_list)
        
        # Retrain confidence predictor if we have enough data
        if len(feedback_list) >= 200:
            training_success = self.confidence_predictor.train(feedback_list)
            if training_success:
                self.logger.info("Confidence predictor retrained successfully")
    
    def _record_routing_decision(self, features: DocumentFeatures, confidence: float,
                               decision: RoutingDecision, threshold: float, 
                               decision_time_ms: float):
        """Record routing decision for analysis"""
        record = {
            'timestamp': time.time(),
            'features': features,
            'predicted_confidence': confidence,
            'threshold': threshold,
            'decision': decision,
            'decision_time_ms': decision_time_ms
        }
        self.routing_history.append(record)
        
        # Record for cost tracking
        cost = 0.01 if decision == RoutingDecision.LLM else 0.001
        self.threshold_manager.record_routing(decision, cost)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.routing_history:
            return {}
        
        recent_decisions = list(self.routing_history)[-1000:]
        
        # Basic stats
        ml_count = sum(1 for d in recent_decisions if d['decision'] == RoutingDecision.ML)
        llm_count = len(recent_decisions) - ml_count
        
        avg_confidence = np.mean([d['predicted_confidence'] for d in recent_decisions])
        avg_decision_time = np.mean([d['decision_time_ms'] for d in recent_decisions])
        avg_threshold = np.mean([d['threshold'] for d in recent_decisions])
        
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
    
    def save_models(self, directory: str):
        """Save all learned models"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save confidence predictor
        self.confidence_predictor.save_model(f"{directory}/confidence_predictor.pkl")
        
        # Save routing history and feedback
        with open(f"{directory}/routing_history.pkl", 'wb') as f:
            pickle.dump(list(self.routing_history), f)
        
        with open(f"{directory}/feedback_buffer.pkl", 'wb') as f:
            pickle.dump(list(self.feedback_buffer), f)
        
        self.logger.info(f"Models saved to {directory}")
    
    def load_models(self, directory: str):
        """Load previously saved models"""
        import os
        
        # Load confidence predictor
        confidence_path = f"{directory}/confidence_predictor.pkl"
        if os.path.exists(confidence_path):
            self.confidence_predictor.load_model(confidence_path)
        
        # Load history and feedback if available
        history_path = f"{directory}/routing_history.pkl"
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
                self.routing_history.extend(history[-5000:])  # Keep recent history
        
        feedback_path = f"{directory}/feedback_buffer.pkl"
        if os.path.exists(feedback_path):
            with open(feedback_path, 'rb') as f:
                feedback = pickle.load(f)
                self.feedback_buffer.extend(feedback[-1000:])  # Keep recent feedback
        
        self.logger.info(f"Models loaded from {directory}")


# Example usage
if __name__ == "__main__":
    # Initialize routing agent
    agent = LearnedRoutingAgent()
    
    # Test document routing
    sample_document = """
    Technical Implementation Report
    
    This document outlines the methodology for implementing a new classification
    system. The analysis includes performance metrics and architectural considerations.
    """
    
    sample_metadata = DocumentMetadata(
        sender_email="engineer@company.com",
        file_extension="pdf"
    )
    
    # Route document
    decision, confidence = agent.route_document(sample_document, sample_metadata)
    
    print(f"Routing Decision: {decision.value}")
    print(f"Predicted ML Confidence: {confidence:.3f}")
    print(f"Performance Stats: {agent.get_performance_stats()}")