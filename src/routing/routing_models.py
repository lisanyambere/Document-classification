import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import pickle

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_LIGHTGBM = False

from ..core.base_classifier import DocumentMetadata

# Routing decisions enum
class RoutingDecision(Enum):
    ML = "ML"
    LLM = "LLM"

# Fast-extractable features for routing decisions
@dataclass
class DocumentFeatures:
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
    
    # Convert to dictionary for ML model input
    def to_dict(self) -> Dict[str, Any]:
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

# Training example for routing decisions
@dataclass
class RoutingExample:
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

# Predicts ML classifier confidence using lightweight ML model
class MLConfidencePredictor:
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.training_history = []
        self.logger = logging.getLogger(__name__)
    
    # Train confidence prediction model
    def train(self, examples: List[RoutingExample], validation_split: float = 0.2):
        if len(examples) < 50:
            self.logger.warning(f"Insufficient training data: {len(examples)} examples")
            return False
        
        X_data = []
        y_data = []
        
        for example in examples:
            features_dict = example.features.to_dict()
            X_data.append(features_dict)
            y_data.append(example.ml_confidence)
        
        X_df = pd.DataFrame(X_data)
        y = np.array(y_data)
        
        self.feature_names = list(X_df.columns)
        
        n_train = int(len(X_df) * (1 - validation_split))
        X_train, X_val = X_df.iloc[:n_train], X_df.iloc[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        try:
            if HAS_LIGHTGBM:
                self.model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
            else:
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            
            self.model.fit(X_train, y_train)
            
            val_predictions = self.model.predict(X_val)
            mse = np.mean((val_predictions - y_val) ** 2)
            mae = np.mean(np.abs(val_predictions - y_val))
            
            self.logger.info(f"Confidence predictor trained: MSE={mse:.4f}, MAE={mae:.4f}")
            
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
    
    # Predict ML classifier confidence
    def predict(self, features: DocumentFeatures) -> float:
        if not self.is_trained:
            return 0.6
        
        try:
            features_dict = features.to_dict()
            
            X = pd.DataFrame([features_dict])
            for feature in self.feature_names:
                if feature not in X.columns:
                    X[feature] = 0
            
            X = X[self.feature_names]
            
            confidence = self.model.predict(X)[0]
            
            return max(0.1, min(0.95, confidence))
            
        except Exception as e:
            self.logger.warning(f"Prediction failed: {e}")
            return 0.6
    
    # Get feature importance if model supports it
    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    # Save trained model
    def save_model(self, filepath: str):
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
    
    # Load trained model
    def load_model(self, filepath: str) -> bool:
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