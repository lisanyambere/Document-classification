import time
import numpy as np
from typing import Dict, List, Any
from collections import deque
import logging
from .routing_models import RoutingExample, RoutingDecision
from ..core.config_manager import get_config

# Manages adaptive routing thresholds based on system performance
class AdaptiveThresholdManager:
    
    def __init__(self):
        config = get_config()
        self.base_threshold = config.routing.ml_threshold
        self.cost_budget_per_hour = config.routing.cost_budget_per_hour
        self.max_llm_ratio = config.routing.max_llm_ratio
        self.learning_rate = config.routing.learning_rate
        
        self.current_threshold = self.base_threshold
        self.hourly_cost = 0.0
        self.hourly_llm_count = 0
        self.hourly_total_count = 0
        self.last_hour_reset = time.time()
        
        self.performance_window = deque(maxlen=1000)
        
        self.logger = logging.getLogger(__name__)
    
    # Get current routing threshold with adaptive adjustments
    def get_threshold(self, current_load: float = 0.0) -> float:
        self._reset_hourly_counters()
        
        threshold = self.current_threshold
        
        if self.hourly_cost > self.cost_budget_per_hour * 0.8:
            threshold -= 0.1
        elif self.hourly_cost < self.cost_budget_per_hour * 0.5:
            threshold += 0.05
        
        if current_load > 0.8:
            threshold -= 0.1
        
        llm_ratio = self.hourly_llm_count / max(1, self.hourly_total_count)
        if llm_ratio > self.max_llm_ratio:
            threshold -= 0.15
        
        return max(0.1, min(0.9, threshold))
    
    # Update threshold based on routing performance feedback
    def update_from_feedback(self, feedback: List[RoutingExample]):
        if len(feedback) < 10:
            return
        
        correct_routings = 0
        total_routings = 0
        
        for example in feedback:
            if (example.ml_correct is not None and example.llm_correct is not None):
                total_routings += 1
                
                if example.routing_decision == RoutingDecision.ML:
                    if (example.ml_correct and not example.llm_correct) or \
                       (example.ml_correct == example.llm_correct):
                        correct_routings += 1
                else:
                    if (example.llm_correct and not example.ml_correct):
                        correct_routings += 1
        
        if total_routings > 0:
            routing_accuracy = correct_routings / total_routings
            
            if routing_accuracy < 0.7:
                adjustment = self._calculate_threshold_adjustment(feedback)
                self.current_threshold += adjustment * self.learning_rate
                self.current_threshold = max(0.1, min(0.9, self.current_threshold))
                
                self.logger.info(f"Adjusted threshold to {self.current_threshold:.3f} "
                               f"(accuracy: {routing_accuracy:.3f})")
    
    # Calculate how to adjust threshold based on mistakes
    def _calculate_threshold_adjustment(self, feedback: List[RoutingExample]) -> float:
        ml_mistakes = []
        llm_mistakes = []
        
        for example in feedback:
            if (example.ml_correct is not None and example.llm_correct is not None):
                if (example.routing_decision == RoutingDecision.ML and 
                    example.llm_correct and not example.ml_correct):
                    ml_mistakes.append(example.ml_confidence)
                elif (example.routing_decision == RoutingDecision.LLM and
                      example.ml_correct and not example.llm_correct):
                    llm_mistakes.append(example.ml_confidence)
        
        adjustment = 0.0
        if len(ml_mistakes) > len(llm_mistakes):
            avg_mistake_confidence = np.mean(ml_mistakes)
            adjustment = 0.05 if avg_mistake_confidence < 0.8 else 0.02
        elif len(llm_mistakes) > len(ml_mistakes):
            adjustment = -0.03
        
        return adjustment
    
    # Record routing decision for cost tracking
    def record_routing(self, decision: RoutingDecision, cost: float):
        self._reset_hourly_counters()
        
        self.hourly_total_count += 1
        self.hourly_cost += cost
        
        if decision == RoutingDecision.LLM:
            self.hourly_llm_count += 1
    
    # Reset counters if hour has passed
    def _reset_hourly_counters(self):
        now = time.time()
        if now - self.last_hour_reset >= 3600:
            self.hourly_cost = 0.0
            self.hourly_llm_count = 0
            self.hourly_total_count = 0
            self.last_hour_reset = now
    
    # Get current threshold manager status
    def get_status(self) -> Dict[str, Any]:
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