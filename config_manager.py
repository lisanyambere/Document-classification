"""
Centralized configuration management for document classification system
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class GroqConfig:
    """Groq API configuration"""
    api_key: str
    model_name: str = "llama3-8b-8192"
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: int = 30
    base_url: str = "https://api.groq.com/openai/v1"
    
    def __post_init__(self):
        if not self.api_key or self.api_key == "your-groq-api-key-here":
            raise ValueError("GROQ_API_KEY must be set and not be the template value")


@dataclass
class LLMConfig:
    """LLM processing configuration"""
    max_chars: int = 2000
    max_words: int = 300
    cost_per_1k_tokens: float = 0.01  # Estimated cost
    
    def __post_init__(self):
        if self.max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if self.max_words <= 0:
            raise ValueError("max_words must be positive")


@dataclass
class MLConfig:
    """ML classifier configuration"""
    model_path: str = "models/"
    max_features: int = 1000
    min_confidence_threshold: float = 0.7
    batch_size: int = 32
    
    def __post_init__(self):
        if self.min_confidence_threshold < 0 or self.min_confidence_threshold > 1:
            raise ValueError("min_confidence_threshold must be between 0 and 1")


@dataclass
class RoutingConfig:
    """Routing decision configuration"""
    ml_threshold: float = 0.75
    cost_budget_per_hour: float = 10.0
    max_llm_ratio: float = 0.3  # Max 30% to LLM
    decision_timeout_ms: int = 10
    learning_rate: float = 0.1
    
    def __post_init__(self):
        if not 0 <= self.ml_threshold <= 1:
            raise ValueError("ml_threshold must be between 0 and 1")
        if not 0 <= self.max_llm_ratio <= 1:
            raise ValueError("max_llm_ratio must be between 0 and 1")


@dataclass
class SystemConfig:
    """System-wide configuration"""
    log_level: str = "INFO"
    max_workers: int = 4
    cache_size: int = 1000
    metrics_retention_hours: int = 24
    health_check_interval: int = 60
    
    def __post_init__(self):
        if self.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ValueError("Invalid log_level")


@dataclass
class AppConfig:
    """Main application configuration"""
    environment: Environment
    groq: GroqConfig
    llm: LLMConfig
    ml: MLConfig
    routing: RoutingConfig
    system: SystemConfig
    
    # Runtime tracking
    _runtime_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def update_runtime(self, **kwargs):
        """Update configuration at runtime"""
        self._runtime_overrides.update(kwargs)
    
    def get_runtime(self, key: str, default: Any = None) -> Any:
        """Get runtime configuration value"""
        return self._runtime_overrides.get(key, default)


class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self, env_file: str = ".env", config_file: Optional[str] = None):
        self.env_file = env_file
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv(env_file)
        
        # Load configuration
        self.config = self._load_config()
        
        # Configure logging
        self._configure_logging()
    
    def _load_config(self) -> AppConfig:
        """Load configuration from environment and files"""
        
        # Determine environment
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        try:
            environment = Environment(env_name)
        except ValueError:
            self.logger.warning(f"Unknown environment '{env_name}', defaulting to development")
            environment = Environment.DEVELOPMENT
        
        # Load from config file if provided
        file_config = {}
        if self.config_file and os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
        
        # Build configuration with precedence: env vars > file > defaults
        groq_config = GroqConfig(
            api_key=os.getenv("GROQ_API_KEY") or file_config.get("groq", {}).get("api_key", ""),
            model_name=os.getenv("GROQ_MODEL_NAME") or file_config.get("groq", {}).get("model_name", "llama3-8b-8192"),
            temperature=float(os.getenv("GROQ_TEMPERATURE") or file_config.get("groq", {}).get("temperature", 0.1)),
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS") or file_config.get("groq", {}).get("max_tokens", 1000)),
            timeout=int(os.getenv("GROQ_TIMEOUT") or file_config.get("groq", {}).get("timeout", 30)),
            base_url=os.getenv("GROQ_BASE_URL") or file_config.get("groq", {}).get("base_url", "https://api.groq.com/openai/v1")
        )
        
        llm_config = LLMConfig(
            max_chars=int(os.getenv("LLM_MAX_CHARS") or file_config.get("llm", {}).get("max_chars", 2000)),
            max_words=int(os.getenv("LLM_MAX_WORDS") or file_config.get("llm", {}).get("max_words", 300)),
            cost_per_1k_tokens=float(os.getenv("LLM_COST_PER_1K") or file_config.get("llm", {}).get("cost_per_1k_tokens", 0.01))
        )
        
        ml_config = MLConfig(
            model_path=os.getenv("ML_MODEL_PATH") or file_config.get("ml", {}).get("model_path", "models/"),
            max_features=int(os.getenv("ML_MAX_FEATURES") or file_config.get("ml", {}).get("max_features", 1000)),
            min_confidence_threshold=float(os.getenv("ML_MIN_CONFIDENCE") or file_config.get("ml", {}).get("min_confidence_threshold", 0.7)),
            batch_size=int(os.getenv("ML_BATCH_SIZE") or file_config.get("ml", {}).get("batch_size", 32))
        )
        
        routing_config = RoutingConfig(
            ml_threshold=float(os.getenv("ROUTING_ML_THRESHOLD") or file_config.get("routing", {}).get("ml_threshold", 0.75)),
            cost_budget_per_hour=float(os.getenv("ROUTING_COST_BUDGET") or file_config.get("routing", {}).get("cost_budget_per_hour", 10.0)),
            max_llm_ratio=float(os.getenv("ROUTING_MAX_LLM_RATIO") or file_config.get("routing", {}).get("max_llm_ratio", 0.3)),
            decision_timeout_ms=int(os.getenv("ROUTING_TIMEOUT_MS") or file_config.get("routing", {}).get("decision_timeout_ms", 10)),
            learning_rate=float(os.getenv("ROUTING_LEARNING_RATE") or file_config.get("routing", {}).get("learning_rate", 0.1))
        )
        
        system_config = SystemConfig(
            log_level=os.getenv("LOG_LEVEL") or file_config.get("system", {}).get("log_level", "INFO"),
            max_workers=int(os.getenv("MAX_WORKERS") or file_config.get("system", {}).get("max_workers", 4)),
            cache_size=int(os.getenv("CACHE_SIZE") or file_config.get("system", {}).get("cache_size", 1000)),
            metrics_retention_hours=int(os.getenv("METRICS_RETENTION_HOURS") or file_config.get("system", {}).get("metrics_retention_hours", 24)),
            health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL") or file_config.get("system", {}).get("health_check_interval", 60))
        )
        
        return AppConfig(
            environment=environment,
            groq=groq_config,
            llm=llm_config,
            ml=ml_config,
            routing=routing_config,
            system=system_config
        )
    
    def _configure_logging(self):
        """Configure logging based on configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.system.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def get_config(self) -> AppConfig:
        """Get the current configuration"""
        return self.config
    
    def reload_config(self):
        """Reload configuration from sources"""
        load_dotenv(self.env_file, override=True)
        self.config = self._load_config()
        self._configure_logging()
        self.logger.info("Configuration reloaded")
    
    def validate_config(self) -> bool:
        """Validate the current configuration"""
        try:
            # Test Groq API key
            if not self.config.groq.api_key:
                self.logger.error("Groq API key is required")
                return False
            
            # Check model path exists
            if not os.path.exists(self.config.ml.model_path):
                self.logger.warning(f"ML model path does not exist: {self.config.ml.model_path}")
            
            # Validate routing parameters
            if self.config.routing.ml_threshold <= 0 or self.config.routing.ml_threshold >= 1:
                self.logger.error("Invalid ML routing threshold")
                return False
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def save_config_to_file(self, filepath: str):
        """Save current configuration to JSON file"""
        config_dict = {
            "environment": self.config.environment.value,
            "groq": {
                "model_name": self.config.groq.model_name,
                "temperature": self.config.groq.temperature,
                "max_tokens": self.config.groq.max_tokens,
                "timeout": self.config.groq.timeout,
                "base_url": self.config.groq.base_url
            },
            "llm": {
                "max_chars": self.config.llm.max_chars,
                "max_words": self.config.llm.max_words,
                "cost_per_1k_tokens": self.config.llm.cost_per_1k_tokens
            },
            "ml": {
                "model_path": self.config.ml.model_path,
                "max_features": self.config.ml.max_features,
                "min_confidence_threshold": self.config.ml.min_confidence_threshold,
                "batch_size": self.config.ml.batch_size
            },
            "routing": {
                "ml_threshold": self.config.routing.ml_threshold,
                "cost_budget_per_hour": self.config.routing.cost_budget_per_hour,
                "max_llm_ratio": self.config.routing.max_llm_ratio,
                "decision_timeout_ms": self.config.routing.decision_timeout_ms,
                "learning_rate": self.config.routing.learning_rate
            },
            "system": {
                "log_level": self.config.system.log_level,
                "max_workers": self.config.system.max_workers,
                "cache_size": self.config.system.cache_size,
                "metrics_retention_hours": self.config.system.metrics_retention_hours,
                "health_check_interval": self.config.system.health_check_interval
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Configuration saved to {filepath}")


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> AppConfig:
    """Get the current application configuration"""
    return get_config_manager().get_config()


# Example usage
if __name__ == "__main__":
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Validate configuration
    if config_manager.validate_config():
        config = config_manager.get_config()
        
        print(f"Environment: {config.environment.value}")
        print(f"Groq Model: {config.groq.model_name}")
        print(f"LLM Max Chars: {config.llm.max_chars}")
        print(f"ML Threshold: {config.routing.ml_threshold}")
        print(f"Log Level: {config.system.log_level}")
        
        # Save sample configuration
        config_manager.save_config_to_file("sample_config.json")
    else:
        print("Configuration validation failed")