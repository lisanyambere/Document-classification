"""
Resilience patterns for document classification system
Includes circuit breaker, retry logic, and fallback mechanisms
"""

import time
import logging
import threading
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import functools
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeoutError
import random


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5        # Failures before opening
    recovery_timeout: int = 60        # Seconds before trying recovery
    success_threshold: int = 3        # Successes needed to close from half-open
    timeout: int = 30                # Request timeout in seconds


@dataclass  
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    base_delay: float = 1.0          # Base delay between retries
    max_delay: float = 60.0          # Maximum delay
    exponential_base: float = 2.0    # Exponential backoff multiplier
    jitter: bool = True              # Add random jitter to prevent thundering herd


class CircuitBreaker:
    """Circuit breaker implementation for API calls"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            # Execute with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                result = future.result(timeout=self.config.timeout)
            
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset from OPEN state"""
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    def _record_success(self):
        """Record successful execution"""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self.logger.info(f"Circuit breaker {self.name} closed after recovery")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    def _record_failure(self):
        """Record failed execution"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.success_count = 0
                    self.logger.warning(f"Circuit breaker {self.name} opened due to {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        with self._lock:
            return {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'is_open': self.state == CircuitState.OPEN,
                'is_half_open': self.state == CircuitState.HALF_OPEN
            }


class RetryHandler:
    """Retry handler with exponential backoff"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger(__name__)
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.config.max_attempts - 1:
                    break
                
                delay = self._calculate_delay(attempt)
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                time.sleep(delay)
        
        raise RetryExhaustedException(f"All {self.config.max_attempts} attempts failed", last_exception)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with jitter"""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            # Add ±25% jitter
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)


class FallbackHandler:
    """Fallback mechanism for failed operations"""
    
    def __init__(self):
        self.fallback_strategies: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """Register a fallback function for an operation"""
        self.fallback_strategies[operation_name] = fallback_func
        self.logger.info(f"Registered fallback for operation: {operation_name}")
    
    def execute_with_fallback(self, operation_name: str, primary_func: Callable, 
                            fallback_args: tuple = None, fallback_kwargs: dict = None) -> Any:
        """Execute primary function with fallback on failure"""
        try:
            return primary_func()
        except Exception as e:
            self.logger.warning(f"Primary operation {operation_name} failed: {e}")
            
            if operation_name in self.fallback_strategies:
                fallback_func = self.fallback_strategies[operation_name]
                fallback_args = fallback_args or ()
                fallback_kwargs = fallback_kwargs or {}
                
                try:
                    result = fallback_func(*fallback_args, **fallback_kwargs)
                    self.logger.info(f"Fallback successful for operation: {operation_name}")
                    return result
                except Exception as fallback_error:
                    self.logger.error(f"Fallback also failed for {operation_name}: {fallback_error}")
                    raise FallbackFailedException(f"Both primary and fallback failed for {operation_name}", e, fallback_error)
            else:
                self.logger.error(f"No fallback registered for operation: {operation_name}")
                raise e


class ResilientExecutor:
    """Combines circuit breaker, retry, and fallback patterns"""
    
    def __init__(self, name: str, circuit_config: CircuitBreakerConfig = None, 
                 retry_config: RetryConfig = None):
        self.name = name
        self.circuit_breaker = CircuitBreaker(name, circuit_config)
        self.retry_handler = RetryHandler(retry_config)
        self.fallback_handler = FallbackHandler()
        self.logger = logging.getLogger(__name__)
    
    def execute(self, func: Callable, fallback_name: str = None, 
                fallback_args: tuple = None, fallback_kwargs: dict = None, 
                *args, **kwargs) -> Any:
        """Execute function with full resilience patterns"""
        
        def resilient_execution():
            # Circuit breaker + retry wrapper
            def circuit_protected():
                return self.circuit_breaker.call(func, *args, **kwargs)
            
            return self.retry_handler.retry(circuit_protected)
        
        if fallback_name:
            return self.fallback_handler.execute_with_fallback(
                fallback_name, resilient_execution, fallback_args, fallback_kwargs
            )
        else:
            return resilient_execution()
    
    def register_fallback(self, name: str, fallback_func: Callable):
        """Register fallback for this executor"""
        self.fallback_handler.register_fallback(name, fallback_func)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of resilience components"""
        return {
            'name': self.name,
            'circuit_breaker': self.circuit_breaker.get_state(),
            'timestamp': time.time()
        }


# Decorators for easy use
def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """Decorator for circuit breaker protection"""
    breaker = CircuitBreaker(name, config)
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


def retry_on_failure(config: RetryConfig = None):
    """Decorator for retry logic"""
    handler = RetryHandler(config)
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return handler.retry(func, *args, **kwargs)
        return wrapper
    return decorator


def resilient(name: str, circuit_config: CircuitBreakerConfig = None, 
              retry_config: RetryConfig = None, fallback_name: str = None):
    """Decorator combining all resilience patterns"""
    executor = ResilientExecutor(name, circuit_config, retry_config)
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return executor.execute(func, fallback_name, *args, **kwargs)
        wrapper.executor = executor  # Expose executor for fallback registration
        return wrapper
    return decorator


# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class RetryExhaustedException(Exception):
    """Raised when all retry attempts are exhausted"""
    def __init__(self, message: str, last_exception: Exception):
        super().__init__(message)
        self.last_exception = last_exception


class FallbackFailedException(Exception):
    """Raised when both primary and fallback operations fail"""
    def __init__(self, message: str, primary_exception: Exception, fallback_exception: Exception):
        super().__init__(message)
        self.primary_exception = primary_exception
        self.fallback_exception = fallback_exception


# Global resilience manager
class ResilienceManager:
    """Manages all resilience components"""
    
    def __init__(self):
        self.executors: Dict[str, ResilientExecutor] = {}
        self.logger = logging.getLogger(__name__)
    
    def get_executor(self, name: str, circuit_config: CircuitBreakerConfig = None,
                    retry_config: RetryConfig = None) -> ResilientExecutor:
        """Get or create a resilient executor"""
        if name not in self.executors:
            self.executors[name] = ResilientExecutor(name, circuit_config, retry_config)
        return self.executors[name]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all components"""
        return {
            'executors': [executor.get_health_status() for executor in self.executors.values()],
            'total_executors': len(self.executors),
            'timestamp': time.time()
        }


# Global instance
_resilience_manager = ResilienceManager()


def get_resilience_manager() -> ResilienceManager:
    """Get the global resilience manager"""
    return _resilience_manager


# Example usage
if __name__ == "__main__":
    # Example of using resilient decorator
    @resilient("test_operation")
    def unreliable_function(fail_rate: float = 0.5):
        if random.random() < fail_rate:
            raise Exception("Random failure")
        return "Success!"
    
    # Register fallback
    def fallback_function():
        return "Fallback result"
    
    unreliable_function.executor.register_fallback("test_operation", fallback_function)
    
    # Test execution
    try:
        result = unreliable_function(0.8)
        print(f"Result: {result}")
    except Exception as e:
        print(f"All attempts failed: {e}")
    
    # Check health status
    manager = get_resilience_manager()
    print("Health status:", manager.get_health_status())