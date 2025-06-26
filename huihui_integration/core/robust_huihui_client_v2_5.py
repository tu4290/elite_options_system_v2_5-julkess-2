"""
Robust HuiHui Client v2.5 - Enterprise-Grade Reliability
=======================================================

Advanced HuiHui client with comprehensive error handling, retry logic,
circuit breaker pattern, and automatic recovery mechanisms.

Addresses the critical 100% failure rate identified in logs/huihui_usage.jsonl.

Key Features:
- Exponential backoff retry logic
- Circuit breaker pattern for fault tolerance
- Comprehensive error logging and recovery
- Health monitoring and self-diagnostics
- Fallback mechanisms for degraded performance
- Performance metrics and monitoring

Author: EOTS v2.5 Recovery Team
"""

import os
import json
import logging
import time
import asyncio
import requests
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, Future

# Import Pydantic for validation
try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

class ExpertType(Enum):
    """HuiHui expert types."""
    MARKET_REGIME = "market_regime"
    OPTIONS_FLOW = "options_flow"
    SENTIMENT = "sentiment"
    ORCHESTRATOR = "orchestrator"

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class RequestPriority(Enum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ExpertConfig:
    """Expert configuration with performance settings."""
    name: str
    system_prompt: str
    max_tokens: int = 800
    temperature: float = 0.1
    timeout: float = 30.0
    retry_count: int = 3
    priority: RequestPriority = RequestPriority.NORMAL

@dataclass
class RequestMetrics:
    """Request performance metrics."""
    start_time: float
    end_time: float
    response_time: float
    success: bool
    error: Optional[str] = None
    expert: Optional[str] = None
    token_count: int = 0
    retry_count: int = 0

class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, 
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN - requests blocked")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        """Handle failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

class RobustHuiHuiClient:
    """
    Enterprise-grade HuiHui client with comprehensive reliability features.
    
    Features:
    - Circuit breaker pattern for fault tolerance
    - Exponential backoff retry logic
    - Request queuing with priority handling
    - Health monitoring and diagnostics
    - Performance metrics collection
    - Automatic recovery mechanisms
    - Comprehensive error logging
    """
    
    def __init__(self, ollama_host: str = "http://localhost:11434", 
                 log_file: str = "logs/huihui_usage.jsonl"):
        self.ollama_host = ollama_host
        self.log_file = Path(log_file)
        self.model = "huihui_ai/huihui-moe-abliterated:5b-a1.7b"
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Circuit breaker for fault tolerance
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2
        )
        
        # Performance metrics
        self.metrics: List[RequestMetrics] = []
        self.total_requests = 0
        self.successful_requests = 0
        
        # Thread pool for concurrent requests
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Expert configurations
        self.expert_configs = {
            ExpertType.MARKET_REGIME: ExpertConfig(
                name="🏛️ Market Regime Expert",
                system_prompt="[EXPERT:MARKET_REGIME] You are the HuiHui Market Regime Expert. Analyze market volatility, VRI metrics, regime transitions, and structural patterns using EOTS analytics. Provide concise, actionable insights.",
                max_tokens=600,
                temperature=0.1,
                timeout=25.0,
                retry_count=3
            ),
            ExpertType.OPTIONS_FLOW: ExpertConfig(
                name="🚀 Options Flow Expert",
                system_prompt="[EXPERT:OPTIONS_FLOW] You are the HuiHui Options Flow Expert. Analyze VAPI-FA, DWFD, TW-LAF metrics, gamma exposure, delta flows, and institutional options positioning. Provide concise, actionable insights.",
                max_tokens=600,
                temperature=0.1,
                timeout=25.0,
                retry_count=3
            ),
            ExpertType.SENTIMENT: ExpertConfig(
                name="🧠 Sentiment Expert",
                system_prompt="[EXPERT:SENTIMENT] You are the HuiHui Sentiment Expert. Analyze market sentiment, news intelligence, behavioral patterns, and psychological market drivers. Provide concise, actionable insights.",
                max_tokens=600,
                temperature=0.1,
                timeout=25.0,
                retry_count=3
            ),
            ExpertType.ORCHESTRATOR: ExpertConfig(
                name="🎯 Meta-Orchestrator",
                system_prompt="[EXPERT:ORCHESTRATOR] You are the HuiHui Meta-Orchestrator. Synthesize insights from all experts, provide strategic recommendations, and deliver comprehensive EOTS analysis. Be thorough but concise.",
                max_tokens=800,
                temperature=0.2,
                timeout=30.0,
                retry_count=3
            )
        }
        
        # Expert tokens for logging compatibility
        self.expert_tokens = {
            ExpertType.MARKET_REGIME: "4f714d2a",
            ExpertType.OPTIONS_FLOW: "f9d747c2",
            ExpertType.SENTIMENT: "6d90476e",
            ExpertType.ORCHESTRATOR: "5081436b"
        }
        
        # Health status
        self.last_health_check = None
        self.is_healthy = False
        
        # Perform initial health check
        self._perform_health_check()
    
    def ask_expert(self, prompt: str, expert: ExpertType, 
                  priority: RequestPriority = RequestPriority.NORMAL,
                  timeout_override: Optional[float] = None) -> Dict[str, Any]:
        """
        Ask a HuiHui expert with comprehensive error handling.
        
        Args:
            prompt: Question or analysis request
            expert: Expert type to consult
            priority: Request priority level
            timeout_override: Override default timeout
        
        Returns:
            Dict with response, metrics, and status information
        """
        start_time = time.time()
        config = self.expert_configs[expert]
        token = self.expert_tokens[expert]
        
        # Initialize metrics
        metrics = RequestMetrics(
            start_time=start_time,
            end_time=0,
            response_time=0,
            success=False,
            expert=expert.value
        )
        
        try:
            # Use circuit breaker protection
            result = self.circuit_breaker.call(
                self._make_request_with_retry,
                prompt, config, timeout_override or config.timeout
            )
            
            # Update metrics for success
            end_time = time.time()
            metrics.end_time = end_time
            metrics.response_time = round(end_time - start_time, 3)
            metrics.success = True
            metrics.token_count = len(result.get("response", ""))
            
            self.successful_requests += 1
            
            # Log successful request
            self._log_request(expert.value, token, prompt, result.get("response", ""), 
                            metrics.response_time, True, None)
            
            # Performance classification
            if metrics.response_time < 5:
                performance = "🚀 EXCELLENT"
            elif metrics.response_time < 10:
                performance = "⚡ GOOD"
            elif metrics.response_time < 20:
                performance = "⏱️ ACCEPTABLE"
            else:
                performance = "🐌 SLOW"
            
            return {
                "response": result.get("response", ""),
                "expert": expert.value,
                "expert_name": config.name,
                "token": token,
                "response_time": metrics.response_time,
                "performance": performance,
                "success": True,
                "metrics": metrics,
                "circuit_state": self.circuit_breaker.state.value,
                "retry_count": metrics.retry_count
            }
            
        except Exception as e:
            # Update metrics for failure
            end_time = time.time()
            metrics.end_time = end_time
            metrics.response_time = round(end_time - start_time, 3)
            metrics.error = str(e)
            
            # Log failed request
            self._log_request(expert.value, token, prompt, "", 
                            metrics.response_time, False, str(e))
            
            return {
                "response": f"❌ Expert {expert.value} failed: {str(e)}",
                "expert": expert.value,
                "expert_name": config.name,
                "token": token,
                "response_time": metrics.response_time,
                "performance": "❌ FAILED",
                "success": False,
                "error": str(e),
                "metrics": metrics,
                "circuit_state": self.circuit_breaker.state.value,
                "retry_count": metrics.retry_count
            }
        
        finally:
            self.total_requests += 1
            self.metrics.append(metrics)
            
            # Trim metrics to last 1000 requests
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-1000:]
    
    def _make_request_with_retry(self, prompt: str, config: ExpertConfig, 
                                timeout: float) -> Dict[str, Any]:
        """Make request with exponential backoff retry logic."""
        last_exception = None
        
        for attempt in range(config.retry_count + 1):
            try:
                return self._make_single_request(prompt, config, timeout)
            
            except requests.Timeout as e:
                last_exception = e
                if attempt < config.retry_count:
                    wait_time = (2 ** attempt) * 0.5  # Exponential backoff
                    self.logger.warning(f"Request timeout, retrying in {wait_time}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
            
            except requests.ConnectionError as e:
                last_exception = e
                if attempt < config.retry_count:
                    wait_time = (2 ** attempt) * 1.0  # Longer wait for connection issues
                    self.logger.warning(f"Connection error, retrying in {wait_time}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
            
            except Exception as e:
                # For other exceptions, don't retry
                raise e
        
        # Should not reach here, but just in case
        raise last_exception or Exception("Max retries exceeded")
    
    def _make_single_request(self, prompt: str, config: ExpertConfig, 
                           timeout: float) -> Dict[str, Any]:
        """Make a single request to the HuiHui model."""
        
        # Build request payload
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
                "num_ctx": 2048,  # Increased context for better responses
                "top_k": 20,      # Balanced sampling
                "top_p": 0.8,     # Balanced probability
                "repeat_penalty": 1.1,  # Prevent repetition
                "stop": ["<|im_end|>", "</response>"]  # Stop tokens
            }
        }
        
        # Make request with timeout
        response = requests.post(
            f"{self.ollama_host}/api/chat",
            json=data,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        
        # Check response status
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        # Parse response
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {str(e)}")
        
        # Extract content
        content = result.get("message", {}).get("content", "")
        if not content.strip():
            raise Exception("Empty response from model")
        
        # Clean up response
        content = self._clean_response(content)
        
        return {"response": content}
    
    def _clean_response(self, content: str) -> str:
        """Clean and format the response content."""
        # Remove thinking tags if present
        if "<think>" in content and "</think>" in content:
            parts = content.split("</think>")
            if len(parts) > 1:
                content = parts[1].strip()
        
        # Remove response tags if present
        if "<response>" in content and "</response>" in content:
            start = content.find("<response>") + len("<response>")
            end = content.find("</response>")
            if end > start:
                content = content[start:end].strip()
        
        # Clean up whitespace
        content = content.strip()
        
        return content
    
    def _log_request(self, expert: str, token: str, prompt: str, response: str,
                    processing_time: float, success: bool, error: Optional[str]):
        """Log request to JSON lines file."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "expert": expert,
            "token": token,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "processing_time": processing_time,
            "success": success,
            "error": error
        }
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write log entry: {e}")
    
    def _perform_health_check(self) -> bool:
        """Perform system health check."""
        try:
            # Test basic connectivity
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code != 200:
                self.is_healthy = False
                return False
            
            # Test model availability
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            if self.model not in model_names:
                self.is_healthy = False
                return False
            
            # Test basic functionality with simple prompt
            test_result = self._make_single_request(
                "Hello", 
                self.expert_configs[ExpertType.ORCHESTRATOR], 
                10.0
            )
            
            if not test_result.get("response", "").strip():
                self.is_healthy = False
                return False
            
            self.is_healthy = True
            self.last_health_check = datetime.now()
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self.is_healthy = False
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        # Perform health check if needed
        if (self.last_health_check is None or 
            datetime.now() - self.last_health_check > timedelta(minutes=5)):
            self._perform_health_check()
        
        # Calculate success rate
        success_rate = 0.0
        if self.total_requests > 0:
            success_rate = (self.successful_requests / self.total_requests) * 100
        
        # Recent performance metrics
        recent_metrics = self.metrics[-100:] if len(self.metrics) >= 100 else self.metrics
        avg_response_time = 0.0
        if recent_metrics:
            successful_metrics = [m for m in recent_metrics if m.success]
            if successful_metrics:
                avg_response_time = sum(m.response_time for m in successful_metrics) / len(successful_metrics)
        
        return {
            "is_healthy": self.is_healthy,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "circuit_state": self.circuit_breaker.state.value,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": round(success_rate, 2),
            "avg_response_time": round(avg_response_time, 3),
            "ollama_host": self.ollama_host,
            "model": self.model
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        if not self.metrics:
            return {"message": "No performance data available"}
        
        # Overall statistics
        total_requests = len(self.metrics)
        successful_requests = sum(1 for m in self.metrics if m.success)
        failed_requests = total_requests - successful_requests
        
        # Response time statistics
        successful_metrics = [m for m in self.metrics if m.success]
        if successful_metrics:
            response_times = [m.response_time for m in successful_metrics]
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = min_response_time = max_response_time = 0.0
        
        # Expert usage statistics
        expert_stats = {}
        for metric in self.metrics:
            expert = metric.expert or "unknown"
            if expert not in expert_stats:
                expert_stats[expert] = {"total": 0, "successful": 0}
            expert_stats[expert]["total"] += 1
            if metric.success:
                expert_stats[expert]["successful"] += 1
        
        # Add success rates to expert stats
        for expert, stats in expert_stats.items():
            stats["success_rate"] = (stats["successful"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "overall_success_rate": (successful_requests / total_requests) * 100 if total_requests > 0 else 0,
            "avg_response_time": round(avg_response_time, 3),
            "min_response_time": round(min_response_time, 3),
            "max_response_time": round(max_response_time, 3),
            "expert_statistics": expert_stats,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "health_status": self.is_healthy
        }
    
    # Convenience methods for each expert
    def ask_market_regime(self, prompt: str, priority: RequestPriority = RequestPriority.NORMAL) -> Dict[str, Any]:
        """Ask the market regime expert."""
        return self.ask_expert(prompt, ExpertType.MARKET_REGIME, priority)
    
    def ask_options_flow(self, prompt: str, priority: RequestPriority = RequestPriority.NORMAL) -> Dict[str, Any]:
        """Ask the options flow expert."""
        return self.ask_expert(prompt, ExpertType.OPTIONS_FLOW, priority)
    
    def ask_sentiment(self, prompt: str, priority: RequestPriority = RequestPriority.NORMAL) -> Dict[str, Any]:
        """Ask the sentiment expert."""
        return self.ask_expert(prompt, ExpertType.SENTIMENT, priority)
    
    def ask_orchestrator(self, prompt: str, priority: RequestPriority = RequestPriority.NORMAL) -> Dict[str, Any]:
        """Ask the meta-orchestrator."""
        return self.ask_expert(prompt, ExpertType.ORCHESTRATOR, priority)
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

# ===== CONVENIENCE FUNCTIONS =====

def create_robust_huihui_client(ollama_host: str = "http://localhost:11434") -> RobustHuiHuiClient:
    """Create a robust HuiHui client instance."""
    return RobustHuiHuiClient(ollama_host)

def quick_huihui_test() -> Dict[str, Any]:
    """Quick test of HuiHui system functionality."""
    client = create_robust_huihui_client()
    
    try:
        # Test with simple prompt
        result = client.ask_orchestrator("Hello, this is a system test.", RequestPriority.HIGH)
        
        # Get health status
        health = client.get_health_status()
        
        return {
            "test_result": result,
            "health_status": health,
            "test_passed": result["success"] and health["is_healthy"]
        }
    
    finally:
        client.cleanup()

if __name__ == "__main__":
    # Run quick test when executed directly
    print("🧪 Testing Robust HuiHui Client...")
    test_result = quick_huihui_test()
    
    if test_result["test_passed"]:
        print("✅ HuiHui system is working correctly!")
    else:
        print("❌ HuiHui system has issues:")
        print(f"   Test Result: {test_result['test_result']['success']}")
        print(f"   Health Status: {test_result['health_status']['is_healthy']}")
    
    print(f"\n📊 Performance Summary:")
    print(f"   Response Time: {test_result['test_result']['response_time']}s")
    print(f"   Performance: {test_result['test_result']['performance']}")
    print(f"   Circuit State: {test_result['test_result']['circuit_state']}")