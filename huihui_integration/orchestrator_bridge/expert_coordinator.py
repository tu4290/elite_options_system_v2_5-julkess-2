"""
🎯 LEGENDARY EXPERT COORDINATOR V3.0 - ULTIMATE COORDINATION ENGINE
==================================================================

The most advanced expert coordination system ever created for options trading.
Manages 4 legendary experts with AI-powered decision making, dynamic load balancing,
intelligent conflict resolution, and real-time performance optimization.

Features:
- AI-Powered Expert Selection and Weighting
- Dynamic Load Balancing and Performance Optimization  
- Intelligent Conflict Resolution and Consensus Building
- Real-Time Market Condition Adaptation
- Advanced Performance Monitoring and Learning
- Predictive Coordination and Resource Management

Author: EOTS v2.5 Legendary Architecture Division
"""

# Standard library imports
import asyncio
import hashlib
import json
import logging
import random
import time
from collections import defaultdict, deque, namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import lru_cache
from statistics import mean, median
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Set, Tuple, Union

# Third-party imports
import aiohttp
import backoff
import numpy as np
from aiohttp import ClientSession, ClientTimeout
from circuitbreaker import CircuitBreaker, CircuitBreakerError, circuit
from prometheus_client import (
    CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Summary, generate_latest,
    start_http_server
)
from pydantic import BaseModel, Field, model_validator, root_validator, validator

# Custom Types and Enums
class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class AdaptiveWindow:
    """Sliding window for tracking metrics with adaptive sizing."""
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.window = deque(maxlen=max_size)
        self.sum = 0.0
        
    def add(self, value: float) -> None:
        if len(self.window) == self.max_size:
            self.sum -= self.window[0]
        self.window.append(value)
        self.sum += value
        
    @property
    def average(self) -> float:
        return self.sum / len(self.window) if self.window else 0.0
    
    def percentile(self, p: float) -> float:
        if not self.window:
            return 0.0
        return float(np.percentile(list(self.window), p))

# EOTS v2.5 Pydantic schemas
from data_models.eots_schemas_v2_5 import (
    HuiHuiAnalysisRequestV2_5,
    HuiHuiAnalysisResponseV2_5,
    HuiHuiExpertConfigV2_5,
    ProcessedDataBundleV2_5,
    FinalAnalysisBundleV2_5
)

# HuiHui core components
from huihui_integration.core.base_expert import BaseHuiHuiExpert, get_expert_registry
from huihui_integration.core.expert_communication import get_communication_protocol, ExpertMessage, MessageType
from huihui_integration.core.local_llm_client import LocalLLMClient, ModelType, ChatMessage

logger = logging.getLogger(__name__)

class CoordinationMode(Enum):
    """Advanced coordination modes for different market conditions."""
    CONSENSUS = "consensus"          # All experts agree
    WEIGHTED = "weighted"            # Performance-weighted decisions
    COMPETITIVE = "competitive"      # Best expert wins
    COLLABORATIVE = "collaborative"  # Experts work together
    EMERGENCY = "emergency"          # Crisis mode coordination
    ADAPTIVE = "adaptive"           # AI-selected mode

class ExpertPerformanceMetrics(BaseModel):
    """Comprehensive performance metrics for each expert with adaptive learning."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    expert_id: str
    accuracy_score: float = Field(ge=0.0, le=1.0)
    response_time_ms: float = Field(ge=0.0)
    confidence_reliability: float = Field(ge=0.0, le=1.0)
    market_condition_performance: Dict[str, float] = Field(default_factory=dict)
    recent_success_rate: float = Field(ge=0.0, le=1.0)
    specialization_strength: float = Field(ge=0.0, le=1.0)
    coordination_compatibility: float = Field(ge=0.0, le=1.0)
    learning_rate: float = Field(ge=0.0)
    
    # Adaptive tracking
    response_times: AdaptiveWindow = field(default_factory=AdaptiveWindow)
    error_rates: AdaptiveWindow = field(default_factory=lambda: AdaptiveWindow(1000))
    circuit_state: CircuitState = CircuitState.CLOSED
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    
    @property
    def health_score(self) -> float:
        """Calculate overall health score (0-1) based on multiple factors."""
        if self.circuit_state == CircuitState.OPEN:
            return 0.0
            
        weights = {
            'accuracy': 0.3,
            'response_time': 0.2,
            'success_rate': 0.3,
            'reliability': 0.2
        }
        
        # Normalize response time (lower is better, max 5s)
        norm_response = max(0, 1 - (self.response_time_ms / 5000))
        
        # Calculate success rate with smoothing
        success_rate = (self.successful_requests / self.total_requests) if self.total_requests > 0 else 1.0
        
        score = (
            weights['accuracy'] * self.accuracy_score +
            weights['response_time'] * norm_response +
            weights['success_rate'] * success_rate +
            weights['reliability'] * self.confidence_reliability
        )
        
        # Apply circuit breaker penalty
        if self.circuit_state == CircuitState.HALF_OPEN:
            score *= 0.7
            
        return max(0.0, min(1.0, score))
    
    def record_success(self, response_time_ms: float):
        """Record a successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.response_times.add(response_time_ms)
        self.consecutive_failures = 0
        self.response_time_ms = self.response_times.average
        self.recent_success_rate = self.successful_requests / self.total_requests
        
        # Reset circuit if needed
        if self.circuit_state != CircuitState.CLOSED:
            self.circuit_state = CircuitState.CLOSED
    
    def record_failure(self, error: Optional[Exception] = None):
        """Record a failed request."""
        self.total_requests += 1
        self.consecutive_failures += 1
        self.last_failure = datetime.utcnow()
        
        # Update circuit state
        if self.consecutive_failures >= 5:  # Threshold for circuit open
            self.circuit_state = CircuitState.OPEN
        elif self.consecutive_failures >= 3:  # Threshold for half-open
            self.circuit_state = CircuitState.HALF_OPEN
            
        # Update error rate
        error_rate = 1.0 - (self.successful_requests / self.total_requests)
        self.error_rates.add(error_rate)
    
class MarketConditionContext(BaseModel):
    """Advanced market condition context for coordination decisions."""
    volatility_regime: str
    market_trend: str
    options_flow_intensity: str
    sentiment_regime: str
    time_of_day: str
    market_stress_level: float = Field(ge=0.0, le=1.0)
    liquidity_condition: str
    news_impact_level: float = Field(ge=0.0, le=1.0)

class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = Field(
        default=5,
        description="Number of failures before opening the circuit"
    )
    recovery_timeout: int = Field(
        default=60,
        description="Seconds before attempting to close the circuit"
    )
    expected_exception: tuple = Field(
        default=(Exception,),
        description="Exceptions that should trigger the circuit breaker"
    )
    
class LoadBalancingConfig(BaseModel):
    """Configuration for adaptive load balancing."""
    initial_weight: float = Field(
        default=1.0,
        description="Initial weight for new experts"
    )
    min_weight: float = Field(
        default=0.1,
        description="Minimum weight an expert can have"
    )
    max_weight: float = Field(
        default=10.0,
        description="Maximum weight an expert can have"
    )
    weight_decay: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Decay factor for expert weights per hour"
    )
    
class RetryConfig(BaseModel):
    """Configuration for retry behavior."""
    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retry attempts"
    )
    initial_delay: float = Field(
        default=0.1,
        ge=0.0,
        description="Initial delay between retries in seconds"
    )
    max_delay: float = Field(
        default=5.0,
        ge=0.0,
        description="Maximum delay between retries in seconds"
    )
    backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        description="Exponential backoff multiplier"
    )

class CoordinationStrategy(BaseModel):
    """AI-powered coordination strategy with adaptive behaviors."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    mode: CoordinationMode = Field(
        default=CoordinationMode.ADAPTIVE,
        description="Coordination strategy mode"
    )
    expert_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Weights for weighted expert selection"
    )
    timeout_seconds: float = Field(
        default=30.0, 
        ge=0.1,
        description="Global timeout for coordination"
    )
    consensus_threshold: float = Field(
        default=0.7, 
        ge=0.5, 
        le=1.0,
        description="Threshold for consensus decisions"
    )
    confidence_threshold: float = Field(
        default=0.6, 
        ge=0.0, 
        le=1.0,
        description="Minimum confidence for accepting decisions"
    )
    parallel_execution: bool = Field(
        default=True,
        description="Enable parallel execution of expert queries"
    )
    fallback_strategy: Optional[str] = Field(
        default=None,
        description="Fallback strategy name if primary fails"
    )
    priority_expert: Optional[str] = Field(
        default=None,
        description="Expert ID to prioritize in certain modes"
    )
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig,
        description="Circuit breaker configuration"
    )
    load_balancing: LoadBalancingConfig = Field(
        default_factory=LoadBalancingConfig,
        description="Load balancing configuration"
    )
    retry: RetryConfig = Field(
        default_factory=RetryConfig,
        description="Retry configuration"
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        description="Batch size for batch processing mode"
    )
    stream_chunk_size: int = Field(
        default=1024,
        ge=1,
        description="Chunk size for streaming responses"
    )
    
    @model_validator(mode='after')
    def validate_weights(self) -> 'CoordinationStrategy':
        """Validate and normalize expert weights."""
        if self.expert_weights:
            # Normalize weights to sum to 1.0
            total = sum(self.expert_weights.values())
            if total > 0:
                self.expert_weights = {k: v/total for k, v in self.expert_weights.items()}
        return self
    
    def get_effective_weights(self, expert_metrics: Dict[str, ExpertPerformanceMetrics]) -> Dict[str, float]:
        """Get effective weights considering expert health and performance."""
        if not expert_metrics:
            return {}
            
        # Start with configured weights or equal distribution
        weights = self.expert_weights.copy() if self.expert_weights else {
            expert_id: 1.0 for expert_id in expert_metrics
        }
        
        # Apply health scores
        for expert_id, metrics in expert_metrics.items():
            if expert_id in weights:
                weights[expert_id] *= metrics.health_score
        
        # Normalize
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()} if total > 0 else weights

class MetricsCollector:
    """Collects and reports metrics for the coordinator."""
    def __init__(self):
        # Request metrics
        self.requests_total = Counter(
            'coordinator_requests_total',
            'Total number of requests',
            ['expert', 'status']
        )
        self.request_duration = Histogram(
            'coordinator_request_duration_seconds',
            'Request duration in seconds',
            ['expert']
        )
        self.expert_load = Gauge(
            'coordinator_expert_load',
            'Current load per expert',
            ['expert']
        )
        self.circuit_breaker_state = Gauge(
            'coordinator_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=half-open, 2=open)',
            ['expert']
        )
        
        # Performance metrics
        self.success_rate = Gauge(
            'coordinator_success_rate',
            'Success rate of expert decisions',
            ['expert']
        )
        self.error_rate = Gauge(
            'coordinator_error_rate',
            'Error rate of expert decisions',
            ['expert']
        )
        
    def record_request(self, expert: str, duration: float, success: bool = True):
        """Record a request with its outcome."""
        status = 'success' if success else 'error'
        self.requests_total.labels(expert=expert, status=status).inc()
        self.request_duration.labels(expert=expert).observe(duration)
        
        # Update success/error rates
        if success:
            self.success_rate.labels(expert=expert).inc()
        else:
            self.error_rate.labels(expert=expert).inc()
    
    def update_circuit_state(self, expert: str, state: CircuitState):
        """Update circuit breaker state."""
        state_value = {
            CircuitState.CLOSED: 0,
            CircuitState.HALF_OPEN: 1,
            CircuitState.OPEN: 2
        }.get(state, 0)
        self.circuit_breaker_state.labels(expert=expert).set(state_value)

class LegendaryExpertCoordinator:
    """
    🎯 LEGENDARY EXPERT COORDINATOR V4.0 - ULTIMATE EDITION
    
    The ultimate coordination engine with adaptive load balancing, circuit breakers,
    and real-time performance optimization for managing expert decisions.
    """
    
    def __init__(self, db_manager=None):
        self.logger = logger.getChild("LegendaryCoordinator")
        self.db_manager = db_manager
        self.metrics = MetricsCollector()
        
        # Expert tracking
        self.experts: Dict[str, ExpertPerformanceMetrics] = {}
        self.expert_weights: Dict[str, float] = {}
        self.expert_load: Dict[str, int] = defaultdict(int)
        self.expert_last_used: Dict[str, datetime] = {}
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Request deduplication
        self.request_cache: Dict[str, Any] = {}
        self.cache_lock = asyncio.Lock()
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        # Initialize metrics server
        self._start_metrics_server()
    
    def _start_metrics_server(self, port: int = 8000):
        """Start Prometheus metrics server."""
        try:
            import multiprocessing
            from prometheus_client import start_http_server
            
            def run_server():
                start_http_server(port)
                while True:
                    time.sleep(1)
            
            p = multiprocessing.Process(target=run_server, daemon=True)
            p.start()
            self.logger.info(f"Started metrics server on port {port}")
        except Exception as e:
            self.logger.warning(f"Failed to start metrics server: {e}")
    
    async def initialize(self):
        """Initialize the coordinator with background tasks."""
        # Start background tasks
        self._background_tasks.update({
            asyncio.create_task(self._update_metrics_loop()),
            asyncio.create_task(self._cleanup_cache_loop()),
            asyncio.create_task(self._adjust_weights_loop())
        })
        
        # Initialize circuit breakers and expert metrics
        await self._initialize_circuit_breakers()
        self._initialize_expert_metrics()
        
        # Initialize expert weights
        for expert_id in self.experts:
            self.expert_weights[expert_id] = 1.0  # Start with equal weights
            
        self.logger.info("🎯 Legendary Expert Coordinator V4.0 initialized with AI-powered coordination")
    
    async def route_request(
        self,
        request: HuiHuiAnalysisRequestV2_5,
        strategy: Optional[CoordinationStrategy] = None,
        timeout: Optional[float] = None
    ) -> HuiHuiAnalysisResponseV2_5:
        """
        Route a request to the appropriate expert(s) based on the current strategy.
        
        Args:
            request: The analysis request to route
            strategy: Optional strategy override
            timeout: Optional timeout override
            
        Returns:
            Analysis response from the selected expert(s)
        """
        strategy = strategy or self.strategy
        timeout = timeout or strategy.timeout_seconds
        request_id = str(uuid.uuid4())
        
        # Check for duplicate request
        cache_key = self._generate_cache_key(request)
        async with self.cache_lock:
            if cache_key in self.request_cache:
                cached_response, _ = self.request_cache[cache_key]
                if cached_response:
                    return cached_response
        
        try:
            # Select expert(s) based on strategy
            if strategy.mode == CoordinationMode.WEIGHTED:
                expert_id = await self._select_expert_weighted(strategy)
                response = await self.execute_with_expert(expert_id, request, timeout)
                
            elif strategy.mode == CoordinationMode.CONSENSUS:
                responses = await self._gather_consensus(request, strategy, timeout)
                response = self._aggregate_responses(responses, strategy)
                
            elif strategy.mode == CoordinationMode.COMPETITIVE:
                expert_id = await self._select_best_expert(request, strategy, timeout)
                response = await self.execute_with_expert(expert_id, request, timeout)
                
            elif strategy.mode == CoordinationMode.ADAPTIVE:
                # Adaptive mode selects the best strategy based on current conditions
                adaptive_strategy = await self._select_adaptive_strategy(request)
                return await self.route_request(request, adaptive_strategy, timeout)
                
            else:
                raise ValueError(f"Unsupported coordination mode: {strategy.mode}")
            
            # Cache successful response
            async with self.cache_lock:
                self.request_cache[cache_key] = (response, time.time())
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error in request routing: {e}")
            # Fallback to orchestrator if available
            if "orchestrator" in self.experts and "orchestrator" != getattr(request, 'expert_id', None):
                self.logger.info("Falling back to orchestrator expert")
                return await self.execute_with_expert("orchestrator", request, timeout)
            raise
    
    async def _select_expert_weighted(self, strategy: CoordinationStrategy) -> str:
        """Select an expert using weighted random selection."""
        if not self.experts:
            raise ValueError("No experts available")
            
        # Get effective weights considering health and performance
        effective_weights = strategy.get_effective_weights(self.experts)
        
        # If no weights configured, use equal distribution
        if not effective_weights:
            experts = list(self.experts.keys())
            return random.choice(experts)
            
        # Select expert based on weights
        experts, weights = zip(*effective_weights.items())
        return random.choices(experts, weights=weights, k=1)[0]
    
    async def _gather_consensus(
        self,
        request: HuiHuiAnalysisRequestV2_5,
        strategy: CoordinationStrategy,
        timeout: float
    ) -> List[HuiHuiAnalysisResponseV2_5]:
        """Gather responses from multiple experts to reach consensus."""
        # Select experts to query
        expert_ids = list(self.experts.keys())
        if not expert_ids:
            raise ValueError("No experts available for consensus")
            
        # Execute in parallel
        tasks = []
        for expert_id in expert_ids:
            task = asyncio.create_task(
                self.execute_with_expert(expert_id, request, timeout)
            )
            tasks.append(task)
            
        # Wait for all responses or timeout
        done, _ = await asyncio.wait(
            tasks,
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED
        )
        
        # Gather successful responses
        responses = []
        for task in done:
            try:
                response = await task
                responses.append(response)
            except Exception as e:
                self.logger.warning(f"Expert consensus error: {e}")
                
        return responses
    
    async def _select_best_expert(
        self,
        request: HuiHuiAnalysisRequestV2_5,
        strategy: CoordinationStrategy,
        timeout: float
    ) -> str:
        """Select the best expert based on historical performance."""
        if not self.experts:
            raise ValueError("No experts available")
            
        # Get performance metrics
        expert_scores = {}
        for expert_id in self.experts:
            metrics = self.experts.get(expert_id)
            expert_scores[expert_id] = metrics.health_score if metrics else 0.5
            
        # Select expert with highest health score
        return max(expert_scores.items(), key=lambda x: x[1])[0]
    
    async def _select_adaptive_strategy(self, request: HuiHuiAnalysisRequestV2_5) -> CoordinationStrategy:
        """Select the best strategy based on current conditions."""
        # Simple adaptive strategy - can be enhanced with ML later
        market_context = await self._analyze_market_context(request)
        
        if market_context.market_stress_level > 0.7:
            # High stress - use consensus for reliability
            return CoordinationStrategy(
                mode=CoordinationMode.CONSENSUS,
                timeout_seconds=45.0
            )
        elif market_context.volatility_regime == "high":
            # High volatility - use weighted for balanced approach
            return CoordinationStrategy(
                mode=CoordinationMode.WEIGHTED,
                timeout_seconds=30.0
            )
        else:
            # Normal conditions - use competitive for speed
            return CoordinationStrategy(
                mode=CoordinationMode.COMPETITIVE,
                timeout_seconds=15.0
            )
    
    def _generate_cache_key(self, request: HuiHuiAnalysisRequestV2_5) -> str:
        """Generate a cache key for the request."""
        request_data = request.model_dump_json()
        return hashlib.md5(request_data.encode()).hexdigest()
    
    def _aggregate_responses(
        self,
        responses: List[HuiHuiAnalysisResponseV2_5],
        strategy: CoordinationStrategy
    ) -> HuiHuiAnalysisResponseV2_5:
        """Aggregate multiple expert responses into a single response."""
        if not responses:
            raise ValueError("No responses to aggregate")
            
        # For simplicity, return the first response with averaged confidence
        # In a real implementation, you might want to implement more sophisticated
        # aggregation logic based on the response type
        avg_confidence = sum(r.confidence for r in responses) / len(responses)
        result = responses[0].copy()
        result.confidence = avg_confidence
        return result
    
    def _initialize_expert_metrics(self):
        """Initialize performance metrics for all experts."""
        for expert_id in self.experts:
            self.experts[expert_id] = ExpertPerformanceMetrics(
                expert_id=expert_id,
                accuracy_score=0.8,  # Default starting accuracy
                response_time_ms=1000.0,  # Default response time
                confidence_reliability=0.8,
                recent_success_rate=0.8,
                specialization_strength=0.7,
                coordination_compatibility=0.9,
                learning_rate=0.1
            )
    
    async def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for all experts."""
        for expert_id in self.experts:
            self.circuit_breakers[expert_id] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60,
                name=f"expert_{expert_id}",
                state_storage=CircuitRedisStorage()
            )
            # Add metrics listener
            class MetricsListener(CircuitBreakerListener):
                def __init__(self, metrics, expert_id):
                    self.metrics = metrics
                    self.expert_id = expert_id
                    
                def state_change(self, cb, old_state, new_state):
                    self.metrics.update_circuit_state(self.expert_id, new_state)
            
            # Register the listener
            self.circuit_breakers[expert_id].add_listener(
                MetricsListener(self.metrics, expert_id)
            )
    
    async def _update_metrics_loop(self):
        """Background task to update metrics."""
        while not self._shutdown_event.is_set():
            try:
                # Update expert load metrics
                for expert, load in self.expert_load.items():
                    self.metrics.expert_load.labels(expert=expert).set(load)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in metrics update loop: {e}")
                await asyncio.sleep(5)  # Prevent tight loop on error
    
    async def _cleanup_cache_loop(self, ttl: int = 300):
        """Background task to clean up old cache entries."""
        while not self._shutdown_event.is_set():
            try:
                now = time.time()
                async with self.cache_lock:
                    # Remove entries older than TTL
                    expired = [k for k, (_, timestamp) in self.request_cache.items() 
                             if now - timestamp > ttl]
                    for k in expired:
                        del self.request_cache[k]
                
                await asyncio.sleep(60)  # Clean up every minute
                
            except Exception as e:
                self.logger.error(f"Error in cache cleanup loop: {e}")
                await asyncio.sleep(5)  # Prevent tight loop on error
    
    async def _adjust_weights_loop(self):
        """Background task to adjust expert weights based on performance."""
        while not self._shutdown_event.is_set():
            try:
                await self._update_expert_weights()
                await asyncio.sleep(60)  # Adjust weights every minute
                
            except Exception as e:
                self.logger.error(f"Error in weight adjustment loop: {e}")
                await asyncio.sleep(5)  # Prevent tight loop on error
    
    async def _update_expert_weights(self):
        """Update expert weights based on recent performance."""
        for expert_id, metrics in self.experts.items():
            # Get current weight or use initial weight
            current_weight = self.expert_weights.get(expert_id, 1.0)
            
            # Adjust based on health score (0-1)
            health_score = metrics.health_score
            
            # Apply adaptive adjustment
            if health_score < 0.3:  # Poor performance
                new_weight = max(0.1, current_weight * 0.8)
            elif health_score > 0.8:  # Excellent performance
                new_weight = min(10.0, current_weight * 1.1)
            else:  # Normal performance
                new_weight = current_weight
            
            # Smooth the transition
            smoothed_weight = 0.7 * current_weight + 0.3 * new_weight
            self.expert_weights[expert_id] = max(0.1, min(10.0, smoothed_weight))
            
            self.logger.debug(
                f"Adjusted weight for {expert_id}: {current_weight:.2f} -> {self.expert_weights[expert_id]:.2f} "
                f"(health: {health_score:.2f})"
            )
    
    @circuit(failure_threshold=5, recovery_timeout=60)
    async def execute_with_expert(
        self,
        expert_id: str,
        request: HuiHuiAnalysisRequestV2_5,
        timeout: float = 30.0
    ) -> HuiHuiAnalysisResponseV2_5:
        """
        Execute a request with the specified expert, with circuit breaking and retries.
        
        Args:
            expert_id: ID of the expert to use
            request: Analysis request
            timeout: Maximum time to wait for a response (seconds)
            
        Returns:
            Analysis response from the expert
            
        Raises:
            CircuitBreakerError: If the circuit is open
            asyncio.TimeoutError: If the request times out
            Exception: For other errors during execution
        """
        start_time = time.monotonic()
        
        try:
            # Check if expert exists and is healthy
            if expert_id not in self.experts:
                raise ValueError(f"Unknown expert: {expert_id}")
                
            # Update load tracking
            self.expert_load[expert_id] += 1
            self.expert_last_used[expert_id] = datetime.utcnow()
            
            # Get the expert instance
            expert = self.experts[expert_id]
            
            # Execute with timeout
            try:
                async with asyncio.timeout(timeout):
                    response = await expert.analyze(request)
                
                # Record success
                duration = (time.monotonic() - start_time) * 1000  # ms
                expert.record_success(duration)
                self.metrics.record_request(expert_id, duration/1000, success=True)
                
                return response
                
            except asyncio.TimeoutError:
                # Record timeout failure
                expert.record_failure(TimeoutError("Request timed out"))
                self.metrics.record_request(expert_id, timeout, success=False)
                raise
                
        except Exception as e:
            # Record other failures
            duration = (time.monotonic() - start_time) * 1000  # ms
            expert.record_failure(e)
            self.metrics.record_request(expert_id, duration/1000, success=False)
            raise
            
        finally:
            # Update load tracking
            if expert_id in self.expert_load:
                self.expert_load[expert_id] = max(0, self.expert_load[expert_id] - 1)
        
        # Core components
        self.expert_registry = get_expert_registry()
        self.communication_protocol = get_communication_protocol()
        self.llm_client = LocalLLMClient()
        
        # Advanced performance tracking
        self.expert_metrics = {}
        self.coordination_history = deque(maxlen=1000)
        self.market_condition_cache = {}
        
        # Expert performance tracking
        self.expert_performance = {}  # expert_name -> ExpertPerformance
        self.feedback_queue = asyncio.Queue()
        self.feedback_processor_task = None
        self.shutdown_event = asyncio.Event()
        
        # AI-powered coordination
        self.coordination_ai_enabled = True
        self.learning_enabled = True
        self.adaptive_weighting = True
        
        # Start background tasks
        self._start_background_tasks()
        
        # Performance statistics
        self.legendary_stats = {
            "total_coordinations": 0,
            "successful_coordinations": 0,
            "ai_decisions": 0,
            "consensus_achieved": 0,
            "conflict_resolutions": 0,
            "performance_improvements": 0,
            "average_coordination_time_ms": 0.0,
            "expert_utilization": defaultdict(int),
            "market_condition_adaptations": 0,
            "learning_iterations": 0
        }
        
        # Expert specialization mapping (enhanced)
        self.expert_specializations = {
            "market_regime": {
                "keywords": ["regime", "volatility", "VRI", "market", "structure", "trend", "macro", "transition"],
                "market_conditions": ["high_volatility", "regime_change", "trending", "consolidation"],
                "strength_multiplier": 1.2,
                "confidence_boost": 0.1
            },
            "options_flow": {
                "keywords": ["options", "flow", "VAPI-FA", "DWFD", "gamma", "institutional", "volume", "SDAG", "DAG"],
                "market_conditions": ["high_volume", "unusual_activity", "gamma_squeeze", "institutional_flow"],
                "strength_multiplier": 1.3,
                "confidence_boost": 0.15
            },
            "sentiment": {
                "keywords": ["sentiment", "news", "psychology", "fear", "greed", "social", "behavioral", "intelligence"],
                "market_conditions": ["news_driven", "sentiment_extreme", "behavioral_bias", "social_momentum"],
                "strength_multiplier": 1.1,
                "confidence_boost": 0.05
            }
        }
        
        # Initialize expert performance tracking
        self._initialize_expert_metrics()
        
        # Initialize performance tracking for registered experts
        for expert_id in self.expert_registry.list_experts():
            if expert_id not in self.expert_performance:
                self.expert_performance[expert_id] = ExpertPerformance(
                    name=expert_id,
                    total_trades=0,
                    successful_trades=0,
                    total_pnl=0.0,
                    recent_pnl=deque(maxlen=100)
                )
        
        self.logger.info("🎯 Legendary Expert Coordinator V3.0 initialized with AI-powered coordination and feedback loop")
    
    def _start_background_tasks(self):
        """Start background tasks for feedback processing."""
        if not self.feedback_processor_task or self.feedback_processor_task.done():
            self.shutdown_event.clear()
            self.feedback_processor_task = asyncio.create_task(self._process_feedback_loop())
            self.logger.info("Started background feedback processor")
    
    async def shutdown(self):
        """Shutdown the coordinator and clean up resources."""
        self.logger.info("Shutting down LegendaryExpertCoordinator...")
        await self._stop_background_tasks()
        
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'feedback_processor_task') and self.feedback_processor_task:
            self.feedback_processor_task.cancel()
    
    async def _stop_background_tasks(self):
        """Stop background tasks gracefully."""
        if self.feedback_processor_task and not self.feedback_processor_task.done():
            self.shutdown_event.set()
            try:
                await asyncio.wait_for(self.feedback_processor_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("Timeout waiting for feedback processor to stop")
                self.feedback_processor_task.cancel()
    
    async def _process_feedback_loop(self):
        """Process feedback in the background."""
        while not self.shutdown_event.is_set():
            try:
                feedback = await asyncio.wait_for(
                    self.feedback_queue.get(),
                    timeout=1.0
                )
                await self._process_feedback(feedback)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing feedback: {e}")
    
    async def _process_feedback(self, feedback: 'TradeFeedback') -> None:
        """Process trade feedback and update expert performance."""
        try:
            expert_id = feedback.expert_id
            if expert_id not in self.expert_performance:
                self.expert_performance[expert_id] = ExpertPerformance(
                    name=expert_id,
                    total_trades=0,
                    successful_trades=0,
                    total_pnl=0.0,
                    recent_pnl=deque(maxlen=100)
                )
            
            expert = self.expert_performance[expert_id]
            is_success = feedback.pnl > 0
            
            # Update metrics
            expert.total_trades += 1
            expert.total_pnl += feedback.pnl
            expert.recent_pnl.append(feedback.pnl)
            
            if is_success:
                expert.successful_trades += 1
            
            # Update weights if adaptive weighting is enabled
            if self.adaptive_weighting:
                await self._update_expert_weights(feedback)
            
            # Log performance
            self.logger.info(
                f"Expert {expert_id} feedback: PnL={feedback.pnl:.2f}, "
                f"Success={is_success}, Win Rate={expert.win_rate:.1%}"
            )
            
            # Update database if available
            if self.db_manager:
                await self._update_expert_performance_db(expert_id, feedback)
                
        except Exception as e:
            self.logger.error(f"Error processing feedback for expert {getattr(feedback, 'expert_id', 'unknown')}: {e}")
    
    async def _update_expert_weights(self, feedback: 'TradeFeedback') -> None:
        """Update expert weights based on feedback."""
        try:
            expert_id = feedback.expert_id
            if expert_id not in self.expert_metrics:
                return
                
            # Calculate weight adjustment based on PnL
            weight_change = self._calculate_weight_adjustment(feedback)
            
            # Update expert weight
            current_weight = self.expert_metrics[expert_id].get('weight', 1.0)
            new_weight = max(0.1, min(10.0, current_weight + weight_change))
            self.expert_metrics[expert_id]['weight'] = new_weight
            
            self.logger.info(
                f"Updated {expert_id} weight: {current_weight:.2f} -> {new_weight:.2f} "
                f"(Δ: {weight_change:+.2f})"
            )
            
            # Log to database if available
            if self.db_manager:
                await self._log_weight_change(expert_id, current_weight, new_weight, feedback)
                
        except Exception as e:
            self.logger.error(f"Error updating expert weights: {e}")
    
    def _calculate_weight_adjustment(self, feedback: 'TradeFeedback') -> float:
        """Calculate weight adjustment based on trade feedback."""
        # Base adjustment is a percentage of PnL, scaled by recent performance
        base_adjustment = feedback.pnl * 0.01  # 1% of PnL
        
        # Adjust based on win rate (experts with higher win rates get more aggressive updates)
        expert = self.expert_performance.get(feedback.expert_id)
        if expert and expert.total_trades > 0:
            win_rate = expert.win_rate
            win_rate_factor = 0.5 + win_rate  # 0.5-1.5x multiplier based on win rate
            base_adjustment *= win_rate_factor
        
        # Cap the adjustment to prevent extreme changes
        return max(-0.5, min(0.5, base_adjustment))
    
    async def _update_expert_performance_db(self, expert_id: str, feedback: 'TradeFeedback') -> None:
        """Update expert performance in the database."""
        try:
            expert = self.expert_performance[expert_id]
            await self.db_manager.upsert(
                "expert_performance",
                {
                    "expert_name": expert_id,
                    "total_trades": expert.total_trades,
                    "successful_trades": expert.successful_trades,
                    "success_rate": expert.success_rate,
                    "total_pnl": float(expert.total_pnl),
                    "avg_pnl": float(expert.avg_pnl),
                    "last_updated": datetime.utcnow().isoformat()
                },
                ["expert_name"]
            )
        except Exception as e:
            self.logger.error(f"Error updating expert performance in DB: {e}")
    
    async def _log_weight_change(self, expert_id: str, old_weight: float, new_weight: float, 
                              feedback: 'TradeFeedback') -> None:
        """Log weight change to database."""
        try:
            await self.db_manager.insert(
                "expert_weight_history",
                {
                    "expert_name": expert_id,
                    "old_weight": float(old_weight),
                    "new_weight": float(new_weight),
                    "change": float(new_weight - old_weight),
                    "trade_id": getattr(feedback, 'trade_id', None),
                    "pnl": float(getattr(feedback, 'pnl', 0.0)),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        except Exception as e:
            self.logger.error(f"Error logging weight change: {e}")
    
    def _initialize_expert_metrics(self):
        """Initialize performance metrics for all experts."""
        for expert_id in self.expert_registry.list_experts():
            self.expert_metrics[expert_id] = {
                'accuracy_score': 0.8,  # Default starting accuracy
                'response_time_ms': 1000.0,  # Default response time
                'weight': 1.0,  # Default weight
                'last_updated': datetime.utcnow()
            }
    
    async def legendary_coordinate_analysis(self, 
                                          request: HuiHuiAnalysisRequestV2_5,
                                          market_context: Optional[MarketConditionContext] = None) -> Dict[str, Any]:
        """
        🚀 LEGENDARY COORDINATION ENGINE
        
        AI-powered coordination with dynamic expert selection, intelligent weighting,
        conflict resolution, and real-time performance optimization.
        """
        start_time = datetime.now()
        self.legendary_stats["total_coordinations"] += 1
        
        try:
            # 1. Analyze market conditions and context
            if not market_context:
                market_context = await self._analyze_market_context(request)
            
            # 2. AI-powered coordination strategy selection
            coordination_strategy = await self._select_coordination_strategy(request, market_context)
            
            # 3. Dynamic expert selection and weighting
            selected_experts = await self._ai_select_experts(request, market_context, coordination_strategy)
            
            if not selected_experts:
                return await self._handle_no_experts_available(request)
            
            # 4. Execute coordinated analysis with intelligent load balancing
            expert_responses = await self._execute_coordinated_analysis(
                request, selected_experts, coordination_strategy
            )
            
            # 5. AI-powered conflict resolution and consensus building
            consensus_result = await self._build_intelligent_consensus(
                expert_responses, coordination_strategy, market_context
            )
            
            # 6. Performance tracking and learning
            coordination_time = (datetime.now() - start_time).total_seconds() * 1000
            await self._update_performance_metrics(expert_responses, consensus_result, coordination_time)
            
            # 7. Generate final coordinated response
            final_response = await self._generate_legendary_response(
                consensus_result, expert_responses, coordination_strategy, market_context
            )
            
            self.legendary_stats["successful_coordinations"] += 1
            return final_response
            
        except Exception as e:
            self.logger.error(f"Legendary coordination failed: {e}")
            return await self._handle_coordination_failure(request, str(e))
    
    async def _analyze_market_context(self, request: HuiHuiAnalysisRequestV2_5) -> MarketConditionContext:
        """Analyze current market conditions for coordination decisions."""
        try:
            # Extract market context from request data
            data_bundle = getattr(request, 'context', None)
            
            # Default context if no data available
            context = MarketConditionContext(
                volatility_regime="normal",
                market_trend="neutral",
                options_flow_intensity="moderate",
                sentiment_regime="neutral",
                time_of_day="market_hours",
                market_stress_level=0.3,
                liquidity_condition="normal",
                news_impact_level=0.2
            )
            
            # Enhanced context analysis if data available
            if data_bundle and hasattr(data_bundle, 'vri_analysis'):
                vri = data_bundle.vri_analysis
                if hasattr(vri, 'current_regime'):
                    context.volatility_regime = vri.current_regime.lower()
                if hasattr(vri, 'stress_level'):
                    context.market_stress_level = min(1.0, max(0.0, vri.stress_level))
            
            # Cache context for performance
            cache_key = f"market_context_{datetime.now().strftime('%Y%m%d_%H%M')}"
            self.market_condition_cache[cache_key] = context
            
            return context
            
        except Exception as e:
            self.logger.warning(f"Market context analysis failed: {e}")
            return MarketConditionContext(
                volatility_regime="unknown",
                market_trend="unknown", 
                options_flow_intensity="unknown",
                sentiment_regime="unknown",
                time_of_day="unknown",
                market_stress_level=0.5,
                liquidity_condition="unknown",
                news_impact_level=0.5
            )
    
    async def _select_coordination_strategy(self, 
                                          request: HuiHuiAnalysisRequestV2_5,
                                          market_context: MarketConditionContext) -> CoordinationStrategy:
        """AI-powered coordination strategy selection."""
        try:
            if not self.coordination_ai_enabled:
                return self._get_default_strategy()
            
            # Prepare AI prompt for strategy selection
            strategy_prompt = f"""
            Analyze the following market conditions and select the optimal coordination strategy:
            
            Market Context:
            - Volatility Regime: {market_context.volatility_regime}
            - Market Trend: {market_context.market_trend}
            - Options Flow: {market_context.options_flow_intensity}
            - Sentiment: {market_context.sentiment_regime}
            - Stress Level: {market_context.market_stress_level:.2f}
            
            Analysis Type: {request.analysis_type}
            
            Available Coordination Modes:
            1. CONSENSUS - All experts must agree (high confidence, slower)
            2. WEIGHTED - Performance-weighted decisions (balanced)
            3. COMPETITIVE - Best expert wins (fast, specialized)
            4. COLLABORATIVE - Experts work together (comprehensive)
            5. EMERGENCY - Crisis mode (fastest response)
            6. ADAPTIVE - AI-selected optimal mode
            
            Select the best coordination mode and provide expert weights (0.0-1.0).
            Respond in JSON format:
            {
                "mode": "selected_mode",
                "expert_weights": {"market_regime": 0.0, "options_flow": 0.0, "sentiment": 0.0},
                "reasoning": "explanation"
            }
            """
            
            # Get AI recommendation
            messages = [ChatMessage(role="user", content=strategy_prompt)]
            ai_response = self.llm_client._make_request(ModelType.HUIHUI_MOE, messages)
            
            # Parse AI response
            import json
            try:
                ai_strategy = json.loads(ai_response)
                mode = CoordinationMode(ai_strategy.get("mode", "weighted"))
                weights = ai_strategy.get("expert_weights", {})
                
                self.legendary_stats["ai_decisions"] += 1
                
                return CoordinationStrategy(
                    mode=mode,
                    expert_weights=weights,
                    timeout_seconds=self._calculate_timeout(mode, market_context),
                    consensus_threshold=self._calculate_consensus_threshold(mode, market_context),
                    confidence_threshold=0.6,
                    parallel_execution=mode != CoordinationMode.EMERGENCY
                )
                
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"AI strategy parsing failed: {e}")
                return self._get_adaptive_strategy(market_context)
                
        except Exception as e:
            self.logger.warning(f"AI strategy selection failed: {e}")
            return self._get_default_strategy()
    
    def _get_default_strategy(self) -> CoordinationStrategy:
        """Get default coordination strategy."""
        return CoordinationStrategy(
            mode=CoordinationMode.WEIGHTED,
            expert_weights={"market_regime": 0.33, "options_flow": 0.33, "sentiment": 0.34},
            timeout_seconds=30.0,
            consensus_threshold=0.7,
            confidence_threshold=0.6,
            parallel_execution=True
        )
    
    def _get_adaptive_strategy(self, market_context: MarketConditionContext) -> CoordinationStrategy:
        """Get adaptive strategy based on market conditions."""
        # High stress = emergency mode
        if market_context.market_stress_level > 0.8:
            return CoordinationStrategy(
                mode=CoordinationMode.EMERGENCY,
                expert_weights={"market_regime": 0.5, "options_flow": 0.3, "sentiment": 0.2},
                timeout_seconds=10.0,
                consensus_threshold=0.5,
                parallel_execution=False
            )
        
        # High volatility = competitive mode
        elif market_context.volatility_regime in ["high", "extreme"]:
            return CoordinationStrategy(
                mode=CoordinationMode.COMPETITIVE,
                expert_weights={"market_regime": 0.6, "options_flow": 0.3, "sentiment": 0.1},
                timeout_seconds=20.0,
                consensus_threshold=0.6
            )
        
        # Default to collaborative
        else:
            return CoordinationStrategy(
                mode=CoordinationMode.COLLABORATIVE,
                expert_weights={"market_regime": 0.35, "options_flow": 0.35, "sentiment": 0.3},
                timeout_seconds=30.0,
                consensus_threshold=0.7
            )
    
    def _calculate_timeout(self, mode: CoordinationMode, market_context: MarketConditionContext) -> float:
        """Calculate optimal timeout based on mode and market conditions."""
        base_timeouts = {
            CoordinationMode.EMERGENCY: 5.0,
            CoordinationMode.COMPETITIVE: 15.0,
            CoordinationMode.WEIGHTED: 25.0,
            CoordinationMode.COLLABORATIVE: 35.0,
            CoordinationMode.CONSENSUS: 45.0,
            CoordinationMode.ADAPTIVE: 30.0
        }
        
        base_timeout = base_timeouts.get(mode, 30.0)
        
        # Adjust for market stress
        stress_multiplier = 1.0 - (market_context.market_stress_level * 0.5)
        
        return max(5.0, base_timeout * stress_multiplier)
    
    def _calculate_consensus_threshold(self, mode: CoordinationMode, market_context: MarketConditionContext) -> float:
        """Calculate consensus threshold based on mode and conditions."""
        base_thresholds = {
            CoordinationMode.EMERGENCY: 0.5,
            CoordinationMode.COMPETITIVE: 0.6,
            CoordinationMode.WEIGHTED: 0.7,
            CoordinationMode.COLLABORATIVE: 0.75,
            CoordinationMode.CONSENSUS: 0.9,
            CoordinationMode.ADAPTIVE: 0.7
        }
        
        return base_thresholds.get(mode, 0.7)
    
    async def _ai_select_experts(self, 
                               request: HuiHuiAnalysisRequestV2_5,
                               market_context: MarketConditionContext,
                               strategy: CoordinationStrategy) -> List[str]:
        """AI-powered expert selection with dynamic weighting."""
        available_experts = list(self.expert_registry.get_all_experts().keys())
        
        # Filter based on expert availability and performance
        viable_experts = []
        for expert_id in available_experts:
            expert = self.expert_registry.get_expert(expert_id)
            if expert and expert.is_initialized:
                metrics = self.expert_metrics.get(expert_id)
                if metrics and metrics.recent_success_rate > 0.3:  # Minimum performance threshold
                    viable_experts.append(expert_id)
        
        if not viable_experts:
            self.logger.warning("No viable experts available")
            return []
        
        # Apply specialization-based selection
        request_text = f"{request.analysis_type} {getattr(request, 'data_types', [])}".lower()
        
        selected_experts = []
        for expert_id in viable_experts:
            if expert_id in self.expert_specializations:
                spec = self.expert_specializations[expert_id]
                
                # Check keyword relevance
                keyword_match = any(keyword in request_text for keyword in spec["keywords"])
                
                # Check market condition relevance
                condition_match = any(
                    condition in [market_context.volatility_regime, market_context.market_trend,
                                market_context.options_flow_intensity, market_context.sentiment_regime]
                    for condition in spec["market_conditions"]
                )
                
                # Include expert if relevant or if using consensus/collaborative mode
                if keyword_match or condition_match or strategy.mode in [CoordinationMode.CONSENSUS, CoordinationMode.COLLABORATIVE]:
                    selected_experts.append(expert_id)
        
        # Ensure at least one expert is selected
        if not selected_experts and viable_experts:
            selected_experts = viable_experts[:1]
        
        self.logger.debug(f"Selected experts: {selected_experts} for strategy: {strategy.mode}")
        return selected_experts
    
    async def _execute_coordinated_analysis(self,
                                          request: HuiHuiAnalysisRequestV2_5,
                                          selected_experts: List[str],
                                          strategy: CoordinationStrategy) -> Dict[str, Any]:
        """Execute analysis across selected experts with intelligent coordination."""
        expert_responses = {}
        
        if strategy.parallel_execution:
            # Parallel execution for speed
            tasks = {}
            for expert_id in selected_experts:
                expert = self.expert_registry.get_expert(expert_id)
                if expert and expert.is_initialized:
                    task = asyncio.create_task(expert.analyze(request))
                    tasks[expert_id] = task
            
            # Wait for responses with timeout
            for expert_id, task in tasks.items():
                try:
                    response = await asyncio.wait_for(task, timeout=strategy.timeout_seconds)
                    expert_responses[expert_id] = response
                    self._update_expert_utilization(expert_id)
                except asyncio.TimeoutError:
                    self.logger.warning(f"Expert {expert_id} timed out")
                    expert_responses[expert_id] = {"status": "timeout", "error": "Analysis timed out"}
                except Exception as e:
                    self.logger.error(f"Expert {expert_id} failed: {e}")
                    expert_responses[expert_id] = {"status": "error", "error": str(e)}
        
        else:
            # Sequential execution for emergency mode
            for expert_id in selected_experts:
                try:
                    expert = self.expert_registry.get_expert(expert_id)
                    if expert and expert.is_initialized:
                        response = await asyncio.wait_for(
                            expert.analyze(request), 
                            timeout=strategy.timeout_seconds / len(selected_experts)
                        )
                        expert_responses[expert_id] = response
                        self._update_expert_utilization(expert_id)
                        
                        # In emergency mode, stop after first successful response
                        if strategy.mode == CoordinationMode.EMERGENCY and response.get("status") != "error":
                            break
                            
                except Exception as e:
                    self.logger.error(f"Expert {expert_id} failed in sequential mode: {e}")
                    expert_responses[expert_id] = {"status": "error", "error": str(e)}
        
        return expert_responses
    
    async def _build_intelligent_consensus(self,
                                         expert_responses: Dict[str, Any],
                                         strategy: CoordinationStrategy,
                                         market_context: MarketConditionContext) -> Dict[str, Any]:
        """Build intelligent consensus with AI-powered conflict resolution."""
        successful_responses = {
            expert_id: response for expert_id, response in expert_responses.items()
            if response.get("status") not in ["error", "timeout"]
        }
        
        if not successful_responses:
            return {"status": "no_successful_responses", "consensus": None}
        
        consensus_result = {
            "status": "success",
            "mode": strategy.mode.value,
            "expert_count": len(successful_responses),
            "consensus_confidence": 0.0,
            "primary_recommendation": None,
            "confidence_scores": {},
            "expert_weights_used": strategy.expert_weights,
            "conflict_resolution": None
        }
        
        # Extract confidence scores and recommendations
        recommendations = []
        confidence_scores = {}
        
        for expert_id, response in successful_responses.items():
            if isinstance(response, dict):
                confidence = response.get("confidence_score", 0.5)
                recommendation = response.get("recommendation", "neutral")
            else:
                confidence = getattr(response, "confidence_score", 0.5)
                recommendation = getattr(response, "recommendation", "neutral")
            
            confidence_scores[expert_id] = confidence
            recommendations.append((expert_id, recommendation, confidence))
        
        consensus_result["confidence_scores"] = confidence_scores
        
        # Apply coordination strategy
        if strategy.mode == CoordinationMode.CONSENSUS:
            consensus_result = await self._build_consensus_agreement(
                recommendations, strategy, consensus_result
            )
        elif strategy.mode == CoordinationMode.WEIGHTED:
            consensus_result = await self._build_weighted_consensus(
                recommendations, strategy, consensus_result
            )
        elif strategy.mode == CoordinationMode.COMPETITIVE:
            consensus_result = await self._build_competitive_consensus(
                recommendations, strategy, consensus_result
            )
        elif strategy.mode == CoordinationMode.COLLABORATIVE:
            consensus_result = await self._build_collaborative_consensus(
                recommendations, strategy, consensus_result, market_context
            )
        elif strategy.mode == CoordinationMode.EMERGENCY:
            consensus_result = await self._build_emergency_consensus(
                recommendations, strategy, consensus_result
            )
        else:  # ADAPTIVE
            consensus_result = await self._build_adaptive_consensus(
                recommendations, strategy, consensus_result, market_context
            )
        
        return consensus_result
    
    async def _build_consensus_agreement(self, recommendations: List[Tuple], 
                                       strategy: CoordinationStrategy, 
                                       result: Dict[str, Any]) -> Dict[str, Any]:
        """Build consensus requiring agreement between experts."""
        from collections import Counter
        
        rec_only = [rec[1] for rec in recommendations]
        rec_counts = Counter(rec_only)
        
        if rec_counts:
            most_common = rec_counts.most_common(1)[0]
            agreement_level = most_common[1] / len(recommendations)
            
            if agreement_level >= strategy.consensus_threshold:
                result["primary_recommendation"] = most_common[0]
                result["consensus_confidence"] = agreement_level
                result["status"] = "consensus_achieved"
                self.legendary_stats["consensus_achieved"] += 1
            else:
                result["status"] = "consensus_failed"
                result["primary_recommendation"] = most_common[0]
                result["consensus_confidence"] = agreement_level
                result["conflict_resolution"] = "insufficient_agreement"
        
        return result
    
    async def _build_weighted_consensus(self, recommendations: List[Tuple],
                                      strategy: CoordinationStrategy,
                                      result: Dict[str, Any]) -> Dict[str, Any]:
        """Build weighted consensus based on expert performance and confidence."""
        weighted_scores = defaultdict(float)
        total_weight = 0.0
        
        for expert_id, recommendation, confidence in recommendations:
            # Get expert weight from strategy
            expert_weight = strategy.expert_weights.get(expert_id, 1.0)
            
            # Get performance-based weight
            metrics = self.expert_metrics.get(expert_id)
            performance_weight = metrics.accuracy_score if metrics else 0.5
            
            # Combined weight
            combined_weight = expert_weight * performance_weight * confidence
            weighted_scores[recommendation] += combined_weight
            total_weight += combined_weight
        
        if weighted_scores and total_weight > 0:
            # Normalize scores
            for rec in weighted_scores:
                weighted_scores[rec] /= total_weight
            
            # Select highest weighted recommendation
            best_rec = max(weighted_scores.items(), key=lambda x: x[1])
            result["primary_recommendation"] = best_rec[0]
            result["consensus_confidence"] = best_rec[1]
            result["status"] = "weighted_consensus"
        
        return result
    
    async def _build_competitive_consensus(self, recommendations: List[Tuple],
                                         strategy: CoordinationStrategy,
                                         result: Dict[str, Any]) -> Dict[str, Any]:
        """Build competitive consensus - best expert wins."""
        if not recommendations:
            return result
        
        # Find expert with highest combined score (performance * confidence)
        best_expert = None
        best_score = 0.0
        
        for expert_id, recommendation, confidence in recommendations:
            metrics = self.expert_metrics.get(expert_id)
            performance = metrics.accuracy_score if metrics else 0.5
            combined_score = performance * confidence
            
            if combined_score > best_score:
                best_score = combined_score
                best_expert = (expert_id, recommendation, confidence)
        
        if best_expert:
            result["primary_recommendation"] = best_expert[1]
            result["consensus_confidence"] = best_expert[2]
            result["winning_expert"] = best_expert[0]
            result["status"] = "competitive_winner"
        
        return result
    
    async def _build_collaborative_consensus(self, recommendations: List[Tuple],
                                           strategy: CoordinationStrategy,
                                           result: Dict[str, Any],
                                           market_context: MarketConditionContext) -> Dict[str, Any]:
        """Build collaborative consensus with AI-powered synthesis."""
        try:
            if not self.coordination_ai_enabled:
                return await self._build_weighted_consensus(recommendations, strategy, result)
            
            # Prepare collaborative analysis prompt
            collab_prompt = f"""
            Synthesize the following expert recommendations into a unified trading decision:
            
            Market Context:
            - Volatility: {market_context.volatility_regime}
            - Trend: {market_context.market_trend}
            - Stress Level: {market_context.market_stress_level:.2f}
            
            Expert Recommendations:
            """
            
            for expert_id, recommendation, confidence in recommendations:
                collab_prompt += f"- {expert_id}: {recommendation} (confidence: {confidence:.2f})\n"
            
            collab_prompt += """
            
            Provide a synthesized recommendation that considers all expert inputs.
            Respond in JSON format:
            {
                "recommendation": "synthesized_recommendation",
                "confidence": 0.0-1.0,
                "reasoning": "explanation"
            }
            """
            
            # Get AI synthesis
            messages = [ChatMessage(role="user", content=collab_prompt)]
            ai_response = self.llm_client._make_request(ModelType.HUIHUI_MOE, messages)
            
            import json
            ai_synthesis = json.loads(ai_response)
            
            result["primary_recommendation"] = ai_synthesis.get("recommendation", "neutral")
            result["consensus_confidence"] = ai_synthesis.get("confidence", 0.5)
            result["ai_synthesis"] = ai_synthesis.get("reasoning", "")
            result["status"] = "collaborative_synthesis"
            
        except Exception as e:
            self.logger.warning(f"Collaborative synthesis failed: {e}")
            return await self._build_weighted_consensus(recommendations, strategy, result)
        
        return result
    
    async def _build_emergency_consensus(self, recommendations: List[Tuple],
                                       strategy: CoordinationStrategy,
                                       result: Dict[str, Any]) -> Dict[str, Any]:
        """Build emergency consensus - fastest reliable response."""
        if not recommendations:
            return result
        
        # In emergency mode, take first high-confidence recommendation
        for expert_id, recommendation, confidence in recommendations:
            if confidence >= 0.6:  # Minimum confidence for emergency
                result["primary_recommendation"] = recommendation
                result["consensus_confidence"] = confidence
                result["emergency_expert"] = expert_id
                result["status"] = "emergency_response"
                return result
        
        # If no high-confidence, take best available
        if recommendations:
            best = max(recommendations, key=lambda x: x[2])
            result["primary_recommendation"] = best[1]
            result["consensus_confidence"] = best[2]
            result["emergency_expert"] = best[0]
            result["status"] = "emergency_fallback"
        
        return result
    
    async def _build_adaptive_consensus(self, recommendations: List[Tuple],
                                      strategy: CoordinationStrategy,
                                      result: Dict[str, Any],
                                      market_context: MarketConditionContext) -> Dict[str, Any]:
        """Build adaptive consensus based on current conditions."""
        # Choose best consensus method based on conditions
        if market_context.market_stress_level > 0.8:
            return await self._build_emergency_consensus(recommendations, strategy, result)
        elif len(recommendations) >= 3:
            return await self._build_collaborative_consensus(recommendations, strategy, result, market_context)
        else:
            return await self._build_weighted_consensus(recommendations, strategy, result)
    
    async def _generate_legendary_response(self,
                                         consensus_result: Dict[str, Any],
                                         expert_responses: Dict[str, Any],
                                         strategy: CoordinationStrategy,
                                         market_context: MarketConditionContext) -> Dict[str, Any]:
        """Generate the final legendary coordinated response."""
        legendary_response = {
            "status": "legendary_coordination_complete",
            "coordination_timestamp": datetime.now(),
            "coordination_mode": strategy.mode.value,
            "market_context": market_context.model_dump(),
            "expert_count": len(expert_responses),
            "successful_experts": len([r for r in expert_responses.values() if r.get("status") not in ["error", "timeout"]]),
            "consensus_result": consensus_result,
            "expert_responses": expert_responses,
            "coordination_quality": self._calculate_coordination_quality(consensus_result, expert_responses),
            "performance_metrics": self._get_current_performance_summary(),
            "recommendations": {
                "primary": consensus_result.get("primary_recommendation", "neutral"),
                "confidence": consensus_result.get("consensus_confidence", 0.5),
                "risk_level": self._assess_risk_level(consensus_result, market_context),
                "action_urgency": self._assess_action_urgency(consensus_result, market_context)
            },
            "legendary_insights": await self._generate_legendary_insights(consensus_result, market_context)
        }
        
        return legendary_response
    
    def _calculate_coordination_quality(self, consensus_result: Dict[str, Any], 
                                      expert_responses: Dict[str, Any]) -> float:
        """Calculate the quality of coordination (0.0 to 1.0)."""
        quality_factors = []
        
        # Response success rate
        successful = len([r for r in expert_responses.values() if r.get("status") not in ["error", "timeout"]])
        total = len(expert_responses)
        success_rate = successful / total if total > 0 else 0.0
        quality_factors.append(success_rate)
        
        # Consensus confidence
        consensus_confidence = consensus_result.get("consensus_confidence", 0.0)
        quality_factors.append(consensus_confidence)
        
        # Expert agreement (if available)
        if "agreement_level" in consensus_result:
            quality_factors.append(consensus_result["agreement_level"])
        
        # Average confidence scores
        confidence_scores = consensus_result.get("confidence_scores", {})
        if confidence_scores:
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
            quality_factors.append(avg_confidence)
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5
    
    def _assess_risk_level(self, consensus_result: Dict[str, Any], 
                          market_context: MarketConditionContext) -> str:
        """Assess risk level based on consensus and market conditions."""
        base_risk = 0.3
        
        # Market stress contribution
        stress_risk = market_context.market_stress_level * 0.4
        
        # Consensus confidence contribution (inverse)
        consensus_confidence = consensus_result.get("consensus_confidence", 0.5)
        confidence_risk = (1.0 - consensus_confidence) * 0.3
        
        total_risk = base_risk + stress_risk + confidence_risk
        
        if total_risk < 0.3:
            return "low"
        elif total_risk < 0.6:
            return "moderate"
        elif total_risk < 0.8:
            return "high"
        else:
            return "extreme"
    
    def _assess_action_urgency(self, consensus_result: Dict[str, Any],
                              market_context: MarketConditionContext) -> str:
        """Assess urgency of action based on consensus and conditions."""
        urgency_score = 0.0
        
        # Market stress urgency
        urgency_score += market_context.market_stress_level * 0.4
        
        # Consensus strength urgency
        consensus_confidence = consensus_result.get("consensus_confidence", 0.5)
        if consensus_confidence > 0.8:
            urgency_score += 0.3
        
        # News impact urgency
        urgency_score += market_context.news_impact_level * 0.3
        
        if urgency_score < 0.3:
            return "low"
        elif urgency_score < 0.6:
            return "moderate"
        elif urgency_score < 0.8:
            return "high"
        else:
            return "immediate"
    
    async def _generate_legendary_insights(self, consensus_result: Dict[str, Any],
                                         market_context: MarketConditionContext) -> Dict[str, Any]:
        """Generate AI-powered legendary insights."""
        try:
            if not self.coordination_ai_enabled:
                return {"status": "ai_disabled", "insights": []}
            
            insights_prompt = f"""
            Generate strategic trading insights based on the coordination results:
            
            Consensus: {consensus_result.get('primary_recommendation', 'neutral')}
            Confidence: {consensus_result.get('consensus_confidence', 0.5):.2f}
            Market Stress: {market_context.market_stress_level:.2f}
            Volatility Regime: {market_context.volatility_regime}
            
            Provide 3-5 actionable insights for options trading strategy.
            """
            
            messages = [ChatMessage(role="user", content=insights_prompt)]
            ai_insights = self.llm_client._make_request(ModelType.HUIHUI_MOE, messages)
            
            return {
                "status": "ai_generated",
                "insights": ai_insights.split('\n') if ai_insights else [],
                "generation_timestamp": datetime.now()
            }
            
        except Exception as e:
            self.logger.warning(f"Legendary insights generation failed: {e}")
            return {"status": "generation_failed", "error": str(e)}
    
    async def _update_performance_metrics(self, expert_responses: Dict[str, Any],
                                        consensus_result: Dict[str, Any],
                                        coordination_time_ms: float):
        """Update performance metrics and learning systems."""
        # Update coordination stats
        self._update_coordination_stats(coordination_time_ms, len(expert_responses) > 0)
        
        # Update expert performance metrics
        for expert_id, response in expert_responses.items():
            if expert_id in self.expert_metrics:
                metrics = self.expert_metrics[expert_id]
                
                # Update response time
                if isinstance(response, dict) and "response_time_ms" in response:
                    old_time = metrics.response_time_ms
                    new_time = response["response_time_ms"]
                    metrics.response_time_ms = (old_time * 0.9) + (new_time * 0.1)  # Exponential moving average
                
                # Update success rate
                success = response.get("status") not in ["error", "timeout"]
                old_rate = metrics.recent_success_rate
                metrics.recent_success_rate = (old_rate * 0.95) + (0.05 if success else 0.0)
        
        # Store coordination history for learning
        coordination_record = {
            "timestamp": datetime.now(),
            "expert_responses": expert_responses,
            "consensus_result": consensus_result,
            "coordination_time_ms": coordination_time_ms,
            "quality_score": self._calculate_coordination_quality(consensus_result, expert_responses)
        }
        
        self.coordination_history.append(coordination_record)
        
        # Trigger learning if enabled
        if self.learning_enabled and len(self.coordination_history) % 10 == 0:
            await self._trigger_learning_cycle()
    
    async def _trigger_learning_cycle(self):
        """Trigger learning cycle to improve coordination."""
        try:
            self.legendary_stats["learning_iterations"] += 1
            
            # Analyze recent coordination performance
            recent_coords = list(self.coordination_history)[-10:]
            avg_quality = sum(c["quality_score"] for c in recent_coords) / len(recent_coords)
            
            # Adjust expert weights based on performance
            if self.adaptive_weighting:
                for expert_id in self.expert_metrics:
                    metrics = self.expert_metrics[expert_id]
                    
                    # Increase learning rate if performance is improving
                    if metrics.recent_success_rate > 0.8:
                        metrics.learning_rate = min(0.2, metrics.learning_rate * 1.1)
                    else:
                        metrics.learning_rate = max(0.05, metrics.learning_rate * 0.9)
            
            self.logger.debug(f"Learning cycle completed. Average quality: {avg_quality:.3f}")
            
        except Exception as e:
            self.logger.warning(f"Learning cycle failed: {e}")
    
    def _update_expert_utilization(self, expert_id: str):
        """Update expert utilization statistics."""
        self.legendary_stats["expert_utilization"][expert_id] += 1
    
    def _update_coordination_stats(self, coordination_time_ms: float, success: bool):
        """Update coordination performance statistics."""
        # Update average coordination time
        current_avg = self.legendary_stats["average_coordination_time_ms"]
        total_coords = self.legendary_stats["total_coordinations"]
        
        self.legendary_stats["average_coordination_time_ms"] = (
            (current_avg * (total_coords - 1) + coordination_time_ms) / total_coords
        )
    
    def _get_current_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        total_coords = self.legendary_stats["total_coordinations"]
        successful_coords = self.legendary_stats["successful_coordinations"]
        
        return {
            "success_rate": successful_coords / total_coords if total_coords > 0 else 0.0,
            "average_coordination_time_ms": self.legendary_stats["average_coordination_time_ms"],
            "total_coordinations": total_coords,
            "ai_decisions": self.legendary_stats["ai_decisions"],
            "consensus_achieved": self.legendary_stats["consensus_achieved"],
            "expert_utilization": dict(self.legendary_stats["expert_utilization"])
        }
    
    async def _handle_no_experts_available(self, request: HuiHuiAnalysisRequestV2_5) -> Dict[str, Any]:
        """Handle case when no experts are available."""
        return {
            "status": "no_experts_available",
            "error": "No viable experts available for analysis",
            "fallback_recommendation": "neutral",
            "confidence": 0.1,
            "timestamp": datetime.now()
        }
    
    async def _handle_coordination_failure(self, request: HuiHuiAnalysisRequestV2_5, error: str) -> Dict[str, Any]:
        """Handle coordination failure with graceful degradation."""
        return {
            "status": "coordination_failed",
            "error": error,
            "fallback_recommendation": "neutral",
            "confidence": 0.0,
            "timestamp": datetime.now(),
            "recovery_suggestions": [
                "Check expert availability",
                "Verify data bundle integrity", 
                "Review system logs",
                "Consider manual analysis"
            ]
        }
    
    # Legacy compatibility methods
    async def coordinate_analysis(self, request: HuiHuiAnalysisRequestV2_5, 
                                criteria: Optional[Any] = None) -> Dict[str, Any]:
        """Legacy compatibility method."""
        return await self.legendary_coordinate_analysis(request)
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination status."""
        available_experts = self.expert_registry.get_expert_status()
        
        return {
            "coordinator_status": "legendary_active",
            "version": "3.0",
            "ai_enabled": self.coordination_ai_enabled,
            "learning_enabled": self.learning_enabled,
            "available_experts": available_experts,
            "legendary_stats": self.legendary_stats.copy(),
            "expert_metrics": {eid: metrics.model_dump() for eid, metrics in self.expert_metrics.items()},
            "coordination_history_size": len(self.coordination_history),
            "market_context_cache_size": len(self.market_condition_cache)
        }
    
    async def legendary_health_check(self) -> Dict[str, Any]:
        """Comprehensive legendary health check."""
        health_status = {
            "coordinator": "legendary_healthy",
            "version": "3.0",
            "ai_systems": {
                "coordination_ai": "active" if self.coordination_ai_enabled else "disabled",
                "learning_system": "active" if self.learning_enabled else "disabled",
                "local_llm": "unknown"
            },
            "experts": {},
            "performance": self._get_current_performance_summary(),
            "system_load": {
                "coordination_history": len(self.coordination_history),
                "market_context_cache": len(self.market_condition_cache),
                "expert_metrics_tracked": len(self.expert_metrics)
            }
        }
        
        # Test local LLM connection
        try:
            test_messages = [ChatMessage(role="user", content="Health check")]
            self.llm_client._make_request(ModelType.HUIHUI_MOE, test_messages)
            health_status["ai_systems"]["local_llm"] = "healthy"
        except Exception as e:
            health_status["ai_systems"]["local_llm"] = f"error: {str(e)}"
        
        # Check each expert
        for expert_id, expert in self.expert_registry.get_all_experts().items():
            try:
                expert_health = {
                    "status": "healthy" if expert.is_initialized else "not_initialized",
                    "last_analysis": getattr(expert, 'last_analysis_time', None),
                    "performance_metrics": self.expert_metrics.get(expert_id, {}).model_dump() if expert_id in self.expert_metrics else {}
                }
                health_status["experts"][expert_id] = expert_health
            except Exception as e:
                health_status["experts"][expert_id] = {"status": "error", "error": str(e)}
        
        return health_status

# Global legendary coordinator instance
_legendary_coordinator = None

def get_legendary_coordinator() -> LegendaryExpertCoordinator:
    """Get the global legendary coordinator instance."""
    global _legendary_coordinator
    if _legendary_coordinator is None:
        _legendary_coordinator = LegendaryExpertCoordinator()
    return _legendary_coordinator

async def initialize_legendary_coordinator(db_manager=None) -> LegendaryExpertCoordinator:
    """Initialize the legendary expert coordinator."""
    global _legendary_coordinator
    _legendary_coordinator = LegendaryExpertCoordinator(db_manager)
    return _legendary_coordinator

# Dataclass for expert performance tracking
@dataclass
class ExpertPerformance:
    name: str
    total_trades: int = 0
    successful_trades: int = 0
    total_pnl: float = 0.0
    recent_pnl: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def win_rate(self) -> float:
        """Calculate the win rate of the expert."""
        if self.total_trades == 0:
            return 0.0
        return self.successful_trades / self.total_trades
    
    @property
    def success_rate(self) -> float:
        """Alias for win_rate for backward compatibility."""
        return self.win_rate
    
    @property
    def avg_pnl(self) -> float:
        """Calculate the average PnL per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

# Trade feedback data class
@dataclass
class TradeFeedback:
    expert_id: str
    trade_id: str
    pnl: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

# Backward compatibility
ExpertCoordinator = LegendaryExpertCoordinator
get_coordinator = get_legendary_coordinator
initialize_coordinator = initialize_legendary_coordinator

