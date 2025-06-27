"""
🚀 EXPERT ROUTER - Consolidated AI Routing Engine
==============================================

UNIFIED ROUTING SYSTEM FOR EOTS v2.5

🔹 Combines vectorized AI routing with adaptive expert selection
🔹 Real-time performance tracking and dynamic weight adjustment
🔹 Ultra-fast vector search with local embeddings
🔹 Seamless integration with Ollama LLM backend

PERFORMANCE FEATURES:
- 🚀 Sub-millisecond routing decisions
- ⚡ Async/await with connection pooling
- 🧠 Vector-based expert matching
- 📈 Continuous performance optimization
"""

import asyncio
import time
import logging
import json
import re
import numpy as np
import aiohttp
import socket
import weakref
import threading
from typing import Dict, List, Optional, Tuple, Any, Set, Deque, AsyncGenerator, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

# Enhanced Cache Manager integration
try:
    from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5
    ENHANCED_CACHE_AVAILABLE = True
except ImportError:
    ENHANCED_CACHE_AVAILABLE = False
    EnhancedCacheManagerV2_5 = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_WEIGHT = 1.0
MIN_WEIGHT = 0.1
MAX_WEIGHT = 10.0
WEIGHT_ADJUSTMENT_STEP = 0.1
CONFIDENCE_THRESHOLD = 0.7
SLIDING_WINDOW_SIZE = 100
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MAX_CONNECTIONS = 20

class HuiHuiExpertType(Enum):
    """🧠 HuiHui Expert Types - The Core Intelligence Pillars"""
    MARKET_REGIME = "market_regime"
    OPTIONS_FLOW = "options_flow"
    SENTIMENT = "sentiment"
    ORCHESTRATOR = "orchestrator"

@dataclass
class PerformanceMetrics:
    """📊 Tracks performance metrics for the router"""
    total_requests: int = 0
    total_time: float = 0.0
    avg_response_time: float = 0.0
    fastest_response: float = float('inf')
    slowest_response: float = 0.0
    vector_routing_used: int = 0
    fallback_routing_used: int = 0
    vector_accuracy: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    connection_reuses: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def update(self, response_time: float, used_vector: bool = False, 
               cache_hit: bool = False, connection_reused: bool = False):
        """Update metrics with new data point"""
        self.total_requests += 1
        self.total_time += response_time
        self.avg_response_time = self.total_time / self.total_requests
        self.fastest_response = min(self.fastest_response, response_time)
        self.slowest_response = max(self.slowest_response, response_time)
        
        if used_vector:
            self.vector_routing_used += 1
        else:
            self.fallback_routing_used += 1
            
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
        if connection_reused:
            self.connection_reuses += 1

@dataclass
class ExpertPerformance:
    """📊 Tracks performance metrics for each expert"""
    name: str
    total_queries: int = 0
    successful_responses: int = 0
    total_response_time: float = 0.0
    recent_response_times: Deque[float] = field(default_factory=lambda: deque(maxlen=SLIDING_WINDOW_SIZE))
    error_count: int = 0
    last_used: Optional[datetime] = None
    weight: float = DEFAULT_WEIGHT
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the expert"""
        if self.total_queries == 0:
            return 0.0
        return self.successful_responses / self.total_queries
    
    @property
    def avg_response_time(self) -> float:
        """Calculate the average response time"""
        if not self.recent_response_times:
            return 0.0
        return sum(self.recent_response_times) / len(self.recent_response_times)
    
    @property
    def score(self) -> float:
        """Calculate a composite performance score"""
        success_score = self.success_rate * 0.6
        speed_score = (1.0 / (1.0 + self.avg_response_time)) * 0.4 if self.avg_response_time > 0 else 0.0
        return (success_score + speed_score) * self.weight

@dataclass
class RoutingDecision:
    """Represents a routing decision with confidence and metadata"""
    expert_type: HuiHuiExpertType
    confidence: float
    alternatives: List[Tuple[HuiHuiExpertType, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "expert_type": self.expert_type.value,
            "confidence": self.confidence,
            "alternatives": [(e.value, c) for e, c in self.alternatives],
            "metadata": self.metadata
        }

class UltraFastEmbeddingCache:
    """🚀 Ultra-fast embedding cache with async operations and intelligent TTL"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._timestamps = {}
        self._lock = threading.RLock()
    
    async def get_async(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache asynchronously"""
        return await asyncio.get_event_loop().run_in_executor(None, self._get_sync, key)
    
    def _get_sync(self, key: str) -> Optional[np.ndarray]:
        """Synchronous cache get with TTL check"""
        with self._lock:
            if key not in self._cache:
                return None
                
            # Check TTL
            if time.time() - self._timestamps[key] > self.ttl_seconds:
                del self._cache[key]
                del self._timestamps[key]
                return None
                
            # Update timestamp
            self._timestamps[key] = time.time()
            return self._cache[key]
    
    async def set_async(self, key: str, value: np.ndarray) -> None:
        """Set embedding in cache asynchronously"""
        await asyncio.get_event_loop().run_in_executor(None, self._set_sync, key, value)
    
    def _set_sync(self, key: str, value: np.ndarray) -> None:
        """Synchronous cache set with LRU eviction"""
        with self._lock:
            # Evict if needed (LRU)
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Find and remove oldest entry
                oldest_key = min(self._timestamps.keys(), key=lambda k: self._timestamps[k])
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            
            # Add/update entry
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hit_rate": 0.0,  # Tracked by PerformanceMetrics
            }

class ExpertRouter:
    """
    🧠 Consolidated Expert Router with Vectorized AI and Adaptive Learning
    
    Combines the best of both worlds:
    - Ultra-fast vectorized routing from VectorizedAIRouter
    - Adaptive learning and performance tracking from ExpertRouter
    - Seamless integration with Ollama LLM backend
    """
    
    def __init__(self, ollama_host: str = DEFAULT_OLLAMA_HOST, 
                 max_connections: int = DEFAULT_MAX_CONNECTIONS):
        """Initialize the Expert Router"""
        self.ollama_host = ollama_host
        self.max_connections = max_connections
        
        # Connection management
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=max_connections,
            keepalive_timeout=120,
            enable_cleanup_closed=True,
            use_dns_cache=True,
            ttl_dns_cache=300,
            family=socket.AF_UNSPEC,
            ssl=False
        )
        self.session = None
        
        # Performance tracking
        self.performance = {expert_type: ExpertPerformance(name=expert_type.value) 
                          for expert_type in HuiHuiExpertType}
        self.performance_metrics = PerformanceMetrics()
        self.query_history = deque(maxlen=1000)
        self.last_weight_update = datetime.now()
        
        # Vector search components
        self.embedding_model = None
        self.expert_embeddings = None
        self.vector_search_enabled = False
        self.embedding_cache = UltraFastEmbeddingCache(max_size=2000, ttl_seconds=1800)
        
        # Expert configurations
        self.experts = {
            HuiHuiExpertType.MARKET_REGIME: {
                "name": "huihui_ai/huihui-moe-abliterated:5b-a1.7b",
                "display_name": "🏛️ HuiHui Market Regime Expert",
                "max_tokens": 350,  # ULTRA-OPTIMIZED for speed
                "temperature": 0.05,  # Lower for more consistency
                "system_prompt": "[EXPERT:MARKET_REGIME] You are the HuiHui Market Regime Expert specializing in EOTS v2.5 analytics. VRI means 'Volatility Regime Indicator' - a core EOTS metric that measures volatility regime strength and transitions. VRI values indicate: positive = bullish volatility regime, negative = bearish volatility regime. Analyze volatility patterns, regime transitions, and market structure using EOTS terminology. Be concise and EOTS-focused.",
                "semantic_keywords": [
                    "market regime analysis", "volatility regime indicator", "VRI analysis",
                    "volatility patterns", "regime transitions", "vol expansion", "vol contraction",
                    "market structure", "volatility regimes", "regime detection", "VFI analysis",
                    "VVR analysis", "VCI analysis", "volatility clustering", "regime shifts"
                ]
            },
            HuiHuiExpertType.OPTIONS_FLOW: {
                "name": "huihui_ai/huihui-moe-abliterated:5b-a1.7b",
                "display_name": "🚀 HuiHui Options Flow Expert",
                "max_tokens": 350, # ULTRA-OPTIMIZED for speed
                "temperature": 0.05, # Lower for more consistency
                "system_prompt": "[EXPERT:OPTIONS_FLOW] You are the HuiHui Options Flow Expert specializing in EOTS v2.5 metrics. MSPI means 'Market Structure Pressure Index' - an EOTS composite metric combining A-DAG, D-TDPI, VRI-2.0, and E-SDAG components to measure structural market pressure. Analyze VAPI-FA (Volume-Adjusted Premium Intensity), DWFD (Delta-Weighted Flow Dynamics), and institutional flow patterns using EOTS terminology. Be concise and EOTS-focused.",
                "semantic_keywords": [
                    "options flow analysis", "MSPI analysis", "market structure pressure index",
                    "VAPI-FA analysis", "DWFD analysis", "TW-LAF analysis", "gamma exposure",
                    "delta flow", "institutional flow", "options volume", "flow dynamics",
                    "A-DAG analysis", "D-TDPI analysis", "E-SDAG analysis", "VRI-2.0 analysis",
                    "structural pressure", "flow patterns", "options positioning"
                ]
            },
            HuiHuiExpertType.SENTIMENT: {
                "name": "huihui_ai/huihui-moe-abliterated:5b-a1.7b",
                "display_name": "🧠 HuiHui Sentiment Expert",
                "max_tokens": 350, # ULTRA-OPTIMIZED for speed
                "temperature": 0.05, # Consistent with other focused experts here
                "system_prompt": "[EXPERT:SENTIMENT] You are the HuiHui Sentiment Expert specializing in EOTS v2.5 behavioral analysis. Analyze market sentiment, psychology, and behavioral patterns that impact EOTS trading decisions. Focus on sentiment indicators, news intelligence, and crowd psychology. Be concise and EOTS-focused.",
                "semantic_keywords": [
                    "market sentiment analysis", "investor psychology", "behavioral analysis",
                    "news intelligence", "crowd psychology", "sentiment indicators",
                    "market mood", "fear and greed", "sentiment shifts", "behavioral patterns",
                    "news sentiment", "social sentiment", "market psychology", "investor behavior"
                ]
            },
            HuiHuiExpertType.ORCHESTRATOR: {
                "name": "huihui_ai/huihui-moe-abliterated:5b-a1.7b",
                "display_name": "🎯 HuiHui Meta-Orchestrator",
                "max_tokens": 500,  # Slightly larger for synthesis
                "temperature": 0.1,   # Slightly higher for strategic thinking
                "system_prompt": "[EXPERT:ORCHESTRATOR] You are the HuiHui Meta-Orchestrator specializing in EOTS v2.5 strategic synthesis. Combine insights from all EOTS metrics including VRI (Volatility Regime Indicator), MSPI (Market Structure Pressure Index), VAPI-FA, DWFD, and regime analysis to provide comprehensive trading recommendations. Be strategic and EOTS-focused.",
                "semantic_keywords": [
                    "strategic analysis", "comprehensive recommendation", "trade strategy",
                    "position recommendation", "overall assessment", "strategic synthesis",
                    "trading decision", "risk management", "portfolio strategy", "market outlook",
                    "strategic planning", "investment strategy", "comprehensive analysis"
                ]
            }
        }
        
        # Background tasks
        self._bg_tasks = set()
        self._shutdown_event = asyncio.Event()
        self._warming_lock = asyncio.Lock()
        
        logger.info(f"🚀 Expert Router initialized with {len(self.performance)} experts")
    
    async def __aenter__(self):
        """Async context manager entry"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(connector=self.connector)
        
        # Initialize vector search if available
        if not self.vector_search_enabled:
            await self._initialize_vector_search()
        
        # Start background tasks
        self._start_background_tasks()
        
        # Warm up experts in the background
        asyncio.create_task(self._safe_warm_experts())
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown()
    
    async def _safe_warm_experts(self):
        """Safely warm up experts with error handling"""
        try:
            await self.warm_experts_async()
        except Exception as e:
            logger.warning(f"Expert warming failed: {e}")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        if not hasattr(self, '_bg_tasks'):
            self._bg_tasks = set()
        
        # Start weight adjustment task
        task = asyncio.create_task(self._periodic_weight_adjustment())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
    
    async def _periodic_weight_adjustment(self):
        """Periodically adjust expert weights based on performance"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._adjust_weights()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in weight adjustment: {e}")
    
    async def _adjust_weights(self):
        """Adjust expert weights based on recent performance"""
        logger.info("🔄 Adjusting expert weights based on performance...")
        
        # Calculate performance scores for each expert
        scores = {}
        for expert_type, perf in self.performance.items():
            scores[expert_type] = perf.score
        
        # Normalize scores
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            for expert_type in scores:
                scores[expert_type] /= max_score
        
        # Update weights with momentum
        for expert_type, score in scores.items():
            current_weight = self.performance[expert_type].weight
            target_weight = MIN_WEIGHT + (MAX_WEIGHT - MIN_WEIGHT) * score
            
            # Apply momentum to smooth weight changes
            new_weight = current_weight * 0.8 + target_weight * 0.2
            
            # Clamp to valid range
            new_weight = max(MIN_WEIGHT, min(MAX_WEIGHT, new_weight))
            
            self.performance[expert_type].weight = new_weight
            logger.debug(f"  {expert_type.value}: {current_weight:.2f} → {new_weight:.2f} (score: {score:.2f})")
        
        self.last_weight_update = datetime.now()
    
    async def _initialize_vector_search(self):
        """Initialize vector search components if available"""
        if not VECTOR_SEARCH_AVAILABLE:
            logger.warning("Vector search not available. Install sentence-transformers for better routing.")
            return
            
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Pre-compute expert embeddings
            expert_texts = {}
            for expert_type, config in self.experts.items():
                expert_text = ' '.join([
                    config.get('display_name', ''),
                    config.get('system_prompt', ''),
                    ' '.join(config.get('semantic_keywords', []))
                ])
                expert_texts[expert_type] = expert_text
            
            # Get embeddings for all experts
            expert_texts_list = list(expert_texts.values())
            expert_embeddings = self.embedding_model.encode(expert_texts_list)
            
            # Store embeddings
            self.expert_embeddings = {
                expert_type: embedding 
                for (expert_type, _), embedding in zip(expert_texts.items(), expert_embeddings)
            }
            
            self.vector_search_enabled = True
            logger.info("✅ Vector search initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector search: {e}")
            self.vector_search_enabled = False
    
    async def detect_expert_type_async(self, prompt: str) -> HuiHuiExpertType:
        """
        Detect the most appropriate expert for a given prompt using vector similarity
        
        Args:
            prompt: The input prompt to analyze
            
        Returns:
            The most appropriate expert type
        """
        if not self.vector_search_enabled or not self.expert_embeddings:
            # Fallback to simple keyword matching if vector search is not available
            return self._fallback_expert_detection(prompt)
        
        try:
            # Get embedding for the prompt
            prompt_embedding = self.embedding_model.encode([prompt])[0]
            
            # Calculate similarities to all expert embeddings
            similarities = {}
            for expert_type, expert_embedding in self.expert_embeddings.items():
                similarity = np.dot(prompt_embedding, expert_embedding) / (
                    np.linalg.norm(prompt_embedding) * np.linalg.norm(expert_embedding) + 1e-10
                )
                similarities[expert_type] = float(similarity)
            
            # Get expert with highest similarity
            best_expert = max(similarities.items(), key=lambda x: x[1])[0]
            
            # Apply confidence threshold
            if similarities[best_expert] < CONFIDENCE_THRESHOLD:
                logger.debug(f"Low confidence in expert detection ({similarities[best_expert]:.2f}), falling back to orchestrator")
                return HuiHuiExpertType.ORCHESTRATOR
                
            return best_expert
            
        except Exception as e:
            logger.error(f"Error in expert detection: {e}")
            return HuiHuiExpertType.ORCHESTRATOR
    
    def _fallback_expert_detection(self, prompt: str) -> HuiHuiExpertType:
        """Fallback expert detection using keyword matching"""
        prompt_lower = prompt.lower()
        
        # Simple keyword matching
        keyword_matches = {}
        for expert_type, config in self.experts.items():
            keywords = config.get('semantic_keywords', []) + [expert_type.value]
            match_count = sum(1 for kw in keywords if kw.lower() in prompt_lower)
            if match_count > 0:
                keyword_matches[expert_type] = match_count
        
        if keyword_matches:
            return max(keyword_matches.items(), key=lambda x: x[1])[0]
        
        # Default to orchestrator if no good match
        return HuiHuiExpertType.ORCHESTRATOR
    
    async def ask_async(self, prompt: str, expert_type: Optional[HuiHuiExpertType] = None) -> Dict:
        """
        Get a response from the specified expert (or auto-detect)
        
        Args:
            prompt: The input prompt
            expert_type: Optional expert to force
            
        Returns:
            Dict with response and metadata
        """
        start_time = time.time()
        
        try:
            # Auto-detect expert if not specified
            if expert_type is None:
                expert_type = await self.detect_expert_type_async(prompt)
            
            # Get expert config
            expert_config = self.experts.get(expert_type, {})
            
            # Update performance tracking
            self.performance[expert_type].total_queries += 1
            self.performance[expert_type].last_used = datetime.now()
            
            # Prepare the request
            model_name = expert_config.get('name', 'llama2')
            system_prompt = expert_config.get('system_prompt', '')
            temperature = expert_config.get('temperature', 0.1)
            max_tokens = expert_config.get('max_tokens', 400)
            
            # Make the API call
            response_start = time.time()
            response = await self._query_ollama(
                prompt=prompt,
                model=model_name,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            response_time = time.time() - response_start
            
            # Update metrics
            self.performance[expert_type].successful_responses += 1
            self.performance[expert_type].total_response_time += response_time
            self.performance[expert_type].recent_response_times.append(response_time)
            
            # Log query
            query_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                "expert": expert_type.value,
                "response_time": response_time,
                "success": True
            }
            self.query_history.append(query_log)
            
            return {
                "response": response,
                "expert_type": expert_type.value,
                "confidence": 1.0,  # Could be enhanced with actual confidence
                "metadata": {
                    "response_time": response_time,
                    "total_time": time.time() - start_time,
                    "expert_weights": {e.value: p.weight for e, p in self.performance.items()}
                }
            }
            
        except Exception as e:
            logger.error(f"Error in expert routing: {e}")
            
            # Update error count
            if expert_type is not None:
                self.performance[expert_type].error_count += 1
            
            # Log failed query
            if expert_type is None:
                expert_type = HuiHuiExpertType.ORCHESTRATOR
                
            query_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                "expert": expert_type.value,
                "error": str(e),
                "success": False
            }
            self.query_history.append(query_log)
            
            # Return error response
            return {
                "response": f"Error processing request: {str(e)}",
                "expert_type": expert_type.value if expert_type else "error",
                "error": str(e),
                "metadata": {
                    "error": True,
                    "fallback": expert_type == HuiHuiExpertType.ORCHESTRATOR
                }
            }
    
    async def _query_ollama(self, prompt: str, model: str, system_prompt: str = "", 
                          temperature: float = 0.1, max_tokens: int = 400) -> str:
        """
        Query the Ollama API
        
        Args:
            prompt: The input prompt
            model: The model to use
            system_prompt: System prompt for the model
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            The generated response
        """
        if self.session is None or self.session.closed:
            raise RuntimeError("Session not initialized. Use async with or call __aenter__ first.")
        
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            async with self.session.post(url, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                return result.get('response', '').strip()
                
        except aiohttp.ClientError as e:
            logger.error(f"Error querying Ollama: {e}")
            raise RuntimeError(f"Failed to get response from Ollama: {str(e)}")
    
    async def ask_batch(self, prompts: List[str]) -> List[Dict]:
        """
        Process multiple prompts in parallel
        
        Args:
            prompts: List of input prompts
            
        Returns:
            List of responses with metadata
        """
        if not prompts:
            return []
            
        # Process prompts in parallel
        tasks = [self.ask_async(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def warm_experts_async(self) -> None:
        """
        Warm up expert models by sending a simple prompt to each
        """
        if self._warming_lock.locked():
            logger.debug("Expert warming already in progress")
            return
            
        async with self._warming_lock:
            logger.info("🔥 Warming up experts...")
            
            # Create a simple test prompt for each expert
            test_prompts = {
                expert_type: f"[TEST] This is a warm-up request for {config['display_name']}"
                for expert_type, config in self.experts.items()
            }
            
            # Send warm-up requests in parallel
            tasks = []
            for expert_type, prompt in test_prompts.items():
                task = asyncio.create_task(
                    self.ask_async(prompt, expert_type=expert_type)
                )
                tasks.append(task)
            
            # Wait for all warm-ups to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log results
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            logger.info(f"✅ Warmed up {success_count}/{len(results)} experts")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics
        
        Returns:
            Dict with performance metrics
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "experts": {
                expert_type.value: {
                    "total_queries": perf.total_queries,
                    "success_rate": perf.success_rate,
                    "avg_response_time": perf.avg_response_time,
                    "current_weight": perf.weight,
                    "error_count": perf.error_count,
                    "last_used": perf.last_used.isoformat() if perf.last_used else None
                }
                for expert_type, perf in self.performance.items()
            },
            "system": {
                "total_queries": sum(p.total_queries for p in self.performance.values()),
                "avg_response_time": (
                    sum(p.avg_response_time * p.total_queries for p in self.performance.values()) /
                    max(1, sum(p.total_queries for p in self.performance.values()))
                    if any(p.total_queries > 0 for p in self.performance.values())
                    else 0.0
                ),
                "last_weight_update": self.last_weight_update.isoformat(),
                **self.performance_metrics.__dict__
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown of the router"""
        self._shutdown_event.set()
        
        # Cancel all background tasks
        for task in self._bg_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._bg_tasks:
            await asyncio.wait(self._bg_tasks, timeout=5.0)
        
        # Close session
        if self.session and not self.session.closed:
            await self.session.close()
        
        logger.info("🛑 Expert Router shutdown complete")

# Backward compatibility aliases
HuiHuiRouter = ExpertRouter
AIRouter = ExpertRouter

# Convenience function for quick usage
async def create_expert_router(ollama_host: str = DEFAULT_OLLAMA_HOST) -> ExpertRouter:
    """Create and initialize an ExpertRouter instance"""
    router = ExpertRouter(ollama_host=ollama_host)
    await router.__aenter__()
    return router

# Example usage
async def example_usage():
    """Example usage of the ExpertRouter"""
    router = await create_expert_router()
    
    try:
        # Example prompts
        prompts = [
            "Analyze the current market regime",
            "What's the options flow telling us?",
            "What's the market sentiment?",
            "Provide a strategic analysis of SPY"
        ]
        
        # Process prompts
        for prompt in prompts:
            print(f"\n📝 Prompt: {prompt}")
            response = await router.ask_async(prompt)
            print(f"🧠 Expert: {response.get('expert_type', 'unknown')}")
            print(f"💡 Response: {response.get('response', '')[:200]}...")
        
        # Show performance metrics
        print("\n📊 Performance Metrics:")
        print(json.dumps(router.performance_metrics.__dict__, indent=2))
        
    finally:
        await router.shutdown()

if __name__ == "__main__":
    asyncio.run(example_usage())
