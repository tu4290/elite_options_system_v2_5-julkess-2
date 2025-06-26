#!/usr/bin/env python3
"""
Optimized HuiHui Client - Performance-First Implementation
Bypasses router overhead for 5-10 second response times instead of 30+ seconds
"""

import time
import requests
from typing import Dict, Optional
from enum import Enum

class OptimizedHuiHuiExpert(Enum):
    """Optimized expert types for fast routing."""
    MARKET_REGIME = "market_regime"
    OPTIONS_FLOW = "options_flow"
    SENTIMENT = "sentiment"
    ORCHESTRATOR = "orchestrator"

class OptimizedHuiHuiClient:
    """
    High-performance HuiHui client optimized for speed.
    
    Performance improvements:
    - Direct Ollama API calls (no router overhead)
    - Optimized context window (1024 vs 4096)
    - Focused sampling (top_k=10, top_p=0.7)
    - Shorter responses (800 tokens vs 2000+)
    - Faster timeouts (15s vs 45s)
    
    Expected performance: 5-10 seconds vs 30+ seconds
    """
    
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.model = "huihui_ai/huihui-moe-abliterated:5b-a1.7b"
        
        # Optimized expert configurations
        self.expert_configs = {
            OptimizedHuiHuiExpert.MARKET_REGIME: {
                "name": "🏛️ Market Regime Expert",
                "system_prompt": "[EXPERT:MARKET_REGIME] You are the HuiHui Market Regime Expert. Analyze market volatility, VRI metrics, regime transitions, and structural patterns using EOTS analytics. Be concise and focused.",
                "max_tokens": 600,
                "temperature": 0.1
            },
            OptimizedHuiHuiExpert.OPTIONS_FLOW: {
                "name": "🚀 Options Flow Expert", 
                "system_prompt": "[EXPERT:OPTIONS_FLOW] You are the HuiHui Options Flow Expert. Analyze VAPI-FA, DWFD, TW-LAF metrics, gamma exposure, delta flows, and institutional options positioning. Be concise and focused.",
                "max_tokens": 600,
                "temperature": 0.1
            },
            OptimizedHuiHuiExpert.SENTIMENT: {
                "name": "🧠 Sentiment Expert",
                "system_prompt": "[EXPERT:SENTIMENT] You are the HuiHui Sentiment Expert. Analyze market sentiment, news intelligence, behavioral patterns, and psychological market drivers. Be concise and focused.",
                "max_tokens": 600,
                "temperature": 0.1
            },
            OptimizedHuiHuiExpert.ORCHESTRATOR: {
                "name": "🎯 Meta-Orchestrator",
                "system_prompt": "[EXPERT:ORCHESTRATOR] You are the HuiHui Meta-Orchestrator. Synthesize insights from all experts, provide strategic recommendations, and deliver comprehensive EOTS analysis. Be thorough but concise.",
                "max_tokens": 800,
                "temperature": 0.2
            }
        }
    
    def _detect_expert(self, prompt: str) -> OptimizedHuiHuiExpert:
        """Fast expert detection based on keywords."""
        prompt_lower = prompt.lower()
        
        # Options flow keywords
        if any(keyword in prompt_lower for keyword in ['vapi', 'dwfd', 'tw-laf', 'gamma', 'delta', 'flow', 'options']):
            return OptimizedHuiHuiExpert.OPTIONS_FLOW
        
        # Market regime keywords  
        elif any(keyword in prompt_lower for keyword in ['vri', 'regime', 'volatility', 'vix', 'market structure']):
            return OptimizedHuiHuiExpert.MARKET_REGIME
        
        # Sentiment keywords
        elif any(keyword in prompt_lower for keyword in ['sentiment', 'news', 'fed', 'earnings', 'psychology']):
            return OptimizedHuiHuiExpert.SENTIMENT
        
        # Default to orchestrator for complex queries
        else:
            return OptimizedHuiHuiExpert.ORCHESTRATOR
    
    def ask_fast(self, prompt: str, expert: Optional[OptimizedHuiHuiExpert] = None) -> Dict:
        """
        Fast HuiHui query with optimized performance.
        
        Args:
            prompt: Your question or analysis request
            expert: Optional specific expert (auto-detected if None)
            
        Returns:
            Dict with response, expert info, and performance metrics
        """
        # Auto-detect expert if not specified
        if expert is None:
            expert = self._detect_expert(prompt)
        
        config = self.expert_configs[expert]
        
        print(f"🚀 Fast Route → {config['name']}")
        
        # Build optimized request
        messages = [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": prompt}
        ]
        
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config["temperature"],
                "num_predict": config["max_tokens"],
                "num_ctx": 1024,      # Optimized: Small context for speed
                "top_k": 10,          # Optimized: Focused sampling
                "top_p": 0.7,         # Optimized: Narrow probability
                "stop": ["</think>"]  # Stop after thinking (optional)
            }
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json=data,
                timeout=15  # Fast timeout - expect 5-10s responses
            )
            response.raise_for_status()
            
            end_time = time.time()
            response_time = round(end_time - start_time, 2)
            
            result = response.json()
            content = result.get("message", {}).get("content", "")
            
            # Clean up <think> tags if present
            if "<think>" in content and "</think>" in content:
                # Extract only the content after </think>
                parts = content.split("</think>")
                if len(parts) > 1:
                    content = parts[1].strip()
            
            # Performance classification
            if response_time < 5:
                performance = "🚀 EXCELLENT"
            elif response_time < 10:
                performance = "⚡ GOOD"
            elif response_time < 15:
                performance = "⏱️ ACCEPTABLE"
            else:
                performance = "🐌 SLOW"
            
            print(f"   ✅ {response_time}s - {performance}")
            
            return {
                "response": content,
                "expert": expert.value,
                "expert_name": config["name"],
                "response_time": response_time,
                "performance": performance,
                "success": True,
                "response_length": len(content)
            }
            
        except requests.Timeout:
            end_time = time.time()
            response_time = round(end_time - start_time, 2)
            return {
                "response": f"⏰ Timeout after {response_time}s - try a shorter prompt",
                "expert": expert.value,
                "expert_name": config["name"],
                "response_time": response_time,
                "performance": "❌ TIMEOUT",
                "success": False,
                "error": "timeout"
            }
            
        except Exception as e:
            end_time = time.time()
            response_time = round(end_time - start_time, 2)
            return {
                "response": f"❌ Error: {str(e)}",
                "expert": expert.value,
                "expert_name": config["name"],
                "response_time": response_time,
                "performance": "❌ ERROR",
                "success": False,
                "error": str(e)
            }
    
    def ask_market_regime(self, prompt: str) -> Dict:
        """Fast market regime analysis."""
        return self.ask_fast(prompt, OptimizedHuiHuiExpert.MARKET_REGIME)
    
    def ask_options_flow(self, prompt: str) -> Dict:
        """Fast options flow analysis."""
        return self.ask_fast(prompt, OptimizedHuiHuiExpert.OPTIONS_FLOW)
    
    def ask_sentiment(self, prompt: str) -> Dict:
        """Fast sentiment analysis."""
        return self.ask_fast(prompt, OptimizedHuiHuiExpert.SENTIMENT)
    
    def ask_orchestrator(self, prompt: str) -> Dict:
        """Fast strategic orchestration."""
        return self.ask_fast(prompt, OptimizedHuiHuiExpert.ORCHESTRATOR)

# Convenience function for quick access
def create_fast_huihui_client() -> OptimizedHuiHuiClient:
    """Create an optimized HuiHui client for fast responses."""
    return OptimizedHuiHuiClient()

# Example usage
if __name__ == "__main__":
    client = create_fast_huihui_client()
    
    # Test fast performance
    print("🧪 Testing Optimized HuiHui Performance")
    print("=" * 40)
    
    test_cases = [
        "What is SPY VRI_2_0 = 0.75 indicating?",
        "VAPI-FA +1.5, DWFD -0.8. Analysis?",
        "Fed hawkish, market sentiment?",
        "SPY strategy: VIX 18, Put/Call 1.2?"
    ]
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {prompt}")
        result = client.ask_fast(prompt)
        print(f"   📝 Response: {result['response'][:100]}...")
        print(f"   ⚡ Performance: {result['performance']}") 