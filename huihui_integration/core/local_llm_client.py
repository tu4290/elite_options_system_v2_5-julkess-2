"""
Local LLM Client for EOTS v2.5 - "YOUR PERSONAL AI API"
======================================================

Simple client for accessing your local HuiHui-MoE and DeepSeek V2 models
from IDEs, external tools, or any Python environment.

Usage Examples:
    # Quick chat with HuiHui-MoE
    from huihui_integration.core.local_llm_client import LocalLLMClient
    
    client = LocalLLMClient()
    response = client.chat_huihui("Analyze SPY market regime")
    print(response)
    
    # Coding assistance with DeepSeek V2
    code = client.chat_deepseek("Write a Python function for Black-Scholes pricing")
    print(code)

Author: EOTS v2.5 AI Liberation Division
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    HUIHUI_MOE = "huihui_moe"
    DEEPSEEK_V2 = "deepseek_v2"

@dataclass
class ChatMessage:
    role: str  # "user", "assistant", "system"
    content: str

class LocalLLMClient:
    """
    Simple client for your local LLM models.
    
    Provides easy access to HuiHui-MoE and DeepSeek V2 with proper API keys.
    """
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.session = requests.Session()
        
        # Load API keys
        self.api_keys = {
            ModelType.HUIHUI_MOE: "huihui-moe-specialist-expert-system",
            ModelType.DEEPSEEK_V2: "deepseek-v2-coding-assistant-elite"
        }
        
        # Model configurations
        self.models = {
            ModelType.HUIHUI_MOE: {
                "name": "huihui_ai/huihui-moe-abliterated:5b-a1.7b",
                "temperature": 0.1,
                "max_tokens": 2000
            },
            ModelType.DEEPSEEK_V2: {
                "name": "deepseek-coder-v2:16b", 
                "temperature": 0.1,
                "max_tokens": 4000
            }
        }
    
    def _make_request(self, model_type: ModelType, messages: List[ChatMessage], 
                     temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Make a request to the local LLM API."""
        
        model_config = self.models[model_type]
        api_key = self.api_keys[model_type]
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_config["name"],
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "stream": False,
            "options": {
                "temperature": temperature or model_config["temperature"],
                "num_predict": max_tokens or model_config["max_tokens"]
            }
        }
        
        try:
            response = self.session.post(
                f"{self.host}/api/chat",
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "")
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def chat_huihui(self, prompt: str, specialist: str = "general", 
                   temperature: float = 0.1) -> str:
        """
        Chat with HuiHui-MoE specialist system.
        
        Args:
            prompt: Your question or request
            specialist: Type of specialist ("market_regime", "options_flow", "sentiment", "orchestrator", "general")
            temperature: Response creativity (0.1 = focused, 0.3 = creative)
        """
        
        # Specialist system prompts
        specialist_prompts = {
            "market_regime": "You are a market regime and volatility analysis expert. Focus on regime detection, volatility patterns, and structural analysis.",
            "options_flow": "You are an options flow and institutional behavior expert. Focus on flow analysis, gamma dynamics, and institutional positioning.",
            "sentiment": "You are a sentiment and news intelligence expert. Focus on market psychology, news analysis, and crowd behavior.",
            "orchestrator": "You are the meta-orchestrator. Synthesize all available information and provide strategic trading decisions.",
            "general": "You are an elite trading AI assistant with expertise across all market domains."
        }
        
        messages = []
        if specialist in specialist_prompts:
            messages.append(ChatMessage("system", specialist_prompts[specialist]))
        
        messages.append(ChatMessage("user", prompt))
        
        return self._make_request(ModelType.HUIHUI_MOE, messages, temperature)
    
    def chat_deepseek(self, prompt: str, mode: str = "development", 
                     temperature: float = 0.1) -> str:
        """
        Chat with DeepSeek V2 coding assistant.
        
        Args:
            prompt: Your coding question or request
            mode: Coding mode ("development", "debugging", "architecture", "review")
            temperature: Response creativity (0.05 = very focused, 0.15 = balanced)
        """
        
        mode_prompts = {
            "development": "You are an elite Python developer specializing in trading systems. Provide clean, efficient, well-documented code.",
            "debugging": "You are a debugging expert. Analyze code issues and provide precise solutions with explanations.",
            "architecture": "You are a system architect. Design scalable, maintainable trading system architectures.",
            "review": "You are a code reviewer. Analyze code quality, performance, and suggest improvements."
        }
        
        messages = []
        if mode in mode_prompts:
            messages.append(ChatMessage("system", mode_prompts[mode]))
        
        messages.append(ChatMessage("user", prompt))
        
        return self._make_request(ModelType.DEEPSEEK_V2, messages, temperature)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of available models."""
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def test_connection(self) -> bool:
        """Test connection to local LLM server."""
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

# ===== CONVENIENCE FUNCTIONS =====

def quick_huihui(prompt: str, specialist: str = "general") -> str:
    """Quick function to chat with HuiHui-MoE."""
    client = LocalLLMClient()
    return client.chat_huihui(prompt, specialist)

def quick_deepseek(prompt: str, mode: str = "development") -> str:
    """Quick function to chat with DeepSeek V2."""
    client = LocalLLMClient()
    return client.chat_deepseek(prompt, mode)

def test_models() -> Dict[str, bool]:
    """Test both models are working."""
    client = LocalLLMClient()
    
    results = {
        "connection": client.test_connection(),
        "huihui_moe": False,
        "deepseek_v2": False
    }
    
    if results["connection"]:
        try:
            huihui_response = client.chat_huihui("Hello, are you working?")
            results["huihui_moe"] = len(huihui_response) > 0 and "error" not in huihui_response.lower()
            
            deepseek_response = client.chat_deepseek("Write a simple hello world function")
            results["deepseek_v2"] = len(deepseek_response) > 0 and "error" not in deepseek_response.lower()
        except:
            pass
    
    return results

# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Test the client
    print("🚀 Testing Local LLM Client...")
    
    client = LocalLLMClient()
    
    # Test connection
    if client.test_connection():
        print("✅ Connected to local LLM server")
        
        # Test HuiHui-MoE
        print("\n🧠 Testing HuiHui-MoE...")
        response = client.chat_huihui("What is your role in the EOTS system?", "orchestrator")
        print(f"Response: {response[:200]}...")
        
        # Test DeepSeek V2
        print("\n💻 Testing DeepSeek V2...")
        code = client.chat_deepseek("Write a simple function to calculate moving average")
        print(f"Code: {code[:200]}...")
        
    else:
        print("❌ Cannot connect to local LLM server")
        print("Make sure Ollama is running: ollama serve")
