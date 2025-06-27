"""
MOE SUPABASE IMPLEMENTATION SCRIPTS
==================================
Python scripts to interact with the MOE databases
Handles data insertion, querying, and learning operations
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from supabase import create_client, Client
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
import json

# =====================================================
# SUPABASE CLIENT SETUP
# =====================================================

class SupabaseConfig:
    """Configuration for Supabase connection"""
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL", "your-supabase-url")
        self.key = os.getenv("SUPABASE_ANON_KEY", "your-supabase-anon-key")
        self.client: Client = create_client(self.url, self.key)

# =====================================================
# PYDANTIC MODELS FOR DATA VALIDATION
# =====================================================

class MarketRegimeData(BaseModel):
    """Market Regime MOE training data model"""
    timestamp: datetime
    regime_type: str = Field(..., regex="^(bull|bear|sideways|volatile)$")
    regime_strength: float = Field(..., ge=0.0, le=1.0)
    regime_duration: int = Field(..., ge=0)
    transition_probability: float = Field(..., ge=0.0, le=1.0)
    
    # Core EOTS Metrics
    vri_2_0: Optional[float] = None
    vri_raw: Optional[float] = None
    vri_smoothed: Optional[float] = None
    volatility_rank: Optional[float] = Field(None, ge=0.0, le=1.0)
    volatility_percentile: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Market Context
    spy_price: Optional[float] = Field(None, gt=0)
    spy_change_pct: Optional[float] = None
    spy_volume: Optional[int] = Field(None, ge=0)
    vix_level: Optional[float] = Field(None, gt=0)
    vix_change_pct: Optional[float] = None
    
    # Regime Indicators
    trend_strength: Optional[float] = Field(None, ge=0.0, le=1.0)
    momentum_score: Optional[float] = None
    mean_reversion_signal: Optional[float] = None
    breakout_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    consolidation_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Time-based Features
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    days_to_expiry: Optional[int] = Field(None, ge=0)
    is_earnings_week: bool = False
    is_fomc_week: bool = False
    
    # Outcome Data
    actual_regime_next_hour: Optional[str] = None
    actual_regime_next_day: Optional[str] = None
    actual_regime_next_week: Optional[str] = None
    regime_change_occurred: Optional[bool] = None
    prediction_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Metadata
    data_source: str = "EOTS_System"
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)

class OptionsFlowData(BaseModel):
    """Options Flow MOE training data model"""
    timestamp: datetime
    
    # Core Flow Metrics
    vapi_fa: Optional[float] = None
    dwfd: Optional[float] = None
    tw_laf: Optional[float] = None
    
    # Custom Flow Metrics
    lwpai: Optional[float] = None
    vabai: Optional[float] = None
    aofm: Optional[float] = None
    lidb: Optional[float] = None
    
    # Market Context
    spy_price: Optional[float] = Field(None, gt=0)
    spy_iv: Optional[float] = Field(None, gt=0)
    total_volume: Optional[int] = Field(None, ge=0)
    call_volume: Optional[int] = Field(None, ge=0)
    put_volume: Optional[int] = Field(None, ge=0)
    call_put_ratio: Optional[float] = Field(None, gt=0)
    
    # Flow Characteristics
    institutional_flow_score: Optional[float] = None
    retail_flow_score: Optional[float] = None
    smart_money_indicator: Optional[float] = None
    aggressive_flow_pct: Optional[float] = Field(None, ge=0.0, le=1.0)
    passive_flow_pct: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Gamma Metrics
    total_gamma: Optional[float] = None
    call_gamma: Optional[float] = None
    put_gamma: Optional[float] = None
    gamma_imbalance: Optional[float] = None
    gex_level: Optional[float] = None
    
    # Strike Analysis
    max_pain: Optional[float] = Field(None, gt=0)
    gamma_flip_level: Optional[float] = Field(None, gt=0)
    resistance_levels: Optional[str] = None  # JSON string
    support_levels: Optional[str] = None     # JSON string
    
    # Time Decay Factors
    avg_dte: Optional[float] = Field(None, gt=0)
    theta_exposure: Optional[float] = None
    vega_exposure: Optional[float] = None
    
    # Flow Outcomes
    price_move_1h: Optional[float] = None
    price_move_4h: Optional[float] = None
    price_move_1d: Optional[float] = None
    volatility_realized_1d: Optional[float] = Field(None, gt=0)
    flow_prediction_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Metadata
    data_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)

class MarketIntelligenceData(BaseModel):
    """Market Intelligence MOE training data model"""
    timestamp: datetime
    
    # Intelligence Metrics
    mspi: Optional[float] = None
    aofm: Optional[float] = None
    sentiment_score: Optional[float] = None
    momentum_intelligence: Optional[float] = None
    
    # Market Psychology
    fear_greed_index: Optional[float] = Field(None, ge=0, le=100)
    put_call_ratio: Optional[float] = Field(None, gt=0)
    vix_term_structure: Optional[float] = None
    skew_indicator: Optional[float] = None
    
    # News & Sentiment
    news_sentiment_score: Optional[float] = None
    social_sentiment_score: Optional[float] = None
    analyst_sentiment_score: Optional[float] = None
    insider_activity_score: Optional[float] = None
    
    # Technical Intelligence
    technical_score: Optional[float] = None
    momentum_score: Optional[float] = None
    mean_reversion_score: Optional[float] = None
    breakout_score: Optional[float] = None
    
    # Cross-Asset Intelligence
    bond_equity_correlation: Optional[float] = Field(None, ge=-1.0, le=1.0)
    dollar_strength_impact: Optional[float] = None
    commodity_correlation: Optional[float] = Field(None, ge=-1.0, le=1.0)
    crypto_correlation: Optional[float] = Field(None, ge=-1.0, le=1.0)
    
    # Sector Intelligence
    sector_rotation_score: Optional[float] = None
    growth_value_ratio: Optional[float] = Field(None, gt=0)
    large_small_cap_ratio: Optional[float] = Field(None, gt=0)
    defensive_cyclical_ratio: Optional[float] = Field(None, gt=0)
    
    # Economic Intelligence
    economic_surprise_index: Optional[float] = None
    earnings_revision_trend: Optional[float] = None
    credit_spread_indicator: Optional[float] = None
    yield_curve_signal: Optional[float] = None
    
    # Contrarian Indicators
    contrarian_signal_strength: Optional[float] = None
    extreme_sentiment_flag: bool = False
    reversal_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Intelligence Outcomes
    intelligence_accuracy_1d: Optional[float] = Field(None, ge=0.0, le=1.0)
    intelligence_accuracy_1w: Optional[float] = Field(None, ge=0.0, le=1.0)
    signal_strength_validation: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Metadata
    intelligence_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    data_freshness_score: Optional[float] = Field(None, ge=0.0, le=1.0)

# =====================================================
# MOE DATABASE MANAGERS
# =====================================================

class MarketRegimeMOEDatabase:
    """Database manager for Market Regime MOE"""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.table_training = "market_regime_training"
        self.table_patterns = "regime_patterns"
        self.table_performance = "regime_moe_performance"
    
    async def insert_training_data(self, data: MarketRegimeData) -> Dict[str, Any]:
        """Insert training data for regime MOE"""
        try:
            # Convert Pydantic model to dict and handle None values
            data_dict = data.dict(exclude_none=True)
            
            result = self.client.table(self.table_training).insert(data_dict).execute()
            return {"success": True, "data": result.data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_recent_training_data(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent training data for analysis"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            result = self.client.table(self.table_training)\
                .select("*")\
                .gte("timestamp", cutoff_time.isoformat())\
                .order("timestamp", desc=True)\
                .execute()
            
            return result.data
        except Exception as e:
            print(f"Error fetching training data: {e}")
            return []
    
    async def update_pattern_recognition(self, pattern_name: str, 
                                       success: bool, detection_time: datetime) -> Dict[str, Any]:
        """Update pattern recognition statistics"""
        try:
            # First, try to get existing pattern
            existing = self.client.table(self.table_patterns)\
                .select("*")\
                .eq("pattern_name", pattern_name)\
                .execute()
            
            if existing.data:
                # Update existing pattern
                pattern = existing.data[0]
                new_detections = pattern["times_detected"] + 1
                new_successes = pattern["times_correct"] + (1 if success else 0)
                new_success_rate = new_successes / new_detections
                
                result = self.client.table(self.table_patterns)\
                    .update({
                        "times_detected": new_detections,
                        "times_correct": new_successes,
                        "success_rate": new_success_rate,
                        "last_seen": detection_time.isoformat()
                    })\
                    .eq("pattern_name", pattern_name)\
                    .execute()
            else:
                # Create new pattern
                result = self.client.table(self.table_patterns)\
                    .insert({
                        "pattern_name": pattern_name,
                        "pattern_type": "auto_discovered",
                        "times_detected": 1,
                        "times_correct": 1 if success else 0,
                        "success_rate": 1.0 if success else 0.0,
                        "last_seen": detection_time.isoformat(),
                        "confidence_level": 0.5  # Initial confidence
                    })\
                    .execute()
            
            return {"success": True, "data": result.data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def log_performance(self, prediction: str, confidence: float, 
                            actual_outcome: str, accuracy_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Log MOE performance metrics"""
        try:
            performance_data = {
                "timestamp": datetime.now().isoformat(),
                "regime_prediction": prediction,
                "regime_confidence": confidence,
                "actual_outcome": actual_outcome,
                "prediction_correct": prediction == actual_outcome,
                **accuracy_metrics
            }
            
            result = self.client.table(self.table_performance)\
                .insert(performance_data)\
                .execute()
            
            return {"success": True, "data": result.data}
        except Exception as e:
            return {"success": False, "error": str(e)}

class OptionsFlowMOEDatabase:
    """Database manager for Options Flow MOE"""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.table_training = "options_flow_training"
        self.table_patterns = "flow_patterns"
        self.table_performance = "flow_moe_performance"
    
    async def insert_training_data(self, data: OptionsFlowData) -> Dict[str, Any]:
        """Insert training data for flow MOE"""
        try:
            data_dict = data.dict(exclude_none=True)
            
            result = self.client.table(self.table_training).insert(data_dict).execute()
            return {"success": True, "data": result.data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def analyze_flow_patterns(self, lookback_hours: int = 168) -> Dict[str, Any]:
        """Analyze flow patterns for learning"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            
            # Get recent flow data
            result = self.client.table(self.table_training)\
                .select("vapi_fa, dwfd, tw_laf, institutional_flow_score, price_move_1d")\
                .gte("timestamp", cutoff_time.isoformat())\
                .not_.is_("price_move_1d", "null")\
                .execute()
            
            if not result.data:
                return {"success": False, "error": "No data available"}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(result.data)
            
            # Analyze patterns
            patterns = {
                "high_vapi_fa_outcomes": df[df["vapi_fa"] > df["vapi_fa"].quantile(0.8)]["price_move_1d"].mean(),
                "institutional_flow_correlation": df["institutional_flow_score"].corr(df["price_move_1d"]),
                "flow_convergence_rate": len(df[(df["vapi_fa"] > 1.5) & (df["dwfd"] > 1.5)]) / len(df)
            }
            
            return {"success": True, "patterns": patterns}
        except Exception as e:
            return {"success": False, "error": str(e)}

class MarketIntelligenceMOEDatabase:
    """Database manager for Market Intelligence MOE"""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.table_training = "intelligence_training"
        self.table_patterns = "intelligence_patterns"
        self.table_performance = "intelligence_moe_performance"
    
    async def insert_training_data(self, data: MarketIntelligenceData) -> Dict[str, Any]:
        """Insert training data for intelligence MOE"""
        try:
            data_dict = data.dict(exclude_none=True)
            
            result = self.client.table(self.table_training).insert(data_dict).execute()
            return {"success": True, "data": result.data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def detect_sentiment_extremes(self, threshold: float = 0.8) -> Dict[str, Any]:
        """Detect sentiment extremes for contrarian signals"""
        try:
            # Get recent sentiment data
            result = self.client.table(self.table_training)\
                .select("timestamp, sentiment_score, fear_greed_index, contrarian_signal_strength")\
                .gte("timestamp", (datetime.now() - timedelta(days=7)).isoformat())\
                .order("timestamp", desc=True)\
                .limit(100)\
                .execute()
            
            if not result.data:
                return {"success": False, "error": "No sentiment data available"}
            
            df = pd.DataFrame(result.data)
            
            # Detect extremes
            extremes = {
                "fear_extreme": (df["fear_greed_index"] < 20).any(),
                "greed_extreme": (df["fear_greed_index"] > 80).any(),
                "sentiment_extreme_count": len(df[abs(df["sentiment_score"]) > threshold]),
                "contrarian_opportunity": df["contrarian_signal_strength"].iloc[0] > threshold if len(df) > 0 else False
            }
            
            return {"success": True, "extremes": extremes}
        except Exception as e:
            return {"success": False, "error": str(e)}

class MetaOrchestratorDatabase:
    """Database manager for Meta-Orchestrator"""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.table_decisions = "orchestrator_decisions"
        self.table_communication = "expert_communication"
        self.table_performance = "system_performance"
    
    async def log_decision(self, expert_signals: Dict[str, Any], 
                          final_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Log orchestrator decision with expert inputs"""
        try:
            decision_data = {
                "timestamp": datetime.now().isoformat(),
                **expert_signals,
                **final_decision
            }
            
            result = self.client.table(self.table_decisions)\
                .insert(decision_data)\
                .execute()
            
            return {"success": True, "data": result.data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def log_expert_communication(self, from_expert: str, to_expert: str,
                                     message: str, data_payload: Dict[str, Any] = None) -> Dict[str, Any]:
        """Log communication between experts"""
        try:
            comm_data = {
                "timestamp": datetime.now().isoformat(),
                "from_expert": from_expert,
                "to_expert": to_expert,
                "message_type": "learning",
                "message_content": message,
                "data_payload": json.dumps(data_payload) if data_payload else None,
                "priority_level": 3
            }
            
            result = self.client.table(self.table_communication)\
                .insert(comm_data)\
                .execute()
            
            return {"success": True, "data": result.data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_system_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get system-wide performance summary"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            result = self.client.table(self.table_performance)\
                .select("*")\
                .gte("timestamp", cutoff_time.isoformat())\
                .order("timestamp", desc=True)\
                .execute()
            
            if not result.data:
                return {"success": False, "error": "No performance data available"}
            
            df = pd.DataFrame(result.data)
            
            summary = {
                "avg_system_accuracy": df["overall_system_accuracy"].mean(),
                "avg_consensus_rate": df["expert_consensus_rate"].mean(),
                "avg_decision_latency": df["decision_latency_ms"].mean(),
                "learning_trend": df["collective_learning_rate"].iloc[-1] - df["collective_learning_rate"].iloc[0] if len(df) > 1 else 0,
                "system_confidence": df["system_confidence"].iloc[0] if len(df) > 0 else 0
            }
            
            return {"success": True, "summary": summary}
        except Exception as e:
            return {"success": False, "error": str(e)}

# =====================================================
# MAIN MOE DATABASE MANAGER
# =====================================================

class MOEDatabaseManager:
    """Main manager for all MOE databases"""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        # Initialize Supabase client
        self.config = SupabaseConfig()
        if supabase_url:
            self.config.url = supabase_url
        if supabase_key:
            self.config.key = supabase_key
        
        self.client = create_client(self.config.url, self.config.key)
        
        # Initialize MOE database managers
        self.regime_db = MarketRegimeMOEDatabase(self.client)
        self.flow_db = OptionsFlowMOEDatabase(self.client)
        self.intelligence_db = MarketIntelligenceMOEDatabase(self.client)
        self.orchestrator_db = MetaOrchestratorDatabase(self.client)
    
    async def bulk_insert_historical_data(self, historical_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Bulk insert historical data for all MOEs"""
        results = {}
        
        try:
            # Insert regime data
            if "regime" in historical_data:
                regime_results = []
                for data_point in historical_data["regime"]:
                    regime_data = MarketRegimeData(**data_point)
                    result = await self.regime_db.insert_training_data(regime_data)
                    regime_results.append(result)
                results["regime"] = regime_results
            
            # Insert flow data
            if "flow" in historical_data:
                flow_results = []
                for data_point in historical_data["flow"]:
                    flow_data = OptionsFlowData(**data_point)
                    result = await self.flow_db.insert_training_data(flow_data)
                    flow_results.append(result)
                results["flow"] = flow_results
            
            # Insert intelligence data
            if "intelligence" in historical_data:
                intel_results = []
                for data_point in historical_data["intelligence"]:
                    intel_data = MarketIntelligenceData(**data_point)
                    result = await self.intelligence_db.insert_training_data(intel_data)
                    intel_results.append(result)
                results["intelligence"] = intel_results
            
            return {"success": True, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_learning_progress_report(self) -> Dict[str, Any]:
        """Get comprehensive learning progress report for all MOEs"""
        try:
            # Get performance summaries
            regime_data = await self.regime_db.get_recent_training_data(hours=168)  # 1 week
            flow_patterns = await self.flow_db.analyze_flow_patterns(lookback_hours=168)
            intelligence_extremes = await self.intelligence_db.detect_sentiment_extremes()
            system_summary = await self.orchestrator_db.get_system_performance_summary(days=7)
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "regime_moe": {
                    "training_data_points": len(regime_data),
                    "data_coverage_days": 7
                },
                "flow_moe": flow_patterns,
                "intelligence_moe": intelligence_extremes,
                "system_performance": system_summary,
                "overall_health": "healthy" if all([
                    len(regime_data) > 0,
                    flow_patterns.get("success", False),
                    intelligence_extremes.get("success", False),
                    system_summary.get("success", False)
                ]) else "needs_attention"
            }
            
            return {"success": True, "report": report}
        except Exception as e:
            return {"success": False, "error": str(e)}

# =====================================================
# EXAMPLE USAGE AND TESTING
# =====================================================

async def example_usage():
    """Example of how to use the MOE database system"""
    
    # Initialize the database manager
    db_manager = MOEDatabaseManager()
    
    # Example: Insert regime training data
    regime_data = MarketRegimeData(
        timestamp=datetime.now(),
        regime_type="bull",
        regime_strength=0.85,
        regime_duration=5,
        transition_probability=0.15,
        vri_2_0=-2.53,
        spy_price=450.25,
        spy_change_pct=1.2,
        vix_level=18.5,
        hour_of_day=14,
        day_of_week=2,
        confidence_score=0.9
    )
    
    result = await db_manager.regime_db.insert_training_data(regime_data)
    print(f"Regime data insertion: {result}")
    
    # Example: Insert flow training data
    flow_data = OptionsFlowData(
        timestamp=datetime.now(),
        vapi_fa=1.84,
        dwfd=1.96,
        tw_laf=0.33,
        lwpai=2.1,
        vabai=1.7,
        spy_price=450.25,
        total_volume=1000000,
        call_volume=600000,
        put_volume=400000,
        call_put_ratio=1.5
    )
    
    result = await db_manager.flow_db.insert_training_data(flow_data)
    print(f"Flow data insertion: {result}")
    
    # Example: Get learning progress report
    report = await db_manager.get_learning_progress_report()
    print(f"Learning progress report: {report}")

if __name__ == "__main__":
    # Run example usage
    asyncio.run(example_usage())

