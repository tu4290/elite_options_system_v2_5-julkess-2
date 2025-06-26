from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import uuid

from data_models.eots_schemas_v2_5 import (
    MarketPredictionV2_5,
    PredictionPerformanceV2_5,
    PredictionConfigV2_5,
    UnifiedPredictionResult
)

class AIPredictionMetricsV2_5(BaseModel):
    """Pydantic model for tracking AI prediction metrics."""
    total_predictions: int = Field(default=0, ge=0)
    successful_predictions: int = Field(default=0, ge=0)
    failed_predictions: int = Field(default=0, ge=0)
    average_confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    prediction_cycles_completed: int = Field(default=0, ge=0)
    total_processing_time_ms: float = Field(default=0.0, ge=0.0)
    
    class Config:
        extra = 'forbid'

class AIPredictionsManagerV2_5:
    """
    AI Predictions Manager for Elite Options Trading System v2.5
    
    This class manages the AI prediction process, generating and tracking
    market predictions using advanced machine learning models.
    """
    
    def __init__(self, config: Optional[PredictionConfigV2_5] = None):
        """Initialize the AI predictions manager."""
        self.config = config or PredictionConfigV2_5()
        self.metrics = AIPredictionMetricsV2_5()
        self.predictions: List[MarketPredictionV2_5] = []
        self.performance_history: List[PredictionPerformanceV2_5] = []
        self.start_time = datetime.now()
        
    def generate_prediction(self, market_data: Dict[str, Any]) -> Optional[MarketPredictionV2_5]:
        """Generate a new market prediction based on input data."""
        try:
            # Validate confidence threshold
            if not self._validate_prediction_criteria(market_data):
                return None
                
            # Create prediction
            prediction = self._create_prediction(market_data)
            
            # Track metrics
            self.predictions.append(prediction)
            self.metrics.total_predictions += 1
            
            return prediction
            
        except Exception as e:
            self.metrics.failed_predictions += 1
            return None
            
    def _validate_prediction_criteria(self, market_data: Dict[str, Any]) -> bool:
        """Validate if market data meets prediction criteria."""
        # Check data completeness
        required_fields = ['price', 'volume', 'indicators']
        if not all(field in market_data for field in required_fields):
            return False
            
        # Check data freshness
        if 'timestamp' in market_data:
            data_age = (datetime.now() - market_data['timestamp']).total_seconds()
            if data_age > self.config.max_data_age_seconds:
                return False
                
        return True
        
    def _create_prediction(self, market_data: Dict[str, Any]) -> MarketPredictionV2_5:
        """Create a market prediction from input data."""
        return MarketPredictionV2_5(
            prediction_id=str(uuid.uuid4()),
            symbol=market_data.get('symbol', 'UNKNOWN'),
            prediction_type="market_direction",
            confidence_score=self._calculate_confidence(market_data),
            prediction_horizon="short_term",
            market_context={
                "price": market_data.get('price'),
                "volume": market_data.get('volume'),
                "indicators": market_data.get('indicators', {})
            },
            prediction_timestamp=datetime.now()
        )
        
    def _calculate_confidence(self, market_data: Dict[str, Any]) -> float:
        """Calculate confidence score for a prediction."""
        # Implement confidence calculation logic
        base_confidence = 0.7  # Default base confidence
        
        # Adjust based on data quality
        if self._check_data_quality(market_data):
            base_confidence += 0.1
            
        # Adjust based on market conditions
        if self._check_favorable_conditions(market_data):
            base_confidence += 0.1
            
        return min(base_confidence, 1.0)
        
    def _check_data_quality(self, market_data: Dict[str, Any]) -> bool:
        """Check quality of input market data."""
        # Implement data quality checks
        return all([
            'price' in market_data and isinstance(market_data['price'], (int, float)),
            'volume' in market_data and isinstance(market_data['volume'], (int, float)),
            'indicators' in market_data and isinstance(market_data['indicators'], dict)
        ])
        
    def _check_favorable_conditions(self, market_data: Dict[str, Any]) -> bool:
        """Check if market conditions are favorable for prediction."""
        # Implement market condition checks
        return True  # Placeholder
        
    def update_prediction_performance(self, prediction_id: str, actual_outcome: Dict[str, Any]):
        """Update performance metrics for a prediction."""
        try:
            # Find prediction
            prediction = next((p for p in self.predictions if p.prediction_id == prediction_id), None)
            if not prediction:
                return
                
            # Create performance record
            performance = PredictionPerformanceV2_5(
                prediction_id=prediction_id,
                symbol=prediction.symbol,
                prediction_type=prediction.prediction_type,
                confidence_score=prediction.confidence_score,
                actual_outcome=actual_outcome,
                performance_score=self._calculate_performance_score(prediction, actual_outcome),
                evaluation_timestamp=datetime.now()
            )
            
            # Update metrics
            self.performance_history.append(performance)
            if performance.performance_score >= self.config.success_threshold:
                self.metrics.successful_predictions += 1
                
            # Update average confidence
            self._update_average_confidence()
            
        except Exception as e:
            # Log error but continue
            pass
            
    def _calculate_performance_score(self, prediction: MarketPredictionV2_5, actual_outcome: Dict[str, Any]) -> float:
        """Calculate performance score for a prediction."""
        try:
            # Implement performance scoring logic
            predicted_direction = prediction.market_context.get('predicted_direction')
            actual_direction = actual_outcome.get('actual_direction')
            
            if predicted_direction == actual_direction:
                return 1.0
            return 0.0
            
        except Exception:
            return 0.0
            
    def _update_average_confidence(self):
        """Update average confidence score metric."""
        if not self.predictions:
            return
            
        total_confidence = sum(p.confidence_score for p in self.predictions)
        self.metrics.average_confidence_score = total_confidence / len(self.predictions)
        
    def get_prediction_summary(self) -> UnifiedPredictionResult:
        """Generate a summary of prediction performance."""
        return UnifiedPredictionResult(
            symbol="SYSTEM",  # System-wide predictions
            total_predictions=self.metrics.total_predictions,
            successful_predictions=self.metrics.successful_predictions,
            average_confidence=self.metrics.average_confidence_score,
            prediction_accuracy=self._calculate_accuracy(),
            recent_predictions=self._get_recent_predictions(),
            performance_trend=self._calculate_performance_trend(),
            next_prediction_cycle=self._calculate_next_cycle(),
            prediction_metadata={
                "start_time": self.start_time.isoformat(),
                "total_processing_time_ms": self.metrics.total_processing_time_ms,
                "prediction_cycles_completed": self.metrics.prediction_cycles_completed
            }
        )
        
    def _calculate_accuracy(self) -> float:
        """Calculate overall prediction accuracy."""
        if not self.metrics.total_predictions:
            return 0.0
        return self.metrics.successful_predictions / self.metrics.total_predictions
        
    def _get_recent_predictions(self) -> List[Dict[str, Any]]:
        """Get most recent predictions."""
        sorted_predictions = sorted(
            self.predictions,
            key=lambda x: x.prediction_timestamp,
            reverse=True
        )
        return [pred.model_dump() for pred in sorted_predictions[:5]]
        
    def _calculate_performance_trend(self) -> str:
        """Calculate the trend in prediction performance."""
        if len(self.performance_history) < 2:
            return "STABLE"
            
        recent_scores = [p.performance_score for p in self.performance_history[-5:]]
        if not recent_scores:
            return "STABLE"
            
        recent_avg = sum(recent_scores) / len(recent_scores)
        overall_avg = self._calculate_accuracy()
        
        if recent_avg > overall_avg * 1.1:
            return "IMPROVING"
        elif recent_avg < overall_avg * 0.9:
            return "DECLINING"
        return "STABLE"
        
    def _calculate_next_cycle(self) -> datetime:
        """Calculate the next prediction cycle timestamp."""
        from datetime import timedelta
        return datetime.now() + timedelta(minutes=15)  # Default to 15-minute cycles


# API compatibility functions
def get_ai_predictions_manager(config: Optional[PredictionConfigV2_5] = None) -> AIPredictionsManagerV2_5:
    """Get AI predictions manager instance."""
    return AIPredictionsManagerV2_5(config)

def generate_market_prediction(market_data: Dict[str, Any], config: Optional[PredictionConfigV2_5] = None) -> Optional[MarketPredictionV2_5]:
    """Generate a market prediction."""
    manager = get_ai_predictions_manager(config)
    return manager.generate_prediction(market_data) 