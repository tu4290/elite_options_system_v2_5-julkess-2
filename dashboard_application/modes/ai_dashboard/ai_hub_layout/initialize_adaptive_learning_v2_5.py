"""
Initialize Adaptive Learning System for EOTS v2.5
==================================================

This script initializes the Pydantic AI-powered self-learning system
and integrates it with your existing EOTS infrastructure.

Run this once to set up the adaptive learning capabilities.

Author: EOTS v2.5 Development Team
"""

import logging
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core_analytics_engine.adaptive_learning_integration_v2_5 import AdaptiveLearningIntegrationV2_5
from data_management.database_manager_v2_5 import DatabaseManagerV2_5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def initialize_adaptive_learning():
    """Initialize the adaptive learning system."""
    try:
        logger.info("🚀 Initializing EOTS v2.5 Adaptive Learning System")
        
        # 1. Initialize database manager (use your existing one)
        logger.info("📊 Connecting to database...")
        db_manager = DatabaseManagerV2_5()
        
        # 2. Initialize config manager (placeholder - use your existing one)
        config_manager = None  # Replace with your actual config manager
        
        # 3. Initialize adaptive learning system
        logger.info("🧠 Setting up adaptive learning integration...")
        learning_system = AdaptiveLearningIntegrationV2_5(db_manager, config_manager)
        
        # 4. Initialize the system
        success = learning_system.initialize_adaptive_learning()
        
        if success:
            logger.info("✅ Adaptive Learning System initialized successfully!")
            
            # 5. Run initial learning cycle to test the system
            logger.info("🔄 Running initial learning cycle...")
            initial_result = await learning_system.run_daily_learning_cycle("SPY")
            
            logger.info(f"📈 Initial learning cycle completed:")
            logger.info(f"   - Insights generated: {len(initial_result.insights_generated)}")
            logger.info(f"   - Parameters updated: {len(initial_result.parameters_updated)}")
            logger.info(f"   - Confidence score: {initial_result.confidence_score:.2f}")
            logger.info(f"   - Summary: {initial_result.learning_summary}")
            
            # 6. Display system status
            status = learning_system.get_learning_status()
            logger.info("📊 Learning System Status:")
            logger.info(f"   - System active: {status['learning_system_active']}")
            logger.info(f"   - Recent cycles: {len(status['recent_learning_cycles'])}")
            logger.info(f"   - Recent updates: {len(status['recent_parameter_updates'])}")
            
            return learning_system
            
        else:
            logger.error("❌ Failed to initialize adaptive learning system")
            return None
            
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return None

async def test_learning_capabilities(learning_system):
    """Test the learning system capabilities."""
    try:
        logger.info("🧪 Testing learning system capabilities...")
        
        # Test daily learning
        logger.info("📅 Testing daily learning cycle...")
        daily_result = await learning_system.run_daily_learning_cycle("SPY")
        logger.info(f"   ✅ Daily learning: {len(daily_result.insights_generated)} insights")
        
        # Test weekly learning
        logger.info("📊 Testing weekly deep learning...")
        weekly_result = await learning_system.run_weekly_deep_learning("SPY")
        logger.info(f"   ✅ Weekly learning: {len(weekly_result.insights_generated)} insights")
        
        # Test monthly review (commented out for quick testing)
        # logger.info("🔍 Testing monthly comprehensive review...")
        # monthly_result = await learning_system.run_monthly_comprehensive_review("SPY")
        # logger.info(f"   ✅ Monthly review: {monthly_result.get('system_health_score', 'N/A')}")
        
        logger.info("✅ All learning capabilities tested successfully!")
        
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")

def create_learning_config():
    """Verify learning configuration exists in consolidated pydantic AI config."""
    try:
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)

        config_path = config_dir / "pydantic_ai_config.json"

        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)

            if 'adaptive_learning_settings' in config:
                logger.info("✅ Adaptive learning settings found in consolidated pydantic AI config")
                return True
            else:
                logger.warning("⚠️ adaptive_learning_settings missing from pydantic AI config")
                return False
        else:
            logger.error("❌ Pydantic AI config file not found")
            return False
            
            default_config = {
                "daily_learning_enabled": True,
                "weekly_deep_learning_enabled": True,
                "monthly_comprehensive_review_enabled": True,
                "real_time_adaptation_enabled": True,
                "min_confidence_for_auto_update": 0.8,
                "max_parameter_change_per_cycle": 0.15,
                "learning_history_retention_days": 90,
                "learning_schedule": {
                    "daily_time": "02:00",
                    "weekly_day": "sunday",
                    "weekly_time": "03:00",
                    "monthly_day": 1
                },
                "parameter_constraints": {
                    "ai_confidence_threshold": {"min": 0.3, "max": 0.95},
                    "vapi_fa_threshold": {"min": 0.5, "max": 3.0},
                    "dwfd_threshold": {"min": 0.5, "max": 3.0},
                    "tw_laf_threshold": {"min": 0.5, "max": 3.0},
                    "regime_confidence_threshold": {"min": 0.4, "max": 0.9}
                },
                "notification_settings": {
                    "email_notifications": False,
                    "log_level": "INFO",
                    "alert_on_major_changes": True
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            logger.info(f"📋 Created learning configuration: {config_path}")
        else:
            logger.info(f"📋 Learning configuration already exists: {config_path}")
            
    except Exception as e:
        logger.error(f"Failed to create learning config: {e}")

async def main():
    """Main initialization function."""
    try:
        logger.info("=" * 60)
        logger.info("🧠 EOTS v2.5 Adaptive Learning System Initialization")
        logger.info("=" * 60)
        
        # Create configuration
        create_learning_config()
        
        # Initialize learning system
        learning_system = await initialize_adaptive_learning()
        
        if learning_system:
            # Test capabilities
            await test_learning_capabilities(learning_system)
            
            logger.info("=" * 60)
            logger.info("🎉 Adaptive Learning System is ready!")
            logger.info("=" * 60)
            logger.info("📋 Next Steps:")
            logger.info("   1. The system will run automatic learning cycles:")
            logger.info("      - Daily: Every day at 02:00")
            logger.info("      - Weekly: Every Sunday at 03:00")
            logger.info("      - Monthly: First day of each month")
            logger.info("   2. Monitor learning progress in the database tables:")
            logger.info("      - learning_cycle_results")
            logger.info("      - parameter_updates")
            logger.info("      - learning_performance_tracking")
            logger.info("   3. Check logs for learning insights and parameter updates")
            logger.info("   4. The system will automatically improve over time!")
            logger.info("=" * 60)
            
            # Keep the system running for demonstration
            logger.info("🔄 System is now running. Press Ctrl+C to stop.")
            try:
                while True:
                    await asyncio.sleep(60)  # Keep running
            except KeyboardInterrupt:
                logger.info("🛑 Stopping adaptive learning system...")
                learning_system.stop_learning_scheduler()
                logger.info("✅ System stopped gracefully")
        
        else:
            logger.error("❌ Failed to initialize learning system")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Main execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the initialization
    asyncio.run(main())
