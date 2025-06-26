import time
from datetime import datetime, time as dtime
import logging
import shutil
from pathlib import Path
import json

from utils.config_manager_v2_5 import ConfigManagerV2_5
from data_management.database_manager_v2_5 import DatabaseManagerV2_5
from data_management.historical_data_manager_v2_5 import HistoricalDataManagerV2_5
from data_management.performance_tracker_v2_5 import PerformanceTrackerV2_5
from core_analytics_engine.metrics_calculator_v2_5 import MetricsCalculatorV2_5
from data_management.initial_processor_v2_5 import InitialDataProcessorV2_5
from core_analytics_engine.market_regime_engine_v2_5 import MarketRegimeEngineV2_5
from core_analytics_engine.market_intelligence_engine_v2_5 import MarketIntelligenceEngineV2_5
from core_analytics_engine.atif_engine_v2_5 import ATIFEngineV2_5
from core_analytics_engine.its_orchestrator_v2_5 import ITSOrchestratorV2_5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IntradayCollector")

def is_market_open():
    now = datetime.now().time()
    return dtime(9, 30) <= now <= dtime(16, 0)

def build_orchestrator():
    config_manager = ConfigManagerV2_5()
    db_manager = DatabaseManagerV2_5(config_manager)
    historical_data_manager = HistoricalDataManagerV2_5(config_manager, db_manager)
    performance_tracker = PerformanceTrackerV2_5(config_manager)
    metrics_calculator = MetricsCalculatorV2_5(config_manager, historical_data_manager)
    initial_processor = InitialDataProcessorV2_5(config_manager, metrics_calculator)
    
    # Initialize market regime engine with config manager
    market_regime_engine = MarketRegimeEngineV2_5(config_manager)
    
    # Initialize market intelligence engine with metrics calculator
    market_intelligence_engine = MarketIntelligenceEngineV2_5(
        config_manager=config_manager,
        metrics_calculator=metrics_calculator
    )
    
    # Initialize ATIF engine with config manager
    adaptive_trade_idea_framework = ATIFEngineV2_5(config_manager=config_manager)
    
    # Initialize orchestrator with config manager only (db_manager is created internally)
    orchestrator = ITSOrchestratorV2_5(config_manager=config_manager)
    
    return orchestrator

def sanitize_symbol(symbol: str) -> str:
    """
    Sanitize a ticker symbol for safe use in file paths and cache keys.
    Replaces '/' and ':' with '_'.
    """
    return symbol.replace('/', '_').replace(':', '_')

def main():
    config_manager = ConfigManagerV2_5()
    collector_cfg = config_manager.config.intraday_collector_settings
    watched_tickers = collector_cfg.watched_tickers
    gauge_metrics = collector_cfg.metrics
    cache_dir = Path(collector_cfg.cache_dir)
    collection_interval = collector_cfg.collection_interval_seconds
    market_open = collector_cfg.market_open_time
    market_close = collector_cfg.market_close_time
    reset_at_eod = collector_cfg.reset_at_eod
    calibration_threshold = getattr(collector_cfg, 'calibration_threshold', 25)
    calibrated_pairs = set()

    # Initialize enhanced cache manager
    from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5, CacheLevel
    enhanced_cache = EnhancedCacheManagerV2_5(
        cache_root="cache/enhanced_v2_5",
        memory_limit_mb=50,
        disk_limit_mb=500,
        default_ttl_seconds=86400  # 24 hours for intraday data
    )
    logger.info("✨ Enhanced Cache Manager initialized for intraday collection")

    # Enhanced Cache is already initialized above - no need for Redis
    # This eliminates Redis limits and network overhead for better performance
    logger.info("✅ Enhanced Cache enabled for intraday collector - no Redis limits!")

    logger.info(f"Loaded intraday collector config: {collector_cfg}")

    def clear_intraday_cache():
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Intraday cache cleared for new trading day.")
        calibrated_pairs.clear()

    orchestrator = build_orchestrator()
    last_reset_date = None

    try:
        while True:
            today = datetime.now().date()
            if last_reset_date != today:
                clear_intraday_cache()
                last_reset_date = today
            if is_market_open():
                for symbol in watched_tickers:
                    try:
                        logger.info(f"Processing {symbol}...")
                        # Handle async orchestrator call properly
                        import asyncio
                        bundle = asyncio.run(orchestrator.run_full_analysis_cycle(symbol))
                        logger.info(f"Completed {symbol}.")
                        # 🚀 PYDANTIC-FIRST: Extract enriched underlying data as Pydantic model
                        und_data = None
                        if bundle and hasattr(bundle, 'processed_data_bundle') and bundle.processed_data_bundle is not None:
                            und_data = getattr(bundle.processed_data_bundle, 'underlying_data_enriched', None)
                        if not und_data or not hasattr(und_data, 'model_dump'):
                            logger.warning(f"No valid Pydantic underlying data for {symbol}, skipping metric collection.")
                            continue
                        today = datetime.now().strftime('%Y-%m-%d')
                        for metric in gauge_metrics:
                            # 🚀 PYDANTIC-FIRST: Extract value directly from Pydantic model
                            value = getattr(und_data, metric, None)
                            if value is None:
                                logger.warning(f"Metric {metric} missing for {symbol}.")
                                continue
                            # Always store as a list for histories/arrays, wrap scalars
                            if isinstance(value, (list, tuple)):
                                values = list(value)
                            elif isinstance(value, dict):
                                # For dicts (e.g., rolling_flows, greek_flows, flow_ratios), store as-is
                                values = value
                            else:
                                values = [value]
                            safe_symbol = sanitize_symbol(symbol)

                            # Use enhanced cache as primary storage
                            try:
                                # Determine cache level based on data size
                                data_size_mb = len(str(values)) / (1024 * 1024)
                                cache_level = CacheLevel.COMPRESSED if data_size_mb > 0.5 else CacheLevel.MEMORY

                                success = enhanced_cache.put(
                                    symbol=symbol,
                                    metric_name=metric,
                                    data=values,
                                    cache_level=cache_level,
                                    tags=[f"intraday_{today}", "collector", safe_symbol]
                                )

                                # Enhanced Cache already stores the data above - no need for Redis
                                # This eliminates Redis limits and provides better performance
                                logger.debug(f"✅ PYDANTIC-FIRST: Stored {symbol} {metric} in Enhanced Cache for real-time access")

                                if success:
                                    sample_count = len(values) if isinstance(values, list) else (len(values) if hasattr(values, '__len__') else 1)
                                    pair_key = (safe_symbol, metric)
                                    if sample_count >= calibration_threshold and pair_key not in calibrated_pairs:
                                        logger.info(f"Metric for {symbol} ({metric}) is now fully calibrated with {sample_count} samples.")
                                        calibrated_pairs.add(pair_key)
                                else:
                                    logger.warning(f"Enhanced cache storage failed for {symbol} {metric}")

                            except Exception as e:
                                logger.warning(f"Enhanced cache error for {symbol} {metric}: {e}")

                                # 🚀 PYDANTIC-FIRST: Fallback to legacy cache with Pydantic model
                                cache_file = cache_dir / f"{safe_symbol}_{metric}_{today}.json"
                                try:
                                    from data_models.eots_schemas_v2_5 import IntradayMetricDataV2_5
                                    cache_data = IntradayMetricDataV2_5(
                                        values=values,
                                        last_updated=datetime.now(),
                                        sample_count=len(values) if isinstance(values, list) else 1
                                    )
                                    with open(cache_file, 'w') as f:
                                        json.dump(cache_data.model_dump(), f)  # 🚀 PYDANTIC-FIRST: Use model_dump()
                                    logger.debug(f"Stored {symbol} {metric} in legacy cache as fallback")
                                except Exception as fallback_e:
                                    logger.error(f"Both enhanced and legacy cache failed for {symbol} {metric}: {fallback_e}")
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                try:
                    time.sleep(collection_interval)
                except KeyboardInterrupt:
                    logger.info("🛑 Received interrupt signal during market hours sleep. Shutting down gracefully...")
                    break
            else:
                logger.info("Market closed. Sleeping until next check.")
                try:
                    time.sleep(60 * 10)  # Sleep 10 minutes when market is closed
                except KeyboardInterrupt:
                    logger.info("🛑 Received interrupt signal during market closed sleep. Shutting down gracefully...")
                    break
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal. Shutting down gracefully...")
    except Exception as e:
        logger.error(f"💥 Unexpected error in main loop: {e}")
    finally:
        logger.info("🏁 Intraday collector shutdown complete.")

if __name__ == "__main__":
    main()