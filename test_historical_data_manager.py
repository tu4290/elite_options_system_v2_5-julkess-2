#!/usr/bin/env python3
"""
Test script to validate HistoricalDataManagerV2_5 integration with DatabaseManagerV2_5.
This script tests the fixes for DataFrame creation from cursor results.
"""

import sys
import os
from datetime import date, timedelta
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_management.historical_data_manager_v2_5 import HistoricalDataManagerV2_5
from data_management.database_manager_v2_5 import DatabaseManagerV2_5
from utils.config_manager_v2_5 import ConfigManagerV2_5

def test_historical_data_manager():
    """Test the HistoricalDataManagerV2_5 functionality."""
    print("🧪 Testing HistoricalDataManagerV2_5 Integration")
    print("=" * 50)
    
    try:
        # Initialize components
        print("📋 Initializing components...")
        config_manager = ConfigManagerV2_5()
        db_manager = DatabaseManagerV2_5(config_manager)
        historical_manager = HistoricalDataManagerV2_5(config_manager, db_manager)
        
        print("✅ Components initialized successfully")
        
        # Test 1: Test OHLCV data retrieval
        print("\n🔍 Test 1: Testing OHLCV data retrieval...")
        test_symbol = "AAPL"
        lookback_days = 30
        
        ohlcv_data = historical_manager.get_historical_ohlcv(test_symbol, lookback_days)
        
        if ohlcv_data is not None:
            print(f"✅ OHLCV data retrieved successfully for {test_symbol}")
            print(f"   - Shape: {ohlcv_data.shape}")
            print(f"   - Columns: {list(ohlcv_data.columns)}")
            print(f"   - Date range: {ohlcv_data['date'].min()} to {ohlcv_data['date'].max()}")
        else:
            print(f"⚠️ No OHLCV data found for {test_symbol} (this may be expected if no data exists)")
        
        # Test 2: Test metric data retrieval
        print("\n🔍 Test 2: Testing metric data retrieval...")
        lookback_days = 30
        
        # Test a common metric (adjust based on your actual metrics)
        test_metrics = ["volume", "close", "high", "low", "open"]
        
        for metric in test_metrics:
            try:
                metric_data = historical_manager.get_historical_metric(
                    symbol=test_symbol,
                    metric_name=metric,
                    lookback_days=lookback_days
                )
                
                if metric_data is not None:
                    print(f"✅ Metric '{metric}' retrieved successfully")
                    print(f"   - Data points: {len(metric_data)}")
                    print(f"   - Value range: {metric_data.min():.2f} to {metric_data.max():.2f}")
                else:
                    print(f"⚠️ No data found for metric '{metric}' (may be expected)")
                    
            except Exception as e:
                print(f"❌ Error retrieving metric '{metric}': {e}")
        
        # Test 3: Test database connection and schema
        print("\n🔍 Test 3: Testing database connection...")
        if db_manager._conn:
            print("✅ Database connection is active")
            
            # Test cursor description functionality
            try:
                with db_manager._conn.cursor() as cur:
                    # Try a simple query to test cursor.description
                    cur.execute("SELECT 1 as test_column, 'test_value' as test_string")
                    rows = cur.fetchall()
                    columns = [desc[0] for desc in cur.description]
                    print(f"✅ Cursor description working: columns = {columns}")
                    
            except Exception as e:
                print(f"❌ Cursor description test failed: {e}")
        else:
            print("❌ Database connection is not active")
        
        print("\n🎉 Historical Data Manager test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    success = test_historical_data_manager()
    sys.exit(0 if success else 1)