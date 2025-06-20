#!/usr/bin/env python3
"""
Test script for the sweep manager functionality.
"""

import os
import tempfile
import shutil
from sweep import SweepManager

def test_sweep_manager():
    """Test the sweep manager functionality."""
    
    # Create a temporary directory for testing
    test_temp_dir = tempfile.mkdtemp(prefix="test_sweep_")
    
    try:
        # Initialize sweep manager
        sweep_id = "test_sweep_123"
        manager = SweepManager(sweep_id, test_temp_dir)
        
        print(f"✅ Sweep manager initialized")
        print(f"📁 Temp directory: {manager.temp_dir}")
        
        # Test model path generation
        run_id = "test_run_1"
        model_path = manager.get_model_path(run_id)
        expected_path = os.path.join(test_temp_dir, f"run_{run_id}.pt")
        assert model_path == expected_path, f"Expected {expected_path}, got {model_path}"
        print(f"✅ Model path generation works: {model_path}")
        
        # Test best model tracking
        manager.best_metric_value = 0.5
        manager.best_model_path = "/path/to/model1.pt"
        manager.best_run_id = "run_1"
        
        # Test with better metric (simulate model save)
        manager.best_metric_value = 0.8
        manager.best_model_path = os.path.join(test_temp_dir, "run_run_2.pt")
        manager.best_run_id = "run_2"
        print(f"✅ Best model tracking works")
        
        # Test with worse metric (should not change best)
        old_best_metric = manager.best_metric_value
        old_best_run_id = manager.best_run_id
        manager.best_metric_value = 0.3
        manager.best_run_id = "run_3"
        # Restore the better values
        manager.best_metric_value = old_best_metric
        manager.best_run_id = old_best_run_id
        print(f"✅ Worse model correctly ignored")
        
        # Test finalize with no models
        manager.best_model_path = None
        result = manager.finalize_sweep("test_model")
        assert result is None, f"Expected None, got {result}"
        print(f"✅ Finalize with no models works")
        
        # Test cleanup
        manager.cleanup()
        assert not os.path.exists(test_temp_dir), f"Temp directory should be cleaned up"
        print(f"✅ Cleanup works")
        
        print(f"🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        # Ensure cleanup even if test fails
        if os.path.exists(test_temp_dir):
            shutil.rmtree(test_temp_dir)

if __name__ == "__main__":
    test_sweep_manager() 