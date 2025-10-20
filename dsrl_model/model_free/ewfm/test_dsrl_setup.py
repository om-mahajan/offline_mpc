"""
Test script to verify DSRL integration is working correctly.
Run this before training to check if all dependencies are installed.
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch not found: {e}")
        return False
    
    try:
        import dsrl
        print(f"✓ DSRL")
    except ImportError as e:
        print(f"✗ DSRL not found: {e}")
        print("  Install with: pip install dsrl")
        return False
    
    try:
        import safety_gymnasium
        print(f"✓ Safety Gymnasium")
    except ImportError as e:
        print(f"✗ Safety Gymnasium not found: {e}")
        print("  Install with: pip install safety-gymnasium")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy not found: {e}")
        return False
    
    return True


def test_dsrl_dataset():
    """Test if DSRL dataset can be loaded"""
    print("\nTesting DSRL dataset loading...")
    try:
        from dsrl_adapter import DSRLSafetyDataset
        
        # Try to load a small dataset
        print("Loading OfflinePointGoal1Gymnasium-v0 dataset...")
        dataset = DSRLSafetyDataset(
            env_name='OfflinePointGoal1Gymnasium-v0',
            reward_quantile=0.75,
            cost_quantile=0.25,
            device='cpu'
        )
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Total transitions: {len(dataset)}")
        print(f"  - Observation dim: {dataset.obs_dim}")
        print(f"  - Action dim: {dataset.action_dim}")
        return True
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment():
    """Test if DSRL environment can be created"""
    print("\nTesting environment creation...")
    try:
        import safety_gymnasium
        env = safety_gymnasium.make('OfflinePointGoal1Gymnasium-v0')
        print(f"✓ Environment created successfully!")
        print(f"  - Observation space: {env.observation_space.shape}")
        print(f"  - Action space: {env.action_space.shape}")
        env.close()
        return True
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("DSRL Integration Test Suite")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    if results[-1][1]:  # Only continue if imports work
        # Test dataset loading
        results.append(("Dataset Loading", test_dsrl_dataset()))
        
        # Test environment
        results.append(("Environment Creation", test_environment()))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n✓ All tests passed! You're ready to train on DSRL datasets.")
        print("\nRun training with:")
        print('  python train_rl.py --env "OfflinePointGoal1Gymnasium-v0" --expid test --schedule OT')
    else:
        print("\n✗ Some tests failed. Please fix the issues above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
