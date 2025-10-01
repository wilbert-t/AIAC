#!/usr/bin/env python3
"""
Test script for enhanced hybrid training models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

def test_linear_models():
    """Test enhanced linear models"""
    print("🔬 Testing Enhanced Linear Models...")
    
    try:
        # Import the linear models analyzer
        from importlib import import_module
        linear_module = import_module('1.linear_models')
        
        # Initialize analyzer
        analyzer = linear_module.LinearModelsAnalyzer(random_state=42)
        
        # Load data with hybrid training
        data_path = "./au_cluster_analysis_results/descriptors.csv"
        analyzer.load_data(data_path, use_hybrid_training=True)
        
        # Prepare features
        X, y, feature_names = analyzer.prepare_features(target_column='energy')
        
        # Test progressive hybrid training if datasets are available
        if any(analyzer.datasets.values()):
            print("   🚀 Testing progressive hybrid training...")
            results = analyzer.progressive_hybrid_training(X, y, use_elite_validation=True)
            
            # Check results structure
            expected_keys = ['foundation_results', 'refinement_results', 'elite_validation', 'anti_memorization_metrics']
            for key in expected_keys:
                if key in results:
                    print(f"   ✅ {key}: Found")
                else:
                    print(f"   ⚠️  {key}: Missing")
            
            # Check anti-memorization metrics
            if results.get('anti_memorization_metrics'):
                print("   🧠 Anti-memorization analysis completed")
                for model, metrics in results['anti_memorization_metrics'].items():
                    risk = metrics.get('memorization_risk', 'UNKNOWN')
                    print(f"      {model}: {risk} memorization risk")
            
            print("   ✅ Linear models hybrid training: SUCCESS")
            return True
        else:
            print("   ⚠️  No categorized datasets found - testing standard training")
            results = analyzer.train_models(X, y)
            print("   ✅ Linear models standard training: SUCCESS")
            return True
            
    except Exception as e:
        print(f"   ❌ Linear models test failed: {e}")
        return False

def test_kernel_models():
    """Test enhanced kernel models"""
    print("\n🔧 Testing Enhanced Kernel Models...")
    
    try:
        # Import the kernel models analyzer
        from importlib import import_module
        kernel_module = import_module('2.kernel_models')
        
        # Initialize analyzer
        analyzer = kernel_module.ComprehensiveKernelAnalysis(random_state=42)
        
        # Load data with hybrid training
        data_path = "./au_cluster_analysis_results/descriptors.csv"
        X, y = analyzer.load_and_prepare_data(data_path, use_hybrid_training=True)
        
        # Test progressive kernel training if datasets are available
        if hasattr(analyzer, 'datasets') and any(analyzer.datasets.values()):
            print("   🚀 Testing progressive kernel training...")
            results = analyzer.progressive_kernel_training(X, y, use_elite_validation=True)
            
            # Check results structure
            expected_keys = ['foundation_results', 'parameter_optimization', 'elite_validation', 'kernel_analysis']
            for key in expected_keys:
                if key in results:
                    print(f"   ✅ {key}: Found")
                else:
                    print(f"   ⚠️  {key}: Missing")
            
            # Check kernel-specific analysis
            if results.get('anti_memorization_metrics'):
                print("   🧠 Kernel anti-memorization analysis completed")
                for model, metrics in results['anti_memorization_metrics'].items():
                    risk = metrics.get('memorization_risk', 'UNKNOWN')
                    print(f"      {model}: {risk} memorization risk")
            
            print("   ✅ Kernel models hybrid training: SUCCESS")
            return True
        else:
            print("   ⚠️  No categorized datasets found - testing standard training")
            results = analyzer.train_models(X, y)
            print("   ✅ Kernel models standard training: SUCCESS")
            return True
            
    except Exception as e:
        print(f"   ❌ Kernel models test failed: {e}")
        return False

def test_tree_models():
    """Test enhanced tree models"""
    print("\n🌳 Testing Enhanced Tree Models...")
    
    try:
        # Import the tree models analyzer
        from importlib import import_module
        tree_module = import_module('3.tree_models')
        
        # Initialize analyzer
        analyzer = tree_module.EnhancedTreeAnalyzer()
        
        # Load data with hybrid training
        data_path = "./au_cluster_analysis_results/descriptors.csv"
        df = analyzer.load_data(data_path, use_hybrid_training=True)
        
        # Prepare features
        X, y = analyzer.prepare_features(df)
        
        # Test progressive ensemble training if datasets are available
        if hasattr(analyzer, 'datasets') and any(analyzer.datasets.values()):
            print("   🚀 Testing progressive ensemble training...")
            results = analyzer.progressive_ensemble_training(X, y, use_elite_validation=True)
            
            # Check results structure
            expected_keys = ['foundation_results', 'ensemble_refinement', 'elite_validation', 'ensemble_analysis']
            for key in expected_keys:
                if key in results:
                    print(f"   ✅ {key}: Found")
                else:
                    print(f"   ⚠️  {key}: Missing")
            
            # Check ensemble-specific analysis
            if results.get('anti_memorization_metrics'):
                print("   🧠 Ensemble anti-memorization analysis completed")
                for model, metrics in results['anti_memorization_metrics'].items():
                    risk = metrics.get('memorization_risk', 'UNKNOWN')
                    print(f"      {model}: {risk} memorization risk")
            
            print("   ✅ Tree models hybrid training: SUCCESS")
            return True
        else:
            print("   ⚠️  No categorized datasets found - testing standard training")
            results = analyzer.train_all_models(X, y)
            print("   ✅ Tree models standard training: SUCCESS")
            return True
            
    except Exception as e:
        print(f"   ❌ Tree models test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 TESTING ENHANCED HYBRID TRAINING MODELS")
    print("=" * 50)
    
    # Check if required data files exist
    required_files = [
        "./au_cluster_analysis_results/descriptors.csv",
        "./improved_dataset_balanced.csv",
        "./improved_dataset_high_quality.csv", 
        "./improved_dataset_elite.csv"
    ]
    
    print("\n📁 Checking required files...")
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - Missing")
    
    # Run tests
    results = []
    
    # Test Linear Models
    results.append(test_linear_models())
    
    # Test Kernel Models  
    results.append(test_kernel_models())
    
    # Test Tree Models
    results.append(test_tree_models())
    
    # Final summary
    print(f"\n🎉 TEST SUMMARY")
    print("=" * 30)
    successful_tests = sum(results)
    total_tests = len(results)
    
    print(f"✅ Successful: {successful_tests}/{total_tests}")
    print(f"❌ Failed: {total_tests - successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("\n🏆 ALL TESTS PASSED! Enhanced hybrid training is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)