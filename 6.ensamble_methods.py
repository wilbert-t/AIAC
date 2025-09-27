#!/usr/bin/env python3
"""
Streamlined Ensemble Methods for Au Cluster Energy Prediction
Works with parsed data from task1 - no data loading needed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
import xgboost as xgb

warnings.filterwarnings('ignore')

# Optional libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class EnsembleAnalyzer:
    """
    Streamlined Ensemble Methods for Au Cluster Analysis
    Assumes you have parsed data from task1
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_models = {}
        self.results = {}
    
    def create_base_models(self):
        """Create diverse base models"""
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'ridge': Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=1.0))
            ]),
            'svr': Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='rbf', C=1.0))
            ]),
            'mlp': Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, 
                                   random_state=self.random_state))
            ])
        }
        
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(n_estimators=100, random_state=self.random_state)
        
        return models
    
    class StackingEnsemble(BaseEstimator, RegressorMixin):
        """Stacking ensemble with proper cross-validation"""
        
        def __init__(self, base_models, meta_learner=None, cv=5, random_state=42):
            self.base_models = base_models
            self.meta_learner = meta_learner or Ridge(alpha=1.0)
            self.cv = cv
            self.random_state = random_state
            self.trained_models_ = {}
        
        def fit(self, X, y):
            # Generate meta-features using cross-validation
            kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            meta_features = np.zeros((len(X), len(self.base_models)))
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
                X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
                y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                
                for i, (name, model) in enumerate(self.base_models.items()):
                    fold_model = clone(model)
                    fold_model.fit(X_train_fold, y_train_fold)
                    val_pred = fold_model.predict(X_val_fold)
                    meta_features[val_idx, i] = val_pred
            
            # Train meta-learner
            self.meta_learner.fit(meta_features, y)
            
            # Train base models on full data
            for name, model in self.base_models.items():
                final_model = clone(model)
                final_model.fit(X, y)
                self.trained_models_[name] = final_model
            
            return self
        
        def predict(self, X):
            # Get base predictions
            base_preds = np.zeros((len(X), len(self.trained_models_)))
            for i, model in enumerate(self.trained_models_.values()):
                base_preds[:, i] = model.predict(X)
            
            # Meta-learner prediction
            return self.meta_learner.predict(base_preds)
    
    def simple_voting_ensemble(self, models, X):
        """Simple averaging ensemble"""
        predictions = []
        for model in models.values():
            predictions.append(model.predict(X))
        return np.mean(predictions, axis=0)
    
    def weighted_voting_ensemble(self, models, weights, X):
        """Weighted averaging ensemble"""
        predictions = []
        for model in models.values():
            predictions.append(model.predict(X))
        
        weighted_preds = np.average(predictions, axis=0, weights=weights)
        return weighted_preds
    
    def run_ensemble_analysis(self, X, y, target_name='energy'):
        """Run complete ensemble analysis"""
        print("Running Ensemble Analysis for Au Cluster Energy Prediction")
        print("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Create and train base models
        print("\nTraining base models...")
        base_models = self.create_base_models()
        trained_models = {}
        individual_scores = {}
        
        for name, model in base_models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                trained_models[name] = model
                individual_scores[name] = {'r2': score, 'rmse': rmse}
                print(f"  {name}: R² = {score:.3f}, RMSE = {rmse:.3f}")
            except Exception as e:
                print(f"  Failed to train {name}: {e}")
        
        if len(trained_models) < 2:
            print("Need at least 2 base models for ensemble")
            return {}
        
        # Ensemble methods
        print("\nTraining ensemble methods...")
        results = {}
        
        # 1. Simple Voting
        try:
            voting_pred = self.simple_voting_ensemble(trained_models, X_test)
            voting_r2 = r2_score(y_test, voting_pred)
            voting_rmse = np.sqrt(mean_squared_error(y_test, voting_pred))
            
            results['simple_voting'] = {
                'r2': voting_r2,
                'rmse': voting_rmse,
                'predictions': voting_pred,
                'description': 'Equal weight averaging'
            }
            print(f"  Simple Voting: R² = {voting_r2:.3f}, RMSE = {voting_rmse:.3f}")
        except Exception as e:
            print(f"  Simple voting failed: {e}")
        
        # 2. Weighted Voting (by performance)
        try:
            weights = [individual_scores[name]['r2'] for name in trained_models.keys()]
            weights = np.array(weights) / sum(weights)  # Normalize
            
            weighted_pred = self.weighted_voting_ensemble(trained_models, weights, X_test)
            weighted_r2 = r2_score(y_test, weighted_pred)
            weighted_rmse = np.sqrt(mean_squared_error(y_test, weighted_pred))
            
            results['weighted_voting'] = {
                'r2': weighted_r2,
                'rmse': weighted_rmse,
                'predictions': weighted_pred,
                'weights': dict(zip(trained_models.keys(), weights)),
                'description': 'Performance-weighted averaging'
            }
            print(f"  Weighted Voting: R² = {weighted_r2:.3f}, RMSE = {weighted_rmse:.3f}")
        except Exception as e:
            print(f"  Weighted voting failed: {e}")
        
        # 3. Stacking
        try:
            stacking = self.StackingEnsemble(trained_models, random_state=self.random_state)
            stacking.fit(X_train, y_train)
            stacking_pred = stacking.predict(X_test)
            stacking_r2 = r2_score(y_test, stacking_pred)
            stacking_rmse = np.sqrt(mean_squared_error(y_test, stacking_pred))
            
            results['stacking'] = {
                'r2': stacking_r2,
                'rmse': stacking_rmse,
                'predictions': stacking_pred,
                'model': stacking,
                'description': 'Meta-learner optimized combination'
            }
            print(f"  Stacking: R² = {stacking_r2:.3f}, RMSE = {stacking_rmse:.3f}")
        except Exception as e:
            print(f"  Stacking failed: {e}")
        
        # Store test data for plotting
        self.X_test = X_test
        self.y_test = y_test
        self.individual_scores = individual_scores
        self.results = results
        
        return results
    
    def analyze_results(self):
        """Analyze and summarize ensemble results"""
        if not self.results:
            print("No results to analyze")
            return
        
        print("\n" + "=" * 60)
        print("ENSEMBLE RESULTS ANALYSIS")
        print("=" * 60)
        
        # Sort by R²
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['r2'], reverse=True)
        
        print("\nPerformance Ranking:")
        for i, (name, result) in enumerate(sorted_results, 1):
            print(f"{i}. {name.upper():<18} | R² = {result['r2']:.3f} | "
                  f"RMSE = {result['rmse']:.3f} | {result['description']}")
        
        # Best individual model vs best ensemble
        best_individual = max(self.individual_scores.items(), key=lambda x: x[1]['r2'])
        best_ensemble = sorted_results[0]
        
        print(f"\nBest Individual Model: {best_individual[0]} (R² = {best_individual[1]['r2']:.3f})")
        print(f"Best Ensemble Method: {best_ensemble[0]} (R² = {best_ensemble[1]['r2']:.3f})")
        
        improvement = best_ensemble[1]['r2'] - best_individual[1]['r2']
        print(f"Ensemble Improvement: {improvement:.3f} R² points")
        
        return sorted_results
    
    def create_visualizations(self, output_dir='./ensemble_results'):
        """Create ensemble visualization plots"""
        if not self.results:
            print("No results to visualize")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 1. Performance comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Include individual models
        all_methods = list(self.individual_scores.keys()) + list(self.results.keys())
        all_r2 = ([self.individual_scores[k]['r2'] for k in self.individual_scores.keys()] + 
                  [self.results[k]['r2'] for k in self.results.keys()])
        all_rmse = ([self.individual_scores[k]['rmse'] for k in self.individual_scores.keys()] + 
                    [self.results[k]['rmse'] for k in self.results.keys()])
        
        # Color individual vs ensemble differently
        colors = ['lightblue'] * len(self.individual_scores) + ['orange'] * len(self.results)
        
        # R² plot
        bars1 = axes[0].bar(all_methods, all_r2, color=colors, alpha=0.8)
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('Individual Models vs Ensemble Methods (R²)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Add legend
        axes[0].legend([bars1[0], bars1[len(self.individual_scores)]], 
                      ['Individual Models', 'Ensemble Methods'])
        
        # RMSE plot
        bars2 = axes[1].bar(all_methods, all_rmse, color=colors, alpha=0.8)
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Individual Models vs Ensemble Methods (RMSE)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ensemble_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Best ensemble predictions
        best_method = max(self.results.keys(), key=lambda k: self.results[k]['r2'])
        best_predictions = self.results[best_method]['predictions']
        
        plt.figure(figsize=(8, 8))
        plt.scatter(self.y_test, best_predictions, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), best_predictions.min())
        max_val = max(self.y_test.max(), best_predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        plt.xlabel('Actual Energy (eV)')
        plt.ylabel('Predicted Energy (eV)')
        plt.title(f'Best Ensemble Predictions: {best_method.upper()}')
        plt.grid(True, alpha=0.3)
        
        # Add metrics
        r2 = self.results[best_method]['r2']
        rmse = self.results[best_method]['rmse']
        plt.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}', 
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'best_ensemble_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    
    def save_results(self, output_dir='./ensemble_results'):
        """Save ensemble results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create summary DataFrame
        summary_data = []
        
        # Individual models
        for name, scores in self.individual_scores.items():
            summary_data.append({
                'method': name,
                'type': 'individual',
                'r2_score': scores['r2'],
                'rmse': scores['rmse']
            })
        
        # Ensemble methods
        for name, result in self.results.items():
            summary_data.append({
                'method': name,
                'type': 'ensemble',
                'r2_score': result['r2'],
                'rmse': result['rmse']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('r2_score', ascending=False)
        summary_df.to_csv(output_dir / 'ensemble_summary.csv', index=False)
        
        print(f"Results saved to {output_dir}")
        return summary_df

# Example usage function
def run_ensemble_analysis_on_data(X, y, target_name='energy'):
    """
    Main function to run ensemble analysis on your parsed data
    
    Parameters:
    X: Feature matrix (pandas DataFrame or numpy array)
    y: Target values (pandas Series or numpy array)
    target_name: Name of target variable for labeling
    
    Returns:
    analyzer: Fitted EnsembleAnalyzer object
    results: Dictionary of ensemble results
    """
    
    analyzer = EnsembleAnalyzer(random_state=42)
    
    # Run analysis
    results = analyzer.run_ensemble_analysis(X, y, target_name)
    
    if results:
        # Analyze results
        analyzer.analyze_results()
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Save results
        summary_df = analyzer.save_results()
        
        print("\nEnsemble Analysis Complete!")
        print("\nKey Insights:")
        print("- Ensemble methods often outperform individual models")
        print("- Stacking learns optimal model combinations")
        print("- Weighted voting balances model strengths")
        print("- Simple voting provides robust baseline")
        
        return analyzer, results
    else:
        print("No ensemble results generated")
        return analyzer, {}

if __name__ == "__main__":
    print("Ensemble Analyzer Module")
    print("Use run_ensemble_analysis_on_data(X, y) with your parsed data from task1")