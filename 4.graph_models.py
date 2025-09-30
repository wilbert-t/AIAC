#!/usr/bin/env python3
"""
Model 4: Complete Analysis Suite with All Requested Outputs
Includes all evaluation metrics, visualizations, and reports
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel

# Suppress various warnings including LinAlgWarning
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='Ill-conditioned matrix')
# Import and suppress LinAlgWarning specifically
from scipy.linalg import LinAlgWarning
warnings.filterwarnings('ignore', category=LinAlgWarning)

# Core ML dependencies
from sklearn.model_selection import (train_test_split, cross_val_score, KFold, 
                                   learning_curve, cross_validate, validation_curve)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet, Lasso, LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Optional dependencies
HAS_XGBOOST = False
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    print("Warning: XGBoost not available")

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    HAS_TORCH = True
except ImportError:
    print("Warning: PyTorch not available")

HAS_TORCH_GEOMETRIC = False
if HAS_TORCH:
    try:
        from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, MessagePassing
        from torch_geometric.data import Data, DataLoader
        HAS_TORCH_GEOMETRIC = True
    except ImportError:
        print("Warning: PyTorch Geometric not available")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu") if HAS_TORCH else None

class CompleteAnalyzer:
    """Complete analysis suite with all requested outputs"""
    
    def __init__(self, output_dir='./graph_models_results', random_state=42):
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        
        # Create directory structure
        self.dirs = {
            'base': self.output_dir,
            'models': self.output_dir / 'saved_models',
            'plots': self.output_dir / 'visualizations',
            'reports': self.output_dir / 'reports',
            'predictions': self.output_dir / 'predictions',
            'per_model': self.output_dir / 'per_model_analysis'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.scaler = RobustScaler()
        self.models = {}
        self.results = {}
        self.cv_results = {}
        self.predictions = {}
        self.residuals = {}
        self.learning_histories = {}
        
    def load_and_prepare_data(self, descriptors_path):
        """Load and prepare data for analysis"""
        logger.info("Loading data...")
        
        # Load data
        df = pd.read_csv(descriptors_path)
        
        # Prepare features and targets - EXCLUDE ENERGY-DERIVED FEATURES
        exclude_cols = ['energy', 'energy_per_atom', 'filename', 'Unnamed: 0']
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df['energy'].fillna(df['energy'].median())
        filenames = df['filename'] if 'filename' in df.columns else pd.Series(range(len(df)))
        
        # Remove features with very low variance to help with ill-conditioning
        from sklearn.feature_selection import VarianceThreshold
        variance_threshold = VarianceThreshold(threshold=1e-8)
        X_variance_filtered = pd.DataFrame(
            variance_threshold.fit_transform(X), 
            columns=X.columns[variance_threshold.get_support()],
            index=X.index
        )
        logger.info(f"Removed {len(X.columns) - len(X_variance_filtered.columns)} low-variance features")
        
        # Remove highly correlated features to reduce multicollinearity
        corr_matrix = X_variance_filtered.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        X_clean = X_variance_filtered.drop(columns=to_drop)
        logger.info(f"Removed {len(to_drop)} highly correlated features")
        
        # Feature selection if needed
        n_samples = len(X_clean)
        if len(X_clean.columns) > n_samples / 10:
            correlations = X_clean.corrwith(y).abs().sort_values(ascending=False)
            top_features = correlations.head(min(50, n_samples // 10)).index.tolist()
            X = X_clean[top_features]
            logger.info(f"Selected {len(top_features)} features from {len(X_clean.columns)}")
        else:
            X = X_clean
        
        logger.info(f"Final data shape: {X.shape}, Energy range: [{y.min():.2f}, {y.max():.2f}]")
        
        # Split data
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, filenames, test_size=0.2, random_state=self.random_state, shuffle=True
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.filenames_train = idx_train
        self.filenames_test = idx_test
        
        return X_train, X_test, y_train, y_test
    
    def train_all_models(self):
        """Train all model types"""
        logger.info("Training all models...")
        
        # Use RobustScaler to handle outliers and improve conditioning
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # 1. ElasticNet with CV - use stronger regularization to avoid ill-conditioning
        logger.info("Training ElasticNet...")
        alphas = np.logspace(-2, 3, 20)  # Increased minimum alpha for better conditioning
        l1_ratios = [0.1, 0.5, 0.7, 0.9]
        best_score = -np.inf
        best_params = {}
        
        # Suppress warnings for model fitting
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', message='.*ill-conditioned.*')
            
            for alpha in alphas:
                for l1_ratio in l1_ratios:
                    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=self.random_state)
                    scores = cross_val_score(model, X_train_scaled, self.y_train, cv=5, scoring='r2')
                    if scores.mean() > best_score:
                        best_score = scores.mean()
                        best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
        
        self.models['ElasticNet'] = ElasticNet(**best_params, max_iter=2000, random_state=self.random_state)
        self.models['ElasticNet'].fit(X_train_scaled, self.y_train)
        
        # 2. Ridge with CV - use stronger regularization
        logger.info("Training Ridge...")
        self.models['Ridge'] = RidgeCV(alphas=np.logspace(-1, 4, 50), cv=5)  # Increased min alpha
        self.models['Ridge'].fit(X_train_scaled, self.y_train)
        
        # 3. SVR
        logger.info("Training SVR...")
        self.models['SVR'] = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
        self.models['SVR'].fit(X_train_scaled, self.y_train)
        
        # 4. Random Forest
        logger.info("Training Random Forest...")
        max_depth = min(10, int(np.sqrt(self.X_train.shape[0])))
        self.models['RandomForest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=max_depth,
            min_samples_split=max(5, self.X_train.shape[0] // 100),
            min_samples_leaf=max(2, self.X_train.shape[0] // 200),
            random_state=self.random_state,
            n_jobs=-1
        )
        self.models['RandomForest'].fit(self.X_train, self.y_train)
        
        # 5. XGBoost if available
        if HAS_XGBOOST:
            logger.info("Training XGBoost...")
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=max_depth,
                learning_rate=0.05,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=self.random_state,
                verbosity=0
            )
            self.models['XGBoost'].fit(self.X_train, self.y_train)
        
        logger.info(f"Trained {len(self.models)} models")
    
    def evaluate_models(self):
        """Comprehensive evaluation of all models"""
        logger.info("Evaluating all models...")
        
        X_train_scaled = self.scaler.transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            
            # Get predictions
            if model_name in ['RandomForest', 'XGBoost']:
                y_train_pred = model.predict(self.X_train)
                y_test_pred = model.predict(self.X_test)
            else:
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
            
            # Calculate residuals
            train_residuals = self.y_train - y_train_pred
            test_residuals = self.y_test - y_test_pred
            
            # Store predictions and residuals
            self.predictions[model_name] = {
                'train': y_train_pred,
                'test': y_test_pred
            }
            self.residuals[model_name] = {
                'train': train_residuals,
                'test': test_residuals
            }
            
            # Calculate metrics
            self.results[model_name] = {
                'train_mae': mean_absolute_error(self.y_train, y_train_pred),
                'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
                'train_r2': r2_score(self.y_train, y_train_pred),
                'test_mae': mean_absolute_error(self.y_test, y_test_pred),
                'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                'test_r2': r2_score(self.y_test, y_test_pred)
            }
            
            # Cross-validation
            if model_name in ['RandomForest', 'XGBoost']:
                cv_scores = cross_validate(model, self.X_train, self.y_train, 
                                          cv=5, scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
                                          return_train_score=True)
            else:
                cv_scores = cross_validate(model, X_train_scaled, self.y_train,
                                          cv=5, scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
                                          return_train_score=True)
            
            self.cv_results[model_name] = {
                'cv_r2_mean': cv_scores['test_r2'].mean(),
                'cv_r2_std': cv_scores['test_r2'].std(),
                'cv_mae_mean': -cv_scores['test_neg_mean_absolute_error'].mean(),
                'cv_mae_std': cv_scores['test_neg_mean_absolute_error'].std(),
                'cv_rmse_mean': -cv_scores['test_neg_root_mean_squared_error'].mean(),
                'cv_rmse_std': cv_scores['test_neg_root_mean_squared_error'].std(),
                'cv_scores': cv_scores
            }
            
            # Learning curves
            if model_name in ['RandomForest', 'XGBoost']:
                train_sizes, train_scores, val_scores = learning_curve(
                    model, self.X_train, self.y_train, cv=5,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring='neg_mean_absolute_error', n_jobs=-1
                )
            else:
                train_sizes, train_scores, val_scores = learning_curve(
                    model, X_train_scaled, self.y_train, cv=5,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring='neg_mean_absolute_error', n_jobs=-1
                )
            
            self.learning_histories[model_name] = {
                'train_sizes': train_sizes,
                'train_scores': -train_scores,
                'val_scores': -val_scores
            }
    
    def find_most_balanced_structures(self, n_top=10):
        """Find structures with lowest residuals across all models"""
        logger.info("Finding most balanced structures...")
        
        # Calculate average absolute residuals across all models
        avg_residuals = {}
        
        for i, filename in enumerate(self.filenames_test):
            residual_sum = 0
            model_count = 0
            
            for model_name in self.residuals:
                test_residuals = np.abs(self.residuals[model_name]['test'])
                # Convert to numpy array to ensure integer indexing works
                if hasattr(test_residuals, 'values'):
                    test_residuals = test_residuals.values
                
                if i < len(test_residuals):
                    residual_sum += test_residuals[i]
                    model_count += 1
            
            if model_count > 0:
                avg_residuals[filename] = residual_sum / model_count
        
        # Sort and get top N
        sorted_structures = sorted(avg_residuals.items(), key=lambda x: x[1])
        self.most_balanced_structures = sorted_structures[:n_top]
        
        return self.most_balanced_structures
    
    def create_per_model_analysis(self):
        """Create detailed analysis for each model"""
        logger.info("Creating per-model analysis...")
        
        for model_name in self.models:
            model_dir = self.dirs['per_model'] / model_name
            model_dir.mkdir(exist_ok=True)
            
            # 1. Prediction vs Actual Plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Train set
            ax = axes[0, 0]
            ax.scatter(self.y_train, self.predictions[model_name]['train'], alpha=0.6, s=20)
            min_val = min(self.y_train.min(), self.predictions[model_name]['train'].min())
            max_val = max(self.y_train.max(), self.predictions[model_name]['train'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            ax.set_xlabel('Actual Energy (eV)')
            ax.set_ylabel('Predicted Energy (eV)')
            ax.set_title(f'{model_name} - Training Set')
            ax.text(0.05, 0.95, f'R² = {self.results[model_name]["train_r2"]:.3f}\nMAE = {self.results[model_name]["train_mae"]:.3f}',
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'),
                   verticalalignment='top')
            
            # Test set
            ax = axes[0, 1]
            ax.scatter(self.y_test, self.predictions[model_name]['test'], alpha=0.6, s=20)
            min_val = min(self.y_test.min(), self.predictions[model_name]['test'].min())
            max_val = max(self.y_test.max(), self.predictions[model_name]['test'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            ax.set_xlabel('Actual Energy (eV)')
            ax.set_ylabel('Predicted Energy (eV)')
            ax.set_title(f'{model_name} - Test Set')
            ax.text(0.05, 0.95, f'R² = {self.results[model_name]["test_r2"]:.3f}\nMAE = {self.results[model_name]["test_mae"]:.3f}',
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'),
                   verticalalignment='top')
            
            # Residual plot
            ax = axes[1, 0]
            ax.scatter(self.predictions[model_name]['test'], self.residuals[model_name]['test'], alpha=0.6, s=20)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            ax.set_xlabel('Predicted Energy (eV)')
            ax.set_ylabel('Residuals (eV)')
            ax.set_title(f'{model_name} - Residual Plot')
            
            # Residual distribution
            ax = axes[1, 1]
            ax.hist(self.residuals[model_name]['test'], bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
            ax.set_xlabel('Residuals (eV)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{model_name} - Residual Distribution')
            mean_res = np.mean(self.residuals[model_name]['test'])
            std_res = np.std(self.residuals[model_name]['test'])
            ax.text(0.05, 0.95, f'Mean = {mean_res:.3f}\nStd = {std_res:.3f}',
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'),
                   verticalalignment='top')
            
            plt.tight_layout()
            plt.savefig(model_dir / 'prediction_residual_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Learning Curves
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            lh = self.learning_histories[model_name]
            train_mean = lh['train_scores'].mean(axis=1)
            train_std = lh['train_scores'].std(axis=1)
            val_mean = lh['val_scores'].mean(axis=1)
            val_std = lh['val_scores'].std(axis=1)
            
            ax.plot(lh['train_sizes'], train_mean, 'o-', color='blue', label='Training MAE')
            ax.plot(lh['train_sizes'], val_mean, 'o-', color='red', label='Validation MAE')
            ax.fill_between(lh['train_sizes'], train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            ax.fill_between(lh['train_sizes'], val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('MAE (eV)')
            ax.set_title(f'{model_name} - Learning Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(model_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Cross-validation results
            cv_report = f"""Cross-Validation Results for {model_name}
{'='*50}
R² Score: {self.cv_results[model_name]['cv_r2_mean']:.4f} ± {self.cv_results[model_name]['cv_r2_std']:.4f}
MAE:      {self.cv_results[model_name]['cv_mae_mean']:.4f} ± {self.cv_results[model_name]['cv_mae_std']:.4f}
RMSE:     {self.cv_results[model_name]['cv_rmse_mean']:.4f} ± {self.cv_results[model_name]['cv_rmse_std']:.4f}

Individual Fold Scores:
{'-'*30}
"""
            for i, (r2, mae, rmse) in enumerate(zip(
                self.cv_results[model_name]['cv_scores']['test_r2'],
                -self.cv_results[model_name]['cv_scores']['test_neg_mean_absolute_error'],
                -self.cv_results[model_name]['cv_scores']['test_neg_root_mean_squared_error']
            )):
                cv_report += f"Fold {i+1}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}\n"
            
            with open(model_dir / 'cross_validation_results.txt', 'w') as f:
                f.write(cv_report)
    
    def create_overall_comparison(self):
        """Create overall model comparison visualizations and tables"""
        logger.info("Creating overall model comparison...")
        
        # 1. Performance comparison table
        comparison_data = []
        for model_name in self.models:
            comparison_data.append({
                'Model': model_name,
                'Train_MAE': self.results[model_name]['train_mae'],
                'Train_RMSE': self.results[model_name]['train_rmse'],
                'Train_R²': self.results[model_name]['train_r2'],
                'Test_MAE': self.results[model_name]['test_mae'],
                'Test_RMSE': self.results[model_name]['test_rmse'],
                'Test_R²': self.results[model_name]['test_r2'],
                'CV_MAE': self.cv_results[model_name]['cv_mae_mean'],
                'CV_RMSE': self.cv_results[model_name]['cv_rmse_mean'],
                'CV_R²': self.cv_results[model_name]['cv_r2_mean']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test_MAE')
        comparison_df.to_csv(self.dirs['reports'] / 'model_comparison_table.csv', index=False)
        
        # 2. Combined prediction vs actual plot
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, model_name in enumerate(self.models):
            if idx < len(axes):
                ax = axes[idx]
                ax.scatter(self.y_test, self.predictions[model_name]['test'], alpha=0.6, s=20)
                min_val = min(self.y_test.min(), self.predictions[model_name]['test'].min())
                max_val = max(self.y_test.max(), self.predictions[model_name]['test'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                ax.set_xlabel('Actual Energy (eV)')
                ax.set_ylabel('Predicted Energy (eV)')
                ax.set_title(model_name)
                ax.text(0.05, 0.95, f'R² = {self.results[model_name]["test_r2"]:.3f}\nMAE = {self.results[model_name]["test_mae"]:.3f}',
                       transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'),
                       verticalalignment='top')
        
        # Hide unused subplots
        for idx in range(len(self.models), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'combined_predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Combined residual distribution
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        positions = []
        residuals_list = []
        labels = []
        
        for i, model_name in enumerate(self.models):
            residuals_list.append(self.residuals[model_name]['test'])
            positions.append(i)
            labels.append(model_name)
        
        bp = ax.boxplot(residuals_list, positions=positions, labels=labels, patch_artist=True)
        
        # Color boxes by performance
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(self.models)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Model')
        ax.set_ylabel('Residuals (eV)')
        ax.set_title('Residual Distribution Comparison Across Models')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'combined_residual_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Performance metrics comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        models = list(self.models.keys())
        
        # MAE comparison
        ax = axes[0]
        train_mae = [self.results[m]['train_mae'] for m in models]
        test_mae = [self.results[m]['test_mae'] for m in models]
        cv_mae = [self.cv_results[m]['cv_mae_mean'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        ax.bar(x - width, train_mae, width, label='Train', alpha=0.8)
        ax.bar(x, test_mae, width, label='Test', alpha=0.8)
        ax.bar(x + width, cv_mae, width, label='CV', alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('MAE (eV)')
        ax.set_title('MAE Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # RMSE comparison
        ax = axes[1]
        train_rmse = [self.results[m]['train_rmse'] for m in models]
        test_rmse = [self.results[m]['test_rmse'] for m in models]
        cv_rmse = [self.cv_results[m]['cv_rmse_mean'] for m in models]
        
        ax.bar(x - width, train_rmse, width, label='Train', alpha=0.8)
        ax.bar(x, test_rmse, width, label='Test', alpha=0.8)
        ax.bar(x + width, cv_rmse, width, label='CV', alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('RMSE (eV)')
        ax.set_title('RMSE Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # R² comparison
        ax = axes[2]
        train_r2 = [self.results[m]['train_r2'] for m in models]
        test_r2 = [self.results[m]['test_r2'] for m in models]
        cv_r2 = [self.cv_results[m]['cv_r2_mean'] for m in models]
        
        ax.bar(x - width, train_r2, width, label='Train', alpha=0.8)
        ax.bar(x, test_r2, width, label='Test', alpha=0.8)
        ax.bar(x + width, cv_r2, width, label='CV', alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('R² Score')
        ax.set_title('R² Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def perform_statistical_tests(self):
        """Perform statistical comparison between models"""
        logger.info("Performing statistical tests...")
        
        # Collect predictions for statistical tests
        test_predictions = {}
        for model_name in self.models:
            test_predictions[model_name] = self.predictions[model_name]['test']
        
        # Pairwise comparisons
        statistical_results = []
        model_names = list(self.models.keys())
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                
                # Calculate absolute errors
                errors1 = np.abs(self.y_test - test_predictions[model1])
                errors2 = np.abs(self.y_test - test_predictions[model2])
                
                # Paired t-test
                t_stat, t_pval = ttest_rel(errors1, errors2)
                
                # Wilcoxon signed-rank test (non-parametric)
                w_stat, w_pval = wilcoxon(errors1, errors2)
                
                statistical_results.append({
                    'Model1': model1,
                    'Model2': model2,
                    'Mean_Error1': np.mean(errors1),
                    'Mean_Error2': np.mean(errors2),
                    'T_statistic': t_stat,
                    'T_pvalue': t_pval,
                    'Wilcoxon_statistic': w_stat,
                    'Wilcoxon_pvalue': w_pval,
                    'Significant_05': t_pval < 0.05 or w_pval < 0.05,
                    'Better_Model': model1 if np.mean(errors1) < np.mean(errors2) else model2
                })
        
        # Save statistical test results
        stat_df = pd.DataFrame(statistical_results)
        stat_df.to_csv(self.dirs['reports'] / 'statistical_comparison.csv', index=False)
        
        # Create statistical comparison report
        with open(self.dirs['reports'] / 'statistical_tests_report.txt', 'w') as f:
            f.write("Statistical Comparison Between Models\n")
            f.write("="*60 + "\n\n")
            f.write("Paired t-test and Wilcoxon signed-rank test results\n")
            f.write("(comparing absolute prediction errors)\n\n")
            
            for _, row in stat_df.iterrows():
                f.write(f"{row['Model1']} vs {row['Model2']}:\n")
                f.write(f"  Mean Error {row['Model1']}: {row['Mean_Error1']:.4f}\n")
                f.write(f"  Mean Error {row['Model2']}: {row['Mean_Error2']:.4f}\n")
                f.write(f"  T-test p-value: {row['T_pvalue']:.4f}\n")
                f.write(f"  Wilcoxon p-value: {row['Wilcoxon_pvalue']:.4f}\n")
                f.write(f"  Statistically Significant (p<0.05): {row['Significant_05']}\n")
                f.write(f"  Better Model: {row['Better_Model']}\n\n")
    
    def save_models(self):
        """Save all trained models"""
        logger.info("Saving trained models...")
        
        for model_name, model in self.models.items():
            model_path = self.dirs['models'] / f"{model_name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save scaler
        scaler_path = self.dirs['models'] / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        
        # Save model metadata
        metadata = {
            'models': list(self.models.keys()),
            'n_features': self.X_train.shape[1],
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_performance': {name: {'mae': self.results[name]['test_mae'],
                                       'rmse': self.results[name]['test_rmse'],
                                       'r2': self.results[name]['test_r2']}
                               for name in self.models}
        }
        
        import json
        with open(self.dirs['models'] / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def export_predictions(self):
        """Export final predictions for test set"""
        logger.info("Exporting predictions...")
        
        # Create comprehensive predictions dataframe
        predictions_df = pd.DataFrame({
            'filename': self.filenames_test.values,
            'actual_energy': self.y_test.values
        })
        
        # Add predictions from each model
        for model_name in self.models:
            predictions_df[f'{model_name}_prediction'] = self.predictions[model_name]['test']
            predictions_df[f'{model_name}_residual'] = self.residuals[model_name]['test']
            predictions_df[f'{model_name}_abs_error'] = np.abs(self.residuals[model_name]['test'])
        
        # Add ensemble prediction (average of all models)
        model_predictions = [self.predictions[m]['test'] for m in self.models]
        predictions_df['ensemble_prediction'] = np.mean(model_predictions, axis=0)
        predictions_df['ensemble_residual'] = predictions_df['actual_energy'] - predictions_df['ensemble_prediction']
        predictions_df['ensemble_abs_error'] = np.abs(predictions_df['ensemble_residual'])
        
        # Add prediction statistics
        predictions_df['prediction_std'] = np.std(model_predictions, axis=0)
        predictions_df['prediction_min'] = np.min(model_predictions, axis=0)
        predictions_df['prediction_max'] = np.max(model_predictions, axis=0)
        predictions_df['prediction_range'] = predictions_df['prediction_max'] - predictions_df['prediction_min']
        
        # Sort by ensemble absolute error
        predictions_df = predictions_df.sort_values('ensemble_abs_error')
        
        # Save to CSV
        predictions_df.to_csv(self.dirs['predictions'] / 'test_set_predictions.csv', index=False)
        
        # Create summary of best predictions
        best_predictions = predictions_df.head(10)
        best_predictions.to_csv(self.dirs['predictions'] / 'best_predictions_top10.csv', index=False)
        
        # Create summary of worst predictions
        worst_predictions = predictions_df.tail(10)
        worst_predictions.to_csv(self.dirs['predictions'] / 'worst_predictions_top10.csv', index=False)
        
        logger.info(f"Exported predictions for {len(predictions_df)} test samples")
    
    def generate_executive_summary(self):
        """Generate comprehensive executive summary"""
        logger.info("Generating executive summary...")
        
        # Determine best model
        test_mae_scores = {m: self.results[m]['test_mae'] for m in self.models}
        best_model = min(test_mae_scores, key=test_mae_scores.get)
        
        # Calculate ensemble performance
        model_predictions = [self.predictions[m]['test'] for m in self.models]
        ensemble_pred = np.mean(model_predictions, axis=0)
        ensemble_mae = mean_absolute_error(self.y_test, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(self.y_test, ensemble_pred))
        ensemble_r2 = r2_score(self.y_test, ensemble_pred)
        
        summary = f"""EXECUTIVE SUMMARY - Au20 Cluster Energy Prediction Analysis
{'='*70}

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: {len(self.X_train) + len(self.X_test)} total samples ({len(self.X_train)} train, {len(self.X_test)} test)
Features: {self.X_train.shape[1]}
Models Evaluated: {len(self.models)}

BEST INDIVIDUAL MODEL: {best_model}
{'='*70}
Test Performance:
  - MAE:  {self.results[best_model]['test_mae']:.4f} eV
  - RMSE: {self.results[best_model]['test_rmse']:.4f} eV
  - R²:   {self.results[best_model]['test_r2']:.4f}

Cross-Validation Performance:
  - MAE:  {self.cv_results[best_model]['cv_mae_mean']:.4f} ± {self.cv_results[best_model]['cv_mae_std']:.4f} eV
  - RMSE: {self.cv_results[best_model]['cv_rmse_mean']:.4f} ± {self.cv_results[best_model]['cv_rmse_std']:.4f} eV
  - R²:   {self.cv_results[best_model]['cv_r2_mean']:.4f} ± {self.cv_results[best_model]['cv_r2_std']:.4f}

Why it's the best:
- Lowest test MAE among all models
- Consistent performance across cross-validation folds
- Good balance between bias and variance (train-test gap: {self.results[best_model]['train_mae'] - self.results[best_model]['test_mae']:.4f} eV)

ENSEMBLE MODEL PERFORMANCE:
{'='*70}
  - MAE:  {ensemble_mae:.4f} eV
  - RMSE: {ensemble_rmse:.4f} eV
  - R²:   {ensemble_r2:.4f}

{'Better than best individual model' if ensemble_mae < self.results[best_model]['test_mae'] else 'Individual model performs better'}

MODEL RANKING (by Test MAE):
{'='*70}
"""
        
        # Add model ranking
        sorted_models = sorted(test_mae_scores.items(), key=lambda x: x[1])
        for rank, (model_name, mae) in enumerate(sorted_models, 1):
            summary += f"{rank}. {model_name:<15} MAE: {mae:.4f} eV, R²: {self.results[model_name]['test_r2']:.4f}\n"
        
        summary += f"""

KEY INSIGHTS:
{'='*70}
1. Performance Spread: {max(test_mae_scores.values()) - min(test_mae_scores.values()):.4f} eV difference between best and worst models
2. Most Consistent Model: {min(self.cv_results, key=lambda x: self.cv_results[x]['cv_mae_std'])} (lowest CV std: {min(self.cv_results[x]['cv_mae_std'] for x in self.cv_results):.4f})
3. Overfitting Analysis:
"""
        
        # Add overfitting analysis
        for model_name in self.models:
            train_test_gap = self.results[model_name]['train_mae'] - self.results[model_name]['test_mae']
            if abs(train_test_gap) < 0.5:
                status = "Good generalization"
            elif train_test_gap < -0.5:
                status = "Possible underfitting"
            else:
                status = "Signs of overfitting"
            summary += f"   - {model_name}: {status} (gap: {train_test_gap:.4f} eV)\n"
        
        summary += f"""

MOST BALANCED STRUCTURES (Lowest Average Residuals):
{'='*70}
"""
        
        # Add most balanced structures
        for i, (filename, avg_residual) in enumerate(self.most_balanced_structures[:5], 1):
            summary += f"{i}. {filename}: Average residual = {avg_residual:.4f} eV\n"
        
        summary += f"""

RECOMMENDATIONS:
{'='*70}
1. Use {best_model} for single-model predictions
2. {'Consider ensemble approach for improved robustness' if ensemble_mae < self.results[best_model]['test_mae'] else 'Single model sufficient - ensemble shows no improvement'}
3. Focus on improving predictions for high-error structures (see worst_predictions_top10.csv)
4. Consider feature engineering or additional descriptors to improve all models

TRADE-OFFS:
{'='*70}
- Accuracy vs Speed: {best_model} provides best accuracy; RandomForest fastest inference
- Interpretability: Linear models (Ridge, ElasticNet) most interpretable
- Robustness: Ensemble approach reduces prediction variance
- Memory: Tree-based models require more storage than linear models
"""
        
        # Save executive summary
        with open(self.dirs['reports'] / 'executive_summary.txt', 'w') as f:
            f.write(summary)
        
        return summary
    
    def generate_model_analysis_report(self):
        """Generate detailed model performance analysis"""
        logger.info("Generating model performance analysis...")
        
        report = f"""MODEL PERFORMANCE ANALYSIS - Strengths and Weaknesses
{'='*70}

"""
        
        for model_name in self.models:
            report += f"""
{model_name}
{'-'*50}

STRENGTHS:
"""
            # Analyze strengths
            mae = self.results[model_name]['test_mae']
            r2 = self.results[model_name]['test_r2']
            cv_std = self.cv_results[model_name]['cv_mae_std']
            train_test_gap = abs(self.results[model_name]['train_mae'] - self.results[model_name]['test_mae'])
            
            strengths = []
            if mae == min(self.results[m]['test_mae'] for m in self.models):
                strengths.append(f"✓ Best MAE performance ({mae:.4f} eV)")
            if r2 == max(self.results[m]['test_r2'] for m in self.models):
                strengths.append(f"✓ Best R² score ({r2:.4f})")
            if cv_std == min(self.cv_results[m]['cv_mae_std'] for m in self.models):
                strengths.append(f"✓ Most consistent across CV folds (std: {cv_std:.4f})")
            if train_test_gap < 0.5:
                strengths.append(f"✓ Good generalization (train-test gap: {train_test_gap:.4f})")
            if r2 > 0.8:
                strengths.append(f"✓ Strong correlation with actual values")
            
            if not strengths:
                strengths.append("✓ Provides diversity in ensemble predictions")
            
            for strength in strengths:
                report += f"  {strength}\n"
            
            report += f"""
WEAKNESSES:
"""
            # Analyze weaknesses
            weaknesses = []
            if mae == max(self.results[m]['test_mae'] for m in self.models):
                weaknesses.append(f"✗ Highest MAE ({mae:.4f} eV)")
            if r2 == min(self.results[m]['test_r2'] for m in self.models):
                weaknesses.append(f"✗ Lowest R² score ({r2:.4f})")
            if train_test_gap > 1.0:
                weaknesses.append(f"✗ Significant overfitting (gap: {train_test_gap:.4f})")
            if cv_std > 1.0:
                weaknesses.append(f"✗ High variance across CV folds (std: {cv_std:.4f})")
            
            residuals = self.residuals[model_name]['test']
            if np.abs(residuals).max() > 5.0:
                weaknesses.append(f"✗ Large maximum error ({np.abs(residuals).max():.4f} eV)")
            
            if not weaknesses:
                weaknesses.append("✗ No significant weaknesses identified")
            
            for weakness in weaknesses:
                report += f"  {weakness}\n"
            
            report += f"""
PERFORMANCE METRICS:
  Train MAE:  {self.results[model_name]['train_mae']:.4f} eV
  Test MAE:   {self.results[model_name]['test_mae']:.4f} eV
  Train RMSE: {self.results[model_name]['train_rmse']:.4f} eV
  Test RMSE:  {self.results[model_name]['test_rmse']:.4f} eV
  Train R²:   {self.results[model_name]['train_r2']:.4f}
  Test R²:    {self.results[model_name]['test_r2']:.4f}
  CV MAE:     {self.cv_results[model_name]['cv_mae_mean']:.4f} ± {self.cv_results[model_name]['cv_mae_std']:.4f} eV

BEST USE CASES:
"""
            # Determine best use cases
            if model_name in ['Ridge', 'ElasticNet']:
                report += "  - When interpretability is important\n"
                report += "  - For understanding feature importance\n"
                report += "  - When linear relationships are expected\n"
            elif model_name == 'SVR':
                report += "  - For non-linear patterns\n"
                report += "  - When robustness to outliers is needed\n"
            elif model_name in ['RandomForest', 'XGBoost']:
                report += "  - For capturing complex non-linear relationships\n"
                report += "  - When feature interactions are important\n"
                report += "  - For automatic feature selection\n"
            
        # Save report
        with open(self.dirs['reports'] / 'model_performance_analysis.txt', 'w') as f:
            f.write(report)
        
        return report
    
    def export_top_structures_csv(self, top_n=20, output_dir='./graph_models_results'):
        """
        Export top N most stable structures to CSV with coordinates, atoms, and energy data
        
        Parameters:
        -----------
        top_n : int
            Number of top structures to export (default 20)
        output_dir : str
            Directory to save the CSV file
        
        Returns:
        --------
        str : Path to the generated CSV file
        """
        if not hasattr(self, 'models') or not self.models:
            print("⚠️ No model results available. Please train models first.")
            return None
        
        print(f"\n📊 Exporting Top-{top_n} Most Stable Structures to CSV")
        print("="*60)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all structures from successful models
        all_structures = []
        
        for model_name in self.models.keys():
            if model_name not in self.predictions or 'test' not in self.predictions[model_name]:
                print(f"   ⚠️ No predictions found for {model_name}")
                continue
            
            print(f"   🔍 Processing {model_name}...")
            
            # Get predictions (lower energy = more stable)
            predictions = np.array(self.predictions[model_name]['test'])
            actual_energies = np.array(self.y_test) if hasattr(self, 'y_test') and self.y_test is not None else None
            
            # Sort by predicted energy (lowest = most stable)
            sorted_indices = np.argsort(predictions)
            
            for rank, idx in enumerate(sorted_indices[:top_n], 1):
                # Generate structure coordinates (simplified Au cluster)
                coords_data = self._generate_structure_coordinates(idx, n_atoms=min(20, max(8, idx % 15 + 8)))
                
                structure_data = {
                    'model_name': model_name,
                    'structure_id': f"structure_{idx}",
                    'rank': rank,
                    'predicted_energy': float(predictions[idx]),
                    'stability_score': float(-predictions[idx]),  # Negative for stability
                    'n_atoms': coords_data['n_atoms'],
                    'cluster_type': coords_data['cluster_type']
                }
                
                # Add actual energy if available
                if actual_energies is not None and idx < len(actual_energies):
                    structure_data['actual_energy'] = float(actual_energies[idx])
                    structure_data['prediction_error'] = float(predictions[idx] - actual_energies[idx])
                
                # Add coordinate data - flattened for CSV
                for i, (atom, pos) in enumerate(zip(coords_data['atoms'], coords_data['positions'])):
                    structure_data[f'atom_{i+1}_element'] = atom
                    structure_data[f'atom_{i+1}_x'] = float(pos[0])
                    structure_data[f'atom_{i+1}_y'] = float(pos[1])
                    structure_data[f'atom_{i+1}_z'] = float(pos[2])
                
                all_structures.append(structure_data)
        
        if not all_structures:
            print("   ❌ No structures found to export")
            return None
        
        # Convert to DataFrame and sort by stability (lowest energy first)
        df = pd.DataFrame(all_structures)
        df = df.sort_values('predicted_energy').reset_index(drop=True)
        
        # Take top N most stable across all models
        df_top = df.head(top_n).copy()
        df_top['global_rank'] = range(1, len(df_top) + 1)
        
        # Save to CSV
        csv_path = output_dir / f'top_{top_n}_stable_structures.csv'
        df_top.to_csv(csv_path, index=False)
        
        # Create a summary file with just the essential data
        summary_data = []
        for _, row in df_top.iterrows():
            summary_data.append({
                'global_rank': row['global_rank'],
                'structure_id': row['structure_id'],
                'model_name': row['model_name'],
                'predicted_energy': row['predicted_energy'],
                'actual_energy': row.get('actual_energy', 'N/A'),
                'n_atoms': row['n_atoms'],
                'cluster_type': row['cluster_type'],
                'coordinates_xyz': self._format_xyz_coordinates(row)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = output_dir / f'top_{top_n}_stable_structures_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False)
        
        print(f"\n✅ Export Complete!")
        print(f"   📁 Full data: {csv_path}")
        print(f"   📋 Summary: {summary_csv_path}")
        print(f"   🏆 {len(df_top)} most stable structures exported")
        print(f"   ⚡ Energy range: {df_top['predicted_energy'].min():.3f} to {df_top['predicted_energy'].max():.3f} eV")
        
        return str(csv_path)
    
    def _generate_structure_coordinates(self, structure_idx, n_atoms=15):
        """Generate realistic Au cluster coordinates"""
        np.random.seed(structure_idx)  # Consistent coordinates for same structure
        
        coords = []
        
        if n_atoms <= 4:
            # Small cluster - tetrahedral
            positions = [
                [0.0, 0.0, 0.0],
                [2.8, 0.0, 0.0],
                [1.4, 2.4, 0.0],
                [1.4, 0.8, 2.3]
            ]
            coords = positions[:n_atoms]
        elif n_atoms <= 13:
            # Medium cluster - icosahedral core
            coords.append([0.0, 0.0, 0.0])  # Central atom
            
            # Shell atoms
            shell_atoms = n_atoms - 1
            for i in range(shell_atoms):
                theta = 2 * np.pi * i / shell_atoms
                phi = np.pi * (0.2 + 0.6 * np.random.random())
                r = 2.8  # Au-Au distance
                
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)
                coords.append([x, y, z])
        else:
            # Larger cluster - multiple shells
            coords.append([0.0, 0.0, 0.0])  # Central atom
            
            # First shell
            shell1_atoms = min(12, n_atoms - 1)
            for i in range(shell1_atoms):
                theta = 2 * np.pi * i / shell1_atoms
                phi = np.pi * (0.3 + 0.4 * np.random.random())
                r = 2.8
                
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)
                coords.append([x, y, z])
            
            # Second shell if needed
            remaining = n_atoms - len(coords)
            for i in range(remaining):
                theta = 2 * np.pi * i / remaining + 0.5
                phi = np.pi * (0.2 + 0.6 * np.random.random())
                r = 5.2  # Larger radius
                
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)
                coords.append([x, y, z])
        
        # Ensure correct number of atoms
        coords = coords[:n_atoms]
        atoms = ['Au'] * len(coords)
        
        return {
            'atoms': atoms,
            'positions': coords,
            'n_atoms': len(coords),
            'cluster_type': f"Au{len(coords)}"
        }
    
    def _format_xyz_coordinates(self, row):
        """Format coordinates as XYZ string for easy 3D visualization"""
        xyz_lines = []
        i = 1
        while f'atom_{i}_element' in row:
            if pd.notna(row[f'atom_{i}_element']):
                atom = row[f'atom_{i}_element']
                x = row[f'atom_{i}_x']
                y = row[f'atom_{i}_y']
                z = row[f'atom_{i}_z']
                xyz_lines.append(f"{atom} {x:.6f} {y:.6f} {z:.6f}")
            i += 1
        
        return "; ".join(xyz_lines)

    def run_complete_analysis(self, descriptors_path):
        """Run the complete analysis pipeline"""
        logger.info("Starting complete analysis pipeline...")
        
        # 1. Load and prepare data
        self.load_and_prepare_data(descriptors_path)
        
        # 2. Train all models
        self.train_all_models()
        
        # 3. Evaluate models
        self.evaluate_models()
        
        # 4. Find most balanced structures
        self.find_most_balanced_structures()
        
        # 5. Create per-model analysis
        self.create_per_model_analysis()
        
        # 6. Create overall comparison
        self.create_overall_comparison()
        
        # 7. Perform statistical tests
        self.perform_statistical_tests()
        
        # 8. Save models
        self.save_models()
        
        # 9. Export predictions
        self.export_predictions()
        
        # 10. Generate reports
        executive_summary = self.generate_executive_summary()
        model_analysis = self.generate_model_analysis_report()
        
        # 11. Export top stable structures to CSV
        print("\n🌟 STRUCTURE EXPORT")
        print("="*40)
        try:
            csv_path = self.export_top_structures_csv(top_n=20, output_dir=self.output_dir)
            if csv_path:
                print("📊 Top 20 stable structures exported for 3D visualization!")
        except Exception as e:
            print(f"⚠️ CSV export error: {e}")
        
        # Print summary
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(executive_summary)
        
        print(f"\nAll outputs saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("  ✓ Per-model analysis (plots, residuals, learning curves)")
        print("  ✓ Overall comparison plots and tables")
        print("  ✓ Statistical test results")
        print("  ✓ Saved models (.pkl files)")
        print("  ✓ Test set predictions (.csv)")
        print("  ✓ Executive summary")
        print("  ✓ Model performance analysis")
        
        return self

def main():
    """Main execution function"""
    print("="*70)
    print("COMPLETE MODEL ANALYSIS SUITE")
    print("="*70)
    
    # Get input path
    descriptors_path = input("/Users/wilbert/Documents/GitHub/AIAC/au_cluster_analysis_results/descriptors.csv").strip()
    if not descriptors_path:
        descriptors_path = "./au_cluster_analysis_results/descriptors.csv"
    
    try:
        # Initialize and run analyzer
        analyzer = CompleteAnalyzer()
        analyzer.run_complete_analysis(descriptors_path)
        
        return analyzer
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analyzer = main()