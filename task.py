#!/usr/bin/env python3
"""
Task 1: Au Cluster Analysis - Complete Implementation with Advanced Features
Parse xyz files, compute descriptors, and generate comprehensive statistical analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial import distance_matrix, ConvexHull
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
import re
import warnings
from multiprocessing import Pool
import logging
warnings.filterwarnings('ignore')

class PlotManager:
    """Separate class to handle all plotting functionality"""
    
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir or './plots')
        self.output_dir.mkdir(exist_ok=True)
        
        # Set default style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_basic_plots(self, df):
        """Generate all basic plots"""
        self._plot_energy_size(df)
        self._plot_distributions(df)
        self._plot_correlations(df)
    
    def _plot_energy_size(self, df):
        if df['energy'].notna().sum() > 1:
            plt.figure(figsize=(10, 6))
            valid_data = df.dropna(subset=['energy', 'n_atoms'])
            plt.scatter(valid_data['n_atoms'], valid_data['energy'], alpha=0.7, s=50)
            plt.xlabel('Number of Atoms')
            plt.ylabel('Total Energy (eV)')
            plt.title('Energy vs Cluster Size')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'energy_vs_size.png', dpi=300)
            plt.close()
    
    def _plot_distributions(self, df):
        key_features = ['mean_bond_length', 'mean_coordination', 'radius_of_gyration', 'surface_fraction']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, feature in enumerate(key_features):
            if feature in df.columns and df[feature].notna().sum() > 0:
                data = df[feature].dropna()
                axes[i].hist(data, bins=20, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{feature.replace("_", " ").title()}')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'descriptors_distribution.png', dpi=300)
        plt.close()
    
    def _plot_correlations(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300)
            plt.close()

class AuClusterAnalyzer:
    """Complete analyzer for Au cluster xyz files with advanced features"""
    
    def __init__(self, data_dir, chunk_size=1000):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.structures = []
        self.setup_logging()
        self.logger = logging.getLogger('AuClusterAnalyzer')
        self.plot_manager = PlotManager()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('au_cluster_analysis.log'),
                logging.StreamHandler()
            ]
        )
    
    # ... (keep all your existing methods from parse_xyz_file through compute_descriptors)
    
    def process_in_chunks(self, files):
        """Process files in chunks to manage memory"""
        for i in range(0, len(files), self.chunk_size):
            chunk = files[i:i + self.chunk_size]
            structures = [self.parse_xyz_file(f) for f in chunk]
            yield from (s for s in structures if s is not None)
    
    def _get_numeric_features(self, df, exclude_cols=None):
        """Helper method to get numeric features"""
        exclude_cols = exclude_cols or ['energy', 'energy_per_atom']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return [col for col in numeric_cols if col not in exclude_cols]
    
    def _prepare_output_dir(self, base_dir, subdir=None):
        """Helper method to prepare output directory"""
        output_dir = Path(base_dir)
        if subdir:
            output_dir = output_dir / subdir
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir
    
    def analyze_feature_target_relationships(self, df, target_col='energy'):
        """Analyze correlations between features and target variable"""
        # ... (keep existing implementation)
    
    def perform_pca_analysis(self, df, output_dir=None):
        """Perform PCA analysis and visualization"""
        # ... (keep existing implementation)
    
    def detect_outliers(self, df, output_dir=None):
        """Detect and analyze outliers in the dataset"""
        # ... (keep existing implementation)
    
    def run_advanced_analysis(self, df, output_dir=None):
        """Run all advanced analysis features"""
        print("\n🔬 Running Advanced Analysis Features...")
        
        output_dir = self._prepare_output_dir(output_dir or './advanced_analysis')
        
        # Run analyses
        corr_df = self.analyze_feature_target_relationships(df)
        pca_results = self.perform_pca_analysis(df, output_dir)
        outliers_info = self.detect_outliers(df, output_dir)
        
        # Save correlation results if available
        if corr_df is not None:
            corr_df.to_csv(output_dir / 'feature_correlations.csv', index=False)
        
        print(f"\n✅ Advanced analysis complete! Results saved to: {output_dir}")
        
        return {
            'correlations': corr_df,
            'pca_results': pca_results,
            'outliers': outliers_info
        }

def main():
    """Main execution function"""
    # ... (keep existing implementation)

if __name__ == "__main__":
    result = main()
    if result is not None:
        df, analyzer, advanced_results = result
        print("Analysis completed successfully!")
        
        # Print key findings
        if advanced_results['correlations'] is not None:
            top_corr = advanced_results['correlations'].head(3)
            print(f"\n🔍 Key Findings:")
            print(f"Top 3 features correlated with energy:")
            for _, row in top_corr.iterrows():
                print(f"  • {row['feature']}: r = {row['pearson_r']:.3f}")
    else:
        print("Analysis failed - no data to process")