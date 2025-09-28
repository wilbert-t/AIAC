"""
Residual analysis plotting function for graph neural network models.
To be added to the DualGPUGraphNeuralNetworkAnalyzer class.
"""

def _plot_residual_analysis(self, output_dir):
    """
    Generate detailed residual analysis plots for each model.
    Plots include:
    - Residual vs Predicted values
    - Residual distribution histogram
    - QQ plot for residuals
    - Residual vs index (to check for patterns)
    """
    if not self.results:
        return
    
    for name, result in self.results.items():
        try:
            predictions = np.array(result['predictions'])
            targets = np.array(result['targets'])
            
            # Calculate residuals
            residuals = targets - predictions
            
            # Create figure with 2x2 subplot grid
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle(f'Residual Analysis: {name.replace("_", " ").title()}', fontsize=16)
            
            # 1. Residual vs Predicted
            axes[0, 0].scatter(predictions, residuals, alpha=0.6, s=30)
            axes[0, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
            axes[0, 0].set_xlabel('Predicted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residual vs Predicted')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add metrics
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            axes[0, 0].text(0.05, 0.95, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}', 
                         transform=axes[0, 0].transAxes,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                         verticalalignment='top')
            
            # 2. Residual Histogram
            sns.histplot(residuals, kde=True, ax=axes[0, 1], bins=30)
            axes[0, 1].set_xlabel('Residual Value')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Residual Distribution')
            
            # Add normal distribution fit line
            mu, std = np.mean(residuals), np.std(residuals)
            x = np.linspace(min(residuals), max(residuals), 100)
            p = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * std**2))
            axes[0, 1].text(0.05, 0.95, f'Mean: {mu:.4f}\nStd Dev: {std:.4f}', 
                         transform=axes[0, 1].transAxes,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                         verticalalignment='top')
            
            # 3. QQ Plot of Residuals
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Normal Q-Q Plot of Residuals')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Residual vs Index (order)
            axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.6, s=30)
            axes[1, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
            axes[1, 1].set_xlabel('Index')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residual vs Index')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add autocorrelation info
            from scipy.stats import pearsonr
            if len(residuals) > 1:
                lag_1_autocorr, _ = pearsonr(residuals[:-1], residuals[1:])
                axes[1, 1].text(0.05, 0.95, f'Lag-1 Autocorr: {lag_1_autocorr:.4f}', 
                            transform=axes[1, 1].transAxes,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            verticalalignment='top')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            # Save figure
            plt.savefig(output_dir / f'{name}_residual_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating residual analysis for {name}: {e}")
            import traceback
            traceback.print_exc()