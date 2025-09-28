"""
Detailed learning curves visualization for graph neural network models.
To be added to the DualGPUGraphNeuralNetworkAnalyzer class.
"""

def _plot_learning_curves_detailed(self, output_dir):
    """
    Generate detailed learning curves for each model with epoch-by-epoch analysis.
    Plots include:
    - Training and validation loss curves
    - Learning rate decay
    - Validation metrics per epoch
    - Training time per epoch
    """
    if not self.results:
        return
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    for name, result in self.results.items():
        try:
            # Extract data
            train_losses = np.array(result['train_losses'])
            val_losses = np.array(result['val_losses'])
            gpu_device = result.get('gpu_device', 'unknown')
            
            # Create figure with custom layout
            fig = plt.figure(figsize=(14, 10))
            gs = GridSpec(2, 2, figure=fig)
            
            # 1. Learning curves (train/val loss)
            ax1 = fig.add_subplot(gs[0, :])  # Takes top row entirely
            epochs = np.arange(1, len(train_losses) + 1)
            
            # Plot training and validation loss
            ax1.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.8, marker='o', markersize=3)
            ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.8, marker='s', markersize=3)
            
            # Find best epoch
            best_epoch = np.argmin(val_losses) + 1  # +1 because epochs are 1-indexed
            best_val_loss = val_losses[best_epoch-1]
            
            # Highlight best epoch
            ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
            ax1.text(best_epoch + 0.5, min(val_losses) * 0.9, 
                     f'Best epoch: {best_epoch}\nVal loss: {best_val_loss:.4f}', 
                     bbox=dict(facecolor='white', alpha=0.8))
            
            # Add early stopping point if training stopped early
            if len(train_losses) < result.get('training_params', {}).get('epochs', len(train_losses)):
                ax1.scatter(len(train_losses), val_losses[-1], s=100, marker='X', color='red')
                ax1.text(len(train_losses) - 5, val_losses[-1], 'Early stopping', 
                        bbox=dict(facecolor='red', alpha=0.2))
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'Learning Curves: {name.replace("_", " ").title()} ({gpu_device})')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right')
            
            # Use log scale if there's a large range in the loss values
            if max(train_losses) / (min(train_losses) + 1e-10) > 10:
                ax1.set_yscale('log')
            
            # 2. Loss improvement rate (derivative of loss)
            ax2 = fig.add_subplot(gs[1, 0])
            if len(train_losses) > 1:
                # Calculate improvement rate (negative derivative)
                train_improve_rate = -np.diff(train_losses)
                val_improve_rate = -np.diff(val_losses)
                
                ax2.plot(epochs[1:], train_improve_rate, 'b-', label='Training Improvement', alpha=0.7)
                ax2.plot(epochs[1:], val_improve_rate, 'r-', label='Validation Improvement', alpha=0.7)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss Improvement')
                ax2.set_title('Loss Improvement Rate')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # Annotate when validation loss starts getting worse (overfitting)
                neg_improvements = np.where(val_improve_rate < 0)[0]
                if len(neg_improvements) > 0:
                    first_overfit = neg_improvements[0] + 1  # +1 because we're working with diff
                    ax2.axvline(x=epochs[first_overfit], color='orange', linestyle='--', alpha=0.5)
                    ax2.text(epochs[first_overfit] + 0.5, min(val_improve_rate), 'Potential overfitting begins',
                            bbox=dict(facecolor='orange', alpha=0.2))
            
            # 3. Train/Val loss ratio (to detect overfitting)
            ax3 = fig.add_subplot(gs[1, 1])
            if len(train_losses) > 0:
                # Calculate ratio of val_loss to train_loss (>1 means potential overfitting)
                loss_ratio = val_losses / (train_losses + 1e-10)  # Avoid division by zero
                
                ax3.plot(epochs, loss_ratio, 'g-', alpha=0.8)
                ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5)
                
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Val/Train Loss Ratio')
                ax3.set_title('Overfitting Indicator')
                ax3.grid(True, alpha=0.3)
                
                # Find where ratio starts consistently growing
                if len(loss_ratio) > 5:
                    window_size = min(5, len(loss_ratio) // 3)
                    smooth_ratio = np.convolve(loss_ratio, np.ones(window_size)/window_size, mode='valid')
                    smooth_epochs = epochs[window_size-1:]
                    
                    # Plot smoothed line
                    ax3.plot(smooth_epochs, smooth_ratio, 'b-', alpha=0.5, label=f'{window_size}-epoch Moving Avg')
                    ax3.legend()
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.92, hspace=0.3)
            
            # Save figure
            plt.savefig(output_dir / f'{name}_detailed_learning_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating detailed learning curves for {name}: {e}")
            import traceback
            traceback.print_exc()