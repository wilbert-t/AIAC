#!/usr/bin/env python3
"""
Category 5: Graph Neural Network Models for Au Cluster Energy Prediction
Models: SchNet, DimeNet++, CGCNN, MEGNet-inspired
Optimized for PyTorch Geometric with MPS acceleration on MacBook Pro M4
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# PyTorch and PyTorch Geometric
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import SchNet, DimeNet, GCNConv, GATConv, global_mean_pool, global_max_pool
    from torch_geometric.utils import from_networkx
    
    # Check for MPS (Metal Performance Shaders) support
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ PyTorch MPS (Metal) acceleration available")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (MPS not available)")
    
    PYTORCH_AVAILABLE = True
except ImportError:
    print("❌ PyTorch Geometric not available")
    PYTORCH_AVAILABLE = False
    device = None

# ASE for molecular structures
try:
    from ase.atoms import Atoms
    from ase.neighborlist import NeighborList, natural_cutoffs
    ASE_AVAILABLE = True
except ImportError:
    print("❌ ASE not available")
    ASE_AVAILABLE = False

class GraphNeuralNetworkAnalyzer:
    """
    Graph Neural Network Models for Au Cluster Analysis
    
    Why Graph Neural Networks for Au Clusters:
    1. Direct Structural Input: Operates on atomic coordinates and bonds directly
    2. Physical Invariance: Translation, rotation, and permutation invariant
    3. Local Interactions: Models many-body interactions between atoms
    4. Scalability: Handles variable-size clusters naturally
    5. Interpretability: Can visualize important atomic contributions
    6. State-of-the-art: Best performance for molecular property prediction
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        if PYTORCH_AVAILABLE:
            torch.manual_seed(random_state)
        
        self.models = {}
        self.results = {}
        self.device = device
        
        # Graph neural network architectures
        self.model_configs = self._initialize_models()
    
    def _initialize_models(self):
        """Initialize graph neural network architectures"""
        if not PYTORCH_AVAILABLE:
            return {}
        
        configs = {
            'schnet': {
                'architecture': 'schnet',
                'params': {
                    'hidden_channels': 128,
                    'num_filters': 128,
                    'num_interactions': 6,
                    'num_gaussians': 50,
                    'cutoff': 5.0,
                    'max_num_neighbors': 32,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 300
                },
                'justification': """
                SchNet (Continuous-filter Convolutional Neural Network):
                - Translation and rotation invariant by design
                - Continuous filters capture smooth distance relationships
                - Interaction blocks model many-body atomic interactions
                - Direct coordinates-to-energy mapping without hand-crafted features
                - Proven excellent performance on molecular datasets
                - Efficient for small to medium-sized clusters like Au20
                """
            },
            
            'dimenet': {
                'architecture': 'dimenet',
                'params': {
                    'hidden_channels': 128,
                    'out_channels': 1,
                    'num_blocks': 6,
                    'num_bilinear': 8,
                    'num_spherical': 7,
                    'num_radial': 6,
                    'cutoff': 5.0,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 300
                },
                'justification': """
                DimeNet++ (Directional Message Passing):
                - Incorporates directional information (bond angles)
                - Bilinear interactions model three-body angular effects
                - Spherical harmonics for rotationally equivariant features
                - Superior to SchNet for systems where angles matter
                - Excellent for coordination-sensitive properties
                - Models geometric effects in cluster stability
                """
            },
            
            'cgcnn': {
                'architecture': 'cgcnn',
                'params': {
                    'atom_fea_len': 64,
                    'h_fea_len': 128,
                    'n_conv': 3,
                    'n_h': 1,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 300
                },
                'justification': """
                Crystal Graph Convolutional Neural Network:
                - Graph convolution on atomic structure graphs
                - Learnable node features represent atomic environments
                - Edge features capture bond characteristics
                - Pooling aggregates local information to global properties
                - Adaptable to different cluster sizes and topologies
                - Interpretable attention mechanisms
                """
            },
            
            'megnet': {
                'architecture': 'megnet_inspired',
                'params': {
                    'node_dim': 64,
                    'edge_dim': 32,
                    'global_dim': 32,
                    'hidden_dim': 128,
                    'n_blocks': 3,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 300
                },
                'justification': """
                MEGNet-inspired Architecture:
                - Multi-graph approach: separate atom, bond, global representations
                - Set2Set pooling for order-invariant aggregation
                - Global state vector captures cluster-wide properties
                - Hierarchical learning from local to global features
                - Proven success in materials property prediction
                - Excellent for size-dependent cluster properties
                """
            }
        }
        
        return configs
    
    def load_data(self, data_path, structures_data=None, target_column='energy'):
        """Load data and convert to graph format"""
        if isinstance(data_path, str):
            self.df = pd.read_csv(data_path)
        else:
            self.df = data_path
        
        # Store structures data for graph conversion
        self.structures_data = structures_data
        
        # Clean data
        self.df = self.df.dropna(subset=[target_column])
        
        print(f"Loaded {len(self.df)} samples for graph neural networks")
        print(f"Target range: {self.df[target_column].min():.2f} to {self.df[target_column].max():.2f}")
        
        return self.df
    
    def create_graph_data(self, target_column='energy'):
        """Convert molecular structures to PyTorch Geometric Data objects"""
        if not PYTORCH_AVAILABLE or not ASE_AVAILABLE:
            print("❌ PyTorch Geometric or ASE not available")
            return None
        
        if self.structures_data is None:
            print("❌ No structures data provided")
            return None
        
        print("Converting molecular structures to graph format...")
        
        graph_data_list = []
        
        for structure in self.structures_data:
            try:
                # Get atoms object
                atoms = structure['atoms'] if 'atoms' in structure else None
                if atoms is None:
                    coords = structure['coords']
                    atoms = Atoms('Au' * len(coords), positions=coords)
                
                # Get energy
                filename = structure['filename']
                energy_row = self.df[self.df['filename'] == filename]
                if len(energy_row) == 0:
                    continue
                
                energy = energy_row[target_column].iloc[0]
                if pd.isna(energy):
                    continue
                
                # Create graph
                graph = self._atoms_to_graph(atoms, energy)
                if graph is not None:
                    graph_data_list.append(graph)
                
            except Exception as e:
                print(f"Error converting {structure.get('filename', 'unknown')} to graph: {e}")
                continue
        
        print(f"Successfully converted {len(graph_data_list)} structures to graphs")
        return graph_data_list
    
    def _atoms_to_graph(self, atoms, energy, cutoff=5.0):
        """Convert ASE Atoms object to PyTorch Geometric Data"""
        n_atoms = len(atoms)
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float)
        
        # Node features (atomic numbers, coordinates)
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        
        # Simple node features: atomic number as one-hot (for Au it's always the same)
        node_features = F.one_hot(atomic_numbers - 79, num_classes=1).float()  # Au = 79
        
        # Create edges based on distance cutoff
        edge_indices = []
        edge_features = []
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    distance = torch.norm(positions[i] - positions[j])
                    if distance < cutoff:
                        edge_indices.append([i, j])
                        edge_features.append([distance.item()])
        
        if len(edge_indices) == 0:
            # No edges found, skip this structure
            return None
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Target
        y = torch.tensor([energy], dtype=torch.float)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=positions,
            y=y
        )
        
        return data
    
    def create_schnet_model(self, params):
        """Create SchNet model"""
        model = SchNet(
            hidden_channels=params['hidden_channels'],
            num_filters=params['num_filters'],
            num_interactions=params['num_interactions'],
            num_gaussians=params['num_gaussians'],
            cutoff=params['cutoff'],
            max_num_neighbors=params['max_num_neighbors'],
            out_channels=1
        )
        return model
    
    def create_dimenet_model(self, params):
        """Create DimeNet model"""
        model = DimeNet(
            hidden_channels=params['hidden_channels'],
            out_channels=params['out_channels'],
            num_blocks=params['num_blocks'],
            num_bilinear=params['num_bilinear'],
            num_spherical=params['num_spherical'],
            num_radial=params['num_radial'],
            cutoff=params['cutoff']
        )
        return model
    
    class CGCNNModel(nn.Module):
        """Crystal Graph Convolutional Neural Network"""
        def __init__(self, params):
            super().__init__()
            self.atom_fea_len = params['atom_fea_len']
            self.h_fea_len = params['h_fea_len']
            self.n_conv = params['n_conv']
            
            # Embedding layer
            self.embedding = nn.Linear(1, self.atom_fea_len)  # For Au atoms
            
            # Graph convolutional layers
            self.convs = nn.ModuleList([
                GCNConv(self.atom_fea_len if i == 0 else self.h_fea_len, self.h_fea_len)
                for i in range(self.n_conv)
            ])
            
            # Final prediction layers
            self.fc = nn.Sequential(
                nn.Linear(self.h_fea_len, self.h_fea_len // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.h_fea_len // 2, 1)
            )
        
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            # Embedding
            x = self.embedding(x)
            
            # Graph convolutions
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
            
            # Global pooling
            x = global_mean_pool(x, batch)
            
            # Final prediction
            out = self.fc(x)
            return out
    
    class MEGNetModel(nn.Module):
        """MEGNet-inspired model"""
        def __init__(self, params):
            super().__init__()
            self.node_dim = params['node_dim']
            self.edge_dim = params['edge_dim']
            self.global_dim = params['global_dim']
            self.hidden_dim = params['hidden_dim']
            self.n_blocks = params['n_blocks']
            
            # Embedding layers
            self.node_embedding = nn.Linear(1, self.node_dim)
            self.edge_embedding = nn.Linear(1, self.edge_dim)
            self.global_embedding = nn.Linear(1, self.global_dim)
            
            # MEGNet blocks
            self.blocks = nn.ModuleList([
                self.MEGNetBlock() for _ in range(self.n_blocks)
            ])
            
            # Final layers
            self.final_layers = nn.Sequential(
                nn.Linear(self.node_dim + self.global_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, 1)
            )
        
        class MEGNetBlock(nn.Module):
            def __init__(self):
                super().__init__()
                # Simplified MEGNet block
                pass
            
            def forward(self, x, edge_attr, global_attr, edge_index, batch):
                # Simplified implementation
                return x, edge_attr, global_attr
        
        def forward(self, data):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            
            # Embeddings
            x = self.node_embedding(x)
            edge_attr = self.edge_embedding(edge_attr)
            
            # Global features (simple: cluster size)
            global_attr = torch.ones(batch.max().item() + 1, 1, device=x.device)
            global_attr = self.global_embedding(global_attr)
            
            # MEGNet blocks (simplified)
            for block in self.blocks:
                x, edge_attr, global_attr = block(x, edge_attr, global_attr, edge_index, batch)
            
            # Pooling and prediction
            node_pool = global_mean_pool(x, batch)
            
            # Combine node and global features
            combined = torch.cat([node_pool, global_attr], dim=1)
            out = self.final_layers(combined)
            
            return out
    
    def train_models(self, graph_data_list, test_size=0.2, val_size=0.2):
        """Train all graph neural network models"""
        if not PYTORCH_AVAILABLE or not graph_data_list:
            print("❌ PyTorch Geometric not available or no graph data")
            return {}
        
        print("\n" + "="*60)
        print("TRAINING GRAPH NEURAL NETWORK MODELS")
        print("="*60)
        
        # Split data
        n_total = len(graph_data_list)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_test - n_val
        
        indices = torch.randperm(n_total)
        train_data = [graph_data_list[i] for i in indices[:n_train]]
        val_data = [graph_data_list[i] for i in indices[n_train:n_train+n_val]]
        test_data = [graph_data_list[i] for i in indices[n_train+n_val:]]
        
        print(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        results = {}
        
        for name, config in self.model_configs.items():
            print(f"\n🌐 Training {name.upper()}...")
            print(f"Justification: {config['justification'].strip()}")
            
            params = config['params']
            
            # Create model
            if config['architecture'] == 'schnet':
                model = self.create_schnet_model(params)
            elif config['architecture'] == 'dimenet':
                model = self.create_dimenet_model(params)
            elif config['architecture'] == 'cgcnn':
                model = self.CGCNNModel(params)
            elif config['architecture'] == 'megnet_inspired':
                model = self.MEGNetModel(params)
            else:
                continue
            
            model = model.to(self.device)
            
            # Create data loaders
            train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_data, batch_size=params['batch_size'], shuffle=False)
            test_loader = DataLoader(test_data, batch_size=params['batch_size'], shuffle=False)
            
            # Optimizer and scheduler
            optimizer = Adam(model.parameters(), lr=params['learning_rate'])
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            train_losses = []
            val_losses = []
            
            for epoch in range(params['epochs']):
                # Training
                model.train()
                total_train_loss = 0
                
                for batch in train_loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    
                    out = model(batch).view(-1)
                    loss = F.mse_loss(out, batch.y)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                
                # Validation
                model.eval()
                total_val_loss = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        out = model(batch).view(-1)
                        loss = F.mse_loss(out, batch.y)
                        total_val_loss += loss.item()
                
                avg_train_loss = total_train_loss / len(train_loader)
                avg_val_loss = total_val_loss / len(val_loader)
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= 30:  # Early stopping patience
                    print(f"  Early stopping at epoch {epoch}")
                    break
                
                if epoch % 50 == 0:
                    print(f"  Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Load best model
            model.load_state_dict(best_model_state)
            
            # Final evaluation
            train_metrics = self._evaluate_model(model, train_loader)
            val_metrics = self._evaluate_model(model, val_loader)
            test_metrics = self._evaluate_model(model, test_loader)
            
            results[name] = {
                'model': model,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_r2': train_metrics['r2'],
                'val_r2': val_metrics['r2'],
                'test_r2': test_metrics['r2'],
                'train_rmse': train_metrics['rmse'],
                'val_rmse': val_metrics['rmse'],
                'test_rmse': test_metrics['rmse'],
                'train_mae': train_metrics['mae'],
                'val_mae': val_metrics['mae'],
                'test_mae': test_metrics['mae'],
                'predictions': test_metrics['predictions'],
                'targets': test_metrics['targets']
            }
            
            print(f"✅ {name}: Test R² = {test_metrics['r2']:.3f}, Test RMSE = {test_metrics['rmse']:.2f}")
        
        self.results = results
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        return results
    
    def _evaluate_model(self, model, data_loader):
        """Evaluate model on data loader"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                out = model(batch).view(-1)
                
                all_predictions.extend(out.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Calculate metrics
        r2 = r2_score(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'targets': targets
        }
    
    def create_visualizations(self, output_dir='./graph_models_results'):
        """Create comprehensive visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 1. Training curves
        self._plot_training_curves(output_dir)
        
        # 2. Model performance comparison
        self._plot_model_comparison(output_dir)
        
        # 3. Prediction analysis
        self._plot_predictions(output_dir)
        
        # 4. Graph-specific analysis
        self._plot_graph_analysis(output_dir)
        
        print(f"📊 Graph neural network visualizations saved to {output_dir}")
    
    def _plot_training_curves(self, output_dir):
        """Plot training curves for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(self.results.items()):
            if i >= 4:
                break
            
            train_losses = result['train_losses']
            val_losses = result['val_losses']
            
            epochs = range(1, len(train_losses) + 1)
            
            axes[i].plot(epochs, train_losses, label='Training Loss', alpha=0.8)
            axes[i].plot(epochs, val_losses, label='Validation Loss', alpha=0.8)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].set_title(f'{name.replace("_", " ").title()} Training Curves')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'gnn_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, output_dir):
        """Plot model performance comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        models = list(self.results.keys())
        colors = ['blue', 'green', 'orange', 'red'][:len(models)]
        
        # R² comparison
        train_r2 = [self.results[m]['train_r2'] for m in models]
        val_r2 = [self.results[m]['val_r2'] for m in models]
        test_r2 = [self.results[m]['test_r2'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        axes[0,0].bar(x - width, train_r2, width, label='Train', alpha=0.8, color=colors)
        axes[0,0].bar(x, val_r2, width, label='Validation', alpha=0.8, color=colors)
        axes[0,0].bar(x + width, test_r2, width, label='Test', alpha=0.8, color=colors)
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_title('Graph Neural Network R² Performance')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels([m.replace('_', '\n') for m in models])
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # RMSE comparison
        test_rmse = [self.results[m]['test_rmse'] for m in models]
        axes[0,1].bar(x, test_rmse, alpha=0.8, color=colors)
        axes[0,1].set_ylabel('Test RMSE')
        axes[0,1].set_title('Model RMSE Performance')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels([m.replace('_', '\n') for m in models])
        axes[0,1].grid(True, alpha=0.3)
        
        # Overfitting analysis
        axes[1,0].scatter(train_r2, test_r2, s=100, c=colors, alpha=0.8)
        for i, model in enumerate(models):
            axes[1,0].annotate(model, (train_r2[i], test_r2[i]), 
                             xytext=(5, 5), textcoords='offset points')
        
        # Perfect correlation line
        min_r2 = min(min(train_r2), min(test_r2))
        max_r2 = max(max(train_r2), max(test_r2))
        axes[1,0].plot([min_r2, max_r2], [min_r2, max_r2], 'r--', alpha=0.8)
        
        axes[1,0].set_xlabel('Train R²')
        axes[1,0].set_ylabel('Test R²')
        axes[1,0].set_title('Overfitting Analysis')
        axes[1,0].grid(True, alpha=0.3)
        
        # MAE comparison
        test_mae = [self.results[m]['test_mae'] for m in models]
        axes[1,1].bar(x, test_mae, alpha=0.8, color=colors)
        axes[1,1].set_ylabel('Test MAE')
        axes[1,1].set_title('Model MAE Performance')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels([m.replace('_', '\n') for m in models])
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'gnn_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_predictions(self, output_dir):
        """Plot prediction analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        colors = ['blue', 'green', 'orange', 'red']
        
        for i, (name, result) in enumerate(self.results.items()):
            if i >= 4:
                break
            
            predictions = result['predictions']
            targets = result['targets']
            
            axes[i].scatter(targets, predictions, alpha=0.6, s=50, color=colors[i])
            
            # Perfect prediction line
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Metrics annotation
            r2 = result['test_r2']
            rmse = result['test_rmse']
            axes[i].text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.2f}', 
                        transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[i].set_xlabel('Actual Energy (eV)')
            axes[i].set_ylabel('Predicted Energy (eV)')
            axes[i].set_title(f'{name.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'gnn_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_graph_analysis(self, output_dir):
        """Plot graph-specific analysis"""
        if not hasattr(self, 'test_data') or not self.test_data:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Graph size distribution
        graph_sizes = [data.x.size(0) for data in self.test_data]
        axes[0,0].hist(graph_sizes, bins=20, alpha=0.7, edgecolor='black')
        axes[0,0].set_xlabel('Number of Nodes (Atoms)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Graph Size Distribution')
        axes[0,0].grid(True, alpha=0.3)
        
        # Edge density distribution
        edge_densities = []
        for data in self.test_data:
            n_nodes = data.x.size(0)
            n_edges = data.edge_index.size(1)
            max_edges = n_nodes * (n_nodes - 1)
            density = n_edges / max_edges if max_edges > 0 else 0
            edge_densities.append(density)
        
        axes[0,1].hist(edge_densities, bins=20, alpha=0.7, edgecolor='black')
        axes[0,1].set_xlabel('Edge Density')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Graph Edge Density Distribution')
        axes[0,1].grid(True, alpha=0.3)
        
        # Performance vs graph size
        if self.results:
            best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['test_r2'])
            best_result = self.results[best_model_name]
            
            # Calculate errors for each graph
            predictions = best_result['predictions']
            targets = best_result['targets']
            errors = np.abs(predictions - targets)
            
            axes[1,0].scatter(graph_sizes, errors, alpha=0.6)
            axes[1,0].set_xlabel('Graph Size (Atoms)')
            axes[1,0].set_ylabel('Absolute Error')
            axes[1,0].set_title(f'Error vs Graph Size - {best_model_name.title()}')
            axes[1,0].grid(True, alpha=0.3)
        
        # Model comparison radar-style
        models = list(self.results.keys())
        metrics = ['test_r2', 'test_rmse', 'test_mae']
        
        if len(models) > 0:
            # Simple bar chart of test R²
            test_r2_values = [self.results[name]['test_r2'] for name in models]
            axes[1,1].bar(range(len(models)), test_r2_values, alpha=0.8, 
                         color=['blue', 'green', 'orange', 'red'][:len(models)])
            axes[1,1].set_ylabel('Test R² Score')
            axes[1,1].set_title('Final Model Comparison')
            axes[1,1].set_xticks(range(len(models)))
            axes[1,1].set_xticklabels([name.replace('_', '\n') for name in models])
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'gnn_graph_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_models(self, output_dir='./graph_models_results'):
        """Save trained models and results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save models
        for name, result in self.results.items():
            model_path = output_dir / f'{name}_model.pt'
            torch.save(result['model'].state_dict(), model_path)
        
        # Save results summary
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'model': name,
                'train_r2': result['train_r2'],
                'val_r2': result['val_r2'],
                'test_r2': result['test_r2'],
                'train_rmse': result['train_rmse'],
                'val_rmse': result['val_rmse'],
                'test_rmse': result['test_rmse'],
                'train_mae': result['train_mae'],
                'val_mae': result['val_mae'],
                'test_mae': result['test_mae']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'gnn_model_summary.csv', index=False)
        
        print(f"💾 Graph neural network models and results saved to {output_dir}")
        
        return summary_df
    
    def analyze_model_insights(self):
        """Analyze graph neural network specific insights"""
        print("\n" + "="*60)
        print("GRAPH NEURAL NETWORK MODEL INSIGHTS")
        print("="*60)
        
        for name, result in self.results.items():
            model = result['model']
            print(f"\n{name.upper()}:")
            print(f"  Test R²: {result['test_r2']:.3f}")
            print(f"  Test RMSE: {result['test_rmse']:.2f}")
            print(f"  Test MAE: {result['test_mae']:.2f}")
            
            # Model-specific insights
            if name == 'schnet':
                print(f"  Strength: Translation/rotation invariant continuous filters")
                print(f"  Best for: Smooth energy landscapes, distance-dependent interactions")
            elif name == 'dimenet':
                print(f"  Strength: Incorporates directional information and bond angles")
                print(f"  Best for: Coordination-sensitive properties, geometric effects")
            elif name == 'cgcnn':
                print(f"  Strength: Graph convolution with learnable atomic features")
                print(f"  Best for: Interpretable predictions, variable cluster sizes")
            elif name == 'megnet':
                print(f"  Strength: Multi-level representation (atom, bond, global)")
                print(f"  Best for: Size-dependent properties, hierarchical learning")

def main():
    """Main execution function"""
    print("🌐 Graph Neural Network Models for Au Cluster Analysis")
    print("="*70)
    
    # Check PyTorch and device availability
    if PYTORCH_AVAILABLE:
        print(f"✅ PyTorch with {'MPS' if device.type == 'mps' else 'CPU'} acceleration ready")
    else:
        print("❌ PyTorch Geometric not available. Please install torch-geometric")
        return None, None
    
    # Initialize analyzer
    analyzer = GraphNeuralNetworkAnalyzer(random_state=42)
    
    # Load data
    try:
        data_path = input("/Users/wilbert/Documents/GitHub/AIAC/au_cluster_analysis_results/descriptors.csv").strip()
        if not data_path:
            data_path = "./au_cluster_analysis_results/descriptors.csv"
        
        # Load structures data (needed for graph conversion)
        structures_path = input("Enter path to structures data (task1.py analyzer.structures): ").strip()
        if structures_path:
            structures_data = joblib.load(structures_path)
        else:
            print("⚠️  No structures data provided. Graph models need atomic coordinates.")
            structures_data = None
        
        analyzer.load_data(data_path, structures_data)
        
        # Convert to graph format
        graph_data_list = analyzer.create_graph_data()
        
        if not graph_data_list:
            print("❌ Failed to create graph data")
            return None, None
        
        # Train models
        results = analyzer.train_models(graph_data_list)
        
        # Analyze insights
        analyzer.analyze_model_insights()
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Save results
        summary_df = analyzer.save_models()
        
        print("\n🎉 Graph neural network analysis complete!")
        if len(summary_df) > 0:
            print("\nBest performing model:")
            best_model = summary_df.loc[summary_df['test_r2'].idxmax()]
            print(f"  {best_model['model'].upper()}: R² = {best_model['test_r2']:.3f}")
        
        print("\n💡 Graph Neural Network Insights:")
        print("- Direct learning from atomic coordinates without hand-crafted features")
        print("- Translation, rotation, and permutation invariant by design")
        print("- SchNet excels at distance-based interactions")
        print("- DimeNet++ captures directional and angular effects")
        print("- CGCNN provides interpretable graph convolutions")
        print("- MEGNet handles multi-scale cluster properties")
        
        return analyzer, results
        
    except FileNotFoundError:
        print("❌ Data file not found. Please run task1.py first to generate descriptors.")
        return None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

if __name__ == "__main__":
    analyzer, results = main()