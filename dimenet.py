import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_mean_pool
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import warnings
warnings.filterwarnings('ignore')

# ASE for molecular structures
try:
    from ase.atoms import Atoms
    ASE_AVAILABLE = True
except ImportError:
    print("ASE not available")
    ASE_AVAILABLE = False

# Device detection
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = torch.device("cpu")
    print("Using CPU")

class DimeNetPlusPlus(nn.Module):
    """
    Custom DimeNet++ implementation without torch-cluster dependency
    
    DimeNet++ captures directional information and 3-body interactions
    making it highly effective for molecular property prediction
    """
    
    def __init__(self, hidden_channels=128, out_channels=1, num_blocks=4, 
                 int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
                 num_spherical=7, num_radial=6, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.cutoff = cutoff
        
        # Atomic embeddings
        self.atom_emb = nn.Embedding(100, hidden_channels)
        
        # Radial basis functions
        self.rbf = RadialBasisFunction(num_radial, cutoff, envelope_exponent)
        
        # Spherical basis functions  
        self.sbf = SphericalBasisFunction(num_spherical, num_radial, cutoff, envelope_exponent)
        
        # Embedding layers
        self.emb = EmbeddingBlock(num_radial, hidden_channels, int_emb_size)
        
        # Interaction blocks
        self.interaction_blocks = nn.ModuleList([
            InteractionPPBlock(
                hidden_channels, int_emb_size, basis_emb_size, 
                num_spherical, num_radial, cutoff
            ) for _ in range(num_blocks)
        ])
        
        # Output blocks
        self.output_blocks = nn.ModuleList([
            OutputPPBlock(
                num_radial, hidden_channels, out_emb_channels, 
                out_channels, num_layers=3
            ) for _ in range(num_blocks + 1)
        ])
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.atom_emb.reset_parameters()
        for block in self.interaction_blocks:
            block.reset_parameters()
        for block in self.output_blocks:
            block.reset_parameters()
    
    def triplets(self, edge_index, num_nodes):
        """Generate triplets (i, j, k) where j is the central atom"""
        row, col = edge_index
        value = torch.arange(row.size(0), device=row.device)
        
        # Find neighbors for each node
        adj_t = torch.sparse_coo_tensor(
            torch.stack([col, row]), value, (num_nodes, num_nodes)
        ).coalesce()
        
        triplets = []
        for node in range(num_nodes):
            neighbors = adj_t[node].indices()
            if len(neighbors) >= 2:
                for i in range(len(neighbors)):
                    for k in range(i + 1, len(neighbors)):
                        # Triplet: (neighbors[i], node, neighbors[k])
                        triplets.append([neighbors[i].item(), node, neighbors[k].item()])
        
        if len(triplets) == 0:
            return torch.empty((0, 3), dtype=torch.long, device=edge_index.device)
        
        return torch.tensor(triplets, dtype=torch.long, device=edge_index.device)
    
    def forward(self, z, pos, batch):
        # Build edges
        edge_index, edge_weight = self.build_edges(pos, batch)
        
        if edge_index.size(1) == 0:
            # No edges, return simple embedding
            x = self.atom_emb(z)
            return global_mean_pool(x, batch).sum(dim=1, keepdim=True)
        
        # Compute distances and angles
        edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
        edge_attr = torch.norm(edge_vec, dim=1)
        
        # Get triplets for angle computation
        num_nodes = pos.size(0)
        triplets = self.triplets(edge_index, num_nodes)
        
        if triplets.size(0) == 0:
            # No triplets, use simpler approach
            x = self.atom_emb(z)
            edge_attr_emb = self.rbf(edge_attr)
            x = self.emb(x, edge_index, edge_attr_emb)
            
            # Simple message passing
            for interaction in self.interaction_blocks:
                x = interaction.simple_forward(x, edge_index, edge_attr_emb)
            
            return global_mean_pool(x, batch).sum(dim=1, keepdim=True)
        
        # Compute angles
        angle_indices = triplets[:, [0, 2]]  # i, k indices
        j_indices = triplets[:, 1]  # j indices (central atoms)
        
        vec1 = pos[angle_indices[:, 0]] - pos[j_indices]
        vec2 = pos[angle_indices[:, 1]] - pos[j_indices]
        
        cos_angles = F.cosine_similarity(vec1, vec2, dim=1)
        angles = torch.acos(torch.clamp(cos_angles, -1 + 1e-7, 1 - 1e-7))
        
        # Basis functions
        rbf = self.rbf(edge_attr)
        
        # For spherical basis, we need edge distances for the triplets
        triplet_edge_distances = []
        for triplet in triplets:
            i, j, k = triplet
            dist_ij = torch.norm(pos[i] - pos[j])
            dist_jk = torch.norm(pos[j] - pos[k])
            triplet_edge_distances.append([dist_ij, dist_jk])
        
        if len(triplet_edge_distances) > 0:
            triplet_distances = torch.tensor(triplet_edge_distances, device=pos.device)
            sbf = self.sbf(triplet_distances, angles)
        else:
            sbf = torch.empty((0, self.sbf.num_spherical * self.sbf.num_radial), device=pos.device)
        
        # Initialize node features
        x = self.atom_emb(z)
        x = self.emb(x, edge_index, rbf)
        
        # Collect outputs for skip connections
        P = self.output_blocks[0](x, rbf, edge_index)
        
        # Interaction blocks
        for i, interaction in enumerate(self.interaction_blocks):
            x = interaction(x, rbf, sbf, edge_index, triplets)
            P += self.output_blocks[i + 1](x, rbf, edge_index)
        
        # Global pooling
        return global_mean_pool(P, batch)
    
    def build_edges(self, pos, batch, cutoff=None):
        """Build edges based on distance cutoff"""
        if cutoff is None:
            cutoff = self.cutoff
        
        batch_size = batch.max().item() + 1
        edge_indices = []
        edge_weights = []
        
        for b in range(batch_size):
            mask = batch == b
            if mask.sum() < 2:
                continue
            
            batch_pos = pos[mask]
            n_atoms = batch_pos.size(0)
            
            for i in range(n_atoms):
                for j in range(n_atoms):
                    if i != j:
                        dist = torch.norm(batch_pos[i] - batch_pos[j])
                        if dist <= cutoff:
                            global_i = torch.where(mask)[0][i]
                            global_j = torch.where(mask)[0][j]
                            edge_indices.append([global_i.item(), global_j.item()])
                            edge_weights.append(dist.item())
        
        if len(edge_indices) == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=pos.device), torch.zeros(0, device=pos.device)
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long, device=pos.device).t()
        edge_weight = torch.tensor(edge_weights, device=pos.device)
        
        return edge_index, edge_weight

class RadialBasisFunction(nn.Module):
    def __init__(self, num_radial, cutoff, envelope_exponent=5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        
        self.freq = nn.Parameter(torch.Tensor(num_radial))
        self.reset_parameters()
    
    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, len(self.freq) + 1, out=self.freq).mul_(np.pi)
    
    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * torch.sin(self.freq * dist)

class SphericalBasisFunction(nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent=5):
        super().__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        
        # Bessel basis
        self.freq = nn.Parameter(torch.Tensor(num_radial))
        self.reset_parameters()
    
    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, len(self.freq) + 1, out=self.freq).mul_(np.pi)
    
    def forward(self, dist, angle):
        # Simplified spherical harmonics
        dist = dist.unsqueeze(-1) / self.cutoff
        rbf = self.envelope(dist) * torch.sin(self.freq * dist)
        
        # Legendre polynomials for angular part
        angle = angle.unsqueeze(-1)
        spherical = torch.ones_like(angle)
        
        if self.num_spherical > 1:
            spherical = torch.cat([spherical, angle], dim=-1)
        if self.num_spherical > 2:
            spherical = torch.cat([spherical, 0.5 * (3 * angle**2 - 1)], dim=-1)
        if self.num_spherical > 3:
            spherical = torch.cat([spherical, 0.5 * (5 * angle**3 - 3 * angle)], dim=-1)
        
        # Truncate to desired size
        spherical = spherical[:, :self.num_spherical]
        
        # Combine radial and angular parts
        sbf = rbf.unsqueeze(-2) * spherical.unsqueeze(-1)
        return sbf.reshape(sbf.size(0), -1)

class Envelope(nn.Module):
    def __init__(self, exponent):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2
    
    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2

class EmbeddingBlock(nn.Module):
    def __init__(self, num_radial, hidden_channels, int_emb_size):
        super().__init__()
        self.emb = nn.Linear(num_radial, int_emb_size)
        self.lin = nn.Linear(int_emb_size, hidden_channels)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.emb.reset_parameters()
        self.lin.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr):
        edge_emb = self.emb(edge_attr)
        edge_emb = F.silu(edge_emb)
        edge_emb = self.lin(edge_emb)
        
        # Simple message passing
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, col, edge_emb)
        
        return x + out

class InteractionPPBlock(nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size, 
                 num_spherical, num_radial, cutoff):
        super().__init__()
        
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)
        
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)
        
        self.lin_t1 = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_t2 = nn.Linear(int_emb_size, hidden_channels)
        
        self.lin_up = nn.Linear(hidden_channels, hidden_channels)
        self.lin_down = nn.Linear(hidden_channels, hidden_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in [self.lin_rbf1, self.lin_rbf2, self.lin_sbf1, self.lin_sbf2,
                      self.lin_t1, self.lin_t2, self.lin_up, self.lin_down]:
            layer.reset_parameters()
    
    def forward(self, x, rbf, sbf, edge_index, triplets):
        # If no triplets, use simple forward
        if triplets.size(0) == 0:
            return self.simple_forward(x, edge_index, rbf)
        
        # Standard DimeNet++ interaction
        x_up = self.lin_up(x)
        x_down = self.lin_down(x)
        
        # Edge message passing
        rbf_emb = self.lin_rbf1(rbf)
        rbf_emb = F.silu(rbf_emb)
        rbf_emb = self.lin_rbf2(rbf_emb)
        
        row, col = edge_index
        x_edge = x_up[row] * rbf_emb
        
        # Triplet interactions
        if sbf.size(0) > 0:
            sbf_emb = self.lin_sbf1(sbf)
            sbf_emb = F.silu(sbf_emb)
            sbf_emb = self.lin_sbf2(sbf_emb)
            
            t = self.lin_t1(x_down)
            t = F.silu(t)
            t = self.lin_t2(t)
            
            # Aggregate triplet contributions
            for i, triplet in enumerate(triplets):
                if i < sbf_emb.size(0):
                    j_idx = triplet[1]
                    x_edge += t[j_idx] * sbf_emb[i]
        
        # Aggregate messages
        out = torch.zeros_like(x)
        out.index_add_(0, col, x_edge)
        
        return x + out
    
    def simple_forward(self, x, edge_index, rbf):
        """Simplified forward without triplet interactions"""
        rbf_emb = self.lin_rbf1(rbf)
        rbf_emb = F.silu(rbf_emb)
        rbf_emb = self.lin_rbf2(rbf_emb)
        
        row, col = edge_index
        x_up = self.lin_up(x)
        edge_msg = x_up[row] * rbf_emb
        
        out = torch.zeros_like(x)
        out.index_add_(0, col, edge_msg)
        
        return x + out

class OutputPPBlock(nn.Module):
    def __init__(self, num_radial, hidden_channels, out_emb_channels, 
                 out_channels, num_layers=2):
        super().__init__()
        
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(hidden_channels, out_emb_channels // 2))
            elif i == num_layers - 1:
                layers.append(nn.Linear(out_emb_channels // 2, out_channels))
            else:
                layers.append(nn.Linear(out_emb_channels // 2, out_emb_channels // 2))
            
            if i < num_layers - 1:
                layers.append(nn.SiLU())
        
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_rbf.reset_parameters()
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def forward(self, x, rbf, edge_index):
        rbf_emb = self.lin_rbf(rbf)
        row, col = edge_index
        
        x_edge = x[row] * rbf_emb
        
        # Aggregate to nodes
        out = torch.zeros_like(x)
        out.index_add_(0, col, x_edge)
        
        return self.mlp(out)

class DimeNetTrainer:
    """Standalone DimeNet trainer for Au cluster analysis"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
        
        self.device = device
        self.model = None
        self.results = {}
    
    def load_data(self, data_path):
        """Load data from CSV file"""
        print(f"Loading data from: {data_path}")
        
        if isinstance(data_path, str):
            self.df = pd.read_csv(data_path)
        else:
            self.df = data_path
        
        # Check if CSV has coordinate data
        coord_cols = ['x', 'y', 'z']
        if all(col in self.df.columns for col in coord_cols):
            print("CSV contains coordinate data - reconstructing molecular structures")
            self.structures_data = self._reconstruct_structures_from_csv()
        else:
            raise ValueError("CSV must contain coordinate columns: x, y, z")
        
        return self.df
    
    def _reconstruct_structures_from_csv(self):
        """Reconstruct molecular structures from CSV"""
        structures = []
        grouped = self.df.groupby('filename')
        
        for filename, group in grouped:
            try:
                coords = group[['x', 'y', 'z']].values
                elements = group['element'].values if 'element' in group.columns else ['Au'] * len(coords)
                energy = group['energy'].iloc[0] if 'energy' in group.columns else None
                
                if energy is None or pd.isna(energy):
                    continue
                
                if ASE_AVAILABLE:
                    atoms = Atoms(symbols=elements, positions=coords)
                else:
                    atoms = None
                
                structures.append({
                    'filename': filename,
                    'n_atoms': len(coords),
                    'energy': energy,
                    'atoms': atoms,
                    'coords': coords,
                    'elements': elements
                })
                
            except Exception as e:
                print(f"Error reconstructing structure {filename}: {e}")
                continue
        
        print(f"Successfully reconstructed {len(structures)} molecular structures")
        return structures
    
    def create_graph_data(self):
        """Convert structures to graph format"""
        print("Converting molecular structures to graph format...")
        
        graph_data_list = []
        
        for structure in self.structures_data:
            try:
                atoms = structure['atoms']
                energy = structure['energy']
                
                if atoms is None and ASE_AVAILABLE:
                    coords = structure['coords']
                    elements = structure['elements']
                    atoms = Atoms(symbols=elements, positions=coords)
                elif atoms is None:
                    continue
                
                graph = self._atoms_to_graph(atoms, energy)
                if graph is not None:
                    graph_data_list.append(graph)
                
            except Exception as e:
                print(f"Error converting {structure.get('filename', 'unknown')} to graph: {e}")
                continue
        
        print(f"Successfully converted {len(graph_data_list)} structures to graphs")
        return graph_data_list
    
    def _atoms_to_graph(self, atoms, energy, cutoff=5.0):
        """Convert ASE Atoms to PyTorch Geometric Data"""
        n_atoms = len(atoms)
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float)
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        
        # For DimeNet, we just need atomic numbers and positions
        y = torch.tensor([energy], dtype=torch.float)
        
        data = Data(
            z=atomic_numbers,
            pos=positions,
            y=y
        )
        
        return data
    
    def train_model(self, graph_data_list, test_size=0.2, val_size=0.2):
        """Train DimeNet model"""
        print("\nTraining DimeNet++ Model")
        print("=" * 50)
        
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
        
        # Create model
        self.model = DimeNetPlusPlus(
            hidden_channels=128,
            out_channels=1,
            num_blocks=4,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=256,
            num_spherical=7,
            num_radial=6,
            cutoff=5.0
        ).to(self.device)
        
        # Training parameters
        batch_size = 32
        learning_rate = 0.001
        epochs = 200
        
        # Data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        # Optimizer and scheduler
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            
            for batch in train_loader:
                try:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    
                    out = self.model(batch.z, batch.pos, batch.batch).view(-1)
                    loss = F.mse_loss(out, batch.y)
                    
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  CUDA OOM, clearing cache...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        batch = batch.to(self.device)
                        out = self.model(batch.z, batch.pos, batch.batch).view(-1)
                        loss = F.mse_loss(out, batch.y)
                        total_val_loss += loss.item()
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise e
            
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= 30:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 25 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Time = {elapsed:.1f}s")
        
        # Load best model
        if 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        train_metrics = self._evaluate_model(self.model, train_loader)
        val_metrics = self._evaluate_model(self.model, val_loader)
        test_metrics = self._evaluate_model(self.model, test_loader)
        
        total_time = time.time() - start_time
        
        self.results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_r2': train_metrics['r2'],
            'val_r2': val_metrics['r2'],
            'test_r2': test_metrics['r2'],
            'train_rmse': train_metrics['rmse'],
            'val_rmse': val_metrics['rmse'],
            'test_rmse': test_metrics['rmse'],
            'predictions': test_metrics['predictions'],
            'targets': test_metrics['targets'],
            'training_time': total_time
        }
        
        print(f"\nDimeNet++ Results:")
        print(f"Test R² = {test_metrics['r2']:.3f}")
        print(f"Test RMSE = {test_metrics['rmse']:.3f}")
        print(f"Training time = {total_time:.1f}s")
        
        return self.results
    
    def _evaluate_model(self, model, data_loader):
        """Evaluate model performance"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                try:
                    batch = batch.to(self.device)
                    out = model(batch.z, batch.pos, batch.batch).view(-1)
                    
                    all_predictions.extend(out.cpu().numpy())
                    all_targets.extend(batch.y.cpu().numpy())
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
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
    
    def create_visualizations(self, output_dir='./dimenet_results', show_plots=True):
        """Create and save result visualizations and export predictions.
        
        Produces:
          - train_val_loss.png: training and validation loss curves
          - parity_plot.png: predicted vs target scatter with identity line
          - residual_hist.png: histogram of residuals
          - predictions.csv: CSV with predictions and targets
          - results.json: summary metrics and paths
        """
        if not self.results:
            print("No results to visualize. Run train_model first.")
            return None

        os.makedirs(output_dir, exist_ok=True)

        # 1) Loss curves (if present)
        train_losses = self.results.get('train_losses', [])
        val_losses = self.results.get('val_losses', [])

        if train_losses and val_losses:
            plt.figure(figsize=(8, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.title('Training & Validation Loss')
            plt.legend()
            loss_path = os.path.join(output_dir, 'train_val_loss.png')
            plt.tight_layout()
            plt.savefig(loss_path, dpi=150)
            if show_plots:
                plt.show()
            plt.close()
            print(f"Saved loss curve to: {loss_path}")
        else:
            print("Train/Val loss history not found in results.")

        # 2) Parity plot (predictions vs targets)
        preds = np.asarray(self.results.get('predictions', []))
        targets = np.asarray(self.results.get('targets', []))

        if preds.size and targets.size:
            # Flatten in case shapes are weird
            preds_flat = preds.reshape(-1)
            targets_flat = targets.reshape(-1)

            plt.figure(figsize=(6, 6))
            plt.scatter(targets_flat, preds_flat, alpha=0.6)
            # identity line
            minv = min(targets_flat.min(), preds_flat.min())
            maxv = max(targets_flat.max(), preds_flat.max())
            plt.plot([minv, maxv], [minv, maxv], linestyle='--', linewidth=1)
            plt.xlabel('Target')
            plt.ylabel('Prediction')
            plt.title(f'Parity Plot (R²={self.results.get("test_r2", np.nan):.3f}, RMSE={self.results.get("test_rmse", np.nan):.4f})')
            plt.tight_layout()
            parity_path = os.path.join(output_dir, 'parity_plot.png')
            plt.savefig(parity_path, dpi=150)
            if show_plots:
                plt.show()
            plt.close()
            print(f"Saved parity plot to: {parity_path}")

            # 3) Residual histogram
            residuals = preds_flat - targets_flat
            plt.figure(figsize=(6, 4))
            plt.hist(residuals, bins=40, density=False, alpha=0.7)
            plt.xlabel('Residual (pred - target)')
            plt.ylabel('Count')
            plt.title('Residual Histogram')
            plt.tight_layout()
            resid_path = os.path.join(output_dir, 'residual_hist.png')
            plt.savefig(resid_path, dpi=150)
            if show_plots:
                plt.show()
            plt.close()
            print(f"Saved residual histogram to: {resid_path}")

            # 4) Save predictions CSV
            df_out = pd.DataFrame({
                'target': targets_flat,
                'prediction': preds_flat,
                'residual': residuals
            })
            csv_path = os.path.join(output_dir, 'predictions.csv')
            df_out.to_csv(csv_path, index=False)
            print(f"Saved predictions to: {csv_path}")
        else:
            print("Predictions or targets are empty; skipping parity/residual plots and CSV export.")

        # 5) Save a JSON summary of key metrics and file paths
        summary = {
            'train_r2': float(self.results.get('train_r2', np.nan)),
            'val_r2': float(self.results.get('val_r2', np.nan)),
            'test_r2': float(self.results.get('test_r2', np.nan)),
            'train_rmse': float(self.results.get('train_rmse', np.nan)),
            'val_rmse': float(self.results.get('val_rmse', np.nan)),
            'test_rmse': float(self.results.get('test_rmse', np.nan)),
            'training_time_sec': float(self.results.get('training_time', np.nan)),
            'files': {}
        }

        # Add files if they exist
        for fname in ['train_val_loss.png', 'parity_plot.png', 'residual_hist.png', 'predictions.csv']:
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                summary['files'][fname] = fpath

        json_path = os.path.join(output_dir, 'results_summary.json')
        with open(json_path, 'w') as fh:
            json.dump(summary, fh, indent=2)
        print(f"Saved summary JSON to: {json_path}")

        return summary

    def save_model(self, path='./dimenet_results/model.pt'):
        """Save model state_dict and trainer metadata."""
        if self.model is None:
            print("No model to save.")
            return None

        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_obj = {
            'state_dict': self.model.state_dict(),
            'device': str(self.device)
        }
        torch.save(save_obj, path)
        print(f"Saved model state_dict to: {path}")
        return path

    def load_model(self, path='./dimenet_results/model.pt', map_location=None):
        """Load model state_dict into self.model. If model architecture is not built,
           instantiate a new DimeNetPlusPlus with default args before calling this."""
        if map_location is None:
            map_location = self.device

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        ckpt = torch.load(path, map_location=map_location)
        if self.model is None:
            # instantiate default model with same constructor used in train_model
            self.model = DimeNetPlusPlus(
                hidden_channels=128,
                out_channels=1,
                num_blocks=4,
                int_emb_size=64,
                basis_emb_size=8,
                out_emb_channels=256,
                num_spherical=7,
                num_radial=6,
                cutoff=5.0
            ).to(self.device)

        self.model.load_state_dict(ckpt['state_dict'])
        print(f"Loaded model from: {path}")
        return self.model


# Example usage entry point
if __name__ == "__main__":
    import os
    import json
    # Demonstrate usage of DimeNetTrainer
    # Placeholder file paths
    input_csv = "./raw_coordinates.csv"
    output_dir = "./dimenet_results"
    model_save_path = os.path.join(output_dir, "model.pt")

    # 1. Instantiate the trainer
    trainer = DimeNetTrainer(random_state=42)

    # 2. Load data from CSV
    try:
        trainer.load_data(input_csv)
    except Exception as e:
        print(f"Failed to load data from {input_csv}: {e}")
        exit(1)

    # 3. Convert to graph data
    graph_data_list = trainer.create_graph_data()
    if not graph_data_list:
        print("No graph data was created. Exiting.")
        exit(1)

    # 4. Train the model
    results = trainer.train_model(graph_data_list)

    # 5. Create visualizations and save predictions/plots
    summary = trainer.create_visualizations(output_dir=output_dir, show_plots=False)

    # 6. Save the trained model
    trainer.save_model(path=model_save_path)

    print("\nAll done. Results and model saved in:", output_dir)