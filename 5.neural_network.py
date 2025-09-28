#!/usr/bin/env python3
"""
Category 4: Neural Network Models for Au Cluster Energy Prediction
Models: MLP, Bayesian NN (MC Dropout), Deep Ensemble, VAE
Optimized for PyTorch 2.1.2 with Metal Performance Shaders on MacBook Pro M4
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# PyTorch with CUDA optimization (Metal alternative commented)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    # Device configuration - CUDA primary, Metal alternative
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ PyTorch CUDA GPU available: {device}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    # Uncomment below for Metal Performance Shaders on M4 Mac
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print(f"✅ PyTorch Metal GPU available: {device}")
    else:
        device = torch.device("cpu")
        print("⚠️  No CUDA GPU found, using CPU")
        # print("⚠️  No CUDA/Metal GPU found, using CPU")  # When Metal enabled
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)  # For multi-GPU setups
    # Uncomment below for Metal
    # if torch.backends.mps.is_available():
    #     torch.mps.manual_seed(42)
    
    PYTORCH_AVAILABLE = True
except ImportError:
    print("❌ PyTorch not available")
    PYTORCH_AVAILABLE = False
    device = None

# SOAP descriptors
try:
    from dscribe.descriptors import SOAP
    from ase.atoms import Atoms
    SOAP_AVAILABLE = True
except ImportError:
    print("Warning: DScribe not available. Using basic descriptors only.")
    SOAP_AVAILABLE = False

class MLPModel(nn.Module):
    """Multi-Layer Perceptron for regression"""
    
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.2, activation='relu'):
        super(MLPModel, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        for layer, dropout in zip(self.layers, self.dropout_layers):
            x = self.activation(layer(x))
            x = dropout(x)
        
        return self.output_layer(x)

class BayesianMLP(nn.Module):
    """Bayesian MLP with MC Dropout for uncertainty quantification"""
    
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.1, activation='relu'):
        super(BayesianMLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers with dropout
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            # Use dropout with training=True during inference for MC sampling
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x, training=False):
        for layer, dropout in zip(self.layers, self.dropout_layers):
            x = self.activation(layer(x))
            # Force dropout to be active if training=True (for MC sampling)
            if training:
                x = F.dropout(x, p=dropout.p, training=True)
            else:
                x = dropout(x)
        
        return self.output_layer(x)

class VAEPredictor(nn.Module):
    """Variational Autoencoder + Energy Predictor"""
    
    def __init__(self, input_dim, encoder_layers, latent_dim, decoder_layers, 
                 predictor_layers, dropout_rate=0.2, beta=1.0):
        super(VAEPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        self.encoder = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in encoder_layers:
            self.encoder.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Latent space
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_var = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.ModuleList()
        prev_dim = latent_dim
        for hidden_dim in decoder_layers:
            self.decoder.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.decoder_output = nn.Linear(prev_dim, input_dim)
        
        # Predictor
        self.predictor = nn.ModuleList()
        prev_dim = latent_dim
        for hidden_dim in predictor_layers:
            self.predictor.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.predictor_output = nn.Linear(prev_dim, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
    
    def encode(self, x):
        h = x
        for layer in self.encoder:
            h = self.activation(layer(h))
            h = self.dropout(h)
        
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = z
        for layer in self.decoder:
            h = self.activation(layer(h))
            h = self.dropout(h)
        
        return self.decoder_output(h)
    
    def predict_energy(self, z):
        h = z
        for layer in self.predictor:
            h = self.activation(layer(h))
            h = self.dropout(h)
        
        return self.predictor_output(h)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        energy_pred = self.predict_energy(z)
        
        return reconstruction, energy_pred, mu, log_var

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=20, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class NeuralNetworkAnalyzer:
    """
    Neural Network Models for Au Cluster Analysis using PyTorch
    
    Why Neural Networks for Au Clusters:
    1. Universal Approximation: Can model any continuous function with sufficient neurons
    2. Feature Learning: Automatically discovers optimal feature combinations from SOAP
    3. Non-linear Modeling: Captures complex energy landscapes and phase transitions
    4. Uncertainty Quantification: Bayesian methods provide prediction confidence
    5. Transfer Learning: Pre-trained features can generalize to new cluster types
    6. Scalability: Leverages M4 Metal GPU for accelerated computation
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.device = device
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
        # Uncomment below for Metal
        # if torch.backends.mps.is_available():
        #     torch.mps.manual_seed(random_state)
        
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.soap_features = None
        
        # Neural network configurations
        self.model_configs = self._initialize_models()
    
    def _initialize_models(self):
        """Initialize neural network architectures with justifications"""
        if not PYTORCH_AVAILABLE:
            return {}
        
        configs = {
            'mlp_basic': {
                'architecture': 'basic_mlp',
                'params': {
                    'hidden_layers': [256, 128, 64],
                    'dropout_rate': 0.2,
                    'activation': 'relu',
                    'learning_rate': 0.001,
                    'batch_size': 16,
                    'epochs': 200,
                    'weight_decay': 0.001
                },
                'justification': """
                Basic Multi-Layer Perceptron:
                - Universal function approximator for energy surfaces
                - Hidden layers learn hierarchical feature representations
                - Dropout prevents overfitting on limited cluster data
                - Weight decay ensures smooth energy predictions
                - Optimized for SOAP descriptor input vectors
                - Leverages CUDA GPU acceleration
                """
            },
            
            'mlp_deep': {
                'architecture': 'deep_mlp',
                'params': {
                    'hidden_layers': [512, 256, 128, 64, 32],
                    'dropout_rate': 0.3,
                    'activation': 'relu',
                    'learning_rate': 0.0005,
                    'batch_size': 32,
                    'epochs': 300,
                    'weight_decay': 0.01
                },
                'justification': """
                Deep Multi-Layer Perceptron:
                - Deeper architecture captures more complex patterns
                - Progressive layer size reduction creates feature hierarchy
                - Higher dropout for deeper regularization
                - Slower learning rate for stable deep training
                - Excellent for high-dimensional SOAP features
                - Benefits from CUDA's parallel matrix operations
                """
            },
            
            'bayesian_nn': {
                'architecture': 'mc_dropout',
                'params': {
                    'hidden_layers': [256, 128, 64],
                    'dropout_rate': 0.1,
                    'activation': 'relu',
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 250,
                    'weight_decay': 0.001,
                    'mc_samples': 100
                },
                'justification': """
                Bayesian Neural Network (MC Dropout):
                - Uncertainty quantification for energy predictions
                - Monte Carlo sampling provides confidence intervals
                - Critical for identifying unreliable predictions
                - Epistemic uncertainty captures model limitations
                - Essential for active learning and experimental design
                - Dropout during inference approximates Bayesian posterior
                """
            },
            
            'autoencoder': {
                'architecture': 'vae_predictor',
                'params': {
                    'encoder_layers': [256, 128],
                    'latent_dim': 32,
                    'decoder_layers': [128, 256],
                    'predictor_layers': [64, 32],
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 300,
                    'weight_decay': 0.001,
                    'beta': 1.0
                },
                'justification': """
                Variational Autoencoder + Predictor:
                - Learns compressed representation of SOAP features
                - Latent space captures underlying structural patterns
                - Regularized latent features improve generalization
                - Can generate new cluster structures in latent space
                - Transfer learning: encoder features for related tasks
                - Dimensionality reduction for interpretability
                """
            }
        }
        
        return configs
    
    def load_data(self, data_path, target_column='energy'):
        """Load data from task1.py output"""
        if isinstance(data_path, str):
            self.df = pd.read_csv(data_path)
        else:
            self.df = data_path
        
        # Clean data
        self.df = self.df.dropna(subset=[target_column])
        
        print(f"Loaded {len(self.df)} samples for neural networks")
        print(f"Target range: {self.df[target_column].min():.2f} to {self.df[target_column].max():.2f}")
        
        return self.df
    
    def create_soap_features(self, structures_data=None):
        """Create SOAP descriptors optimized for neural networks"""
        if not SOAP_AVAILABLE or structures_data is None:
            print("Using basic descriptors only")
            return None
        
        print("Creating SOAP descriptors for neural networks...")
        
        # SOAP parameters optimized for neural networks
        soap = SOAP(
            species=['Au'],
            r_cut=5.0,      # Au-Au interaction range
            n_max=10,       # Higher resolution for NNs
            l_max=8,        # More angular terms for NNs
            sigma=0.5,      # Balanced smoothness
            periodic=False, # Clusters
            sparse=False,   # Dense for neural networks
            average='inner' # Average over atoms
        )
        
        soap_features = []
        filenames = []
        
        for structure in structures_data:
            try:
                atoms = structure['atoms'] if 'atoms' in structure else None
                if atoms is None:
                    coords = structure['coords']
                    atoms = Atoms('Au' * len(coords), positions=coords)
                
                soap_desc = soap.create(atoms)
                soap_features.append(soap_desc)
                filenames.append(structure['filename'])
                
            except Exception as e:
                print(f"Error creating SOAP for {structure.get('filename', 'unknown')}: {e}")
                continue
        
        if soap_features:
            soap_array = np.array(soap_features)
            soap_df = pd.DataFrame(
                soap_array, 
                columns=[f'soap_{i}' for i in range(soap_array.shape[1])]
            )
            soap_df['filename'] = filenames
            
            # Merge with existing data
            self.df = self.df.merge(soap_df, on='filename', how='inner')
            
            print(f"Added {soap_array.shape[1]} SOAP features for neural networks")
            self.soap_features = [col for col in self.df.columns if col.startswith('soap_')]
            
        return self.soap_features
    
    def prepare_features(self, target_column='energy', include_soap=True):
        """Prepare features optimized for neural networks"""
        feature_cols = []
        
        # Basic structural features
        basic_features = [
            'mean_bond_length', 'std_bond_length', 'n_bonds',
            'mean_coordination', 'std_coordination', 'max_coordination',
            'radius_of_gyration', 'asphericity', 'surface_fraction',
            'x_range', 'y_range', 'z_range', 'anisotropy',
            'compactness', 'bond_variance'
        ]
        
        available_basic = [f for f in basic_features if f in self.df.columns]
        feature_cols.extend(available_basic)
        
        # Add SOAP features (crucial for NN performance)
        if include_soap and self.soap_features:
            feature_cols.extend(self.soap_features)
            print(f"Using {len(self.soap_features)} SOAP features for neural networks")
        
        # Clean data
        feature_cols = [f for f in feature_cols if f in self.df.columns]
        data_clean = self.df[feature_cols + [target_column]].dropna()
        
        X = data_clean[feature_cols]
        y = data_clean[target_column]
        
        print(f"Neural network input shape: {X.shape}")
        print(f"High-dimensional features benefit neural network learning")
        
        return X, y, feature_cols
    
    def train_single_model(self, model, train_loader, val_loader, epochs, lr, weight_decay):
        """Train a single PyTorch model"""
        
        # Move model to device
        model = model.to(self.device)
        
        # Optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
        criterion = nn.MSELoss()
        
        # Early stopping
        early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                if isinstance(model, VAEPredictor):
                    # VAE training
                    reconstruction, energy_pred, mu, log_var = model(batch_x)
                    
                    # Reconstruction loss
                    recon_loss = F.mse_loss(reconstruction, batch_x, reduction='sum')
                    
                    # KL divergence loss
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                    
                    # Prediction loss
                    pred_loss = criterion(energy_pred.squeeze(), batch_y)
                    
                    # Total loss
                    loss = recon_loss + model.beta * kl_loss + pred_loss
                else:
                    # Standard training
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    if isinstance(model, VAEPredictor):
                        reconstruction, energy_pred, mu, log_var = model(batch_x)
                        recon_loss = F.mse_loss(reconstruction, batch_x, reduction='sum')
                        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                        pred_loss = criterion(energy_pred.squeeze(), batch_y)
                        loss = recon_loss + model.beta * kl_loss + pred_loss
                    else:
                        outputs = model(batch_x)
                        loss = criterion(outputs.squeeze(), batch_y)
                    
                    val_loss += loss.item()
            
            # Average losses
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Early stopping check
            if early_stopping(val_loss, model):
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return model, history
    
    def mc_predict(self, model, data_loader, n_samples=100):
        """Monte Carlo predictions with uncertainty quantification"""
        model.eval()
        
        all_predictions = []
        
        for _ in range(n_samples):
            predictions = []
            
            with torch.no_grad():
                for batch_x, _ in data_loader:
                    batch_x = batch_x.to(self.device)
                    # Force training mode for dropout during inference
                    outputs = model(batch_x, training=True)
                    predictions.append(outputs.cpu().numpy())
            
            predictions = np.concatenate(predictions).flatten()
            all_predictions.append(predictions)
        
        predictions_array = np.array(all_predictions)
        
        # Calculate mean and uncertainty
        mean_pred = np.mean(predictions_array, axis=0)
        uncertainty = np.std(predictions_array, axis=0)
        
        return mean_pred, uncertainty
    
    def predict_model(self, model, data_loader, mc_samples=None):
        """Standard model prediction"""
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                
                if isinstance(model, VAEPredictor):
                    _, energy_pred, _, _ = model(batch_x)
                    outputs = energy_pred
                else:
                    outputs = model(batch_x)
                
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions).flatten()
    
    def train_models(self, X, y, test_size=0.2, val_size=0.2):
        """Train all neural network models"""
        if not PYTORCH_AVAILABLE:
            print("❌ PyTorch not available")
            return {}
        
        print("\n" + "="*60)
        print("TRAINING NEURAL NETWORK MODELS WITH PYTORCH")
        print("="*60)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=self.random_state
        )
        
        # Scale features (important for neural networks)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_train_tensor = torch.FloatTensor(y_train.values)
        y_val_tensor = torch.FloatTensor(y_val.values)
        y_test_tensor = torch.FloatTensor(y_test.values)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Store data
        self.X_train, self.X_val, self.X_test = X_train_scaled, X_val_scaled, X_test_scaled
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        self.scaler = scaler
        
        results = {}
        input_dim = X_train_scaled.shape[1]
        
        for name, config in self.model_configs.items():
            print(f"\n🧠 Training {name.upper()}...")
            print(f"Justification: {config['justification'].strip()}")
            
            params = config['params']
            
            # Create data loaders for this model
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
            
            # Create model based on architecture
            if config['architecture'] == 'basic_mlp' or config['architecture'] == 'deep_mlp':
                model = MLPModel(
                    input_dim=input_dim,
                    hidden_layers=params['hidden_layers'],
                    dropout_rate=params['dropout_rate'],
                    activation=params['activation']
                )
            elif config['architecture'] == 'mc_dropout':
                model = BayesianMLP(
                    input_dim=input_dim,
                    hidden_layers=params['hidden_layers'],
                    dropout_rate=params['dropout_rate'],
                    activation=params['activation']
                )
            elif config['architecture'] == 'vae_predictor':
                model = VAEPredictor(
                    input_dim=input_dim,
                    encoder_layers=params['encoder_layers'],
                    latent_dim=params['latent_dim'],
                    decoder_layers=params['decoder_layers'],
                    predictor_layers=params['predictor_layers'],
                    dropout_rate=params['dropout_rate'],
                    beta=params['beta']
                )
            else:
                continue
            
            print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Train model
            trained_model, history = self.train_single_model(
                model, train_loader, val_loader, 
                params['epochs'], params['learning_rate'], params['weight_decay']
            )
            
            # Predictions
            if config['architecture'] == 'mc_dropout':
                # Monte Carlo predictions for uncertainty
                y_train_pred, train_uncertainty = self.mc_predict(trained_model, 
                                                                DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False),
                                                                params['mc_samples'])
                y_val_pred, val_uncertainty = self.mc_predict(trained_model, val_loader, params['mc_samples'])
                y_test_pred, test_uncertainty = self.mc_predict(trained_model, test_loader, params['mc_samples'])
            else:
                # Standard predictions
                y_train_pred = self.predict_model(trained_model, 
                                                DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False))
                y_val_pred = self.predict_model(trained_model, val_loader)
                y_test_pred = self.predict_model(trained_model, test_loader)
            
            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            results[name] = {
                'model': trained_model,
                'history': history,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'test_rmse': test_rmse,
                'y_train_pred': y_train_pred,
                'y_val_pred': y_val_pred,
                'y_test_pred': y_test_pred
            }
            
            # Add uncertainty for Bayesian models
            if config['architecture'] == 'mc_dropout':
                results[name]['train_uncertainty'] = train_uncertainty
                results[name]['val_uncertainty'] = val_uncertainty
                results[name]['test_uncertainty'] = test_uncertainty
            
            print(f"✅ {name}: Test R² = {test_r2:.3f}, Test RMSE = {test_rmse:.2f}")
        
        self.results = results
        return results
    
    def create_visualizations(self, output_dir='./neural_networks_results'):
        """Create comprehensive visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Training history
        self._plot_training_history(output_dir)
        
        # 2. Model performance comparison
        self._plot_model_comparison(output_dir)
        
        # 3. Prediction analysis
        self._plot_predictions(output_dir)
        
        # 4. Uncertainty analysis (for Bayesian models)
        self._plot_uncertainty_analysis(output_dir)
        
        print(f"📊 Neural network visualizations saved to {output_dir}")
    
    def _plot_training_history(self, output_dir):
        """Plot training history for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(self.results.items()):
            if i >= 4:
                break
                
            history = result['history']
            ax = axes[i]
            
            # Loss curves
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], label='Training Loss', linewidth=2)
            ax.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
            
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel('Loss', fontweight='bold')
            ax.set_title(f'{name.replace("_", " ").title()} Training History', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add final loss values
            final_train = history['train_loss'][-1]
            final_val = history['val_loss'][-1]
            ax.text(0.02, 0.98, f'Final Train: {final_train:.4f}\nFinal Val: {final_val:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(len(self.results), 4):
            axes[i].set_visible(False)
        
        plt.suptitle('Training History - PyTorch Neural Networks', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, output_dir):
        """Plot model performance comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        models = list(self.results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        # R² comparison
        train_r2 = [self.results[m]['train_r2'] for m in models]
        val_r2 = [self.results[m]['val_r2'] for m in models]
        test_r2 = [self.results[m]['test_r2'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        axes[0,0].bar(x - width, train_r2, width, label='Train', alpha=0.8, color=colors)
        axes[0,0].bar(x, val_r2, width, label='Validation', alpha=0.8, color=colors)
        axes[0,0].bar(x + width, test_r2, width, label='Test', alpha=0.8, color=colors)
        axes[0,0].set_ylabel('R² Score', fontweight='bold')
        axes[0,0].set_title('Neural Network R² Performance', fontweight='bold')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, (tr, vr, ter) in enumerate(zip(train_r2, val_r2, test_r2)):
            axes[0,0].text(i-width, tr+0.01, f'{tr:.3f}', ha='center', va='bottom', fontsize=8)
            axes[0,0].text(i, vr+0.01, f'{vr:.3f}', ha='center', va='bottom', fontsize=8)
            axes[0,0].text(i+width, ter+0.01, f'{ter:.3f}', ha='center', va='bottom', fontsize=8)
        
        # RMSE comparison
        test_rmse = [self.results[m]['test_rmse'] for m in models]
        bars = axes[0,1].bar(x, test_rmse, alpha=0.8, color=colors, edgecolor='black')
        axes[0,1].set_ylabel('Test RMSE', fontweight='bold')
        axes[0,1].set_title('Model RMSE Performance', fontweight='bold')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
        axes[0,1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rmse in zip(bars, test_rmse):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{rmse:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Overfitting analysis
        axes[1,0].scatter(train_r2, test_r2, s=100, c=colors, alpha=0.8, edgecolors='black')
        for i, model in enumerate(models):
            axes[1,0].annotate(model.replace('_', '\n'), (train_r2[i], test_r2[i]), 
                             xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Perfect correlation line
        min_r2 = min(min(train_r2), min(test_r2))
        max_r2 = max(max(train_r2), max(test_r2))
        axes[1,0].plot([min_r2, max_r2], [min_r2, max_r2], 'r--', alpha=0.8, linewidth=2)
        
        axes[1,0].set_xlabel('Train R²', fontweight='bold')
        axes[1,0].set_ylabel('Test R²', fontweight='bold')
        axes[1,0].set_title('Overfitting Analysis', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # Model complexity (parameters)
        param_counts = []
        for name in models:
            model = self.results[name]['model']
            param_count = sum(p.numel() for p in model.parameters())
            param_counts.append(param_count)
        
        axes[1,1].scatter(param_counts, test_r2, s=100, c=colors, alpha=0.8, edgecolors='black')
        for i, model in enumerate(models):
            axes[1,1].annotate(model.replace('_', '\n'), (param_counts[i], test_r2[i]), 
                             xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        axes[1,1].set_xlabel('Model Parameters', fontweight='bold')
        axes[1,1].set_ylabel('Test R²', fontweight='bold')
        axes[1,1].set_title('Complexity vs Performance', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_xscale('log')
        
        plt.suptitle('PyTorch Neural Network Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'nn_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_predictions(self, output_dir):
        """Plot prediction analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.results)))
        
        for i, (name, result) in enumerate(self.results.items()):
            if i >= 4:
                break
            
            y_true = self.y_test
            y_pred = result['y_test_pred']
            
            axes[i].scatter(y_true, y_pred, alpha=0.6, s=50, color=colors[i], edgecolors='black', linewidth=0.5)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            # Metrics annotation
            r2 = result['test_r2']
            rmse = result['test_rmse']
            axes[i].text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}', 
                        transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontweight='bold')
            
            axes[i].set_xlabel('Actual Energy (eV)', fontweight='bold')
            axes[i].set_ylabel('Predicted Energy (eV)', fontweight='bold')
            axes[i].set_title(f'{name.replace("_", " ").title()}', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(self.results), 4):
            axes[i].set_visible(False)
        
        plt.suptitle('Prediction Analysis - PyTorch Models', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'nn_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_uncertainty_analysis(self, output_dir):
        """Plot uncertainty analysis for Bayesian models"""
        bayesian_models = {k: v for k, v in self.results.items() if 'test_uncertainty' in v}
        
        if not bayesian_models:
            print("No Bayesian models found for uncertainty analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for i, (name, result) in enumerate(bayesian_models.items()):
            if i >= 1:  # Only one Bayesian model expected
                break
            
            y_true = self.y_test
            y_pred = result['y_test_pred']
            uncertainty = result['test_uncertainty']
            
            # Prediction vs uncertainty
            scatter = axes[0,0].scatter(y_pred, uncertainty, alpha=0.6, s=50, c=np.abs(y_true - y_pred), 
                                      cmap='viridis', edgecolors='black', linewidth=0.5)
            axes[0,0].set_xlabel('Predicted Energy (eV)', fontweight='bold')
            axes[0,0].set_ylabel('Prediction Uncertainty', fontweight='bold')
            axes[0,0].set_title(f'{name.title()} - Prediction vs Uncertainty', fontweight='bold')
            axes[0,0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0,0], label='Absolute Error')
            
            # Error vs uncertainty
            errors = np.abs(y_true - y_pred)
            axes[0,1].scatter(uncertainty, errors, alpha=0.6, s=50, color='orange', edgecolors='black', linewidth=0.5)
            
            # Add correlation line
            correlation = np.corrcoef(uncertainty, errors)[0, 1]
            z = np.polyfit(uncertainty, errors, 1)
            p = np.poly1d(z)
            axes[0,1].plot(uncertainty, p(uncertainty), "r--", alpha=0.8, linewidth=2)
            
            axes[0,1].set_xlabel('Prediction Uncertainty', fontweight='bold')
            axes[0,1].set_ylabel('Absolute Error', fontweight='bold')
            axes[0,1].set_title(f'Uncertainty vs Error (r={correlation:.3f})', fontweight='bold')
            axes[0,1].grid(True, alpha=0.3)
            
            # Uncertainty distribution
            axes[1,0].hist(uncertainty, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
            axes[1,0].axvline(np.mean(uncertainty), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(uncertainty):.3f}')
            axes[1,0].axvline(np.median(uncertainty), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(uncertainty):.3f}')
            axes[1,0].set_xlabel('Prediction Uncertainty', fontweight='bold')
            axes[1,0].set_ylabel('Frequency', fontweight='bold')
            axes[1,0].set_title('Uncertainty Distribution', fontweight='bold')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
            # Calibration plot
            # Sort by uncertainty and bin
            sorted_indices = np.argsort(uncertainty)
            n_bins = 10
            bin_size = len(sorted_indices) // n_bins
            
            bin_uncertainties = []
            bin_errors = []
            
            for j in range(n_bins):
                start_idx = j * bin_size
                end_idx = (j + 1) * bin_size if j < n_bins - 1 else len(sorted_indices)
                bin_indices = sorted_indices[start_idx:end_idx]
                
                bin_uncertainty = np.mean(uncertainty[bin_indices])
                bin_error = np.mean(errors[bin_indices])
                
                bin_uncertainties.append(bin_uncertainty)
                bin_errors.append(bin_error)
            
            axes[1,1].scatter(bin_uncertainties, bin_errors, s=100, alpha=0.8, color='green', edgecolors='black')
            axes[1,1].plot([0, max(bin_uncertainties)], [0, max(bin_uncertainties)], 'r--', alpha=0.8, linewidth=2, label='Perfect Calibration')
            
            # Add correlation for calibration
            cal_corr = np.corrcoef(bin_uncertainties, bin_errors)[0, 1]
            axes[1,1].text(0.05, 0.95, f'Calibration r={cal_corr:.3f}', transform=axes[1,1].transAxes,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')
            
            axes[1,1].set_xlabel('Mean Uncertainty', fontweight='bold')
            axes[1,1].set_ylabel('Mean Absolute Error', fontweight='bold')
            axes[1,1].set_title('Uncertainty Calibration', fontweight='bold')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle('Bayesian Neural Network Uncertainty Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'uncertainty_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_models(self, output_dir='./neural_networks_results'):
        """Save trained models and results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, output_dir / 'feature_scaler.joblib')
        
        # Save models
        for name, result in self.results.items():
            model_path = output_dir / f'{name}_model.pth'
            torch.save({
                'model_state_dict': result['model'].state_dict(),
                'model_class': result['model'].__class__.__name__,
                'model_config': self.model_configs[name]['params']
            }, model_path)
        
        # Save results summary
        summary_data = []
        for name, result in self.results.items():
            param_count = sum(p.numel() for p in result['model'].parameters())
            summary_data.append({
                'model': name,
                'train_r2': result['train_r2'],
                'val_r2': result['val_r2'],
                'test_r2': result['test_r2'],
                'train_rmse': result['train_rmse'],
                'val_rmse': result['val_rmse'],
                'test_rmse': result['test_rmse'],
                'parameters': param_count,
                'has_uncertainty': 'test_uncertainty' in result
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'nn_model_summary.csv', index=False)
        
        # Save predictions
        predictions_dir = output_dir / 'predictions'
        predictions_dir.mkdir(parents=True, exist_ok=True)
        
        for name, result in self.results.items():
            pred_data = {
                'y_train_actual': self.y_train.values,
                'y_train_pred': result['y_train_pred'],
                'y_val_actual': self.y_val.values,
                'y_val_pred': result['y_val_pred'],
                'y_test_actual': self.y_test.values,
                'y_test_pred': result['y_test_pred']
            }
            
            # Add uncertainties if available
            if 'test_uncertainty' in result:
                pred_data['y_train_uncertainty'] = result['train_uncertainty']
                pred_data['y_val_uncertainty'] = result['val_uncertainty']
                pred_data['y_test_uncertainty'] = result['test_uncertainty']
            
            pred_df = pd.DataFrame(pred_data)
            pred_df.to_csv(predictions_dir / f'{name}_predictions.csv', index=False)
        
        print(f"💾 PyTorch models and results saved to {output_dir}")
        
        return summary_df
    
    def analyze_model_insights(self):
        """Analyze neural network specific insights"""
        print("\n" + "="*60)
        print("PYTORCH NEURAL NETWORK MODEL INSIGHTS")
        print("="*60)
        
        for name, result in self.results.items():
            model = result['model']
            param_count = sum(p.numel() for p in model.parameters())
            
            print(f"\n{name.upper()}:")
            print(f"  Parameters: {param_count:,}")
            print(f"  Test R²: {result['test_r2']:.3f}")
            print(f"  Test RMSE: {result['test_rmse']:.3f}")
            print(f"  Device: {next(model.parameters()).device}")
            
            # Training efficiency
            history = result['history']
            final_epoch = len(history['train_loss'])
            min_val_loss_epoch = np.argmin(history['val_loss']) + 1
            print(f"  Training epochs: {final_epoch}")
            print(f"  Best validation at epoch: {min_val_loss_epoch}")
            
            # Overfitting analysis
            overfitting_gap = result['train_r2'] - result['test_r2']
            if overfitting_gap > 0.1:
                print(f"  ⚠️ Overfitting detected: {overfitting_gap:.3f}")
            else:
                print(f"  ✅ Good generalization: {overfitting_gap:.3f}")
            
            # Uncertainty analysis for Bayesian models
            if 'test_uncertainty' in result:
                uncertainty = result['test_uncertainty']
                print(f"  Mean uncertainty: {np.mean(uncertainty):.3f}")
                print(f"  Uncertainty range: {np.min(uncertainty):.3f} - {np.max(uncertainty):.3f}")
                
                # High uncertainty predictions
                high_uncertainty_threshold = np.percentile(uncertainty, 95)
                high_uncertainty_count = np.sum(uncertainty > high_uncertainty_threshold)
                print(f"  High uncertainty predictions: {high_uncertainty_count} ({high_uncertainty_count/len(uncertainty)*100:.1f}%)")
                
                # Uncertainty-error correlation
                errors = np.abs(self.y_test.values - result['y_test_pred'])
                correlation = np.corrcoef(uncertainty, errors)[0, 1]
                print(f"  Uncertainty-error correlation: {correlation:.3f}")

def main():
    """Main execution function"""
    print("🧠 PyTorch Neural Networks for Au Cluster Analysis")
    print("="*65)
    
    # Check PyTorch CUDA availability
    if PYTORCH_AVAILABLE:
        print(f"✅ PyTorch ready with device: {device}")
        if torch.cuda.is_available():
            print(f"🔥 CUDA acceleration enabled")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        # Uncomment below for Metal
        # elif torch.backends.mps.is_available():
        #     print("🔥 Metal Performance Shaders acceleration enabled")
        else:
            print("⚠️ Using CPU (CUDA GPU not detected)")
            # print("⚠️ Using CPU (CUDA/Metal GPU not detected)")  # When Metal enabled
    else:
        print("❌ PyTorch not available. Please install PyTorch 2.1.2 with CUDA")
        return None, None
    
    # Initialize analyzer
    analyzer = NeuralNetworkAnalyzer(random_state=42)
    
    # Load data
    try:
        data_path = input("/Users/wilbert/Documents/GitHub/AIAC/au_cluster_analysis_results/descriptors.csv").strip()
        if not data_path:
            data_path = "./au_cluster_analysis_results/descriptors.csv"
        
        analyzer.load_data(data_path)
        
        # Prepare features
        X, y, feature_names = analyzer.prepare_features(target_column='energy')
        
        print(f"\n📊 Dataset prepared:")
        print(f"   Features: {len(feature_names)}")
        print(f"   Samples: {len(X)}")
        print(f"   Device: {device}")
        
        # Train models
        results = analyzer.train_models(X, y)
        
        # Analyze insights
        analyzer.analyze_model_insights()
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Save results
        summary_df = analyzer.save_models()
        
        print("\n🎉 PyTorch neural network analysis complete!")
        
        if not summary_df.empty:
            print("\nModel Performance Summary:")
            print(summary_df[['model', 'test_r2', 'test_rmse', 'parameters']].round(3).to_string(index=False))
            
            best_model = summary_df.loc[summary_df['test_r2'].idxmax()]
            print(f"\n🏆 Best performing model:")
            print(f"   {best_model['model'].upper()}: R² = {best_model['test_r2']:.3f}")
            print(f"   Parameters: {best_model['parameters']:,}")
        
        print("\n💡 PyTorch Neural Network Insights:")
        print("- Deep learning captures complex non-linear energy relationships")
        print("- SOAP descriptors provide rich input for neural feature learning")
        print("- Bayesian models quantify prediction uncertainty")
        print("- CUDA GPU acceleration enables efficient training")
        print("- Early stopping and regularization prevent overfitting")
        print("- PyTorch provides flexible model architecture design")
        # print("- Metal GPU acceleration enables efficient training on M4")  # For Metal
        
        return analyzer, results
        
    except FileNotFoundError:
        print("❌ Data file not found. Please run task1.py first to generate descriptors.")
        return None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    analyzer, results = main()