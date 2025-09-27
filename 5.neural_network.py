#!/usr/bin/env python3
"""
Category 4: Neural Network Models for Au Cluster Energy Prediction
Models: MLP, Bayesian NN (MC Dropout), Deep Ensemble, VAE
Optimized for TensorFlow + Metal Performance Shaders on MacBook Pro M4
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

# TensorFlow with Metal optimization
try:
    import tensorflow as tf
    # Configure for Metal Performance Shaders
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"✅ TensorFlow Metal GPU available: {len(physical_devices)} devices")
    else:
        print("⚠️  No Metal GPU found, using CPU")
    
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, optimizers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("❌ TensorFlow not available")
    TENSORFLOW_AVAILABLE = False

# SOAP descriptors
try:
    from dscribe.descriptors import SOAP
    from ase.atoms import Atoms
    SOAP_AVAILABLE = True
except ImportError:
    print("Warning: DScribe not available. Using basic descriptors only.")
    SOAP_AVAILABLE = False

class NeuralNetworkAnalyzer:
    """
    Neural Network Models for Au Cluster Analysis
    
    Why Neural Networks for Au Clusters:
    1. Universal Approximation: Can model any continuous function with sufficient neurons
    2. Feature Learning: Automatically discovers optimal feature combinations from SOAP
    3. Non-linear Modeling: Captures complex energy landscapes and phase transitions
    4. Uncertainty Quantification: Bayesian methods provide prediction confidence
    5. Transfer Learning: Pre-trained features can generalize to new cluster types
    6. Scalability: Leverages M4 Neural Engine for accelerated computation
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.soap_features = None
        
        # Neural network architectures
        self.model_configs = self._initialize_models()
    
    def _initialize_models(self):
        """Initialize neural network architectures with justifications"""
        if not TENSORFLOW_AVAILABLE:
            return {}
        
        configs = {
            'mlp_basic': {
                'architecture': 'basic_mlp',
                'params': {
                    'hidden_layers': [256, 128, 64],
                    'dropout_rate': 0.2,
                    'activation': 'relu',
                    'l2_reg': 0.001,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 200
                },
                'justification': """
                Basic Multi-Layer Perceptron:
                - Universal function approximator for energy surfaces
                - Hidden layers learn hierarchical feature representations
                - Dropout prevents overfitting on limited cluster data
                - L2 regularization ensures smooth energy predictions
                - Optimized for SOAP descriptor input vectors
                - Leverages Metal GPU acceleration on M4
                """
            },
            
            'mlp_deep': {
                'architecture': 'deep_mlp',
                'params': {
                    'hidden_layers': [512, 256, 128, 64, 32],
                    'dropout_rate': 0.3,
                    'activation': 'relu',
                    'l2_reg': 0.01,
                    'learning_rate': 0.0005,
                    'batch_size': 32,
                    'epochs': 300
                },
                'justification': """
                Deep Multi-Layer Perceptron:
                - Deeper architecture captures more complex patterns
                - Progressive layer size reduction creates feature hierarchy
                - Higher dropout for deeper regularization
                - Slower learning rate for stable deep training
                - Excellent for high-dimensional SOAP features
                - Benefits from Metal's parallel matrix operations
                """
            },
            
            'bayesian_nn': {
                'architecture': 'mc_dropout',
                'params': {
                    'hidden_layers': [256, 128, 64],
                    'dropout_rate': 0.1,
                    'activation': 'relu',
                    'l2_reg': 0.001,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 250,
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
        """
        Create SOAP descriptors optimized for neural networks
        
        Why SOAP + Neural Networks:
        - High-dimensional SOAP vectors ideal for deep learning
        - Smooth, differentiable features for gradient descent
        - Translation/rotation invariance preserves physical meaning
        - Neural networks excel at learning from raw SOAP vectors
        """
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
    
    def create_mlp_model(self, input_dim, params):
        """Create Multi-Layer Perceptron model"""
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for i, units in enumerate(params['hidden_layers']):
            model.add(layers.Dense(
                units, 
                activation=params['activation'],
                kernel_regularizer=keras.regularizers.l2(params['l2_reg']),
                name=f'hidden_{i+1}'
            ))
            model.add(layers.Dropout(params['dropout_rate']))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear', name='output'))
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=params['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_bayesian_model(self, input_dim, params):
        """Create Bayesian Neural Network with MC Dropout"""
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers with dropout for MC sampling
        for i, units in enumerate(params['hidden_layers']):
            model.add(layers.Dense(
                units, 
                activation=params['activation'],
                kernel_regularizer=keras.regularizers.l2(params['l2_reg']),
                name=f'bayesian_hidden_{i+1}'
            ))
            # Dropout remains active during inference for MC sampling
            model.add(layers.Dropout(params['dropout_rate'], name=f'mc_dropout_{i+1}'))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear', name='bayesian_output'))
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=params['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_vae_model(self, input_dim, params):
        """Create Variational Autoencoder + Predictor"""
        # Encoder
        encoder_input = layers.Input(shape=(input_dim,))
        x = encoder_input
        
        for units in params['encoder_layers']:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(params['dropout_rate'])(x)
        
        # Latent space
        z_mean = layers.Dense(params['latent_dim'], name='z_mean')(x)
        z_log_var = layers.Dense(params['latent_dim'], name='z_log_var')(x)
        
        # Sampling layer
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = layers.Lambda(sampling, output_shape=(params['latent_dim'],))([z_mean, z_log_var])
        
        # Decoder
        decoder_input = layers.Input(shape=(params['latent_dim'],))
        x = decoder_input
        
        for units in params['decoder_layers']:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(params['dropout_rate'])(x)
        
        decoder_output = layers.Dense(input_dim, activation='linear')(x)
        
        # Predictor from latent space
        predictor_input = layers.Input(shape=(params['latent_dim'],))
        x = predictor_input
        
        for units in params['predictor_layers']:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(params['dropout_rate'])(x)
        
        energy_output = layers.Dense(1, activation='linear', name='energy_prediction')(x)
        
        # Create models
        encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
        decoder = keras.Model(decoder_input, decoder_output, name='decoder')
        predictor = keras.Model(predictor_input, energy_output, name='predictor')
        
        # VAE model
        class VAE(keras.Model):
            def __init__(self, encoder, decoder, predictor, beta=1.0, **kwargs):
                super(VAE, self).__init__(**kwargs)
                self.encoder = encoder
                self.decoder = decoder
                self.predictor = predictor
                self.beta = beta
                self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
                self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
                self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
                self.prediction_loss_tracker = keras.metrics.Mean(name="prediction_loss")
            
            @property
            def metrics(self):
                return [
                    self.total_loss_tracker,
                    self.reconstruction_loss_tracker,
                    self.kl_loss_tracker,
                    self.prediction_loss_tracker,
                ]
            
            def call(self, inputs):
                z_mean, z_log_var, z = self.encoder(inputs)
                reconstruction = self.decoder(z)
                energy_pred = self.predictor(z)
                return reconstruction, energy_pred
            
            def train_step(self, data):
                x, y = data
                
                with tf.GradientTape() as tape:
                    z_mean, z_log_var, z = self.encoder(x)
                    reconstruction = self.decoder(z)
                    energy_pred = self.predictor(z)
                    
                    # Reconstruction loss
                    reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum(keras.losses.mse(x, reconstruction), axis=1)
                    )
                    
                    # KL divergence loss
                    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                    
                    # Prediction loss
                    prediction_loss = tf.reduce_mean(keras.losses.mse(y, energy_pred))
                    
                    # Total loss
                    total_loss = reconstruction_loss + self.beta * kl_loss + prediction_loss
                
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                
                self.total_loss_tracker.update_state(total_loss)
                self.reconstruction_loss_tracker.update_state(reconstruction_loss)
                self.kl_loss_tracker.update_state(kl_loss)
                self.prediction_loss_tracker.update_state(prediction_loss)
                
                return {
                    "loss": self.total_loss_tracker.result(),
                    "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                    "kl_loss": self.kl_loss_tracker.result(),
                    "prediction_loss": self.prediction_loss_tracker.result(),
                }
        
        vae = VAE(encoder, decoder, predictor, beta=params['beta'])
        vae.compile(optimizer=optimizers.Adam(learning_rate=params['learning_rate']))
        
        return vae
    
    def train_models(self, X, y, test_size=0.2, val_size=0.2):
        """Train all neural network models"""
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available")
            return {}
        
        print("\n" + "="*60)
        print("TRAINING NEURAL NETWORK MODELS")
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
            
            # Create model based on architecture
            if config['architecture'] == 'basic_mlp' or config['architecture'] == 'deep_mlp':
                model = self.create_mlp_model(input_dim, params)
            elif config['architecture'] == 'mc_dropout':
                model = self.create_bayesian_model(input_dim, params)
            elif config['architecture'] == 'vae_predictor':
                model = self.create_vae_model(input_dim, params)
            else:
                continue
            
            print(f"  Model architecture: {model.summary()}")
            
            # Callbacks
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
            
            # Train model
            if config['architecture'] == 'vae_predictor':
                # VAE training
                history = model.fit(
                    X_train_scaled, [X_train_scaled, y_train],
                    validation_data=(X_val_scaled, [X_val_scaled, y_val]),
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1
                )
                
                # Predictions for VAE
                _, y_train_pred = model(X_train_scaled)
                _, y_val_pred = model(X_val_scaled) 
                _, y_test_pred = model(X_test_scaled)
                
                y_train_pred = y_train_pred.numpy().flatten()
                y_val_pred = y_val_pred.numpy().flatten()
                y_test_pred = y_test_pred.numpy().flatten()
                
            else:
                # Standard neural network training
                history = model.fit(
                    X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1
                )
                
                # Predictions
                if config['architecture'] == 'mc_dropout':
                    # Monte Carlo predictions for uncertainty
                    y_train_pred, train_uncertainty = self.mc_predict(model, X_train_scaled, params['mc_samples'])
                    y_val_pred, val_uncertainty = self.mc_predict(model, X_val_scaled, params['mc_samples'])
                    y_test_pred, test_uncertainty = self.mc_predict(model, X_test_scaled, params['mc_samples'])
                else:
                    # Standard predictions
                    y_train_pred = model.predict(X_train_scaled, verbose=0).flatten()
                    y_val_pred = model.predict(X_val_scaled, verbose=0).flatten()
                    y_test_pred = model.predict(X_test_scaled, verbose=0).flatten()
            
            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            results[name] = {
                'model': model,
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
    
    def mc_predict(self, model, X, n_samples=100):
        """Monte Carlo predictions with uncertainty quantification"""
        predictions = []
        
        for _ in range(n_samples):
            # Enable dropout during prediction
            pred = model(X, training=True)
            predictions.append(pred.numpy().flatten())
        
        predictions = np.array(predictions)
        
        # Calculate mean and uncertainty
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty
    
    def create_visualizations(self, output_dir='./neural_networks_results'):
        """Create comprehensive visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
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
        
        for i, (name, result) in enumerate(self.results.items()):
            if i >= 4:
                break
                
            history = result['history']
            
            # Loss curves
            ax_row = i // 2
            ax_col = i % 2
            
            axes[ax_row, ax_col].plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                axes[ax_row, ax_col].plot(history.history['val_loss'], label='Validation Loss')
            
            axes[ax_row, ax_col].set_xlabel('Epoch')
            axes[ax_row, ax_col].set_ylabel('Loss')
            axes[ax_row, ax_col].set_title(f'{name.replace("_", " ").title()} Training History')
            axes[ax_row, ax_col].legend()
            axes[ax_row, ax_col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
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
        axes[0,0].set_title('Neural Network R² Performance')
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
        
        # Model complexity (parameters)
        param_counts = []
        for name in models:
            model = self.results[name]['model']
            param_count = model.count_params()
            param_counts.append(param_count)
        
        axes[1,1].scatter(param_counts, test_r2, s=100, c=colors, alpha=0.8)
        for i, model in enumerate(models):
            axes[1,1].annotate(model, (param_counts[i], test_r2[i]), 
                             xytext=(5, 5), textcoords='offset points')
        
        axes[1,1].set_xlabel('Model Parameters')
        axes[1,1].set_ylabel('Test R²')
        axes[1,1].set_title('Complexity vs Performance')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'nn_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_predictions(self, output_dir):
        """Plot prediction analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        colors = ['blue', 'green', 'orange', 'red']
        
        for i, (name, result) in enumerate(self.results.items()):
            if i >= 4:
                break
            
            y_true = self.y_test
            y_pred = result['y_test_pred']
            
            axes[i].scatter(y_true, y_pred, alpha=0.6, s=50, color=colors[i])
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
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
        plt.savefig(output_dir / 'nn_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_uncertainty_analysis(self, output_dir):
        """Plot uncertainty analysis for Bayesian models"""
        bayesian_models = {k: v for k, v in self.results.items() if 'uncertainty' in v}
        
        if not bayesian_models:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for i, (name, result) in enumerate(bayesian_models.items()):
            if i >= 4:
                break
            
            y_true = self.y_test
            y_pred = result['y_test_pred']
            uncertainty = result['test_uncertainty']
            
            # Prediction vs uncertainty
            axes[0,0].scatter(y_pred, uncertainty, alpha=0.6, s=50)
            axes[0,0].set_xlabel('Predicted Energy')
            axes[0,0].set_ylabel('Prediction Uncertainty')
            axes[0,0].set_title(f'{name.title()} - Prediction vs Uncertainty')
            axes[0,0].grid(True, alpha=0.3)
            
            # Error vs uncertainty
            errors = np.abs(y_true - y_pred)
            axes[0,1].scatter(uncertainty, errors, alpha=0.6, s=50)
            axes[0,1].set_xlabel('Prediction Uncertainty')
            axes[0,1].set_ylabel('Absolute Error')
            axes[0,1].set_title('Uncertainty vs Actual Error')
            axes[0,1].grid(True, alpha=0.3)
            
            # Uncertainty distribution
            axes[1,0].hist(uncertainty, bins=20, alpha=0.7, edgecolor='black')
            axes[1,0].set_xlabel('Prediction Uncertainty')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Uncertainty Distribution')
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
            
            axes[1,1].scatter(bin_uncertainties, bin_errors, s=100, alpha=0.8)
            axes[1,1].plot([0, max(bin_uncertainties)], [0, max(bin_uncertainties)], 'r--', alpha=0.8)
            axes[1,1].set_xlabel('Mean Uncertainty')
            axes[1,1].set_ylabel('Mean Absolute Error')
            axes[1,1].set_title('Uncertainty Calibration')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'uncertainty_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_models(self, output_dir='./neural_networks_results'):
        """Save trained models and results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, output_dir / 'feature_scaler.joblib')
        
        # Save models
        for name, result in self.results.items():
            model_path = output_dir / f'{name}_model'
            result['model'].save(model_path)
        
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
                'parameters': result['model'].count_params()
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'nn_model_summary.csv', index=False)
        
        print(f"💾 Neural network models and results saved to {output_dir}")
        
        return summary_df
    
    def analyze_model_insights(self):
        """Analyze neural network specific insights"""
        print("\n" + "="*60)
        print("NEURAL NETWORK MODEL INSIGHTS")
        print("="*60)
        
        for name, result in self.results.items():
            model = result['model']
            print(f"\n{name.upper()}:")
            print(f"  Parameters: {model.count_params():,}")
            print(f"  Test R²: {result['test_r2']:.3f}")
            print(f"  Test RMSE: {result['test_rmse']:.2f}")
            
            # Architecture analysis
            if hasattr(model, 'layers'):
                hidden_layers = [layer for layer in model.layers if 'hidden' in layer.name or 'dense' in layer.name.lower()]
                print(f"  Hidden layers: {len(hidden_layers)}")
                
                if hidden_layers:
                    total_params = sum([layer.count_params() for layer in hidden_layers])
                    print(f"  Hidden layer parameters: {total_params:,}")
            
            # Uncertainty analysis for Bayesian models
            if 'uncertainty' in result:
                uncertainty = result['test_uncertainty']
                print(f"  Mean uncertainty: {np.mean(uncertainty):.3f}")
                print(f"  Uncertainty range: {np.min(uncertainty):.3f} - {np.max(uncertainty):.3f}")
                
                # High uncertainty predictions (potential outliers)
                high_uncertainty_threshold = np.percentile(uncertainty, 95)
                high_uncertainty_count = np.sum(uncertainty > high_uncertainty_threshold)
                print(f"  High uncertainty predictions: {high_uncertainty_count} ({high_uncertainty_count/len(uncertainty)*100:.1f}%)")

def main():
    """Main execution function"""
    print("🧠 Neural Network Models for Au Cluster Analysis")
    print("="*65)
    
    # Check TensorFlow Metal availability
    if TENSORFLOW_AVAILABLE:
        print("✅ TensorFlow with Metal optimization ready")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🔥 Metal GPU acceleration: {len(gpus)} device(s)")
        else:
            print("⚠️  Using CPU (Metal GPU not detected)")
    else:
        print("❌ TensorFlow not available. Please install tensorflow-macos and tensorflow-metal")
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
        
        # Train models
        results = analyzer.train_models(X, y)
        
        # Analyze insights
        analyzer.analyze_model_insights()
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Save results
        summary_df = analyzer.save_models()
        
        print("\n🎉 Neural network analysis complete!")
        print("\nBest performing model:")
        best_model = summary_df.loc[summary_df['test_r2'].idxmax()]
        print(f"  {best_model['model'].upper()}: R² = {best_model['test_r2']:.3f}")
        
        print("\n💡 Neural Network Insights:")
        print("- Deep learning captures complex non-linear energy relationships")
        print("- SOAP descriptors provide rich input for neural feature learning")
        print("- Bayesian models quantify prediction uncertainty")
        print("- Metal GPU acceleration enables efficient training on M4")
        print("- Regularization and early stopping prevent overfitting")
        
        return analyzer, results
        
    except FileNotFoundError:
        print("❌ Data file not found. Please run task1.py first to generate descriptors.")
        return None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

if __name__ == "__main__":
    analyzer, results = main()#!/usr/bin/env python3
"""
Category 4: Neural Network Models for Au Cluster Energy Prediction
Models: MLP, Bayesian NN (MC Dropout), Deep Ensemble, VAE
Optimized for TensorFlow + Metal Performance Shaders on MacBook Pro M4
"""
