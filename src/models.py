"""
Deep learning models for AQI prediction.
"""
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class NeuralNetworkRegressor:
    """Multi-Layer Perceptron (MLP) for tabular regression."""
    
    def __init__(self, input_dim: int, hidden_layers: Tuple[int, ...] = (64, 32, 16), 
                 dropout: float = 0.2, learning_rate: float = 0.001, random_state: int = 42):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for neural network models. Install with: pip install tensorflow")
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        
    def _build_model(self):
        """Build the MLP architecture."""
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Dense(self.hidden_layers[0], activation='relu', input_shape=(self.input_dim,)))
        model.add(layers.Dropout(self.dropout))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(self.dropout))
        
        # Output layer (single value for regression)
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features (required for neural networks)."""
        if fit:
            self.scaler_mean = X.mean(axis=0)
            self.scaler_std = X.std(axis=0)
            # Avoid division by zero
            self.scaler_std = np.where(self.scaler_std == 0, 1.0, self.scaler_std)
        
        return (X - self.scaler_mean) / self.scaler_std
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32, 
            validation_split: float = 0.1, verbose: int = 0):
        """Train the neural network."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # Normalize features
        X_norm = self._normalize(X, fit=True)
        
        # Build model
        self.model = self._build_model()
        
        # Early stopping to prevent overfitting
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        self.model.fit(
            X_norm, y,
            epochs=epochs,
            batch_size=min(batch_size, len(X)),
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        X_norm = self._normalize(X, fit=False)
        predictions = self.model.predict(X_norm, verbose=0)
        return predictions.flatten()


class LSTMModelRegressor:
    """LSTM model for time series regression (requires sequential data)."""
    
    def __init__(self, sequence_length: int = 24, features_per_step: int = 1,
                 lstm_units: Tuple[int, ...] = (50, 50), dropout: float = 0.2,
                 learning_rate: float = 0.001, random_state: int = 42):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models. Install with: pip install tensorflow")
        
        self.sequence_length = sequence_length
        self.features_per_step = features_per_step
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        
    def _create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences from time series data."""
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(data)):
            X_seq.append(data[i - self.sequence_length:i])
            y_seq.append(targets[i])
        return np.array(X_seq), np.array(y_seq)
    
    def _build_model(self):
        """Build the LSTM architecture."""
        model = models.Sequential()
        
        # First LSTM layer
        model.add(layers.LSTM(
            self.lstm_units[0],
            return_sequences=len(self.lstm_units) > 1,
            input_shape=(self.sequence_length, self.features_per_step)
        ))
        model.add(layers.Dropout(self.dropout))
        
        # Additional LSTM layers
        for units in self.lstm_units[1:-1]:
            model.add(layers.LSTM(units, return_sequences=True))
            model.add(layers.Dropout(self.dropout))
        
        if len(self.lstm_units) > 1:
            model.add(layers.LSTM(self.lstm_units[-1], return_sequences=False))
            model.add(layers.Dropout(self.dropout))
        
        # Dense layers
        model.add(layers.Dense(25, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features."""
        if fit:
            # Reshape for normalization (flatten time steps)
            X_flat = X.reshape(-1, X.shape[-1])
            self.scaler_mean = X_flat.mean(axis=0)
            self.scaler_std = X_flat.std(axis=0)
            self.scaler_std = np.where(self.scaler_std == 0, 1.0, self.scaler_std)
        
        # Normalize
        X_norm = (X - self.scaler_mean) / self.scaler_std
        return X_norm
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32,
            validation_split: float = 0.1, verbose: int = 0):
        """Train the LSTM model."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # For LSTM, we need to reshape data into sequences
        # Use the target variable itself as the sequence feature (simplified approach)
        # Reshape to (samples, sequence_length, features_per_step)
        if len(X) < self.sequence_length + 10:
            raise ValueError(f"Need at least {self.sequence_length + 10} samples for LSTM, got {len(X)}")
        
        # Use target values to create sequences (autoregressive approach)
        y_padded = np.pad(y, (self.sequence_length, 0), mode='edge')
        X_seq, y_seq = self._create_sequences(
            y_padded.reshape(-1, 1), 
            y
        )
        
        # Normalize
        X_seq_norm = self._normalize(X_seq, fit=True)
        
        # Build model
        self.model = self._build_model()
        
        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        self.model.fit(
            X_seq_norm, y_seq,
            epochs=epochs,
            batch_size=min(batch_size, len(X_seq)),
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions (requires recent history)."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # For prediction, we'll use a simplified approach
        # In production, you'd maintain a rolling window of predictions
        # Here we use the last sequence_length values from target
        # This is a limitation - in practice, you'd need to maintain state
        raise NotImplementedError(
            "LSTM prediction requires maintaining state/sequence history. "
            "For production use, implement a stateful prediction method."
        )

