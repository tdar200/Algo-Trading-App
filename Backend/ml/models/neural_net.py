"""
PyTorch Neural Network for Pattern Discovery

GPU-accelerated neural network with:
- CUDA support for RTX 3050
- Mixed precision training (FP16)
- BatchNorm + Dropout regularization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from typing import Tuple, Optional, List
import warnings


class PatternNet(nn.Module):
    """
    Neural network for stock pattern recognition.

    Architecture: Input -> 256 -> 128 -> 64 -> Output
    Features:
    - BatchNorm for stable training
    - Dropout for regularization
    - LeakyReLU activation
    - Residual connections for deeper variants
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        task: str = 'regression'
    ):
        """
        Initialize PatternNet.

        Args:
            input_dim: Number of input features
            output_dim: Number of output classes (1 for regression, 2+ for classification)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            task: 'regression' or 'classification'
        """
        super(PatternNet, self).__init__()

        self.task = task
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        output = self.network(x)

        if self.task == 'classification' and self.output_dim > 1:
            # Don't apply softmax here - use CrossEntropyLoss which includes it
            pass

        return output


class PatternNetTrainer:
    """
    Trainer for PatternNet with GPU acceleration and mixed precision.
    """

    def __init__(
        self,
        model: PatternNet,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_gpu: bool = True,
        use_mixed_precision: bool = True
    ):
        """
        Initialize trainer.

        Args:
            model: PatternNet model
            learning_rate: Learning rate
            weight_decay: L2 regularization
            use_gpu: Whether to use GPU
            use_mixed_precision: Whether to use FP16 mixed precision
        """
        self.model = model
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.use_mixed_precision = use_mixed_precision and self.use_gpu

        # Move model to GPU if available
        if self.use_gpu:
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Loss function
        if model.task == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_mixed_precision else None

        # Training history
        self.history = {'train_loss': [], 'val_loss': []}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 256,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> dict:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: Epochs without improvement before stopping
            verbose: Whether to print progress

        Returns:
            Training history dict
        """
        # Prepare data
        train_loader = self._prepare_dataloader(X_train, y_train, batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_loader = self._prepare_dataloader(X_val, y_val, batch_size, shuffle=False)
        else:
            val_loader = None

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            # Validation
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.history['val_loss'].append(val_loss)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.6f} - "
                          f"Val Loss: {val_loss:.6f}")

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history

    def _prepare_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool
    ) -> DataLoader:
        """Prepare PyTorch DataLoader."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y) if self.model.task == 'regression' else torch.LongTensor(y)

        if self.model.task == 'regression' and len(y_tensor.shape) == 1:
            y_tensor = y_tensor.unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=self.use_gpu,
            num_workers=0  # Keep at 0 for Windows compatibility
        )

        return loader

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(X_batch)
                        loss = self.criterion(outputs, y_batch)
                else:
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def predict(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features
            batch_size: Batch size for inference

        Returns:
            Predictions array
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X)

        predictions = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size].to(self.device)

                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)

                if self.model.task == 'classification':
                    # Get class probabilities
                    outputs = torch.softmax(outputs, dim=1)

                predictions.append(outputs.cpu().numpy())

        return np.vstack(predictions)

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'model_config': {
                'input_dim': self.model.input_dim,
                'output_dim': self.model.output_dim,
                'task': self.model.task
            }
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})


def get_gpu_info() -> dict:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {'available': False}

    return {
        'available': True,
        'device_name': torch.cuda.get_device_name(0),
        'device_count': torch.cuda.device_count(),
        'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
        'memory_allocated_gb': torch.cuda.memory_allocated(0) / 1e9,
        'cuda_version': torch.version.cuda
    }


if __name__ == '__main__':
    # Test the module
    print("GPU Info:", get_gpu_info())

    # Create a test model
    model = PatternNet(input_dim=100, output_dim=1, task='regression')
    trainer = PatternNetTrainer(model)

    # Generate random test data
    X_train = np.random.randn(1000, 100).astype(np.float32)
    y_train = np.random.randn(1000).astype(np.float32)
    X_val = np.random.randn(200, 100).astype(np.float32)
    y_val = np.random.randn(200).astype(np.float32)

    # Train
    print("\nTraining test model...")
    history = trainer.fit(X_train, y_train, X_val, y_val, epochs=20, verbose=True)

    # Predict
    predictions = trainer.predict(X_val)
    print(f"\nPredictions shape: {predictions.shape}")
