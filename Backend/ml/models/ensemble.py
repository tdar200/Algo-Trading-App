"""
GPU-Accelerated Ensemble Model for Pattern Discovery

Combines multiple models for robust predictions:
- XGBoost (GPU-accelerated via cuda_hist)
- LightGBM (GPU-accelerated)
- Random Forest (sklearn, optionally GPU via cuML)
- PyTorch Neural Network (CUDA)

Stacking meta-learner combines base model predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
from pathlib import Path

from .neural_net import PatternNet, PatternNetTrainer, get_gpu_info


class UniversalPatternEnsemble:
    """
    Ensemble model combining multiple ML algorithms for pattern discovery.

    GPU Acceleration:
    - XGBoost: tree_method='cuda_hist', device='cuda'
    - LightGBM: device='gpu'
    - Neural Network: PyTorch CUDA

    Stacking: Meta-learner trained on base model predictions.
    """

    def __init__(
        self,
        task: str = 'regression',
        use_gpu: bool = True,
        n_jobs: int = -1
    ):
        """
        Initialize ensemble.

        Args:
            task: 'regression' or 'classification'
            use_gpu: Whether to use GPU acceleration
            n_jobs: Number of CPU jobs for parallel processing
        """
        self.task = task
        self.use_gpu = use_gpu and self._check_gpu_available()
        self.n_jobs = n_jobs

        # Feature scaler
        self.scaler = StandardScaler()

        # Base models
        self.base_models: Dict[str, Any] = {}
        self.meta_model = None

        # Training state
        self.is_fitted = False
        self.feature_names: List[str] = []

        print(f"Ensemble initialized - GPU: {self.use_gpu}, Task: {task}")

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available."""
        gpu_info = get_gpu_info()
        if gpu_info['available']:
            print(f"GPU detected: {gpu_info['device_name']}")
            print(f"VRAM: {gpu_info['memory_total_gb']:.1f} GB")
            return True
        return False

    def _create_xgboost_model(self, input_dim: int) -> xgb.XGBRegressor:
        """Create XGBoost model with GPU support."""
        params = {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': self.n_jobs
        }

        if self.use_gpu:
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'

        if self.task == 'regression':
            return xgb.XGBRegressor(**params)
        else:
            params['objective'] = 'binary:logistic'
            return xgb.XGBClassifier(**params)

    def _create_lightgbm_model(self, input_dim: int) -> lgb.LGBMRegressor:
        """Create LightGBM model with GPU support."""
        params = {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': self.n_jobs,
            'verbose': -1
        }

        if self.use_gpu:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0

        if self.task == 'regression':
            return lgb.LGBMRegressor(**params)
        else:
            return lgb.LGBMClassifier(**params)

    def _create_random_forest_model(self, input_dim: int):
        """Create Random Forest model."""
        params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': self.n_jobs
        }

        if self.task == 'regression':
            return RandomForestRegressor(**params)
        else:
            return RandomForestClassifier(**params)

    def _create_neural_net_model(self, input_dim: int) -> Tuple[PatternNet, PatternNetTrainer]:
        """Create PyTorch neural network with GPU support."""
        output_dim = 1 if self.task == 'regression' else 2

        model = PatternNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[256, 128, 64],
            dropout=0.3,
            task=self.task
        )

        trainer = PatternNetTrainer(
            model=model,
            learning_rate=1e-3,
            weight_decay=1e-4,
            use_gpu=self.use_gpu,
            use_mixed_precision=self.use_gpu
        )

        return model, trainer

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ) -> 'UniversalPatternEnsemble':
        """
        Train the ensemble model.

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: List of feature names
            progress_callback: Callback function(model_name, progress)

        Returns:
            self
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        input_dim = X.shape[1]

        # Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None

        # Handle NaN/Inf values
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        if X_val_scaled is not None:
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0, posinf=0, neginf=0)

        # Train base models
        base_predictions_train = []
        base_predictions_val = [] if X_val is not None else None

        # 1. XGBoost
        print("\nTraining XGBoost...")
        if progress_callback:
            progress_callback('xgboost', 0)

        self.base_models['xgboost'] = self._create_xgboost_model(input_dim)
        self.base_models['xgboost'].fit(
            X_scaled, y,
            eval_set=[(X_val_scaled, y_val)] if X_val is not None else None,
            verbose=False
        )

        xgb_pred_train = self.base_models['xgboost'].predict(X_scaled)
        base_predictions_train.append(xgb_pred_train)
        if X_val is not None:
            base_predictions_val.append(self.base_models['xgboost'].predict(X_val_scaled))

        if progress_callback:
            progress_callback('xgboost', 100)

        # 2. LightGBM
        print("\nTraining LightGBM...")
        if progress_callback:
            progress_callback('lightgbm', 0)

        self.base_models['lightgbm'] = self._create_lightgbm_model(input_dim)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.base_models['lightgbm'].fit(
                X_scaled, y,
                eval_set=[(X_val_scaled, y_val)] if X_val is not None else None
            )

        lgb_pred_train = self.base_models['lightgbm'].predict(X_scaled)
        base_predictions_train.append(lgb_pred_train)
        if X_val is not None:
            base_predictions_val.append(self.base_models['lightgbm'].predict(X_val_scaled))

        if progress_callback:
            progress_callback('lightgbm', 100)

        # 3. Random Forest
        print("\nTraining Random Forest...")
        if progress_callback:
            progress_callback('random_forest', 0)

        self.base_models['random_forest'] = self._create_random_forest_model(input_dim)
        self.base_models['random_forest'].fit(X_scaled, y)

        rf_pred_train = self.base_models['random_forest'].predict(X_scaled)
        base_predictions_train.append(rf_pred_train)
        if X_val is not None:
            base_predictions_val.append(self.base_models['random_forest'].predict(X_val_scaled))

        if progress_callback:
            progress_callback('random_forest', 100)

        # 4. Neural Network
        print("\nTraining Neural Network...")
        if progress_callback:
            progress_callback('neural_net', 0)

        nn_model, nn_trainer = self._create_neural_net_model(input_dim)
        self.base_models['neural_net'] = (nn_model, nn_trainer)

        nn_trainer.fit(
            X_scaled.astype(np.float32),
            y.astype(np.float32),
            X_val_scaled.astype(np.float32) if X_val_scaled is not None else None,
            y_val.astype(np.float32) if y_val is not None else None,
            epochs=100,
            batch_size=256,
            early_stopping_patience=10,
            verbose=False
        )

        nn_pred_train = nn_trainer.predict(X_scaled.astype(np.float32)).flatten()
        base_predictions_train.append(nn_pred_train)
        if X_val is not None:
            base_predictions_val.append(
                nn_trainer.predict(X_val_scaled.astype(np.float32)).flatten()
            )

        if progress_callback:
            progress_callback('neural_net', 100)

        # 5. Train meta-learner (stacking)
        print("\nTraining meta-learner...")
        if progress_callback:
            progress_callback('meta_learner', 0)

        meta_features_train = np.column_stack(base_predictions_train)

        if self.task == 'regression':
            self.meta_model = Ridge(alpha=1.0)
        else:
            self.meta_model = LogisticRegression(C=1.0, max_iter=1000)

        self.meta_model.fit(meta_features_train, y)

        if progress_callback:
            progress_callback('meta_learner', 100)

        self.is_fitted = True
        print("\nEnsemble training complete!")

        # Calculate validation metrics
        if X_val is not None:
            val_pred = self.predict(X_val)
            if self.task == 'regression':
                from sklearn.metrics import mean_squared_error, r2_score
                mse = mean_squared_error(y_val, val_pred)
                r2 = r2_score(y_val, val_pred)
                print(f"Validation MSE: {mse:.6f}, R2: {r2:.4f}")
            else:
                from sklearn.metrics import accuracy_score
                accuracy = accuracy_score(y_val, (val_pred > 0.5).astype(int))
                print(f"Validation Accuracy: {accuracy:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble.

        Args:
            X: Input features

        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

        # Get base model predictions
        base_predictions = []

        # XGBoost
        base_predictions.append(self.base_models['xgboost'].predict(X_scaled))

        # LightGBM
        base_predictions.append(self.base_models['lightgbm'].predict(X_scaled))

        # Random Forest
        base_predictions.append(self.base_models['random_forest'].predict(X_scaled))

        # Neural Network
        nn_model, nn_trainer = self.base_models['neural_net']
        base_predictions.append(nn_trainer.predict(X_scaled.astype(np.float32)).flatten())

        # Stack and predict with meta-learner
        meta_features = np.column_stack(base_predictions)
        predictions = self.meta_model.predict(meta_features)

        return predictions

    def predict_with_models(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model.

        Args:
            X: Input features

        Returns:
            Dict mapping model name to predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

        predictions = {}

        predictions['xgboost'] = self.base_models['xgboost'].predict(X_scaled)
        predictions['lightgbm'] = self.base_models['lightgbm'].predict(X_scaled)
        predictions['random_forest'] = self.base_models['random_forest'].predict(X_scaled)

        nn_model, nn_trainer = self.base_models['neural_net']
        predictions['neural_net'] = nn_trainer.predict(X_scaled.astype(np.float32)).flatten()

        predictions['ensemble'] = self.predict(X)

        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.

        Returns:
            DataFrame with feature importance from each model
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        importance_dict = {'feature': self.feature_names}

        # XGBoost importance
        importance_dict['xgboost'] = self.base_models['xgboost'].feature_importances_

        # LightGBM importance
        importance_dict['lightgbm'] = self.base_models['lightgbm'].feature_importances_

        # Random Forest importance
        importance_dict['random_forest'] = self.base_models['random_forest'].feature_importances_

        df = pd.DataFrame(importance_dict)

        # Average importance
        df['average'] = df[['xgboost', 'lightgbm', 'random_forest']].mean(axis=1)
        df = df.sort_values('average', ascending=False)

        return df

    def save(self, path: str):
        """Save the ensemble model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save sklearn models
        joblib.dump(self.scaler, path / 'scaler.joblib')
        joblib.dump(self.base_models['xgboost'], path / 'xgboost.joblib')
        joblib.dump(self.base_models['lightgbm'], path / 'lightgbm.joblib')
        joblib.dump(self.base_models['random_forest'], path / 'random_forest.joblib')
        joblib.dump(self.meta_model, path / 'meta_model.joblib')

        # Save neural network
        nn_model, nn_trainer = self.base_models['neural_net']
        nn_trainer.save(str(path / 'neural_net.pt'))

        # Save metadata
        metadata = {
            'task': self.task,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        joblib.dump(metadata, path / 'metadata.joblib')

        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load the ensemble model."""
        path = Path(path)

        # Load metadata
        metadata = joblib.load(path / 'metadata.joblib')
        self.task = metadata['task']
        self.feature_names = metadata['feature_names']
        self.is_fitted = metadata['is_fitted']

        # Load sklearn models
        self.scaler = joblib.load(path / 'scaler.joblib')
        self.base_models['xgboost'] = joblib.load(path / 'xgboost.joblib')
        self.base_models['lightgbm'] = joblib.load(path / 'lightgbm.joblib')
        self.base_models['random_forest'] = joblib.load(path / 'random_forest.joblib')
        self.meta_model = joblib.load(path / 'meta_model.joblib')

        # Load neural network
        input_dim = len(self.feature_names)
        nn_model, nn_trainer = self._create_neural_net_model(input_dim)
        nn_trainer.load(str(path / 'neural_net.pt'))
        self.base_models['neural_net'] = (nn_model, nn_trainer)

        print(f"Model loaded from {path}")


if __name__ == '__main__':
    # Test the ensemble
    print("Testing UniversalPatternEnsemble...")

    # Generate random test data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1

    # Split
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Train ensemble
    ensemble = UniversalPatternEnsemble(task='regression', use_gpu=True)
    ensemble.fit(X_train, y_train, X_val, y_val)

    # Get feature importance
    importance = ensemble.get_feature_importance()
    print("\nTop 10 features:")
    print(importance.head(10))
