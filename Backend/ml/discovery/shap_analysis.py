"""
SHAP Analysis for Feature Importance

Uses SHAP (SHapley Additive exPlanations) for:
- Global feature importance rankings
- Per-prediction explanations
- Feature interaction detection
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Optional, Tuple, Any
import warnings


class SHAPAnalyzer:
    """
    SHAP-based feature importance and explanation analysis.

    Provides:
    - Global feature importance
    - Local explanations for individual predictions
    - Feature interactions
    - Summary statistics
    """

    def __init__(self, model: Any, feature_names: List[str]):
        """
        Initialize SHAP analyzer.

        Args:
            model: Trained model (must have predict method)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.base_value = None

    def compute_shap_values(
        self,
        X: np.ndarray,
        max_samples: int = 1000,
        background_samples: int = 100
    ) -> np.ndarray:
        """
        Compute SHAP values for the dataset.

        Args:
            X: Feature matrix
            max_samples: Maximum samples to explain (for performance)
            background_samples: Background samples for KernelExplainer

        Returns:
            SHAP values array
        """
        # Sample if dataset is large
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
            indices = np.arange(len(X))

        # Create background dataset
        if len(X) > background_samples:
            bg_indices = np.random.choice(len(X), background_samples, replace=False)
            X_background = X[bg_indices]
        else:
            X_background = X

        # Try TreeExplainer first (faster for tree models)
        try:
            # Check if model has tree structure
            if hasattr(self.model, 'get_booster') or hasattr(self.model, 'booster_'):
                self.explainer = shap.TreeExplainer(self.model)
            elif hasattr(self.model, 'estimators_'):  # Random Forest
                self.explainer = shap.TreeExplainer(self.model)
            else:
                raise ValueError("Not a tree model")

            self.shap_values = self.explainer.shap_values(X_sample)
            self.base_value = self.explainer.expected_value

        except Exception:
            # Fall back to KernelExplainer (works with any model)
            print("Using KernelExplainer (slower but universal)...")

            def predict_fn(x):
                return self.model.predict(x)

            self.explainer = shap.KernelExplainer(predict_fn, X_background)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.shap_values = self.explainer.shap_values(X_sample, nsamples=100)

            self.base_value = self.explainer.expected_value

        # Handle multi-output models
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[0]

        print(f"SHAP values computed for {len(X_sample)} samples")
        return self.shap_values

    def get_global_importance(self) -> pd.DataFrame:
        """
        Get global feature importance based on mean absolute SHAP values.

        Returns:
            DataFrame with features ranked by importance
        """
        if self.shap_values is None:
            raise ValueError("Run compute_shap_values() first")

        # Mean absolute SHAP value
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap,
            'mean_shap': self.shap_values.mean(axis=0),
            'std_shap': self.shap_values.std(axis=0)
        })

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        importance_df = importance_df.reset_index(drop=True)

        return importance_df

    def get_top_features(self, n: int = 20) -> List[str]:
        """
        Get top N most important features.

        Args:
            n: Number of top features

        Returns:
            List of feature names
        """
        importance_df = self.get_global_importance()
        return importance_df['feature'].head(n).tolist()

    def explain_prediction(
        self,
        X_single: np.ndarray,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Explain a single prediction.

        Args:
            X_single: Single sample feature vector
            top_n: Number of top contributing features to show

        Returns:
            DataFrame with feature contributions
        """
        if self.explainer is None:
            raise ValueError("Run compute_shap_values() first")

        # Compute SHAP values for single sample
        if len(X_single.shape) == 1:
            X_single = X_single.reshape(1, -1)

        shap_vals = self.explainer.shap_values(X_single)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        shap_vals = shap_vals.flatten()

        # Create explanation DataFrame
        explanation_df = pd.DataFrame({
            'feature': self.feature_names,
            'value': X_single.flatten(),
            'shap_value': shap_vals,
            'contribution': np.abs(shap_vals)
        })

        explanation_df = explanation_df.sort_values('contribution', ascending=False)

        # Add direction
        explanation_df['direction'] = np.where(
            explanation_df['shap_value'] > 0, 'increases', 'decreases'
        )

        return explanation_df.head(top_n)

    def get_feature_interactions(
        self,
        X: np.ndarray,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Compute feature interaction strengths.

        Args:
            X: Feature matrix
            top_n: Number of top interactions to return

        Returns:
            DataFrame with feature pair interactions
        """
        if self.shap_values is None:
            # Compute SHAP values if not already done
            self.compute_shap_values(X)

        n_features = len(self.feature_names)
        interactions = []

        # Compute correlation between SHAP values (proxy for interaction)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = np.corrcoef(self.shap_values[:, i], self.shap_values[:, j])[0, 1]
                if not np.isnan(corr):
                    interactions.append({
                        'feature_1': self.feature_names[i],
                        'feature_2': self.feature_names[j],
                        'interaction_strength': abs(corr),
                        'correlation': corr
                    })

        interactions_df = pd.DataFrame(interactions)
        interactions_df = interactions_df.sort_values('interaction_strength', ascending=False)

        return interactions_df.head(top_n)

    def get_feature_effects(self) -> Dict[str, Dict]:
        """
        Analyze the direction and magnitude of feature effects.

        Returns:
            Dict mapping feature name to effect statistics
        """
        if self.shap_values is None:
            raise ValueError("Run compute_shap_values() first")

        effects = {}

        for i, feature in enumerate(self.feature_names):
            shap_col = self.shap_values[:, i]

            positive_effect = shap_col[shap_col > 0]
            negative_effect = shap_col[shap_col < 0]

            effects[feature] = {
                'mean_effect': shap_col.mean(),
                'abs_mean_effect': np.abs(shap_col).mean(),
                'positive_pct': len(positive_effect) / len(shap_col) * 100,
                'negative_pct': len(negative_effect) / len(shap_col) * 100,
                'max_positive': positive_effect.max() if len(positive_effect) > 0 else 0,
                'max_negative': negative_effect.min() if len(negative_effect) > 0 else 0
            }

        return effects

    def generate_summary(self) -> str:
        """
        Generate a text summary of feature importance.

        Returns:
            Human-readable summary string
        """
        importance_df = self.get_global_importance()
        top_10 = importance_df.head(10)

        summary_lines = [
            "=" * 60,
            "FEATURE IMPORTANCE SUMMARY (SHAP Analysis)",
            "=" * 60,
            "",
            "Top 10 Most Important Features:",
            "-" * 40
        ]

        for _, row in top_10.iterrows():
            direction = "↑" if row['mean_shap'] > 0 else "↓"
            summary_lines.append(
                f"{row['rank']:2}. {row['feature']:<30} "
                f"Importance: {row['importance']:.4f} {direction}"
            )

        # Add statistics
        summary_lines.extend([
            "",
            "-" * 40,
            f"Total features analyzed: {len(importance_df)}",
            f"Top 10 explain {importance_df['importance'].head(10).sum() / importance_df['importance'].sum() * 100:.1f}% of variation",
            "=" * 60
        ])

        return "\n".join(summary_lines)


def analyze_model_shap(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    max_samples: int = 1000
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Convenience function to run full SHAP analysis.

    Args:
        model: Trained model
        X: Feature matrix
        feature_names: Feature names
        max_samples: Max samples for SHAP computation

    Returns:
        Tuple of (importance_df, interactions_df, summary_text)
    """
    analyzer = SHAPAnalyzer(model, feature_names)
    analyzer.compute_shap_values(X, max_samples=max_samples)

    importance = analyzer.get_global_importance()
    interactions = analyzer.get_feature_interactions(X)
    summary = analyzer.generate_summary()

    return importance, interactions, summary


if __name__ == '__main__':
    # Test the SHAP analyzer
    from sklearn.ensemble import RandomForestRegressor

    print("Testing SHAPAnalyzer...")

    # Generate test data
    np.random.seed(42)
    n_samples = 500
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    # Create target with known feature importance
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] + np.random.randn(n_samples) * 0.5

    feature_names = [f'feature_{i}' for i in range(n_features)]

    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Run SHAP analysis
    importance, interactions, summary = analyze_model_shap(model, X, feature_names)

    print(summary)
    print("\nTop 5 Feature Interactions:")
    print(interactions.head())
