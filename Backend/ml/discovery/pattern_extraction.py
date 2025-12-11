"""
Universal Pattern Extraction

Discovers patterns that work across all stocks:
- Clusters similar prediction conditions
- Extracts human-readable rules
- Identifies regime-dependent patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import warnings


@dataclass
class UniversalPattern:
    """Container for a discovered universal pattern."""
    pattern_id: int
    description: str
    rules: List[str]
    avg_return: float
    win_rate: float
    n_samples: int
    feature_ranges: Dict[str, Tuple[float, float]]
    confidence: float


class UniversalPatternExtractor:
    """
    Extracts universal trading patterns from ML model insights.

    Methods:
    - Rule extraction from decision trees
    - Clustering of high-prediction scenarios
    - Cross-stock pattern validation
    """

    def __init__(
        self,
        feature_names: List[str],
        n_patterns: int = 10,
        min_samples_pattern: int = 100
    ):
        """
        Initialize pattern extractor.

        Args:
            feature_names: List of feature names
            n_patterns: Number of patterns to extract
            min_samples_pattern: Minimum samples for a valid pattern
        """
        self.feature_names = feature_names
        self.n_patterns = n_patterns
        self.min_samples_pattern = min_samples_pattern

        self.patterns: List[UniversalPattern] = []
        self.scaler = StandardScaler()

    def extract_patterns(
        self,
        X: np.ndarray,
        y: np.ndarray,
        predictions: np.ndarray,
        shap_values: Optional[np.ndarray] = None
    ) -> List[UniversalPattern]:
        """
        Extract universal patterns from model predictions.

        Args:
            X: Feature matrix
            y: Actual returns
            predictions: Model predictions
            shap_values: Optional SHAP values for feature importance

        Returns:
            List of discovered patterns
        """
        self.patterns = []

        # Method 1: Extract rules from decision tree approximation
        tree_patterns = self._extract_tree_rules(X, predictions)
        self.patterns.extend(tree_patterns)

        # Method 2: Cluster high-prediction scenarios
        cluster_patterns = self._extract_cluster_patterns(X, y, predictions)
        self.patterns.extend(cluster_patterns)

        # Method 3: Extract patterns from extreme predictions
        extreme_patterns = self._extract_extreme_patterns(X, y, predictions)
        self.patterns.extend(extreme_patterns)

        # Deduplicate and rank patterns
        self.patterns = self._rank_and_filter_patterns(self.patterns)

        print(f"Extracted {len(self.patterns)} universal patterns")
        return self.patterns

    def _extract_tree_rules(
        self,
        X: np.ndarray,
        predictions: np.ndarray,
        max_depth: int = 4
    ) -> List[UniversalPattern]:
        """Extract rules using a surrogate decision tree."""
        patterns = []

        # Fit decision tree to approximate model predictions
        tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=self.min_samples_pattern)
        tree.fit(X, predictions)

        # Extract leaf information
        leaf_ids = tree.apply(X)
        unique_leaves = np.unique(leaf_ids)

        for leaf_id in unique_leaves:
            mask = leaf_ids == leaf_id
            n_samples = mask.sum()

            if n_samples < self.min_samples_pattern:
                continue

            # Get path to this leaf
            rules = self._get_tree_rules(tree, leaf_id, X[mask])

            if not rules:
                continue

            # Calculate statistics
            leaf_predictions = predictions[mask]
            avg_pred = leaf_predictions.mean()

            # Get feature ranges for this leaf
            feature_ranges = self._get_feature_ranges(X[mask])

            pattern = UniversalPattern(
                pattern_id=len(patterns),
                description=f"Tree Rule {len(patterns) + 1}",
                rules=rules,
                avg_return=avg_pred,
                win_rate=(leaf_predictions > 0).mean(),
                n_samples=n_samples,
                feature_ranges=feature_ranges,
                confidence=min(1.0, n_samples / 1000)
            )
            patterns.append(pattern)

        return patterns[:self.n_patterns // 3]

    def _get_tree_rules(
        self,
        tree: DecisionTreeRegressor,
        leaf_id: int,
        X_leaf: np.ndarray
    ) -> List[str]:
        """Extract decision rules leading to a leaf node."""
        rules = []

        # Get tree structure
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right

        # Trace path from root to leaf
        def find_path(node_id, target_leaf, path):
            if node_id == target_leaf:
                return path

            if children_left[node_id] != -1:
                # Try left child
                left_path = find_path(children_left[node_id], target_leaf, path + [('left', node_id)])
                if left_path is not None:
                    return left_path

            if children_right[node_id] != -1:
                # Try right child
                right_path = find_path(children_right[node_id], target_leaf, path + [('right', node_id)])
                if right_path is not None:
                    return right_path

            return None

        path = find_path(0, leaf_id, [])

        if path is None:
            return rules

        for direction, node_id in path:
            feat_idx = feature[node_id]
            if feat_idx < 0:
                continue

            feat_name = self.feature_names[feat_idx]
            thresh = threshold[node_id]

            if direction == 'left':
                rules.append(f"{feat_name} <= {thresh:.4f}")
            else:
                rules.append(f"{feat_name} > {thresh:.4f}")

        return rules

    def _extract_cluster_patterns(
        self,
        X: np.ndarray,
        y: np.ndarray,
        predictions: np.ndarray,
        n_clusters: int = 10
    ) -> List[UniversalPattern]:
        """Extract patterns by clustering high-prediction scenarios."""
        patterns = []

        # Focus on top predictions
        threshold = np.percentile(predictions, 75)
        high_pred_mask = predictions >= threshold

        if high_pred_mask.sum() < self.min_samples_pattern:
            return patterns

        X_high = X[high_pred_mask]
        y_high = y[high_pred_mask]
        pred_high = predictions[high_pred_mask]

        # Scale features for clustering
        X_scaled = self.scaler.fit_transform(X_high)

        # Cluster
        n_clusters = min(n_clusters, len(X_high) // self.min_samples_pattern)
        if n_clusters < 2:
            return patterns

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Analyze each cluster
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            n_samples = mask.sum()

            if n_samples < self.min_samples_pattern:
                continue

            # Get cluster center in original feature space
            center = self.scaler.inverse_transform(kmeans.cluster_centers_[cluster_id:cluster_id+1])[0]

            # Generate rules from cluster characteristics
            rules = self._generate_cluster_rules(center, X_high[mask])

            # Calculate statistics
            cluster_returns = y_high[mask]
            cluster_preds = pred_high[mask]

            pattern = UniversalPattern(
                pattern_id=len(patterns) + 100,
                description=f"Cluster Pattern {cluster_id + 1}",
                rules=rules,
                avg_return=cluster_returns.mean(),
                win_rate=(cluster_returns > 0).mean(),
                n_samples=n_samples,
                feature_ranges=self._get_feature_ranges(X_high[mask]),
                confidence=min(1.0, n_samples / 500)
            )
            patterns.append(pattern)

        return patterns[:self.n_patterns // 3]

    def _generate_cluster_rules(
        self,
        center: np.ndarray,
        X_cluster: np.ndarray,
        top_n: int = 5
    ) -> List[str]:
        """Generate human-readable rules from cluster center."""
        rules = []

        # Find features with most distinct values vs overall mean
        cluster_means = X_cluster.mean(axis=0)
        cluster_stds = X_cluster.std(axis=0)

        # Score features by how much they deviate from center
        deviations = np.abs(cluster_means - center)
        top_features = np.argsort(deviations)[-top_n:][::-1]

        for feat_idx in top_features:
            feat_name = self.feature_names[feat_idx]
            feat_mean = cluster_means[feat_idx]
            feat_std = cluster_stds[feat_idx]

            if feat_std > 0:
                lower = feat_mean - feat_std
                upper = feat_mean + feat_std
                rules.append(f"{feat_name} in [{lower:.3f}, {upper:.3f}]")

        return rules

    def _extract_extreme_patterns(
        self,
        X: np.ndarray,
        y: np.ndarray,
        predictions: np.ndarray
    ) -> List[UniversalPattern]:
        """Extract patterns from extreme prediction scenarios."""
        patterns = []

        # Top 10% predictions
        top_threshold = np.percentile(predictions, 90)
        top_mask = predictions >= top_threshold

        if top_mask.sum() >= self.min_samples_pattern:
            patterns.append(self._create_extreme_pattern(
                X[top_mask], y[top_mask], predictions[top_mask],
                "High Confidence Bullish", pattern_id=200
            ))

        # Bottom 10% predictions
        bottom_threshold = np.percentile(predictions, 10)
        bottom_mask = predictions <= bottom_threshold

        if bottom_mask.sum() >= self.min_samples_pattern:
            patterns.append(self._create_extreme_pattern(
                X[bottom_mask], y[bottom_mask], predictions[bottom_mask],
                "High Confidence Bearish", pattern_id=201
            ))

        return [p for p in patterns if p is not None]

    def _create_extreme_pattern(
        self,
        X: np.ndarray,
        y: np.ndarray,
        predictions: np.ndarray,
        description: str,
        pattern_id: int
    ) -> Optional[UniversalPattern]:
        """Create pattern from extreme prediction subset."""
        if len(X) < self.min_samples_pattern:
            return None

        # Find distinguishing features
        means = X.mean(axis=0)
        stds = X.std(axis=0)

        # Top features by absolute mean (normalized)
        global_std = stds.mean() + 1e-10
        normalized_means = means / global_std
        top_features = np.argsort(np.abs(normalized_means))[-5:][::-1]

        rules = []
        for feat_idx in top_features:
            feat_name = self.feature_names[feat_idx]
            feat_mean = means[feat_idx]
            feat_std = stds[feat_idx]

            if feat_mean > 0:
                rules.append(f"{feat_name} high ({feat_mean:.3f} ± {feat_std:.3f})")
            else:
                rules.append(f"{feat_name} low ({feat_mean:.3f} ± {feat_std:.3f})")

        return UniversalPattern(
            pattern_id=pattern_id,
            description=description,
            rules=rules,
            avg_return=y.mean(),
            win_rate=(y > 0).mean(),
            n_samples=len(y),
            feature_ranges=self._get_feature_ranges(X),
            confidence=min(1.0, len(y) / 500)
        )

    def _get_feature_ranges(self, X: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Get feature value ranges for a subset."""
        ranges = {}
        for i, name in enumerate(self.feature_names):
            ranges[name] = (float(X[:, i].min()), float(X[:, i].max()))
        return ranges

    def _rank_and_filter_patterns(
        self,
        patterns: List[UniversalPattern]
    ) -> List[UniversalPattern]:
        """Rank patterns by quality and remove duplicates."""
        if not patterns:
            return []

        # Score patterns
        scored_patterns = []
        for p in patterns:
            score = (
                p.avg_return * 100 +  # Favor higher returns
                p.win_rate * 50 +      # Favor higher win rates
                np.log(p.n_samples) * 10 +  # Favor more samples
                p.confidence * 20      # Favor higher confidence
            )
            scored_patterns.append((score, p))

        # Sort by score
        scored_patterns.sort(key=lambda x: x[0], reverse=True)

        # Take top patterns
        return [p for _, p in scored_patterns[:self.n_patterns]]

    def get_pattern_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all patterns."""
        data = []
        for p in self.patterns:
            data.append({
                'pattern_id': p.pattern_id,
                'description': p.description,
                'avg_return': p.avg_return,
                'win_rate': p.win_rate,
                'n_samples': p.n_samples,
                'confidence': p.confidence,
                'num_rules': len(p.rules),
                'rules': ' AND '.join(p.rules[:3])  # First 3 rules
            })

        return pd.DataFrame(data)

    def generate_report(self) -> str:
        """Generate human-readable pattern report."""
        lines = [
            "=" * 70,
            "UNIVERSAL PATTERN DISCOVERY REPORT",
            "=" * 70,
            ""
        ]

        for i, pattern in enumerate(self.patterns, 1):
            lines.extend([
                f"Pattern #{i}: {pattern.description}",
                "-" * 50,
                f"  Avg Return: {pattern.avg_return:.4f} ({pattern.avg_return * 100:.2f}%)",
                f"  Win Rate: {pattern.win_rate:.2%}",
                f"  Samples: {pattern.n_samples}",
                f"  Confidence: {pattern.confidence:.2f}",
                "",
                "  Rules:",
            ])

            for rule in pattern.rules[:5]:
                lines.append(f"    • {rule}")

            lines.append("")

        lines.extend([
            "=" * 70,
            f"Total Patterns Discovered: {len(self.patterns)}",
            "=" * 70
        ])

        return "\n".join(lines)

    def to_dict(self) -> List[Dict]:
        """Convert patterns to JSON-serializable format."""
        return [
            {
                'pattern_id': p.pattern_id,
                'description': p.description,
                'rules': p.rules,
                'avg_return': float(p.avg_return),
                'win_rate': float(p.win_rate),
                'n_samples': int(p.n_samples),
                'confidence': float(p.confidence)
            }
            for p in self.patterns
        ]


if __name__ == '__main__':
    # Test pattern extraction
    print("Testing UniversalPatternExtractor...")

    # Generate test data
    np.random.seed(42)
    n_samples = 2000
    n_features = 50

    X = np.random.randn(n_samples, n_features)
    # Create patterns
    y = (
        X[:, 0] > 0.5 * (X[:, 1] < 0) * 0.05 +  # Pattern 1
        (X[:, 2] > 1) * 0.03 +                    # Pattern 2
        np.random.randn(n_samples) * 0.01
    )
    predictions = y + np.random.randn(n_samples) * 0.005

    feature_names = [f'feature_{i}' for i in range(n_features)]

    # Extract patterns
    extractor = UniversalPatternExtractor(feature_names, n_patterns=10)
    patterns = extractor.extract_patterns(X, y, predictions)

    # Print report
    print(extractor.generate_report())

    # Print summary
    print("\nPattern Summary:")
    print(extractor.get_pattern_summary())
