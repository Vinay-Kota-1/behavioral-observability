"""
Explainer - Feature-based Anomaly Explanations

This module provides explanations for why an event was flagged as anomalous.
Uses feature importance and deviation analysis to identify contributing factors.

Usage:
    from explainer import AnomalyExplainer
    
    explainer = AnomalyExplainer(model_path="models/anomaly_xgb.pkl")
    explanations = explainer.explain_batch(df)
    
    # Or from CLI
    python explainer.py --explain --limit 1000
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Import ModelConfig for pickle loading compatibility
from anomaly_model import ModelConfig

import warnings
warnings.filterwarnings('ignore')


@dataclass
class FeatureContribution:
    """A single feature's contribution to the anomaly score."""
    feature_name: str
    feature_value: float
    zscore: float  # How many std dev from mean
    importance: float  # Model importance weight
    contribution: float  # Combined score
    direction: str  # "high" or "low"
    
    def to_dict(self) -> Dict:
        return {
            "feature": self.feature_name,
            "value": self.feature_value,
            "zscore": self.zscore,
            "importance": self.importance,
            "contribution": self.contribution,
            "direction": self.direction
        }


@dataclass  
class Explanation:
    """Full explanation for a single prediction."""
    entity_id: str
    source_dataset: str
    anomaly_score: float
    is_anomaly: bool
    top_contributors: List[FeatureContribution]
    summary: str
    
    def to_dict(self) -> Dict:
        return {
            "entity_id": self.entity_id,
            "source_dataset": self.source_dataset,
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "top_contributors": [c.to_dict() for c in self.top_contributors],
            "summary": self.summary
        }


class AnomalyExplainer:
    """
    Feature importance-based explainer for anomaly detection.
    
    Uses:
    - XGBoost native feature importance
    - Z-score deviation from training data means
    - Combined contribution scoring
    """
    
    def __init__(self, model_path: str = "models/anomaly_xgb.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and compute feature statistics."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]
        
        # Extract feature importance from model
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(self.feature_names, importance))
        
        # Load or compute feature statistics
        stats_path = self.model_path.replace(".pkl", "_feature_stats.json")
        if Path(stats_path).exists():
            with open(stats_path) as f:
                self.feature_stats = json.load(f)
        else:
            # Will compute on first batch
            self.feature_stats = {}
        
        print(f"Loaded model with {len(self.feature_names)} features")
    
    def compute_feature_stats(self, df: pd.DataFrame):
        """Compute mean and std for each feature from data."""
        for feature in self.feature_names:
            if feature in df.columns:
                values = df[feature].dropna()
                if len(values) > 0:
                    self.feature_stats[feature] = {
                        "mean": float(values.mean()),
                        "std": float(max(values.std(), 1e-6))
                    }
        
        # Save for future use
        stats_path = self.model_path.replace(".pkl", "_feature_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.feature_stats, f, indent=2)
    
    def _get_zscore(self, feature: str, value: float) -> float:
        """Get z-score for a feature value."""
        if feature not in self.feature_stats or pd.isna(value):
            return 0.0
        stats = self.feature_stats[feature]
        return (value - stats["mean"]) / stats["std"]
    
    def _generate_summary(
        self, 
        contributors: List[FeatureContribution],
        anomaly_score: float
    ) -> str:
        """Generate human-readable explanation."""
        if not contributors:
            return f"Anomaly score: {anomaly_score:.2%}. No significant contributors."
        
        # Get top 2 positive contributors
        top = sorted(contributors, key=lambda c: c.contribution, reverse=True)[:2]
        
        parts = []
        for c in top:
            name = c.feature_name.replace("_", " ").title()
            
            if c.zscore > 2:
                intensity = "very high"
            elif c.zscore > 1:
                intensity = "high"
            elif c.zscore < -2:
                intensity = "very low"
            elif c.zscore < -1:
                intensity = "low"
            else:
                intensity = "unusual"
            
            if "count" in c.feature_name:
                parts.append(f"{intensity} {name} ({int(c.feature_value)})")
            elif "seconds" in c.feature_name:
                hours = c.feature_value / 3600
                if hours > 24:
                    parts.append(f"{intensity} time gap ({hours/24:.1f} days)")
                else:
                    parts.append(f"{intensity} time gap ({hours:.1f}h)")
            else:
                parts.append(f"{intensity} {name}")
        
        reason = " and ".join(parts)
        return f"Anomaly (score: {anomaly_score:.2%}) due to {reason}."
    
    def explain_row(
        self, 
        row: pd.Series,
        top_n: int = 5
    ) -> Explanation:
        """Generate explanation for a single row."""
        
        # Get prediction
        X = row[self.feature_names].fillna(0).values.reshape(1, -1)
        proba = self.model.predict_proba(X)[0, 1]
        is_anomaly = proba >= 0.5
        
        # Compute contributions for each feature
        contributions = []
        for feature in self.feature_names:
            if feature not in row:
                continue
            
            value = float(row[feature]) if pd.notna(row[feature]) else 0.0
            zscore = self._get_zscore(feature, value)
            importance = self.feature_importance.get(feature, 0)
            
            # Contribution = |zscore| * importance (higher for unusual + important features)
            contribution = abs(zscore) * importance
            direction = "high" if zscore > 0 else "low"
            
            contributions.append(FeatureContribution(
                feature_name=feature,
                feature_value=value,
                zscore=zscore,
                importance=importance,
                contribution=contribution,
                direction=direction
            ))
        
        # Sort by contribution
        contributions.sort(key=lambda c: c.contribution, reverse=True)
        top_contributors = contributions[:top_n]
        
        # Generate summary
        summary = self._generate_summary(top_contributors, proba)
        
        return Explanation(
            entity_id=str(row.get("entity_id", "unknown")),
            source_dataset=str(row.get("source_dataset", "unknown")),
            anomaly_score=float(proba),
            is_anomaly=is_anomaly,
            top_contributors=top_contributors,
            summary=summary
        )
    
    def explain_batch(
        self, 
        df: pd.DataFrame,
        top_n: int = 5
    ) -> List[Explanation]:
        """Generate explanations for a batch."""
        
        # Compute stats if not already done
        if not self.feature_stats:
            print("Computing feature statistics...")
            self.compute_feature_stats(df)
        
        explanations = []
        for _, row in df.iterrows():
            exp = self.explain_row(row, top_n)
            explanations.append(exp)
        
        return explanations
    
    def explain_anomalies(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
        top_n: int = 5
    ) -> List[Explanation]:
        """Explain only anomalies, sorted by score."""
        
        # Compute stats if needed
        if not self.feature_stats:
            self.compute_feature_stats(df)
        
        # Get predictions
        X = df[self.feature_names].fillna(0)
        proba = self.model.predict_proba(X)[:, 1]
        
        # Filter anomalies
        anomaly_mask = proba >= threshold
        anomaly_df = df[anomaly_mask].copy()
        anomaly_df["_score"] = proba[anomaly_mask]
        anomaly_df = anomaly_df.sort_values("_score", ascending=False)
        
        # Generate explanations
        explanations = []
        for _, row in anomaly_df.iterrows():
            exp = self.explain_row(row, top_n)
            explanations.append(exp)
        
        return explanations
    
    def format_explanation(self, exp: Explanation) -> str:
        """Format explanation as text."""
        lines = [
            "=" * 60,
            f"Entity: {exp.entity_id} ({exp.source_dataset})",
            f"Anomaly Score: {exp.anomaly_score:.2%}",
            f"Is Anomaly: {'YES ⚠️' if exp.is_anomaly else 'NO ✓'}",
            "-" * 60,
            f"Summary: {exp.summary}",
            "",
            "Contributing Factors:"
        ]
        
        for i, c in enumerate(exp.top_contributors, 1):
            arrow = "↑" if c.direction == "high" else "↓"
            lines.append(
                f"  {i}. {c.feature_name}: {c.feature_value:.2f} "
                f"(z={c.zscore:+.2f}{arrow}, importance={c.importance:.3f})"
            )
        
        return "\n".join(lines)
    
    def get_global_importance(self) -> pd.DataFrame:
        """Get global feature importance."""
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": [self.feature_importance[f] for f in self.feature_names]
        }).sort_values("importance", ascending=False)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Anomaly Explainer")
    parser.add_argument("--explain", action="store_true", help="Explain anomalies")
    parser.add_argument("--limit", type=int, default=1000, help="Sample size")
    parser.add_argument("--top", type=int, default=10, help="Number to show")
    parser.add_argument("--importance", action="store_true", help="Show feature importance")
    
    args = parser.parse_args()
    
    engine = create_engine("postgresql://vinaykota:12345678@localhost:5432/fintech_lab")
    explainer = AnomalyExplainer()
    
    if args.importance:
        print("\nGlobal Feature Importance:")
        print(explainer.get_global_importance().to_string(index=False))
        return
    
    if args.explain:
        print(f"Loading {args.limit} samples...")
        with engine.connect() as conn:
            df = pd.read_sql(text(f"""
                SELECT * FROM sentinelrisk.features
                ORDER BY RANDOM()
                LIMIT {args.limit}
            """), conn)
        
        print(f"Loaded {len(df)} rows")
        print("Finding and explaining anomalies...\n")
        
        explanations = explainer.explain_anomalies(df, threshold=0.5, top_n=5)
        
        print(f"Found {len(explanations)} anomalies\n")
        
        for exp in explanations[:args.top]:
            print(explainer.format_explanation(exp))
            print()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
