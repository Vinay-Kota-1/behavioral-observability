"""
Anomaly Model - XGBoost Classifier for Anomaly Detection

This module trains and uses an XGBoost model for anomaly detection.
It combines:
- Features from the feature store (sentinelrisk.features)
- Labels from labels_outcomes table
- Business impact rules from adversarial_config.yaml

Usage:
    # Train the model
    python anomaly_model.py --train
    
    # Score new data
    python anomaly_model.py --score --limit 1000
    
    # Get predictions with business context
    python anomaly_model.py --predict --entity USER_123
"""

import numpy as np
import pandas as pd
import pickle
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    confusion_matrix, f1_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """Configuration for the anomaly model."""
    model_path: str = "models/anomaly_xgb.pkl"
    config_path: str = "adversarial_config.yaml"
    
    # Features to use for training
    # These come from feature_builder.py
    numeric_features: List[str] = None
    
    # XGBoost hyperparameters
    xgb_params: Dict = None
    
    def __post_init__(self):
        if self.numeric_features is None:
            self.numeric_features = [
                "event_count_1h", "event_count_24h",
                "metric_mean_24h", "metric_std_24h", "metric_z_24h",
                "delta_1", "pct_change_1h", "pct_change_24h",
                "time_since_last_event_seconds",
                "amount_diff_1", "amount_diff_2",
                "hour_of_day", "day_of_week",
                "sin_hourly", "cos_hourly", "sin_daily", "cos_daily"
            ]
        
        if self.xgb_params is None:
            self.xgb_params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "use_label_encoder": False,
                "random_state": 42
            }


class AnomalyModel:
    """
    XGBoost-based anomaly detection model.
    
    - Trains on features + labels
    - Outputs probability scores
    - Maps predictions to business impact
    """
    
    def __init__(self, engine: Engine, config: Optional[ModelConfig] = None):
        self.engine = engine
        self.config = config or ModelConfig()
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []
        self.business_rules: Dict = {}
        self.training_metadata: Dict = {}
        
        # Load business rules from adversarial_config.yaml
        self._load_business_rules()
    
    def _load_business_rules(self):
        """Load business impact rules from config."""
        config_path = Path(self.config.config_path)
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            
            self.business_rules = {}
            for scenario_name, scenario in cfg.get("scenarios", {}).items():
                if "business_impact" in scenario:
                    self.business_rules[scenario_name] = {
                        "source_dataset": scenario.get("source_dataset"),
                        "entity_type": scenario.get("entity_type"),
                        "anomaly_label": scenario.get("anomaly_label", {}),
                        "business_impact": scenario.get("business_impact")
                    }
            
            self.escalation_paths = cfg.get("escalation_paths", {})
            print(f"Loaded {len(self.business_rules)} business rules")
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    
    def load_training_data(self, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load features and labels for training.
        
        Returns:
            X: Feature DataFrame
            y: Label Series (1 = anomaly, 0 = normal)
        """
        print("Loading training data...")
        
        # Join features with labels - handle NULL ts_end and adversarial labels
        query = """
            SELECT f.*, 
                   CASE WHEN l.label IS NOT NULL AND l.label > 0 THEN 1 ELSE 0 END as is_anomaly,
                   l.label_type,
                   l.label as label_value
            FROM sentinelrisk.features f
            LEFT JOIN sentinelrisk.labels_outcomes l ON (
                -- Match by entity_id/scope_id and time window
                (f.entity_id = l.scope_id
                 AND f.ts >= l.ts_start
                 AND (l.ts_end IS NULL OR f.ts <= l.ts_end))
                -- OR match adversarial labels by ingest_batch_id
                OR (f.ingest_batch_id LIKE 'adversarial%' 
                    AND f.ingest_batch_id = l.ingest_batch_id
                    AND f.entity_id = l.scope_id)
            )
        """
        
        if sample_size:
            query += f" ORDER BY RANDOM() LIMIT {sample_size}"
        
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        
        print(f"Loaded {len(df)} rows")
        print(f"Anomaly rate: {df['is_anomaly'].mean():.2%}")
        
        # Select features
        available_features = [f for f in self.config.numeric_features if f in df.columns]
        self.feature_names = available_features
        
        X = df[available_features].copy()
        y = df["is_anomaly"].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        return X, y, df
    
    # =========================================================================
    # Training
    # =========================================================================
    
    def train(
        self, 
        sample_size: Optional[int] = 100000,
        test_size: float = 0.2,
        balance_classes: bool = True
    ) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Args:
            sample_size: Limit training data size
            test_size: Fraction for validation
            balance_classes: Upsample minority class
            
        Returns:
            Dictionary with training metrics
        """
        X, y, df = self.load_training_data(sample_size)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if y.sum() > 10 else None
        )
        
        print(f"\nTraining set: {len(X_train)} rows")
        print(f"Test set: {len(X_test)} rows")
        print(f"Features: {len(self.feature_names)}")
        
        # Handle class imbalance
        if balance_classes and y_train.sum() > 0:
            scale_pos_weight = (y_train == 0).sum() / max(y_train.sum(), 1)
            self.config.xgb_params["scale_pos_weight"] = min(scale_pos_weight, 100)
            print(f"Class weight: {self.config.xgb_params['scale_pos_weight']:.1f}")
        
        # Train model
        print("\nTraining XGBoost...")
        self.model = xgb.XGBClassifier(**self.config.xgb_params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_pred_proba) if y_test.sum() > 0 else 0,
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "precision": (y_pred & y_test).sum() / max(y_pred.sum(), 1),
            "recall": (y_pred & y_test).sum() / max(y_test.sum(), 1),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_count": len(self.feature_names),
            "anomaly_rate_train": float(y_train.mean()),
            "anomaly_rate_test": float(y_test.mean())
        }
        
        print(f"\n{'='*50}")
        print("TRAINING RESULTS")
        print(f"{'='*50}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        print(f"\nTop features:")
        print(importance.head(10).to_string(index=False))
        
        # Save training metadata
        self.training_metadata = {
            "trained_at": datetime.now().isoformat(),
            "metrics": metrics,
            "features": self.feature_names,
            "feature_importance": importance.to_dict(orient="records"),
            "config": {
                "sample_size": sample_size,
                "test_size": test_size,
                "xgb_params": self.config.xgb_params
            }
        }
        
        return metrics
    
    # =========================================================================
    # Saving/Loading
    # =========================================================================
    
    def save(self, path: Optional[str] = None):
        """Save model and metadata."""
        path = path or self.config.model_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Don't pickle the config dataclass - just save feature_names
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "training_metadata": self.training_metadata
        }
        
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        
        # Also save metadata as JSON for easy reading
        meta_path = path.replace(".pkl", "_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(self.training_metadata, f, indent=2, default=str)
        
        print(f"Saved model to {path}")
        print(f"Saved metadata to {meta_path}")
    
    def load(self, path: Optional[str] = None):
        """Load model from file."""
        path = path or self.config.model_path
        
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found at {path}")
        
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]
        self.training_metadata = model_data.get("training_metadata", {})
        
        print(f"Loaded model from {path}")
        print(f"Features: {len(self.feature_names)}")
        
        print(f"Loaded model from {path}")
        print(f"Features: {len(self.feature_names)}")
    
    # =========================================================================
    # Prediction
    # =========================================================================
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomaly scores for a DataFrame.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            DataFrame with added columns:
            - anomaly_score: 0-1 probability
            - is_anomaly: binary prediction
            - scenario_match: matched business rule
            - business_impact: impact details
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Prepare features
        X = df[self.feature_names].fillna(0)
        
        # Get predictions
        proba = self.model.predict_proba(X)[:, 1]
        
        result = df.copy()
        result["anomaly_score"] = proba
        result["is_anomaly"] = (proba >= 0.5).astype(int)
        
        # Match to business rules
        result["scenario_match"] = result.apply(
            lambda row: self._match_scenario(row), axis=1
        )
        
        result["business_impact"] = result["scenario_match"].apply(
            lambda s: self.business_rules.get(s, {}).get("business_impact", {})
        )
        
        return result
    
    def _match_scenario(self, row: pd.Series) -> Optional[str]:
        """Match a row to the most likely scenario based on source and entity type."""
        source = row.get("source_dataset", "")
        entity = row.get("entity_type", "")
        
        for scenario_name, rule in self.business_rules.items():
            if rule.get("source_dataset") == source and rule.get("entity_type") == entity:
                return scenario_name
        
        return None
    
    def score_batch(self, limit: int = 1000) -> pd.DataFrame:
        """Score a batch of data from the features table."""
        with self.engine.connect() as conn:
            df = pd.read_sql(text(f"""
                SELECT * FROM sentinelrisk.features
                ORDER BY RANDOM()
                LIMIT {limit}
            """), conn)
        
        return self.predict(df)
    
    def get_top_anomalies(self, limit: int = 1000, top_n: int = 20) -> pd.DataFrame:
        """Get top anomalies by score."""
        scored = self.score_batch(limit)
        anomalies = scored[scored["is_anomaly"] == 1].copy()
        anomalies = anomalies.sort_values("anomaly_score", ascending=False)
        return anomalies.head(top_n)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Anomaly Model Training and Inference")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--score", action="store_true", help="Score sample data")
    parser.add_argument("--sample-size", type=int, default=100000, help="Training sample size")
    parser.add_argument("--limit", type=int, default=1000, help="Scoring limit")
    
    args = parser.parse_args()
    
    engine = create_engine("postgresql://vinaykota:12345678@localhost:5432/fintech_lab")
    model = AnomalyModel(engine)
    
    if args.train:
        print("="*60)
        print("TRAINING ANOMALY MODEL")
        print("="*60)
        
        metrics = model.train(sample_size=args.sample_size)
        model.save()
        
    elif args.score:
        print("="*60)
        print("SCORING DATA")
        print("="*60)
        
        model.load()
        anomalies = model.get_top_anomalies(limit=args.limit, top_n=20)
        
        if len(anomalies) > 0:
            print(f"\nTop {len(anomalies)} anomalies:")
            print("-"*60)
            cols = ["entity_id", "source_dataset", "anomaly_score", "scenario_match"]
            print(anomalies[cols].to_string())
        else:
            print("No anomalies detected in sample")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
