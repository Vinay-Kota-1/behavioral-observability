"""
Baseline Detector - Statistical Anomaly Detection

This module provides fast, always-on statistical anomaly detection as the
first layer of the detection pipeline. It uses no ML training, just math.

Methods:
- Z-score: How many standard deviations from the mean?
- IQR (Interquartile Range): Is this outside the whiskers?
- Rolling statistics: Is this unusual compared to recent history?
- Percentile-based: Is this in the extreme tails?

Usage:
    from baseline_detector import BaselineDetector
    
    detector = BaselineDetector(engine)
    scores = detector.score_batch(df)  # Returns anomaly scores 0-1
    
    # Or for single events
    score = detector.score_event(event_dict)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import json
import yaml
from pathlib import Path


@dataclass
class SignalProfile:
    """Baseline statistics for a signal (source_dataset + entity_type combo)."""
    signal_id: str
    source_dataset: str
    entity_type: str
    
    # Numeric feature statistics
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Format: {feature_name: {mean, std, p1, p5, p25, p50, p75, p95, p99, iqr}}
    
    # Temporal patterns
    hourly_counts: Optional[np.ndarray] = None  # 24-element array
    daily_counts: Optional[np.ndarray] = None   # 7-element array
    
    # Sample size used to compute stats
    sample_size: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "signal_id": self.signal_id,
            "source_dataset": self.source_dataset,
            "entity_type": self.entity_type,
            "feature_stats": self.feature_stats,
            "hourly_counts": self.hourly_counts.tolist() if self.hourly_counts is not None else None,
            "daily_counts": self.daily_counts.tolist() if self.daily_counts is not None else None,
            "sample_size": self.sample_size
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'SignalProfile':
        profile = cls(
            signal_id=d["signal_id"],
            source_dataset=d["source_dataset"],
            entity_type=d["entity_type"],
            feature_stats=d.get("feature_stats", {}),
            sample_size=d.get("sample_size", 0)
        )
        if d.get("hourly_counts"):
            profile.hourly_counts = np.array(d["hourly_counts"])
        if d.get("daily_counts"):
            profile.daily_counts = np.array(d["daily_counts"])
        return profile


class BaselineDetector:
    """
    Statistical baseline anomaly detector.
    
    Computes anomaly scores based on statistical deviation from learned baselines.
    No ML training required - just compute statistics from historical data.
    """
    
    # Features to analyze (numeric columns from feature store)
    NUMERIC_FEATURES = [
        "event_count_1h", "event_count_24h",
        "metric_mean_24h", "metric_std_24h", "metric_z_24h",
        "delta_1", "pct_change_1h", "pct_change_24h",
        "time_since_last_event_seconds",
        "amount_diff_1", "amount_diff_2",
        "amount"
    ]
    
    def __init__(self, engine: Engine, profiles_path: Optional[str] = None):
        """
        Initialize detector.
        
        Args:
            engine: SQLAlchemy database engine
            profiles_path: Path to save/load signal profiles
        """
        self.engine = engine
        self.profiles_path = profiles_path or "baseline_profiles.json"
        self.profiles: Dict[str, SignalProfile] = {}
        
    def _signal_id(self, source_dataset: str, entity_type: str) -> str:
        """Generate unique signal identifier."""
        return f"{source_dataset}::{entity_type}"
    
    # =========================================================================
    # Profile Learning
    # =========================================================================
    
    def learn_profiles(self, sample_size: int = 100000) -> Dict[str, SignalProfile]:
        """
        Learn baseline statistics for each signal from the features table.
        
        Args:
            sample_size: Max rows to sample per signal
            
        Returns:
            Dict of signal_id -> SignalProfile
        """
        print("Learning baseline profiles from features table...")
        
        with self.engine.connect() as conn:
            # Get distinct signals
            signals = pd.read_sql(text("""
                SELECT DISTINCT source_dataset, entity_type
                FROM sentinelrisk.features
                WHERE source_dataset IS NOT NULL
            """), conn)
        
        for _, row in signals.iterrows():
            source = row["source_dataset"]
            entity = row["entity_type"]
            signal_id = self._signal_id(source, entity)
            
            print(f"  Learning {signal_id}...")
            
            # Sample data for this signal
            with self.engine.connect() as conn:
                df = pd.read_sql(text(f"""
                    SELECT * FROM sentinelrisk.features
                    WHERE source_dataset = :source AND entity_type = :entity
                    ORDER BY RANDOM()
                    LIMIT :limit
                """), conn, params={"source": source, "entity": entity, "limit": sample_size})
            
            if df.empty:
                continue
            
            profile = SignalProfile(
                signal_id=signal_id,
                source_dataset=source,
                entity_type=entity,
                sample_size=len(df)
            )
            
            # Compute feature statistics
            for feature in self.NUMERIC_FEATURES:
                if feature in df.columns:
                    values = df[feature].dropna()
                    if len(values) > 10:
                        profile.feature_stats[feature] = {
                            "mean": float(values.mean()),
                            "std": float(values.std()),
                            "p1": float(values.quantile(0.01)),
                            "p5": float(values.quantile(0.05)),
                            "p25": float(values.quantile(0.25)),
                            "p50": float(values.quantile(0.50)),
                            "p75": float(values.quantile(0.75)),
                            "p95": float(values.quantile(0.95)),
                            "p99": float(values.quantile(0.99)),
                            "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
                            "min": float(values.min()),
                            "max": float(values.max())
                        }
            
            # Compute temporal patterns
            if "hour_of_day" in df.columns:
                hourly = df.groupby(df["hour_of_day"].astype(int) % 24).size()
                profile.hourly_counts = np.zeros(24)
                for h, c in hourly.items():
                    if 0 <= h < 24:
                        profile.hourly_counts[h] = c
                profile.hourly_counts = profile.hourly_counts / profile.hourly_counts.sum()
            
            if "day_of_week" in df.columns:
                daily = df.groupby(df["day_of_week"].astype(int) % 7).size()
                profile.daily_counts = np.zeros(7)
                for d, c in daily.items():
                    if 0 <= d < 7:
                        profile.daily_counts[d] = c
                profile.daily_counts = profile.daily_counts / profile.daily_counts.sum()
            
            self.profiles[signal_id] = profile
        
        print(f"Learned {len(self.profiles)} signal profiles")
        self.save_profiles()
        return self.profiles
    
    def save_profiles(self, path: Optional[str] = None):
        """Save profiles to JSON file."""
        path = path or self.profiles_path
        data = {k: v.to_dict() for k, v in self.profiles.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved profiles to {path}")
    
    def load_profiles(self, path: Optional[str] = None):
        """Load profiles from JSON file."""
        path = path or self.profiles_path
        if not Path(path).exists():
            print(f"No profiles found at {path}")
            return
        with open(path, "r") as f:
            data = json.load(f)
        self.profiles = {k: SignalProfile.from_dict(v) for k, v in data.items()}
        print(f"Loaded {len(self.profiles)} profiles from {path}")
    
    # =========================================================================
    # Anomaly Scoring
    # =========================================================================
    
    def _zscore(self, value: float, mean: float, std: float) -> float:
        """Compute absolute z-score."""
        if std == 0 or pd.isna(value):
            return 0.0
        return abs((value - mean) / std)
    
    def _iqr_score(self, value: float, stats: Dict) -> float:
        """
        Compute IQR-based outlier score.
        Returns 0 if within normal range, higher if outlier.
        """
        if pd.isna(value):
            return 0.0
        
        q1 = stats.get("p25", 0)
        q3 = stats.get("p75", 0)
        iqr = stats.get("iqr", 1)
        
        if iqr == 0:
            return 0.0
        
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        if value < lower:
            return min(abs(value - lower) / iqr, 5.0)
        elif value > upper:
            return min(abs(value - upper) / iqr, 5.0)
        return 0.0
    
    def _percentile_score(self, value: float, stats: Dict) -> float:
        """
        Score based on how extreme the percentile is.
        Returns ~0 for median values, ~1 for extreme values.
        """
        if pd.isna(value):
            return 0.0
        
        p1, p99 = stats.get("p1", 0), stats.get("p99", 1)
        p5, p95 = stats.get("p5", 0), stats.get("p95", 1)
        
        if value <= p1 or value >= p99:
            return 1.0
        elif value <= p5 or value >= p95:
            return 0.7
        elif value <= stats.get("p25", 0) or value >= stats.get("p75", 1):
            return 0.3
        return 0.0
    
    def score_row(self, row: pd.Series, profile: SignalProfile) -> Dict[str, float]:
        """
        Compute anomaly scores for a single row.
        
        Returns dict with:
        - feature-level scores
        - aggregated score
        """
        scores = {}
        feature_scores = []
        
        for feature in self.NUMERIC_FEATURES:
            if feature not in profile.feature_stats:
                continue
            if feature not in row or pd.isna(row[feature]):
                continue
            
            stats = profile.feature_stats[feature]
            value = float(row[feature])
            
            # Compute multiple score types
            z = self._zscore(value, stats["mean"], stats["std"])
            iqr = self._iqr_score(value, stats)
            pct = self._percentile_score(value, stats)
            
            # Combined score (weighted average)
            combined = 0.4 * min(z / 5, 1) + 0.3 * min(iqr / 3, 1) + 0.3 * pct
            
            scores[f"{feature}_zscore"] = z
            scores[f"{feature}_combined"] = combined
            feature_scores.append(combined)
        
        # Aggregate scores
        if feature_scores:
            scores["baseline_score_mean"] = float(np.mean(feature_scores))
            scores["baseline_score_max"] = float(np.max(feature_scores))
            scores["baseline_score"] = float(np.mean(feature_scores) * 0.6 + np.max(feature_scores) * 0.4)
        else:
            scores["baseline_score"] = 0.0
        
        return scores
    
    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a batch of events.
        
        Args:
            df: DataFrame with features (must have source_dataset, entity_type)
            
        Returns:
            DataFrame with anomaly scores added
        """
        result_rows = []
        
        for idx, row in df.iterrows():
            signal_id = self._signal_id(
                row.get("source_dataset", "unknown"),
                row.get("entity_type", "unknown")
            )
            
            profile = self.profiles.get(signal_id)
            if profile is None:
                # No profile for this signal - score as 0
                scores = {"baseline_score": 0.0}
            else:
                scores = self.score_row(row, profile)
            
            result_rows.append(scores)
        
        scores_df = pd.DataFrame(result_rows, index=df.index)
        return pd.concat([df, scores_df], axis=1)
    
    def get_anomalies(
        self, 
        df: pd.DataFrame, 
        threshold: float = 0.5,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get anomalous events from a DataFrame.
        
        Args:
            df: DataFrame to score
            threshold: Minimum baseline_score to be considered anomaly
            top_n: If set, return only top N anomalies by score
            
        Returns:
            DataFrame of anomalies sorted by score
        """
        scored = self.score_batch(df)
        anomalies = scored[scored["baseline_score"] >= threshold].copy()
        anomalies = anomalies.sort_values("baseline_score", ascending=False)
        
        if top_n:
            anomalies = anomalies.head(top_n)
        
        return anomalies


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Detector")
    parser.add_argument("--learn", action="store_true", help="Learn profiles from data")
    parser.add_argument("--score", action="store_true", help="Score sample data")
    parser.add_argument("--sample-size", type=int, default=50000, help="Sample size for scoring")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly threshold")
    
    args = parser.parse_args()
    
    engine = create_engine("postgresql://vinaykota:12345678@localhost:5432/fintech_lab")
    detector = BaselineDetector(engine)
    
    if args.learn:
        detector.learn_profiles()
    else:
        detector.load_profiles()
    
    if args.score:
        print(f"\nScoring sample of {args.sample_size} rows...")
        with engine.connect() as conn:
            df = pd.read_sql(text(f"""
                SELECT * FROM sentinelrisk.features
                ORDER BY RANDOM()
                LIMIT {args.sample_size}
            """), conn)
        
        anomalies = detector.get_anomalies(df, threshold=args.threshold, top_n=20)
        
        print(f"\nTop anomalies (threshold={args.threshold}):")
        print("-" * 60)
        
        if len(anomalies) > 0:
            cols = ["entity_id", "source_dataset", "baseline_score"]
            cols += [c for c in anomalies.columns if c.endswith("_zscore")][:3]
            print(anomalies[cols].to_string())
        else:
            print("No anomalies found above threshold")
        
        # Score distribution
        scored = detector.score_batch(df)
        print(f"\nScore distribution:")
        print(scored["baseline_score"].describe())


if __name__ == "__main__":
    main()
