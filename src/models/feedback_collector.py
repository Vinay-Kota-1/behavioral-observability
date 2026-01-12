"""
Feedback Collector - Track Predictions and User Corrections

Stores predictions with timestamps for tracking model performance over time.
Allows users to mark predictions as correct/incorrect for model improvement.

Usage:
    from feedback_collector import FeedbackCollector
    
    collector = FeedbackCollector()
    collector.log_prediction(entity_id, score, predicted, source)
    collector.record_feedback(prediction_id, actual_label, notes)
"""

import pandas as pd
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@dataclass
class Prediction:
    """A single model prediction."""
    prediction_id: str
    timestamp: str
    entity_id: str
    source_dataset: str
    anomaly_score: float
    predicted_label: int  # 0 or 1
    feedback_received: bool
    actual_label: Optional[int]
    is_correct: Optional[bool]
    notes: Optional[str]


class FeedbackCollector:
    """
    Collects and stores predictions and user feedback.
    
    Uses both a local JSON file and Postgres table for storage.
    """
    
    def __init__(
        self, 
        engine: Optional[Engine] = None,
        local_path: str = "feedback/predictions.json"
    ):
        self.engine = engine
        self.local_path = Path(local_path)
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.predictions: List[Prediction] = []
        self._load_local()
        
        if self.engine:
            self._ensure_table()
    
    def _load_local(self):
        """Load predictions from local JSON file."""
        if self.local_path.exists():
            with open(self.local_path) as f:
                data = json.load(f)
            self.predictions = [Prediction(**p) for p in data]
            print(f"Loaded {len(self.predictions)} predictions from {self.local_path}")
    
    def _save_local(self):
        """Save predictions to local JSON file."""
        data = [asdict(p) for p in self.predictions]
        with open(self.local_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def _ensure_table(self):
        """Create Postgres table if it doesn't exist."""
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sentinelrisk.prediction_feedback (
                    prediction_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    entity_id TEXT,
                    source_dataset TEXT,
                    anomaly_score NUMERIC,
                    predicted_label SMALLINT,
                    feedback_received BOOLEAN DEFAULT FALSE,
                    actual_label SMALLINT,
                    is_correct BOOLEAN,
                    notes TEXT
                )
            """))
        print("Feedback table ready")
    
    def log_prediction(
        self,
        entity_id: str,
        anomaly_score: float,
        predicted_label: int,
        source_dataset: str = "unknown"
    ) -> str:
        """
        Log a new prediction.
        
        Returns:
            prediction_id for later feedback
        """
        prediction_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        pred = Prediction(
            prediction_id=prediction_id,
            timestamp=timestamp,
            entity_id=entity_id,
            source_dataset=source_dataset,
            anomaly_score=anomaly_score,
            predicted_label=predicted_label,
            feedback_received=False,
            actual_label=None,
            is_correct=None,
            notes=None
        )
        
        self.predictions.append(pred)
        self._save_local()
        
        # Also save to Postgres if available
        if self.engine:
            with self.engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO sentinelrisk.prediction_feedback 
                    (prediction_id, entity_id, source_dataset, anomaly_score, predicted_label)
                    VALUES (:pid, :eid, :src, :score, :label)
                    ON CONFLICT (prediction_id) DO NOTHING
                """), {
                    "pid": prediction_id,
                    "eid": entity_id,
                    "src": source_dataset,
                    "score": anomaly_score,
                    "label": predicted_label
                })
        
        return prediction_id
    
    def log_batch(self, df: pd.DataFrame) -> List[str]:
        """Log a batch of predictions from a DataFrame."""
        ids = []
        for _, row in df.iterrows():
            pid = self.log_prediction(
                entity_id=str(row.get("entity_id", "unknown")),
                anomaly_score=float(row.get("anomaly_score", 0)),
                predicted_label=int(row.get("is_anomaly", 0)),
                source_dataset=str(row.get("source_dataset", "unknown"))
            )
            ids.append(pid)
        return ids
    
    def record_feedback(
        self,
        prediction_id: str,
        actual_label: int,
        notes: Optional[str] = None
    ) -> bool:
        """
        Record user feedback on a prediction.
        
        Args:
            prediction_id: ID from log_prediction
            actual_label: 1 if actually anomaly, 0 if normal
            notes: Optional user notes
            
        Returns:
            True if feedback recorded successfully
        """
        # Find in local list
        for pred in self.predictions:
            if pred.prediction_id == prediction_id:
                pred.feedback_received = True
                pred.actual_label = actual_label
                pred.is_correct = (pred.predicted_label == actual_label)
                pred.notes = notes
                break
        else:
            print(f"Prediction {prediction_id} not found")
            return False
        
        self._save_local()
        
        # Update Postgres
        if self.engine:
            with self.engine.begin() as conn:
                conn.execute(text("""
                    UPDATE sentinelrisk.prediction_feedback
                    SET feedback_received = TRUE,
                        actual_label = :actual,
                        is_correct = (predicted_label = :actual),
                        notes = :notes
                    WHERE prediction_id = :pid
                """), {
                    "pid": prediction_id,
                    "actual": actual_label,
                    "notes": notes
                })
        
        return True
    
    def get_accuracy_stats(self) -> Dict[str, Any]:
        """Calculate accuracy statistics from feedback."""
        with_feedback = [p for p in self.predictions if p.feedback_received]
        
        if not with_feedback:
            return {"total": 0, "with_feedback": 0}
        
        correct = sum(1 for p in with_feedback if p.is_correct)
        
        # Breakdown by prediction type
        true_positives = sum(1 for p in with_feedback if p.predicted_label == 1 and p.actual_label == 1)
        false_positives = sum(1 for p in with_feedback if p.predicted_label == 1 and p.actual_label == 0)
        true_negatives = sum(1 for p in with_feedback if p.predicted_label == 0 and p.actual_label == 0)
        false_negatives = sum(1 for p in with_feedback if p.predicted_label == 0 and p.actual_label == 1)
        
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        
        return {
            "total_predictions": len(self.predictions),
            "with_feedback": len(with_feedback),
            "accuracy": correct / len(with_feedback),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall
        }
    
    def get_recent_predictions(self, n: int = 20) -> List[Prediction]:
        """Get most recent predictions."""
        return sorted(self.predictions, key=lambda p: p.timestamp, reverse=True)[:n]
    
    def get_pending_feedback(self) -> List[Prediction]:
        """Get predictions awaiting feedback (predicted as anomaly)."""
        return [p for p in self.predictions 
                if p.predicted_label == 1 and not p.feedback_received]


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Feedback Collector")
    parser.add_argument("--stats", action="store_true", help="Show accuracy stats")
    parser.add_argument("--pending", action="store_true", help="Show pending feedback")
    parser.add_argument("--feedback", type=str, help="Record feedback: ID,label[,notes]")
    
    args = parser.parse_args()
    
    engine = create_engine("postgresql://vinaykota:12345678@localhost:5432/fintech_lab")
    collector = FeedbackCollector(engine)
    
    if args.stats:
        stats = collector.get_accuracy_stats()
        print("\nPrediction Statistics:")
        print(json.dumps(stats, indent=2))
    
    elif args.pending:
        pending = collector.get_pending_feedback()
        print(f"\nPending Feedback ({len(pending)} predictions):")
        for p in pending[:10]:
            print(f"  {p.prediction_id}: {p.entity_id} (score: {p.anomaly_score:.2%})")
    
    elif args.feedback:
        parts = args.feedback.split(",")
        pid = parts[0]
        label = int(parts[1])
        notes = parts[2] if len(parts) > 2 else None
        
        success = collector.record_feedback(pid, label, notes)
        print(f"Feedback recorded: {success}")
    
    else:
        print(f"Total predictions: {len(collector.predictions)}")
        parser.print_help()
