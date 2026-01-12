"""
Stream Simulator - Simulate Real-time Event Streaming

Replays events from raw_events in time order to simulate a live data stream.
Runs the anomaly detection pipeline on each event batch.

Usage:
    # Simulate streaming with 1 second between batches
    python stream_simulator.py --interval 1 --batch-size 100
"""

import time
import argparse
from datetime import datetime
from typing import Generator, Optional
import pandas as pd

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from anomaly_model import AnomalyModel, ModelConfig
from explainer import AnomalyExplainer
from business_mapper import BusinessMapper
from feedback_collector import FeedbackCollector


class StreamSimulator:
    """
    Simulates streaming events from raw_events table.
    
    Processes events through the full anomaly detection pipeline:
    1. Load batch of events
    2. Join with features (or compute on-the-fly)
    3. Run anomaly model
    4. Generate explanations for anomalies
    5. Map to business impact
    6. Log predictions for feedback
    """
    
    def __init__(self, engine: Engine):
        self.engine = engine
        
        # Initialize components
        print("Initializing pipeline components...")
        self.model = AnomalyModel(engine)
        self.model.load()
        
        self.explainer = AnomalyExplainer()
        self.mapper = BusinessMapper()
        self.feedback = FeedbackCollector(engine)
        
        # Streaming state
        self.last_ts = None
        self.total_processed = 0
        self.total_anomalies = 0
    
    def get_batch(
        self, 
        batch_size: int = 100, 
        source_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """Get a batch of features for scoring."""
        
        query = """
            SELECT * FROM sentinelrisk.features
            WHERE ts > :last_ts
        """
        
        params = {"last_ts": self.last_ts or "1900-01-01"}
        
        if source_filter:
            query += " AND source_dataset = :source"
            params["source"] = source_filter
        
        query += f" ORDER BY ts LIMIT {batch_size}"
        
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        
        if len(df) > 0:
            self.last_ts = df["ts"].max()
        
        return df
    
    def process_batch(self, df: pd.DataFrame) -> dict:
        """Process a batch through the pipeline."""
        if df.empty:
            return {"processed": 0, "anomalies": []}
        
        # Run model
        scored = self.model.predict(df)
        
        # Get anomalies
        anomalies = scored[scored["is_anomaly"] == 1]
        
        results = {
            "processed": len(df),
            "anomaly_count": len(anomalies),
            "anomalies": []
        }
        
        # Process each anomaly
        for _, row in anomalies.iterrows():
            # Get explanation
            exp = self.explainer.explain_row(row)
            
            # Get business impact
            impact = self.mapper.get_impact(
                row.get("source_dataset", ""),
                row.get("entity_type", "")
            )
            
            # Log prediction
            pred_id = self.feedback.log_prediction(
                entity_id=str(row.get("entity_id", "")),
                anomaly_score=float(row.get("anomaly_score", 0)),
                predicted_label=1,
                source_dataset=str(row.get("source_dataset", ""))
            )
            
            results["anomalies"].append({
                "prediction_id": pred_id,
                "entity_id": exp.entity_id,
                "source_dataset": exp.source_dataset,
                "score": exp.anomaly_score,
                "summary": exp.summary,
                "business_impact": impact.to_dict() if impact else None
            })
        
        self.total_processed += len(df)
        self.total_anomalies += len(anomalies)
        
        return results
    
    def run_stream(
        self, 
        batch_size: int = 100,
        interval: float = 1.0,
        max_batches: Optional[int] = None,
        source_filter: Optional[str] = None
    ):
        """Run the streaming simulation."""
        
        print(f"\n{'='*60}")
        print("STARTING STREAM SIMULATION")
        print(f"Batch size: {batch_size}, Interval: {interval}s")
        print(f"{'='*60}\n")
        
        batch_num = 0
        
        while True:
            batch_num += 1
            
            if max_batches and batch_num > max_batches:
                break
            
            # Get and process batch
            df = self.get_batch(batch_size, source_filter)
            
            if df.empty:
                print("No more events to process.")
                break
            
            results = self.process_batch(df)
            
            # Print results
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] Batch {batch_num}: "
                  f"processed={results['processed']}, "
                  f"anomalies={results['anomaly_count']}")
            
            # Print anomaly details
            for anom in results["anomalies"]:
                impact = anom.get("business_impact", {})
                risk = impact.get("risk_level", "?") if impact else "?"
                print(f"    ⚠️ {anom['entity_id']} ({anom['source_dataset']}): "
                      f"{anom['score']:.2%} | {risk.upper()}")
                print(f"       {anom['summary'][:60]}...")
            
            # Wait for next batch
            time.sleep(interval)
        
        print(f"\n{'='*60}")
        print("STREAM COMPLETE")
        print(f"Total processed: {self.total_processed}")
        print(f"Total anomalies: {self.total_anomalies}")
        print(f"Anomaly rate: {self.total_anomalies/max(self.total_processed,1):.2%}")
        print(f"{'='*60}")


# CLI
def main():
    parser = argparse.ArgumentParser(description="Stream Simulator")
    parser.add_argument("--batch-size", type=int, default=100, help="Events per batch")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between batches")
    parser.add_argument("--max-batches", type=int, default=10, help="Max batches (0=unlimited)")
    parser.add_argument("--source", type=str, help="Filter by source_dataset")
    
    args = parser.parse_args()
    
    engine = create_engine("postgresql://vinaykota:12345678@localhost:5432/fintech_lab")
    simulator = StreamSimulator(engine)
    
    simulator.run_stream(
        batch_size=args.batch_size,
        interval=args.interval,
        max_batches=args.max_batches or None,
        source_filter=args.source
    )


if __name__ == "__main__":
    main()
