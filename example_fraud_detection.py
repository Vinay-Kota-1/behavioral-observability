"""
Example: Fraud Detection Pipeline

This script demonstrates how to use the fraud detection system.
"""

from datetime import datetime, timedelta
from anomaly_detection.fraud_detector import FraudDetector

# Database configuration
DB_URL = "postgresql://vinaykota:12345678@localhost:5432/fintech_lab"
CONFIG_PATH = "config/default.yaml"

def main():
    """Run fraud detection pipeline example."""
    
    print("=" * 60)
    print("Fraud Detection System - Example")
    print("=" * 60)
    
    # Step 1: Initialize Fraud Detector
    print("\n1. Initializing Fraud Detector...")
    detector = FraudDetector(
        db_url=DB_URL,
        config_path=CONFIG_PATH
    )
    print("✓ Fraud Detector initialized")
    
    # Step 2: Define time range
    print("\n2. Setting up time range...")
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    print(f"  Processing events from {start_time} to {end_time}")
    
    # Step 3: Run pipeline
    print("\n3. Running fraud detection pipeline...")
    try:
        results = detector.run_pipeline(
            start_time=start_time,
            end_time=end_time,
            limit=100,  # Process up to 100 events for demo
            generate_alerts=True,
            generate_explanations=True,
            monitor=True
        )
        
        # Step 4: Display results
        print("\n4. Pipeline Results:")
        print("-" * 60)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        summary = results.get('summary', {})
        print(f"  Events Processed: {summary.get('n_events_processed', 0)}")
        print(f"  Alerts Created: {summary.get('n_alerts_created', 0)}")
        print(f"  Processing Time: {summary.get('processing_time_seconds', 0):.2f} seconds")
        
        # Step 5: Display alerts
        print("\n5. Recent Alerts:")
        print("-" * 60)
        
        alerts = detector.alerting_system.get_alerts(
            start_time=start_time,
            limit=10
        )
        
        if not alerts.empty:
            print(alerts[['entity_id', 'score', 'severity', 'model_used', 'timestamp']].to_string())
        else:
            print("  No alerts found")
        
        # Step 6: Display monitoring metrics
        print("\n6. Monitoring Metrics:")
        print("-" * 60)
        
        monitoring = results.get('monitoring', {})
        if monitoring:
            alert_volume = monitoring.get('alert_volume', {})
            print(f"  Total Alerts (24h): {alert_volume.get('total_alerts', 0)}")
            print(f"  Average Score: {alert_volume.get('avg_score', 0):.3f}")
            
            drift_metrics = monitoring.get('drift_metrics', {})
            if 'psi' in drift_metrics:
                print(f"  PSI (Drift): {drift_metrics.get('psi', 0):.3f}")
                print(f"  JS Divergence: {drift_metrics.get('js_divergence', 0):.3f}")
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

