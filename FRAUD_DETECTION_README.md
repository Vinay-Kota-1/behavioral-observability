# Fraud Detection System

A comprehensive fraud detection system based on the architecture diagram, featuring dynamic routing, state space models, and complete pipeline orchestration.

## System Architecture

The system follows a complete data pipeline:

1. **Data Ingestion** → Postgres (System of Record)
2. **Feature Building** → Point-in-time correct features
3. **Model Router** → Dynamic model selection
4. **Models** → Supervised, Unsupervised, and Time Series models
5. **Score Ensembler** → Combines scores from multiple models
6. **Explainability** → SHAP explanations and evidence packs
7. **Alerting** → Deduplication and budget control
8. **Monitoring** → Drift metrics and alert volume tracking

## Components

### 1. Feature Builder (`feature_builder.py`)
- Builds point-in-time correct features for users, merchants, and API clients
- Prevents data leakage by using only historical data available at event time
- Supports multiple time windows (1h, 6h, 24h, 1 week)

### 2. Model Router (`model_router.py`)
- Dynamically selects models based on:
  - Label availability (supervised vs unsupervised)
  - Historical data availability (time series models)
  - Alert budget constraints
- Routes events to appropriate models

### 3. State Space Models (`timeseries.py`)
- **Kalman Filter**: Adaptive state space model for time series forecasting
- **State Space (Structural Time Series)**: Unobserved components model with trend and seasonality
- **EWMA**: Exponentially Weighted Moving Average
- **STL**: Seasonal-Trend decomposition using Loess

### 4. Score Ensembler (`score_ensembler.py`)
- Combines scores from multiple models
- Methods: weighted average, max, min, median, mean
- Configurable weights per model type

### 5. Alerting System (`alerting.py`)
- **Deduplication Window**: Prevents duplicate alerts within a time window
- **Daily Alert Budget**: Limits alerts per day to prevent fatigue
- **Severity Classification**: Low, Medium, High, Critical

### 6. Monitoring System (`monitoring.py`)
- **Alert Volume Tracking**: Monitors alert generation rates
- **Drift Detection**: 
  - Population Stability Index (PSI)
  - Jensen-Shannon Divergence
- Tracks model score distributions over time

### 7. Explainability (`explainability.py`)
- **SHAP Values**: For supervised models
- **Evidence Packs**: Features, timeline, model information
- Feature importance ranking

### 8. Fraud Detector (`fraud_detector.py`)
- Main pipeline orchestrator
- Integrates all components
- End-to-end processing

## Installation

### Prerequisites
- Python 3.8+
- PostgreSQL database
- Required Python packages (see `requirements_anomaly.txt`)

### Install Dependencies

```bash
pip install -r requirements_anomaly.txt

# Additional dependencies for state space models
pip install pykalman  # For Kalman Filter
pip install shap      # For explainability (optional)
pip install xgboost   # For supervised models (optional)
pip install pyyaml    # For configuration
```

## Configuration

Configuration is stored in `config/default.yaml`. Key parameters:

- **Feature Windows**: Time windows for feature calculation
- **Router Settings**: Model selection criteria
- **Thresholds**: Model-specific thresholds
- **Alerting**: Dedup window and daily budget
- **Monitoring**: Drift detection thresholds

## Usage

### Step 1: Database Setup

Ensure your PostgreSQL database has the required schema:

```sql
CREATE SCHEMA IF NOT EXISTS sentinelrisk;

-- Tables will be created automatically by the system
-- raw_events, labels_outcomes, alerts, monitoring_metrics
```

### Step 2: Load Data

Use the existing data loaders to populate the database:

```python
from ingest_postgres import make_engine, insert_raw_events, insert_labels

engine = make_engine("postgresql://user:password@localhost:5432/dbname")

# Load your events
events_df = ...  # Your events DataFrame
insert_raw_events(engine, events_df)

# Load labels (if available)
labels_df = ...  # Your labels DataFrame
insert_labels(engine, labels_df)
```

### Step 3: Initialize Fraud Detector

```python
from anomaly_detection.fraud_detector import FraudDetector

# Initialize with database URL and config
detector = FraudDetector(
    db_url="postgresql://user:password@localhost:5432/dbname",
    config_path="config/default.yaml"
)
```

### Step 4: Run Pipeline

#### Process Events

```python
from datetime import datetime, timedelta

# Process events from the last 24 hours
end_time = datetime.now()
start_time = end_time - timedelta(hours=24)

results = detector.run_pipeline(
    start_time=start_time,
    end_time=end_time,
    limit=1000,  # Process up to 1000 events
    generate_alerts=True,
    generate_explanations=True,
    monitor=True
)

print(f"Processed {results['summary']['n_events_processed']} events")
print(f"Created {results['summary']['n_alerts_created']} alerts")
```

#### Process Specific Events

```python
# Load events from database
events_df = detector.load_events(
    start_time=start_time,
    end_time=end_time,
    limit=100
)

# Process with custom options
results = detector.process_events(
    events_df,
    generate_alerts=True,
    generate_explanations=True
)
```

### Step 5: Access Results

```python
# Get processing results
results_df = results['processing']['results_df']

# Get alerts
alerts_df = results['processing']['alerts_df']

# View alerts
print(alerts_df[alerts_df['alert_created']][
    ['event_id', 'entity_id', 'score', 'severity', 'model_used']
])

# Get monitoring metrics
monitoring = results['monitoring']
print(f"Alert volume: {monitoring['alert_volume']}")
print(f"Drift metrics: {monitoring['drift_metrics']}")
```

### Step 6: Query Alerts

```python
from datetime import datetime, timedelta

# Get alerts from the last 24 hours
start_time = datetime.now() - timedelta(hours=24)

alerts = detector.alerting_system.get_alerts(
    start_time=start_time,
    severity='high',  # Filter by severity
    limit=100
)

print(alerts[['alert_id', 'entity_id', 'score', 'severity', 'timestamp']])
```

### Step 7: Monitor System

```python
# Run monitoring
monitoring_results = detector.monitor_system()

print("Alert Volume:", monitoring_results['alert_volume'])
print("Drift Metrics:", monitoring_results['drift_metrics'])

# Get historical metrics
metrics_df = detector.monitoring_system.get_metrics(
    metric_type='drift',
    limit=100
)
```

## Advanced Usage

### Custom Feature Building

```python
from anomaly_detection.feature_builder import FeatureBuilder

feature_builder = FeatureBuilder(engine, schema="sentinelrisk")

# Build features for a specific user at a point in time
features = feature_builder.build_user_features(
    user_id="user_123",
    timestamp=datetime.now(),
    windows=[1, 6, 24, 168]  # 1h, 6h, 24h, 1 week
)
```

### Custom Model Routing

```python
from anomaly_detection.model_router import ModelRouter

router = ModelRouter(engine, schema="sentinelrisk", config={
    'has_labels': True,
    'min_history_days': 30,
    'alert_budget_per_day': 100
})

# Route a batch of events
routing_df = router.route_batch(events_df)
```

### Score Ensembling

```python
from anomaly_detection.score_ensembler import ScoreEnsembler

ensembler = ScoreEnsembler(
    method="weighted_average",
    weights={
        'supervised': 0.4,
        'unsupervised': 0.3,
        'time_series': 0.3
    }
)

# Combine scores from multiple models
scores = {
    'isolation_forest': np.array([0.7, 0.8, 0.6]),
    'kalman_filter': np.array([0.6, 0.7, 0.5]),
    'state_space': np.array([0.8, 0.9, 0.7])
}

ensemble_result = ensembler.ensemble_with_metadata(scores)
print(ensemble_result['scores'])
print(ensemble_result['metadata'])
```

### Explainability

```python
from anomaly_detection.explainability import ExplainabilitySystem

explainer = ExplainabilitySystem()

# Create evidence pack
evidence = explainer.create_evidence_pack(
    event_id="event_123",
    entity_type="user",
    entity_id="user_123",
    timestamp=datetime.now(),
    score=0.85,
    model_used="isolation_forest",
    features={'feature1': 1.5, 'feature2': 2.3}
)

print(evidence['explanation'])
print(evidence['top_features'])
```

## Database Schema

The system uses the following tables in the `sentinelrisk` schema:

### `raw_events`
- Event data with columns: event_id, ts, entity_type, entity_id, event_type, status, amount, etc.

### `labels_outcomes`
- Fraud labels with columns: ts_start, ts_end, label_scope, scope_id, label_type, label, severity

### `alerts`
- Generated alerts with columns: alert_id, event_id, entity_type, entity_id, timestamp, score, severity, model_used, features, evidence_pack

### `monitoring_metrics`
- Monitoring metrics with columns: metric_id, metric_type, metric_name, metric_value, timestamp, metadata

## Configuration Reference

### Feature Windows
```yaml
feature_windows:
  user: [1, 6, 24, 168]      # hours
  merchant: [1, 6, 24]
  api_client: [1, 6, 24]
```

### Router Configuration
```yaml
router:
  has_labels: true
  min_history_days: 30
  alert_budget_per_day: 100
  weights:
    supervised: 0.4
    unsupervised: 0.3
    time_series: 0.3
```

### Alerting Configuration
```yaml
alerting:
  dedup_window_hours: 24
  daily_alert_budget: 100
  severity_thresholds:
    low: 0.3
    medium: 0.5
    high: 0.7
    critical: 0.9
```

## Model Types

### Supervised Models
- **XGBoost**: Gradient boosting (requires labels)

### Unsupervised Models
- **Isolation Forest**: Tree-based anomaly detection
- **Local Outlier Factor**: Density-based
- **One-Class SVM**: Support vector machine
- **Autoencoder**: Neural network reconstruction error

### Time Series Models
- **Kalman Filter**: State space model with adaptive filtering
- **State Space (Structural)**: Trend and seasonal decomposition
- **EWMA**: Exponentially weighted moving average
- **STL**: Seasonal-Trend decomposition using Loess
- **ARIMA**: Auto-regressive integrated moving average

## Monitoring and Drift Detection

The system tracks:

1. **Alert Volume**: Total alerts, by severity, by entity type
2. **PSI (Population Stability Index)**: 
   - < 0.1: No significant change
   - 0.1-0.2: Minor change
   - > 0.2: Significant change
3. **Jensen-Shannon Divergence**: Distribution similarity (0 = identical, 1 = completely different)

## Troubleshooting

### Database Connection Issues
- Verify PostgreSQL is running
- Check connection URL format: `postgresql://user:password@host:port/dbname`
- Ensure schema `sentinelrisk` exists

### Missing Dependencies
```bash
pip install pykalman shap xgboost pyyaml
```

### Model Training
- Models need to be trained before use
- Use the existing framework classes to train models on historical data
- Save trained models for reuse

## Next Steps

1. **Train Models**: Train models on historical data
2. **Tune Thresholds**: Adjust thresholds in config based on performance
3. **Customize Features**: Add domain-specific features
4. **Add Models**: Integrate additional models as needed
5. **Deploy**: Set up scheduled jobs for continuous processing

## Example Workflow

```python
# 1. Initialize
detector = FraudDetector(
    db_url="postgresql://user:pass@localhost:5432/db",
    config_path="config/default.yaml"
)

# 2. Process recent events
results = detector.run_pipeline(
    start_time=datetime.now() - timedelta(hours=24),
    end_time=datetime.now(),
    limit=1000
)

# 3. Review alerts
alerts = detector.alerting_system.get_alerts(
    start_time=datetime.now() - timedelta(hours=1),
    severity='high'
)

# 4. Monitor system
monitoring = detector.monitor_system()
print(f"PSI: {monitoring['drift_metrics']['psi']}")
print(f"Alert Volume: {monitoring['alert_volume']['total_alerts']}")
```

## Support

For issues or questions, refer to the code documentation in each module.

