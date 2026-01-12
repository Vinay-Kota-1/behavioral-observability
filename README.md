# SentinelRisk: Intelligent Anomaly Detection Platform

> *From noise to signal: Catching the needles in haystacks of operational data*

![Dashboard](./dashboard_preview.png)

---

## 🎯 The Problem

Modern organizations drown in signals:
- **Millions of events daily** across security logs, transactions, and infrastructure metrics
- **High false positive rates** from rule-based systems → alert fatigue
- **No business context** → analysts don't know *why* something matters or *what to do*
- **Siloed detection** → fraud team, security team, SRE team all build separate systems

**The real challenge isn't detection—it's *actionable insight*:**

| What teams ask | What current tools provide |
|----------------|---------------------------|
| "What changed?" | Raw alerts |
| "How bad is it?" | No severity context |
| "Is this recurring?" | No pattern memory |
| "Can I trust this signal?" | No confidence score |
| "Is this worth my attention?" | Everything looks equal |

---

## 💡 The "Aha" Moment

**Insight**: The best anomaly detection doesn't just flag outliers—it **explains them in business terms**.

```
Traditional approach:           SentinelRisk approach:
─────────────────────           ─────────────────────
"Event count: 289"      →       "Account takeover attempt detected.
                                 20 logins in 5 minutes from user_3576664.
                                 Risk: HIGH | Est. Cost: $500 | SLA: 15 min
                                 Action: Lock account, notify user"
```

**The magic is in the judgment layer**—connecting statistical anomalies to:
1. **Why** it's anomalous (explainability)
2. **What** it means for the business (impact mapping)  
3. **How** to respond (escalation paths)
4. **Whether** the model was right (feedback loop)

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                │
│   CERT (Insider Threat) │ IEEE CIS (Fraud) │ NAB (Metrics) │ ...  │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                              │
│   Point-in-time features via SQL window functions                  │
│   event_count_24h │ time_since_last │ metric_std │ z-scores        │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                    ADVERSARIAL INJECTION (Training)                 │
│   Synthetic attack patterns → Labeled anomalies → Model learns     │
│   login_burst │ velocity_spike │ off_hours │ high_value_sudden     │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                    ML DETECTION ENGINE                              │
│   XGBoost Classifier (ROC AUC: 0.96)                               │
│   Outputs: probability score + is_anomaly flag                     │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                    JUDGMENT LAYER ⭐                                │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│   │  Explainer   │  │ Business     │  │  Feedback    │            │
│   │  "Why?"      │  │ Mapper       │  │  Collector   │            │
│   │              │  │ "So what?"   │  │  "Was I      │            │
│   │  Top features│  │ Risk, cost,  │  │   right?"    │            │
│   │  + z-scores  │  │ action, SLA  │  │              │            │
│   └──────────────┘  └──────────────┘  └──────────────┘            │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                    LIVE DASHBOARD (Streamlit)                       │
│   Real-time anomaly feed │ Score distributions │ Business impact   │
│   Feedback interface │ Model performance tracking                  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📊 V1: Current State (MVP)

### What's Working

| Component | Status | Description |
|-----------|--------|-------------|
| **Feature Store** | ✅ | 17 point-in-time features via SQL |
| **Adversarial Injection** | ✅ | 7 attack scenarios with business rules |
| **XGBoost Model** | ✅ | ROC AUC: 0.96, F1: 0.69 |
| **Explainer** | ✅ | Feature importance + z-score explanations |
| **Business Mapper** | ✅ | Risk levels, costs, escalation paths |
| **Feedback Loop** | ✅ | Prediction tracking, accuracy stats |
| **Streamlit Dashboard** | ✅ | Live visualization across 4 data sources |

### Key Metrics

```
Model Performance:
├── ROC AUC:    0.957  (excellent discrimination)
├── Precision:  0.621  (62% true positive rate)
├── Recall:     0.784  (catches 78% of anomalies)
└── F1 Score:   0.693  (good balance)

Data Scale:
├── CERT:        915K events (insider threat)
├── CREDITCARD:  285K events (fraud detection)
├── IEEE_CIS:    592K events (transaction fraud)
└── NAB:          70K events (infrastructure metrics)
```

### Quick Start

```bash
# 1. Train the model
python anomaly_model.py --train --sample-size 50000

# 2. Run the dashboard
streamlit run app.py

# 3. Or run stream simulation
python stream_simulator.py --batch-size 100 --max-batches 10
```

---

## 🚀 V2+: Future Roadmap

### Phase 2: Intelligence Layer
- [ ] **Multi-model ensemble** — Combine XGBoost, Isolation Forest, Autoencoders
- [ ] **Automatic retraining** — Learn from feedback loop
- [ ] **Drift detection** — Alert when data distribution shifts
- [ ] **Temporal patterns** — Detect seasonality-aware anomalies

### Phase 3: Operational Excellence
- [ ] **Real-time streaming** — Kafka/Flink integration
- [ ] **Alerting** — PagerDuty, Slack, email notifications
- [ ] **SOAR integration** — Trigger automated response playbooks
- [ ] **Multi-tenant** — Support for multiple teams/customers

### Phase 4: Advanced Analytics
- [ ] **Attack chain detection** — Link related anomalies
- [ ] **Entity risk scoring** — Aggregate risk at user/device level
- [ ] **What-if analysis** — "What would happen if..."
- [ ] **Natural language queries** — "Show me suspicious activity last week"

### Phase 5: Scale & Deploy
- [ ] **Cloud deployment** — AWS/GCP/Azure templates
- [ ] **Kubernetes operator** — Self-managing deployment
- [ ] **API layer** — REST/GraphQL for integrations
- [ ] **Role-based access** — Analyst vs. Admin views

---

## 📁 Project Structure

```
sentinelrisk/
├── config/
│   ├── feature_config.yaml           # Feature definitions
│   └── adversarial_config.yaml        # Attack scenarios + business rules
├── data/
│   ├── raw/                           # Raw data files
│   ├── samples/                       # Sample datasets
│   └── models/                        # Trained model files
│       ├── anomaly_xgb.pkl
│       └── anomaly_xgb_metadata.json
├── docs/
│   ├── assets/                        # Screenshots, diagrams, recordings
│   │   ├── main_dashboard.png
│   │   ├── distributions_tab.png
│   │   └── dashboard_working.webp
│   └── implementation/                # Technical docs
├── notebooks/                         # Jupyter notebooks
├── src/
│   ├── app.py                         # Streamlit dashboard
│   ├── ingestion/
│   │   ├── ingest_postgres.py         # Data loading to Postgres
│   │   ├── adversarial_injector.py    # Synthetic anomaly generation
│   │   └── stream_simulator.py        # Real-time simulation
│   ├── features/
│   │   ├── feature_builder.py         # Point-in-time features
│   │   └── baseline_detector.py       # Statistical detection
│   ├── models/
│   │   ├── anomaly_model.py           # XGBoost classifier
│   │   ├── explainer.py               # Feature-based explanations
│   │   ├── business_mapper.py         # Business impact mapping
│   │   └── feedback_collector.py      # Prediction tracking
│   └── utils/
│       └── paths.py                   # Shared path utilities
├── tests/                             # Unit and integration tests
└── README.md
```

---

## 🤝 Contributing

1. Fork the repository
2. Add new scenarios to `adversarial_config.yaml`
3. Run `python adversarial_injector.py --scenario your_scenario`
4. Regenerate features: `python feature_builder.py --batch`
5. Retrain: `python anomaly_model.py --train`
6. Test in dashboard: `streamlit run app.py`

---

## 📜 License

MIT License - Build something great with it.

---

*Built with ❤️ for security, fraud, and SRE teams who are tired of alert fatigue.*
