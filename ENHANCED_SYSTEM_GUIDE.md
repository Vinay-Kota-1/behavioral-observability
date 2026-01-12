# Enhanced Fraud Detection System - Complete Guide

## Philosophy: From Prediction to Reasoning

This system represents a fundamental shift from "AI that predicts" to **"AI that reasons"**. 

### Core Principle

> **This system represents reality as entity-centric, time-indexed state so that any form of intelligence — statistical, machine learning, or agentic — can reason over it consistently, explainably, and robustly.**

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Reality Representation Layer              │
│  (Entity-centric, time-indexed state for reasoning)         │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Event Stream │   │ State Store │   │ Graph Repr   │
│ (raw_events) │   │ (snapshots) │   │ (relations)  │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Features    │   │ Transitions  │   │ NL State     │
│ (point-time) │   │ (what changed)│  │ (LLM-ready)  │
└──────────────┘   └──────────────┘   └──────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │   Intelligence Layer               │
        │   (Models, Agents, LLMs)           │
        └───────────────────────────────────┘
```

## Component Overview

### 1. Entity State Store (`entity_state_store.py`)

**Purpose**: Materialize entity state snapshots over time

**What it does**:
- Creates hourly snapshots of entity state
- Stores "Given everything that happened so far, what is the state now?"
- Enables fast lookups (no on-demand computation)
- Tracks state evolution over time

**Key Methods**:
```python
# Create a snapshot
snapshot = state_store.create_snapshot('user', 'user_123', datetime.now())

# Get entity memory (what this entity "remembers")
memory = state_store.get_entity_memory('user', 'user_123', lookback_hours=168)

# Get state history
history = state_store.get_state_history('user', 'user_123', start_time, end_time)
```

**Database Table**: `entity_state_snapshots`
- Stores: entity_type, entity_id, snapshot_ts, state_features (JSONB)
- Indexed for fast temporal queries

### 2. Scenario Injector (`scenario_injector.py`)

**Purpose**: Simulate intelligent adversaries for robustness testing

**What it does**:
- Injects realistic attack patterns (ATO, API abuse, merchant abuse)
- Creates labeled counterfactual worlds
- Tests model robustness ("when does it fail, and why?")

**Key Methods**:
```python
# Inject Account Takeover scenario
scenario = injector.inject_ato_scenario(
    user_id='user_123',
    start_time=datetime.now(),
    duration_hours=2,
    intensity='medium'
)

# Inject API abuse
scenario = injector.inject_api_abuse_scenario(
    api_client_id='client_456',
    start_time=datetime.now(),
    duration_hours=1,
    intensity='high'
)
```

**Database Table**: `scenario_runs`
- Tracks: scenario_type, entity, start/end time, events injected, labels created

### 3. State Transitions (`state_transitions.py`)

**Purpose**: Track what changed between snapshots

**What it does**:
- Computes: "Given state at T-1, what changed to reach state at T?"
- Detects significant changes (spikes, drops, multi-feature changes)
- Generates human-readable explanations

**Key Methods**:
```python
# Compute transition
transition = tracker.compute_transition(
    'user', 'user_123',
    from_timestamp=datetime.now() - timedelta(hours=1),
    to_timestamp=datetime.now()
)

# Detect anomalous transitions
anomalies = tracker.detect_anomalous_transitions('user', 'user_123')

# Explain change in natural language
explanation = tracker.explain_change('user', 'user_123', from_ts, to_ts)
```

**Database Table**: `state_transitions`
- Stores: from/to snapshots, transition_delta (JSONB), change_magnitude, change_type

### 4. Graph Builder (`graph_builder.py`)

**Purpose**: Build explicit entity relationship graphs

**What it does**:
- Creates graph where nodes = entities, edges = relationships
- Enables graph neural networks and relationship reasoning
- Tracks interaction patterns

**Key Methods**:
```python
# Build graph from events
graph_data = builder.build_graph_from_events(
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now()
)

# Get entity neighbors
neighbors = builder.get_entity_neighbors('user', 'user_123')

# Get entity community (connected entities)
community = builder.get_entity_community('user', 'user_123', max_hops=2)
```

**Database Tables**: 
- `graph_edges`: from/to entities, relationship_type, interaction_count, edge_weight
- `graph_nodes`: entity, node_degree, total_interactions

### 5. Natural Language State (`natural_language_state.py`)

**Purpose**: Convert features to human-readable statements

**What it does**:
- Transforms: `{user_24h_event_count: 150}` → "User has 150 events in the last 1 day"
- Enables LLM reasoning over state directly
- Generates explanations for anomalies

**Key Methods**:
```python
# Describe entity state
description = nl_state.describe_state(
    'user', 'user_123',
    state_features={'user_24h_event_count': 150, 'user_24h_success_rate': 0.95},
    baseline_features={'user_24h_event_count': 50, 'user_24h_success_rate': 0.98}
)

# Describe change
change_desc = nl_state.describe_change(from_state, to_state, 'user', 'user_123')

# Generate anomaly explanation
explanation = nl_state.generate_anomaly_explanation(
    'user', 'user_123', state_features, anomaly_score=0.85
)
```

## Complete Workflow Example

```python
from anomaly_detection.fraud_detector import FraudDetector
from datetime import datetime, timedelta

# Initialize system
detector = FraudDetector(
    db_url="postgresql://user:pass@localhost:5432/db",
    config_path="config/default.yaml"
)

# 1. Materialize entity states (background job)
entity_ids = ['user_123', 'user_456', 'merchant_789']
detector.entity_state_store.materialize_snapshots_batch(
    'user', entity_ids,
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now()
)

# 2. Inject test scenarios
scenario = detector.scenario_injector.inject_ato_scenario(
    user_id='user_123',
    start_time=datetime.now(),
    duration_hours=2,
    intensity='medium'
)

# 3. Process events (standard pipeline)
results = detector.run_pipeline(
    start_time=datetime.now() - timedelta(hours=24),
    end_time=datetime.now()
)

# 4. Analyze state transitions
transition = detector.state_transition_tracker.compute_transition(
    'user', 'user_123',
    from_timestamp=datetime.now() - timedelta(hours=2),
    to_timestamp=datetime.now()
)

# 5. Get natural language explanation
explanation = detector.natural_language_state.describe_state(
    'user', 'user_123',
    state_features=transition['deltas'],
    baseline_features=baseline
)

# 6. Build graph for relationship analysis
graph_data = detector.graph_builder.build_graph_from_events(
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now()
)

# 7. Get entity memory (for agent reasoning)
memory = detector.entity_state_store.get_entity_memory(
    'user', 'user_123', lookback_hours=168
)
```

## Use Cases

### 1. Real-Time Fraud Detection
```python
# Process new events
events = detector.load_events(
    start_time=datetime.now() - timedelta(minutes=5),
    end_time=datetime.now()
)

# Get current state (from materialized snapshots)
state = detector.entity_state_store.get_snapshot('user', 'user_123', datetime.now())

# Generate natural language description
description = detector.natural_language_state.describe_state(
    'user', 'user_123', state['state_features']
)

# Route to appropriate model
model_name, model, threshold = detector.model_router.select_model(
    'user', 'user_123', datetime.now()
)
```

### 2. Anomaly Explanation
```python
# Detect anomaly
anomaly_score = 0.85

# Get state
state = detector.entity_state_store.get_snapshot('user', 'user_123', datetime.now())

# Get what changed
transition = detector.state_transition_tracker.compute_transition(
    'user', 'user_123',
    from_timestamp=datetime.now() - timedelta(hours=1),
    to_timestamp=datetime.now()
)

# Generate explanation
explanation = detector.natural_language_state.generate_anomaly_explanation(
    'user', 'user_123',
    state['state_features'],
    anomaly_score
)
```

### 3. Scenario Testing
```python
# Inject attack scenario
scenario = detector.scenario_injector.inject_ato_scenario(
    user_id='user_123',
    start_time=datetime.now(),
    duration_hours=2,
    intensity='high'
)

# Process events (including injected scenario)
results = detector.run_pipeline(
    start_time=datetime.now() - timedelta(hours=3),
    end_time=datetime.now()
)

# Check if model detected the attack
alerts = detector.alerting_system.get_alerts(
    start_time=datetime.now() - timedelta(hours=3),
    entity_id='user_123'
)
```

### 4. Graph-Based Analysis
```python
# Build graph
graph_data = detector.graph_builder.build_graph_from_events(
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now()
)

# Get entity community
community = detector.graph_builder.get_entity_community(
    'user', 'user_123', max_hops=2
)

# Analyze relationships
neighbors = detector.graph_builder.get_entity_neighbors(
    'user', 'user_123', relationship_type='transaction'
)
```

### 5. Agent Memory Interface
```python
# Get entity memory (what this entity "remembers")
memory = detector.entity_state_store.get_entity_memory(
    'user', 'user_123', lookback_hours=168
)

# Use in agent reasoning
agent_prompt = f"""
Entity {memory['entity_id']} has the following memory:
- Latest state: {memory['latest_state']}
- Patterns detected: {memory['patterns']}
- State evolution: {memory['state_evolution']}

Based on this memory, should we flag this entity as suspicious?
"""
```

## Database Schema

### New Tables Created

1. **`entity_state_snapshots`**
   - Stores materialized entity states
   - Columns: entity_type, entity_id, snapshot_ts, state_features (JSONB)
   - Indexed for fast temporal queries

2. **`scenario_runs`**
   - Tracks injected scenarios
   - Columns: scenario_type, entity_type, entity_id, start_time, end_time, events_injected

3. **`state_transitions`**
   - Stores state changes
   - Columns: from_snapshot_ts, to_snapshot_ts, transition_delta (JSONB), change_magnitude, change_type

4. **`graph_edges`**
   - Entity relationships
   - Columns: from_entity_type, from_entity_id, to_entity_type, to_entity_id, relationship_type, edge_weight

5. **`graph_nodes`**
   - Entity nodes
   - Columns: entity_type, entity_id, node_degree, total_interactions

## Configuration

All components use the existing `config/default.yaml`. No additional configuration needed.

## Dependencies

New optional dependencies:
- `networkx` - For graph operations (install with: `pip install networkx`)

## Key Benefits

1. **State Persistence**: Fast lookups without recomputation
2. **Temporal Reasoning**: Track what changed, not just current state
3. **Robustness Testing**: Scenario injection for model validation
4. **Graph Reasoning**: Explicit relationship representation
5. **LLM Integration**: Natural language state descriptions
6. **Agent Memory**: Entity memory interface for agentic systems

## Next Steps

1. **Materialize States**: Run background job to materialize entity snapshots
2. **Inject Scenarios**: Create test scenarios for model validation
3. **Build Graphs**: Construct relationship graphs for analysis
4. **Integrate LLMs**: Use natural language state for LLM reasoning
5. **Agent Integration**: Connect to agentic systems using entity memory

## Summary

This enhanced system transforms the fraud detection framework from a **prediction system** into a **reasoning infrastructure**. It provides:

- **Entity-centric state** (not row-centric data)
- **Temporal continuity** (state over time, not isolated events)
- **Explicit relationships** (graph representation, not implicit features)
- **Human-interpretable** (natural language, not just numbers)
- **Agent-ready** (memory interface, not just predictions)

This is the foundation for AI systems that **reason about reality**, not just classify it.

