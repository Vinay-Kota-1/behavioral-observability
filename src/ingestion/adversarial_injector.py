"""
Adversarial Injector - Synthetic Anomaly Generation for Testing

This module injects synthetic anomalous events into raw_events table,
then uses the feature_builder to generate point-in-time features.

Workflow:
1. Select real entities from raw_events as templates
2. Generate synthetic anomalous events based on scenario patterns
3. Insert into raw_events with unique event_ids
4. Add labels to labels_outcomes table
5. Use feature_builder to regenerate features

Usage:
    # List available scenarios
    python adversarial_injector.py --list

    # Inject a specific scenario
    python adversarial_injector.py --scenario suspicious_login_burst --count 10

    # Inject all scenarios
    python adversarial_injector.py --all --count 5
"""

import argparse
import yaml
import uuid
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: str = "adversarial_config.yaml") -> Dict[str, Any]:
    """Load adversarial injection configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_engine(db_url: str) -> Engine:
    """Create SQLAlchemy engine."""
    return create_engine(db_url, future=True)


def stable_event_id(*parts: str) -> str:
    """Generate stable event ID from parts."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# ============================================================================
# Entity Selection
# ============================================================================

def get_sample_entities(
    engine: Engine,
    schema: str,
    table: str,
    source_dataset: str,
    entity_type: str,
    n: int = 10,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Get sample entities from raw_events to use as templates.
    Returns recent events for each entity.
    """
    query = f"""
        WITH entities AS (
            SELECT entity_id FROM (
                SELECT DISTINCT entity_id
                FROM {schema}.{table}
                WHERE source_dataset = :source_dataset
                  AND entity_type = :entity_type
            ) sub
            ORDER BY RANDOM()
            LIMIT :n
        )
        SELECT r.*
        FROM {schema}.{table} r
        INNER JOIN entities e ON r.entity_id = e.entity_id
        WHERE r.source_dataset = :source_dataset
        ORDER BY r.entity_id, r.ts DESC
    """
    
    df = pd.read_sql(
        text(query),
        engine,
        params={"source_dataset": source_dataset, "entity_type": entity_type, "n": n}
    )
    
    # Get one row per entity (most recent)
    return df.groupby("entity_id").first().reset_index()


# ============================================================================
# Event Generation
# ============================================================================

def generate_timestamps(
    base_time: datetime,
    count: int,
    time_window_seconds: int,
    distribution: str = "uniform",
    hour_range: Optional[List[int]] = None
) -> List[datetime]:
    """Generate timestamps for synthetic events."""
    timestamps = []
    
    if distribution == "uniform":
        # Evenly spaced within window
        interval = time_window_seconds / max(count - 1, 1)
        for i in range(count):
            ts = base_time + timedelta(seconds=i * interval)
            timestamps.append(ts)
    
    elif distribution == "random":
        # Random within window
        for _ in range(count):
            offset = random.uniform(0, time_window_seconds)
            ts = base_time + timedelta(seconds=offset)
            timestamps.append(ts)
        timestamps.sort()
    
    elif distribution == "poisson":
        # Poisson-like (exponential inter-arrival times)
        current = base_time
        mean_interval = time_window_seconds / count
        for _ in range(count):
            interval = random.expovariate(1 / mean_interval)
            current = current + timedelta(seconds=min(interval, time_window_seconds / 2))
            timestamps.append(current)
    
    # Apply hour range constraint if specified
    if hour_range:
        adjusted = []
        for ts in timestamps:
            # Adjust to fall within hour range
            hour = random.randint(hour_range[0], hour_range[1])
            adjusted_ts = ts.replace(hour=hour, minute=random.randint(0, 59))
            adjusted.append(adjusted_ts)
        timestamps = adjusted
    
    return timestamps


def generate_amounts(
    count: int,
    config: Dict[str, Any]
) -> List[Optional[float]]:
    """Generate amounts based on distribution config."""
    dist_type = config.get("type", "normal")
    
    if dist_type == "normal":
        mean = config.get("mean", 100)
        std = config.get("std", 30)
        return [max(0.01, np.random.normal(mean, std)) for _ in range(count)]
    
    elif dist_type == "fixed":
        value = config.get("value", 100)
        return [value] * count
    
    elif dist_type == "round":
        values = config.get("values", [100, 200, 500])
        return [random.choice(values) for _ in range(count)]
    
    elif dist_type == "spike":
        baseline = config.get("baseline_mean", 30)
        multiplier = config.get("spike_multiplier", 5)
        # First half normal, second half spiked
        amounts = []
        for i in range(count):
            if i < count // 2:
                amounts.append(baseline + np.random.normal(0, 5))
            else:
                amounts.append(baseline * multiplier + np.random.normal(0, 10))
        return amounts
    
    elif dist_type == "drift":
        start = config.get("start_value", 30)
        end = config.get("end_value", 80)
        noise = config.get("noise", 5)
        return [
            start + (end - start) * i / max(count - 1, 1) + np.random.normal(0, noise)
            for i in range(count)
        ]
    
    return [None] * count


def generate_synthetic_events(
    template: pd.Series,
    scenario: Dict[str, Any],
    scenario_name: str,
    run_id: str
) -> pd.DataFrame:
    """Generate synthetic events based on scenario pattern."""
    pattern = scenario.get("injection_pattern", {})
    count = pattern.get("count", 10)
    time_window = pattern.get("time_window_seconds", 3600)
    time_dist = pattern.get("time_distribution", "uniform")
    hour_range = pattern.get("hour_range")
    amount_config = pattern.get("amount_distribution", {"type": "normal", "mean": 100, "std": 30})
    
    # Base time: use template time or current time
    if pd.notna(template.get("ts")):
        base_time = template["ts"]
        if isinstance(base_time, str):
            base_time = pd.to_datetime(base_time)
    else:
        base_time = datetime.now()
    
    # Handle silence_before (period of inactivity)
    silence_hours = pattern.get("silence_before_hours", 0)
    if silence_hours > 0:
        base_time = base_time + timedelta(hours=silence_hours)
    
    # Generate timestamps
    timestamps = generate_timestamps(base_time, count, time_window, time_dist, hour_range)
    
    # Generate amounts
    if scenario.get("event_type") in ["transaction_auth"]:
        amounts = generate_amounts(count, amount_config)
    elif scenario.get("event_type") == "metric_observation":
        amounts = generate_amounts(count, amount_config)
    else:
        amounts = [None] * count
    
    # Build events
    events = []
    for i, (ts, amount) in enumerate(zip(timestamps, amounts)):
        event_id = stable_event_id(
            "adversarial", run_id, scenario_name, template["entity_id"], str(i)
        )
        
        event = {
            "event_id": event_id,
            "ts": ts,
            "entity_type": scenario.get("entity_type", template.get("entity_type")),
            "entity_id": template["entity_id"],
            "event_type": scenario.get("event_type", template.get("event_type")),
            "status": scenario.get("status", "success"),
            "amount": amount,
            "currency": template.get("currency", "USD"),
            "user_id": template.get("user_id"),
            "merchant_id": template.get("merchant_id"),
            "api_client_id": template.get("api_client_id"),
            "device_id": template.get("device_id"),
            "ip_hash": template.get("ip_hash"),
            "ua_hash": template.get("ua_hash"),
            "endpoint": template.get("endpoint"),
            "error_code": template.get("error_code"),
            "channel": scenario.get("channel", template.get("channel")),
            "source_dataset": scenario.get("source_dataset"),
            "ingest_batch_id": f"adversarial_{run_id}",
            "metadata": json.dumps({"adversarial": True, "scenario": scenario_name, "run_id": run_id})
        }
        events.append(event)
    
    return pd.DataFrame(events)


# ============================================================================
# Injection Functions
# ============================================================================

def inject_scenario(
    config: Dict[str, Any],
    engine: Engine,
    scenario_name: str,
    entity_count: int = 5,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Inject a scenario by:
    1. Selecting template entities from raw_events
    2. Generating synthetic anomalous events
    3. Inserting into raw_events
    4. Adding labels to labels_outcomes
    """
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    
    schema = config["database"]["schema"]
    raw_table = config["database"]["raw_events_table"]
    labels_table = config["database"]["labels_table"]
    
    scenario = config["scenarios"].get(scenario_name)
    if not scenario:
        available = list(config["scenarios"].keys())
        raise ValueError(f"Scenario '{scenario_name}' not found. Available: {available}")
    
    run_id = str(uuid.uuid4())[:8]
    
    print(f"\n{'='*60}")
    print(f"INJECTING SCENARIO: {scenario_name}")
    print(f"{'='*60}")
    print(f"Description: {scenario['description']}")
    print(f"Source: {scenario['source_dataset']}")
    print(f"Run ID: {run_id}")
    
    # Get template entities
    print(f"\nSelecting {entity_count} template entities...")
    templates = get_sample_entities(
        engine, schema, raw_table,
        scenario["source_dataset"],
        scenario["entity_type"],
        n=entity_count,
        seed=seed
    )
    
    if templates.empty:
        print(f"⚠ No entities found for source_dataset={scenario['source_dataset']}")
        return {"status": "skipped", "reason": "no_entities"}
    
    print(f"Found {len(templates)} template entities")
    
    # Generate synthetic events for each entity
    all_events = []
    for _, template in templates.iterrows():
        events_df = generate_synthetic_events(template, scenario, scenario_name, run_id)
        all_events.append(events_df)
    
    all_events_df = pd.concat(all_events, ignore_index=True)
    print(f"Generated {len(all_events_df)} synthetic events")
    
    # Create ingest_batch first (FK constraint)
    ingest_batch_id = f"adversarial_{run_id}"
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO sentinelrisk.ingest_batches (ingest_batch_id, source_dataset, notes)
            VALUES (:bid, :src, :notes)
            ON CONFLICT (ingest_batch_id) DO NOTHING
        """), {
            "bid": ingest_batch_id,
            "src": f"adversarial_{scenario_name}",
            "notes": f"Adversarial injection: {scenario['description']}"
        })
    print(f"Created ingest batch: {ingest_batch_id}")
    
    # Insert into raw_events
    print("Inserting into raw_events...")
    all_events_df.to_sql(
        raw_table,
        engine,
        schema=schema,
        if_exists="append",
        index=False,
        chunksize=50  # Smaller chunks to avoid parameter limits
    )
    
    # Create labels
    if scenario.get("anomaly_label"):
        print("Creating labels...")
        label_config = scenario["anomaly_label"]
        
        labels = []
        for _, template in templates.iterrows():
            # Create label spanning the injection period
            entity_events = all_events_df[all_events_df["entity_id"] == template["entity_id"]]
            if not entity_events.empty:
                ts_start = entity_events["ts"].min()
                ts_end = entity_events["ts"].max()
                
                # Note: label column is integer (1 = anomaly/fraud, 0 = normal)
                label = {
                    "ts_start": ts_start,
                    "ts_end": ts_end,
                    "label_scope": scenario["entity_type"],
                    "scope_id": template["entity_id"],
                    "label_type": label_config.get("label_type", "anomaly"),
                    "label": 1,  # 1 = anomaly/fraud
                    "severity": int(label_config.get("severity", 3)),
                    "source_dataset": f"adversarial_{scenario_name}",
                    "ingest_batch_id": f"adversarial_{run_id}",
                    "metadata": json.dumps({
                        "scenario": scenario_name, 
                        "run_id": run_id,
                        "anomaly_type": label_config.get("label", scenario_name)
                    })
                }
                labels.append(label)
        
        labels_df = pd.DataFrame(labels)
        labels_df.to_sql(
            labels_table,
            engine,
            schema=schema,
            if_exists="append",
            index=False
        )
        print(f"Created {len(labels_df)} labels")
    
    print(f"✓ Injection complete: {len(all_events_df)} events, {len(templates)} entities")
    
    return {
        "status": "completed",
        "run_id": run_id,
        "scenario": scenario_name,
        "events_created": len(all_events_df),
        "entities_affected": len(templates)
    }


def inject_all_scenarios(
    config: Dict[str, Any],
    engine: Engine,
    entity_count: int = 5,
    seed: Optional[int] = None
) -> List[Dict]:
    """Inject all scenarios."""
    results = []
    for scenario_name in config["scenarios"]:
        result = inject_scenario(config, engine, scenario_name, entity_count, seed)
        results.append(result)
    return results


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Adversarial Injection for Anomaly Testing")
    parser.add_argument("--config", default="adversarial_config.yaml", help="Path to config")
    parser.add_argument("--scenario", type=str, help="Scenario name to inject")
    parser.add_argument("--all", action="store_true", help="Inject all scenarios")
    parser.add_argument("--count", type=int, default=5, help="Number of entities per scenario")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--list", action="store_true", help="List scenarios")
    
    args = parser.parse_args()
    
    config_path = Path(__file__).parent / args.config
    config = load_config(str(config_path))
    engine = make_engine(config["database"]["url"])
    
    if args.list:
        print("\nAvailable Scenarios:")
        print("-" * 60)
        for name, scenario in config["scenarios"].items():
            print(f"  {name}")
            print(f"    Source: {scenario['source_dataset']}")
            print(f"    Type: {scenario['event_type']}")
            print(f"    Description: {scenario['description']}")
            print()
        return
    
    if args.all:
        results = inject_all_scenarios(config, engine, args.count, args.seed)
        print("\n" + "=" * 60)
        print("ALL SCENARIOS COMPLETE")
        print("=" * 60)
        total_events = sum(r.get("events_created", 0) for r in results)
        print(f"Total events created: {total_events}")
        for r in results:
            print(f"  • {r.get('scenario', 'unknown')}: {r.get('events_created', 0)} events")
    
    elif args.scenario:
        result = inject_scenario(config, engine, args.scenario, args.count, args.seed)
        print(f"\nResult: {result}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
