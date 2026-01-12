"""
Feature Builder - Point-in-Time Feature Generation for Postgres

This module provides both batch and online feature computation using SQL window functions.
Features are defined in feature_config.yaml.

Usage:
    # Batch mode - rebuild all features
    python feature_builder.py --batch

    # Online mode - get features for a specific entity at a point in time
    python feature_builder.py --entity USER_123 --as-of "2024-01-15 10:30:00"
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: str = "feature_config.yaml") -> Dict[str, Any]:
    """Load feature configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_engine(db_url: str) -> Engine:
    """Create SQLAlchemy engine."""
    return create_engine(db_url, future=True)


# ============================================================================
# SQL Generation Helpers
# ============================================================================

def _window_clause(partition_cols: List[str], order_col: str, interval: Optional[str] = None) -> str:
    """Generate a window clause for SQL."""
    partition = ", ".join(partition_cols)
    if interval:
        return f"OVER (PARTITION BY {partition} ORDER BY {order_col} RANGE BETWEEN INTERVAL '{interval}' PRECEDING AND CURRENT ROW)"
    else:
        return f"OVER (PARTITION BY {partition} ORDER BY {order_col})"


def _lag_window(partition_cols: List[str], order_col: str) -> str:
    """Generate a window clause for LAG functions."""
    partition = ", ".join(partition_cols)
    return f"OVER (PARTITION BY {partition} ORDER BY {order_col})"


# ============================================================================
# Feature SQL Generators
# ============================================================================

def gen_event_count_sql(config: Dict, windows: List[Dict]) -> List[str]:
    """Generate SQL for event count features."""
    if not config.get("enabled", False):
        return []
    
    partition_cols = config.get("partition_columns", ["entity_id"])
    order_col = config.get("order_column", "ts")
    
    sqls = []
    for w in windows:
        if w["name"] in config.get("windows", []):
            window = _window_clause(partition_cols, order_col, w["interval"])
            sqls.append(f"COUNT(*) {window} AS event_count_{w['name']}")
    return sqls


def gen_distinct_count_sql(config: Dict, entity_config: Dict, windows: List[Dict]) -> List[str]:
    """Generate SQL for distinct count features."""
    if not config.get("enabled", False):
        return []
    
    partition_cols = entity_config.get("partition_columns", ["entity_id"])
    order_col = entity_config.get("order_column", "ts")
    
    sqls = []
    for col in config.get("columns", []):
        for w in windows:
            if w["name"] in config.get("windows", []):
                # Note: COUNT(DISTINCT) doesn't work with window functions directly
                # We use a subquery approach or approximate with array_agg
                window = _window_clause(partition_cols, order_col, w["interval"])
                sqls.append(f"(SELECT COUNT(DISTINCT x) FROM UNNEST(ARRAY_AGG({col}) {window}) AS x) AS distinct_{col}s_{w['name']}")
    return sqls


def gen_metric_stats_sql(config: Dict, entity_config: Dict, windows: List[Dict]) -> List[str]:
    """Generate SQL for metric statistics (mean, std, z-score)."""
    if not config.get("enabled", False):
        return []
    
    partition_cols = entity_config.get("partition_columns", ["entity_id"])
    order_col = entity_config.get("order_column", "ts")
    col = config.get("column", "amount")
    
    sqls = []
    for w in windows:
        if w["name"] in config.get("windows", []):
            window = _window_clause(partition_cols, order_col, w["interval"])
            for agg in config.get("aggregations", []):
                if agg == "mean":
                    sqls.append(f"AVG({col}) {window} AS metric_mean_{w['name']}")
                elif agg == "std":
                    sqls.append(f"STDDEV({col}) {window} AS metric_std_{w['name']}")
    
    # Z-score calculation (current value - mean) / std
    for w in windows:
        if w["name"] in config.get("windows", []):
            window = _window_clause(partition_cols, order_col, w["interval"])
            sqls.append(f"""
                CASE 
                    WHEN STDDEV({col}) {window} > 0 
                    THEN ({col} - AVG({col}) {window}) / STDDEV({col}) {window}
                    ELSE 0 
                END AS metric_z_{w['name']}""")
    return sqls


def gen_delta_sql(config: Dict, entity_config: Dict, windows: List[Dict]) -> List[str]:
    """Generate SQL for delta features (change from window start)."""
    if not config.get("enabled", False):
        return []
    
    partition_cols = entity_config.get("partition_columns", ["entity_id"])
    order_col = entity_config.get("order_column", "ts")
    col = config.get("column", "amount")
    lag_window = _lag_window(partition_cols, order_col)
    
    sqls = []
    # Simple lag-based delta
    sqls.append(f"{col} - LAG({col}, 1) {lag_window} AS delta_1")
    
    # Percent change
    if config.get("include_pct_change", False):
        for w in windows:
            if w["name"] in config.get("windows", []):
                window = _window_clause(partition_cols, order_col, w["interval"])
                sqls.append(f"""
                    CASE 
                        WHEN FIRST_VALUE({col}) {window} > 0 
                        THEN ({col} - FIRST_VALUE({col}) {window}) / FIRST_VALUE({col}) {window} * 100
                        ELSE 0 
                    END AS pct_change_{w['name']}""")
    return sqls


def gen_time_since_last_sql(config: Dict, entity_config: Dict) -> List[str]:
    """Generate SQL for time since last event."""
    if not config.get("enabled", False):
        return []
    
    partition_cols = entity_config.get("partition_columns", ["entity_id"])
    order_col = entity_config.get("order_column", "ts")
    lag_window = _lag_window(partition_cols, order_col)
    
    return [f"EXTRACT(EPOCH FROM ({order_col} - LAG({order_col}, 1) {lag_window})) AS time_since_last_event_seconds"]


def gen_higher_order_diff_sql(config: Dict, entity_config: Dict) -> List[str]:
    """Generate SQL for higher-order derivatives (first derivative only - second computed in CTE)."""
    if not config.get("enabled", False):
        return []
    
    partition_cols = entity_config.get("partition_columns", ["entity_id"])
    order_col = entity_config.get("order_column", "ts")
    col = config.get("column", "amount")
    lag_window = _lag_window(partition_cols, order_col)
    
    sqls = []
    # First derivative only - second derivative computed in wrapper CTE
    sqls.append(f"{col} - LAG({col}, 1) {lag_window} AS {col}_diff_1")
    return sqls


def gen_seasonality_sql(config: Dict) -> List[str]:
    """Generate SQL for seasonality/Fourier features."""
    if not config.get("enabled", False):
        return []
    
    sqls = [
        "EXTRACT(HOUR FROM ts) AS hour_of_day",
        "EXTRACT(DOW FROM ts) AS day_of_week",
    ]
    
    for period in config.get("periods", []):
        name = period["name"]
        period_secs = period["period_seconds"]
        # Fourier features: sin and cos of time position within the period
        sqls.append(f"SIN(2 * PI() * EXTRACT(EPOCH FROM ts) / {period_secs}) AS sin_{name}")
        sqls.append(f"COS(2 * PI() * EXTRACT(EPOCH FROM ts) / {period_secs}) AS cos_{name}")
    
    return sqls


# ============================================================================
# Main SQL Builder
# ============================================================================

def build_feature_sql(config: Dict, where_clause: Optional[str] = None) -> str:
    """
    Build the complete SQL query for feature generation.
    
    Args:
        config: Loaded configuration dictionary
        where_clause: Optional WHERE clause for filtering (e.g., for online mode)
    
    Returns:
        Complete SQL query string
    """
    schema = config["database"]["schema"]
    source_table = config["database"]["source_table"]
    entity_config = config["entity"]
    windows = config["time_windows"]
    features_config = config["features"]
    
    # Collect all feature SQL fragments
    feature_sqls = []
    
    # Base columns - include all raw_events columns
    feature_sqls.extend([
        "event_id",
        "ts",
        "entity_type",
        "entity_id",
        "event_type",
        "status",
        "amount",
        "currency",
        "user_id",
        "merchant_id",
        "api_client_id",
        "device_id",
        "ip_hash",
        "ua_hash",
        "endpoint",
        "error_code",
        "channel",
        "source_dataset",
        "ingest_batch_id",
        "metadata"
    ])
    
    # Event counts
    feature_sqls.extend(gen_event_count_sql(
        {**features_config.get("event_counts", {}), **entity_config}, 
        windows
    ))
    
    # Distinct counts (simplified - exact distinct in window is complex)
    # For now, we skip the complex distinct count and use an approximation
    if features_config.get("distinct_counts", {}).get("enabled", False):
        partition_cols = entity_config.get("partition_columns", ["entity_id"])
        order_col = entity_config.get("order_column", "ts")
        for w in windows:
            if w["name"] in features_config["distinct_counts"].get("windows", []):
                # Simplified: just count events as proxy
                window = _window_clause(partition_cols, order_col, w["interval"])
                feature_sqls.append(f"COUNT(*) {window} AS distinct_event_types_{w['name']}_proxy")
    
    # Metric statistics
    feature_sqls.extend(gen_metric_stats_sql(features_config.get("metric_stats", {}), entity_config, windows))
    
    # Deltas
    feature_sqls.extend(gen_delta_sql(features_config.get("deltas", {}), entity_config, windows))
    
    # Time since last
    feature_sqls.extend(gen_time_since_last_sql(features_config.get("time_since_last", {}), entity_config))
    
    # Higher order diffs
    feature_sqls.extend(gen_higher_order_diff_sql(features_config.get("higher_order_diffs", {}), entity_config))
    
    # Seasonality
    feature_sqls.extend(gen_seasonality_sql(features_config.get("seasonality", {})))
    
    # Build the query
    select_clause = ",\n    ".join(feature_sqls)
    
    sql = f"""
SELECT
    {select_clause}
FROM {schema}.{source_table}
"""
    
    if where_clause:
        sql += f"\nWHERE {where_clause}"
    
    sql += "\nORDER BY entity_id, ts"
    
    return sql


def build_label_join_sql(config: Dict, base_sql: str) -> str:
    """
    Wrap the base feature SQL with:
    1. Second derivative computation (to avoid nested window functions)
    2. Lateral join to get time since last label
    """
    schema = config["database"]["schema"]
    labels_table = config["database"]["labels_table"]
    features_config = config["features"]
    entity_config = config["entity"]
    
    # Check if we need to compute second derivative
    higher_order_config = features_config.get("higher_order_diffs", {})
    compute_second_diff = higher_order_config.get("enabled", False) and higher_order_config.get("order", 2) >= 2
    
    # Check if we need label joins
    need_label_joins = features_config.get("time_since_last_label", {}).get("enabled", False)
    
    if not compute_second_diff and not need_label_joins:
        return base_sql
    
    # Build the CTE chain
    partition_cols = entity_config.get("partition_columns", ["entity_id"])
    order_col = entity_config.get("order_column", "ts")
    col = higher_order_config.get("column", "amount")
    partition = ", ".join(partition_cols)
    
    # CTE 1: base features (already has amount_diff_1)
    cte_parts = [f"WITH stage1 AS (\n    {base_sql}\n)"]
    
    # CTE 2: compute second derivative from amount_diff_1
    if compute_second_diff:
        cte_parts.append(f"""
stage2 AS (
    SELECT 
        stage1.*,
        {col}_diff_1 - LAG({col}_diff_1, 1) OVER (PARTITION BY {partition} ORDER BY {order_col}) AS {col}_diff_2
    FROM stage1
)""")
        current_stage = "stage2"
    else:
        current_stage = "stage1"
    
    # Final select with label joins
    if need_label_joins:
        label_types = features_config["time_since_last_label"].get("label_types", ["fraud", "anomaly"])
        
        label_joins = []
        label_selects = []
        
        for i, ltype in enumerate(label_types):
            alias = f"lbl_{i}"
            label_joins.append(f"""
LEFT JOIN LATERAL (
    SELECT ts_start 
    FROM {schema}.{labels_table} 
    WHERE scope_id = {current_stage}.entity_id 
      AND label_type = '{ltype}'
      AND ts_start < {current_stage}.ts
    ORDER BY ts_start DESC
    LIMIT 1
) {alias} ON TRUE""")
            label_selects.append(f"EXTRACT(EPOCH FROM ({current_stage}.ts - {alias}.ts_start)) AS time_since_last_{ltype}_seconds")
        
        sql = ",\n".join(cte_parts) + f"""
SELECT 
    {current_stage}.*,
    {', '.join(label_selects)}
FROM {current_stage}
{' '.join(label_joins)}
"""
    else:
        # Just select from the final stage
        sql = ",\n".join(cte_parts) + f"""
SELECT * FROM {current_stage}
"""
    
    return sql


# ============================================================================
# Batch Mode - Rebuild All Features
# ============================================================================

def rebuild_all_features(config: Dict, engine: Engine) -> int:
    """
    Rebuild the entire features table from raw_events.
    
    Returns:
        Number of rows created
    """
    schema = config["database"]["schema"]
    features_table = config["database"]["features_table"]
    
    # Build the feature SQL
    base_sql = build_feature_sql(config)
    final_sql = build_label_join_sql(config, base_sql)
    
    print("Generated SQL:")
    print(final_sql[:2000] + "..." if len(final_sql) > 2000 else final_sql)
    
    with engine.begin() as conn:
        # Drop and recreate the features table
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}.{features_table}"))
        
        create_sql = f"""
CREATE TABLE {schema}.{features_table} AS
{final_sql}
"""
        conn.execute(text(create_sql))
        
        # Get row count
        result = conn.execute(text(f"SELECT COUNT(*) FROM {schema}.{features_table}"))
        count = result.scalar()
    
    print(f"Created {count} feature rows in {schema}.{features_table}")
    return count


# ============================================================================
# Online Mode - Get Features for Specific Entity
# ============================================================================

def get_features(
    config: Dict, 
    engine: Engine, 
    entity_id: str, 
    as_of_ts: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Get features for a specific entity at a point in time.
    
    This is the "online serving" mode - used when a new event comes in
    and you need to compute features for scoring/inference.
    
    Args:
        config: Loaded configuration
        engine: SQLAlchemy engine
        entity_id: The entity to get features for
        as_of_ts: Point in time (defaults to now). Features are computed
                  using only events BEFORE this timestamp.
    
    Returns:
        DataFrame with computed features
    """
    if as_of_ts is None:
        as_of_ts = datetime.utcnow()
    
    # Build SQL with WHERE clause for this entity
    where_clause = f"entity_id = '{entity_id}' AND ts <= '{as_of_ts.isoformat()}'"
    base_sql = build_feature_sql(config, where_clause)
    final_sql = build_label_join_sql(config, base_sql)
    
    # Get the most recent row (the "current" state)
    wrapped_sql = f"""
WITH features AS (
    {final_sql}
)
SELECT * FROM features
ORDER BY ts DESC
LIMIT 1
"""
    
    df = pd.read_sql(text(wrapped_sql), engine)
    return df


def get_features_batch(
    config: Dict,
    engine: Engine,
    entity_ids: List[str],
    as_of_ts: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Get features for multiple entities at a point in time.
    More efficient than calling get_features in a loop.
    """
    if as_of_ts is None:
        as_of_ts = datetime.utcnow()
    
    entity_list = ", ".join([f"'{e}'" for e in entity_ids])
    where_clause = f"entity_id IN ({entity_list}) AND ts <= '{as_of_ts.isoformat()}'"
    base_sql = build_feature_sql(config, where_clause)
    final_sql = build_label_join_sql(config, base_sql)
    
    # Get the most recent row per entity
    wrapped_sql = f"""
WITH features AS (
    {final_sql}
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY ts DESC) as rn
    FROM features
)
SELECT * FROM ranked WHERE rn = 1
"""
    
    df = pd.read_sql(text(wrapped_sql), engine)
    return df


# ============================================================================
# Cache / Feature Store Table
# ============================================================================

def upsert_feature_cache(
    config: Dict,
    engine: Engine,
    features_df: pd.DataFrame
) -> int:
    """
    Upsert computed features into a cache table for fast lookups.
    This acts as a simple "feature store" for online serving.
    """
    schema = config["database"]["schema"]
    cache_table = "feature_cache"
    
    # Create cache table if not exists
    create_sql = f"""
CREATE TABLE IF NOT EXISTS {schema}.{cache_table} (
    entity_id TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    features JSONB NOT NULL,
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (entity_id, ts)
)
"""
    with engine.begin() as conn:
        conn.execute(text(create_sql))
    
    # Convert features to JSONB and insert
    if features_df.empty:
        return 0
    
    import json
    rows_inserted = 0
    
    with engine.begin() as conn:
        for _, row in features_df.iterrows():
            features_json = json.dumps(row.to_dict(), default=str)
            upsert_sql = f"""
INSERT INTO {schema}.{cache_table} (entity_id, ts, features)
VALUES (:entity_id, :ts, :features::jsonb)
ON CONFLICT (entity_id, ts) DO UPDATE SET
    features = EXCLUDED.features,
    computed_at = NOW()
"""
            conn.execute(text(upsert_sql), {
                "entity_id": row.get("entity_id"),
                "ts": row.get("ts"),
                "features": features_json
            })
            rows_inserted += 1
    
    return rows_inserted


def lookup_cached_features(
    config: Dict,
    engine: Engine,
    entity_id: str,
    as_of_ts: Optional[datetime] = None
) -> Optional[Dict]:
    """
    Look up cached features for an entity.
    Returns the most recent cached features at or before as_of_ts.
    """
    schema = config["database"]["schema"]
    cache_table = "feature_cache"
    
    if as_of_ts is None:
        as_of_ts = datetime.utcnow()
    
    sql = f"""
SELECT features 
FROM {schema}.{cache_table}
WHERE entity_id = :entity_id AND ts <= :as_of_ts
ORDER BY ts DESC
LIMIT 1
"""
    
    with engine.connect() as conn:
        result = conn.execute(text(sql), {"entity_id": entity_id, "as_of_ts": as_of_ts})
        row = result.fetchone()
        if row:
            return row[0]
    return None


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Point-in-Time Feature Builder")
    parser.add_argument("--config", default="feature_config.yaml", help="Path to config file")
    parser.add_argument("--batch", action="store_true", help="Run in batch mode (rebuild all features)")
    parser.add_argument("--entity", type=str, help="Entity ID for online mode")
    parser.add_argument("--as-of", type=str, help="Point-in-time timestamp (ISO format)")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent / args.config
    config = load_config(str(config_path))
    
    # Create engine
    engine = make_engine(config["database"]["url"])
    
    if args.dry_run:
        # Just print the SQL
        sql = build_feature_sql(config)
        sql = build_label_join_sql(config, sql)
        print("=" * 80)
        print("Generated Feature SQL:")
        print("=" * 80)
        print(sql)
        return
    
    if args.batch:
        # Batch mode - rebuild all features
        count = rebuild_all_features(config, engine)
        print(f"Batch complete. Created {count} feature rows.")
    
    elif args.entity:
        # Online mode - get features for specific entity
        as_of_ts = None
        if args.as_of:
            as_of_ts = datetime.fromisoformat(args.as_of)
        
        features = get_features(config, engine, args.entity, as_of_ts)
        print(f"Features for entity {args.entity}:")
        print(features.to_string())
        
        # Optionally cache the result
        upsert_feature_cache(config, engine, features)
        print("Features cached.")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
