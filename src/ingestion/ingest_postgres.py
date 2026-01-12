# %%
# !pip install pandas sqlalchemy psycopg2-binary hashlib pathlib
import hashlib
import json
from typing import Dict, Any, Optional, Iterable, List

import pandas as pd
import uuid
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine  

# %%
DATABASE_URL =  "postgresql://vinaykota:12345678@localhost:5432/fintech_lab"

# %%
RAW_EVENTS_COLS = [
    "event_id", "ts", "entity_type", "entity_id", "event_type", "status",
    "amount", "currency",
    "user_id", "merchant_id", "api_client_id", "device_id",
    "ip_hash", "ua_hash", "endpoint", "error_code", "channel",
    "source_dataset", "ingest_batch_id", "metadata"
]

LABELS_COLS = [
    "ts_start", "ts_end",
    "label_scope", "scope_id",
    "label_type", "label", "severity",
    "source_dataset", "ingest_batch_id", "metadata"
]

# %%
def make_engine(db_url: str) -> Engine:
    return create_engine(db_url, future=True)

# %%
def ensure_ingest_batch(engine: Engine, ingest_batch_id: str, source_dataset: str, notes: str = "") -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO sentinelrisk.ingest_batches (ingest_batch_id, source_dataset, notes)
            VALUES (:bid, :src, :notes)
            ON CONFLICT (ingest_batch_id) DO NOTHING
            """),
            {"bid": ingest_batch_id, "src": source_dataset, "notes": notes}
        )

# %%
def stable_event_id(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

# %%
def coerce_raw_events(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RAW_EVENTS_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"raw_events missing columns: {missing}")

    out = df[RAW_EVENTS_COLS].copy()

    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["event_id", "ts", "entity_type", "entity_id", "event_type", "status", "source_dataset", "ingest_batch_id"])

    # amount numeric
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")

    out["metadata"] = out["metadata"].apply(
        lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
    )

    # json metadata: allow dict/None
    # pandas will keep dict objects; SQLAlchemy will serialize to JSONB
    return out




# %%
def coerce_labels(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in LABELS_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"labels_outcomes missing columns: {missing}")

    out = df[LABELS_COLS].copy()
    out["ts_start"] = pd.to_datetime(out["ts_start"], utc=True, errors="coerce")
    out["ts_end"] = pd.to_datetime(out["ts_end"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts_start", "label_scope", "scope_id", "label_type", "label", "source_dataset", "ingest_batch_id"])
    out["metadata"] = out["metadata"].apply(
        lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
    )

    return out

# %%
def insert_raw_events(engine: Engine, df: pd.DataFrame, chunksize: int = 10_000) -> int:
    df2 = coerce_raw_events(df)
    if df2.empty:
        return 0

    # NOTE: to_sql doesn't support ON CONFLICT easily.
    # We'll insert into a temp table and upsert into raw_events.
    temp_table = "raw_events_staging"

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS pg_temp.{temp_table};"))
        conn.execute(text(f"""
            CREATE TEMP TABLE {temp_table}
            (LIKE sentinelrisk.raw_events INCLUDING DEFAULTS)
            ON COMMIT DROP;
        """))

    df2.to_sql(temp_table, engine, schema="pg_temp", if_exists="append", index=False, chunksize=500, method=None)

    with engine.begin() as conn:
        res = conn.execute(text(f"""
    INSERT INTO sentinelrisk.raw_events (
      {", ".join(RAW_EVENTS_COLS)}
    )
    SELECT
      event_id,
      ts,
      entity_type::sentinelrisk.entity_type_enum,
      entity_id,
      event_type,
      status::sentinelrisk.status_enum,
      amount,
      currency,
      user_id,
      merchant_id,
      api_client_id,
      device_id,
      ip_hash,
      ua_hash,
      endpoint,
      error_code,
      channel,
      source_dataset,
      ingest_batch_id,
      metadata::jsonb
    FROM pg_temp.{temp_table}
    ON CONFLICT (source_dataset, event_id) DO NOTHING;
"""))
        # rowcount may be -1 depending on driver; still fine
    return len(df2)

# %%
def insert_labels(engine: Engine, df: pd.DataFrame, chunksize: int = 5_000) -> int:
    df2 = coerce_labels(df)
    if df2.empty:
        return 0

    # use a unique temp table name to avoid collisions across connections/sessions
    temp_table = f"labels_staging_{uuid.uuid4().hex[:8]}"

    # create a dedicated staging table in the sentinelrisk schema (unique per run)
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS sentinelrisk.{temp_table};"))
        conn.execute(text(f"CREATE TABLE sentinelrisk.{temp_table} (LIKE sentinelrisk.labels_outcomes INCLUDING DEFAULTS);"))

    # upload into the sentinelrisk staging table (separate connection is fine)
    df2.to_sql(temp_table, engine, schema="sentinelrisk", if_exists="append", index=False, chunksize=chunksize, method="multi")

    # upsert/insert from staging into final table, then drop staging
    with engine.begin() as conn:
        conn.execute(text(f"INSERT INTO sentinelrisk.labels_outcomes ({', '.join(LABELS_COLS)}) SELECT {', '.join(LABELS_COLS)} FROM sentinelrisk.{temp_table}"))
        conn.execute(text(f"DROP TABLE IF EXISTS sentinelrisk.{temp_table};"))
    return len(df2)



