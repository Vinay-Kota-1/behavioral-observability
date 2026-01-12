import os, json
import pandas as pd
from datetime import datetime, timedelta, timezone

from ingest_postgres import (
    make_engine, ensure_ingest_batch, stable_event_id,
    insert_raw_events, insert_labels, RAW_EVENTS_COLS, LABELS_COLS
)


CSV_PATH = os.path.join("creditcard.csv")  

DB_URL = "postgresql://vinaykota:12345678@localhost:5432/fintech_lab"
engine = make_engine(DB_URL)

INGEST_BATCH_ID = "creditcard_001"
SOURCE_DATASET = "CREDITCARD"

BASE_TIME = datetime(2013, 9, 1, tzinfo=timezone.utc)

def main():
    ensure_ingest_batch(engine, INGEST_BATCH_ID, SOURCE_DATASET, notes="creditcard.csv")

    df = pd.read_csv(CSV_PATH)

    # Build events
    ts = BASE_TIME + pd.to_timedelta(df["Time"], unit="s")

    entity_id = [f"cc_txn_{i}" for i in df.index]

    events = pd.DataFrame({
        "event_id": [stable_event_id(SOURCE_DATASET, str(i)) for i in df.index],
        "ts": ts,
        "entity_type": "user",
        "entity_id": entity_id,
        "event_type": "transaction_auth",
        "status": "success",
        "amount": pd.to_numeric(df["Amount"], errors="coerce"),
        "currency": "EUR",
        "user_id": entity_id,
        "merchant_id": None,
        "api_client_id": None,
        "device_id": None,
        "ip_hash": None,
        "ua_hash": None,
        "endpoint": None,
        "error_code": None,
        "channel": "card_present",
        "source_dataset": SOURCE_DATASET,
        "ingest_batch_id": INGEST_BATCH_ID,
        # Store the V-features to reuse later for supervised ML
        "metadata": [json.dumps({k: row[k] for k in row.index if k.startswith("V")}) for _, row in df.iterrows()],
    })

    for c in RAW_EVENTS_COLS:
        if c not in events.columns:
            events[c] = None
    events = events[RAW_EVENTS_COLS]

    # Labels
    fraud_df = df[df["Class"] == 1].copy()
    labels = pd.DataFrame({
        "ts_start": (BASE_TIME + pd.to_timedelta(fraud_df["Time"], unit="s")),
        "ts_end": None,
        "label_scope": "user",
        "scope_id": [f"cc_txn_{i}" for i in fraud_df.index],
        "label_type": "fraud",
        "label": 1,
        "severity": 5,
        "source_dataset": SOURCE_DATASET,
        "ingest_batch_id": INGEST_BATCH_ID,
        "metadata": [json.dumps({"class": 1})] * len(fraud_df),
    })
    for c in LABELS_COLS:
        if c not in labels.columns:
            labels[c] = None
    labels = labels[LABELS_COLS]

    n_events = insert_raw_events(engine, events)
    n_labels = insert_labels(engine, labels)

    print(f"Inserted {n_events:,} CREDITCARD events")
    print(f"Inserted {n_labels:,} CREDITCARD fraud labels")

if __name__ == "__main__":
    main()
