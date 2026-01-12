import os
import pandas as pd
from datetime import datetime, timedelta, timezone

from ingest_postgres import (
    make_engine,
    ensure_ingest_batch,
    stable_event_id,
    insert_raw_events,
    insert_labels,
    RAW_EVENTS_COLS,
    LABELS_COLS,
)

# ---------------- CONFIG ----------------
DATA_DIR = "ieee-fraud-detection"
TXN_FILE = os.path.join(DATA_DIR, "train_transaction.csv")

MAX_ROWS = 1_000_000
INGEST_BATCH_ID = "ieee_cis_txn_001"
SOURCE_DATASET = "IEEE_CIS"

DB_URL = "postgresql://vinaykota:12345678@localhost:5432/fintech_lab"
engine = make_engine(DB_URL)

# Fake but consistent anchor time (standard practice)
BASE_TIME = datetime(2017, 12, 1, tzinfo=timezone.utc)


def build_ieee_events_and_labels(df: pd.DataFrame):
    events = []
    labels = []

    for row in df.itertuples(index=False):
        txn_id = str(row.TransactionID)

        ts = BASE_TIME + timedelta(seconds=int(row.TransactionDT))

        entity_id = f"user_{txn_id}"
        event_id = stable_event_id(SOURCE_DATASET, txn_id)

        # ---- raw_events ----
        events.append({
            "event_id": event_id,
            "ts": ts,
            "entity_type": "user",
            "entity_id": entity_id,
            "event_type": "transaction_auth",
            "status": "success",
            "amount": float(row.TransactionAmt) if row.TransactionAmt is not None else None,
            "currency": "USD",

            "user_id": entity_id,
            "merchant_id": None,
            "api_client_id": None,
            "device_id": None,
            "ip_hash": None,
            "ua_hash": None,
            "endpoint": None,
            "error_code": None,
            "channel": "payment",

            "source_dataset": SOURCE_DATASET,
            "ingest_batch_id": INGEST_BATCH_ID,
            "metadata": {
                "transaction_id": txn_id
            }
        })

        # ---- labels_outcomes (fraud) ----
        if getattr(row, "isFraud", 0) == 1:
            labels.append({
                "ts_start": ts,
                "ts_end": None,
                "label_scope": "user",
                "scope_id": entity_id,
                "label_type": "fraud",
                "label": 1,
                "severity": 5,
                "source_dataset": SOURCE_DATASET,
                "ingest_batch_id": INGEST_BATCH_ID,
                "metadata": {
                    "transaction_id": txn_id
                }
            })

    ev_df = pd.DataFrame(events)
    lb_df = pd.DataFrame(labels)

    # Ensure full column coverage
    for c in RAW_EVENTS_COLS:
        if c not in ev_df.columns:
            ev_df[c] = None

    for c in LABELS_COLS:
        if c not in lb_df.columns:
            lb_df[c] = None

    return ev_df, lb_df


def main():
    ensure_ingest_batch(
        engine,
        INGEST_BATCH_ID,
        SOURCE_DATASET,
        notes="IEEE-CIS train_transaction.csv (first 1M rows)"
    )

    print("Reading first 1M rows...")
    df = pd.read_csv(TXN_FILE, nrows=MAX_ROWS)

    print("Building events and labels...")
    events_df, labels_df = build_ieee_events_and_labels(df)

    print("Inserting raw_events...")
    n_events = insert_raw_events(engine, events_df)
    print(f"Inserted {n_events:,} IEEE events")

    print("Inserting labels_outcomes...")
    n_labels = insert_labels(engine, labels_df)
    print(f"Inserted {n_labels:,} fraud labels")


if __name__ == "__main__":
    main()
