import os
import json
import pandas as pd

from ingest_postgres import (
    make_engine,
    ensure_ingest_batch,
    insert_raw_events,
    RAW_EVENTS_COLS,
)

DB_URL = "postgresql://vinaykota:12345678@localhost:5432/fintech_lab"
engine = make_engine(DB_URL)

INGEST_BATCH_ID = "cert_device_001"
SOURCE_DATASET = "CERT"

CERT_DIR = "r1"  # <-- change if needed
DEVICE_PATH = os.path.join(CERT_DIR, "device.csv")     # <-- change if needed


def main():
    ensure_ingest_batch(engine, INGEST_BATCH_ID, SOURCE_DATASET, notes="CERT device connect/disconnect")

    total = 0

    # attempt to auto-detect delimiter and common column names; CSVs may be comma-separated
    for df in pd.read_csv(DEVICE_PATH, sep=None, engine="python", chunksize=500_000):
        # discover column names (case-insensitive) for required fields
        cols_lower = {c.lower(): c for c in df.columns}

        def find_col(candidates):
            for cand in candidates:
                if cand.lower() in cols_lower:
                    return cols_lower[cand.lower()]
            return None

        id_col = find_col(["id", "row_id", "event_id"])
        date_col = find_col(["date", "datetime", "timestamp", "time"])
        user_col = find_col(["user", "username", "userid"])
        pc_col = find_col(["pc", "device", "host"])
        activity_col = find_col(["activity", "action", "event"])

        if not all([id_col, date_col, user_col, pc_col, activity_col]):
            print("Skipping chunk: missing required columns", {"id": id_col, "date": date_col, "user": user_col, "pc": pc_col, "activity": activity_col})
            continue

        # Parse timestamp; allow multiple formats and be tolerant
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
        df = df.dropna(subset=[id_col, date_col, user_col, pc_col, activity_col])

        # If timestamps are timezone-naive, set to UTC
        try:
            if df[date_col].dt.tz is None:
                df[date_col] = df[date_col].dt.tz_localize("UTC")
        except Exception:
            try:
                df[date_col] = df[date_col].dt.tz_localize("UTC")
            except Exception:
                pass

        events = pd.DataFrame({
            "event_id": df[id_col].astype(str),          # CERT already gives a unique id
            "ts": df[date_col],
            "entity_type": "device",                   # matches your enum
            "entity_id": df[pc_col].astype(str),         # device is the entity
            "event_type": df[activity_col].astype(str).str.lower(),  # connect/disconnect
            "status": "success",                       # these are factual logs
            "amount": None,
            "currency": None,

            "user_id": df[user_col].astype(str),
            "merchant_id": None,
            "api_client_id": None,
            "device_id": df["pc"].astype(str),

            "ip_hash": None,
            "ua_hash": None,
            "endpoint": None,
            "error_code": None,
            "channel": "endpoint",

            "source_dataset": SOURCE_DATASET,
            "ingest_batch_id": INGEST_BATCH_ID,

            "metadata": df[id_col].astype(str).apply(lambda x: json.dumps({"cert_row_id": x})),
        })

        # Ensure all canonical columns exist
        for c in RAW_EVENTS_COLS:
            if c not in events.columns:
                events[c] = None
        events = events[RAW_EVENTS_COLS]

        n = insert_raw_events(engine, events)
        total += n
        print(f"Loaded {n:,} CERT device events (running total {total:,})")

    print(f"Done. Total CERT device events inserted: {total:,}")


if __name__ == "__main__":
    main()
