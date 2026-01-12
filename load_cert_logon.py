import os, json
import pandas as pd

from ingest_postgres import (
    make_engine, ensure_ingest_batch, stable_event_id,
    insert_raw_events, RAW_EVENTS_COLS
)

DB_URL = "postgresql://vinaykota:12345678@localhost:5432/fintech_lab"
engine = make_engine(DB_URL)

INGEST_BATCH_ID = "cert_logon_001"
SOURCE_DATASET = "CERT"

CERT_DIR = "r1"   # <-- change
LOGON_PATH = os.path.join(CERT_DIR, "logon.csv")        # <-- change

# Edit these to match your CERT columns
COL_TS = "date"          # timestamp column
COL_USER = "user"
COL_PC = "pc"
COL_SUCCESS = "success"  # bool or 0/1 or "True/False"

def normalize_status(x):
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "success", "successful"):
        return "success"
    if s in ("0", "false", "f", "fail", "failed"):
        return "fail"
    return "unknown"

def main():
    ensure_ingest_batch(engine, INGEST_BATCH_ID, SOURCE_DATASET, notes="CERT logon-only")

    # BIG FILE: read in chunks
    chunks = pd.read_csv(LOGON_PATH, chunksize=500_000)
    total = 0

    for df in chunks:
        df[COL_TS] = pd.to_datetime(df[COL_TS], utc=True, errors="coerce")
        df = df.dropna(subset=[COL_TS, COL_USER])

        # determine source for status: prefer configured COL_SUCCESS, fall back to
        # an `activity` column (e.g., Logon/Logoff), otherwise mark unknown
        if COL_SUCCESS in df.columns:
            status_src = df[COL_SUCCESS].astype(str)
        elif "activity" in df.columns:
            status_src = df["activity"].astype(str).str.strip().str.lower().map(
                lambda s: "success" if s in ("logon", "logoff") else s
            )
        else:
            status_src = pd.Series(["unknown"] * len(df), index=df.index)

        events = pd.DataFrame({
            "ts": df[COL_TS],
            "entity_type": "user",
            "entity_id": df[COL_USER].astype(str),
            "event_type": "login",
            "status": status_src.apply(normalize_status),
            "amount": None,
            "currency": None,
            "user_id": df[COL_USER].astype(str),
            "device_id": df[COL_PC].astype(str) if COL_PC in df.columns else None,
            "source_dataset": SOURCE_DATASET,
            "ingest_batch_id": INGEST_BATCH_ID,
            "metadata": [json.dumps({})] * len(df),
        })

        # stable event_id: (user, ts, pc, status)
        events["event_id"] = [
            stable_event_id(SOURCE_DATASET, u, t.isoformat(), str(p), st)
            for u, t, p, st in zip(
                events["entity_id"],
                events["ts"],
                events["device_id"] if "device_id" in events.columns else [""] * len(events),
                events["status"]
            )
        ]

        # fill required columns
        for c in RAW_EVENTS_COLS:
            if c not in events.columns:
                events[c] = None
        events = events[RAW_EVENTS_COLS]

        n = insert_raw_events(engine, events)
        total += n
        print(f"Loaded {n:,} CERT login events (running total {total:,})")

    print(f"Done. Total CERT login events inserted: {total:,}")

if __name__ == "__main__":
    main()
