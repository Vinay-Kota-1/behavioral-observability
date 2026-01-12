"""
Simple script to retrieve data from sentinelrisk.raw_events table
"""

import pandas as pd
from sqlalchemy import create_engine, text
import json
import os
from datetime import datetime

# Database connection
DB_URL = "postgresql://vinaykota:12345678@localhost:5432/fintech_lab"
SCHEMA = "sentinelrisk"
TABLE = "raw_events"

def retrieve_raw_events(limit=100, start_time=None, end_time=None):
    """
    Retrieve data from raw_events table.
    
    Args:
        limit: Maximum number of rows to retrieve (default: 100)
        start_time: Optional start timestamp filter
        end_time: Optional end timestamp filter
    
    Returns:
        DataFrame with raw_events data
    """
    # #region agent log
    log_path = "/Users/vinaykota/Downloads/Datasets/.cursor/debug.log"
    with open(log_path, "a") as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "retrieve_raw_events.py:25", "message": "Function entry", "data": {"limit": limit, "start_time": str(start_time), "end_time": str(end_time), "db_url": DB_URL, "schema": SCHEMA, "table": TABLE}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
    # #endregion
    
    try:
        # Create engine
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "D", "location": "retrieve_raw_events.py:30", "message": "Creating engine", "data": {"db_url": DB_URL}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
        # #endregion
        engine = create_engine(DB_URL)
        
        # Build query
        query = f"""
            SELECT *
            FROM {SCHEMA}.{TABLE}
            WHERE 1=1
        """
        params = {}
        
        if start_time:
            query += " AND ts >= :start_time"
            params['start_time'] = start_time
        
        if end_time:
            query += " AND ts <= :end_time"
            params['end_time'] = end_time
        
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "retrieve_raw_events.py:50", "message": "Query before LIMIT", "data": {"query": query, "params": {k: str(v) for k, v in params.items()}}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
        # #endregion
        
        # Fix: Use string formatting for LIMIT instead of parameter binding
        query += f" ORDER BY ts DESC LIMIT {limit}"
        # Note: Removed params['limit'] since LIMIT doesn't work well with parameterized queries
        
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "retrieve_raw_events.py:55", "message": "Final query", "data": {"query": query, "params": {k: str(v) for k, v in params.items()}}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
        # #endregion
        
        # Execute query
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "retrieve_raw_events.py:60", "message": "Before pd.read_sql", "data": {}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
        # #endregion
        df = pd.read_sql(text(query), engine, params=params if params else None)
        
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "retrieve_raw_events.py:63", "message": "After pd.read_sql", "data": {"rows": len(df), "columns": list(df.columns) if not df.empty else []}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
        # #endregion
        
        return df
    except Exception as e:
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "retrieve_raw_events.py:68", "message": "Exception caught", "data": {"error_type": type(e).__name__, "error_message": str(e)}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
        # #endregion
        raise

def main():
    """Main function to retrieve and display data."""
    
    print("Retrieving data from sentinelrisk.raw_events...")
    print("-" * 60)
    
    try:
        # #region agent log
        log_path = "/Users/vinaykota/Downloads/Datasets/.cursor/debug.log"
        with open(log_path, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "retrieve_raw_events.py:75", "message": "Main function entry", "data": {}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
        # #endregion
        
        # Retrieve data
        df = retrieve_raw_events(limit=50)
        
        if df.empty:
            print("No data found in raw_events table.")
            return
        
        print(f"\nRetrieved {len(df)} rows")
        print(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Display data
        print("\n" + "=" * 60)
        print("Data Preview:")
        print("=" * 60)
        
        # Set display options
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        
        # Show first 20 rows
        print(df.head(20).to_string(index=False))
        
        # Show basic info
        print("\n" + "=" * 60)
        print("Data Info:")
        print("=" * 60)
        print(f"Total rows retrieved: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        
        # Show data types
        print("\nColumn Data Types:")
        print(df.dtypes.to_string())
        
        # Show null counts
        print("\nNull Value Counts:")
        null_counts = df.isnull().sum()
        null_counts = null_counts[null_counts > 0]
        if len(null_counts) > 0:
            print(null_counts.to_string())
        else:
            print("No null values found")
        
        # Show unique value counts for categorical columns
        print("\nUnique Value Counts (top columns):")
        for col in df.columns:
            # #region agent log
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "retrieve_raw_events.py:151", "message": "Processing column", "data": {"column": col, "dtype": str(df[col].dtype)}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
            # #endregion
            
            # Skip columns that might contain unhashable types (dicts, lists)
            # Check if column contains dict/list objects
            try:
                # Sample a few values to check type
                sample_values = df[col].dropna().head(5)
                has_unhashable = any(isinstance(val, (dict, list)) for val in sample_values)
                
                if has_unhashable:
                    # #region agent log
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "retrieve_raw_events.py:160", "message": "Skipping unhashable column", "data": {"column": col}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
                    # #endregion
                    print(f"  {col}: (contains dict/list - skipping unique count)")
                    continue
                
                # Try to get unique count
                if df[col].dtype == 'object':
                    unique_count = df[col].nunique()
                    if unique_count < 20:
                        print(f"  {col}: {unique_count} unique values")
                        if unique_count <= 10:
                            unique_vals = df[col].dropna().unique().tolist()
                            # Convert to strings if needed for display
                            unique_vals = [str(v) if not isinstance(v, (str, int, float)) else v for v in unique_vals]
                            print(f"    Values: {unique_vals}")
                else:
                    unique_count = df[col].nunique()
                    if unique_count < 20:
                        print(f"  {col}: {unique_count} unique values")
                        if unique_count <= 10:
                            print(f"    Values: {df[col].unique().tolist()}")
            except (TypeError, ValueError) as e:
                # #region agent log
                with open(log_path, "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "retrieve_raw_events.py:180", "message": "Error processing column", "data": {"column": col, "error": str(e)}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
                # #endregion
                print(f"  {col}: (error computing unique count: {type(e).__name__})")
                continue
        
        # Save to CSV if needed
        output_file = "raw_events_sample.csv"
        df.to_csv(output_file, index=False)
        print(f"\nData saved to: {output_file}")
        
    except Exception as e:
        # #region agent log
        log_path = "/Users/vinaykota/Downloads/Datasets/.cursor/debug.log"
        with open(log_path, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "retrieve_raw_events.py:130", "message": "Exception in main", "data": {"error_type": type(e).__name__, "error_message": str(e)}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
        # #endregion
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

