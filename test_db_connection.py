"""
Simple script to test PostgreSQL connection and retrieve data from sentinelrisk.raw_events
"""

import pandas as pd
from sqlalchemy import create_engine, text

# Database connection
DB_URL = "postgresql://vinaykota:12345678@localhost:5432/fintech_lab"
SCHEMA = "sentinelrisk"
TABLE = "raw_events"

def main():
    """Test database connection and retrieve sample data."""
    
    print("=" * 60)
    print("PostgreSQL Connection Test")
    print("=" * 60)
    
    try:
        # Create engine
        print(f"\n1. Connecting to database...")
        print(f"   URL: {DB_URL}")
        engine = create_engine(DB_URL)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"   ✓ Connected successfully!")
            print(f"   PostgreSQL version: {version.split(',')[0]}")
        
        # Check if schema exists
        print(f"\n2. Checking schema '{SCHEMA}'...")
        with engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name = '{SCHEMA}'
            """))
            if result.fetchone():
                print(f"   ✓ Schema '{SCHEMA}' exists")
            else:
                print(f"   ✗ Schema '{SCHEMA}' not found")
                return
        
        # Check if table exists
        print(f"\n3. Checking table '{SCHEMA}.{TABLE}'...")
        with engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{SCHEMA}' 
                  AND table_name = '{TABLE}'
            """))
            if result.fetchone():
                print(f"   ✓ Table '{SCHEMA}.{TABLE}' exists")
            else:
                print(f"   ✗ Table '{SCHEMA}.{TABLE}' not found")
                return
        
        # Get table row count
        print(f"\n4. Getting row count...")
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.{TABLE}"))
            count = result.fetchone()[0]
            print(f"   Total rows: {count:,}")
        
        if count == 0:
            print("\n   No data found in table.")
            return
        
        # Get sample data
        print(f"\n5. Retrieving sample data (first 10 rows)...")
        query = text(f"""
            SELECT 
                event_id,
                ts,
                entity_type,
                entity_id,
                event_type,
                status,
                amount,
                currency,
                user_id,
                merchant_id,
                api_client_id
            FROM {SCHEMA}.{TABLE}
            ORDER BY ts DESC
            LIMIT 10
        """)
        
        df = pd.read_sql(query, engine)
        
        print(f"\n   Retrieved {len(df)} rows")
        print("\n" + "=" * 60)
        print("Sample Data:")
        print("=" * 60)
        
        # Display data
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 30)
        
        print(df.to_string(index=False))
        
        # Display summary statistics
        print("\n" + "=" * 60)
        print("Summary Statistics:")
        print("=" * 60)
        
        if 'amount' in df.columns:
            print(f"\nAmount Statistics:")
            print(f"  Count: {df['amount'].notna().sum()}")
            print(f"  Mean: ${df['amount'].mean():.2f}" if df['amount'].notna().any() else "  Mean: N/A")
            print(f"  Min: ${df['amount'].min():.2f}" if df['amount'].notna().any() else "  Min: N/A")
            print(f"  Max: ${df['amount'].max():.2f}" if df['amount'].notna().any() else "  Max: N/A")
        
        if 'entity_type' in df.columns:
            print(f"\nEntity Type Distribution:")
            print(df['entity_type'].value_counts().to_string())
        
        if 'event_type' in df.columns:
            print(f"\nEvent Type Distribution:")
            print(df['event_type'].value_counts().head(10).to_string())
        
        if 'status' in df.columns:
            print(f"\nStatus Distribution:")
            print(df['status'].value_counts().to_string())
        
        # Date range
        if 'ts' in df.columns and not df['ts'].empty:
            print(f"\nTimestamp Range:")
            print(f"  Earliest: {df['ts'].min()}")
            print(f"  Latest: {df['ts'].max()}")
        
        print("\n" + "=" * 60)
        print("Connection test completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

