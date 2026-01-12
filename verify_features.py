"""
Verification script for the Feature Store

Run this after executing: python feature_builder.py --batch

This script:
1. Loads the features table
2. Joins with labels
3. Shows sample data and statistics
"""

import pandas as pd
from sqlalchemy import create_engine, text

# Database connection
DATABASE_URL = "postgresql://vinaykota:12345678@localhost:5432/fintech_lab"
engine = create_engine(DATABASE_URL, future=True)


def verify_features():
    """Load and verify the features table."""
    
    print("=" * 80)
    print("FEATURE TABLE VERIFICATION")
    print("=" * 80)
    
    # 1. Check if features table exists and has data
    try:
        count_result = pd.read_sql(
            "SELECT COUNT(*) as cnt FROM sentinelrisk.features", 
            engine
        )
        row_count = count_result['cnt'].iloc[0]
        print(f"\n✓ Features table exists with {row_count:,} rows")
    except Exception as e:
        print(f"\n✗ Features table not found. Run 'python feature_builder.py --batch' first.")
        print(f"  Error: {e}")
        return None
    
    # 2. Load a sample of the features
    print("\n" + "-" * 40)
    print("LOADING FEATURES...")
    print("-" * 40)
    
    df = pd.read_sql("""
        SELECT * FROM sentinelrisk.features
        ORDER BY ts DESC
        LIMIT 1000
    """, engine)
    
    print(f"\nLoaded {len(df)} sample rows")
    print(f"\nColumns ({len(df.columns)} total):")
    print(df.columns.tolist())
    
    # 3. Show data types
    print("\n" + "-" * 40)
    print("DATA TYPES")
    print("-" * 40)
    print(df.dtypes)
    
    # 4. Show sample data
    print("\n" + "-" * 40)
    print("SAMPLE DATA (first 5 rows)")
    print("-" * 40)
    print(df.head().to_string())
    
    # 5. Feature statistics
    print("\n" + "-" * 40)
    print("FEATURE STATISTICS")
    print("-" * 40)
    
    # Numeric feature columns
    feature_cols = [c for c in df.columns if any(x in c for x in [
        'event_count', 'metric_', 'delta', 'pct_change', 
        'time_since', 'diff_', 'sin_', 'cos_', 'hour_', 'day_'
    ])]
    
    if feature_cols:
        print(df[feature_cols].describe().round(3).to_string())
    
    # 6. Check for nulls
    print("\n" + "-" * 40)
    print("NULL VALUE CHECK")
    print("-" * 40)
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    null_df = pd.DataFrame({'null_count': null_counts, 'null_pct': null_pct})
    print(null_df[null_df['null_count'] > 0].to_string() or "No nulls found!")
    
    # 7. Join with labels to create training dataset
    print("\n" + "-" * 40)
    print("TRAINING DATASET (Features + Labels)")
    print("-" * 40)
    
    training_df = pd.read_sql("""
        SELECT 
            f.*,
            l.label,
            l.label_type,
            l.severity
        FROM sentinelrisk.features f
        LEFT JOIN sentinelrisk.labels_outcomes l 
            ON f.entity_id = l.scope_id 
            AND f.ts BETWEEN l.ts_start AND COALESCE(l.ts_end, f.ts)
        LIMIT 1000
    """, engine)
    
    print(f"\nTraining dataset shape: {training_df.shape}")
    print(f"\nLabel distribution:")
    print(training_df['label'].value_counts(dropna=False))
    
    return training_df


def get_full_training_data():
    """Load the complete training dataset."""
    
    print("Loading full training dataset...")
    
    df = pd.read_sql("""
        SELECT 
            f.*,
            l.label,
            l.label_type,
            l.severity
        FROM sentinelrisk.features f
        LEFT JOIN sentinelrisk.labels_outcomes l 
            ON f.entity_id = l.scope_id 
            AND f.ts BETWEEN l.ts_start AND COALESCE(l.ts_end, f.ts)
    """, engine)
    
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    
    return df


if __name__ == "__main__":
    # Run verification
    sample_df = verify_features()
    
    if sample_df is not None:
        print("\n" + "=" * 80)
        print("VERIFICATION COMPLETE")
        print("=" * 80)
        print("\nTo load full training data in your code:")
        print("    from verify_features import get_full_training_data")
        print("    df = get_full_training_data()")
