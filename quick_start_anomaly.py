"""
Quick Start Script for Anomaly Detection Framework

This script demonstrates how to use the anomaly detection framework
with the credit card fraud dataset.
"""

from anomaly_detection import (
    create_tabular_detector_suite,
    DataLoader
)
import os


def main():
    print("="*60)
    print("Anomaly Detection Framework - Quick Start")
    print("="*60)
    
    # Check if credit card data exists
    credit_card_path = "creditcard.csv"
    if not os.path.exists(credit_card_path):
        print(f"\nError: {credit_card_path} not found.")
        print("Please ensure the credit card dataset is in the current directory.")
        return
    
    print(f"\n1. Loading data from {credit_card_path}...")
    X, y = DataLoader.load_credit_card(credit_card_path, sample_size=10000)
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Anomaly rate: {y.sum() / len(y):.4f}")
    
    print("\n2. Preparing train/test split...")
    data = DataLoader.prepare_tabular_data(X, y, test_size=0.3, random_state=42)
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    print("\n3. Creating detector suite...")
    framework = create_tabular_detector_suite(contamination=0.01)
    print(f"   Created {len(framework.detectors)} detectors")
    
    print("\n4. Running complete pipeline...")
    results = framework.run_pipeline(
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        generate_report=True
    )
    
    print("\n5. Detector Comparison:")
    if results['comparison'] is not None:
        print(results['comparison'][['Detector', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']].to_string(index=False))
    
    print("\n" + "="*60)
    print("Quick start completed!")
    print("="*60)
    print("\nFor more examples, see:")
    print("  - anomaly_detection/examples/tabular_example.py")
    print("  - anomaly_detection/examples/timeseries_example.py")
    print("\nFor documentation, see:")
    print("  - anomaly_detection/README.md")


if __name__ == "__main__":
    main()

