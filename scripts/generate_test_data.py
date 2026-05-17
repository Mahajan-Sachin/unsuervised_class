"""
Run this once to generate sample test CSV files from creditcard.csv
Usage: python scripts/generate_test_data.py
"""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'creditcard.csv')
OUT_DIR   = os.path.join(BASE_DIR, 'data')

def generate():
    print("Loading creditcard.csv...")
    df = pd.read_csv(DATA_PATH)

    normal = df[df['Class'] == 0]
    fraud  = df[df['Class'] == 1]

    # Drop Class column — we don't give labels to the model
    drop = lambda d: d.drop(columns=['Class'])

    # test_normal.csv — 50 normal transactions
    drop(normal.sample(50, random_state=42)).to_csv(
        os.path.join(OUT_DIR, 'test_normal.csv'), index=False)
    print("✅ test_normal.csv saved (50 normal transactions)")

    # test_fraud.csv — 20 fraud transactions
    drop(fraud.sample(20, random_state=42)).to_csv(
        os.path.join(OUT_DIR, 'test_fraud.csv'), index=False)
    print("✅ test_fraud.csv saved (20 fraud transactions)")

    # test_mixed.csv — 40 normal + 10 fraud (realistic ratio)
    mixed = pd.concat([
        normal.sample(40, random_state=1),
        fraud.sample(10, random_state=1)
    ]).sample(frac=1, random_state=42)   # shuffle
    drop(mixed).to_csv(
        os.path.join(OUT_DIR, 'test_mixed.csv'), index=False)
    print("✅ test_mixed.csv saved (40 normal + 10 fraud, shuffled)")

    print("\nAll test files saved to data/ folder!")

if __name__ == '__main__':
    generate()
