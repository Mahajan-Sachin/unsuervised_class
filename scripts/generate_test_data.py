"""
Run once to generate sample test CSV files from creditcard.csv
Usage: python scripts/generate_test_data.py
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / 'data'


def generate():
    print("Loading creditcard.csv...")
    df = pd.read_csv(DATA_DIR / 'creditcard.csv')

    normal = df[df['Class'] == 0]
    fraud  = df[df['Class'] == 1]

    def save(data, name):
        data.drop(columns=['Class']).to_csv(DATA_DIR / name, index=False)
        print(f"✅ {name} saved ({len(data)} rows)")

    save(normal.sample(50, random_state=42),                           'test_normal.csv')
    save(fraud.sample(20, random_state=42),                            'test_fraud.csv')
    save(pd.concat([normal.sample(40, random_state=1),
                    fraud.sample(10,  random_state=1)
                   ]).sample(frac=1, random_state=42),                 'test_mixed.csv')

    print("\nAll test files saved to data/")


if __name__ == '__main__':
    generate()
