import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_autoencoder

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'creditcard.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────
    print("[1/5] Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape : {df.shape}")
    print(f"  Normal: {(df['Class']==0).sum()}  |  Fraud: {(df['Class']==1).sum()}\n")

    features = [c for c in df.columns if c != 'Class']
    X = df[features].values
    y = df['Class'].values

    # ── Scale ─────────────────────────────────────────────────
    print("[2/5] Scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

    # ── Isolation Forest ──────────────────────────────────────
    print("[3/5] Training Isolation Forest...")
    fraud_ratio = float(y.sum()) / len(y)
    iso = IsolationForest(contamination=fraud_ratio, random_state=42, n_jobs=-1)
    iso.fit(X_scaled)
    joblib.dump(iso, os.path.join(MODELS_DIR, 'isolation_forest.pkl'))
    print(f"  Done! contamination={fraud_ratio:.5f}")

    # ── LOF ───────────────────────────────────────────────────
    print("[4/5] Training LOF (novelty mode on 10k sample)...")
    idx = np.random.RandomState(42).choice(len(X_scaled), size=10000, replace=False)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=fraud_ratio, novelty=True)
    lof.fit(X_scaled[idx])
    joblib.dump(lof, os.path.join(MODELS_DIR, 'lof.pkl'))
    print("  Done!")

    # ── Autoencoder ───────────────────────────────────────────
    print("[5/5] Training Autoencoder on normal transactions only...")
    X_normal = X_scaled[y == 0]
    ae = build_autoencoder(input_dim=X_scaled.shape[1])
    ae.fit(X_normal, X_normal, epochs=20, batch_size=256,
           validation_split=0.1, verbose=1)
    ae.save(os.path.join(MODELS_DIR, 'autoencoder.h5'))

    recon = ae.predict(X_normal, verbose=0)
    mse   = np.mean(np.power(X_normal - recon, 2), axis=1)
    threshold = float(np.percentile(mse, 95))
    joblib.dump(threshold, os.path.join(MODELS_DIR, 'threshold.pkl'))
    print(f"  Done! Threshold={threshold:.6f}")

    print("\n✅ All 3 models saved to /models")


if __name__ == '__main__':
    train()
