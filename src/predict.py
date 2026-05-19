from pathlib import Path
import numpy as np
import joblib
import tensorflow as tf

MODELS_DIR = Path(__file__).parent.parent / 'models'

_cache = {}


def _load():
    if not _cache:
        _cache['iso']       = joblib.load(MODELS_DIR / 'isolation_forest.pkl')
        _cache['scaler']    = joblib.load(MODELS_DIR / 'scaler.pkl')
        _cache['ae']        = tf.keras.models.load_model(MODELS_DIR / 'autoencoder.h5')
        _cache['threshold'] = joblib.load(MODELS_DIR / 'threshold.pkl')


def predict_anomaly(features: list) -> dict:
    _load()

    X        = np.array(features, dtype=np.float32).reshape(1, -1)
    X_scaled = _cache['scaler'].transform(X)

    # Isolation Forest
    iso_label  = _cache['iso'].predict(X_scaled)[0]   # 1=normal, -1=anomaly

    # Autoencoder
    recon      = _cache['ae'].predict(X_scaled, verbose=0)
    mse        = float(np.mean(np.power(X_scaled - recon, 2)))
    ae_anomaly = mse > _cache['threshold']

    is_anomaly = (iso_label == -1) or ae_anomaly

    return {
        "is_anomaly":       bool(is_anomaly),
        "isolation_forest": "Anomaly" if iso_label == -1 else "Normal",
        "autoencoder":      "Anomaly" if ae_anomaly else "Normal",
    }
