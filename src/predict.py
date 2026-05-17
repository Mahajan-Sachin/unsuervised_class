import os
import numpy as np
import joblib
import tensorflow as tf

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

_cache = {}


def _load():
    if not _cache:
        _cache['iso']       = joblib.load(os.path.join(MODELS_DIR, 'isolation_forest.pkl'))
        _cache['lof']       = joblib.load(os.path.join(MODELS_DIR, 'lof.pkl'))
        _cache['scaler']    = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        _cache['ae']        = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'autoencoder.h5'))
        _cache['threshold'] = joblib.load(os.path.join(MODELS_DIR, 'threshold.pkl'))


def predict_anomaly(features: list) -> dict:
    """
    3-model ensemble: Isolation Forest + LOF + Autoencoder.
    Flags as anomaly if ANY model detects it.
    """
    _load()

    X        = np.array(features, dtype=np.float32).reshape(1, -1)
    X_scaled = _cache['scaler'].transform(X)

    # Isolation Forest
    iso_label = _cache['iso'].predict(X_scaled)[0]
    iso_score = float(_cache['iso'].score_samples(X_scaled)[0])

    # LOF
    lof_label = _cache['lof'].predict(X_scaled)[0]
    lof_score = float(_cache['lof'].score_samples(X_scaled)[0])

    # Autoencoder
    recon     = _cache['ae'].predict(X_scaled, verbose=0)
    mse       = float(np.mean(np.power(X_scaled - recon, 2)))
    ae_anomaly = mse > _cache['threshold']

    # Final: fraud if ANY model flags it
    is_anomaly = (iso_label == -1) or (lof_label == -1) or ae_anomaly

    return {
        "result":           "Anomaly" if is_anomaly else "Normal",
        "is_anomaly":       bool(is_anomaly),
        "isolation_forest": "Anomaly" if iso_label == -1 else "Normal",
        "lof":              "Anomaly" if lof_label == -1 else "Normal",
        "autoencoder":      "Anomaly" if ae_anomaly else "Normal",
        "iso_score":        round(iso_score, 6),
        "lof_score":        round(lof_score, 6),
        "recon_error":      round(mse, 6),
        "threshold":        round(_cache['threshold'], 6),
    }
