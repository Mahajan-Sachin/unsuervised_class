import tensorflow as tf


def build_autoencoder(input_dim=30):
    """
    Autoencoder — unsupervised deep learning.
    Trained only on normal data → high reconstruction error = anomaly.
    Uses Sequential API: clean pipeline-style, no branching needed.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_dim=input_dim),  # Encoder
        tf.keras.layers.Dense(8,  activation='relu'),                             # Bottleneck
        tf.keras.layers.Dense(16, activation='relu'),                             # Decoder
        tf.keras.layers.Dense(input_dim, activation='linear'),                   # Reconstruction
    ], name='autoencoder')

    model.compile(optimizer='adam', loss='mse')
    return model
