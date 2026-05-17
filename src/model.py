import tensorflow as tf


def build_autoencoder(input_dim=30):
    """
    Autoencoder — unsupervised deep learning.
    Trained only on normal data → high reconstruction error = anomaly.
    """
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(16, activation='relu')(inputs)
    x = tf.keras.layers.Dense(8,  activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(input_dim, activation='linear')(x)
    model = tf.keras.Model(inputs, outputs, name='autoencoder')
    model.compile(optimizer='adam', loss='mse')
    return model
