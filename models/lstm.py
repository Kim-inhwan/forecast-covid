import tensorflow as tf


def get_model(units, label_width, dropout):
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(label_width)
    ])

    model.compile(loss='mse', optimizer='adam')
    return model