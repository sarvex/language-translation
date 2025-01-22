import numpy as np
import tensorflow as tf

def positional_encoding(position, d_model):
    """
    Create sinusoidal positional encodings.
    """
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # Apply sin to even indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # Apply cos to odd indices
    return tf.constant(angle_rads, dtype=tf.float32)
