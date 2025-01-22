import tensorflow as tf
from spektral.layers import GATConv

class GNNEncoder(tf.keras.Model):
    def __init__(self, hidden_dim, attention_heads=4):
        super().__init__()
        self.gat1 = GATConv(hidden_dim, attn_heads=attention_heads, activation="relu")
        self.gat2 = GATConv(hidden_dim, attn_heads=attention_heads, activation="relu")
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs):
        x, a = inputs
        x = self.gat1([x, a])
        x = self.gat2([x, a])
        return self.global_pool(x)  # Return graph-level embedding
