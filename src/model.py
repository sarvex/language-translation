from tensorflow.keras.layers import Input, Embedding, GRU, Dense, AdditiveAttention
from tensorflow.keras.models import Model
from encoder import GNNEncoder
from utils import positional_encoding

def build_seq2seq_model(vocab_size, embed_dim=128, hidden_dim=256, max_length=100):
    # GNN Encoder
    gnn_inputs = [
        Input(shape=(None, embed_dim), name="node_features"),  # Node features
        Input(shape=(None, None), name="adjacency_matrix"),   # Adjacency matrix
    ]
    gnn_encoder = GNNEncoder(hidden_dim)
    encoded_graph = gnn_encoder(gnn_inputs)

    # Add positional encodings to graph embeddings
    pos_encodings = positional_encoding(max_length, hidden_dim)
    encoded_graph += pos_encodings

    # Decoder
    decoder_inputs = Input(shape=(None,), name="decoder_inputs")
    decoder_embedding = Embedding(vocab_size, embed_dim)(decoder_inputs)
    
    # Attention Mechanism
    attention = AdditiveAttention()
    context_vector = attention([decoder_embedding, encoded_graph])

    # Decoder GRU
    decoder_gru = GRU(hidden_dim, return_sequences=True, return_state=True)
    decoder_outputs, _ = decoder_gru(context_vector)

    # Output layer
    decoder_dense = Dense(vocab_size, activation="softmax")
    outputs = decoder_dense(decoder_outputs)

    return Model(inputs=gnn_inputs + [decoder_inputs], outputs=outputs)
