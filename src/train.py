from preprocess import preprocess_wmt14
from model import build_seq2seq_model
import tensorflow as tf
import numpy as np

# Load WMT14 Data (General Dataset)
language_pair = ("fr", "en")
wmt_data = preprocess_wmt14(language_pair, max_samples=10000)

# Load Domain-Specific Data (e.g., Medical Domain)
domain_data = preprocess_wmt14(language_pair, max_samples=2000, tokenizer_prefix="domain_specific")

# Find maximum dimensions for padding
max_nodes = max(
    max(graph.x.shape[0] for graph in wmt_data["source_graphs"]),
    max(graph.x.shape[0] for graph in domain_data["source_graphs"])
)
feature_dim = wmt_data["source_graphs"][0].x.shape[1]

print("Max nodes:", max_nodes)
print("Feature dimension:", feature_dim)

# Pad node features and adjacency matrices
def pad_graph(graph, max_nodes, feature_dim):
    n_nodes = graph.x.shape[0]
    padded_x = np.pad(
        graph.x,
        ((0, max_nodes - n_nodes), (0, 0)),
        mode='constant'
    )
    padded_a = np.pad(
        graph.a,
        ((0, max_nodes - n_nodes), (0, max_nodes - n_nodes)),
        mode='constant'
    )
    return padded_x, padded_a

# Combine and pad datasets
node_features = []
adj_matrices = []
for graph in wmt_data["source_graphs"] + domain_data["source_graphs"]:
    x_padded, a_padded = pad_graph(graph, max_nodes, feature_dim)
    node_features.append(x_padded)
    adj_matrices.append(a_padded)

# Convert to numpy arrays with explicit types
node_features = np.array(node_features, dtype=np.float32)
adj_matrices = np.array(adj_matrices, dtype=np.float32)
decoder_inputs = np.array(wmt_data["source_sequences"].tolist() + domain_data["source_sequences"].tolist(), dtype=np.int32)
decoder_targets = np.array(wmt_data["target_sequences"].tolist() + domain_data["target_sequences"].tolist(), dtype=np.int32)

print("\nArray shapes:")
print("node_features shape:", node_features.shape)
print("adj_matrices shape:", adj_matrices.shape)
print("decoder_inputs shape:", decoder_inputs.shape)
print("decoder_targets shape:", decoder_targets.shape)

# Build Model with matching dimensions
vocab_size = 8000
embed_dim = feature_dim  # Match the feature dimension
max_length = max(max_nodes, decoder_inputs.shape[1])  # Use the larger of the two maximum lengths
model = build_seq2seq_model(vocab_size, embed_dim=embed_dim, max_length=max_length)

# Compile Model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train on Combined Data
history = model.fit(
    {"node_features": node_features, 
     "adjacency_matrix": adj_matrices, 
     "decoder_inputs": decoder_inputs},
    decoder_targets,
    batch_size=32,
    epochs=10,
    validation_split=0.2
)

# Fine-Tune on Domain-Specific Data
fine_tune_features = [graph.x for graph in domain_data["source_graphs"]]
fine_tune_adj = [graph.a for graph in domain_data["source_graphs"]]
fine_tune_inputs = np.array(domain_data["source_sequences"].tolist())
fine_tune_targets = np.array(domain_data["target_sequences"].tolist())

model.fit(
    [fine_tune_features, fine_tune_adj, fine_tune_inputs],
    fine_tune_targets,
    epochs=3,
    batch_size=16
)

# Save Fine-Tuned Model
model.save_weights("fine_tuned_model.h5")
