from preprocess import preprocess_wmt14
from model import build_seq2seq_model
import tensorflow as tf
import numpy as np

# Load WMT14 Data (General Dataset)
language_pair = ("fr", "en")
wmt_data = preprocess_wmt14(language_pair, max_samples=10000)

# Load Domain-Specific Data (e.g., Medical Domain)
domain_data = preprocess_wmt14(language_pair, max_samples=2000, tokenizer_prefix="domain_specific")

# Combine Datasets
node_features = [graph.x for graph in wmt_data["source_graphs"]] + \
                [graph.x for graph in domain_data["source_graphs"]]
adj_matrices = [graph.a for graph in wmt_data["source_graphs"]] + \
                [graph.a for graph in domain_data["source_graphs"]]
decoder_inputs = np.array(wmt_data["source_sequences"].tolist() + domain_data["source_sequences"].tolist())
decoder_targets = np.array(wmt_data["target_sequences"].tolist() + domain_data["target_sequences"].tolist())

# Build Model
vocab_size = 8000
model = build_seq2seq_model(vocab_size)

# Compile Model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train on General Data
model.fit(
    [node_features, adj_matrices, decoder_inputs],
    decoder_targets,
    epochs=5,
    batch_size=32
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
