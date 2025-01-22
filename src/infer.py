from evaluate import evaluate_bleu, evaluate_meteor
from preprocess import preprocess_wmt14
from seq2seq_model import build_seq2seq_model

# Load Test Data
test_data = preprocess_wmt14(("en", "fr"), max_samples=1000, tokenizer_prefix="wmt14_test")

# Load Model and Tokenizer
model = build_seq2seq_model(vocab_size=8000)
model.load_weights("fine_tuned_model.h5")
tokenizer = spm.SentencePieceProcessor(model_file="wmt14.model")

# Generate Predictions
predicted_sequences = []
target_sequences = test_data["target_sequences"]
for source_graph in test_data["source_graphs"]:
    predicted_sequences.append(translate(source_graph, model, tokenizer))

# Evaluate
bleu_score = evaluate_bleu(target_sequences, predicted_sequences)
meteor = evaluate_meteor(target_sequences, predicted_sequences)
print(f"BLEU Score: {bleu_score}")
print(f"METEOR Score: {meteor}")
