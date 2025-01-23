# Language Translation

This project demonstrates a **graph-based approach to neural machine translation (NMT)** using **Spektral**, a library for graph neural networks. Sentences are represented as dependency graphs where nodes correspond to tokens (words), and edges represent syntactic or semantic relationships. The pipeline integrates a **Graph Neural Network (GNN)** encoder with a sequence-to-sequence decoder to translate English sentences into French using the **WMT14 dataset**.

---

## **Key Features**

1. **Graph Representation**: Sentences are converted into graphs using dependency parsing, with words as nodes and syntactic relationships as edges.
2. **Graph Neural Networks**: Leverages GNNs (e.g., Graph Attention Networks) to encode graph structures.
3. **Seq2Seq Decoder**: A GRU-based decoder generates translations from graph embeddings.
4. **Evaluation Metrics**: Includes BLEU and METEOR scores for translation quality.
5. **Attention Visualization**: Visualizes the model's attention to interpret translation results.
6. **Fine-Tuning**: Supports fine-tuning on domain-specific datasets for improved specialized translations.

---

## **Directory Structure**

```plaintext
language-translation/
├── data/
│   ├── train.csv               # WMT14 training dataset
│   ├── test.csv                # WMT14 test dataset
├── src/
│   ├── preprocess.py           # Preprocessing, graph creation, tokenization
│   ├── encoder.py              # GNN encoder implementation
│   ├── model.py                # Combined GNN-Seq2Seq model
│   ├── train.py                # Training script
│   ├── infer.py                # Inference script
│   ├── evaluate.py             # BLEU and METEOR evaluation
│   ├── utils.py                # Utilities (e.g., positional encodings)
├── models/
│   ├── wmt14.model             # SentencePiece tokenizer model
│   ├── fine_tuned_model.h5     # Trained model weights
├── README.md                   # Project description
```

---

## **Setup Instructions**

### **1. Clone the Repository**

```bash
git clone https://github.com/sarvex/language-translation
cd language-translation
```

### **2. Install Dependencies**

Install the required Python libraries:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### **3. Train the SentencePiece Tokenizer**

Train the tokenizer using the WMT14 dataset:

```bash
python src/preprocess.py
```

### **4. Train the Model**

Train the GNN-Seq2Seq model:

```bash
python src/train.py
```

### **5. Translate a Sentence**

Use the trained model to translate an English sentence into French:

```bash
python src/infer.py
```

---

## **Detailed Workflow**

### **1. Preprocessing**
- Sentences are parsed using spaCy to generate dependency graphs.
- Tokenization is performed using SentencePiece for subword-level modeling.
- Graph representations (`Graph` objects from Spektral) and tokenized sequences are saved for training.

### **2. Graph Neural Network Encoder**
- A GNN encoder (using `GATConv`) processes graph-structured data.
- Attention mechanisms within GNN layers focus on important tokens based on their syntactic roles.

### **3. Seq2Seq Decoder**
- A GRU-based decoder generates target sequences.
- Additive attention is applied between graph embeddings and decoder states.

### **4. Fine-Tuning**
- The model is pre-trained on the general WMT14 dataset and fine-tuned on smaller domain-specific datasets for better performance in specialized contexts.

### **5. Evaluation**
- BLEU and METEOR scores are computed for test translations.

### **6. Attention Visualization**
- Attention weights between source tokens and generated target tokens are visualized for interpretability.

---

## **Evaluation Metrics**

### **1. BLEU (Bilingual Evaluation Understudy)**
Measures n-gram overlap between generated and reference translations.

### **2. METEOR (Metric for Evaluation of Translation with Explicit ORdering)**
Aligns words and phrases to measure semantic similarity.

---

## **Sample Code Usage**

### **1. Translate a Sentence**

```python
from src.infer import translate
from src.seq2seq_model import build_seq2seq_model
import sentencepiece as spm

# Load model and tokenizer
model = build_seq2seq_model(vocab_size=8000)
model.load_weights("models/fine_tuned_model.h5")
tokenizer = spm.SentencePieceProcessor(model_file="models/wmt14.model")

# Translate a sentence
source_sentence = "How are you today?"
translation = translate(source_sentence, model, tokenizer)
print("Translation:", translation)
```

### **2. Evaluate BLEU and METEOR Scores**

```python
from src.evaluate import evaluate_bleu, evaluate_meteor

# Example data
target_sequences = ["Je suis heureux"]
predicted_sequences = ["Je suis content"]

# Evaluate
bleu_score = evaluate_bleu(target_sequences, predicted_sequences)
meteor_score = evaluate_meteor(target_sequences, predicted_sequences)

print(f"BLEU Score: {bleu_score}")
print(f"METEOR Score: {meteor_score}")
```

### **3. Visualize Attention**

```python
from src.infer import visualize_attention

# Visualize attention
visualize_attention("How are you today?", model, tokenizer)
```

---

## **Future Enhancements**

1. **Multilingual Translation**: Extend the pipeline to handle translations for multiple language pairs.
2. **Transformer Decoder**: Replace the GRU decoder with a Transformer for better performance.
3. **Larger Datasets**: Use additional datasets like CCMatrix for richer translations.
4. **Dynamic Graph Construction**: Create semantic graphs based on context rather than fixed dependency parsing.

---

## **License**

This project is licensed under the MIT License. See the LICENSE file for details.

---

## **Contributors**

- **Your Name**: [Sarvex](https://github.com/sarvex)

