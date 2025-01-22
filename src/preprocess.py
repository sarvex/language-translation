import spacy
import numpy as np
import pandas as pd
from spektral.data import Graph
from datasets import load_dataset
from spektral.data import Dataset
import sentencepiece as spm

nlp = spacy.load("en_core_web_sm")

def sentence_to_graph(sentence):
    """
    Converts a sentence into a graph where:
    - Nodes are tokens.
    - Edges are dependency relationships.
    """
    doc = nlp(sentence)
    nodes = [token.text for token in doc]
    edges = [(token.i, child.i) for token in doc for child in token.children]
    
    # Adjacency matrix
    adj = np.zeros((len(nodes), len(nodes)))
    for src, tgt in edges:
        adj[src, tgt] = 1
        adj[tgt, src] = 1  # Make it undirected

    # Node features (e.g., word embeddings)
    features = np.array([token.vector for token in doc])  # SpaCy's word vectors

    return Graph(x=features, a=adj)

def preprocess_wmt14(language_pair, max_samples=None, vocab_size=8000, tokenizer_prefix="wmt14"):
    """
    Load and preprocess the WMT14 dataset.
    Args:
        language_pair (tuple): The source and target language pair, e.g., ('en', 'fr').
        max_samples (int): Number of samples to load.
        vocab_size (int): Vocabulary size for SentencePiece tokenizer.
        tokenizer_prefix (str): Prefix for the tokenizer model.
    """
    dataset = load_dataset("wmt14", f"{language_pair[0]}-{language_pair[1]}")
    print("Dataset structure:", dataset)
    print("First example:", dataset["train"][0])
    
    # Limit samples if max_samples is specified
    if max_samples:
        dataset = dataset["train"].select(range(max_samples))
    else:
        dataset = dataset["train"]

    # Access translations using the 'translation' column
    source_texts = [example['translation'][language_pair[0]] for example in dataset]
    target_texts = [example['translation'][language_pair[1]] for example in dataset]

    # Train SentencePiece tokenizer on combined source and target texts
    combined_texts = source_texts + target_texts
    with open("combined_texts.txt", "w", encoding="utf-8") as f:
        for text in combined_texts:
            f.write(text + "\n")

    spm.SentencePieceTrainer.train(
        input="combined_texts.txt",
        model_prefix=tokenizer_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="unigram"
    )

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=f"{tokenizer_prefix}.model")

    # Create source and target sequences
    source_sequences = []
    target_sequences = []
    source_graphs = []
    
    # Process each sentence pair
    for src_text, tgt_text in zip(source_texts, target_texts):
        # Create graph for source sentence
        source_graph = sentence_to_graph(src_text)
        source_graphs.append(source_graph)
        
        # Tokenize source and target
        source_seq = sp.encode_as_ids(src_text)
        target_seq = sp.encode_as_ids(tgt_text)
        
        source_sequences.append(source_seq)
        target_sequences.append(target_seq)
    
    # Find maximum sequence length
    max_source_len = max(len(seq) for seq in source_sequences)
    max_target_len = max(len(seq) for seq in target_sequences)
    
    # Pad sequences
    padded_source_sequences = [
        seq + [0] * (max_source_len - len(seq)) for seq in source_sequences
    ]
    padded_target_sequences = [
        seq + [0] * (max_target_len - len(seq)) for seq in target_sequences
    ]
    
    return {
        "source_graphs": source_graphs,
        "source_sequences": np.array(padded_source_sequences),
        "target_sequences": np.array(padded_target_sequences),
        "tokenizer": sp
    }

# Example usage:
# wmt_data = preprocess_wmt14(("en", "fr"), max_samples=10000)
