from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

def evaluate_bleu(target_sequences, predicted_sequences):
    """
    Calculate BLEU scores for a set of predictions.
    """
    scores = []
    for reference, prediction in zip(target_sequences, predicted_sequences):
        scores.append(sentence_bleu([reference.split()], prediction.split()))
    return sum(scores) / len(scores)

def evaluate_meteor(target_sequences, predicted_sequences):
    """
    Calculate METEOR scores for a set of predictions.
    """
    scores = []
    for reference, prediction in zip(target_sequences, predicted_sequences):
        scores.append(meteor_score([reference], prediction))
    return sum(scores) / len(scores)

# Example Usage:
# bleu_score = evaluate_bleu(["Je suis content"], ["Je suis heureux"])
# meteor = evaluate_meteor(["Je suis content"], ["Je suis heureux"])
