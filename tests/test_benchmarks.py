import re
from typing import Tuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Normalization Function ---

def normalize_answer(answer: str) -> str:
    """
    Normalize text by converting to lowercase, removing punctuation, articles, and extra whitespace.
    """
    answer = answer.lower().strip()
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)  # Remove common articles
    answer = re.sub(r'[^a-z0-9\s]', '', answer)      # Remove punctuation
    answer = re.sub(r'\s+', ' ', answer)             # Remove extra whitespace
    return answer.strip()

# --- Exact Match and F1 Score for QA ---

def exact_match_score(prediction: str, ground_truth: str) -> int:
    """
    Calculate Exact Match score for QA. Returns 1 if prediction matches ground truth, else 0.
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Calculate F1 score for QA based on token overlap between prediction and ground truth.
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    num_common = sum(min(pred_tokens.count(token), truth_tokens.count(token)) for token in common_tokens)
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)

# --- ROUGE and BLEU Scoring for Summarization ---

def calculate_bleu(reference: list, candidate: list) -> float:
    """
    Calculate BLEU score between reference and candidate summaries.
    """
    # Apply smoothing to BLEU score calculation
    smoothie = SmoothingFunction().method1  # You can use other methods too
    return sentence_bleu([reference], candidate, smoothing_function=smoothie)

def calculate_rouge(reference: str, candidate: str) -> dict:
    """
    Calculate ROUGE-1 and ROUGE-L scores between reference and candidate summaries.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, candidate)

# --- Accuracy, Precision, Recall, and F1 for Classification ---

def classification_metrics(y_true: list, y_pred: list) -> Tuple[float, float, float, float]:
    """
    Calculate accuracy, precision, recall, and F1 for classification tasks (e.g., sentiment analysis).
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0) # average='weighted' could be used
    return accuracy, precision, recall, f1

# main function for model evaluations
def evaluate_model(prediction, reference, task_type):
    """
    main function for model evaluations
    """
    if task_type.lower() == 'qa':
        em_score = exact_match_score(prediction, reference)
        f1 = f1_score(prediction, reference)
        print(f"Exact Match Score: {em_score}")
        print(f"F1 Score: {f1:.2f}")
        return em_score, f1
    
    elif task_type.lower() == 'summarization':
        prediction_list = prediction.strip().split(',')
        reference_list = reference.strip().split(',')
        bleu = calculate_bleu(reference_list, prediction_list)
        rouge_scores = calculate_rouge(prediction, reference)
        print(f"BLEU Score: {bleu:.2f}")
        print(f"ROUGE Scores: {rouge_scores}")
        return bleu, rouge_scorer
    
    elif task_type.lower() == 'sentiment' or task_type.lower() == 'classification':
        accuracy, precision, recall, f1_class = classification_metrics(reference, prediction)
        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1_class:.2f}")
        return accuracy, precision, recall, f1_class

# --- Example Usage ---

# prediction = "Paris, France"
# reference = "The Eiffel Tower is located in Paris, France."
prediction = ["Positive", "Negative"]
reference = ["Positive", "Positive"]
evaluate_model(prediction, reference, 'sentiment')