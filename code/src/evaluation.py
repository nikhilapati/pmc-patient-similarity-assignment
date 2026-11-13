"""Precision@k and recall@k against the similar_patients answer key."""
import numpy as np


def precision_at_k(retrieved, relevant, k=5):
    """Share of the top k results that are in the answer key."""
    hits = sum(1 for uid in retrieved[:k] if uid in relevant)
    return hits / k


def recall_at_k(retrieved, relevant, k=5):
    """Share of the answer key found in the top k. None if the key is empty."""
    if not relevant:
        return None
    hits = sum(1 for uid in retrieved[:k] if uid in relevant)
    return hits / len(relevant)


def evaluate_all(df, retrieval_results, methods, k=5):
    """Per-patient scores: {uid: {"patient_id": ..., method: {"precision_5", "recall_5"}}}."""
    scores = {}
    for row in df.itertuples():
        relevant = set(row.similar_patients)   # labels 1 and 2 both count as relevant
        entry = {"patient_id": row.patient_id}
        for method in methods:
            retrieved = retrieval_results[row.patient_uid][method]
            entry[method] = {
                f"precision_{k}": precision_at_k(retrieved, relevant, k),
                f"recall_{k}": recall_at_k(retrieved, relevant, k),
            }
        scores[row.patient_uid] = entry
    return scores


def summarize(scores, methods, k=5):
    """Average precision over all patients; recall only over patients with an answer key."""
    summary = {}
    for method in methods:
        precisions = [s[method][f"precision_{k}"] for s in scores.values()]
        recalls = [s[method][f"recall_{k}"] for s in scores.values()
                   if s[method][f"recall_{k}"] is not None]
        summary[method] = {
            f"avg_precision_{k}": float(np.mean(precisions)),
            f"avg_recall_{k}": float(np.mean(recalls)),
            "patients_excluded_from_recall": len(precisions) - len(recalls),
        }
    return summary


def overlap_between_methods(retrieval_results, method_a, method_b):
    """Average number of patients (0-5) that two methods' top-5 lists share."""
    shared = [len(set(r[method_a]) & set(r[method_b])) for r in retrieval_results.values()]
    return float(np.mean(shared))