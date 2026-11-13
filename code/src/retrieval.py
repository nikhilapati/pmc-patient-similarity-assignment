"""For every patient, rank all other patients and keep the top k."""
import numpy as np


def top_k_similar(query_matrix, doc_matrix, k=5, batch_size=500):
    """Return an (n_patients, k) array of row numbers of the most similar patients.

    Scores = query_matrix @ doc_matrix.T, computed batch_size patients at a time
    so the full n x n table never sits in memory. A patient never matches itself.
    """
    n_patients = query_matrix.shape[0]
    doc_matrix_t = doc_matrix.T.tocsr()
    results = np.zeros((n_patients, k), dtype=np.int64)

    for start in range(0, n_patients, batch_size):
        end = min(start + batch_size, n_patients)

        # 1. score this batch against everyone
        scores = (query_matrix[start:end] @ doc_matrix_t).toarray()

        # 2. a patient can't be similar to itself
        rows = np.arange(end - start)
        scores[rows, np.arange(start, end)] = -np.inf

        # 3. grab the top k quickly (unordered), then sort those k best -> worst
        candidates = np.argpartition(-scores, k, axis=1)[:, :k]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        results[start:end] = np.take_along_axis(candidates, order, axis=1)

        print(f"    ranked {end}/{n_patients}", end="\r")
    print()
    return results


def indices_to_uids(top_indices, uids):
    """Convert row numbers back into patient_uid strings."""
    uids = np.asarray(uids)
    return [uids[row].tolist() for row in top_indices]