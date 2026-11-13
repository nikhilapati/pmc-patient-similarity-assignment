import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def build_tfidf(cleaned_texts, min_df=2, max_df=0.5, sublinear_tf=True,
                smooth_idf=True, norm="l2"):
    """Fit scikit-learn's TfidfVectorizer. Returns (matrix, list of words)."""
    vectorizer = TfidfVectorizer(
        min_df=min_df,              # ignore words in fewer than 2 notes (typos, rare junk)
        max_df=max_df,              # ignore words in more than 50% of notes (too common)
        sublinear_tf=sublinear_tf,  # use 1 + log(count) instead of the raw count
        smooth_idf=smooth_idf,      # add 1 to avoid dividing by zero
        norm=norm,                  # make every row length 1, so cosine = dot product
        dtype=np.float32,           # half the memory of the default float64
    )
    matrix = vectorizer.fit_transform(cleaned_texts)
    vocabulary = vectorizer.get_feature_names_out().tolist()
    return matrix, vocabulary

def build_bm25(cleaned_texts, k1=1.5, b=0.75, min_df=2, max_df=0.5):
    """Build BM25 as two sparse matrices.

    doc_weights[d, t]  = BM25 weight of word t in patient d
    query_matrix[q, t] = 1 if word t appears in patient q, else 0
    BM25 score(q, d)   = query_matrix[q] . doc_weights[d]
    """
    # 1. Raw word counts (same vocabulary filters as TF-IDF, for a fair comparison)
    counter = CountVectorizer(min_df=min_df, max_df=max_df, dtype=np.float32)
    counts = counter.fit_transform(cleaned_texts).tocsr()
    n_docs = counts.shape[0]

    # 2. IDF: how rare each word is across all patients
    doc_freq = np.bincount(counts.indices, minlength=counts.shape[1])
    idf = np.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    # 3. Length normalization: each note's length compared with the average
    doc_lengths = np.asarray(counts.sum(axis=1)).ravel()
    length_norm = k1 * (1 - b + b * doc_lengths / doc_lengths.mean())

    # 4. Apply the BM25 formula to every non-zero (patient, word) cell
    doc_weights = counts.copy()
    row_of_each_value = np.repeat(np.arange(n_docs), np.diff(counts.indptr))
    tf = doc_weights.data
    doc_weights.data = idf[doc_weights.indices] * tf * (k1 + 1) / (tf + length_norm[row_of_each_value])

    # 5. Query matrix: 1 wherever a word appears
    query_matrix = counts.copy()
    query_matrix.data[:] = 1.0

    return doc_weights.astype(np.float32), query_matrix.astype(np.float32)