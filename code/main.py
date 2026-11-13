"""Patient similarity retrieval on PMC-Patients: TF-IDF vs BM25.

Usage:
    python main.py PATH_TO_INPUT_DATA_FOLDER
    python main.py PATH_TO_INPUT_DATA_FOLDER --sample_size 5000   (quick test)
"""
import argparse
import os
import time

from src.data_loader import load_patients
from src.evaluation import evaluate_all, overlap_between_methods, summarize
from src.io_utils import save_json, save_tfidf
from src.preprocess import preprocess_corpus
from src.representations import build_bm25, build_tfidf
from src.retrieval import indices_to_uids, top_k_similar

TOP_K = 5
METHODS = ["tfidf", "bm25"]
DEFAULT_OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def parse_args():
    """Read the command-line arguments."""
    parser = argparse.ArgumentParser(description="TF-IDF vs BM25 patient similarity retrieval")
    parser.add_argument("data_folder", help="Folder containing the patient .json files")
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Use a random subset of patients (same subset for all methods)")
    return parser.parse_args()


def run_retrieval(df, tfidf_matrix, bm25_docs, bm25_queries):
    """Task 2: top-5 similar patients per method, keyed by patient_uid."""
    uids = df["patient_uid"].tolist()

    print("  TF-IDF + cosine similarity...")
    tfidf_top = indices_to_uids(top_k_similar(tfidf_matrix, tfidf_matrix, TOP_K), uids)

    print("  BM25...")
    bm25_top = indices_to_uids(top_k_similar(bm25_queries, bm25_docs, TOP_K), uids)

    return {
        uid: {"patient_id": pid, "tfidf": tfidf_top[i], "bm25": bm25_top[i]}
        for i, (uid, pid) in enumerate(zip(uids, df["patient_id"]))
    }


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    start = time.time()

    print("Loading data...")
    df = load_patients(args.data_folder, sample_size=args.sample_size)
    print(f"  {len(df)} patients loaded")

    print("Pre-processing text...")
    cleaned = preprocess_corpus(df["text"].tolist())

    print("Task 1: building TF-IDF...")
    tfidf_matrix, vocabulary = build_tfidf(cleaned)
    save_tfidf(tfidf_matrix, df["patient_uid"], vocabulary, args.output_folder)
    print(f"  TF-IDF shape: {tfidf_matrix.shape}")

    print("Building BM25...")
    bm25_docs, bm25_queries = build_bm25(cleaned)

    print("Task 2: retrieving top 5 similar patients...")
    results = run_retrieval(df, tfidf_matrix, bm25_docs, bm25_queries)
    save_json(results, os.path.join(args.output_folder, "top5_similar_patients.json"))

    print("Task 3: evaluating...")
    scores = evaluate_all(df, results, METHODS, TOP_K)
    save_json(scores, os.path.join(args.output_folder, "performance.json"))

    summary = summarize(scores, METHODS, TOP_K)
    summary["avg_overlap_tfidf_bm25"] = overlap_between_methods(results, "tfidf", "bm25")
    summary["n_patients"] = len(df)
    save_json(summary, os.path.join(args.output_folder, "summary.json"))

    print("Summary:", summary)
    print(f"Done in {time.time() - start:.0f} seconds. Files saved to {args.output_folder}")


if __name__ == "__main__":
    main()