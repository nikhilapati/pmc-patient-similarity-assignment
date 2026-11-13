"""Save and load the output files."""
import json
import os

from scipy import sparse


def save_json(obj, path):
    """Save any dictionary/list as a .json file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def save_tfidf(matrix, uids, vocabulary, output_folder):
    """Save TF-IDF as a sparse matrix + its row labels (uids) and column labels (terms)."""
    sparse.save_npz(os.path.join(output_folder, "tfidf_matrix.npz"), matrix.tocsr())
    save_json({"patient_uids": list(uids), "terms": list(vocabulary)},
              os.path.join(output_folder, "tfidf_labels.json"))


class TfidfLookup:
    """Load the saved TF-IDF files and read one weight.

    Example:
        lookup = TfidfLookup("outputs")
        lookup.weight("1634757-1", "nodule")
    """

    def __init__(self, output_folder):
        self.matrix = sparse.load_npz(os.path.join(output_folder, "tfidf_matrix.npz")).tocsr()
        with open(os.path.join(output_folder, "tfidf_labels.json"), encoding="utf-8") as f:
            labels = json.load(f)
        self.row_of = {uid: i for i, uid in enumerate(labels["patient_uids"])}
        self.col_of = {term: j for j, term in enumerate(labels["terms"])}

    def weight(self, patient_uid, term):
        """TF-IDF weight of `term` in `patient_uid` (0.0 if the word isn't in the vocabulary)."""
        if term not in self.col_of:
            return 0.0
        return float(self.matrix[self.row_of[patient_uid], self.col_of[term]])