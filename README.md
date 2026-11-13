# Patient Similarity Retrieval on PMC-Patients (HI 744 Programming Assignment)

Finds the 5 most similar patients for every patient in the PMC-Patients dataset
(167,034 patient case descriptions from PubMed Central) using two retrieval methods:

- **TF-IDF (Term Frequency – Inverse Document Frequency) + cosine similarity**: the baseline
- **BM25 (Best Matching 25)**: a keyword-ranking function used by search engines

Both methods are evaluated with **precision@5** and **recall@5** against the
`similar_patients` field in the data.

## Repository layout
```
code/
  main.py                      entry point, runs the whole pipeline
  requirements.txt             Python packages needed
  src/
    data_loader.py             read the patient .json files
    preprocess.py              lowercase, remove punctuation, stop words, lemmatize
    representations.py         build the TF-IDF and BM25 matrices
    retrieval.py               rank patients and return the top 5
    evaluation.py              precision@5, recall@5, overlap between methods
    io_utils.py                save/load outputs, TF-IDF lookup helper
outputs/
  tfidf_matrix.npz             Task 1: TF-IDF matrix (sparse, stored with Git LFS)
  tfidf_labels.json            Task 1: row labels (patient_uids) and column labels (terms)
  top5_similar_patients.json   Task 2: top 5 similar patients per method
  performance.json             Task 3: precision@5 and recall@5 per patient per method
  summary.json                 averaged results used in the report
report.pdf
```

## How to run
```
pip install -r code/requirements.txt
python code/main.py PATH_TO_INPUT_DATA_FOLDER
```
All output files are written to `outputs/`.

**Expected run time:** about 32 minutes on the full dataset (167,034 patients) on a
MacBook Air. The program prints its progress (`ranked N/167034`).
For a quick check, add `--sample_size 5000`; this finishes in a few seconds.

NLTK stop words and WordNet are downloaded automatically on the first run.

## Results (full dataset, 167,034 patients)

| Method | Average precision@5 | Average recall@5 |
|---|---|---|
| TF-IDF + cosine similarity | 0.133 | 0.407 |
| BM25 | 0.121 | 0.377 |

- Patients excluded from recall (no similar patients listed): 62,570
- Average overlap between the two methods' top 5 lists: 2.5 patients

## Loading the outputs
Run Python from inside the `code/` folder:
```python
import json
from src.io_utils import TfidfLookup

# Task 1: TF-IDF weight of a term for a patient
lookup = TfidfLookup("../outputs")
lookup.weight("1634757-1", "nodule")

# Task 2: top 5 similar patients per method
top5 = json.load(open("../outputs/top5_similar_patients.json"))
top5["1634757-1"]["tfidf"]

# Task 3: precision@5 and recall@5 per method
perf = json.load(open("../outputs/performance.json"))
perf["1634757-1"]["bm25"]
```

## Note on the large file
`outputs/tfidf_matrix.npz` (~150 MB) is stored with **Git LFS (Large File Storage)**
because GitHub blocks files over 100 MB. Clone with Git LFS installed, or run
`git lfs pull` after cloning, to download the real file.