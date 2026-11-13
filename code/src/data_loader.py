"""Reads the patient JSON files into a pandas DataFrame."""
import ast
import json
import os
import random

import pandas as pd


def parse_similar_patients(raw_value):
    """Turns the text "{'123-1': 2}" into a real Python dictionary."""
    if not raw_value:
        return {}
    return ast.literal_eval(raw_value)


def load_patients(data_folder, sample_size=None, seed=42):
    """Reads every .json file in data_folder (or a random sample of them)."""
    file_names = sorted(f for f in os.listdir(data_folder) if f.endswith(".json"))

    if sample_size:
        random.Random(seed).shuffle(file_names)
        file_names = sorted(file_names[:sample_size])

    records = []
    for file_name in file_names:
        with open(os.path.join(data_folder, file_name), encoding="utf-8") as f:
            record = json.load(f)
        records.append({
            "patient_uid": record["patient_uid"],
            "patient_id": str(record["patient_id"]),
            "text": record["patient"],
            "similar_patients": parse_similar_patients(record.get("similar_patients")),
        })
    return pd.DataFrame(records)