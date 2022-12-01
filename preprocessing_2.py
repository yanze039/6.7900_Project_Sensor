import gpn.mlm
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import re
import os
import json

"""
1. Convert csv file to JSON files.
2. Generate the embedding for Seq.
"""


# model_path = "gonzalobenegas/gpn-arabidopsis"

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModel.from_pretrained(model_path)
# model.eval()

df = pd.read_csv("data/intermediate_07-Aug-2020.csv")
n_rows = df.shape[0]

dna_mapping = {}
json_data = {}
one_hot_encoder = OneHotEncoder(sparse = False)
one_hot_encoder.fit([['A'],['T'],['C'],['G']])

def string_to_array(seq_string):
   seq_string = re.sub('[^ACGT]', 'n', seq_string)
   seq_string = np.array(list(seq_string))
   return seq_string.reshape(-1,1)

for rr in range(n_rows):
    # print(rr)
    data = {}
    sub = df.iloc[rr]
    seq = sub['DNA']
    if not (seq in dna_mapping):
        current_index = len(dna_mapping)
        dna_mapping[seq] = current_index
        embedding = one_hot_encoder.transform(string_to_array(seq))
        print(embedding.shape)
        np.save(f"data/dna_onehot/{current_index:d}.npy", embedding)
    data["seq"] = str(seq)
    data["ph"] = int(sub["pH"])
    data["analyte"] = sub["Analyte"]
    data["shape_term1"] = float(sub["shape_term1"])
    data["shape_term2"] = float(sub["shape_term2"])
    json_data[f"{dna_mapping[seq]}_{sub['pH']}_{sub['Analyte']}"] = data

with open("data/data.json", 'w') as fp:
    json.dump(json_data, fp, indent=4)
