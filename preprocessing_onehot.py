import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import re
import os
import json

"""
Generate onehot embedding of DNAs.
"""

df = pd.read_csv("data.csv")
n_rows = df.shape[0]

dna_mapping = {}
json_data = {}
one_hot_encoder = OneHotEncoder(sparse = False)
one_hot_encoder.fit([['A'],['T'],['C'],['G']])

def string_to_array(seq_string):
   seq_string = re.sub('[^ACGT]', 'n', seq_string)
   seq_string = np.array(list(seq_string))
   return seq_string.reshape(-1,1)

if not os.path.exists("data/dna_onehot"):
    os.makedirs("data/dna_onehot")

print("\nGenerating embeddings.\n")
for rr in range(n_rows):
    data = {}
    sub = df.iloc[rr]
    seq = sub['DNA']
    if not (seq in dna_mapping):
        current_index = len(dna_mapping)
        dna_mapping[seq] = current_index
        embedding = one_hot_encoder.transform(string_to_array(seq))
        np.save(f"data/dna_onehot/{current_index:d}.npy", embedding)
    data["seq"] = str(seq)
    data["ph"] = int(sub["pH"])
    data["analyte"] = sub["Analyte"]
    data["shape_term1"] = float(sub["shape_term1"])
    data["shape_term2"] = float(sub["shape_term2"])
    json_data[f"{dna_mapping[seq]}_{sub['pH']}_{sub['Analyte']}"] = data
print("embeddings are generated.")

rdata = json_data

ndata = {}
for kk, item in rdata.items():
    rname = str(kk).split("_")
    name = "_".join(rname[:2])
    if not (name in ndata):
        ndata[name] = {
            "seq": item["seq"],
            "seq_idx": int(rname[0]),
            "ph": item["ph"],
            "analyte": {}
        }
    ndata[name]["analyte"].update(
        {item["analyte"]: {"shape_term1": item["shape_term1"], "shape_term2": item["shape_term2"]}}
    )

with open("data/data.json", "w") as fp:
    json.dump(ndata, fp, indent=4)


for rr, item in ndata.items():
    for analyte in ["chlor", "cd", "enro", "semi"]:
        ndata[rr]["analyte"][analyte]["shape_term1"] = item["analyte"][analyte]["shape_term1"] * 0.0113

with open("data/normalized_data.json", "w") as fp:
    json.dump(ndata, fp, indent=4)


response1 = []
response2 = []

for condi, item in ndata.items():
    for analyte in ["chlor", "cd", "enro", "semi"]:
        response1.append(item["analyte"][analyte]["shape_term1"])
        response2.append(item["analyte"][analyte]["shape_term2"])

print(f"for response 1, mean: {np.mean(response1)}, std: {np.std(response1)}")
print(f"for response 2, mean: {np.mean(response2)}, std: {np.std(response2)}")
