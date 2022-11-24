import json
import numpy as np
import matplotlib.pyplot as plt

with open("normalized_data.json", "r") as fp:
    data = json.load(fp)

all_keys = list(data.keys())
n_data = len(all_keys)
print(len(all_keys))

# 8:2 ratio is used.
val_rate = 0.2
val_num = int(val_rate * n_data)
np.random.seed(20221124)

validation_idx = sorted(np.random.choice(n_data, val_num, replace=False))
validation_keys = [all_keys[kk] for kk in validation_idx]
training_keys = [kk for kk in all_keys if kk not in validation_keys]
print(len(training_keys))


validation_data = {}
for v_k in validation_keys:
    validation_data.update({v_k: data[v_k]})

with open("validation_data.json", 'w') as vd:
    json.dump(validation_data, vd, indent=4)

training_data = {}
for t_k in training_keys:
    training_data.update({t_k: data[t_k]})

with open("training_data.json", 'w') as td:
    json.dump(training_data, td, indent=4)


#  Here we need to find the max length of Seq so that we can pad the matrix to the same shape.
with open("data.json", "r") as fp:
    data = json.load(fp)

seq_length = []
for key, item in data.items():
    print(item)
    seq_length.append(len(item['seq']))

print(seq_length)
print(np.max(seq_length))
plt.figure()
plt.hist(seq_length)
plt.savefig("len.png")