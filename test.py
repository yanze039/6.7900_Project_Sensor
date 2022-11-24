from dna_model import MLPmodel
import torch
import numpy as np
reg_model = MLPmodel(n_feature=40, n_embedding=512, n_out=8)
print(reg_model)

total = 0
for parameter in reg_model.parameters():
    num = np.sum(list(parameter.size()))
    print(num)
    total += num
print(total)
