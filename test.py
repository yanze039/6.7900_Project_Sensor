import gpn.mlm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

seq = "TTAATTAAGGGGAACCAAGGGGAAGG"

model_path = "gonzalobenegas/gpn-arabidopsis"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model.eval()
input_ids = tokenizer(seq, return_tensors="pt")["input_ids"]

with torch.no_grad():
    embedding = model(input_ids=input_ids).last_hidden_state[0].numpy()
print(embedding.shape)
print(embedding)
print(np.mean(embedding))
print(np.std(embedding))
