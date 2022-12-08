import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os


class DNASensorDataset(Dataset):

    def __init__(self, json_file, embedding_dir, max_length=39) -> None:
        super().__init__()
        with open(json_file, "r") as fp:
            data = json.load(fp)
        self.data = data
        self.keys = list(self.data.keys())
        self.n_data = len(self.keys)
        self.embedding_dir = embedding_dir
        self.ph_mapping = {6: 0., 8: 1.}
        self.analyte = ["chlor", "cd", "enro", "semi"]
        self.response_type = ["shape_term1", "shape_term2"]
        self.max_length = max_length
    
    def __len__(self):
        return self.n_data
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        sample = self.data[key]
        seq_embedding = np.load(os.path.join(self.embedding_dir, f"{sample['seq_idx']}.npy"))
        ph_embedding = self.ph_mapping[sample['ph']] * np.ones(shape=[1, seq_embedding.shape[1]])
        mask = np.ones([self.max_length+1,])
        n_padding = self.max_length - seq_embedding.shape[0]
        if n_padding > 0:
            mask[seq_embedding.shape[0]+1: ] = 0.
            padding = np.zeros(shape=[n_padding, seq_embedding.shape[1]])
            seq_embedding = np.concatenate((seq_embedding, padding), axis=0)
            
        response = []
        for analyte in self.analyte:
            for rsp in self.response_type:
                response.append(sample["analyte"][analyte][rsp])
        X = np.concatenate((ph_embedding, seq_embedding), axis=0)
        Y = np.array(response)
        return torch.from_numpy(X).float(), torch.from_numpy(Y).float(), torch.from_numpy(mask).float()


def shuffle_data(
        data_json="data/normalized_data.json", 
        validation_rate=0.2,
        validation_out="data/validation_data.json",
        training_out="data/training_data.json"
    ):

    with open(data_json, "r") as fp:
        data = json.load(fp)

    all_keys = list(data.keys())
    n_data = len(all_keys)

    # 8:2 ratio is used.
    val_num = int(validation_rate * n_data)

    validation_idx = sorted(np.random.choice(n_data, val_num, replace=False))
    validation_keys = [all_keys[kk] for kk in validation_idx]
    training_keys = [kk for kk in all_keys if kk not in validation_keys]

    validation_data = {}
    for v_k in validation_keys:
        validation_data.update({v_k: data[v_k]})

    with open(validation_out, 'w') as vd:
        json.dump(validation_data, vd, indent=4)

    training_data = {}
    for t_k in training_keys:
        training_data.update({t_k: data[t_k]})

    with open(training_out, 'w') as td:
        json.dump(training_data, td, indent=4)
    

class MLPmodel(nn.Module):

    def __init__(self, n_feature, n_embedding, n_out, dropout_rate=0.1) -> None:
        super().__init__()
        self.n_feature = n_feature
        self.n_embedding = n_embedding
        self.n_out = n_out
        self.linear1 = torch.nn.Linear(n_embedding, n_out)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(n_feature, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std= 2.0 / (self.n_feature*self.n_embedding))
            if module.bias is not None:
                module.bias.data.zero_()

    
    def forward(self, x, mask):
        x = self.linear1(x)  # [*, 40, 8]
        x = self.dropout(x)
        x = self.activation(x)  # [*, 40, 8]
        x = x * torch.unsqueeze(mask, -1)  # [*, 40, 8]
        x = torch.transpose(x, -2, -1)  # [*, 8, 40]
        x = self.linear2(x)  # [*, 4, 8, 1]
        return torch.squeeze(x)

