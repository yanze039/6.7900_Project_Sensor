import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dna_model import DNASensorDataset, MLPmodel
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
print(os.getcwd())

with open("data/normalized_data.json", "r") as fp:
    data = json.load(fp)

all_keys = list(data.keys())
n_data = len(all_keys)
print(len(all_keys))

# 8:2 ratio is used.
val_rate = 0.2
val_num = int(val_rate * n_data)
np.random.seed(20221124)



train_dataset = []
train_dataloader = []
validation_dataset = []
validation_dataloader = []
num_parallel = 2

for i in range(num_parallel):
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

    train_dataset.append(DNASensorDataset("data/training_data.json", "data/dna_onehot"))
    train_dataloader.append(DataLoader(train_dataset[i], batch_size=1, shuffle=True, num_workers=0))

    validation_dataset.append(DNASensorDataset("data/validation_data.json", "data/dna_onehot"))
    validation_dataloader.append(DataLoader(validation_dataset[i], batch_size=len(validation_dataset[i]), shuffle=False,
                                       num_workers=0))

criterion = nn.MSELoss()

def validation(your_model, dataloader):
    inputs, labels, mask = next(iter(dataloader))
    your_model.eval()
    with torch.no_grad():
        outputs = your_model(inputs, mask)  # [*, 8]
        loss = criterion(outputs, labels)
    return loss.item()


def main(model, epoches = 5):
    
    train_loss_record = []
    val_loss_record = []
    for parallel in range(num_parallel):

        a = []
        b = []
        for epoch in range(epoches):

            running_loss = 0.0
            for i, data in enumerate(train_dataloader[parallel], 0):
                inputs, labels, mask = data
                optimizer.zero_grad()

                outputs = model(inputs, mask)  # [*, 8]
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2 == 0:
                    test_loss = validation(model, validation_dataloader[parallel])
                 #   print(f'[{epoch + 1}, {i + 1:5d}] train loss: {running_loss/2:.4f} val loss: {test_loss:.4f}')
                    a.append(running_loss)
                    b.append(test_loss)

                    running_loss = 0.0

        train_loss_record.append(a)
        val_loss_record.append(b)
    print('Finished Training')

    return np.array(train_loss_record).mean(axis = 0), np.array(val_loss_record).mean(axis = 0)

def llasso():
    lasso = Lasso(alpha=1.0)
    result_lasso = []
    for parallel in range(num_parallel):
        X_lasso = []
        Y_lasso = []
        for i, data in enumerate(train_dataloader[parallel], 0):
            inputs, labels, mask = data

            X_lasso.append(inputs.numpy().flatten())
            Y_lasso.append(labels.numpy().flatten())

        lasso.fit(X_lasso, Y_lasso)

        for i, data in enumerate(validation_dataloader[parallel], 0):
            inputs, labels, mask = data
            lasso_predict = lasso.predict(np.transpose(inputs.reshape(160, -1)))
            result_lasso.append(criterion(torch.from_numpy(lasso_predict), labels))  # 0.0393
    return np.array(result_lasso).mean()

if __name__ == "__main__":
    ''' lasso = Lasso(alpha=1.0)
    X_lasso = []
    Y_lasso = []
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels, mask = data
        X_lasso.append(inputs.numpy().flatten())
        Y_lasso.append(labels.numpy().flatten())
    lasso.fit(X_lasso, Y_lasso)
    for i, data in enumerate(validation_dataloader, 0):
        inputs, labels, mask = data
        lasso_predict = lasso.predict(np.transpose(inputs.reshape(160, -1)))
        print(criterion(torch.from_numpy(lasso_predict), labels)) #0.0393
        '''
    reg_model = MLPmodel(n_feature=40, n_embedding=4, n_out=8)
    optimizer = optim.Adam(reg_model.parameters(), lr=0.0001)
    train_loss_record, val_loss_record = main(reg_model)
    X = np.arange(len(train_loss_record)) * 2
    plt.figure()
    plt.yscale("log")
    plt.plot(X, train_loss_record, label="train")
    plt.plot(X, val_loss_record, label="validation")
    plt.title("loss curve")
    plt.xlabel("steps")
    plt.ylabel("MSELoss")
    lasso_y = llasso()
    print(lasso_y)
    plt.axhline(y=lasso_y, color='g', linestyle='-', label = "LASSO_baseline")
    plt.legend()
    plt.savefig("loss_curve.png")