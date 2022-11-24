import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dna_model import DNASensorDataset, MLPmodel
import matplotlib.pyplot as plt
import numpy as np


train_dataset = DNASensorDataset("data/training_data.json", "data/dna_embedding")
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

validation_dataset = DNASensorDataset("data/validation_data.json", "data/dna_embedding")
validation_dataloader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False, num_workers=0)

criterion = nn.MSELoss()

def validation(your_model, dataloader):
    inputs, labels, mask = next(iter(dataloader))
    your_model.eval()
    with torch.no_grad():
        outputs = your_model(inputs, mask)  # [*, 8]
        loss = criterion(outputs, labels)
    return loss.item()


def main(model, epoches = 15):
    
    train_loss_record = []
    val_loss_record = []

    for epoch in range(epoches):

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels, mask = data
            optimizer.zero_grad()

            outputs = model(inputs, mask)  # [*, 8]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2 == 0:   
                test_loss = validation(model, validation_dataloader)
                print(f'[{epoch + 1}, {i + 1:5d}] train loss: {running_loss/2:.4f} val loss: {test_loss:.4f}')
                train_loss_record.append(running_loss)
                val_loss_record.append(test_loss)

                running_loss = 0.0
    
    print('Finished Training')

    return train_loss_record, val_loss_record



if __name__ == "__main__":
    reg_model = MLPmodel(n_feature=40, n_embedding=512, n_out=8)
    optimizer = optim.Adam(reg_model.parameters(), lr=0.0001)
    train_loss_record, val_loss_record = main(reg_model)
    X = np.arange(len(train_loss_record)) * 2
    plt.figure()
    plt.plot(X, train_loss_record, label="train")
    plt.plot(X, val_loss_record, label="validation")
    plt.title("loss curve")
    plt.xlabel("steps")
    plt.ylabel("MSELoss")
    plt.legend()
    plt.savefig("loss_curve.png")