import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dna_model import DNASensorDataset, MLPmodel, shuffle_data
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import Lasso

criterion = nn.MSELoss()


def llasso(training_dataset, validation_dataset, alpha=0.1):
    train_dataloader = DataLoader(training_dataset, batch_size=len(training_dataset), shuffle=True, num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False, num_workers=0)
    inputs, labels, _ = next(iter(train_dataloader))
    feature_1, feature_2 = inputs.shape[-2], inputs.shape[-1]
    flatten_feature_dim = int(feature_1 * feature_2)
    inputs = inputs.reshape([-1, flatten_feature_dim])
    inputs, labels = inputs.numpy(), labels.numpy()
    lasso = Lasso(alpha=alpha)
    lasso.fit(inputs, labels)
    
    validation_X, validation_Y, _ = next(iter(validation_dataloader))
    validation_X, validation_Y = validation_X.numpy(), validation_Y.numpy()
    lasso_predict = lasso.predict(validation_X.reshape(-1, flatten_feature_dim))
    error = (np.mean((lasso_predict - validation_Y) ** 2))
    accuracy = (np.mean(((lasso_predict - validation_Y)/validation_Y) ** 2)) ** 0.5
    return error, accuracy


def validation(your_model, dataset):
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    inputs, labels, mask = next(iter(dataloader))
    your_model.eval()
    with torch.no_grad():
        outputs = your_model(inputs, mask)  # [*, 8]
        loss = criterion(outputs, labels)
    rel_err = (np.mean(((outputs.numpy() - labels.numpy()) / labels.numpy())**2)) ** 0.5
    return loss.item(), rel_err


def neural_net(training_dataset, validation_dataset, dropout_rate=0.1, learning_rate=1e-6, epoches = 5, out_feq=50):
    
    train_loss_record = []
    val_loss_record = []
    rel_err_record = []

    train_dataloader = DataLoader(training_dataset, batch_size=8, shuffle=True, num_workers=0)
    model = MLPmodel(n_feature=40, n_embedding=4, n_out=8, dropout_rate=dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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
            if i % out_feq == 0:
                test_loss, rel_err = validation(model, validation_dataset)
                print(f'[{epoch + 1}, {i + 1:5d}] train loss: {running_loss/out_feq:.4f} val loss: {test_loss:.4f}')
                train_loss_record.append(running_loss/out_feq)
                val_loss_record.append(test_loss)
                rel_err_record.append(rel_err)
                running_loss = 0.0
    print('Finished Training')

    return np.array(train_loss_record), np.array(val_loss_record), np.array(rel_err_record)



def main():
    
    err_LASSO = []
    acc_LASSO = []
    train_loss_parallel = []
    validation_loss_parallel = []
    accuracy_NN = []

    for idx in range(NUM_PARALLEL):
        # reshuffle data / resplit data
        shuffle_data(data_json="data/normalized_data.json")
        training_dataset = DNASensorDataset("data/training_data.json", "data/dna_onehot")
        validation_dataset = DNASensorDataset("data/validation_data.json", "data/dna_onehot")

        # LASSO
        err, acc = llasso(training_dataset=training_dataset, validation_dataset=validation_dataset, alpha=ALPHA)
        err_LASSO.append(err)
        acc_LASSO.append(acc)
        print(f"Error of LASSO: {err:.4f}, Accuracy of LASSO: {acc:.4f}")
        
        # Neural nets
        train_loss_record, val_loss_record, rel_err_record = neural_net(
            training_dataset=training_dataset, 
            validation_dataset=validation_dataset,
            dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE,
            epoches = EPOCHES, out_feq=OUT_FEQ
        )
        train_loss_parallel.append(train_loss_record)
        validation_loss_parallel.append(val_loss_record)
        accuracy_NN.append(rel_err_record)
    
    err_LASSO = np.array(err_LASSO)
    acc_LASSO = np.array(acc_LASSO)
    train_loss_parallel = np.array(train_loss_parallel)
    validation_loss_parallel = np.array(validation_loss_parallel)
    accuracy_NN = np.array(accuracy_NN)
    
    return err_LASSO, acc_LASSO, train_loss_parallel, validation_loss_parallel, accuracy_NN


if __name__ == "__main__":
    NUM_PARALLEL = 2
    OUT_FEQ = 50
    EPOCHES = 40
    ALPHA = 0.1
    DROPOUT_RATE = 0.1
    LEARNING_RATE = 1e-4

    err_LASSO, acc_LASSO, train_loss_parallel, validation_loss_parallel, accuracy_NN = main()

    name_list = ["err_LASSO", "acc_LASSO", "train_loss_parallel", "validation_loss_parallel", "accuracy_NN"]
    var_list = [err_LASSO, acc_LASSO, train_loss_parallel, validation_loss_parallel, accuracy_NN]

    if not os.path.exists("result"):
        os.mkdir("result")

    for ii in range(len(var_list)):
        np.save(f"result/{name_list[ii]}.npy", var_list[ii])

    X = np.arange(train_loss_parallel.shape[1]) * OUT_FEQ
    plt.figure()
    plt.yscale("log")
    plt.plot(X, np.mean(train_loss_parallel, axis=0), label="train MSE loss")
    plt.plot(X, np.mean(validation_loss_parallel, axis=0), label="validation MSE loss")
    plt.title("loss curve")
    plt.xlabel("steps")
    plt.ylabel("MSELoss")
    plt.axhline(y=np.mean(err_LASSO), color='g', linestyle='-', label = "LASSO_baseline")
    plt.legend()
    plt.savefig("loss_curve.png")

