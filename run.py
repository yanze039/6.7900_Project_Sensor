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
    """Perform LASSP regression.

    Args:
        training_dataset (nn.Dataset): training dataset
        validation_dataset (nn.Dataset): validation dataset
        alpha (float, optional): the facter before regularzation term. Defaults to 0.1.

    Returns:
        error (numpy.array): the breakdowm Error of the predictions of different analytes.
        rel_error (numpy.array): the relative Error of predictions
    """

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
    error = np.mean((lasso_predict - validation_Y) ** 2, axis=0)
    rel_err = (np.mean(((lasso_predict - validation_Y)/validation_Y) ** 2, axis=0)) ** 0.5
    return error, rel_err


def validation(your_model, dataset):
    """do validation evaluation.
    Args:
        your_model (nn.Module): the model evaluated.
        dataset (nn.Dataset): the validation dataset.
    
    Returns:
        loss (float): Overall loss one the dataset.
        error (numpy.array): the breakdowm Error of the predictions of different analytes.
        rel_error (numpy.array): the relative Error of predictions
    """
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    inputs, labels, mask = next(iter(dataloader))
    your_model.eval()
    with torch.no_grad():
        outputs = your_model(inputs, mask)  # [*, 8]
        loss = criterion(outputs, labels)
    error = np.mean(((outputs.numpy() - labels.numpy()))**2, axis=0)
    rel_err = (np.mean(((outputs.numpy() - labels.numpy()) / labels.numpy())**2, axis=0)) ** 0.5
    return loss.item(), error, rel_err


def neural_net(training_dataset, validation_dataset, dropout_rate=0.1, learning_rate=1e-4, epoches = 5, out_feq=50, idx_parallel=0):

    """Train a neural network.
    """
    
    train_loss_record = []
    val_loss_record = []
    rel_err_record = []
    error_record = []

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
                test_loss, error, rel_err = validation(model, validation_dataset)
                print(f'[{epoch + 1}, {idx_parallel}] train loss: {running_loss/out_feq:.4f} val loss: {np.mean(test_loss):.4f}')
                train_loss_record.append(running_loss/out_feq)
                val_loss_record.append(test_loss)
                rel_err_record.append(rel_err)
                error_record.append(error)
                running_loss = 0.0
    print('Finished.')

    return np.array(train_loss_record), np.array(val_loss_record), np.array(error_record), np.array(rel_err_record), model



def main():
    """Do experiments here.

    We ran LASSO and NN models in order. The experiments will run multiple times to get the statistics.
    Before each run, we will reshuffle the data split to eliminate the randomness of data splitting.
    """
    
    err_LASSO = []
    acc_LASSO = []
    train_loss_parallel = []
    validation_loss_parallel = []
    rel_err_NN = []
    error_NN = []

    for idx_parallel in range(NUM_PARALLEL):
        # reshuffle data / resplit data
        shuffle_data(data_json="data/normalized_data.json")
        training_dataset = DNASensorDataset("data/training_data.json", "data/dna_onehot")
        validation_dataset = DNASensorDataset("data/validation_data.json", "data/dna_onehot")

        # LASSO
        err, acc = llasso(training_dataset=training_dataset, validation_dataset=validation_dataset, alpha=ALPHA)
        err_LASSO.append(err)
        acc_LASSO.append(acc)
        print(f"Error of LASSO: {np.mean(err):.4f}, Rel err of LASSO: {np.mean(acc):.4f}")
        
        # Neural nets
        train_loss_record, val_loss_record, err_record, rel_err_record, _ = neural_net(
            training_dataset=training_dataset, 
            validation_dataset=validation_dataset,
            dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE,
            epoches = EPOCHES, out_feq=OUT_FEQ, idx_parallel=idx_parallel
        )
        train_loss_parallel.append(train_loss_record)
        validation_loss_parallel.append(val_loss_record)
        error_NN.append(err_record)
        rel_err_NN.append(rel_err_record)

    
    err_LASSO = np.array(err_LASSO)
    acc_LASSO = np.array(acc_LASSO)
    train_loss_parallel = np.array(train_loss_parallel)
    validation_loss_parallel = np.array(validation_loss_parallel)
    error_NN = np.array(error_NN)
    rel_err_NN = np.array(rel_err_NN)
    
    return err_LASSO, acc_LASSO, train_loss_parallel, validation_loss_parallel, error_NN, rel_err_NN


if __name__ == "__main__":

    # ALL hyper-parameters
    NUM_PARALLEL = 15
    OUT_FEQ = 50
    EPOCHES = 70
    ALPHA = 0.1
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 1e-4
    
    # RUN the code!
    err_LASSO, acc_LASSO, train_loss_parallel, validation_loss_parallel, error_NN, rel_err_NN = main()

    name_list = ["err_LASSO", "acc_LASSO", "train_loss_parallel", "validation_loss_parallel", "rel_err_NN", "error_NN"]
    var_list = [err_LASSO, acc_LASSO, train_loss_parallel, validation_loss_parallel, rel_err_NN, error_NN]


    # The results above will be stored in .npy format.
    # in "result" folder within current directory.
    if not os.path.exists("result"):
        os.mkdir("result")

    for ii in range(len(var_list)):
        np.save(f"result/{name_list[ii]}.npy", var_list[ii])
    
    # the name of these analyte / response.
    analyte = ["analyte_1", "analyte_2", "analyte_3", "analyte_4"]
    response_type = ["response_1", "response_2"]
    
    # x-axis, unit: steps
    X = np.arange(train_loss_parallel.shape[1]) * OUT_FEQ
    
    # Analysis. Plot the results.
    for ii in range(len(analyte)):
        plt.figure()
        plt.yscale("log")
        for jj in range(len(response_type)):
            plt.plot(X, np.mean(error_NN, axis=0)[:,ii*2+jj], label=f"error_{analyte[ii]}_{response_type[jj]}")
            plt.fill_between(
            X, 
                np.mean(error_NN, axis=0)[:,ii*2+jj] - np.std(error_NN, axis=0)[:,ii*2+jj], 
                np.mean(error_NN, axis=0)[:,ii*2+jj] + np.std(error_NN, axis=0)[:,ii*2+jj],color='blue', 
            alpha=0.2)
            baseline_color = "black" if jj == 0 else "red"
            plt.axhline(y=np.mean(err_LASSO, axis=0)[ii*2+jj], color=baseline_color, linestyle=":", lw=4, label = f"error_LASSO_{analyte[ii]}_{response_type[jj]}")
            lower_bound = np.mean(err_LASSO, axis=0)[ii*2+jj]-np.std(err_LASSO, axis=0)[ii*2+jj]
            upper_bound = np.mean(err_LASSO, axis=0)[ii*2+jj]+np.std(err_LASSO, axis=0)[ii*2+jj]
            plt.fill_between(
                [X[0], X[-1]], 
                [lower_bound, lower_bound], [upper_bound, upper_bound],color='grey', 
                alpha=0.2)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.title(f"loss curve {analyte[ii]}")
        plt.xlabel("steps")
        plt.ylabel("Loss / Error")
        plt.style.use('ggplot')
        plt.savefig(f"loss_curve_{analyte[ii]}.png", bbox_inches='tight')