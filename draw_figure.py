import numpy as np
import matplotlib.pyplot as plt

OUT_FEQ = 50

err_LASSO = np.load("err_LASSO.npy")
err_NN = np.load("error_NN.npy")

print(err_LASSO.shape)
print(err_NN.shape)

analyte = ["analyte_1", "analyte_2", "analyte_3", "analyte_4"]
response_type = ["response_1", "response_2"]

X = np.arange(err_NN.shape[1]) * OUT_FEQ

for ii in range(len(analyte)):
    plt.figure()
    plt.yscale("log")
    for jj in range(len(response_type)):
        plt.plot(X, np.mean(err_NN, axis=0)[:,ii*2+jj], label=f"error_{analyte[ii]}_{response_type[jj]}")
        plt.fill_between(
            X, 
            np.mean(err_NN, axis=0)[:,ii*2+jj] - np.std(err_NN, axis=0)[:,ii*2+jj], 
            np.mean(err_NN, axis=0)[:,ii*2+jj] + np.std(err_NN, axis=0)[:,ii*2+jj],color='blue', 
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
    # plt.style.use('ggplot')
    plt.savefig(f"loss_curve_{analyte[ii]}.png", bbox_inches='tight')