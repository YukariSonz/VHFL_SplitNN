"""
==========
plot(x, y)
==========

See `~matplotlib.axes.Axes.plot`.
"""

import matplotlib.pyplot as plt
import numpy as np

import csv
import pandas as pd
import argparse


def read_results(file_name):
       df = pd.read_csv(file_name)
       accuracy = df[["Training_Accuracy", "Testing_Accuracy"]]
       loss = df[["Training_loss", "Testing_loss"]]

       return accuracy, loss

parser = argparse.ArgumentParser()
parser.add_argument("--csv1", type = str)
parser.add_argument("--csv2", type = str)
args = parser.parse_args()

accuracy_SGD, loss_SGD = read_results(args.csv1)
accuracy_Adam, loss_Adam = read_results(args.csv2)

fig, axes = plt.subplots(nrows=2, ncols=2)


# print(df)

accuracy_SGD.plot(ax=axes[0][0])
loss_SGD.plot(ax=axes[1][0])

accuracy_Adam.plot(ax=axes[0][1])
loss_Adam.plot(ax=axes[1][1])

title_list = [["Accuracy (SGD)", "Accuracy (Adam)"], ["Loss (SGD)", "Loss (Adam)" ]]
for i in range(2):
    for j in range(2):
        ax = axes[i,j]
        ax.set_title(title_list[i][j])

        # Accuracy
        if i == 0:
            ax.set_ylim([0, 100])
        else:
            ax.set_ylim([0, 2])


plt.legend()
plt.xlabel("Communication Rounds")
plt.show()
