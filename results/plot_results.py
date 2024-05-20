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
parser.add_argument("--csv", type = str)
args = parser.parse_args()

accuracy, loss = read_results(args.csv)
fig, axes = plt.subplots(nrows=2, ncols=1)


# print(df)

accuracy.plot(ax=axes[0])
loss.plot(ax=axes[1])

title_list = ["Accuracy (SGD)", "Loss (SGD)"]
for i, ax in enumerate(axes.ravel()): 
    ax.set_title(title_list[i])


plt.legend()
plt.xlabel("Communication Rounds")
plt.show()
