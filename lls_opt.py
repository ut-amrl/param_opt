import numpy as np
import torch
import pandas as pd 
import csv

def read_data():
    # read a csv file
    # the first row is (length, n_samples, n_features, 1)
    # each row is the row major enumeration of a n_samples x n_features x 1 tensor followed by a n_samples x 1 tensor
    # stores the data in a list of tuples of tensors
    data = []
    with open("simple_straight.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        header = next(reader)
        n_samples = int(header[1])
        n_features = int(header[2])
        for row in reader:
            state = np.array(torch.tensor(
                [float(x) for x in row[: n_samples * n_features]]
            ).view(1, n_samples, n_features).detach())
            cost = np.array(torch.tensor(
                [
                    float(x)
                    for x in row[
                        n_samples * n_features : n_samples * n_features + n_samples
                    ]
                ]
            ).view(1, n_samples).detach())
            data.append((state, cost))
    return data


def test_data(data, weights):
    ct = 0
    for i in data:
        y_hat = np.argmin(np.matmul(i[0][0], weights))
        y = np.argmin(i[1][0])

        diff = abs(y_hat - y)

        if (diff < 4):
            ct += 1

    return ct / len(data)


data = read_data()

A = []
B = []
for i in data:
    for j in i[1][0]:
        B.append(j)
    for j in i[0][0]:
        A.append(j)
A = np.array(A)

# for one-hot
# B = np.array([0 if i != 0 else 1 for i in B])

B = np.array(B)


c = np.linalg.lstsq(A, B)

weights = np.transpose(c[0])

print("accuracy: ", end="")
print(test_data(data, weights))