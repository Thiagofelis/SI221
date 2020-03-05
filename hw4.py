#! /usr/bin/python3

import numpy as np

def GeneratePoints(size, sigma):
    x1_1 = np.random.normal(1, sigma**2, int (size/2))
    x1_2 = np.random.normal(0, sigma**2, int (size/2))
    x1 = np.array ((x1_1, x1_2)).T
    x2_1 = np.random.normal(-1, sigma**2, int (size/2))
    x2_2 = np.random.normal(0, sigma**2, int (size/2))
    x2 = np.array ((x2_1, x2_2)).T
    x = np.concatenate ([x1, x2])
    y_1 = [1 for i in x1_1]
    y_2 = [0 for i in x1_2]
    y = np.concatenate ([y_1, y_2])
    S = list(zip(x, y))
    return S

print(GeneratePoints(10, 0.01))
