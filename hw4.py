import numpy as np

def GeneratePoints(size, sigma):
# quase ctz q ta funcionando
    x1_1 = np.random.normal(1, sigma**2, size/2)
    x1_2 = np.random.normal(0, sigma**2, size/2)
    x1 = np.concatenate ([x1_1, x1_2])
    x2 = np.random.normal(0, sigma**2, size)
    x = np.array((x1, x2)).T
    y_1 = [1 for i in x1_1]
    y_2 = [0 for i in x1_2]
    y = np.concatenate ([y_1, y_2])
    S = list(zip(x, y))
    return S
