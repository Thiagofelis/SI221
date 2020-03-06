#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

def generatePoints(size, sigma_sqr):
    x1_1 = np.random.normal(1, sigma_sqr, int (size/2))
    x1_2 = np.random.normal(0, sigma_sqr, int (size/2))
    x1 = np.array ((x1_1, x1_2)).T
    x2_1 = np.random.normal(-1, sigma_sqr, int (size/2))
    x2_2 = np.random.normal(0, sigma_sqr, int (size/2))
    x2 = np.array ((x2_1, x2_2)).T
    x = np.concatenate ([x1, x2])
    y_1 = [1 for i in x1_1]
    y_2 = [0 for i in x1_2]
    y = np.concatenate ([y_1, y_2])
    S = list(zip(x, y))
    return S

def trainPerceptron(S):
    w = np.array([0, 0])
    for count, sample in enumerate(S):
        if (np.dot(w, sample[0]) > 0):
            if (sample[1] == 0):
                w = w - sample[0]
        else:
            if (sample[1] == 1):
                w = w + sample[0]
    return w

def calcError(w, S):
    numErrors = 0
    for sample in S:
        if (np.dot(sample[0], w) > 0 and sample[1] == 0):
            numErrors = numErrors + 1
        if (np.dot(sample[0], w) <= 0 and sample[1] == 1):
            numErrors = numErrors + 1
    return float(numErrors) / float(len(S))

mean = []
std = []
for sigma_sqr in [0.05, 0.25, 0.50, 0.75]:
    e = []
    for it in range(50):
        s = generatePoints(200, sigma_sqr)
        w = trainPerceptron(s)
        e = e + [calcError(w, s)]
    e = np.array(e)
    mean = mean + [np.mean(e)]
    std = std + [np.std(e)]
    print ("for sigma  _sqr = " + str(sigma_sqr) + ": mu(e) = " + str(np.mean(e))
    + " and sigma_sqr(e) = " + str(np.std(e)))

plt.errorbar([0.05, 0.25, 0.50, 0.75], mean, yerr = std, fmt = '.')
plt.grid()
plt.xlabel("Noise variance [sigma(x_i)^2]")
plt.ylabel("Mean and standard deviation of the error [mu(e) and sigma(e)]")
plt.savefig('ex1_1.pdf')
