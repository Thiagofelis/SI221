#! /usr/bin/python3

import numpy as np
import random
import copy
import matplotlib.pyplot as plt

# Generates datasets of size 'size' where:
#
# First size/2 samples are couples:
# . Bivariate normal random var of mean [1 0] and standard deviation I*'sigma_sqr'
# . Label 1
#
# Last size/2 samples are couples:
# . Bivariate normal random var of mean [-1 0] and standard deviation I*'sigma_sqr'
# . Label 0
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
    S = [list(a) for a in zip(x,y)] #Converts tuples into lists so they can be eddited
    return S

# Obtains the perceptron weights from its training dataset S by the method proposed
# in the practical work instructions
def trainPerceptron(S):
    w = np.array([0, 0])
    for sample in S:
        if (np.dot(w, sample[0]) > 0):
            if (sample[1] == 0):
                w = w - sample[0]
        else:
            if (sample[1] == 1):
                w = w + sample[0]
    return w

# Calculate the perceptron (characterised by weight w) error with respect to the testing dataset S
# obtaining mean error and its standard deviation
def calcError(w, S):
    numErrors = 0
    for sample in S:
        if (np.dot(sample[0], w) > 0 and sample[1] == 0):
            numErrors = numErrors + 1
        if (np.dot(sample[0], w) <= 0 and sample[1] == 1):
            numErrors = numErrors + 1
    return float(numErrors) / float(len(S))

def labelFlip(S, p):
    Sflipped = copy.deepcopy(S)
    for count, sample in enumerate(Sflipped):
        # Using the property that the random function generates a uniform random number
        # between [0.0 1.0) we can flip with probability p a label if the generated
        # number is less than p because of the cummulative distribution function of
        # the uniform distribution defined
        if(random.random() < p):
            if(sample[1] == 0):
                sample[1] = 1
            elif(sample[1] == 1):
                sample[1] = 0
    return Sflipped

def ex1_1():
    mean = []
    std = []
    sigmaSet = [0.05, 0.25, 0.50, 0.75]
    for sigma_sqr in sigmaSet:
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

    plt.errorbar(sigmaSet, mean, yerr = std, fmt = '.')
    plt.grid()
    plt.xlabel("Noise variance [sigma(x_i)^2]")
    plt.ylabel("Mean and std deviation of the error [$\mu(e)$ and $\sigma(e)$]")
    plt.savefig("ex1_1.pdf")
    plt.close()

def ex1_2():
    mean = []
    std = []
    pSet = [0.00, 0.05, 0.10, 0.20]
    for p in pSet:
        e = []
        for it in range(50):
            s = generatePoints(200, 0.15)
            w = trainPerceptron(s)
            s_flipped = labelFlip(s, p)
            e = e + [calcError(w, s_flipped)]
        e = np.array(e)
        mean = mean + [np.mean(e)]
        std = std + [np.std(e)]
        print ("for sigma  _sqr = 0.15 and flipping probability " + str(p)
               + ": mu(e) = " + str(np.mean(e)) + " and sigma_sqr(e) = "
               + str(np.std(e)))

    plt.errorbar(pSet, mean, yerr = std, fmt = '.')
    plt.grid()
    plt.xlabel("Flipping Probability")
    plt.ylabel("Mean and std deviation of the error [$\mu(e)$ and $\sigma(e)$]")
    plt.savefig("ex1_2.pdf")
    plt.close()

ex1_1()
ex1_2()
