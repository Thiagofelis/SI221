import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.io
from numpy.random import normal as normal
from numpy.linalg import norm as norm


def generatePoints(size, sigma_sqr):
    # generate dataset of given size and sigma_sqr
    S = []
    for i in range(size):
        y = int (np.random.uniform(0, 3))
        if (y == 0):
            mu = [-1, 1]
        elif (y == 1):
            mu = [1, 1]
        else:
            mu = [0, 1]
        x_0 = np.random.normal(mu[0], math.sqrt(sigma_sqr))
        x_1 = np.random.normal(mu[1], math.sqrt(sigma_sqr))
        _x = np.array((x_0, x_1))
        S.append([_x, y])
    return S

def updateCenter(c, X, norm=2):
    if (len(X) == 0):
        return c
    if (norm == 1):
        return (np.median([a for (a,b) in X]), np.median([b for (a,b) in X]))
    if (norm == 2):
        return (np.mean([a for (a,b) in X]), np.mean([b for (a,b) in X]))

def Kmeans(S, initCentroidsDist, updateNorm=2, costNorm=2):
    labeledS = [(v, 0) for v in S]
    previousCentroids = 0
    centroids = [ (normal(mu[0], math.sqrt(sigma_sqr)), normal(mu[1], math.sqrt(sigma_sqr)))
                                                    for (mu, sigma_sqr) in initCentroidsDist ]
    cost = []

    while (not np.array_equal (centroids, previousCentroids)):
        previousCentroids = centroids

        labeledS = [ (v, np.argmin( [ norm( np.subtract(c, v), ord=updateNorm ) ** 2 for c in centroids] ) ) for (v, l) in labeledS ]
        centroids = [ updateCenter(c , [v for (v, l) in labeledS if l == index] , updateNorm)
                                                                for (index, c) in enumerate(centroids) ]

        cost.append( sum( [ norm( np.subtract(centroids[l], v), ord=costNorm ) ** 2 for (v, l) in labeledS] ) )

    return (labeledS, centroids, cost)

def ex1_1():
    initCentroidsDist = [ ( [-1,1], 0.05), ( [1,1], 0.05), ( [0,1], 0.05) ]
    S = [v for (v,l) in generatePoints(300, 0.05)]
    labeledS, centroids, cost = Kmeans(S, initCentroidsDist)
    #colors = ['litghtcoral', 'lime', 'skyblue']
    colors = ['lightcoral', 'lime', 'dodgerblue']
    plt.scatter([v[0] for (v,l) in labeledS], [v[1] for (v,l) in labeledS], c=[colors[l] for (v,l) in labeledS], s=10)
    colorsCentroids = ['darkred', 'darkgreen', 'darkblue']
    plt.title("Initial centroids picked per gaussian")
    plt.scatter([x for (x,y) in centroids], [y for (x,y) in centroids], c=colorsCentroids, marker='x', s=200)
    plt.grid()
    plt.savefig("hw6_graphs/ex1_1.pdf",  bbox_inches='tight')
    plt.close()


def ex1_2():
    initCentroidsDist = [ ( [-1,1], 0.05), ( [1,1], 0.05), ( [0,1], 0.05) ]
    S = [v for (v,l) in generatePoints(300, 0.05)]
    labeledS, centroids, cost = Kmeans(S, initCentroidsDist)

    x = [i + 1 for i in range(len(cost))]
    plt.plot(x, cost, marker='x')
    cost = [(i, c) for (i, c) in enumerate(cost)]
    for (i, c) in cost:
        plt.annotate("{:.2f}".format(c), (i + 1, c))
    plt.title("Cost as function of the iteration")
    plt.xlabel("Iteration")
    plt.xticks(np.arange(1, len(cost) + 1, 1))
    plt.ylabel("Cost")
    plt.grid()
    plt.savefig("hw6_graphs/ex1_2_pt1.pdf",  bbox_inches='tight')
    plt.close()

    numIt = []
    for i in range(50):
        S = [v for (v,l) in generatePoints(300, 0.05)]
        labeledS, centroids, cost = Kmeans(S, initCentroidsDist)
        numIt.append(len(cost))

    plt.bar(np.unique(numIt), [numIt.count(i) / len(numIt) for i in np.unique(numIt)])
    plt.grid()
    plt.title("Empirical ditribution of number of iterations to finish")
    plt.xlabel("Number of iterations")
    plt.ylabel("Occourences")
    plt.savefig("hw6_graphs/ex1_2_pt2.pdf",  bbox_inches='tight')
    plt.close()

def ex1_3():
    initCentroidsDist = [ ( [0, 0], 0.05), ( [0, 0], 0.05), ( [0, 0], 0.05) ]
    S = [v for (v,l) in generatePoints(300, 0.05)]
    labeledS, centroids, cost = Kmeans(S, initCentroidsDist)

    x = [i + 1 for i in range(len(cost))]
    plt.plot(x, cost, marker='x')
    cost = [(i, c) for (i, c) in enumerate(cost)]
    for (i, c) in cost:
        plt.annotate("{:.2f}".format(c), (i + 1, c))
    plt.title("Cost as function of the iteration")
    plt.xlabel("Iteration")
    plt.xticks(np.arange(1, len(cost) + 1, 1))
    plt.ylabel("Cost")
    plt.grid()
    plt.savefig("hw6_graphs/ex1_3_pt1.pdf",  bbox_inches='tight')
    plt.close()

    numIt = []
    for i in range(50):
        S = [v for (v,l) in generatePoints(300, 0.05)]
        labeledS, centroids, cost = Kmeans(S, initCentroidsDist)
        numIt.append(len(cost))

    plt.bar(np.unique(numIt), [numIt.count(i) / len(numIt) for i in np.unique(numIt)])
    plt.grid()
    plt.title("Empirical ditribution of number of iterations to finish")
    plt.xlabel("Number of iterations")
    plt.ylabel("Occourences")
    plt.savefig("hw6_graphs/ex1_3_pt2.pdf",  bbox_inches='tight')
    plt.close()

def ex1_4():
    for sigma_sqr in [0.1, 0.2, 0.3]:
        initCentroidsDist = [ ( [0, 0], sigma_sqr), ( [0, 0], sigma_sqr), ( [0, 0], sigma_sqr) ]
        S = [v for (v,l) in generatePoints(300, sigma_sqr)]
        labeledS, centroids, cost = Kmeans(S, initCentroidsDist)

        x = [i + 1 for i in range(len(cost))]
        plt.plot(x, cost, marker='x')
        cost = [(i, c) for (i, c) in enumerate(cost)]
        for (i, c) in cost:
            plt.annotate("{:.2f}".format(c), (i + 1, c))
        plt.title("Cost as function of the iteration")
        plt.xlabel("Iteration")
        plt.xticks(np.arange(1, len(cost) + 1, 1))
        plt.ylabel("Cost")
        plt.grid()
        plt.savefig("hw6_graphs/ex1_3_pt1" + "_sigma_sqr=" + str(sigma_sqr) + ".pdf",  bbox_inches='tight')
        plt.close()

        numIt = []
        for i in range(50):
            S = [v for (v,l) in generatePoints(300, sigma_sqr)]
            labeledS, centroids, cost = Kmeans(S, initCentroidsDist)
            numIt.append(len(cost))

        plt.bar(np.unique(numIt), [numIt.count(i) / len(numIt) for i in np.unique(numIt)])
        plt.grid()
        plt.title("Empirical ditribution of number of iterations to finish")
        plt.xlabel("Number of iterations")
        plt.ylabel("Occourences")
        plt.savefig("hw6_graphs/ex1_3_pt2" + "_sigma_sqr=" + str(sigma_sqr) + ".pdf",  bbox_inches='tight')
        plt.close()

def ex1_5a():
    initCentroidsDist = [ ( [0, 0], 0.05), ( [0, 0], 0.05), ( [0, 0], 0.05) ]
    S = [v for (v,l) in generatePoints(300, 0.05)]
    labeledS, centroids, cost = Kmeans(S, initCentroidsDist, updateNorm=2, costNorm=1)

    x = [i + 1 for i in range(len(cost))]
    plt.plot(x, cost, marker='x')
    cost = [(i, c) for (i, c) in enumerate(cost)]
    for (i, c) in cost:
        plt.annotate("{:.2f}".format(c), (i + 1, c))
    plt.title("Cost as function of the iteration")
    plt.xlabel("Iteration")
    plt.xticks(np.arange(1, len(cost) + 1, 1))
    plt.ylabel("Cost")
    plt.grid()
    plt.savefig("hw6_graphs/ex1_5a_pt1.pdf",  bbox_inches='tight')
    plt.close()

    numIt = []
    for i in range(50):
        S = [v for (v,l) in generatePoints(300, 0.05)]
        labeledS, centroids, cost = Kmeans(S, initCentroidsDist, updateNorm=2, costNorm=1)
        numIt.append(len(cost))

    plt.bar(np.unique(numIt), [numIt.count(i) / len(numIt) for i in np.unique(numIt)])
    plt.grid()
    plt.title("Empirical ditribution of number of iterations to finish")
    plt.xlabel("Number of iterations")
    plt.ylabel("Occourences")
    plt.savefig("hw6_graphs/ex1_5a_pt2.pdf",  bbox_inches='tight')
    plt.close()

def ex1_5b():
    initCentroidsDist = [ ( [0, 0], 0.05), ( [0, 0], 0.05), ( [0, 0], 0.05) ]
    S = [v for (v,l) in generatePoints(300, 0.05)]
    labeledS, centroids, cost = Kmeans(S, initCentroidsDist, updateNorm=1, costNorm=1)

    x = [i + 1 for i in range(len(cost))]
    plt.plot(x, cost, marker='x')
    cost = [(i, c) for (i, c) in enumerate(cost)]
    for (i, c) in cost:
        plt.annotate("{:.2f}".format(c), (i + 1, c))
    plt.title("Cost as function of the iteration")
    plt.xlabel("Iteration")
    plt.xticks(np.arange(1, len(cost) + 1, 1))
    plt.ylabel("Cost")
    plt.grid()
    plt.savefig("hw6_graphs/ex1_5b_pt1.pdf",  bbox_inches='tight')
    plt.close()

    numIt = []
    for i in range(50):
        S = [v for (v,l) in generatePoints(300, 0.05)]
        labeledS, centroids, cost = Kmeans(S, initCentroidsDist, updateNorm=1, costNorm=1)
        numIt.append(len(cost))

    plt.bar(np.unique(numIt), [numIt.count(i) / len(numIt) for i in np.unique(numIt)])
    plt.grid()
    plt.title("Empirical ditribution of number of iterations to finish")
    plt.xlabel("Number of iterations")
    plt.ylabel("Occourences")
    plt.savefig("hw6_graphs/ex1_5b_pt2.pdf",  bbox_inches='tight')
    plt.close()

ex1_1()
ex1_2()
ex1_3()
ex1_4()
ex1_5a()
ex1_5b()
