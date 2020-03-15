import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.io

def generatePoints(size, sigma_sqr):
    S = []
    for i in range(size):
        y = int (np.random.uniform(0, 3))
        if (y == 0):
            mu = [-1, 0]
        elif (y == 1):
            mu = [1, 0]
        else:
            mu = [0, 1]
        x_0 = np.random.normal(mu[0], sigma_sqr)
        x_1 = np.random.normal(mu[1], sigma_sqr)
        _x = np.array((x_0, x_1))
        S.append([_x, y])
    return S

def kNN(k, S, x):
    dist = [[math.sqrt((x[0] - s[0][0])**2 + (x[1] - s[0][1])**2), s[1]] for s in S]
    dist = sorted(dist, key=lambda entry: entry[0])
    dist = dist[0: k]
    dist_y = [s[1] for s in dist]
    return max(set(dist_y), key = dist_y.count)

def predictionBinaryError(k, S, X):
    errors = 0
    for x in X:
        if (x[1] != kNN(k, S, x[0])):
            errors = errors + 1
    return float(errors) / float(len(X))

def predictionAbsError(k, S, X):
    errors = 0
    for x in X:
        errors = errors + 1
    return float(errors) / float(len(X))

def ex_1_1():
    S = generatePoints(300, 0.10)
    K = [1, 2, 5, 10]
    for k in K:
        x_plane = [0.01 * a for a in range(-150, 150)]
        discrete_domain = [[a,b] for a in x_plane for b in x_plane]
        discrete_image = [[kNN(k, S, [a,b]) for b in x_plane] for a in x_plane]
        plt.contourf(x_plane, x_plane, discrete_image)
        plt.grid()
        plt.title("k = " + str(k))
        plt.savefig("hw5_graphs/E1Q1/decisionBoundary k = " + str(k) + ".pdf",\
                    bbox_inches='tight')
        plt.close()

def ex_1_2():
    sigma_sqr = [0.05, 0.10, 0.15, 0.20, 0.25]
    k = [1, 2, 5, 10]
    errors = []
    for ssq in sigma_sqr:
        errors_ssq = []
        for _k in k:
            errors_k = []
            for i in range(50):
                S = generatePoints(300, ssq)
                errors_k = errors_k +\
                    [predictionBinaryError(_k, S[0 : 200], S[200 : 300])]
            errors_ssq.append([np.mean(errors_k), np.std(errors_k)])
        errors = errors + [errors_ssq]
    print (errors)
    plt.errorbar(k, [a[0] for a in errors[4]], yerr = [a[1] for a in errors[4]],\
                 fmt = '.')
    plt.grid()
    plt.title("$\sigma(x_i)^2$ fixed at 0.25")
    plt.xlabel("k")
    plt.ylabel("Mean and std deviation of the error [$\mu(e)$ and $\sigma(e)$]")
    plt.savefig("hw5_graphs/E1Q2/sigma_fixed.pdf",  bbox_inches='tight')
    plt.close()

    plt.errorbar(sigma_sqr, [a[2][0] for a in errors],\
                 yerr = [a[2][1] for a in errors], fmt = '.')
    plt.grid()
    plt.title("K fixed at 5")
    plt.xlabel("$\sigma(x_i)^2$")
    plt.ylabel("Mean and std deviation of the error [$\mu(e)$ and $\sigma(e)$]")
    plt.savefig("hw5_graphs/E1Q2/k_fixed.pdf",  bbox_inches='tight')
    plt.close()

def ex_2_1():
    df = pd.read_csv("weatherHistory.csv")
    temp = df['Temperature (C)'].values
    hum = df['Humidity'].values
    app = df['Apparent Temperature (C)'].values
    plt.tricontourf(temp, hum, app, 100, cmap = 'jet')
    plt.colorbar()
    plt.title("Apparent temperature in $^\circ$C as a function of\
              temperature and humidity")
    plt.xlabel("Temperature [$^\circ$C]")
    plt.ylabel("Humidity")
    plt.savefig("hw5_graphs/E2Q1.pdf",  bbox_inches='tight')
    plt.close()

def ex_2_2():
    df = pd.read_csv("weatherHistory.csv")
    df = df.iloc[:2000, :]
    df = df.loc[:,['Temperature (C)', 'Humidity', 'Apparent Temperature (C)']]
    df = df.sample(frac=1)
    df = df.values
    S = [list(a) for a in zip(df[:, [0,1]], df[:, 2])]
    train = S[:int(4/5*nbRows), :]
    test = S[int(4/5*nbRows):, :]


def ex_3():
    train = scipy.io.loadmat('data_app.mat')
    test = scipy.io.loadmat('data_test.mat')

#ex_1_1()
#ex_1_2()
#ex_2_1()
