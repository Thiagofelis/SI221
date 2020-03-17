import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.io

def generatePoints(size, sigma_sqr):
    # generate dataset of given size and sigma_sqr, according to
    # the specifications given in exercise 1
    S = []
    for i in range(size):
        y = int (np.random.uniform(0, 3))
        if (y == 0):
            mu = [-1, 0]
        elif (y == 1):
            mu = [1, 0]
        else:
            mu = [0, 1]
        x_0 = np.random.normal(mu[0], math.sqrt(sigma_sqr))
        x_1 = np.random.normal(mu[1], math.sqrt(sigma_sqr))
        _x = np.array((x_0, x_1))
        S.append([_x, y])
    return S

def kNN(k, S, x):
    # given a k and a dataset S, where each element s of the dataset is a tuple of
    # coordinates (s[0]) and a label(s[1]), it runs kNN to calculate the label on point x
    dist = [[math.sqrt((x[0] - s[0][0])**2 + (x[1] - s[0][1])**2), s[1]] for s in S]
    dist = sorted(dist, key=lambda entry: entry[0])
    dist = dist[0: k]
    dist_y = [s[1] for s in dist]
    return max(set(dist_y), key = dist_y.count)

# Runs knn where the labels can take real numbers and returns the mean of the k
# nearest neighbours
def kNNReal(k, S, x):
    dist = [[math.sqrt((x[0] - s[0][0])**2 + (x[1] - s[0][1])**2), s[1]] for s in S]
    dist = sorted(dist, key=lambda entry: entry[0])
    dist = dist[0: k]
    dist_y = [s[1] for s in dist]
    return np.mean(dist_y)

# Runs knn where the input is a vector and the distance considered is the euclidian
# distance between it and the training set vectors
def kNNVector(k, S, x):
    dist = []
    for s in S:
        dist_vv_sqr = 0
        for i in range(len(x)):
            dist_vv_sqr = dist_vv_sqr + (x[i]-s[0][i])**2
        dist.append([math.sqrt(dist_vv_sqr), s[1]])
    dist = sorted(dist, key=lambda entry: entry[0])
    dist = dist[0: k]
    dist_y = [s[1] for s in dist]
    return max(set(dist_y), key = dist_y.count)

def kNNpredictionBinaryError(k, S, X):
    # calculates prediction error on X, for the given k and the given dataset S.
    # each element x of X is a tuple of cordinates and a label.
    errors = 0
    for x in X:
        if (x[1] != kNN(k, S, x[0])):
            errors = errors + 1
    return float(errors) / float(len(X))

# Computes the number of labels wrongly predicted by the algorithm
def predictionBinaryError(label, pred):
    errors = 0
    for i in range(len(pred)):
        if (label[i] != pred[i]):
            errors = errors + 1
    return float(errors) / float(len(pred))

# Computes the sum of the absolute errors commited by the prediction in a label set
def predictionAbsError(label, pred):
    errors = 0
    for i in range(len(pred)):
        errors = errors + abs(label[i] - pred[i])
    return float(errors) / float(len(pred))

# Returns a matrix where the element a_i_j represents that an element i was labelled
# as being j (from 0 to 9)
def confusion_matrix(l, r):
    m = np.zeros((10, 10))
    ll = [int(0) if x == 10 else int(x) for x in l]
    rr = [int(0) if x == 10 else int(x) for x in r]
    for i in range(len(r)):
        m[ll[i]][rr[i]] = m[ll[i]][rr[i]] + 1
    return m

def ex_1_1():
    S = generatePoints(300, 0.10)
    K = [1, 2, 5, 10]
    for k in K:
        x_plane = [0.01 * a for a in range(-150, 150)]
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
                    [kNNpredictionBinaryError(_k, S[0 : 200], S[200 : 300])]
            errors_ssq.append([np.mean(errors_k), np.std(errors_k)])
        errors = errors + [errors_ssq]
    plt.errorbar(k, [a[0] for a in errors[3]], yerr = [a[1] for a in errors[3]],\
                 fmt = '.')
    plt.grid()
    plt.title("$\sigma(x_i)^2$ fixed at 0.20")
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
    nbRows = 2000
    df = pd.read_csv("weatherHistory.csv")
    df = df.iloc[:nbRows, :]
    df = df.loc[:,['Temperature (C)', 'Humidity', 'Apparent Temperature (C)']]
    K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    meanError = []
    stdError = []
    for k in K:
        error = []
        for i in range(5):
            dfShuffled = df.sample(frac=1)
            dfShuffled = dfShuffled.values
            S = [list(a) for a in zip(dfShuffled[:, [0,1]], dfShuffled[:, 2])]
            train = S[:int(4/5*nbRows)]
            test = S[int(4/5*nbRows):]
            labels = [b for a,b in test]
            result_2_2 = [kNNReal(k, S, x) for x,y in test]
            error.append(predictionAbsError(labels, result_2_2))
        meanError.append(np.mean(error))
        stdError.append(np.std(error))
    plt.errorbar(K, meanError, yerr = stdError, fmt = '.')
    plt.grid()
    plt.xlabel("Number K of neighbours considered")
    plt.ylabel("Mean and std deviation of the error [$\mu(e)$ and $\sigma(e)$]")
    plt.savefig("hw5_graphs/E2Q2.pdf",  bbox_inches='tight')
    plt.close()

def ex_3_1():
    train = scipy.io.loadmat('data_app.mat')
    test = scipy.io.loadmat('data_test.mat')
    train['x'] = train['x']/255
    test['x'] = test['x']/255

    fig1=plt.figure(figsize=(5,8))
    columns = 25
    rows = 40
    for i in range(1, len(train['x']) + 1):
        img = train['x'][i-1].reshape(28, 28)
        fig1.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("hw5_graphs/E3Q1/Training.pdf",  bbox_inches='tight')
    plt.close()

    fig2=plt.figure(figsize=(4,3))
    columns = 20
    rows = 15
    for i in range(1, len(test['x']) + 1):
        img = test['x'][i-1].reshape(28, 28)
        fig2.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("hw5_graphs/E3Q1/Testing.pdf",  bbox_inches='tight')
    plt.close()
    bins = [x-0.5 for x in range(1,12)]
    plt.hist([train['S'], test['S']], bins, label=['Train', 'Test'])
    plt.xticks(range(1, 11))
    plt.legend(loc='upper right')
    plt.savefig("hw5_graphs/E3Q1/DataDistrib.pdf",  bbox_inches='tight')
    plt.close()

def ex_3_2():
    train = scipy.io.loadmat('data_app.mat')
    test = scipy.io.loadmat('data_test.mat')
    train['x'] = train['x']/255
    test['x'] = test['x']/255

    trainLabels = [x[0] for x in train['S']]
    testLabels = [x[0] for x in test['S']]
    Strain = [list(a) for a in zip(train['x'], trainLabels)]
    K = [1,3,5]
    error = []
    result_3_2 = {}
    for k in K:
        result_3_2[k] = [kNNVector(k, Strain, x) for x in test['x']]
        error.append(predictionBinaryError(testLabels, result_3_2[k]))
    plt.scatter(K, error)
    plt.grid()
    plt.xlabel("Number K of neighbours considered")
    plt.ylabel("Proportional error labelling the numbers")
    plt.savefig("hw5_graphs/E3Q2.pdf",  bbox_inches='tight')
    plt.close()
    return result_3_2

def ex_3_3(result):
    test = scipy.io.loadmat('data_test.mat')
    for k in result:
        print ("Confusion matrix for k = " + str(k))
        print (confusion_matrix(test['S'], result[k]))

ex_1_1()
ex_1_2()
ex_2_1()
ex_2_2()
ex_3_1()
ex_3_3(ex_3_2())
