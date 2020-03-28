#KMeans for color quantization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.spatial import distance

def codebook (source, n_points):
    codebook = shuffle(source, random_state=0)[:n_points]
    codebook = [np.asarray(x) for x in set(tuple(x) for x in codebook)]
    return codebook

def assign_samples (codebook, samples):
    cl =[]
    for i in range(samples.shape[0]):
        s_i = np.transpose(samples[i].reshape(samples[i].shape[0],-1)) #to convert 1d array in 2d array
        dist_i = distance.cdist(s_i, codebook, 'euclidean')
        cl_i = np.argmin(dist_i)
        cl.append(cl_i)
    return cl
    #for s in samples:
        #dist = [np.linalg.norm(np.subtract(s,label)) for label in codebook]
        #cl.append(np.argmin(dist))
    #return cl

def calculate_centroids(samples, codebook, cl, n_points):
    centroid = []
    for i in range(n_points):
        cluster_i = [index for index, x in enumerate(cl) if x ==i]
        centroid.append(np.mean(samples[cluster_i],0))
    return centroid

def convergence(samples, codebook_old, codebook_cur, thr, cl_old, cl_cur):
    delta_inertia = inertia(samples, codebook_old, cl_old) - inertia(samples, codebook_cur, cl_cur)
    if (delta_inertia < thr):
        return True
    else:
        return False

def inertia (samples, codebook, cl):
    dist =[]
    for i in range(len(codebook)):
        cluster_i = [index for index, x in enumerate(cl) if x ==i]
        s_i = samples[cluster_i]
        c_i = np.tile(codebook[i],[len(s_i),1])
        dist_i_sqr = np.linalg.norm(s_i-c_i)**2
        dist.append(dist_i_sqr)
    value_inertia = np.sum(dist)
    print(value_inertia)
    return value_inertia

def KMeans (codebook, samples, thr, max_it):
    count = 1
    n_points = len(codebook)
    cl_old = assign_samples(codebook, samples)
    centroids = calculate_centroids(samples, codebook, cl_old, n_points)
    cl_cur = assign_samples(centroids, samples)

    while (convergence(samples, codebook, centroids, thr, cl_old, cl_cur)== False and count <= max_it):
        codebook = centroids
        cl_old = cl_cur
        centroids = calculate_centroids(samples, codebook, cl_cur, n_points)
        cl_cur = assign_samples(centroids, samples)
        count = count + 1
    return (centroids, cl_cur)

def recreate_image(codb, lab, w, h, d):
    img = np.zeros((w,h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            img[i][j]=codb[lab[label_idx]]
            label_idx = label_idx +1
    return img

#Reading image
image = plt.imread('landscape.jpg')
image_ = image.astype(float) / 255
plt.imshow(image)

w, h, d = original_shape = tuple(image.shape)

assert d == 3
image_array = np.reshape(image, (w * h, d))

n_colors = 64
threshold = 0.01
max_iterations = 5

#using 64 points from the fist 500 top points 
codebook_random_1 = codebook(image_array[:500], n_colors)
(codebook1, l) = KMeans(codebook_random_1, image_array, threshold, max_iterations)
image_1 = recreate_image(codebook1, l, w, h, d)
image_1 = image_1.astype(float) / 255

plt.figure()
plt.imshow(image_1)
plt.savefig("hw6_graphs/500First.pdf")
plt.close()

#using 64 points from the whole image
codebook_random_2 = codebook(image_array, n_colors)

(codebook2, ll) = KMeans(codebook_random_2, image_array, threshold, max_iterations)

image_2 = recreate_image(codebook2, ll, w, h, d)
image_2 = image_2.astype(float) / 255

plt.figure()
plt.imshow(image_2)
plt.savefig("hw6_graphs/AllPOints.pdf")
plt.close()
