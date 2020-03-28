#KMeans for color quantization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.spatial import distance

def codebook (source, n_points):
    codebook = shuffle(source, random_state=0)[:n_points]
    return codebook

def assign_samples (codebook, samples):
    cl =[]
    for i in range(samples.shape[0]):
        s_i = np.transpose(samples[i].reshape(samples[i].shape[0],-1)) #to convert 1d array in 2d array
        dist_i = distance.cdist(s_i, codebook, 'euclidean')
        cl_i = np.argmin(dist_i)
        cl.append(cl_i)
    return cl

def calculate_centroids(samples, codebook, cl, n_points):
    centroid = codebook.copy() 
    #TO FILL HERE!
    return centroid

def convergence(samples, codebook_old, codebook_current, thr, cl):
    # FILL HERE!
    # Compute here delta_inertia as the difference between the current inertia (compute on the current codebook) 
    # and the inertia at the previous step (computed on the old codebook)
    
        
    if (delta_inertia > 0 and delta_inertia < thr):
        return True
    else:
        return False
    
def inertia (samples, codebook, cl):
    dist =[]
    for i in range(codebook.shape[0]):
        s_i = samples[cl==i]
        s_i = np.transpose(samples[i].reshape(samples[i].shape[0],-1)) #to convert 1d array in 2d array
        c_i = np.transpose(codebook[i].reshape(codebook[i].shape[0],-1))
        dist_i_sqr = (distance.cdist(s_i, c_i, 'euclidean'))**2
        dist.append(dist_i_sqr)
    value_inertia = np.sum(dist)
    return value_inertia

def KMeans (codebook, samples, thr, max_it):
    count = 1
    n_points = len(codebook)
    cls = assign_samples(codebook, samples)
    centroids = calculate_centroids(samples, codebook, cls, n_points) 
    
    while (convergence(samples, codebook, centroids, thr, cls)== False and count <= max_it): 
        #TO FILL HERE!
    return  cls

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
max_iterations = 3

#using 64 points from the fist 500 top points 
codebook_random_1 = codebook(image_array[:500], n_colors)
l = KMeans(codebook_random_1, image_array, threshold, max_iterations)
image_1 = recreate_image(codebook_random_1, l, w, h, d)
image_1 = image_1.astype(float) / 255

plt.figure()
plt.imshow(image_1)

#using 64 poitns from the whole image
codebook_random_2 = codebook(image_array, n_colors)

ll = KMeans(codebook_random_2, image_array, threshold, max_iterations)

image_2 = recreate_image(codebook_random_2, l, w, h, d)
image_2 = image_2.astype(float) / 255

plt.figure()
plt.imshow(image_2)

