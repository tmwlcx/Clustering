from copy import deepcopy
import numpy as np
    
def my_kmeans(image_data, K, threshold = 0):


    N = image_data.shape[0]
    p = image_data.shape[1]
    
    image_crap = deepcopy(image_data)
    image_crap = np.prod(image_crap, axis = 1)
    image_crap = image_data[np.argsort(image_crap)]

    centroids = np.zeros((K,p))

    for i in range(K):
        centroids[i] = np.mean(image_crap[i*int(image_data.shape[0]/K):(i+1)*int(image_data.shape[0]/K)], axis = 0)

    labels = np.zeros(N)

    centroids_old = np.zeros(centroids.shape)
    centroids_new = deepcopy(centroids)
    

    DisMat = np.zeros((N,K))
    error = np.linalg.norm(centroids_new - centroids_old)
    iter_count = 0

    while error > threshold:
        iter_count += 1
        centroids_old = deepcopy(centroids_new)
        for i in range(K): # assign image_data points to closest centroids
            DisMat[:,i] = np.linalg.norm(image_data- centroids_new[i], axis = 1)
        labels  = np.argmin(DisMat, axis = 1)
        for i in range(K):
            if centroids_new[i].size == 0: # if new centroid has no cluster, use old centroid
                centroids_new[i] = centroids_old[i]
            else:
                centroids_new[i] = np.mean(image_data[labels == i], axis = 0)
        error = np.linalg.norm(centroids_new - centroids_old)
    return labels.astype(int), centroids_new.astype(int)
