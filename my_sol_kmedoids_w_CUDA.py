from copy import deepcopy
from my_sol_kmeans import my_kmeans
import numpy as np
import cupy as cp

def my_kmedoids(image_data, K, threshold = 0):
    """
    This is the cuda implementation of my_kmedoids. Requires cuda to function properly.
    """
    
    N = image_data.shape[0]
    p = image_data.shape[1]    
   
    medoids = np.zeros((K, p))
    _, medoids = my_kmeans(image_data, K)

    medoids = cp.asarray(medoids)
    medoids_old = cp.zeros(medoids.shape)
    medoids_new = deepcopy(medoids) 
    error = cp.linalg.norm(medoids_new - medoids_old)
    image_data = cp.asarray(image_data)
    labels = cp.zeros(N)

    DisMat = cp.zeros((N,K))
    iter_ct = 0        
    while error > threshold:
        iter_ct+= 1
        #print('K-medoids iteration {}'.format(iter_ct))
        medoids_old = deepcopy(medoids_new)
        for i in range(K): # assign image_data points to closest centroids
            DisMat[:,i] = cp.linalg.norm(image_data- medoids_new[i], axis = 1)
        labels  = cp.argmin(DisMat, axis = 1)
        for i in range(K):
            cluster = image_data[labels == i]
            DMC = sum(cp.linalg.norm(cluster - medoids_new[i], axis = 1))
            DMP = cp.zeros(cluster.shape[0])
        if cluster.shape[0] == 0:
            medoids_new[i] = medoids_old[i]
        else:
            for j in range(cluster.shape[0]):
                DMP[j] = cp.sum(cp.linalg.norm(cluster - cluster[j], axis = 1))
            small_cost_idx = cp.argmin(DMP)
            if DMP[small_cost_idx] < DMC:
                medoids_new[i] = cluster[small_cost_idx]
        error = cp.linalg.norm(medoids_new - medoids_old)
    return cp.asnumpy(labels.astype(int)), cp.asnumpy(medoids)
    
