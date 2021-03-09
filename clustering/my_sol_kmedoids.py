from copy import deepcopy
from my_sol_kmeans import my_kmeans
import numpy as np

def my_kmedoids(image_data, K, threshold = 0):

    
    N = image_data.shape[0]
    p = image_data.shape[1]    
   
    medoids = np.zeros((K, p))
    _, medoids = my_kmeans(image_data, K)

    medoids = np.asarray(medoids)
    medoids_old = np.zeros(medoids.shape)
    medoids_new = deepcopy(medoids) 
    error = np.linalg.norm(medoids_new - medoids_old)
    image_data = np.asarray(image_data)
    labels = np.zeros(N)

    DisMat = np.zeros((N,K))
    iter_ct = 0        
    while error > threshold:
        iter_ct+= 1
#        print('K-medoids iteration {}'.format(iter_ct))
        medoids_old = deepcopy(medoids_new)
        for i in range(K): # assign image_data points to closest centroids
            DisMat[:,i] = np.linalg.norm(image_data- medoids_new[i], axis = 1, ord = 2)
        labels  = np.argmin(DisMat, axis = 1)
        for i in range(K):
            cluster = image_data[labels == i]
            DMC = sum(np.linalg.norm(cluster - medoids_new[i], axis = 1, ord = 2))
            DMP = np.zeros(cluster.shape[0])
        if cluster.shape[0] == 0:
            medoids_new[i] = medoids_old[i]
        else:
            for j in range(cluster.shape[0]):
                DMP[j] = np.sum(np.linalg.norm(cluster - cluster[j], axis = 1, ord = 2))
            small_cost_idx = np.argmin(DMP)
            if DMP[small_cost_idx] < DMC:
                medoids_new[i] = cluster[small_cost_idx]
        error = np.linalg.norm(medoids_new - medoids_old)
    return labels.astype(int), medoids
    
