#% Your goal of this assignment is implementing your own K-medoids.
#% Please refer to the instructions carefully, and we encourage you to
#% consult with other resources about this algorithm on the web.
#%
#% Input:
#%     pixels: data set. Each row contains one data point. For image
#%     dataset, it contains 3 columns, each column corresponding to Red,
#%     Green, and Blue component.
#%
#%     K: the number of desired clusters. Too high value of K may result in
#%     empty cluster error. Then, you need to reduce it.
#%
#% Output:
#%     class: the class assignment of each data point in pixels. The
#%     assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
#%     of class should be either 1, 2, 3, 4, or 5. The output should be a
#%     column vector with size(pixels, 1) elements.
#%
#%     centroid: the location of K centroids in your result. With images,
#%     each centroid corresponds to the representative color of each
#%     cluster. The output should be a matrix with K rows and
#%     3 columns. The range of values should be [0, 255].
#%     
#%
#% You may run the following line, then you can see what should be done.
#% For submission, you need to code your own implementation without using
#% the kmeans matlab function directly. That is, you need to comment it out.

# from sklearn.cluster import KMeans

# def my_kmedoids(image_data, K):
#     kmeans = KMeans(n_clusters=K).fit(image_data)
#     label = kmeans.labels_
#     centroid = kmeans.cluster_centers_
#     return label, centroid

def my_kmedoids(image_data, K, threshold = 0):
    from copy import deepcopy
    from my_sol_kmeans import my_kmeans
    import numpy as np
    import cupy as cp
    
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
    