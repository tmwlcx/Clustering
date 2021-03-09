#% Your goal of this assignment is implementing your own K-means.
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

# def my_kmeans(image_data, K):
#     kmeans = KMeans(n_clusters=K).fit(image_data)
#     label = kmeans.labels_
#     centroid = kmeans.cluster_centers_
#     return label, centroid

def my_kmeans(image_data, K, threshold = 0):
    from copy import deepcopy
    import numpy as np

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
