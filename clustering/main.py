

import matplotlib.pyplot as plt
import numpy as np

from my_sol_kmeans import my_kmeans
#from my_sol_kmedoids_w_CUDA import my_kmedoids
from my_sol_kmedoids import my_kmedoids


def score(image_name, K):
    image = plt.imread(image_name)
    rows = image.shape[0]
    cols = image.shape[1]
    pixels = np.zeros((rows*cols,3))

    for i in range(rows):
        for j in range(cols):
            pixels[j*rows+i,:] = image[i,j,:]
    
    class1, centroid1 = my_kmeans(pixels, K)
    class2, centroid2 = my_kmedoids(pixels, K)
    
    converted_image1 = np.zeros((rows, cols, 3))
    converted_image2 = np.zeros((rows, cols, 3))
    
    for i in range(rows):
        for j in range(cols):
            converted_image1[i,j,:] = centroid1[class1[j*rows+i],:]
            converted_image2[i,j,:] = centroid2[class2[j*rows+i],:]
            
    converted_image1 /= 255
    converted_image2 /= 255
    
    f = plt.figure()
    f.set_figheight(30)
    f.set_figwidth(30)
    
    plt.subplot(1,3,1)
    plt.title('Origin')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.title('K-means')
    plt.imshow(converted_image1)
    plt.axis('off')
    
    plt.subplot(1,3,3)
    plt.title('K-medoids')
    plt.imshow(converted_image2)
    plt.axis('off')
    
    plt.show()
    return None

#score('ball.jpg', 5)
