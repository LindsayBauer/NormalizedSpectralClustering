import mykmeanssp as capi
import numpy as np

'''
This module implements the K-means++ algorithm used to choose initial centroids
for the K-means algorithm. The function k_means_pp() passes the chosen centroids
to the capi, which implements the K-means algorithm, and returns the vectors
organized by cluster. 
'''

def distance(v1, v2):
    return np.sum(np.power((v1 - v2),2))

def update_DList(new_index, initial_centroids_py, input_vectors_py, Dlist):
    ''' Find min distance from vectors to closest cluster '''
    new_distances = np.apply_along_axis(distance, 1, input_vectors_py, input_vectors_py[initial_centroids_py[new_index]])
    return np.minimum(Dlist, new_distances)

def k_means_pp(K, N, d, MAX_ITER, initial_vectors):
    ''' K-means++ algorithm for clustering '''
    # Regard each row of initial_vectors as a vector
    np.random.seed(0)

    # The indexes of the K initial centroids 
    initial_centroids_py = np.full(K, fill_value=(-1)) 
    Dlist = np.full(N,fill_value=(np.inf)) 

    # Index of first initial centroid to be selected 
    initial_centroids_py[0] = np.random.choice(N, 1)
    
    for j in range(1,K):
        Dlist = update_DList(j-1, initial_centroids_py, initial_vectors, Dlist)
        distance_sum = np.sum(Dlist)
        probs_list = np.true_divide(Dlist, distance_sum)
        initial_centroids_py[j] = np.random.choice(N, 1, p=probs_list)

    # Convert NumPy arrays into python lists
    input_vectors_py = initial_vectors.tolist()
    initial_centroids_py = initial_centroids_py.tolist()

    ## Call C program to get an N sized array where index i holds the 
    # index of the cluster that vector i belongs to. 
    n_sized_results = capi.Ckmeans(K, N, d, MAX_ITER, input_vectors_py, initial_centroids_py)
    
    # Create an array of K subarrays
    # Subarray i holds the indexes of the vectors belonging to cluster i 
    k_sized_results = [[] for _ in range(K)]
    for i in range(N):
        k_sized_results[n_sized_results[i]].append(i)
    return k_sized_results