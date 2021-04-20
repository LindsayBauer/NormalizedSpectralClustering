import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.backends.backend_pdf import PdfPages

'''
The functions of this module are used to generate a visualization of the calculated clusters
resulting from Normalized Spectral Clustering and K-means, along with the following descriptive
information: The k and n that the clusters were generated from, the k used for both algorithms, 
and the Jaccard measure for each algorithm.
'''

def generate_visualization(N, K, gen_K, d, initial_vectors, kmeans_results, spectral_results, make_blobs_clusters):
    ''' Create a PDF file containing visualizations of and descriptive information for the calculated clusters
    Input:
    initial_vectors - Array of N d-dimmension vectors
    results_kmeans - Array of k arrays. Index i holds indexes of vectors that 
            kmeans algorithm determined belong to cluster i
    results_spectral - Array of k arrays. Index i holds indexes of vectors that spectral clustering 
            algorithm determined belong to cluster i
    make_blobs_clusters - N length array. Index i holds the cluster that the vector in index i belongs to '''

    # Get Jaccard measure for each clustering algorithm to display quality of their clustering
    jac_spectral = round(get_jaccard(K, gen_K, spectral_results, make_blobs_clusters), 2)
    jac_kmeans = round(get_jaccard(K, gen_K, kmeans_results, make_blobs_clusters), 2)
    sentence1 = "Data was generated from the values:"
    sentence2 = f"n = {N} , k = {gen_K}"
    sentence3 = f"The k that was used for both algorithms was {K}"
    sentence4 = f"The Jaccard measure for Spectral Clustering: {jac_spectral}"
    sentence5 = f"The Jaccard measure for K-Means: {jac_kmeans}"

    with PdfPages('clusters.pdf') as pdf:
        # Get scatter plots of clusters for both implementations 
        if d == 2:
            plt = generate_scatter_plot_2d(
                initial_vectors, kmeans_results, spectral_results)
        else:
            plt = generate_scatter_plot_3d(
                initial_vectors, kmeans_results, spectral_results)

        # Add descriptive text noting the k and n, as well as 
        # the Jaccard measure calculated for each clustering algorithm
        plt.figtext(0.5, 0.23, s=sentence1, size=13,
                    horizontalalignment='center', verticalalignment='center')
        plt.figtext(0.5, 0.18, s=sentence2, size=13,
                    horizontalalignment='center', verticalalignment='center')
        plt.figtext(0.5, 0.13, s=sentence3, size=13,
                    horizontalalignment='center', verticalalignment='center')
        plt.figtext(0.5, 0.08, s=sentence4, size=13,
                    horizontalalignment='center', verticalalignment='center')
        plt.figtext(0.5, 0.03, s=sentence5, size=13,
                    horizontalalignment='center', verticalalignment='center')
        pdf.savefig()
        plt.close()


def get_jaccard(results_K,make_blobs_K,results, make_blobs_clusters):
    X = results_K**3
    cluster_sizes= [0]*results_K
    max_cluster_size = 0

    # Create an array with the size of each cluster, and save the max_cluster_size
    for i in range(results_K):
        cluster_sizes[i]=len(results[i])
        if cluster_sizes[i]>max_cluster_size:
            max_cluster_size = cluster_sizes[i]

    # Calculate the number of pairs in each cluster and the number of pairs overall
    num_of_pairs_in_each_cluster = np.apply_along_axis(get_num_of_pairs,0,cluster_sizes)
    num_of_pairs = np.sum(num_of_pairs_in_each_cluster)

    # Turn results into a NumPy array
    np_results = np.zeros((results_K, max_cluster_size), np.int)
    for i in range(results_K):
        one_row= np.array(results[i],np.int)
        one_row = np.pad(one_row,(0,max_cluster_size-cluster_sizes[i]),'constant', constant_values=(0, X))
        np_results[i]= one_row

    # Calculate the num of shared pairs in each cluster by changing each element in the cluster (in results)
    # into the index of the cluster that it belongs to in make_blobs,
    # and then counting the number of pairs with the same index in the same cluster.
    # Sum them to get the num of shared pairs overall
    cluster_in_make_blobs = np.apply_along_axis(get_cluster_in_make_blobs,1,np_results,make_blobs_clusters,X)
    num_of_shared_pairs_in_each_cluster = np.apply_along_axis(get_num_of_shared_pairs,1,cluster_in_make_blobs,make_blobs_K,X)
    num_of_shared_pairs = np.sum(num_of_shared_pairs_in_each_cluster)
    return num_of_shared_pairs/num_of_pairs

def get_num_of_pairs(a):
    return a*(a-1)/2

def get_cluster_in_make_blobs(cluster,make_blobs_clusters,X):
    ni = len(cluster)
    clusters_in_make_blobs = np.zeros(ni,np.int)
    for i in range(ni):
        index = cluster[i]
        if index==X:
            clusters_in_make_blobs[i]= X
        if index!=X:
            clusters_in_make_blobs[i]= make_blobs_clusters[index]
    return clusters_in_make_blobs

def get_num_of_shared_pairs(cluster,make_blobs_K,X):
    ni = len(cluster)
    index_counter = np.zeros(make_blobs_K,np.int)
    for i in range(ni):
        index= cluster[i]
        if index!=X:
            index_counter[index] = index_counter[index]+1
    pairs_counter = np.apply_along_axis(get_num_of_pairs,0,index_counter)
    return np.sum(pairs_counter)


def generate_scatter_plot_3d(vectors, clusters_kmeans, clusters_spectral):
    ''' Generate side-by-side 3D scatter plots of spectral and k-means clustering '''
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # Plot Normalized Spectral Clustering subplot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    for cluster in clusters_spectral:
        x = vectors[cluster, 0]
        y = vectors[cluster, 1]
        z = vectors[cluster, 2]
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        ax.scatter(x, y, z, color=color)
    plt.subplots_adjust(bottom=0.3)
    plt.title("Normalized Spectral Clustering")

    # Plot k-means subplot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    for cluster in clusters_kmeans:
        x = vectors[cluster, 0]
        y = vectors[cluster, 1]
        z = vectors[cluster, 2]
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        ax.scatter(x, y, z, color=color)
    plt.subplots_adjust(bottom=0.3)
    plt.title("K-means")
    return plt

def generate_scatter_plot_2d(vectors, clusters_kmeans, clusters_spectral):
    ''' Generate side-by-side 2D scatter plots of spectral and k-means clustering '''
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # Plot Normalized Spectral Clustering subplot
    ax = fig.add_subplot(1, 2, 1)
    for cluster in clusters_spectral:
        x = vectors[cluster, 0]
        y = vectors[cluster, 1]
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        ax.scatter(x, y, color=color)
    plt.subplots_adjust(bottom=0.3)
    plt.title("Normalized Spectral Clustering")

    # Plot k-means subplot
    ax = fig.add_subplot(1, 2, 2)
    for cluster in clusters_kmeans:
        x = vectors[cluster, 0]
        y = vectors[cluster, 1]
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        ax.scatter(x, y, color=color)
    plt.subplots_adjust(bottom=0.3)
    plt.title("K-means")
    return plt