from random import randrange
import numpy as np
import argparse
from sklearn.datasets import make_blobs
from normalized_laplacian import get_l_norm
from kmeans_pp import k_means_pp
import mykmeanssp as capi
import csv
from visualization import generate_visualization
from eigenvs import *

'''
This module orchestrates the entire program execution. It verifies the input received from the 
command line; sets the values of d, K and N accordingly; generates data; exports data to a .txt 
file; calls to the functions responsible for performing K-means clustering and normalized spectral 
clustering; and outputs the results to clusters.txt. Lastly, it calls to a function that 
generates a visualization of the clustering as a pdf.
'''

MAX_CAP_N_2 = 380
MAX_CAP_K_2 = 10
MAX_CAP_N_3 = 400
MAX_CAP_K_3 = 15
MAX_CAP_N = 1000
MAX_CAP_K = 1000
MAX_ITER = 300


def eigen_gap(eigen_values):
    ''' Return optimal K for spectral clustering '''
    limit = int(len(eigen_values)/2)
    gaps = eigen_values[1:limit+1] - eigen_values[:limit]
    return np.argmax(gaps) + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("K", type=int, help="The number of clusters")
    parser.add_argument("N", type=int, help="The number of observations")
    parser.add_argument("Random", type=int,
                        help="Indicates way data is generated")
    args = parser.parse_args()
    args.Random = True if args.Random == 1 else False

    # Determine d
    d = randrange(2,4)
    if d == 2:
        MAX_CAP_N = MAX_CAP_N_2
        MAX_CAP_K = MAX_CAP_K_2
    if d == 3:
        MAX_CAP_N = MAX_CAP_N_3
        MAX_CAP_K = MAX_CAP_K_3

    # Print informative message stating the maximum capacity of the project
    print(
        f"The maximum capacity for N is {MAX_CAP_N_2}, and for K is {MAX_CAP_K_2} when d is 2")
    print(
        f"The maximum capacity for N is {MAX_CAP_N_3}, and for K is {MAX_CAP_K_3} when d is 3")

    # Determine N and K
    if args.Random:
        N = randrange(MAX_CAP_N // 2, MAX_CAP_N)
        gen_K = randrange(MAX_CAP_K // 2, MAX_CAP_K)
    if not args.Random:
        if args.N <= 0 or args.K <= 0:
            raise Exception("Invalid input. Must provide positive N and K such that K < N.")
        elif args.K >= args.N:
            raise Exception("Invalid K, K must be smaller than N")
        else:
            gen_K = args.K
            N = args.N

    # Generate N d-dimensional random vectors
    initial_vectors, make_blobs_clusters = make_blobs(
        n_samples=N, centers=gen_K, n_features=d, random_state=0)
    np.array(initial_vectors)

    # Write the N generated data points into data.txt file along with
    # the integer label for cluster membership of each point
    writer = csv.writer(open('data.txt', 'w'))
    writer.writerows(tuple(f)+tuple(i) for f, i in zip(initial_vectors.copy().round(
        decimals=8), np.array(make_blobs_clusters, dtype=np.int32).reshape(N, 1)))
    del writer

    # Calculate Lnorm, the eigenvalues and the eigenvectors
    normalized_laplacian = get_l_norm(initial_vectors, N)
    Abar, Qbar = QR(normalized_laplacian, N)
    sorted_eigenvalues = np.sort(Abar.diagonal())
    sorted_eigenvectors = get_sorted_eigenvectors(Abar, Qbar, N)

    # Determine K that's to be used in both implementations
    # If Random is True - K is set to be the generated random K from earlier
    # If Random is False - K is set to be the best k, using eigen_gap
    if not args.Random:
        K = gen_K
    if args.Random:
        K = eigen_gap(sorted_eigenvalues)

    # Run normalized spectral clustering and write results into file
    if K == 1:
        print("K is 1, all vectors belong to same cluster. Terminating program.")
    else:
        U = sorted_eigenvectors[:, 0:K]  # Eigenvectors are received as columns
        T = (U.transpose() / np.linalg.norm(U, axis=1)).transpose()
        print(K, N, d)  # REMOVE THIS BEFORE SUBMITTING
        spectral_results = k_means_pp(K, N, K, MAX_ITER, T)
        with open("clusters.txt", "w") as f:
            wr = csv.writer(f)
            # Append number of clusters to file
            f.write(f"{len(spectral_results)}\n")
            # Append clusters computed by normalized spectral clustering
            wr.writerows(spectral_results)

        # Run K-Means and write results into file
        kmeans_results = k_means_pp(K, N, d, MAX_ITER, initial_vectors)
        with open("clusters.txt", "a") as f:
            wr = csv.writer(f)
            wr.writerows(kmeans_results)

        # Generate PDF of visualizations for the 2 implementations
        generate_visualization(N, K, gen_K, d, initial_vectors,
                               kmeans_results, spectral_results, make_blobs_clusters)
