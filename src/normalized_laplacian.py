import numpy as np

'''
The functions in this module are used to generate the normalized graph 
laplacian: A nxn matrix that has n non-negative real-valued eigenvalues 
and orthogonal real-values eigenvectors. 
'''

def generate_weighted_matrix(initial_vectors, N):
    ''' Construct weighted adjacency matrix '''
    W = np.zeros((N, N))  # Initialize a matrix of zeros
    for i in range(N):
        initial_vectors_minus_vec_i = initial_vectors - initial_vectors[i, :]
        W[:, i] = np.exp(np.linalg.norm(
            initial_vectors_minus_vec_i, 2, axis=1)/-2)
    np.fill_diagonal(W, val=0.0)
    return W


def get_l_norm(initial_vectors, N):
    ''' Calculate and return the l_norm matrix calculated from x '''
    W = generate_weighted_matrix(initial_vectors, N)
    D_inv = np.diag((np.apply_along_axis(np.sum, 1, W))**(-0.5))
    L_norm = np.identity(N) - D_inv @ W @ D_inv
    return L_norm
