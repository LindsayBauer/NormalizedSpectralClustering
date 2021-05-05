import numpy as np

'''
The functions in this module are used to lay the foundation needed to find all eigenvalues 
and eigenvectors of a real, symmetric, full rank matrix. 
'''

def gram_schmidt(A, N):
    ''' Decompose matrix A into an orthogonal matrix Q
     and an upper triangular matrix R ''' 
    U= A.copy()
    R= np.zeros((N, N))
    Q= np.zeros((N, N))
    for i in range(N):
        R[i,i]= np.linalg.norm(U[:,i])
        if R[i,i]!=0:
            Q_col_i = U[:, i] / R[i, i]
        if R[i,i]==0:
            Q_col_i = np.zeros(N)
        Q[:, i] = Q_col_i
        j = N-i-1
        if(j!=0):
            Qi_times_U = Q_col_i @ U
            R[i,i+1:N]= Qi_times_U[i+1:N]
            Q_col_i_broadcasted = np.array([Q_col_i]*j).transpose()
            U[:,i+1:N]= U[:,i+1:N] - (R[i,i+1:N]*Q_col_i_broadcasted)
    return Q,R


def QR(A, N):
    ''' Return a diagonal matrix Abar whose elements approach eigenvalues of A, 
    and an orthogonal matrix Qbar whose columns approach eigenvectors of A '''
    e = 0.0001
    Abar = A
    Qbar = np.identity(N)
    for _ in range(N):
        Q,R = gram_schmidt(Abar, N)
        Abar = R @ Q
        Qcurr = Qbar @ Q
        if np.allclose(np.abs(Qbar),np.abs(Qcurr),e):
            return Abar, Qbar
        Qbar = Qcurr
    return Abar, Qbar


def get_sorted_eigenvectors(Abar,Qbar,N):
    ''' Sort eigenvectors via corresponding eigenvalues 
    and return the sorted eigenvectors as matrix columns '''
    eigen_vectors = Qbar.transpose() # Each row is an eigenvector 
    eigen_values = Abar.diagonal()
    eigen_pairs = np.empty(N, dtype=object)
    for i in range(N):
        eigen_pairs[i] = (eigen_values[i], eigen_vectors[i])
    indexes = np.argsort(eigen_pairs)
    eigen_pairs = eigen_pairs[indexes]
    sorted_eigen_vectors = eigen_pairs[0][1]
    for i in range(1,len(indexes)):
        sorted_eigen_vectors = np.vstack((sorted_eigen_vectors, eigen_pairs[i][1]))
    return sorted_eigen_vectors.transpose()