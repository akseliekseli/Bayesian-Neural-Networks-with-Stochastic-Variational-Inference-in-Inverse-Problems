"""
---------------------------------------------------------------------------
Created by: 
Felipe Uribe (felipe.uribe@lut.fi)
---------------------------------------------------------------------------
Version 2024-01
---------------------------------------------------------------------------
"""
import numpy as np
import scipy as sp
from scipy.sparse import spdiags

#=============================================================================
def Nystrom(x, d_k, n_GL, R_nu, sigma2):
    # domain data    
    n = x.size
    a = (x[-1]-x[0])/2     # scale

# compute the Gauss-Legendre abscissas and weights
    xi, w = np.polynomial.legendre.leggauss(n_GL)

    # transform nodes and weights to [0, L]
    xi_s = a*xi + a
    w_s = a*w

    # compute diagonal matrix 
    D = spdiags(np.sqrt(w_s), 0, n_GL, n_GL).toarray()
    S1 = np.tile( np.sqrt(w_s).reshape(1, n_GL), (n_GL, 1))
    S2 = np.tile( np.sqrt(w_s).reshape(n_GL, 1), (1, n_GL))
    S = S1*S2

    # compute covariance matrix at quadrature nodes
    Sigma_nu = np.empty((n_GL, n_GL))
    for i in range(n_GL):
        for j in range(i, n_GL):
            Sigma_nu[i, j] = sigma2*R_nu(xi_s[i], xi_s[j])
            Sigma_nu[j, i] = Sigma_nu[i, j]
            # if (i == j):
            #     Sigma_nu[i, j] += sigma2
    # plt.figure(),    plt.imshow(Sigma_nu),    plt.pause(1)

    # solve the eigenvalue problem
    A = Sigma_nu*S     # D_sqrt*Sigma_nu*D_sqrt
    L, h = sp.sparse.linalg.eigsh(A, d_k, which='LM')     # np.linalg.eig(A)         
    idx = np.argsort(-np.real(L))     # index sorting descending

    # order the results
    eigval = np.real(L[idx])
    h = h[:, idx]

    # replace for the actual eigenvectors
    phi = np.linalg.solve(D, h)

    # Nystrom's interpolation formula
    # recompute covariance matrix on partition nodes and quadrature nodes
    Sigma_nu = np.array([sigma2*R_nu(x[i], xi_s[j]) for i in range(n) for j in range(n_GL)]).reshape(n, n_GL)
    M1 = Sigma_nu * np.tile(w_s.reshape(n_GL, 1), (1, n)).T
    M2 = phi @ np.diag(1/eigval)
    eigvec = M1 @ M2
    return eigval, eigvec