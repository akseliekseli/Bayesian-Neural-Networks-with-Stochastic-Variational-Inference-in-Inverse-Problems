import numpy as np
from scipy.ndimage import convolve1d


class deconvolution:
    def __init__(self, PSF_size, PSF_param, BC):
        self.BC = BC # {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}
        self.P = self.PSF_Gauss(PSF_size, PSF_param)

    def PSF_Gauss(self, dim, s):
        # for Gaussian blur (astronomic turbulence)
        # Set up grid points to evaluate the Gaussian function
        x = np.arange(-np.fix(dim/2), np.ceil(dim/2))

        # Compute the Gaussian, and normalize the PSF.
        PSF = np.exp(-0.5 * ((x**2) / (s**2)))
        PSF /= PSF.sum()

        # find the center
        # center = np.where(PSF == PSF.max())[0][0]
        return PSF

    def linear_operator(self, n):
        # build the matrix
        ee = np.eye(n)
        A = np.array([self.forward(ee[:, i]) for i in range(n)])
        return A

    def forward(self, X):
        # forward projection
        return convolve1d(X, self.P, mode=self.BC, axis=0)