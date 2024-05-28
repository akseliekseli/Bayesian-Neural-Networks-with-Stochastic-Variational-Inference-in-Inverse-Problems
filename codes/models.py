import numpy as np
from scipy.ndimage import convolve1d
# import matplotlib.pyplot as plt

#================================================================================
class diffusion:
    def __init__(self, t, s, opt):
        self.t = t            # discretization points
        self.n = len(t)       # number of discretization points
        self.h = t[1]-t[0]    # discretization size
        self.u_r = 1          # Dirichlet datum
        self.s = s            # set source

        # set source integrated
        if (opt == 1):
            # source is spatially uniform
            self.s_int = self.s*self.t

        elif (opt == 2):
            # source is spatially variable
            self.s_int = self.h*np.cumsum(self.s, dtype=float)

    def forward(self, x, F):
        # integration constants
        c1 = -F
        c2 = self.u_r - np.trapz((c1 - self.s_int)/x, self.t)

        # semi-analytical solution
        y = self.h*np.cumsum((c1 - self.s_int)/x, dtype=float) + c2
        return y

#================================================================================
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