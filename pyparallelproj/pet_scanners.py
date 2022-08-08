import numpy as np
import matplotlib.pyplot as plt
try:
    import cupy as cp
except:
    import numpy as np


class RegularPolygonPETScanner:
    """Geometry defition of a cylindical PET scanner with pixelized detectors aranged in module
       that form a regular polygon in the trans-axial plane

       Parameters
       ----------
       R : float
         the radium of the scanner (in mm)

       ncrystals_per_module : numpy array of two ints
         number of crystals per module in the trans-axial and axial direction

       crystal_size : numpy array of two floats
         the crystal diameter in trans-axial and axial direction (in mm)

       nmodules : numpy array of two ints
         the number of modules in the angular and axial direction

       module_gap_axial : float
         the gap between two modules in axial direction (in mm)
    """

    def __init__(self,
                 R=325.,
                 ncrystals_per_module=np.array([16, 9]),
                 crystal_size=np.array([4., 5.]),
                 nmodules=np.array([28, 5]),
                 module_gap_axial=5.,
                 on_gpu=False):

        self._on_gpu = on_gpu

        self.R = R
        self.ncrystals_per_module = ncrystals_per_module
        self.crystal_size = crystal_size
        self.nmodules = nmodules
        self.module_gap_axial = module_gap_axial

        self.ncrystals_per_plane = nmodules[0] * ncrystals_per_module[0]
        self.ncrystals_axial = nmodules[1] * ncrystals_per_module[1]

        self.calculate_crystal_coordinates()

    def calculate_crystal_coordinates(self):
        # the distance of a crystal to the center of the module
        d = (np.arange(self.ncrystals_per_module[0]) -
             self.ncrystals_per_module[0] / 2 + 0.5) * self.crystal_size[0]

        # x0 and x1 crystal coordinates of one ring (single layer of crystals)
        self.xc0 = np.zeros(self.ncrystals_per_plane, dtype=np.float32)
        self.xc1 = np.zeros(self.ncrystals_per_plane, dtype=np.float32)

        self.alpha_module = np.linspace(0, 2 * np.pi,
                                        self.nmodules[0] + 1)[:-1]

        for i, alpha in enumerate(self.alpha_module):
            self.xc0[i * self.ncrystals_per_module[0]:(i + 1) *
                     self.ncrystals_per_module[0]] = self.R * np.cos(
                         alpha) - d * np.sin(alpha)
            self.xc1[i * self.ncrystals_per_module[0]:(i + 1) *
                     self.ncrystals_per_module[0]] = self.R * np.sin(
                         alpha) + d * np.cos(alpha)

        self.xc2 = np.zeros(self.ncrystals_axial, dtype=np.float32)

        for i in range(self.nmodules[1]):
            self.xc2[i * self.ncrystals_per_module[1]:(i + 1) *
                     self.ncrystals_per_module[1]] = (
                         np.arange(self.ncrystals_per_module[1]) *
                         self.crystal_size[1] + i *
                         (self.ncrystals_per_module[1] * self.crystal_size[1] +
                          self.module_gap_axial))

        # shift center in x2 direction to 0
        self.xc2 -= 0.5 * self.xc2.max()

        # move crystal coordinate arrays to GPU
        if self._on_gpu:
            self.xc0 = cp.asarray(self.xc0)
            self.xc1 = cp.asarray(self.xc1)
            self.xc2 = cp.asarray(self.xc2)

    def show_crystal_config(self, show_crystal_numbers=False):

        if self._on_gpu:
            xc0 = cp.asnumpy(self.xc0)
            xc1 = cp.asnumpy(self.xc1)
            xc2 = cp.asnumpy(self.xc2)
        else:
            xc0 = self.xc0
            xc1 = self.xc1
            xc2 = self.xc2

        fig, ax = plt.subplots(1, 2, figsize=(12, 7))
        ax[0].plot(xc0, xc1, 'r.')

        ax[1].plot(xc2, np.full(self.ncrystals_axial, xc1.max()), 'r.')
        ax[1].plot(xc2, np.full(self.ncrystals_axial, xc1.min()), 'r.')

        ax[0].set_xlabel('xc0')
        ax[0].set_ylabel('xc1')
        ax[1].set_xlabel('xc2')
        ax[1].set_ylabel('xc1')

        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        for axx in ax.flatten():
            axx.grid(ls=':')

        if show_crystal_numbers:
            for i in range(self.ncrystals_per_plane):
                ax[0].text(xc0[i], xc1[i], str(i))
            for i in range(self.ncrystals_axial):
                ax[1].text(xc2[i], xc1.max(), str(i))
                ax[1].text(xc2[i], xc1.min(), str(i))

        fig.tight_layout()
        fig.show()

        return fig, ax

    def get_crystal_coordinates(self, crystal_inds):
        """ get the world coordinates for a number of crystals specified with a (transaxial, axial) 
            crystal ID

            Parameters
            ----------
            crystal_inds : 2D numpy int array of shape (n,2)
              containing the trans-axial and axial crystal ID of n detectors for which to calculate
              the world coordinates

            Returns
            -------
            2D numpy or cupy array of shape (n,3) containing the three world coordinates 
            of the detectors
        """
        if self._on_gpu:
            return cp.dstack(
                (self.xc0[crystal_inds[:, 0]], self.xc1[crystal_inds[:, 0]],
                 self.xc2[crystal_inds[:, 1]])).squeeze()
        else:
            return np.dstack(
                (self.xc0[crystal_inds[:, 0]], self.xc1[crystal_inds[:, 0]],
                 self.xc2[crystal_inds[:, 1]])).squeeze()
