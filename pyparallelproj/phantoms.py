import numpy as np
import os
import pyparallelproj

from scipy.ndimage import zoom


def ellipse_inds(nx, ny, rx, ry=None, x0=0, y0=0):

    if ry == None:
        ry = rx

    x = np.arange(nx) - nx / 2 + 0.5
    y = np.arange(ny) - ny / 2 + 0.5

    X, Y = np.meshgrid(x, y, indexing='ij')

    return np.where((((X - x0) / rx)**2 + ((Y - y0) / ry)**2) <= 1)


# --------------------------------------------------------------
def ellipse2d_phantom(n=256, c=3):
    r = n / 6

    img = np.zeros((n, n), dtype=np.float32)
    i0 = ellipse_inds(n, n, n / 4, n / 2.2)
    i1 = ellipse_inds(n, n, n / 32, n / 32)

    phis = np.linspace(0, 2 * np.pi, 9)[:-1]

    img[i0] = 1
    img[i1] = 0

    for i, phi in enumerate(phis):
        i = ellipse_inds(n,
                         n,
                         np.sqrt(i + 1) * n / 80,
                         np.sqrt(i + 1) * n / 80,
                         x0=r * np.sin(phi),
                         y0=r * np.cos(phi))
        img[i] = c

    return img


# --------------------------------------------------------------


def brain2d_phantom(n=128):
    img = np.load(
        os.path.join(os.path.dirname(pyparallelproj.__file__), 'data',
                     'brain2d.npy'))

    if n != 128:
        img = zoom(img, n / 128, order=1, prefilter=False)

    # due to floating point predicision shape of img can be different from n
    if img.shape[0] > n:
        img = img[:n, :]
    if img.shape[1] > n:
        img = img[:, :n]

    if img.shape[0] < n:
        img = np.pad(img, ((0, n - img.shape[0]), (0, 0)))
    if img.shape[1] < n:
        img = np.pad(img, ((0, 0), (0, n - img.shape[1])))

    return img
