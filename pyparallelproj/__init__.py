from .pet_scanners import RegularPolygonPETScanner
from .projectors   import SinogramProjector, LMProjector
from .sinogram     import PETSinogramParameters

# this is needed to get the package version at runtime
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
