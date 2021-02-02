from .pet_scanners  import RegularPolygonPETScanner
from .projectors    import SinogramProjector, LMProjector
from .sinogram      import PETSinogramParameters
from .models        import pet_fwd_model, pet_back_model, pet_fwd_model_lm, pet_back_model_lm

# this is needed to get the package version at runtime
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
