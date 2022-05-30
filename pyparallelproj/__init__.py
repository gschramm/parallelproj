from .pet_scanners  import RegularPolygonPETScanner
from .projectors    import SinogramProjector
from .sinogram      import PETSinogramParameters
from .models        import PETAcqModel, LMPETAcqModel, ImageBasedResolutionModel, GradientNorm, GradientOperator, GradientBasedPrior
from .utils         import EventMultiplicityCounter
from .algorithms    import OSEM, OSEM_EMTV, SPDHG, LM_OSEM, LM_OSEM_EMTV, LM_SPDHG, PDHG_L2_Denoise
from .phantoms      import ellipse2d_phantom, brain2d_phantom

# this is needed to get the package version at runtime
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
