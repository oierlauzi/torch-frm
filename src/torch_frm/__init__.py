from importlib.metadata import PackageNotFoundError, version

__author__ = "Oier Lauzirika Zarrabeitia"
__email__ = "olauzirika@cnb.csic.es"

from torch_frm.euler import euler_zyz_to_matrix
from torch_frm.frm import frm
from torch_frm.sample_3d import sample_3d
from torch_frm.sh_rotational_correlator import (
    SHRotationalCorrelator,
    find_rcf_peak_angles
)
from torch_frm.sh_volume_decomposer import SHVolumeDecomposer
from torch_frm.legendre import (
    associated_legendre_ortho, 
    ravel_associated_legendre_index
)
from torch_frm.wigner import wigner_matrices

try:
    __version__ = version("torch-frm")
except PackageNotFoundError:
    __version__ = "uninstalled"
    
__all__ = [
    frm,
    associated_legendre_ortho, 
    ravel_associated_legendre_index,
    wigner_matrices,
    sample_3d,
    find_rcf_peak_angles,
    euler_zyz_to_matrix,
    SHVolumeDecomposer,
    SHRotationalCorrelator
]
