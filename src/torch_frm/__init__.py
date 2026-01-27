from importlib.metadata import PackageNotFoundError, version

__author__ = "Oier Lauzirika Zarrabeitia"
__email__ = "olauzirika@cnb.csic.es"

from torch_frm.spherical_harmonics import (
    spherical_harmonics, 
    ravel_spherical_harmonic_index
)

from torch_frm.wigner import wigner_matrices

try:
    __version__ = version("torch-frm")
except PackageNotFoundError:
    __version__ = "uninstalled"
    
__all__ = [
    spherical_harmonics,
    ravel_spherical_harmonic_index,
    wigner_matrices
]