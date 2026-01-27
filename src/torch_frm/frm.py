import torch

from .euler import euler_zyz_to_matrix
from .sh_volume_decomposer import SHVolumeDecomposer
from .sh_rotational_correlation import (
    SHRotationalCorrelation, 
    find_rcf_peak_angles
)

def frm(x: torch.Tensor, r: torch.Tensor, bandwidth: int = 32) -> torch.Tensor:
    """
    Perform a Fast Rotational Matching for a pair of volumes.
    
    Parameters
    -----------
    x: torch.Tensor
        Volume to be aligned. Shape (N, N, N)
    r: torch.Tensor
        Reference volume for alignment. Shape (N, N, N)
    bandwidth: int
        Number of spherical harmonic components to be used for alignment. The
        higher the number, the higher the precision but lower performance.
    
    Returns
    -------
    out: torch.Tensor
        A 2D view into a particular level of a wigner pyramid. Shape of 
        (2*l+1, 2*l+1)
    """
    
    if x.shape != r.shape:
        raise ValueError('Both input operands must have the same shape')
    
    if x.device != r.device:
        raise ValueError('Both input arguments must reside in the same device')
    
    dtype = torch.promote_types(x.dtype, r.dtype)
    device = r.device
    
    N = len(x)
    decomposer = SHVolumeDecomposer(
        bandwidth=bandwidth, 
        n_radii=N//2, 
        dtype=dtype, 
        device=device
    )
    correlation_function = SHRotationalCorrelation(
        bandwidth=bandwidth,
        dtype=dtype,
        device=device
    )
    
    sh_x = decomposer.transform(x)
    sh_r = decomposer.transform(r)
    rcf = correlation_function.rcf(sh_x, sh_r)

    alpha, beta, gamma = find_rcf_peak_angles(rcf)
    return euler_zyz_to_matrix(alpha, beta, gamma)
