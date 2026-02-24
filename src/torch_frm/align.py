from typing import Sequence, Optional
import torch

from .sh_rotational_correlator import (
    SHRotationalCorrelator, 
    find_rcf_peak_angles
)
from .euler import euler_zyz_to_matrix

def find_optimal_shift_fourier(
    x_ft: torch.Tensor, 
    reference_ft: torch.Tensor,
    shift_mask: Optional[torch.Tensor] = None
) -> Sequence[int]:
    correlation_ft = x_ft*reference_ft.conj()
    correlation = torch.fft.irfftn(correlation_ft)
    
    if shift_mask is not None:
        correlation *= shift_mask
        
    # TODO detect peaks
    
def find_optimal_rotation_sh(
    x_sh: torch.Tensor,
    reference_sh: torch.Tensor,
    correlator: SHRotationalCorrelator
) -> torch.Tensor:
    rcf = correlator.rcf(reference_sh, x_sh)
    alpha, beta, gamma = find_rcf_peak_angles(rcf)
    return euler_zyz_to_matrix(alpha[0], beta[0], gamma[0])

def align_volumes(
    x: torch.Tensor,
    ref: torch.Tensor,
    bandwidth: int = 64,
    max_shift: Optional[float] = None
) -> torch.Tensor:
    
    if max_shift is None:
        shift_mask = None
    else:
        shift_mask = None # TODO

    pass