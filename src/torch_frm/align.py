from typing import Sequence, Optional
import torch
import numpy as np

from .sh_volume_decomposer import SHVolumeDecomposer
from .sh_rotational_correlator import SHRotationalCorrelator
from .peak_detection import _rcf_peak_indices_to_euler_zyz
from .euler import euler_zyz_to_matrix

def _make_shift_mask(max_shift: float, n: int) -> np.ndarray:
    max_shift2 = max_shift*max_shift
    index_to_shift = np.fft.fftfreq(n, d=1/n) # shifts have the same origin as fft
    index_to_shift2 = np.square(index_to_shift)
    total_shift2_grid = \
        index_to_shift2[:,None,None] + \
        index_to_shift2[None,:,None] + \
        index_to_shift2[None,None,:]

    out = np.zeros((n, )*3, dtype=np.bool_)
    out[total_shift2_grid <= max_shift2] = 1
    return out

def _make_shift_filter_fourier(
    shift: torch.Tensor, 
    n: int
) -> torch.Tensor:
    k = torch.fft.fftfreq(n, device=shift.device, dtype=shift.dtype)
    kp = torch.fft.rfftfreq(n, device=shift.device, dtype=shift.dtype)
    kz, ky, kx = torch.meshgrid(k, k, kp, indexing='ij')
    phase = -2*torch.pi * (kx*shift[2] + ky*shift[1] + kz*shift[0])
    return torch.exp(1j * phase)

def _index_to_shift(index: torch.Tensor, n: int) -> torch.Tensor:
    half = n // 2
    return torch.where(index < half, index, index-n)

def _find_optimal_shift_fourier(
    x_ft: torch.Tensor, 
    reference_ft: torch.Tensor,
    shift_mask: Optional[torch.Tensor] = None
) -> Sequence[int]:
    correlation_ft = x_ft*reference_ft.conj()
    correlation = torch.fft.irfftn(correlation_ft).numpy()
    
    if shift_mask is not None:
        correlation *= shift_mask
        
    N = len(correlation)
    indices = torch.unravel_index(torch.argmax(correlation), correlation.shape)
    return (_index_to_shift(index, N) for index in reversed(indices))
    
def _find_optimal_rotation_sh(
    x_sh: torch.Tensor,
    reference_sh: torch.Tensor,
    correlator: SHRotationalCorrelator
) -> Sequence[float]:
    rcf = correlator.rcf(reference_sh, x_sh)
    N = len(rcf)
    rcf = rcf[:,:(N//2+1), :]
    
    indices = torch.unravel_index(torch.argmax(rcf), rcf.shape)
    return _rcf_peak_indices_to_euler_zyz(indices, N)

def align_volumes(
    x: torch.Tensor,
    ref: torch.Tensor,
    bandwidth: int = 64,
    max_shift: Optional[float] = None,
    max_iter: int = 16,
    max_frequency: float = 0.25
) -> torch.Tensor:
    N = len(x)
    # TODO check same size
    
    if max_shift is None:
        shift_mask = None
    else:
        shift_mask = _make_shift_mask(max_shift, N)

    decomposer = SHVolumeDecomposer(
        bandwidth=bandwidth,
        n_radii=round(max_frequency*N),
        max_radius=max_frequency,
        device=ref.device,
        dtype=ref.dtype
    )

    ref_ft = torch.fft.rfftn(ref)
    ref_sh = decomposer.transform(ref_ft)
    rotation = torch.eye(3, device=x.device, dtype=x.dtype)
    for _ in range(max_iter):
        x_rotated = None # TODO
        
        x_ft = torch.fft.rfftn(x_rotated)
        shift = _find_optimal_shift_fourier(x_ft, ref_ft, shift_mask=shift_mask)
        x_ft *= _make_shift_filter_fourier(shift, N) # TODO decide sign
        
        x_sh = decomposer.transform(x_ft)
        angles = _find_optimal_rotation_sh(x_sh, ref_sh)
        delta_rotation = euler_zyz_to_matrix(angles[0], angles[1], angles[2]) # TODO decide inverse and intrinsic/extrinsic
        rotation = delta_rotation @ rotation # TODO decide ordering
    
    matrix = torch.empty((4, 4), device=x.device, dtype=x.dtype)
    matrix[:3,:3] = rotation
    matrix[:3,3] = rotation @ shift # TODO decide sign
    matrix[3,:3] = 0.0
    matrix[3,3] = 1.0
    
    return matrix
    