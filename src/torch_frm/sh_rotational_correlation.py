from typing import Optional, Tuple
import torch
import math

from .wigner import wigner_matrices

class SHRotationalCorrelation:
    """
    Class to compute the rotational cross-correlation function between two sets
    of spherical harmonic coefficients.
    """
    
    def __init__(
        self, 
        bandwidth: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
    ):
        self.bandwidth_ = bandwidth
        self._wigner_half_pi = wigner_matrices(
            torch.tensor(0.5*math.pi, dtype=dtype, device=device),
            self.bandwidth_
        )
    
    def rcf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the rotational cross-correlation function (RCF) between two sets
        of spherical harmonic coefficients.
        
        Parameters
        -----------
        x: torch.Tensor
            Spherical harmonic coefficients of the first volume. Shape
            (n_radii, bandwidth^2). Must have been computed with this 
            decomposer.
        y: torch.Tensor
            Spherical harmonic coefficients of the second volume. Shape
            (n_radii, bandwidth^2). Must have been computed with this 
            decomposer.
            
        Returns
        -------
        out: torch.Tensor
            The rotational cross-correlation function between the two volumes. 
            Shape (2*bandwidth, 2*bandwidth, 2*bandwidth).
        """
        dtype = torch.promote_types(x.dtype, y.dtype)
        rct_ft = torch.zeros(
            (2*self.bandwidth_, )*3, 
            dtype=dtype, 
            device=x.device
        )

        start_1d = 0
        start_2d = 0
        for l in range(self.bandwidth_):
            count = 2*l + 1
            end_1d = start_1d + count
            end_2d = start_2d + count*count
            
            d = self._wigner_half_pi[start_2d:end_2d].view(count, count)
            i = torch.einsum(
                'ki,kj->ij', 
                x[:,start_1d:end_1d], 
                y[:,start_1d:end_1d].conj()
            )
            
            term = d[:,:,None]*d[None,:,:]*i[:,None,:]
            central_range = slice(self.bandwidth_ - l, self.bandwidth_ + l + 1)
            rct_ft[central_range,central_range,central_range] += term
            
            start_1d = end_1d
            start_2d = end_2d
        
        rct_ft = torch.fft.fftshift(rct_ft)
        rct_ft = rct_ft[..., :rct_ft.shape[-1]//2 + 1]
        return torch.fft.irfftn(rct_ft)
        
def find_rcf_peak_angles(rcf: torch.Tensor) -> Tuple[float, float, float]:
    """
    Find the optimal alignment angles in a Rotational Correlation Function (RCF)
    
    The results are returned in euler ZYZ extrinsic convention.

    Parameters
    -----------
    rcf: torch.Tensor
        The rotational correlation function presumably computed by a 
        `SHVolumeDecomposer`

    Returns
    -------
    alpha: torch.Tensor
        First rotation around Z axis in radians.
    beta: torch.Tensor
        Second rotation around Y axis in radians.
    gamma: torch.Tensor
        Third rotation around Z axis in radians.
    """
    
    indices = torch.unravel_index(torch.argmax(rcf), rcf.shape)
    angles = 2*math.pi * (torch.tensor(indices) / torch.tensor(rcf.shape))
    xi, nu, omega = angles
    
    alpha = -xi
    beta = math.pi - nu
    gamma = math.pi - omega
    
    return alpha, beta, gamma
