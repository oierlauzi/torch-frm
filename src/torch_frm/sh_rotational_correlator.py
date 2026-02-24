from typing import Optional
import torch
import math

from .wigner import wigner_matrices

class SHRotationalCorrelator:
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
            (n_radii, bandwidth^2).
        y: torch.Tensor
            Spherical harmonic coefficients of the second volume. Shape
            (n_radii, bandwidth^2).
            
        Returns
        -------
        out: torch.Tensor
            The rotational cross-correlation function between the two volumes. 
            Shape (2*bandwidth, 2*bandwidth, 2*bandwidth).
        """
        
        n_radii = x.shape[0]
        expected_shape = (n_radii, self.bandwidth_**2)
        expected_device = self._wigner_half_pi.device
        if x.shape != expected_shape:
            raise ValueError(
                f"Expected x to have shape {expected_shape}, "
                f"but got {x.shape}."
            )
        if x.device != expected_device:
            raise ValueError(
                f"x is on device {x.device}, but correlator is on "
                f"device {expected_device}."
            )
        if y.shape != expected_shape:
            raise ValueError(
                f"Expected y to have shape {expected_shape}, "
                f"but got {y.shape}."
            )
        if y.device != expected_device:
            raise ValueError(
                f"y is on device {y.device}, but correlator is on "
                f"device {expected_device}."
            )
        
        dtype = torch.promote_types(x.dtype, y.dtype)
        rcf_ft = torch.zeros(
            (2*self.bandwidth_, )*3, 
            dtype=dtype, 
            device=self._wigner_half_pi.device
        )

        start_1d = 0
        start_2d = 0
        for l in range(self.bandwidth_):
            count = 2*l + 1
            end_1d = start_1d + count
            end_2d = start_2d + count*count
            
            d = self._wigner_half_pi[start_2d:end_2d].view(count, count).to(dtype)
            term = torch.einsum(
                'ip,ir,pq,qr->pqr', 
                x[:,start_1d:end_1d], 
                y[:,start_1d:end_1d].conj(),
                d,
                d
            )
            central_range = slice(self.bandwidth_ - l, self.bandwidth_ + l + 1)
            rcf_ft[central_range,central_range,central_range] += term
            
            start_1d = end_1d
            start_2d = end_2d
        
        rcf_ft = torch.fft.fftshift(rcf_ft)
        rcf_ft = rcf_ft[..., :(rcf_ft.shape[-1]//2 + 1)]
        return torch.fft.irfftn(rcf_ft)
