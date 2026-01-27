from typing import Optional
import torch
import math

from torch_frm import sample_3d, spherical_harmonics, wigner_matrices
from torch_frm import spherical_harmonics

def _spherical_to_cartesian(
    theta: torch.Tensor,
    phi: torch.Tensor
) -> torch.Tensor:
    batch_shape = torch.broadcast_shapes(theta.shape, phi.shape)
    dtype = torch.promote_types(theta.dtype, phi.dtype)

    out = torch.empty(batch_shape + (3, ), dtype=dtype, device=theta.device)

    out[...,2] = torch.cos(theta)
    rho = torch.sin(theta)
    out[...,0] = rho*torch.cos(phi)
    out[...,1] = rho*torch.sin(phi)

    return out

class SHVolumeDecomposer:
    def __init__(
        self, 
        bandwidth: int, 
        n_radii: int, 
        min_radius: float = 0.0,
        max_radius: float = 1.0,
        n_theta: Optional[int] = None,
        n_phi: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        n_theta = n_theta or 2*bandwidth
        n_phi = n_phi or 2*bandwidth
        
        self.bandwidth_ = bandwidth
        
        self.theta_ = torch.linspace(
            0.0,
            math.pi,
            n_theta,
            dtype=dtype, 
            device=device
        )
        
        phi_step = (2*math.pi)/n_phi
        self.phi_ = phi_step*torch.arange(n_phi, dtype=dtype, device=device)
        
        self.radii_ = torch.linspace(
            min_radius, 
            max_radius, 
            n_radii, 
            dtype=dtype, 
            device=device
        )
        
        self.theta_grid_ = self.theta_[None,None,:]
        self.phi_grid_ = self.phi_[None,:,None]
        self.radii_grid_ = self.radii_[:,None,None]
        
        u = _spherical_to_cartesian(self.theta_, self.phi_)
        self.cartesian_ = self.radii_ * u
        
        self.spherical_harmonics_ = spherical_harmonics(
            self.theta_grid_, 
            self.phi_grid_, 
            self.bandwidth_
        )
        
        self._weights = \
            torch.sin(self.theta_grid_)*self.spherical_harmonics_.conj()
            
        self._wigner_half_pi = wigner_matrices(
            torch.tensor(0.5*math.pi, dtype=dtype, device=device),
            self.bandwidth_
        )

    def transform(self, volume: torch.Tensor) -> torch.Tensor:
        shells = sample_3d(volume, self.cartesian_)
        shells = shells.to(self.spherical_harmonics_.dtype)
        return torch.einsum('kij,hij,k->kh', shells, self._weights, self.radii_)
    
    def compute_rcf(self, x_sh: torch.Tensor, y_sh: torch.Tensor) -> torch.Tensor:
        dtype = torch.promote_types(x_sh.dtype, y_sh.dtype)
        rct_ft = torch.zeros(
            (2*self.bandwidth_, )*3, 
            dtype=dtype, 
            device=x_sh.device
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
                x_sh[start_1d:end_1d], 
                y_sh[start_1d:end_1d].conj()
            )
            
            term = d[:,:,None]*d[None,:,:]*i[:,None,:]
            central_range = slice(self.bandwidth_ - l, self.bandwidth_ + l + 1)
            rct_ft[central_range,central_range,central_range] += term
            
            start_1d = end_1d
            start_2d = end_2d
            
        rct_ft = torch.fft.fftshift(rct_ft)
        return torch.fft.ifftn(rct_ft)
            