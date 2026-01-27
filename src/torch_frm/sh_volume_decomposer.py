from typing import Optional
import torch
import math

from torch_frm import sample_3d
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

    def transform(self, volume: torch.Tensor) -> torch.Tensor:
        shells = sample_3d(volume, self.cartesian_)
        shells = shells.to(self.spherical_harmonics_.dtype)
        return torch.einsum('kij,hij->kh', shells, self._weights)
    