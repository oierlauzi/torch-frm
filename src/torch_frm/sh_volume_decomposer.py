from typing import Optional
import torch
import math

from .legendre import associated_legendre_ortho
from .sample_3d import sample_3d

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
    """
    Class to decompose 3D volumes into spherical harmonics.
    """
    
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
        """
        Parameters
        -----------
        bandwidth: int
            Number of spherical harmonic components to be used for alignment. The
            higher the number, the higher the precision but lower performance.
        n_radii: int
            Number of radial shells to be used.
        min_radius: float
            Minimum radius to be extracted. Normalized to the volume radius.
        max_radius: float
            Maximum radius to be extracted. Normalized to the volume radius.
        n_theta: Optional[int]
            Number of polar angles to be used. If None, it will be set to 
            2*bandwidth.
        n_phi: Optional[int]
            Number of azimuthal angles to be used. If None, it will be set to 
            2*bandwidth.
        device: Optional[torch.device]
            Device where to allocate internal tensors. If None, the default
            torch device will be used.
        dtype: Optional[torch.dtype]
            Data type for internal tensors. If None, the default torch dtype
            will be used.
        """
        
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
        
        self.theta_grid_ = self.theta_[:,None]
        self.phi_grid_ = self.phi_[None,:]
        
        u = _spherical_to_cartesian(self.theta_grid_, self.phi_grid_)
        self.cartesian_grid_ = self.radii_[:,None,None,None] * u
        
        self.associated_legendre_ = associated_legendre_ortho(
            self.theta_, 
            self.bandwidth_
        )
        
        self._theta_integration_kernel = \
            torch.sin(self.theta_)*self.associated_legendre_
            
    def transform(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Decompose a volume into spherical harmonics shell by shell.
        
        Parameters
        -----------
        volume: torch.Tensor
            The input volume to be decomposed. Shape (N, N, N). Must be on
            the same device and have the same dtype as the decomposer.
            
        Returns
        -------
        out: torch.Tensor
            The spherical harmonic coefficients of the volume shells. Shape
            (n_radii, bandwidth^2)
        """
        
        shells = sample_3d(volume, self.cartesian_grid_)
        
        if volume.is_complex():
            raise NotImplementedError("Complex volumes are not supported yet.")
        else:
            return self._transform_real(shells)
      
    def _transform_real(self, shells: torch.Tensor) -> torch.Tensor:
        shells_ft = torch.fft.rfft(shells, dim=2) # Phi direction
        
        theta_integration_kernel = self._theta_integration_kernel.to(shells_ft.dtype)
        radii = self.radii_.to(shells_ft.dtype)
        
        out = torch.zeros(
            (len(self.radii_), self.bandwidth_*self.bandwidth_),
            dtype=shells_ft.dtype,
            device=shells_ft.device
        )
        
        for l in range(self.bandwidth_):
            base_out = l*(l+1)
            base_in = base_out // 2
            
            phase = 1
            for m in range(0, l+1):
                out[:,base_out+m] = torch.einsum(
                    'ji,i,j->j', 
                    shells_ft[:,:,m],
                    phase*theta_integration_kernel[base_in+m],
                    radii
                )
                phase = -phase
            
            phase = -1
            for m in range(1, l+1):
                out[:,base_out-m] = phase*out[:,base_out-m].conj()
                phase = -phase
            
        return out
