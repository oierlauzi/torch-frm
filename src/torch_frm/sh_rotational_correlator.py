from typing import Optional, Tuple, Set
import torch
import math
import numpy as np
import scipy.ndimage
import scipy.sparse

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

def _find_common_labels(
    first_face: torch.Tensor, 
    last_face: torch.Tensor
) -> Set[int]:
    merges = set()
    
    both_faces = (first_face > 0) & (last_face > 0)
    for i, j in zip(first_face[both_faces], last_face[both_faces]):
        if i < j:
            merges.add((i-1, j-1))
        elif j < i:
            merges.add((j-1, i-1))
            
    return merges

def _label_rcf_peaks(
    mask: torch.Tensor, 
    structure: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, int]:
    labels, n_labels = scipy.ndimage.label(mask, structure=structure)

    # Trivial case
    if n_labels <= 1:
        return labels, n_labels

    top_face = labels[0,:,:]
    bottom_face = labels[-1,:,:]
    left_face = labels[:,:,0]
    right_face = labels[:,:,-1]

    merges = set()
    merges.update(_find_common_labels(top_face, bottom_face))
    merges.update(_find_common_labels(left_face, right_face))

    if not merges:
        return labels, n_labels

    graph_edges = np.array(list(merges), dtype=int)
    graph = scipy.sparse.coo_array(
        (np.ones(len(graph_edges)), (graph_edges[:, 0], graph_edges[:, 1])),
        shape=(n_labels, n_labels)
    )
    n_components, component_labels = scipy.sparse.csgraph.connected_components(
        graph, 
        directed=False, 
        return_labels=True
    )
    new_mapping = np.zeros(n_labels + 1, dtype=labels.dtype)
    new_mapping[1:] = component_labels + 1
    labels = new_mapping[labels]
    n_labels = n_components
    
    return labels, n_labels
 
def find_rcf_peak_angles(
    rcf: torch.Tensor, 
    threshold_rel: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find the optimal alignment angles in a Rotational Correlation Function (RCF)
    
    The results are returned in euler ZYZ extrinsic convention.

    Parameters
    -----------
    rcf: torch.Tensor
        The rotational correlation function presumably computed by a 
        `SHVolumeDecomposer`. Must have shape 
        (2*bandwidth, 2*bandwidth, 2*bandwidth).
    
    Returns
    -------
    alpha: torch.Tensor
        First rotation around Z axis in radians.
    beta: torch.Tensor
        Second rotation around Y axis in radians.
    gamma: torch.Tensor
        Third rotation around Z axis in radians.
    """
    
    N = rcf.shape[0]
    if rcf.shape != (N, N, N):
        raise ValueError(
            f"Expected rcf to have shape {(N, N, N)}, but got {rcf.shape}."
        )
        
    # Remove redundant part for computation
    rcf = rcf[:, :(N//2+1), :] 
    
    # Find peaks in the RCF
    threshold = threshold_rel*rcf.max()
    mask = rcf > threshold
    labels, n_labels = _label_rcf_peaks(mask, torch.ones(3, 3, 3))
    indices = torch.arange(1, n_labels+1)
    peaks = scipy.ndimage.maximum_position(rcf, labels=labels, index=indices)
    angles = (2*math.pi / N) * torch.tensor(peaks)
    xi = angles[:, 0]
    nu = angles[:, 1]
    omega = angles[:, 2]
    
    # Convert to ZYZ extrinsic convention
    alpha = omega - math.pi/2
    beta = math.pi - nu
    gamma = xi - math.pi/2
    
    return alpha, beta, gamma
