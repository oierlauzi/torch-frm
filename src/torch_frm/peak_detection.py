from typing import Optional, Iterable, Tuple, Set, Union
import math
import numpy as np
import scipy.ndimage
import scipy.sparse
import torch

def _find_common_labels(
    first_face: np.ndarray, 
    last_face: np.ndarray
) -> Set[Tuple[int, int]]:
    """
    Given two opposing 'faces' of a label volume, find label pairs in
    opposing positions that produce periodic edges.
    
    Parameters
    -----------
    first_face: torch.Tensor
        One of the faces.
    last_face: torch.Tensor
        The opposite face.
        
    Returns
    -------
    out: Set[Tuple[int, int]]
        Pairs of indices that are shared across boundaries. Each pair is
        sorted, such that the first index is strictly less than the second.
    """
    merges = set()
    
    both_faces = (first_face > 0) & (last_face > 0)
    for i, j in zip(first_face[both_faces], last_face[both_faces]):
        if i < j:
            merges.add((i-1, j-1))
        elif j < i:
            merges.add((j-1, i-1))
            
    return merges

def _label_mask(
    mask: np.ndarray,
    periodic_axis: Optional[Iterable[int]] = None,
    structure: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, int]:
    """
    Given a binary segmentation label disjoint bodies with unique id-s 
    optionally considering periodic edges.
    
    Parameters
    -----------
    mask: np.ndarray
        Binary segmentation.
    periodic_axis: Optional[Iterable[int]]
        Axis indices that are periodic.
    structure:
        Structure used to determine if two bodies are disjoint. If not provided
        `scipy.ndimage.label`'s default structure is used.
        
    Returns
    -------
    labels: np.ndarray
        Labels assigned to each body of the mask. Labels are in [1, n_labels]
        and 0 label is used to identify the background.
    n_labels: int
        Number of different distinct bodies found in the mask.
    """
    labels, n_labels = scipy.ndimage.label(mask, structure=structure)

    # Trivial case
    if n_labels <= 1 or not periodic_axis:
        return labels, n_labels

    merges = set()
    for periodic_axis_index in periodic_axis:
        first_face = np.take(labels, indices=0, axis=periodic_axis_index)
        last_face = np.take(labels, indices=-1, axis=periodic_axis_index)
        merges.update(_find_common_labels(first_face, last_face))

    if not merges:
        return labels, n_labels

    graph_edges = np.array(list(merges), dtype=int)
    graph = scipy.sparse.csr_array(
        (np.ones(len(graph_edges)), (graph_edges[:, 0], graph_edges[:, 1])),
        shape=(n_labels, n_labels)
    )
    n_components, component_labels = scipy.sparse.csgraph.connected_components(
        graph, 
        directed=False, 
        return_labels=True
    )
    new_mapping = np.empty(n_labels + 1, dtype=labels.dtype)
    new_mapping[0] = 0
    new_mapping[1:] = component_labels + 1
    labels = new_mapping[labels]
    n_labels = n_components
    
    return labels, n_labels

def _find_correlation_peak_indices(
    correlation_function: np.ndarray,
    threshold_rel: float = 0.5,
    periodic_axis: Optional[Iterable[int]] = None
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Given an arbitrary 3D cross-correlation function find peaks on it.
    
    Parameters
    -----------
    correlation_function: np.ndarray
        A 3D cross-correlation function.
    threshold_rel: float
        Ratio of the threshold and the global maximum in the function
    periodic_axis: Optional[Iterable[int]]
        Axis indices that are periodic.
        
    Returns
    -------
    indices: Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of the indices for the peaks in the function.
    values: np.ndarray
        Peak values for each of the detected peaks. In descending order.
    """
    threshold = threshold_rel*correlation_function.max()
    mask = correlation_function > threshold
    labels, n_labels = _label_mask(
        mask=mask,
        periodic_axis=periodic_axis,
        structure=np.ones((3, 3, 3))
    )
    indices = np.arange(1, n_labels+1)
    peak_positions = scipy.ndimage.maximum_position(
        correlation_function, 
        labels=labels, 
        index=indices
    )
    peak_indices = tuple(np.array(col) for col in zip(*peak_positions))
    
    peaks = correlation_function[peak_indices]
    order = np.argsort(peaks)
    order = np.flip(order) # Descending
    return tuple(peak_index[order] for peak_index in peak_indices), peaks[order]

def _rcf_peak_indices_to_euler_zyz(
    indices: Union[np.ndarray, torch.Tensor], 
    n: int
) -> np.ndarray:
    angles = (2*math.pi / n) * indices
    xi = angles[:, 0]
    nu = angles[:, 1]
    omega = angles[:, 2]
    
    # Convert to ZYZ extrinsic convention
    alpha = omega - math.pi/2
    beta = math.pi - nu
    gamma = xi - math.pi/2
    
    return alpha, beta, gamma

def _require_cube(x: np.ndarray) -> int:
    N = x.shape[0]
    if x.shape != (N, N, N):
        raise ValueError(
            f"Expected rcf to have shape {(N, N, N)}, but got {x.shape}."
        )
        
    return N

def find_rcf_peak_angles(
    rcf: np.ndarray, 
    threshold_rel: float = 0.5
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Find the optimal alignment angles in a Rotational Correlation Function (RCF)
    
    The results are returned in euler ZYZ extrinsic convention.

    Parameters
    -----------
    rcf: np.ndarray
        The rotational correlation function presumably computed by a 
        `SHVolumeDecomposer`. Must have shape 
        (2*bandwidth, 2*bandwidth, 2*bandwidth).
    
    Returns
    -------
    angles: Tuple[np.ndarray, np.ndarray, np.ndarray]
        Euler angles in ZYZ convention for which correlation peaks exist.
    values: np.ndarray
        Correlation values for the peaks. Sorted in descending order.
    """
    N = _require_cube(rcf)
        
    # Remove redundant part for computation
    rcf = rcf[:, :(N//2+1), :] 
    
    indices, values = _find_correlation_peak_indices(
        correlation_function=rcf, 
        threshold_rel=threshold_rel, 
        periodic_axis=(0, 2)
    )
    
    angles = _rcf_peak_indices_to_euler_zyz(np.array(indices).T, N)
    return angles, values

def find_cross_correlation_peak_shifts(
    cross_correlation: np.ndarray, 
    threshold_rel: float = 0.5,
    max_shift: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the optimal alignment shifts for a cross-correlation function.
    
    Parameters
    -----------
    cross_correlation: np.ndarray
        Cross correlation function. Must be (N, N, N) shape.
    
    Returns
    -------
    shifts: Tuple[np.ndarray, np.ndarray, np.ndarray]
        Shifts for Z, Y and X in pixels.
    values: np.ndarray
        Correlation values for the peaks. Sorted in descending order.
    """
    N = _require_cube(cross_correlation)
        
    index_to_shift = np.fft.fftfreq(N, d=1/N)
    if max_shift is not None:
        max_shift2 = max_shift*max_shift
        index_to_shift2 = np.square(index_to_shift)
        total_shift2_grid = \
            index_to_shift2[:,None,None] + \
            index_to_shift2[None,:,None] + \
            index_to_shift2[None,None,:]
        
        # Mask out out of bounds peaks
        cross_correlation = cross_correlation.copy()
        cross_correlation[total_shift2_grid > max_shift2] = 0

    indices, values = _find_correlation_peak_indices(
        correlation_function=cross_correlation,
        threshold_rel=threshold_rel, 
        periodic_axis=(0, 1, 2)
    )
    
    shifts = index_to_shift[indices]
    return shifts, values
