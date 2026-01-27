import torch

def _sample_3d(
    volume: torch.Tensor,
    positions: torch.Tensor,
    mode: str,
    padding: str
) -> torch.Tensor:
    return torch.nn.functional.grid_sample(
        volume, 
        positions, 
        align_corners=False,
        mode=mode,
        padding_mode=padding
    )
    
def sample_3d(
    volume: torch.Tensor,
    positions: torch.Tensor,
    mode: str = 'bilinear',
    padding: str = 'zeros'
) -> torch.Tensor:
    """
    Sample a 3D volume at the specified fractional coordinates.
    
    Thin wrapper around `torch.nn.functional.grid_sample`.
    
    Parameters
    -----------
    volume: torch.Tensor
        Volume to be sampled.
    positions: torch.Tensor
        Fractional 3D coordinates at which the volume is sampled. Must have
        (...,3) shape.
    mode: str
        Interpolation used for sampling. `'bilinear'` or `'nearest'`.
    padding: str
        Padding used to fill out of range values.

    Returns
    -------
    out: torch.Tensor
        Values extracted from the volume. Shape (...).
    """
    
    batch_shape = positions.shape[:-1]
    
    volume = volume.view(1, 1, *volume.shape)
    positions = positions.view(1, -1, 1, 1, 3)
    
    if volume.is_complex():
        # torch.grid_sample does not support complex types. This is a
        # workaround and should be replaced if torch.grid_sample starts
        # supporting complex types.
        samples = torch.complex(
            _sample_3d(volume.real, positions, mode=mode, padding=padding),
            _sample_3d(volume.imag, positions, mode=mode, padding=padding)
        )
    else:
        samples = _sample_3d(volume, positions, mode=mode, padding=padding)
        
    return samples.view(batch_shape)
