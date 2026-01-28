import numpy as np
import scipy.ndimage
import torch

from torch_frm import frm

def _make_test_volume() -> np.ndarray:
    volume = np.zeros((64, 64, 64), dtype=np.float32)
    
    rng = np.random.default_rng(42)
    points = rng.integers(low=16, high=48, size=(64, 3))
    for point in points:
        volume[tuple(point)] = 1.0

    volume_ft = np.fft.rfftn(volume)
    wz = np.fft.fftfreq(volume.shape[0])[:,None, None]
    wy = np.fft.fftfreq(volume.shape[1])[None, :, None]
    wx = np.fft.rfftfreq(volume.shape[2])[None, None, :]
    w = np.sqrt(np.square(wx) + np.square(wy) + np.square(wz))
    sigma = 0.25
    volume_ft *= np.exp(-0.5 * np.square(w / sigma))
    
    volume = np.fft.irfftn(volume_ft)
    
    return volume
def _rotate_volume_around_center(
    volume: np.ndarray, 
    rotation: np.ndarray, 
    order: int = 2
) -> np.ndarray:
    rotation = np.flip(rotation)
    center = (np.array(volume.shape) - 1) / 2.0
    offset = center - np.dot(rotation.T, center)
    
    return scipy.ndimage.affine_transform(
        volume, 
        matrix=rotation.T, 
        offset=offset, 
        order=order,
        mode='constant',
        cval=0.0
    )

def test_frm():
    R = np.array([
        [-0.4906, -0.1693, 0.8548],
        [0.5976, -0.7410, 0.3032],
        [0.6339,  0.6500, 0.4199]
    ])

    volume = _make_test_volume()
    volume_exp = torch.tensor(_rotate_volume_around_center(volume, R.T))
    volume_ref = torch.tensor(volume)
    
    alignment = frm(volume_exp, volume_ref, bandwidth=32)
    assert torch.allclose(alignment, torch.tensor(R, dtype=alignment.dtype), atol=0.1)