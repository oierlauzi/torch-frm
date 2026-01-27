import numpy as np
import scipy.ndimage
import mrcfile
import torch
import pytest

from emdb import fetch_emdb_map
from torch_frm import frm

@pytest.fixture
def emdb_2660() -> np.ndarray:
    volume_path = fetch_emdb_map(2660)
    with mrcfile.open(volume_path) as mrc:
        return mrc.data

def _rotate_volume_around_center(
    volume: np.ndarray, 
    rotation: np.ndarray, 
    order: int = 2
) -> np.ndarray:
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

    volume_path = fetch_emdb_map(2660)
    with mrcfile.open(volume_path) as mrc:
        volume_ref = mrc.data

    volume_exp = torch.tensor(_rotate_volume_around_center(volume_ref, R))
    volume_ref = torch.tensor(volume_ref)
    
    alignment = frm(volume_exp, volume_ref, bandwidth=32)
    print(alignment, flush=True)
    fdsfds
    