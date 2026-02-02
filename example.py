import numpy as np
import scipy.ndimage
import torch
import math
import napari

from torch_frm import SHVolumeDecomposer, SHRotationalCorrelator, find_rcf_peak_angles, euler_zyz_to_matrix

def _make_test_volume() -> np.ndarray:
    volume = np.zeros((64, 64, 64), dtype=np.float32)
    
    rng = np.random.default_rng(42)
    points = rng.integers(low=16, high=48, size=(256, 3))
    for point in points:
        volume[tuple(point)] += 1.0

    volume_ft = np.fft.rfftn(volume)
    wz = np.fft.fftfreq(volume.shape[0])[:,None, None]
    wy = np.fft.fftfreq(volume.shape[1])[None, :, None]
    wx = np.fft.rfftfreq(volume.shape[2])[None, None, :]
    w = np.sqrt(np.square(wx) + np.square(wy) + np.square(wz))
    sigma = 0.1
    volume_ft *= np.exp(-0.5 * np.square(w / sigma))
    
    volume = np.fft.irfftn(volume_ft)
    
    if False:
        volume += np.rot90(volume, k=1, axes=(1, 2))
        volume += np.rot90(volume, k=2, axes=(1, 2))

    if False:
        volume += np.rot90(volume, k=2, axes=(1, 2))
        volume += np.rot90(volume, k=2, axes=(0, 1))
    
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

if __name__ == "__main__":
    ang_z1 = 40.0
    ang_y = 60
    ang_z2 = -25
    #ang_z1 =  0.0
    #ang_y = 0.0
    #ang_z2 = 0.0
    s = math.sin(math.radians(ang_z1))
    c = math.cos(math.radians(ang_z1))
    RZ1 = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])
    s = math.sin(math.radians(ang_y))
    c = math.cos(math.radians(ang_y))
    RY = np.array([
        [c, 0.0, -s],
        [0.0,  1.0, 0.0],
        [s, 0.0, c]
    ])
    s = math.sin(math.radians(ang_z2))
    c = math.cos(math.radians(ang_z2))
    RZ2 = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])
    R = RZ1 @ RY @ RZ2

    volume = _make_test_volume()
    volume_ref = torch.tensor(volume)
    volume_exp = torch.tensor(_rotate_volume_around_center(volume, R))
    
    B = 32
    decomposer = SHVolumeDecomposer(
        bandwidth=B, 
        n_radii=len(volume)//2,
    )
    correlator = SHRotationalCorrelator(
        bandwidth=B
    )
    
    sh_x = decomposer.transform(volume_exp)
    sh_r = decomposer.transform(volume_ref)
    rcf = correlator.rcf(sh_x, sh_r)
    
    alpha, beta, gamma = find_rcf_peak_angles(rcf)
    print(math.degrees(alpha), math.degrees(beta), math.degrees(gamma))
    
    #napari.view_image(volume, name='RCF')
    napari.view_image(rcf.numpy(), name='RCF')
    napari.run()
