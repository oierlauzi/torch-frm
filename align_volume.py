import numpy as np
import scipy.ndimage
import torch
import napari

from torch_frm import SHVolumeDecomposer, SHRotationalCorrelator, find_rcf_peak_angles, euler_zyz_to_matrix

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

if __name__ == "__main__":
    R = np.array([
        [-0.4906, -0.1693, 0.8548],
        [0.5976, -0.7410, 0.3032],
        [0.6339,  0.6500, 0.4199]
    ])

    volume = _make_test_volume()
    volume_exp = torch.tensor(_rotate_volume_around_center(volume, R.T))
    volume_ref = torch.tensor(volume)
    
    B = 64
    decomposer = SHVolumeDecomposer(
        bandwidth=B, 
        n_radii=len(volume)//2,
    )
    correlation_function = SHRotationalCorrelator(
        bandwidth=B
    )
    
    sh_x = decomposer.transform(volume_exp)
    sh_r = decomposer.transform(volume_ref)
    rcf = correlation_function.rcf(sh_x, sh_r)

    alpha, beta, gamma = find_rcf_peak_angles(rcf)
    matrix = euler_zyz_to_matrix(alpha, beta, gamma)
    print(matrix)

    
    delta = matrix.T @ torch.tensor(R, dtype=matrix.dtype)
    print("Deviation from true rotation:")
    print(torch.rad2deg(torch.acos((torch.trace(delta) - 1) / 2)))
    
    viewer = napari.view_image(rcf.numpy(), name='RCF')
    napari.run()
