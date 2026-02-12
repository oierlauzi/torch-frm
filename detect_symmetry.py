from typing import Optional
import torch
import argparse
import mrcfile
import napari
from torch_frm import (
    SHVolumeDecomposer, SHRotationalCorrelator, 
    find_rcf_peak_angles, euler_zyz_to_matrix
)

def compute_rotational_self_correlation(
    volume: torch.Tensor,
    bandwidth: int = 64,
    n_radii: Optional[int] = None
) -> torch.Tensor:
    if n_radii is None:
        n_radii = len(volume) // 2
    
    decomposer = SHVolumeDecomposer(bandwidth=bandwidth, n_radii=n_radii)
    correlator = SHRotationalCorrelator(bandwidth=bandwidth)

    volume = volume - volume.mean()
    sh_decomposition = decomposer.transform(volume)
    return correlator.rcf(sh_decomposition, sh_decomposition)

def main():
    parser = argparse.ArgumentParser(description="Detect symmetry of a map.")
    parser.add_argument("volume_path", type=str, help="Path to the input volume.")
    args = parser.parse_args()

    with mrcfile.open(args.volume_path) as mrc:
        volume = torch.tensor(mrc.data)
        
    rcf = compute_rotational_self_correlation(volume)
    
    # Uncomment to visualize the Rotational Cross-Correlation Function (RCF)
    napari.view_image(rcf.numpy(), name='RCF')
    napari.run()
    
    alpha, beta, gamma = find_rcf_peak_angles(rcf, 0.5)
    matrices = euler_zyz_to_matrix(alpha, beta, gamma)
    n_peaks = len(matrices)
    print(n_peaks)
    
    # TODO Given a list of matrices (array with shape (N, 3, 3)),
    # search for the symmetry group to which it belongs and the canonical 
    # orientation.

if __name__ == "__main__":
    main()