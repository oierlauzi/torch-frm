import torch

def euler_zyz_to_matrix(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor
) -> torch.Tensor:
    """
    Convert euler angles in zyz convention to matrix form.
    
    Parameters
    -----------
    alpha: torch.Tensor
        First rotation around Z axis in radians.
    beta: torch.Tensor
        Second rotation around Y axis in radians.
    gamma: torch.Tensor
        Third rotation around Z axis in radians.

    Returns
    -------
    out: torch.Tensor
        The matrices associated to the euler angles. The shape is derived
        from the broadcast of input operand shapes suffixed by (3, 3)
    """
    
    batch_shape = torch.broadcast_shapes(alpha.shape, beta.shape, gamma.shape)
    dtype = torch.promote_types(alpha.dtype, torch.promote_types(beta.dtype, gamma.dtype))
    out = torch.empty(batch_shape + (3, 3), dtype=dtype, device=alpha.device)

    cos_alpha = torch.cos(alpha)
    sin_alpha = torch.sin(alpha)
    cos_beta = torch.cos(beta)
    sin_beta = torch.sin(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma)

    out[...,0,0] = cos_alpha*cos_beta*cos_gamma - sin_alpha*sin_gamma
    out[...,0,1] = -sin_alpha*cos_beta*cos_gamma - cos_alpha*sin_gamma
    out[...,0,2] = sin_beta*sin_gamma

    out[...,1,0] = cos_alpha*cos_beta*sin_gamma + sin_alpha*cos_gamma
    out[...,1,1] = -sin_alpha*cos_beta*sin_gamma + cos_alpha*cos_gamma
    out[...,1,2] = sin_beta*sin_gamma

    out[...,2,0] = -cos_alpha*sin_beta
    out[...,2,1] = sin_alpha*sin_beta
    out[...,2,2] = cos_beta

    return out
