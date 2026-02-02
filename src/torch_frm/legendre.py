import torch
import math

def ravel_associated_legendre_index(l: int, m: int) -> int:
    """
    Get the linear index where a particular associated Legendre polynomial degree
    and order is stored.
    
    Parameters
    -----------
    l: int
        The degree.
    m: int
        The order. Must be in [0, l]

    Returns
    -------
    out: int
        The linear index where the particular polynomial is stored.
    """
    return (l*(l+1))//2 + m

def associated_legendre_ortho(
    theta: torch.Tensor,
    degrees: int
) -> torch.Tensor:
    """
    Computes the orthogonalized values P^l_m(cos(theta)) for all l-s upto a 
    level and their associated possitive m-s. 

    Parameters
    -----------
    theta: torch.Tensor
        Tensor of positions at which to evaluate the Legendre polynomials (after
        cosine)
    degrees: int
        Parameter to define the Maximum order and degree of the Legendre
        polynomials, i.e. the polynomials are generated for all l-s [0, degrees)
        and m-s in [0, l]

    Returns
    -------
    out: torch.Tensor
        Tensor of Legendre polynomial values. The shape is (N, x.shape), where
        N is (degrees*(degrees+1))//2. `ravel_associated_legendre_index` may
        be used to retrieve a particular (l, m) polynomial value tensor.
    """

    N = (degrees*(degrees+1))//2
    out = torch.empty(
        (N, ) + theta.shape,
        dtype=theta.dtype,
        device=theta.device
    )
    sin = torch.sin(theta)
    cos = torch.cos(theta)

    # First row
    if degrees > 0:
        out[0] = 1.0 / math.sqrt(4*math.pi)

    # Rest
    base_upper1 = 0
    base_upper2 = None
    for l in range(1, degrees):
        base = base_upper1 + l
        for m in range(0, l-1):
            upper1 = out[base_upper1 + m]
            upper2 = out[base_upper2 + m]
            k1 = math.sqrt((4*l*l - 1) / (l*l - m*m))
            k2 = math.sqrt(((2*l + 1)*(l + m - 1)*(l - m - 1)) / ((2*l - 3)*(l - m)*(l + m)))
            out[base+m] = k1*cos*upper1 - k2*upper2

        # Last two elements of the row
        upper = out[base_upper1 + l-1]
        out[base+l-1] = math.sqrt(2*l + 1) * cos * upper
        out[base+l] = math.sqrt((2*l + 1) / (2*l)) * sin * upper

        # Prepare next iteration
        base_upper2 = base_upper1
        base_upper1 = base

    assert (base_upper1+degrees) == N
    return out
