import torch
import math

_real_to_complex = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128
}

def _spherical_harmonic_azimuth(
    phi: torch.Tensor,
    degrees: int
):
    """
    Computes the azimuthal part of the spherical harmonic function.

    Parameters
    -----------
    phi: torch.Tensor
        The azimuth angle in radians.
    degrees: int
        Parameter to define the Maximum order and degree of the Legendre
        polynomials, i.e. the polynomials are generated for all l-s in
        0, 1, ... degrees-1 and m-s in [0, l]

    Returns
    -------
    out: torch.Tensor
        Tensor of Legendre polynomial values. The shape is (N, x.shape), where
        N is (degrees*(degrees+1))//2
    """
    
    N = (degrees*(degrees+1))//2
    out = torch.empty(
        (N, ) + phi.shape,
        dtype=_real_to_complex[phi.dtype],
        device=phi.device
    )

    for m in range(degrees):
        value = torch.exp(1j*m*phi)
        for l in range(m, degrees):
            index = (l*(l+1))//2 + m
            out[index] = value

    return out

def _associated_legendre_ortho(
    theta: torch.Tensor,
    degrees: int
) -> torch.Tensor:
    """
    Computes the orthogonalized values P^l_m(cos(theta)) with Condon-Shortley
    phase for all valid l-m
    combinations upto a level.

    Parameters
    -----------
    theta: torch.Tensor
        Tensor of positions at which to evaluate the Legendre polynomials (after
        cosine)
    degrees: int
        Parameter to define the Maximum order and degree of the Legendre
        polynomials, i.e. the polynomials are generated for all l-s in
        0, 1, ... degrees-1 and m-s in 0 to l

    Returns
    -------
    out: torch.Tensor
        Tensor of Legendre polynomial values. The shape is (N, x.shape), where
        N is (degrees*(degrees+1))//2
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
    phase = -1
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
        out[base+l] = phase*math.sqrt((2*l + 1) / (2*l)) * sin * upper

        # Prepare next iteration
        base_upper2 = base_upper1
        base_upper1 = base
        phase = -phase

    assert (base_upper1+degrees) == N
    return out

def ravel_spherical_harmonic_index(l: int, m: int) -> int:
    """
    Get the linear index where a particular spherical harmonic degree and order
    is stored.
    
    Parameters
    -----------
    l: int
        The degree.
    m: int
        The order. Must be in [-l, +l]

    Returns
    -------
    out: int
        The linear index where the particular coefficient is stored.
    """
    return l*(l+1) + m

def spherical_harmonics(
    theta: torch.Tensor,
    phi: torch.Tensor,
    degrees: int
) -> torch.Tensor:
    """
    Evaluate all spherical harmonic coefficients upto a degree for the given
    theta and phi angles.

    Parameters
    -----------
    theta: torch.Tensor
        Elevations at which the spherical harmonic coefficients are evaluated.
        In radians.
    phi: torch.Tensor
        Azimuths at which the spherical harmonic coefficients are evaluated.
        In radians.
    degrees: int
        Parameter to define the Maximum order and degree of the Legendre
        polynomials, i.e. the polynomials are generated for all l-s in
        0, 1, ... degrees-1 and m-s in -l to l (both included)

    Returns
    -------
    out: torch.Tensor
        Tensor of spherical harmonics. The shape of the resulting tensor is 
        (N, broadcast(theta, phi)), where N is degrees*(degrees+1). The a 
        particular degree-order in N may be addressed using 
        `ravel_spherical_harmonic_index`
    """
    
    p = _associated_legendre_ortho(theta, degrees)
    q = _spherical_harmonic_azimuth(phi, degrees)
    pq = p*q

    N = degrees*degrees
    out = torch.empty(
        (N, ) + pq.shape[1:],
        dtype=pq.dtype,
        device=pq.device
    )

    for l in range(degrees):
        base_out = l*(l+1)
        base_in = base_out // 2

        out[base_out] = pq[base_in]
        phase = -1

        for m in range(1, l+1):
            value = pq[base_in+m]
            out[base_out+m] = value
            out[base_out-m] = phase*value.conj()
            phase = -phase

    return out
