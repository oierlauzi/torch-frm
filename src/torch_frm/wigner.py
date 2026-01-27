import torch

def wigner_matrices(
    theta: torch.Tensor,
    degrees: int
) -> torch.Tensor:
    # FIXME: This was translated fron C-code and there is a lot of room to make 
    # it more Pythonic.
    size = 2*degrees
    N = (degrees*(4*degrees*degrees-1)) // 3

    out = torch.empty(
        (N, ) + theta.shape,
        dtype=theta.dtype,
        device=theta.device
    )
    d = torch.empty(
        (size*size, ) + theta.shape,
        dtype=theta.dtype,
        device=theta.device
    )
    dd = torch.empty(
        (size*size, ) + theta.shape,
        dtype=theta.dtype,
        device=theta.device
    )

    sin = torch.sin(theta/2)
    cos = torch.cos(theta/2)

    if degrees > 0:
        out[0] = 1.0

        d[0] = cos
        d[1] = sin
        d[size] = -sin
        d[size+1] = cos

    for l in range(1, degrees):
        for half_degree in range(0, 1 if l == degrees-1 else 2):
            j2 = 2*l + half_degree
            sin2 = sin / j2
            cos2 = cos / j2
            for i in range(0, j2+2):
                start = i*size
                end = start + j2+2
                dd[start:end] = 0.0

            for i in range(0, j2):
                start = i*size
                end = start + j2
                k = torch.arange(0, j2)

                tmp = d[start:end]
                dd[start:end] += torch.sqrt((j2-i)*(j2-k))*tmp*cos2
                dd[start+1:end+1] += torch.sqrt((j2-i)*(k+1))*tmp*sin2
                dd[start+size:end+size] -= torch.sqrt((i+1)*(j2-k))*tmp*sin2
                dd[start+size+1:end+size+1] += torch.sqrt((i+1)*(k+1))*tmp*cos2

            for i in range(0, j2+1):
                start = i*size
                end = start + j2+1
                d[start:end] = dd[start:end]

            if half_degree == 0:
                base = l*(4*l*l-1) // 3
                for i in range(0, j2+1):
                    start_dst = base + i*(j2+1)
                    end_dst = start_dst + j2+1
                    start_src = i*size
                    end_src = start_src + j2+1
                    out[start_dst:end_dst] = d[start_src:end_src]

    return out
