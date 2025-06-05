from cued_sf2_lab.dwt import idwt
from cued_sf2_lab.dwt import dwt
from cued_sf2_lab.laplacian_pyramid import quantise, bpp
import numpy as np
import matplotlib.pyplot as plt

def zero_mean(X):
    return (X - 128.0)


def nlevdwt(X, n):
    """
    This function computes the DWT decomposition of an input image
    for a certain number of speciifed layers.

    input: image X, number of layers
    output: image Y after dwt decomposition
    """
    m = X.shape[0]
    Y =dwt(X)
    for i in range(n-1):
        m = m//2
        Y[:m,:m] = dwt(Y[:m, :m])
    return Y

def nlevidwt(Yq, n):
    """
    This function computes the inverse DWT image
    for a certain number of speciifed layers.

    input: quantised image Yq, number of layers
    output: image X after idwt
    """
    m = (Yq.shape[0])//(2**(n-1))
    Xr = Yq.copy()
    Xr[:m,:m] = idwt(Yq[:m, :m])
    for i in range(n-1):
        m = m*2
        Xr[:m,:m] = idwt(Xr[:m, :m])
    return Xr


def quantdwt(Y: np.ndarray, dwtstep: np.ndarray, k=0.5):
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
        k: the scaling factor for rise1
    Returns:
        Yq: the quantized version of `Y`
        dwtenc: an array of shape `(3, n+1)` containing the entropies
    """
    # your code here
    Yq = Y.copy()
    m = Y.shape[0]
    n = dwtstep.shape[1]-1
    dwtent = np.zeros((dwtstep.shape[0], dwtstep.shape[1]))
    for col in range(n):
        m = m//2
        #top right 
        VU = Yq[:m, m:2*m]
        VU = quantise(VU, dwtstep[0, col], k*dwtstep[0, col])
        Yq[:m, m:2*m] = VU
        dwtent[0, col] = bpp(VU)

        #bottom left
        UV = Yq[m:2*m, :m]
        UV = quantise(UV, dwtstep[1, col], k*dwtstep[1, col])
        Yq[m:2*m, :m] = UV
        dwtent[1, col] = bpp(UV)

        #bottom right
        VV = Yq[m:2*m, m:2*m]
        VV = quantise(VV, dwtstep[2, col], k*dwtstep[2, col])
        Yq[m:2*m, m:2*m] = VV
        dwtent[2, col] = bpp(VV)
    #top left
    UU = Yq[:m, :m]
    UU = quantise(UU, dwtstep[0, n], k*dwtstep[0, n])
    Yq[:m, :m] = UU
    dwtent[0, -1] = bpp(UU)
    return Yq, dwtent

def optimisation_for_DWT(X, Y, n, target_rms, k=0.5, max_iter: int = 100):
    """
    This function computes the equal rms optimum step size for quantisation
    Input: X, Y, n
    Output: opt step_size, opt Yq, opt dwent, opt Z
    """
    ls, hs = 1.0, 50.0           # lower / upper bounds
    step_size   = hs
    tol         = 0.001
    dwtstep = np.full((3, n+1), hs)
    Yq, _ = quantdwt(Y, dwtstep, k)
    Z = nlevidwt(Yq, n)
    rms_error = np.std(X-Z)
    iter_count = 0
    while np.abs(rms_error - target_rms) > tol and iter_count < max_iter:
        Yq, _ = quantdwt(Y, dwtstep, k)
        Z = nlevidwt(Yq, n)
        rms_error = np.std(X-Z)

        if rms_error > target_rms:
            hs = step_size
            step_size = 0.5 * (ls + step_size)
            dwtstep = np.full((3, n+1), step_size)
        else:                       # rms_error < target_rms
            ls = step_size
            step_size = 0.5 * (hs + step_size)
            dwtstep = np.full((3, n+1), step_size)
        iter_count += 1

    if iter_count == max_iter:
        print("Warning: max iterations reached without meeting tolerance")
    Yq, dwtent = quantdwt(Y, dwtstep, k)
    Z = nlevidwt(Yq, n)
    return step_size, Yq, dwtent, Z

def compression_ratio_for_DWT(X, Yq, dwtent):
    """
    This function computes the compression ratio for a given dwt scheme
    Input: X, Yq, dwtent
    Output: compression_ratio
    """
    num_bits = 0.0
    m = Yq.shape[0]
    n = dwtent.shape[1]-1
    for col in range(n):
        m = m//2
        subimg_size = m * m
        num_bits += (
            dwtent[0, col] +  # top right
            dwtent[1, col] +  # bottom left
            dwtent[2, col]    # bottom right
        ) * subimg_size
    num_bits += dwtent[0, n] * subimg_size #top left
    Xq = quantise(X, 17)
    no_bits_ref = bpp(Xq)*Xq.size
    compression_ratio = no_bits_ref/num_bits
    return compression_ratio


def get_sub_img_regions(N, n):
    """
    Returns subband coordinates for an N x N image with n DWT levels.
    Each subband is a tuple: (name, ((row_start, row_end), (col_start, col_end)))
    """
    sub_img_regions = []
    m = N
    for i in range(n):
        half = m // 2

        # VU (top-right)
        sub_img_regions.append((f"VU_{i+1}", ((0, half), (half, m))))

        # UV (bottom-left)
        sub_img_regions.append((f"UV_{i+1}", ((half, m), (0, half))))

        # VV (bottom-right)
        sub_img_regions.append((f"VV_{i+1}", ((half, m), (half, m))))

        m = half  # Prepare for next level

    # Final LL subband
    sub_img_regions.append((f"LL_{n}", ((0, m), (0, m))))

    return sub_img_regions


def energy_from_impulse(N, n, amplitude=100.0):
    sub_img_regions = get_sub_img_regions(N, n)
    energies = np.full((3, n+1), np.nan)

    for idx, ((r0, r1), (c0, c1)) in enumerate([reg for _, reg in sub_img_regions]):
        Yimp = np.zeros((N, N))
        center_row = (r0 + r1) // 2
        center_col = (c0 + c1) // 2
        Yimp[center_row, center_col] = amplitude

        Z = nlevidwt(Yimp, n)
        E = np.sum(Z**2)

        if idx < 3 * n:
            row = idx % 3  # 0=VU, 1=UV, 2=VV
            col = idx // 3
        else:
            row = 0        # UU goes in row 0
            col = n

        energies[row, col] = E

    return energies


def optimisation_for_DWT_MSE(X, Y, n, step_ratios, target_rms, k=0.5, max_iter: int = 100):
    """
    This function computes the equal mse optimum step size for quantisation
    Input: X, Y, n
    Output: opt step_size, opt Yq, opt dwent, opt Z
    """
    ls, hs = 1.0, 50.0           # lower / upper bounds
    step_size   = 0.5 * (ls + hs)
    tol         = 0.001
    dwtstep = step_ratios*hs
    Yq, _ = quantdwt(Y, dwtstep, k)
    Z = nlevidwt(Yq, n)
    rms_error = np.std(X-Z)
    iter_count = 0
    while np.abs(rms_error - target_rms) > tol and iter_count < max_iter:
        Yq, _ = quantdwt(Y, dwtstep, k)
        Z = nlevidwt(Yq, n)
        rms_error = np.std(X-Z)

        if rms_error > target_rms:
            hs = step_size
            step_size = 0.5 * (ls + step_size)
            dwtstep = step_ratios*step_size
        else:                       # rms_error < target_rms
            ls = step_size
            step_size = 0.5 * (hs + step_size)
            dwtstep = step_ratios*step_size
        iter_count += 1

    if iter_count == max_iter:
        print("Warning: max iterations reached without meeting tolerance")
    Yq, dwtent = quantdwt(Y, dwtstep, k)
    Z = nlevidwt(Yq, n)
    return step_size, dwtstep, Yq, dwtent, Z

def diff_step_sizes(X, m, n, target_rms, k=0.5):
    """
    This function scales dwtstep appropriately for an equal mse and equal rms scheme
    Input: X, m (eg 256), n
    Output:scaled
    """
    Y = nlevdwt(X, n)
    energies = energy_from_impulse(m, n)
    step_ratios = 1 / np.sqrt(energies)
    step_ratios /= step_ratios[0][0]
    print(step_ratios)
    _, scaled, Yq, dwtent, Z = optimisation_for_DWT_MSE(X, Y, n, step_ratios, target_rms, k)
    return(scaled, Yq, dwtent, Z)
    

def dwtgroup(X: np.ndarray, n: int) -> np.ndarray:
    '''
    Regroups the rows and columns of ``X``, such that an
    n-level DWT image composed of separate subimages is regrouped into 2^n x
    2^n blocks of coefs from the same spatial region (like the DCT).

    If n is negative the process is reversed.
    '''
    Y = X.copy()

    if n == 0:
        return Y
    elif n < 0:
        n = -n
        invert = 1
    else:
        invert = 0

    sx = X.shape
    N = np.round(2**n)

    if sx[0] % N != 0 or sx[1] % N != 0:
        raise ValueError(
            'Error in dwtgroup: X dimensions are not multiples of 2^n')

    if invert == 0:
        # Determine size of smallest sub-image.
        sx = sx // N

        # Regroup the 4 subimages at each level, starting with the smallest
        # subimages in the top left corner of Y.
        k = 1  # Size of blocks of pels at each level.
        # tm = 1:sx[0];
        # tn = 1:sx[1];
        tm = np.arange(sx[0])
        tn = np.arange(sx[1])

        # Loop for each level.
        for _ in range(n):
            tm2 = np.block([
                [np.reshape(tm, (k, sx[0]//k), order='F')],
                [np.reshape(tm+sx[0], (k, sx[0]//k), order='F')]
            ])
            tn2 = np.block([
                [np.reshape(tn, (k, sx[1]//k), order='F')],
                [np.reshape(tn+sx[1], (k, sx[1]//k), order='F')]
            ])

            sx = sx * 2
            k = k * 2
            tm = np.arange(sx[0])
            tn = np.arange(sx[1])
            Y[np.ix_(tm, tn)] = Y[np.ix_(
                tm2.flatten('F'), tn2.flatten('F'))]

    else:
        # Invert the grouping:
        # Determine size of largest sub-image.
        sx = np.array(X.shape) // 2

        # Regroup the 4 subimages at each level, starting with the largest
        # subimages in Y.
        k = N // 2  # Size of blocks of pels at each level.

        # Loop for each level.
        for _ in np.arange(n):
            tm = np.arange(sx[0])
            tn = np.arange(sx[1])
            tm2 = np.block([
                [np.reshape(tm, (k, sx[0]//k), order='F')],
                [np.reshape(tm+sx[0], (k, sx[0]//k), order='F')]
            ])
            tn2 = np.block([
                [np.reshape(tn, (k, sx[1]//k), order='F')],
                [np.reshape(tn+sx[1], (k, sx[1]//k), order='F')]
            ])

            Y[np.ix_(tm2.flatten('F'), tn2.flatten('F'))] = Y[np.ix_(
                np.arange(sx[0]*2), np.arange(sx[1]*2))]

            sx = sx // 2
            k = k // 2

    return Y


