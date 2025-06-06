from cued_sf2_lab.dwt import idwt
from cued_sf2_lab.dwt import dwt
from cued_sf2_lab.laplacian_pyramid import quantise, quant1, quant2, bpp
import numpy as np

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


def quantdwt(Y: np.ndarray, dwtstep: np.ndarray, rise_ratio=0.5):
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
        rise_ratio: the scaling factor for rise1
    Returns:
        Yq: the quantized version of `Y`
        dwtent: an array of shape `(3, n+1)` containing the entropies
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
        VU = quantise(VU, dwtstep[0, col], rise_ratio*dwtstep[0, col])
        Yq[:m, m:2*m] = VU
        dwtent[0, col] = bpp(VU)

        #bottom left
        UV = Yq[m:2*m, :m]
        UV = quantise(UV, dwtstep[1, col], rise_ratio*dwtstep[1, col])
        Yq[m:2*m, :m] = UV
        dwtent[1, col] = bpp(UV)

        #bottom right
        VV = Yq[m:2*m, m:2*m]
        VV = quantise(VV, dwtstep[2, col], rise_ratio*dwtstep[2, col])
        Yq[m:2*m, m:2*m] = VV
        dwtent[2, col] = bpp(VV)
    #top left
    UU = Yq[:m, :m]
    UU = quantise(UU, dwtstep[0, n], rise_ratio*dwtstep[0, n])
    Yq[:m, :m] = UU
    dwtent[0, -1] = bpp(UU)
    return Yq, dwtent

def optimisation_for_DWT(X, Y, n, target_rms, rise_ratio=0.5, max_iter: int = 100):
    """
    This function computes the equal rms optimum step size for quantisation
    Input: X, Y, n
    Output: opt step_size, opt Yq, opt dwent, opt Z
    """
    ls, hs = 1.0, 50.0           # lower / upper bounds
    step_size   = hs
    tol         = 0.01
    dwtstep = np.full((3, n+1), hs)
    Yq, _ = quantdwt(Y, dwtstep, rise_ratio)
    Z = nlevidwt(Yq, n)
    rms_error = np.std(X-Z)
    iter_count = 0
    while np.abs(rms_error - target_rms) > tol and iter_count < max_iter:
        Yq, _ = quantdwt(Y, dwtstep, rise_ratio)
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
    Yq, dwtent = quantdwt(Y, dwtstep, rise_ratio)
    Z = nlevidwt(Yq, n)
    return step_size, Yq, dwtent, Z

def compression_ratio_for_DWT(X, Yq, dwtent):
    """
    This function computes the compression ratio for a given dwt scheme
    Input: X, Yq, dwtent
    Output: compression_ratio, num_bits
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
    return compression_ratio, num_bits


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


def optimisation_for_DWT_MSE(X, Y, n, step_ratios, target_rms, rise_ratio=0.5, max_iter: int = 1000):
    """
    This function computes the equal mse optimum step size for quantisation
    Input: X, Y, n
    Output: opt step_size, opt Yq, opt dwent, opt Z
    """
    ls, hs = 1.0, 50.0           # lower / upper bounds
    step_size   = 0.5 * (ls + hs)
    tol         = 0.001
    dwtstep = step_ratios*hs
    Yq, _ = quantdwt(Y, dwtstep, rise_ratio)
    Z = nlevidwt(Yq, n)
    rms_error = np.std(X-Z)
    iter_count = 0
    while np.abs(rms_error - target_rms) > tol and iter_count < max_iter:
        Yq, _ = quantdwt(Y, dwtstep, rise_ratio)
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
    Yq, dwtent = quantdwt(Y, dwtstep, rise_ratio)
    Z = nlevidwt(Yq, n)
    return step_size, dwtstep, Yq, dwtent, Z

def diff_step_sizes(X, m, n, target_rms, rise_ratio=0.5):
    """
    This function scales dwtstep appropriately for an equal mse and equal rms scheme
    Input: X, m (eg 256), n
    Output: optimal starting step, scaled steps, Yq, dwtent, Z
    """
    Y = nlevdwt(X, n)
    energies = energy_from_impulse(m, n)
    step_ratios = 1 / np.sqrt(energies)
    step_ratios /= step_ratios[0][0]
    opt_step, scaled, Yq, dwtent, Z = optimisation_for_DWT_MSE(X, Y, n, step_ratios, target_rms, rise_ratio)
    return(opt_step, scaled, Yq, dwtent, Z)
    
def quant1dwt(Y: np.ndarray, dwtstep: np.ndarray, rise_ratio=0.5):
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
        rise_ratio: the scaling factor for rise1
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
        VU = quant1(VU, dwtstep[0, col], rise_ratio*dwtstep[0, col])
        Yq[:m, m:2*m] = VU
        dwtent[0, col] = bpp(VU)

        #bottom left
        UV = Yq[m:2*m, :m]
        UV = quant1(UV, dwtstep[1, col], rise_ratio*dwtstep[1, col])
        Yq[m:2*m, :m] = UV
        dwtent[1, col] = bpp(UV)

        #bottom right
        VV = Yq[m:2*m, m:2*m]
        VV = quant1(VV, dwtstep[2, col], rise_ratio*dwtstep[2, col])
        Yq[m:2*m, m:2*m] = VV
        dwtent[2, col] = bpp(VV)
    #top left
    UU = Yq[:m, :m]
    UU = quant1(UU, dwtstep[0, n], rise_ratio*dwtstep[0, n])
    Yq[:m, :m] = UU
    dwtent[0, -1] = bpp(UU)
    return Yq

def quant2dwt(Yq: np.ndarray, dwtstep: np.ndarray, rise_ratio=0.5) -> np.ndarray:
    """
    Inverse quantisation of DWT coefficients using quant2.

    Parameters:
        Yq: the quantised DWT coefficients
        dwtstep: step sizes of shape (3, n+1)
        rise_ratio: the same scaling factor used during quantisation

    Returns:
        Y: the reconstructed (dequantised) DWT coefficients
    """
    Y = Yq.copy()
    m = Y.shape[0]
    n = dwtstep.shape[1] - 1

    for col in range(n):
        m = m // 2

        # top right (VU)
        VU = Y[:m, m:2*m]
        VU = quant2(VU, dwtstep[0, col], rise_ratio*dwtstep[0, col])
        Y[:m, m:2*m] = VU

        # bottom left (UV)
        UV = Y[m:2*m, :m]
        UV = quant2(UV, dwtstep[1, col], rise_ratio*dwtstep[1, col])
        Y[m:2*m, :m] = UV

        # bottom right (VV)
        VV = Y[m:2*m, m:2*m]
        VV = quant2(VV, dwtstep[2, col], rise_ratio*dwtstep[2, col])
        Y[m:2*m, m:2*m] = VV

    # top left (UU, lowest resolution)
    UU = Y[:m, :m]
    UU = quant2(UU, dwtstep[0, n], rise_ratio*dwtstep[0, n])
    Y[:m, :m] = UU

    return Y


vlc, dhufftab, totalbits = jpegenc_dwt(X, 5, dcbits=10, steps=scaled, rise_ratio=1.0, log=False, opthuff=True)



# ---------------------------------------------------------------------
#  NEW  ✦  Step size that meets a target bit-budget
# ---------------------------------------------------------------------
def step_from_target_bits_DWT(X: np.ndarray,
                              s: float,
                              N: int,
                              k: float,
                              target_bits: float,
                              lo: float = 1.0,
                              hi: float = 50.0,
                              tol_bits: float = 500.0,
                              max_iter: int = 5000):
    """
    Find the step size Δ such that the DWT bit-count ≈ target_bits.

    Parameters
    ----------
    X           : 2-D ndarray (zero-mean image)
    s           : float   – overlap parameter of the POT
    N           : int     – block size (8 by default)
    k           : float   – rise1/Δ factor
    target_bits : float   – desired total bit budget
    lo, hi      : float   – initial search bounds for Δ
    tol_bits    : float   – acceptable absolute error in bits
    max_iter    : int     – failsafe iteration cap

    Returns
    -------
    Δ_opt  : float  – step size meeting the target bit-rate
    bits   : float  – bit-count at Δ_opt
    Yq_opt : ndarray – quantised coefficient array at Δ_opt
    Zp_opt : ndarray – reconstructed spatial image at Δ_opt
    """
    # helper: forward LBT analysis, quantise, count bits
    def _quantise_and_bits(step):
        Yq = find_Yq(X, N, s, step, k)         # quantise only
        bits = lbt_bits(Yq, N)
        return Yq, bits

    vlc, dhufftab, totalbits = jpegenc_dwt(X, 5, dcbits=10, steps=scaled, rise_ratio=1.0, log=False, opthuff=True)
    # binary search for Δ so that bits ≈ target_bits
    for _ in range(max_iter):
        step = 0.5 * (lo + hi)
        vlc, dhufftab, totalbits = jpegenc_dwt(X, 5, dcbits=10, steps=scaled, rise_ratio=1.0, log=False, opthuff=True)

        if abs(totalbits - target_bits) <= tol_bits:
            break

        if bits > target_bits:     # too many bits ⇒ need larger Δ
            lo = step
        else:                       # too few bits ⇒ need smaller Δ
            hi = step
    else:
        print("Warning: max_iter reached without hitting target_bits")

    # inverse-transform to give reconstructed image
    Zp = lbt_reconstruct(X, N, s, step, k)

    return step, bits, Yq, Zp
