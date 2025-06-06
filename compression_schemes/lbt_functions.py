import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise,bpp
from cued_sf2_lab.dct import colxfm, regroup
from cued_sf2_lab.lbt import pot_ii, dct_ii
from compression_schemes.dct_funcs import dctbpp, generate_suppress_mask, suppress_dct_coefficients




#LBT method
def lbt_reconstruct(X, N, s, step, k):
    """
    Apply a POT+block-DCT analysis/synthesis to image X and return Zp.

    Parameters
    ----------
    X    : 2-D ndarray
           Spatial image (zero-mean if you previously subtracted 128).
    N    : int
           Block size of the DCT (default 8).
    s    : float
           Overlap (scale) parameter for the POT (default 1.4).
    step : float or None
           If given, quantise the DCT coefficients with this step
           before the inverse transform.  If None, no quantisation.

    Returns
    -------
    Zp : 2-D ndarray
         Reconstructed image after inverse DCT and post-filter.

    """
    # ----- 1.  matrices -------------------------------------------------
    Pf, Pr = pot_ii(N, s)         # forward & reverse POT filters
    C      = dct_ii(N)            # orthonormal block DCT

    # slice that selects interior rows/cols (overlapped region)
    t = np.s_[N//2 : -N//2]

    # ----- 2.  forward POT ---------------------------------------------
    Xp = X.copy()
    Xp[t, :] = colxfm(Xp[t, :],  Pf)       # columns
    Xp[:, t] = colxfm(Xp[:, t].T, Pf).T    # rows

    # ----- 3.  block DCT ------------------------------------------------
    Y = colxfm(colxfm(Xp.T, C).T, C)

    # optional quantisation
    
    Y = quantise(Y, step,k*step)

    # ----- 4.  inverse DCT ---------------------------------------------
    Z = colxfm(colxfm(Y.T, C.T).T, C.T)

    # ----- 5.  post-filter ---------------------------------------------
    Zp = Z.copy()
    Zp[:, t] = colxfm(Zp[:, t].T, Pr.T).T  # rows
    Zp[t, :] = colxfm(Zp[t, :],  Pr.T)     # columns

    return Zp

def find_Yq(X, N, s, step, k):
    """
    Apply a POT+block-DCT analysis/synthesis to image X and return Yq.

    Parameters
    ----------
    X    : 2-D ndarray
           Spatial image (zero-mean if you previously subtracted 128).
    N    : int
           Block size of the DCT (default 8).
    s    : float
           Overlap (scale) parameter for the POT (default 1.4).
    step : float or None
           If given, quantise the DCT coefficients with this step
           before the inverse transform.  If None, no quantisation.

    Returns
    -------
    Yq : 2-D ndarray
         Quantised Y after LBT transformation.

    """
    # ----- 1.  matrices -------------------------------------------------
    Pf, Pr = pot_ii(N, s)         # forward & reverse POT filters
    C      = dct_ii(N)            # orthonormal block DCT

    # slice that selects interior rows/cols (overlapped region)
    t = np.s_[N//2 : -N//2]

    # ----- 2.  forward POT ---------------------------------------------
    Xp = X.copy()
    Xp[t, :] = colxfm(Xp[t, :],  Pf)       # columns
    Xp[:, t] = colxfm(Xp[:, t].T, Pf).T    # rows

    # ----- 3.  block DCT ------------------------------------------------
    Y = colxfm(colxfm(Xp.T, C).T, C)
    # optional quantisation
    
    Y = quantise(Y, step,k*step)

    return Y


def rms_LBT(X, step:float, s:float, N: int, k:float) -> float:
    """Return RMS error for an N×N LBT with overlap-parameter *s*
       when the DCT coefficients are quantised with step Δ=*step*."""
    
    Zp = lbt_reconstruct(X, N, s, step, k)
    rms_err_Zp = np.std(X- Zp)
    return rms_err_Zp


def find_step_LBT(X,
                  target_err: float, 
                  s: float,
                  N: int,
                  k: float,
                  lo: float = 1.0,
                  hi: float = 50.0,
                  tol: float = 1e-3,
                  max_iter: int = 5000):
    "Returns the optimal step size and the rms error at this step size"

    """Binary-search Δ so that rms_LBt(step) ≈ target_err."""    
    for _ in range(max_iter):          # failsafe upper bound
        mid = 0.5 * (lo + hi)
        e   = rms_LBT(X, mid, s, N, k)

        # stop if we're close enough
        if abs(e - target_err) <= tol:
            return mid, e

        # otherwise shrink the half-interval that gives too much error
        if e > target_err:
            hi = mid          # error too big, we need a smaller step
        if e < target_err:
            lo = mid          # error too small ⇒ we need a larger step

    # If we drop out because of max_iter, return the best we have
    return mid, e   #mid is the matched step size, e is the error



def CPR_LBT(X, N, s, rms_ref, step_ref, k):
    "Returns the compression ratio at optimal step size "

    '''Parameters
    ----------
    X    : 2-D ndarray
           Spatial image (zero-mean if you previously subtracted 128).
    N    : int
           Block size of the DCT (default 8).
    s    : float
           Overlap (scale) parameter for the POT (default 1.4).


    rms_ref: rms value between X and Xq quantised at step_ref
    step_ref :quantise X at this step size
          '''

    C  = dct_ii(N)
    # 1. find Δ* that gives rms_ref for this s
    Δ_star, rms_opt = find_step_LBT(X, rms_ref, s=s, N=N, k=k)

    # 2. forward LBT ----------------------------------------------------
    t_slice  = lambda N: np.s_[N//2:-N//2]
    t         = t_slice(N)
    Pf, Pr    = pot_ii(N, s)
    Xp        = X.copy()
    Xp[t, :]  = colxfm(Xp[t, :],  Pf)
    Xp[:, t]  = colxfm(Xp[:, t].T, Pf).T
    Y         = colxfm(colxfm(Xp.T, C).T, C)

    # 3. quantise and regroup ------------------------------------------
    Yq   = quantise(Y, Δ_star, k*Δ_star)
    Yr   = regroup(Yq, N) / N
    bits = dctbpp(Yr, 16)
    

    Xq       = quantise(X, step_ref)
    bits_ref = bpp(Xq) * Xq.size
    rms_ref  = np.std(X - Xq)

    # 4. compression ratio ---------------------------------------------
    CPR   = bits_ref / bits

    return Δ_star, rms_opt, bits, CPR





def suppress_lbt_coefficients(Y, suppress_mask, N):
    """
    Suppress selected LBT DCT coefficients based on a binary mask.

    Parameters
    ----------
    Y : 2D ndarray
        LBT-transformed image (output of colxfm).
    suppress_mask : 2D ndarray (N x N)
        1 = keep, 0 = suppress.
    N : int
        Block size.

    Returns
    -------
    Y_suppressed : 2D ndarray
        LBT coefficients after suppression.
    """
    Y_suppressed = np.zeros_like(Y)
    for i in range(0, Y.shape[0], N):
        for j in range(0, Y.shape[1], N):
            block = Y[i:i+N, j:j+N]
            block_suppressed = block * suppress_mask
            Y_suppressed[i:i+N, j:j+N] = block_suppressed
    return Y_suppressed




def CPR_LBT_suppressed(X, N, s, rms_ref, step_ref, k, keep_fraction):
    "Like CPR_LBT but applies coefficient suppression"
    C = dct_ii(N)
    Δ_star, rms_opt = find_step_LBT(X, rms_ref, s=s, N=N, k=k)

    # LBT forward
    t = np.s_[N//2:-N//2]
    Pf, Pr = pot_ii(N, s)
    Xp = X.copy()
    Xp[t, :] = colxfm(Xp[t, :], Pf)
    Xp[:, t] = colxfm(Xp[:, t].T, Pf).T
    Y = colxfm(colxfm(Xp.T, C).T, C)

    # Suppress high-freq coefficients
    mask = generate_suppress_mask(N, keep_fraction)
    Y_suppressed = suppress_lbt_coefficients(Y, mask, N)

    # Quantise and regroup
    Yq = quantise(Y_suppressed, Δ_star, k*Δ_star)
    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)

    # ----- 5.  post-filter ---------------------------------------------
    Zp = Z.copy()
    Zp[:, t] = colxfm(Zp[:, t].T, Pr.T).T  # rows
    Zp[t, :] = colxfm(Zp[t, :],  Pr.T)     # columns
    Yr = regroup(Yq, N)/N
    bits = dctbpp(Yr, 16)

    Xq = quantise(X, step_ref)
    bits_ref = bpp(Xq) * Xq.size
    rms_ref = np.std(X - Xq)
    rms_supress = np.std(X-Zp)
    CPR = bits_ref / bits
    return Δ_star, rms_opt, rms_supress, bits, CPR, Zp


def get_equivalent_lbt_error_quant(X, N, M):
    EPSILON = 8.0
    bits_max = 40960.0 - 1424.0 - 8.0*8.0 - EPSILON*2
    

    lbt_quant = 100
    step = lbt_quant/2
    i = 0 # Stop infinite loops
    size = calc_lbt_bit_size(lbt_quant, X, N, M)
    while abs(size - bits_max) > EPSILON and i < 500:
        if size > bits_max:
            lbt_quant += step
        else:
            lbt_quant -= step
        step /= 2
        size = calc_lbt_bit_size(lbt_quant, X, N, M)
        i += 1

    if i == 500:
        print("STEP OPTIMISATION REACHED ITERATION CAP")

    return lbt_quant



# ---------------------------------------------------------------------
#  NEW  ✦  Bit-count for LBT
# ---------------------------------------------------------------------
from dct_funcs import regroup, dctbpp              # already in your repo
# (regroup(Yq,N)/N produces the 16×16 mosaic of sub-images that dctbpp expects)

def lbt_bits(Yq: np.ndarray, N: int) -> float:
    """
    Return the total estimated number of bits required to code the
    quantised LBT coefficients Yq (using the same entropy model as dctbpp).

    Parameters
    ----------
    Yq : 2-D ndarray
         Quantised LBT coefficients (shape = original image size).
    N  : int
         Block size of the DCT inside the LBT (usually 8).

    Returns
    -------
    bits : float
           Total number of bits for the whole image.
    """
    Yr = regroup(Yq, N) / N           # 16×16 mosaic of sub-images
    bits = dctbpp(Yr, N)              # reuse the DCT helper
    return bits


# ---------------------------------------------------------------------
#  NEW  ✦  Step size that meets a target bit-budget
# ---------------------------------------------------------------------
def step_from_target_bits_LBT(X: np.ndarray,
                              s: float,
                              N: int,
                              k: float,
                              target_bits: float,
                              lo: float = 1.0,
                              hi: float = 50.0,
                              tol_bits: float = 500.0,
                              max_iter: int = 5000):
    """
    Find the step size Δ such that the LBT bit-count ≈ target_bits.

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

    Δ_low, Δ_high = lo, hi
    Yq, bits = _quantise_and_bits(Δ_high)

    # binary search for Δ so that bits ≈ target_bits
    for _ in range(max_iter):
        step = 0.5 * (Δ_low + Δ_high)
        Yq, bits = _quantise_and_bits(step)

        if abs(bits - target_bits) <= tol_bits:
            break

        if bits > target_bits:     # too many bits ⇒ need larger Δ
            Δ_low = step
        else:                       # too few bits ⇒ need smaller Δ
            Δ_high = step
    else:
        print("Warning: max_iter reached without hitting target_bits")

    # inverse-transform to give reconstructed image
    Zp = lbt_reconstruct(X, N, s, step, k)

    return step, bits, Yq, Zp


