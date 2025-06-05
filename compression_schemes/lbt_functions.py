import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise,bpp
from cued_sf2_lab.dct import colxfm, regroup
from cued_sf2_lab.lbt import pot_ii, dct_ii
from dct_funcs import dctbpp



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