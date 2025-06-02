import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.lbt import pot_ii, dct_ii

#LBT method
def lbt_optimal_step(X,
                     s: float,
                     N: int,
                     reference_step:float,
                     lo: float = 1.0,
                     hi: float = 50.0,
                     tol: float = 1e-3,
                     max_iter: int = 5000):
    
    "Return the rms_error between Zp and X under refrence_step, reconstructed image Zp, the optimal-step with the given reference step, and the rms error between Zp and X"
    '''Parameters
        ----------
    X    : 2-D ndarray
           Spatial image (zero-mean if you previously subtracted 128).

    N    : int
           Block size of the DCT (default 8).

    s    : float
           Overlap (scale) parameter for the POT (default 1.4).

    reference_step : float
           quantise the DCT coefficients with this step before the inverse transform.'''
    
    Xq = quantise(X, reference_step)
    rms_err_Xq = np.std(X- Xq)

    def lbt_reconstruct(X,
                    N, 
                    s, 
                    reference_step):
        """
        Apply LBT to image X,
        Return the reconstructed image Zp and the rms error.


        

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
        
        Y = quantise(Y, reference_step)

        # ----- 4.  inverse DCT ---------------------------------------------
        Z = colxfm(colxfm(Y.T, C.T).T, C.T)

        # ----- 5.  post-filter ---------------------------------------------
        Zp = Z.copy()
        Zp[:, t] = colxfm(Zp[:, t].T, Pr.T).T  # rows
        Zp[t, :] = colxfm(Zp[t, :],  Pr.T)     # columns

        rms_err_Zp = np.std(X- Zp)

        return Zp, rms_err_Zp
    
    _, rms_err_Zp = lbt_reconstruct(X, N, s, reference_step)

                    
    """Binary-search Δ so that rms_LBt(step) ≈ target_err."""    
    for _ in range(max_iter):          # failsafe upper bound
        mid = 0.5 * (lo + hi)
        Zp, e   = lbt_reconstruct(X, N, s, mid)

        # stop if we're close enough
        if abs(e - rms_err_Xq) <= tol:
            return rms_err_Zp, Zp, mid, e

        # otherwise shrink the half-interval that gives too much error
        if e > rms_err_Xq:
            hi = mid          # error too big, we need a smaller step
        if e < rms_err_Xq:
            lo = mid          # error too small ⇒ we need a larger step


    # If we drop out because of max_iter, return the best we have
    return rms_err_Zp, Zp, mid, e   #mid is the matched step size, e is the error
