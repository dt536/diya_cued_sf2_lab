#this file contains the dct functions
from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.dct import regroup
from cued_sf2_lab.laplacian_pyramid import quantise, bpp
import numpy as np

def perform_dct(N, X):
    """
    Calculates an n point DCT matrix and applies it to X
    input: N, X
    output: Y, Yr
    """
    C = dct_ii(N)
    Y = colxfm(colxfm(X, C).T, C).T
    Yr = regroup(Y, N)/N
    return Y, Yr

def recontruct_dct(N, Y):
    """
    Reconstructs the image Z from Y
    input: N, Y
    output: Z
    """
    C = dct_ii(N)
    Z = colxfm(colxfm(Y.T, C.T).T, C.T)
    return Z

def dctbpp(Yr, N): 
    """
    calculate the total number of bits from a re-grouped image Yr, 
    by using `bpp(Ys)` on each sub-image Ys of Yr, 
    then multiplying each result by the number of pixels in the sub-image, 
    and summing to give the total number of bits.

    input: Yr, N
    output: no_bits
    """
    no_pix = int(256/N)
    no_bits = 0.0
    for row in range(N):
        y_start = no_pix * row
        y_end = y_start + no_pix
        for col in range(N):
            x_start = no_pix * col
            x_end = x_start + no_pix
            Ys = Yr[y_start:y_end, x_start:x_end]
            entropy = bpp(Ys)
            no_bits += Ys.size * entropy
    return(no_bits)


def optimisation_for_DCT(X, Y, C, k, max_iter: int = 100):
    """
    Returns the equal rms optimal step.
    Input: X, Y, C
    Output: optimal step, Yq, Z
    """
    ls, hs = 1e-2, 50.0           # lower / upper bounds
    step_size   = hs
    target_rms  = np.std(X-quantise(X, 17))
    tol         = 0.001
    Yq = quantise(Y, step_size, rise1=k*step_size)
    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)
    rms_error   = np.std(X-Z)

    iter_count = 0
    while np.abs(rms_error - target_rms) > tol and iter_count < max_iter:
        Yq = quantise(Y, step_size, rise1=k*step_size)
        Z = colxfm(colxfm(Yq.T, C.T).T, C.T)
        rms_error = np.std(X-Z)

        if rms_error > target_rms:
            hs = step_size
            step_size = 0.5 * (ls + step_size)
        else:                       # rms_error < target_rms
            ls = step_size
            step_size = 0.5 * (hs + step_size)

        iter_count += 1

    if iter_count == max_iter:
        print("Warning: max iterations reached without meeting tolerance")
    Yq = quantise(Y, step_size, rise1=k*step_size)
    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)

    return step_size, Yq, Z




def compression_ratio_for_DCT(N, X, Yq):
    Xq = quantise(X, 17)
    Yr = regroup(Yq, N)/N
    no_bits_sub_img = dctbpp(Yr, N)
    no_bits_ref = bpp(Xq)*Xq.size
    compression_ratio = no_bits_ref/no_bits_sub_img
    return compression_ratio

print("Investigating suppression of high-frequency DCT coefficients")

# Define suppression masks: keep low frequencies, zero out highs
def generate_suppress_mask(N, keep_fraction):
    mask = np.zeros((N, N))
    limit = int(N * keep_fraction)
    for i in range(limit):
        for j in range(limit - i):
            mask[i, j] = 1
    return mask



def suppress_dct_coefficients(Y, suppress_mask):
    """
    Suppress selected DCT coefficients based on a mask.
    Y: DCT-transformed image
    suppress_mask: binary mask of shape (8, 8) where 1 = keep, 0 = suppress
    Returns: suppressed DCT image
    """
    N = 8
    Y_suppressed = np.zeros_like(Y)
    for i in range(0, Y.shape[0], N):
        for j in range(0, Y.shape[1], N):
            block = Y[i:i+N, j:j+N]
            block_suppressed = block * suppress_mask
            Y_suppressed[i:i+N, j:j+N] = block_suppressed
    return Y_suppressed
