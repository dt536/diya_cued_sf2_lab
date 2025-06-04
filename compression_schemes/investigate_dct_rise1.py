"""
Centre-clipped quantiser study for an 8×8 block DCT front-end.
"""
import numpy as np
import matplotlib.pyplot as plt
from cued_sf2_lab.familiarisation import load_mat_img
from cued_sf2_lab.laplacian_pyramid import quantise, bpp
from dct_funcs import perform_dct, recontruct_dct, dctbpp    
from dwt_funcs import *
from dct_funcs import *
# -------------------------------------------------------------
#  Set-up
# -------------------------------------------------------------
X, _     = load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={})
X        = zero_mean(X)                       # zero-mean as usual
step_ref = 17.0
target_rms = np.std(X - quantise(X, step_ref))     # reference distortion
rise1_factors = [0.5, 1.0, 1.5]                     # multiples of step for the wide zero

# -------------------------------------------------------------
#  DCT analysis (single level, 8×8)
# -------------------------------------------------------------
N        = 8
C        = dct_ii(N)
Y, Yr    = perform_dct(N, X)               # forward transform (unquantised)

bits_ref = bpp(quantise(X, step_ref)) * X.size

results  = []                              # (rise1, rms, bits, cpr)

for k in rise1_factors:
    opt_step, Yq, Z = optimisation_for_DCT(X, Y, C, k=k, max_iter= 100)
    
    #Yq = quantise(Y, step_ref, rise1=k*step_ref)    # centre-clipped
    #Z  = recontruct_dct(N, Yq)                      # inverse transform
    rms = np.std(X - Z)

    Yr = regroup(Yq, N)/N              # regroup for entropy calc

    bits   = dctbpp(Yr, N)                       # always 16×16 regroup
    cpr    = bits_ref / bits
    CPR = compression_ratio_for_DCT(N, X, Yq, k=k)
    
    results.append((k, rms, bits, cpr,CPR))
    print(f"rise1/step = {k:3.1f} → RMS={rms:6.3f}, bits={bits:,.0f}, CPR={cpr:4.2f}, CPR2={CPR}")

# -------------------------------------------------------------
#  Plot
# -------------------------------------------------------------
fig, ax = plt.subplots()
ax.set_title("DCT – centre-clipped zero step")
ax.set_xlabel("RMS error")
ax.set_ylabel("Compression ratio")
ax.grid(True)
for k, rms, bits, cpr, cpr2 in results:
    ax.plot(rms, cpr, 'o', label=f"rise1={k:.1f}×Δ")
ax.legend()
plt.show()
