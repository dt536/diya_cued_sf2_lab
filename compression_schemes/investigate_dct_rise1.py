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
rise1_factors = [0.5, 1.0, 1.5, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]                     # multiples of step for the wide zero

# -------------------------------------------------------------
#  DCT analysis (single level, 8×8)
# -------------------------------------------------------------
N        = 8
C        = dct_ii(N)
Y, Yr    = perform_dct(N, X)               # forward transform (unquantised)

bits_ref = bpp(quantise(X, step_ref)) * X.size


"Find the quantising error for different k"
bits_list, rms_list = [], []             # containers for the plot
print("Find the quantising error for different k")
for k in rise1_factors:
    Yq2 = quantise(Y, step_ref, rise1=k*step_ref)    # centre-clipped
    Yr2 = regroup(Yq2, N)/N
    total_bits   = dctbpp(Yr2, N)
    Z2  = recontruct_dct(N, Yq2)                      # inverse transform
    rms = np.std(X - Z2)
    print(f"k={k}, rms={rms}, total bits={total_bits}")
    bits_list.append(total_bits)
    rms_list.append(rms)

fig, ax = plt.subplots()
ax.plot(bits_list, rms_list, 'o-', linewidth=1.2, markersize=6)

for k, b, r in zip(rise1_factors, bits_list, rms_list):
    ax.annotate(f"k={k}", (b, r), textcoords="offset points",
                xytext=(5, -7), ha='left', fontsize=8)

ax.set_xlabel("Total bits per image")
ax.set_ylabel("RMS error")
ax.set_title("Centre–clipped quantiser: RMS vs. bits")
ax.grid(True)
plt.tight_layout()
plt.show()


"Find the CPR when rms matched to reference for different k values"
results  = []                              # (rise1, rms, bits, cpr)

for k in rise1_factors:
    opt_step, Yq, Z = optimisation_for_DCT(X, Y, C, k=k, max_iter= 100)
    
    #Yq = quantise(Y, step_ref, rise1=k*step_ref)    # centre-clipped
    #Z  = recontruct_dct(N, Yq)                      # inverse transform
    rms = np.std(X - Z)

    Yr = regroup(Yq, N)/N              # regroup for entropy calc

    bits   = dctbpp(Yr, N)                       # always 16×16 regroup
    cpr    = bits_ref / bits
    #CPR = compression_ratio_for_DCT(N, X, Yq)  or use this function to find CPR
    
    results.append((k, rms, bits, cpr))
    print(f"rise1/step = {k:3.1f} → opt_step = {opt_step}, RMS={rms:6.3f}, bits={bits:,.0f}, CPR={cpr:4.2f}")


# -------------------------------------------------------------
#  Plot
# -------------------------------------------------------------
fig, ax = plt.subplots()
ax.set_title("DCT – centre-clipped zero step")
ax.set_xlabel("RMS error")
ax.set_ylabel("Compression ratio")
ax.grid(True)
for k, rms, bits, cpr in results:
    ax.plot(rms, cpr, 'o', label=f"rise1={k:.1f}×Δ")
ax.legend()
plt.show()



