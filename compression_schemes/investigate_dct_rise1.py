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




# -------------------------------------------------------------
# Suppression of some DCT coefficients
# -------------------------------------------------------------

print("Investigating suppression of high-frequency DCT coefficients")

# Define suppression masks: keep low frequencies, zero out highs
def generate_suppress_mask(N, keep_fraction):
    mask = np.zeros((N, N))
    limit = int(N * keep_fraction)
    for i in range(limit):
        for j in range(limit - i):
            mask[i, j] = 1
    return mask

fractions = [1.0, 0.75, 0.5, 0.25]  # keep 100%, 75%, 50%, 25% of low frequencies
results_suppression = []

for f in fractions:
    mask = generate_suppress_mask(N, f)
    Y_suppressed = suppress_dct_coefficients(Y, mask)
    Z_suppressed = recontruct_dct(N, Y_suppressed)
    Yr_suppressed = regroup(Y_suppressed, N)/N
    bits = dctbpp(Yr_suppressed, N)
    rms = np.std(X - Z_suppressed)
    results_suppression.append((f, rms, bits))
    print(f"Keep {f*100:.0f}% low-freq → RMS={rms:.3f}, bits={bits:.0f}")

# Plot suppression results
fig, ax = plt.subplots()
for f, rms, bits in results_suppression:
    ax.plot(bits, rms, 'o', label=f"{int(f*100)}% LF kept")
ax.set_xlabel("Total bits per image")
ax.set_ylabel("RMS error")
ax.set_title("Suppression of DCT coefficients")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()




print("Combining DCT coefficient suppression and rise1 quantisation, followed by matched-rms")

fractions = [1.0, 0.75, 0.5, 0.25]  # percentage of low-freq coefficients to keep
combined_results = []

for f in fractions:
    mask = generate_suppress_mask(N, f)
    for k in rise1_factors:
        Y_suppressed = suppress_dct_coefficients(Y, mask)
        


        # Quantise suppressed DCT with rise1 and optimal step size
        opt_step, Yq, Z = optimisation_for_DCT(X, Y_suppressed, C, k=k, max_iter= 100)

        '''# Reconstruct and measure
        Z = recontruct_dct(N, Yq)
        rms = np.std(X - Z)
        Yr = regroup(Yq, N) / N
        bits = dctbpp(Yr, N)
        cpr = bits_ref / bits
        combined_results.append((f, k, rms, bits, cpr))
        print(f"Keep {int(f*100)}% LF, rise1={k:.1f} → RMS={rms:.3f}, Bits={bits:.0f}, CPR={cpr:.2f}")'''

        rms = np.std(X - Z)

        Yr = regroup(Yq, N)/N              # regroup for entropy calc

        bits   = dctbpp(Yr, N)                       # always 16×16 regroup
        cpr    = bits_ref / bits
        #CPR = compression_ratio_for_DCT(N, X, Yq)  or use this function to find CPR
        
        results.append((k, rms, bits, cpr))
        print(f"rise1/step = {k:3.1f} → opt_step = {opt_step}, RMS={rms:6.3f}, bits={bits:,.0f}, CPR={cpr:4.2f}")




#print("Optimising step size for CPR with suppressed DCT and rise1 quantisation")




"Find the CPR when rms matched to reference for different k values"
results  = []                              # (rise1, rms, bits, cpr)

for k in rise1_factors:
    opt_step, Yq, Z = optimisation_for_DCT(X, Y_suppressed, C, k=k, max_iter= 1000)
    #Yq = quantise(Y, step_ref, rise1=k*step_ref)    # centre-clipped
    #Z  = recontruct_dct(N, Yq)                      # inverse transform
    Yr = regroup(Yq, N)/N              # regroup for entropy calc

    bits   = dctbpp(Yr, N)                       # always 16×16 regroup
    cpr    = bits_ref / bits
    #CPR = compression_ratio_for_DCT(N, X, Yq)  or use this function to find CPR
    
    results.append((k, Yq, bits, cpr))
    print(f"rise1/step = {k:3.1f} → opt_step = {opt_step}, RMS={e:6.3f}, bits={bits:,.0f}, CPR={cpr:4.2f}")

