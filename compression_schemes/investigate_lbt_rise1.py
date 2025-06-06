"""
Centre-clipped quantiser study for an LBT front-end.
"""
import numpy as np
import matplotlib.pyplot as plt
from cued_sf2_lab.familiarisation import load_mat_img
from cued_sf2_lab.laplacian_pyramid import quantise, bpp
from dct_funcs import *    
from lbt_functions import *
from subjective_quality import *
from dwt_funcs import zero_mean
# -------------------------------------------------------------
#  Set-up
# -------------------------------------------------------------
X, _     = load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={})
X        = zero_mean(X)                       # zero-mean as usual
step_ref = 17.0
target_rms = np.std(X - quantise(X, step_ref))     # reference distortion
rise1_factors = [0.5, 1.0, 1.5, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]                     # multiples of step for the wide zero

# -------------------------------------------------------------
#  LBT analysis (s = np.sqrt(2), 8×8)
# -------------------------------------------------------------
N = 8
s = np.sqrt(2)
"Find the quantising error for different k"
bits_list, rms_list = [], []             # containers for the plot
print("Find the quantising error for different k")
for k in rise1_factors:
    # ----- 1.  matrices -------------------------------------------------
    Pf, Pr = pot_ii(N=N, s=s)         # forward & reverse POT filters
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
    
    Yq = quantise(Y, step_ref ,k*step_ref)
    Yr   = regroup(Yq, N) / N
    bits = dctbpp(Yr, 16)

    Zp = lbt_reconstruct(X, N, s=s, step=step_ref, k=k)
    rms = rms_LBT(X, step=step_ref, s=s, N=N, k=k)

    print(f"k={k}, rms={rms}, total bits={bits}")
    bits_list.append(bits)
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
    opt_step, rms_opt, bits, cpr = CPR_LBT(
        X=X, N=N, s=s,
        rms_ref=target_rms, step_ref=step_ref,
        k=k
    )


    results.append((k, rms_opt, bits, cpr))
    print(f"rise1/step = {k:3.1f} → opt_step = {opt_step}, RMS={rms_opt:6.3f}, bits={bits:,.0f}, CPR={cpr:4.2f}")



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




keep_fracs = [1.0, 0.75, 0.5, 0.25]
results = []
for f in keep_fracs:
    for k in rise1_factors:
        Δ, rms, bits, cpr = CPR_LBT_suppressed(X, N=8, s=1.4, rms_ref=target_rms, step_ref=17, k=k, keep_fraction=f)
        Yq = lbt_reconstruct(X, N, np.sqrt(2), Δ, k)
        similarity = ssim(X, Yq) 
        print(f"keep {int(f*100)}%, rise1={k:.1f} → Δ={Δ:.2f}, RMS={rms:.3f}, bits={bits:.0f}, CPR={cpr:.2f}, similarity = {similarity}")
        results.append((f, k, Δ, rms, bits, cpr, similarity))

       
