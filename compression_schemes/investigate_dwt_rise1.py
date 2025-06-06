import numpy as np
import matplotlib.pyplot as plt
from cued_sf2_lab.familiarisation import load_mat_img
from cued_sf2_lab.laplacian_pyramid import quantise, bpp
from dct_funcs import *    
from dwt_funcs import *
from lbt_functions import *
from subjective_quality import *


# Load image
X, _ = load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={})
X = zero_mean(X)
target_rms = np.std(X - quantise(X, 17))
n_levels = 4
Y = nlevdwt(X, n=n_levels)

rise1_factors = [0.25, 0.5, 0.75, 1.0, 1.25]
results_uniform = []
results_equal_mse = []

for k in rise1_factors:
    # --- Uniform Step ---
    step, Yq_u, dwtent_u, Z_u = optimisation_for_DWT(X, Y, n=n_levels, target_rms=target_rms, rise_ratio=k)
    rms_u = np.std(X - Z_u)
    bits_u = 0.0
    m = Yq_u.shape[0]
    for col in range(n_levels):
        m = m // 2
        subimg_size = m * m
        bits_u += (dwtent_u[0, col] + dwtent_u[1, col] + dwtent_u[2, col]) * subimg_size
    bits_u += dwtent_u[0, n_levels] * subimg_size
    cpr_u = (quantise(X, 17).size * bpp(quantise(X, 17))) / bits_u
    ssim_u = ssim(X, Z_u)
    results_uniform.append((k, rms_u, bits_u, cpr_u, ssim_u, Z_u))

    # --- Equal-MSE Step ---
    _, Yq_m, dwtent_m, Z_m = diff_step_sizes(X, m=X.shape[0], n=n_levels, target_rms=target_rms, rise_ratio=k)
    rms_m = np.std(X - Z_m)
    bits_m = 0.0
    m = Yq_m.shape[0]
    for col in range(n_levels):
        m = m // 2
        subimg_size = m * m
        bits_m += (dwtent_m[0, col] + dwtent_m[1, col] + dwtent_m[2, col]) * subimg_size
    bits_m += dwtent_m[0, n_levels] * subimg_size
    cpr_m = (quantise(X, 17).size * bpp(quantise(X, 17))) / bits_m
    ssim_m = ssim(X, Z_m)
    results_equal_mse.append((k, rms_m, bits_m, cpr_m, ssim_m, Z_m))

    # Print results for each k
    print(f"k={k:.2f} | "
          f"Uniform:  RMS={rms_u:.3f}, Bits={bits_u:.0f}, CPR={cpr_u:.2f}, SSIM={ssim_u:.4f} | "
          f"Equal-MSE: RMS={rms_m:.3f}, Bits={bits_m:.0f}, CPR={cpr_m:.2f}, SSIM={ssim_m:.4f}")

# Plot: CPR vs RMS
ks, rms_u, bits_u, cpr_u, ssim_u, _ = zip(*results_uniform)
_, rms_m, bits_m, cpr_m, ssim_m, _ = zip(*results_equal_mse)

plt.figure(figsize=(10, 5))
plt.plot(rms_u, cpr_u, 'o-', label='Uniform Step DWT')
plt.plot(rms_m, cpr_m, 's--', label='Equal-MSE DWT')
for i, k in enumerate(rise1_factors):
    plt.annotate(f'k={k}', (rms_u[i], cpr_u[i]), xytext=(5, -5), textcoords="offset points", fontsize=8)
    plt.annotate(f'k={k}', (rms_m[i], cpr_m[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)
plt.xlabel("RMS error")
plt.ylabel("Compression Ratio (CPR)")
plt.title("DWT: CPR vs RMS for Varying rise1/step (k)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Show reconstructed images
for i, k in enumerate(rise1_factors):
    _, _, _, _, _, Z_u = results_uniform[i]
    _, _, _, _, _, Z_m = results_equal_mse[i]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(Z_u, cmap='gray')
    axs[0].set_title(f'Uniform Step\nk={k}')
    axs[1].imshow(Z_m, cmap='gray')
    axs[1].set_title(f'Equal-MSE Step\nk={k}')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()



# Load and preprocess image
X, _ = load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={})
X = zero_mean(X)
target_rms = np.std(X - quantise(X, 17))
n_levels = 4
Y = nlevdwt(X, n=n_levels)

# Define suppression configs: list of (row, col) in dwtstep to suppress
suppression_configs = {
    "none": [],
    "suppress_HH4": [(2, 3)],
    "suppress_HH4_HH3": [(2, 3), (2, 2)],
    "suppress_all_HH": [(2, 1), (2, 2), (2, 3)],
}

k = 0.5  # fixed rise1/step ratio
results_equal_mse_supp = []

# Get baseline equal-MSE step sizes
base_step, _, _, _ = diff_step_sizes(X, m=X.shape[0], n=n_levels, target_rms=target_rms, rise_ratio=k)

for label, suppressed in suppression_configs.items():
    dwtstep = base_step.copy()
    for (row, col) in suppressed:
        dwtstep[row, col] = 1e6  # suppress by massive step size

    Yq, dwtent = quantdwt(Y, dwtstep, rise_ratio=k)
    Z = nlevidwt(Yq, n_levels)
    rms = np.std(X - Z)
    ssim_val = ssim(X, Z)

    # Bit calculation
    bits = 0.0
    m = Yq.shape[0]
    for col in range(n_levels):
        m = m // 2
        subimg_size = m * m
        bits += (dwtent[0, col] + dwtent[1, col] + dwtent[2, col]) * subimg_size
    bits += dwtent[0, n_levels] * subimg_size

    cpr = (bpp(quantise(X, 17)) * X.size) / bits
    results_equal_mse_supp.append((label, rms, bits, cpr, ssim_val))

# Print results
print(f"{'Suppression':<25} {'RMS':>7} {'Bits':>10} {'CPR':>7} {'SSIM':>7}")
for label, rms, bits, cpr, ssim_val in results_equal_mse_supp:
    print(f"{label:<25} {rms:7.3f} {bits:10.0f} {cpr:7.2f} {ssim_val:7.4f}")
