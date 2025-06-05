from skimage.metrics import structural_similarity

# Structural Similarity Index (SSIM)
def ssim(A, B):
    data_range = A.max() - A.min()  # or use 1.0 if images are normalized
    struct_sim, _ = structural_similarity(A, B, data_range=data_range, full=True)
    return struct_sim
