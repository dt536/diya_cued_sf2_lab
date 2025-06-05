from skimage.metrics import structural_similarity

# Structural Similarity Index (SSIM)
def ssim(A, B):
    data_range = max(A.max(), B.max()) - min(A.min(), B.min())
    struct_sim, _ = structural_similarity(A, B, data_range=data_range, full=True)
    return struct_sim
