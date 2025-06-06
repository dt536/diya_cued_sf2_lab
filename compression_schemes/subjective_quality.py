from skimage.metrics import structural_similarity
import sewar
# Structural Similarity Index (SSIM)
def ssim(A, B):
    """Compare A and B for visual similarity
    
    higher value of structural similarity gives better visual result"""
    data_range = max(A.max(), B.max()) - min(A.min(), B.min())
    struct_sim, _ = structural_similarity(A, B, data_range=data_range, full=True)
    return struct_sim

#Visual Information Fidelity
def vif_index(imageA, imageB):
    return sewar.full_ref.vifp(imageA, imageB)