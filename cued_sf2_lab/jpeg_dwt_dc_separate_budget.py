
#not using this file right now
#jpeg_dwt_dc_separate_budget.py

import numpy as np
from compression_schemes.dwt_funcs import nlevdwt,get_step_ratios, quant1dwt, quant2dwt
from .laplacian_pyramid import bpp
from .jpeg_dwt_dc import runampl, diagscan, huffdflt, huffgen, dwtgroup
 from .jpeg_dwt_dc import decode_ac_stream  # user-provided

def encode_dc_pass(Yg: np.ndarray,
                   n: int,
                   dcbits: int) -> (list, dict, int, float, np.ndarray):
    """
    Extract, DPCM-quantise and Huffman-encode the DC stream.
    Returns:
      vlc_dc     : list of (code, length) pairs
      hufftab_dc : DC Huffman table
      dc_bits    : total bits used for DC
      delta_dc   : quant step for DC
      dc_q       : quantised DPCM indices
    """
    M = 2**n
    dc_vals = Yg[0::M, 0::M].flatten().astype(float)
    prev = 0.0
    diffs = []
    for v in dc_vals:
        diffs.append(v - prev)
        prev = v
    diffs = np.array(diffs)

    delta_dc = np.max(np.abs(diffs)) / (2**dcbits - 1)
    dc_q = np.round(diffs / delta_dc).astype(int)

    hufftab_dc = huffdflt('dc')
    _, ehuf = huffgen(hufftab_dc)
    vlc_dc = []
    for idx in dc_q:
        size = int(np.floor(np.log2(abs(idx))) + 1) if idx != 0 else 0
        code = ehuf[size, 0]
        vlc_dc.append((code, ehuf[size,1] + size))
    dc_bits = sum(length for (_, length) in vlc_dc)

    return vlc_dc, hufftab_dc, dc_bits, delta_dc, dc_q


def jpegenc_dwt_ac_only(Yg: np.ndarray,
                         n: int,
                         steps: np.ndarray,
                         rise_ratio: float,
                         opthuff: bool) -> (list, dict, int):
    """
    Huffman-encode only the AC coefficients with given per-subband steps.
    Returns (vlc_ac, hufftab_ac, ac_bits).
    """
    M = 2**n
    hufftab_ac = huffdflt('ac')
    _, ehuf = huffgen(hufftab_ac)

    total_bits = 0
    vlc_ac = []
    scan = diagscan(M)
    H, W = Yg.shape
    for r in range(0, H, M):
        for c in range(0, W, M):
            block = Yg[r:r+M, c:c+M].flatten()[scan].astype(int)
            ra = runampl(block)
            for run, size, rem in ra:
                idx = run*16 + size
                entry = ehuf[idx] if idx < ehuf.shape[0] else ehuf[0xF0]
                code, length = entry[0], entry[1]
                vlc_ac.append((code, length + size))
                total_bits += length + size
    return vlc_ac, hufftab_ac, total_bits


def step_from_target_bits_DWT_separate(X: np.ndarray,
                                      n: int,
                                      target_bits: float,
                                      dcbits: int = 9,
                                      rise_ratio: float = 1.0,
                                      lo: float = 1.0,
                                      hi: float = 50.0,
                                      tol_bits: float = 500.0,
                                      max_iter: int = 100):
    """
    Two-pass bit-budget: DC gets exactly dcbits per block, AC uses the rest.
    Returns Δ_opt, vlc_dc, hufftab_dc, vlc_ac, hufftab_ac, dc_q, delta_dc.
    """
    Y = nlevdwt(X, n)
    Yg = dwtgroup(Y, n)

    vlc_dc, hufftab_dc, dc_bits, delta_dc, dc_q = encode_dc_pass(Yg, n, dcbits)
    ac_target = target_bits - dc_bits

    ratios = get_step_ratios(X.shape[0], n)
    ratios[0,0] = 0.0
    for _ in range(max_iter):
        Δ = 0.5*(lo + hi)
        steps = Δ * ratios
        Yq = quant1dwt(Y, steps, rise_ratio)
        Yqg = dwtgroup(Yq, n)
        vlc_ac, hufftab_ac, ac_bits = jpegenc_dwt_ac_only(Yqg, n, steps, rise_ratio, opthuff=True)
        if abs(ac_bits - ac_target) <= tol_bits:
            break
        if ac_bits > ac_target:
            lo = Δ
        else:
            hi = Δ
    return Δ, vlc_dc, hufftab_dc, vlc_ac, hufftab_ac, dc_q, delta_dc


def jpegdec_dwt_separate(vlc_dc, hufftab_dc, dc_q, delta_dc,
                         vlc_ac, hufftab_ac,
                         X_shape: tuple, n: int,
                         Δ_opt: float) -> np.ndarray:
    """
    Decode the two-pass DWT bitstream and reconstruct the image.
    """
    H, W = X_shape
    M = 2**n
    # DC decode
    Yg_dc = np.zeros((H, W), dtype=float)
    prev = 0.0
    idx = 0
    for r in range(0, H, M):
        for c in range(0, W, M):
            diff = dc_q[idx] * delta_dc
            val = prev + diff
            Yg_dc[r, c] = val
            prev = val
            idx += 1

    # AC decode placeholder: user must implement parsing
    # Here, one would convert vlc_ac + hufftab_ac into grouped AC coeffs
    # For now: assume decode_ac_stream is available
    from .jpeg_dwt_dc import decode_ac_stream  # user-provided
    Yg_ac = decode_ac_stream(vlc_ac, hufftab_ac, X_shape, n)

    # Combine and invert quantisation
    ratios = get_step_ratios(H, n)
    steps = Δ_opt * ratios
    steps[0,0] = delta_dc
    Yqg = Yg_dc + Yg_ac

    # Inverse grouping and DWT
    Yq = dwtungroup(Yqg, n)
    Z = inv_nlevdwt(Yq, n)
    return Z
