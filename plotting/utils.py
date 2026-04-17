import json
import numpy as np
import matplotlib.pyplot as plt

# Color-Blind Friendly Okabe-Ito Palette
PALETTE = {
    "blue": "#0072B2",  # LAMa
    "orange": "#E69F00",  # Even Less
    "green": "#009E73",  # Remin
    "vermilion": "#D55E00",  # Convex Hull / Error
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "sky_blue": "#56B4E9",  # Ground Truth
    "black": "#000000",
    "gray": "#969696"
}


def set_sleek_style():
    """Configures matplotlib for modern, clean aesthetics."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Linux Biolinum", "DejaVu Sans"],
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 1.5,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "legend.fontsize": 12,
        "legend.frameon": True,
        "legend.edgecolor": "#CCCCCC",
        "figure.titlesize": 22,
        "figure.titleweight": "bold",
    })


def format_metric(val, pos=None):
    """Formats large numbers with 'k', 'M', or 'B' suffixes."""
    if val == 0: return "0"
    abs_val = abs(val)
    if abs_val >= 1e9: return f"{val / 1e9:.1f}B".replace(".0B", "B")
    if abs_val >= 1e6: return f"{val / 1e6:.1f}M".replace(".0M", "M")
    if abs_val >= 1e3: return f"{val / 1e3:.1f}k".replace(".0k", "k")
    return f"{val:.0f}"


def calculate_mse(true_pts, recon_pts):
    return np.mean(np.sum((true_pts - recon_pts) ** 2, axis=1))


def procrustes_align(true_pts, recon_pts, scale=True, rotate=True, shift=True):
    """N-Dimensional Procrustes alignment using SVD."""
    true_pts = np.asarray(true_pts)
    recon_pts = np.asarray(recon_pts)

    mean_true = np.mean(true_pts, axis=0) if shift else np.zeros(true_pts.shape[1])
    mean_recon = np.mean(recon_pts, axis=0) if shift else np.zeros(recon_pts.shape[1])

    centered_true = true_pts - mean_true
    centered_recon = recon_pts - mean_recon

    var_recon = np.sum(centered_recon ** 2)
    R = np.eye(true_pts.shape[1])
    s = 1.0

    if var_recon > 1e-12:
        if rotate:
            H = centered_recon.T @ centered_true
            U, S, Vt = np.linalg.svd(H)
            R_temp = U @ Vt

            # Prevent reflection
            if np.linalg.det(R_temp) < 0:
                U[:, -1] *= -1
                S[-1] *= -1
                R_temp = U @ Vt
            R = R_temp

            if scale:
                s = np.sum(S) / var_recon
        elif scale:
            H = centered_recon.T @ centered_true
            s = np.trace(H) / var_recon

    aligned_recon = (centered_recon @ R * s) + mean_true
    mse = calculate_mse(true_pts, aligned_recon)
    return aligned_recon, mse


def load_data(path, is_limits=False):
    with open(path, 'r') as f:
        data = json.load(f)

    if is_limits:
        return np.array([[pt["true"] for pt in cluster] for cluster in data]), \
            np.array([[pt["reconstructed"] for pt in cluster] for cluster in data])

    true_pts = np.array([pt["true"] for pt in data])
    recon_pts = np.array([pt["reconstructed"] for pt in data])
    return true_pts, recon_pts