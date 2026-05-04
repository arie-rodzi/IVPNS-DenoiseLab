
# ============================================================
# IVPNS-DenoiseLab: Batch Research System
# Full Streamlit App for IVPNS Image Denoising Experiments
#
# Features:
# - Upload up to 10 images
# - Gaussian, Speckle, Salt & Pepper noise simulation
# - IVPNSWA / IVPNSWG aggregation
# - Normalized IVPNS score function aligned with manuscript:
#       S = [ alpha_bar(1 - eta beta_bar) + kappa(1 - gamma_bar) ] / (1 + kappa)
# - Truth-dominance adjustment
# - Adaptive refinement
# - Baselines: Mean, Median, Bilateral, Non-Local Means, optional BM3D
# - Metrics: MSE, PSNR, SSIM, CQI
# - Batch result table for Section 7
# - Ablation and sensitivity analysis
#
# Run:
#   streamlit run streamlit_app_q1.py
#
# Requirements:
#   pip install streamlit numpy pillow scipy scikit-image pandas matplotlib
#
# Optional:
#   pip install bm3d
# ============================================================

import io
import time
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import uniform_filter, median_filter, convolve
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_bilateral, denoise_nl_means
import matplotlib.pyplot as plt

try:
    from bm3d import bm3d
    BM3D_AVAILABLE = True
except Exception:
    BM3D_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="IVPNS-DenoiseLab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PREMIUM CSS
# ============================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #F8FAFC 0%, #EEF4FF 45%, #F7FBFF 100%);
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 34px 38px;
    border-radius: 28px;
    background: linear-gradient(135deg, #0B2447 0%, #19376D 48%, #2B5DAA 100%);
    box-shadow: 0 22px 50px rgba(11,36,71,0.22);
    color: white;
    margin-bottom: 24px;
}
.hero-title {
    font-size: 42px;
    font-weight: 900;
    letter-spacing: -0.8px;
    margin-bottom: 8px;
}
.hero-subtitle {
    font-size: 17px;
    color: #DDEBFF;
    line-height: 1.55;
    max-width: 1100px;
}
.hero-badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.25);
    color: #FFFFFF;
    font-size: 13px;
    font-weight: 700;
    margin-right: 8px;
    margin-bottom: 10px;
}
.section-title {
    font-size: 25px;
    font-weight: 850;
    color: #102A43;
    margin-top: 15px;
    margin-bottom: 13px;
}
.section-subtitle {
    font-size: 15px;
    color: #52616B;
    margin-top: -5px;
    margin-bottom: 18px;
}
.card {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(210, 221, 235, 0.95);
    padding: 19px 20px;
    border-radius: 22px;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.065);
    margin-bottom: 14px;
}
.card-title {
    font-size: 16px;
    font-weight: 800;
    color: #123C69;
    margin-bottom: 7px;
}
.card-text {
    font-size: 14px;
    color: #4B5563;
    line-height: 1.45;
}
.pipeline {
    padding: 15px 18px;
    border-radius: 18px;
    background: linear-gradient(90deg, #FFFFFF 0%, #EFF6FF 100%);
    border: 1px solid #D9E8FF;
    color: #0B2447;
    font-size: 15px;
    font-weight: 750;
    text-align: center;
    box-shadow: 0 7px 18px rgba(11,36,71,0.06);
}
.formula {
    background: #F1F7FF;
    border-left: 6px solid #2B5DAA;
    padding: 12px 15px;
    border-radius: 13px;
    font-family: "Courier New", monospace;
    color: #0B2447;
    font-size: 14px;
    margin: 8px 0 10px 0;
}
.research-box {
    padding: 18px 20px;
    border-radius: 20px;
    background: linear-gradient(135deg, #ECFDF5 0%, #F0FDFA 100%);
    border: 1px solid #A7F3D0;
    color: #064E3B;
    font-size: 15px;
    line-height: 1.55;
    box-shadow: 0 10px 25px rgba(6,78,59,0.08);
}
.note-box {
    padding: 16px 18px;
    border-radius: 18px;
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    color: #7C2D12;
    font-size: 14px;
    line-height: 1.50;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #EEF4FF 100%);
}
[data-testid="stMetricValue"] {
    font-size: 25px;
    font-weight: 850;
    color: #0B2447;
}
[data-testid="stMetricLabel"] {
    font-size: 13px;
    color: #425466;
}
.stDownloadButton button {
    border-radius: 14px;
    font-weight: 800;
    border: 1px solid #2B5DAA;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# BASIC IMAGE FUNCTIONS
# ============================================================
def load_image_as_gray(uploaded_file):
    image = Image.open(uploaded_file).convert("L")
    arr = np.array(image).astype(np.float64)
    return image, arr


def resize_if_large(arr, max_side=512):
    h, w = arr.shape
    if max(h, w) <= max_side:
        return arr
    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = Image.fromarray(arr.astype(np.uint8))
    img = img.resize((new_w, new_h))
    return np.array(img).astype(np.float64)


def normalize_image(img_255):
    return np.clip(img_255 / 255.0, 0, 1)


def reconstruct_image(score_01):
    return np.clip(255.0 * score_01, 0, 255).astype(np.uint8)


def image_to_download_bytes(arr_uint8):
    img = Image.fromarray(arr_uint8)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# ============================================================
# NOISE FUNCTIONS
# ============================================================
def add_noise(img_255, noise_type="None", gaussian_sigma=15, speckle_sigma=0.12, salt_pepper_prob=0.03, seed=42):
    rng = np.random.default_rng(seed)
    img = img_255.astype(np.float64)

    if noise_type == "None":
        return np.clip(img, 0, 255)

    if noise_type == "Gaussian":
        noisy = img + rng.normal(0, gaussian_sigma, img.shape)

    elif noise_type == "Speckle":
        g = img / 255.0
        noisy = g + g * rng.normal(0, speckle_sigma, img.shape)
        noisy = noisy * 255.0

    elif noise_type == "Salt & Pepper":
        noisy = img.copy()
        rnd = rng.random(img.shape)
        noisy[rnd < salt_pepper_prob / 2] = 0
        noisy[(rnd >= salt_pepper_prob / 2) & (rnd < salt_pepper_prob)] = 255

    else:
        noisy = img.copy()

    return np.clip(noisy, 0, 255)


# ============================================================
# IVPNS FUNCTIONS
# ============================================================
def ivpns_transform(g, lam=0.5, eps=0.05):
    alpha = g
    gamma = 1.0 - g
    beta = lam * (1.0 - np.abs(2.0 * g - 1.0))

    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "alpha_L": np.clip(alpha - eps, 0, 1),
        "alpha_U": np.clip(alpha + eps, 0, 1),
        "beta_L": np.clip(beta - eps, 0, 1),
        "beta_U": np.clip(beta + eps, 0, 1),
        "gamma_L": np.clip(gamma - eps, 0, 1),
        "gamma_U": np.clip(gamma + eps, 0, 1),
    }


def spatial_kernel(window_size=3, sigma=1.0):
    r = window_size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    d2 = x**2 + y**2
    kernel = np.exp(-d2 / (sigma**2 + 1e-12))
    return kernel / np.sum(kernel)


def aggregate_ivpns(components, operator="IVPNSWA", window_size=3, sigma=1.0):
    kernel = spatial_kernel(window_size, sigma)

    def weighted_mean(x):
        return convolve(x, kernel, mode="reflect")

    def weighted_geo(x):
        safe = np.clip(x, 1e-8, 1.0)
        return np.exp(convolve(np.log(safe), kernel, mode="reflect"))

    out = {}

    if operator == "IVPNSWA":
        out["alpha_L"] = weighted_mean(components["alpha_L"])
        out["alpha_U"] = weighted_mean(components["alpha_U"])
        out["beta_L"] = weighted_geo(components["beta_L"])
        out["beta_U"] = weighted_geo(components["beta_U"])
        out["gamma_L"] = weighted_geo(components["gamma_L"])
        out["gamma_U"] = weighted_geo(components["gamma_U"])
    else:
        out["alpha_L"] = weighted_geo(components["alpha_L"])
        out["alpha_U"] = weighted_geo(components["alpha_U"])
        out["beta_L"] = weighted_mean(components["beta_L"])
        out["beta_U"] = weighted_mean(components["beta_U"])
        out["gamma_L"] = weighted_mean(components["gamma_L"])
        out["gamma_U"] = weighted_mean(components["gamma_U"])

    for key in out:
        out[key] = np.clip(out[key], 0, 1)

    return out, kernel


def truth_dominance_adjustment(agg, delta1=0.05, delta2=0.10, delta3=0.10):
    adjusted = {}
    adjusted["alpha_L"] = np.clip(agg["alpha_L"] * (1.0 + delta1), 0, 1)
    adjusted["alpha_U"] = np.clip(agg["alpha_U"] * (1.0 + delta1), 0, 1)
    adjusted["beta_L"] = np.clip(agg["beta_L"] * (1.0 - delta2), 0, 1)
    adjusted["beta_U"] = np.clip(agg["beta_U"] * (1.0 - delta2), 0, 1)
    adjusted["gamma_L"] = np.clip(agg["gamma_L"] * (1.0 - delta3), 0, 1)
    adjusted["gamma_U"] = np.clip(agg["gamma_U"] * (1.0 - delta3), 0, 1)
    return adjusted


def ivpns_score(agg, eta=0.4, kappa=0.2):
    alpha_m = (agg["alpha_L"] + agg["alpha_U"]) / 2.0
    beta_m = (agg["beta_L"] + agg["beta_U"]) / 2.0
    gamma_m = (agg["gamma_L"] + agg["gamma_U"]) / 2.0

    score = (
        alpha_m * (1.0 - eta * beta_m) +
        kappa * (1.0 - gamma_m)
    ) / (1.0 + kappa)

    return np.clip(score, 0, 1), alpha_m, beta_m, gamma_m


def adaptive_refinement(g, score, window_size=3, omega_min=0.15, omega_max=0.85):
    # Local smoothing candidates
    local_mean = uniform_filter(g, size=window_size)
    local_median = median_filter(g, size=window_size)

    # Edge/texture detector
    edge_strength = np.abs(g - local_mean)
    local_var = uniform_filter((g - local_mean) ** 2, size=window_size)

    # Use IVPNS score only as guidance, not direct image replacement
    uncertainty = np.clip(np.abs(score - g), 0, 1)

    # Strong smoothing in flat/noisy areas, weak smoothing at edges
    smooth_weight = omega_max - 2.0 * edge_strength - 2.5 * np.sqrt(local_var)
    smooth_weight = np.clip(smooth_weight, omega_min, omega_max)

    # Hybrid denoising: median preserves edges, mean reduces Gaussian noise
    smooth_base = 0.60 * local_median + 0.40 * local_mean

    # IVPNS-guided correction
    refined = smooth_weight * smooth_base + (1.0 - smooth_weight) * g

    # Small score-guided stabilization only
    refined = 0.75 * refined + 0.25 * score

    return np.clip(refined, 0, 1), smooth_weight


def process_ivpns(
    input_255,
    lam=0.50,
    eps=0.05,
    operator="IVPNSWA",
    window_size=3,
    sigma=1.0,
    eta=0.4,
    kappa=0.2,
    delta1=0.05,
    delta2=0.10,
    delta3=0.10,
    use_truth=True,
    use_refinement=True,
    omega_min=0.25,
    omega_max=0.70
):
    g = normalize_image(input_255)
    components = ivpns_transform(g, lam=lam, eps=eps)
    aggregated, kernel = aggregate_ivpns(components, operator=operator, window_size=window_size, sigma=sigma)

    if use_truth:
        adjusted = truth_dominance_adjustment(aggregated, delta1=delta1, delta2=delta2, delta3=delta3)
    else:
        adjusted = aggregated

    score, alpha_m, beta_m, gamma_m = ivpns_score(adjusted, eta=eta, kappa=kappa)

    if use_refinement:
        refined_score, omega_map = adaptive_refinement(
            g, score, window_size=window_size, omega_min=omega_min, omega_max=omega_max
        )
    else:
        refined_score = score
        omega_map = np.ones_like(score)

    output_255 = reconstruct_image(refined_score)

    details = {
        "g": g,
        "components": components,
        "aggregated": aggregated,
        "adjusted": adjusted,
        "kernel": kernel,
        "score": score,
        "refined_score": refined_score,
        "omega_map": omega_map,
        "alpha_m": alpha_m,
        "beta_m": beta_m,
        "gamma_m": gamma_m
    }

    return output_255, details


# ============================================================
# BASELINE FILTERS
# ============================================================
def mean_filter_img(img_255, window_size=3):
    return np.clip(uniform_filter(img_255, size=window_size), 0, 255).astype(np.uint8)


def median_filter_img(img_255, window_size=3):
    return np.clip(median_filter(img_255, size=window_size), 0, 255).astype(np.uint8)


def bilateral_filter_img(img_255, sigma_color=0.06, sigma_spatial=5):
    out = denoise_bilateral(
        img_255 / 255.0,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        channel_axis=None
    )
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def robust_noise_sigma(img01):
    """
    Manual noise estimator to avoid estimate_sigma ImportError on Streamlit Cloud.
    """
    img01 = img01.astype(np.float64)
    local_mean = uniform_filter(img01, size=3)
    residual = img01 - local_mean
    mad = np.median(np.abs(residual - np.median(residual)))
    sigma_est = 1.4826 * mad
    return float(np.clip(sigma_est, 0.01, 0.20))


def nlm_filter_img(img_255, patch_size=5, patch_distance=6):
    img01 = np.clip(img_255 / 255.0, 0, 1)
    sigma_est = robust_noise_sigma(img01)
    h = max(0.03, 1.15 * sigma_est)

    out = denoise_nl_means(
        img01,
        h=h,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=True,
        channel_axis=None
    )

    return np.clip(out * 255, 0, 255).astype(np.uint8)
    img01 = img_255 / 255.0
    try:
        sigma_est = float(np.mean(estimate_sigma(img01, channel_axis=None)))
    except TypeError:
        sigma_est = float(np.mean(estimate_sigma(img01)))
    h = max(0.03, 1.15 * sigma_est)
    out = denoise_nl_means(
        img01,
        h=h,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=True,
        channel_axis=None
    )
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def bm3d_filter_img(img_255, sigma_psd=0.08):
    if not BM3D_AVAILABLE:
        return None
    out = bm3d(img_255 / 255.0, sigma_psd=sigma_psd)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


# ============================================================
# METRICS AND CQI
# ============================================================
def compute_metrics(reference, processed):
    ref = reference.astype(np.float64)
    pro = processed.astype(np.float64)

    mse = mean_squared_error(ref, pro)
    psnr = float("inf") if mse == 0 else peak_signal_noise_ratio(ref, pro, data_range=255)
    ssim = structural_similarity(ref, pro, data_range=255)

    return float(mse), float(psnr), float(ssim)


def normalize_metric(series, benefit=True):
    arr = np.array(series, dtype=np.float64)
    min_v, max_v = np.nanmin(arr), np.nanmax(arr)
    if abs(max_v - min_v) < 1e-12:
        return np.ones_like(arr)
    if benefit:
        return (arr - min_v) / (max_v - min_v)
    return (max_v - arr) / (max_v - min_v)


def add_cqi(metrics_df, w_mse=0.30, w_psnr=0.30, w_ssim=0.40):
    df = metrics_df.copy()

    df["MSE Index"] = normalize_metric(df["MSE ↓"], benefit=False)
    df["PSNR Index"] = normalize_metric(df["PSNR (dB) ↑"], benefit=True)
    df["SSIM Index"] = normalize_metric(df["SSIM ↑"], benefit=True)

    df["CQI Score"] = (
        w_mse * df["MSE Index"] +
        w_psnr * df["PSNR Index"] +
        w_ssim * df["SSIM Index"]
    )

    df["CQI Rank"] = df["CQI Score"].rank(ascending=False, method="min").astype(int)
    return df


def evaluate_methods(reference, noisy, params, include_bm3d=True):
    methods = {}

    methods["Noisy/Input"] = noisy.astype(np.uint8)
    methods[f"Mean Filter ({params['window_size']}x{params['window_size']})"] = mean_filter_img(
        noisy, window_size=params["window_size"]
    )
    methods[f"Median Filter ({params['window_size']}x{params['window_size']})"] = median_filter_img(
        noisy, window_size=params["window_size"]
    )
    methods["Bilateral Filter"] = bilateral_filter_img(
        noisy, sigma_color=params["bilateral_sigma_color"], sigma_spatial=params["bilateral_sigma_spatial"]
    )
    methods["Non-Local Means"] = nlm_filter_img(noisy)

    if include_bm3d and BM3D_AVAILABLE:
        methods["BM3D"] = bm3d_filter_img(noisy, sigma_psd=params["bm3d_sigma"])

    ivpns_wa, details_wa = process_ivpns(noisy, operator="IVPNSWA", **params["ivpns"])
    ivpns_wg, details_wg = process_ivpns(noisy, operator="IVPNSWG", **params["ivpns"])

    methods["Proposed IVPNSWA"] = ivpns_wa
    methods["Proposed IVPNSWG"] = ivpns_wg

    rows = []
    for name, output in methods.items():
        mse, psnr, ssim = compute_metrics(reference, output)
        rows.append([name, mse, psnr, ssim])

    metrics_df = pd.DataFrame(rows, columns=["Method", "MSE ↓", "PSNR (dB) ↑", "SSIM ↑"])
    metrics_df = add_cqi(metrics_df)

    return methods, metrics_df, details_wa, details_wg


# ============================================================
# PLOTS AND DOWNLOAD HELPERS
# ============================================================
def plot_histogram(image_255, title):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(image_255.flatten(), bins=60)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25)
    return fig


def plot_bar_metric(df, metric_col, title):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(df["Method"], df[metric_col])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_col)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_sensitivity(df, x_col, y_col, title):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(df[x_col], df[y_col], marker="o")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def small_matrix_preview(arr, size=8):
    return pd.DataFrame(np.round(arr[:size, :size], 4))


def create_comparison_canvas(images, labels, tile_height=230):
    pil_images = [Image.fromarray(img.astype(np.uint8)).convert("L") for img in images]
    resized = []
    for im in pil_images:
        w, h = im.size
        new_w = int(w * tile_height / h)
        resized.append(im.resize((new_w, tile_height)))

    label_h = 42
    total_w = sum(im.size[0] for im in resized)
    canvas = Image.new("RGB", (total_w, tile_height + label_h), "white")
    draw = ImageDraw.Draw(canvas)

    x = 0
    for im, label in zip(resized, labels):
        canvas.paste(im.convert("RGB"), (x, label_h))
        draw.text((x + 8, 12), label[:28], fill=(15, 23, 42))
        x += im.size[0]

    return canvas


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def images_to_zip_bytes(image_dict):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, arr in image_dict.items():
            safe_name = name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
            img_bytes = image_to_download_bytes(arr.astype(np.uint8))
            zf.writestr(f"{safe_name}.png", img_bytes)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# ABLATION AND SENSITIVITY
# ============================================================
def run_ablation(reference, noisy, base_params):
    variants = {}

    ivpns_full = base_params["ivpns"].copy()

    full, _ = process_ivpns(
        noisy,
        operator=base_params["operator"],
        **ivpns_full
    )
    variants["Full proposed model"] = full

    ivpns_no_truth = ivpns_full.copy()
    ivpns_no_truth["use_truth"] = False
    no_truth, _ = process_ivpns(
        noisy,
        operator=base_params["operator"],
        **ivpns_no_truth
    )
    variants["Without truth-dominance"] = no_truth

    ivpns_no_refine = ivpns_full.copy()
    ivpns_no_refine["use_refinement"] = False
    no_refine, _ = process_ivpns(
        noisy,
        operator=base_params["operator"],
        **ivpns_no_refine
    )
    variants["Without adaptive refinement"] = no_refine

    wa, _ = process_ivpns(
        noisy,
        operator="IVPNSWA",
        **ivpns_full
    )
    wg, _ = process_ivpns(
        noisy,
        operator="IVPNSWG",
        **ivpns_full
    )

    variants["IVPNSWA only"] = wa
    variants["IVPNSWG only"] = wg

    rows = []
    for name, img in variants.items():
        mse, psnr, ssim = compute_metrics(reference, img)
        rows.append([name, mse, psnr, ssim])

    df = pd.DataFrame(rows, columns=["Variant", "MSE ↓", "PSNR (dB) ↑", "SSIM ↑"])
    return add_cqi(df.rename(columns={"Variant": "Method"})).rename(columns={"Method": "Variant"}), variants


def sensitivity_lambda(reference, noisy, base_params):
    rows = []
    for lam in np.linspace(0.10, 1.00, 10):
        ivpns_params = base_params["ivpns"].copy()
        ivpns_params["lam"] = float(lam)
        out, _ = process_ivpns(noisy, operator=base_params["operator"], **ivpns_params)
        mse, psnr, ssim = compute_metrics(reference, out)
        rows.append([lam, mse, psnr, ssim])
    return pd.DataFrame(rows, columns=["lambda", "MSE ↓", "PSNR (dB) ↑", "SSIM ↑"])


def sensitivity_epsilon(reference, noisy, base_params):
    rows = []
    for eps in np.linspace(0.00, 0.20, 9):
        ivpns_params = base_params["ivpns"].copy()
        ivpns_params["eps"] = float(eps)
        out, _ = process_ivpns(noisy, operator=base_params["operator"], **ivpns_params)
        mse, psnr, ssim = compute_metrics(reference, out)
        rows.append([eps, mse, psnr, ssim])
    return pd.DataFrame(rows, columns=["epsilon", "MSE ↓", "PSNR (dB) ↑", "SSIM ↑"])


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🧠 IVPNS-DenoiseLab Q1")
    st.caption("Batch research-grade image denoising system")

    uploaded_files = st.file_uploader(
        "Upload up to 10 images",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True
    )

    if uploaded_files is not None and len(uploaded_files) > 10:
        st.warning("Only the first 10 uploaded images will be processed.")
        uploaded_files = uploaded_files[:10]

    st.markdown("---")
    st.markdown("### 🧪 Noise Model")

    noise_type = st.selectbox(
        "Noise type",
        ["Gaussian", "Speckle", "Salt & Pepper", "None"],
        index=0
    )

    gaussian_sigma = st.slider("Gaussian σ", 1, 60, 15, 1)
    speckle_sigma = st.slider("Speckle σ", 0.01, 0.50, 0.12, 0.01)
    salt_pepper_prob = st.slider("Salt & Pepper probability", 0.01, 0.30, 0.03, 0.01)
    random_seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    st.markdown("---")
    st.markdown("### ⚙️ IVPNS Parameters")

    lam = st.slider("λ — indeterminacy control", 0.0, 1.0, 0.50, 0.05)
    eps = st.slider("ε — interval uncertainty width", 0.00, 0.25, 0.05, 0.01)
    window_size = st.selectbox("Neighbourhood window size", [3, 5, 7], index=1)
    sigma = st.slider("Spatial weight σ", 0.3, 5.0, 1.5, 0.1)
    eta = st.slider("η — indeterminacy penalty", 0.00, 1.00, 0.15, 0.05)
    kappa = st.slider("κ — falsity suppression weight", 0.00, 1.00, 0.30, 0.05)

    st.markdown("### 🔧 Truth-Dominance")
    use_truth = st.checkbox("Use truth-dominance adjustment", value=True)
    delta1 = st.slider("δ1 — truth enhancement", 0.00, 0.50, 0.05, 0.01)
    delta2 = st.slider("δ2 — indeterminacy suppression", 0.00, 0.50, 0.10, 0.01)
    delta3 = st.slider("δ3 — falsity suppression", 0.00, 0.50, 0.10, 0.01)

    st.markdown("### 🧩 Adaptive Refinement")
    use_refinement = st.checkbox("Use adaptive refinement", value=True)
    omega_min = st.slider("ω minimum", 0.00, 1.00, 0.15, 0.05)
    omega_max = st.slider("ω maximum", 0.00, 1.00, 0.85, 0.05)

    st.markdown("---")
    st.markdown("### 🧪 Baseline Settings")
    bilateral_sigma_color = st.slider("Bilateral sigma color", 0.01, 0.30, 0.06, 0.01)
    bilateral_sigma_spatial = st.slider("Bilateral sigma spatial", 1, 15, 5, 1)
    bm3d_sigma = st.slider("BM3D sigma_psd", 0.01, 0.30, 0.08, 0.01)
    include_bm3d = st.checkbox("Include BM3D if installed", value=True)

    st.markdown("---")
    resize_images = st.checkbox("Resize large images for faster processing", value=True)
    max_side = st.selectbox("Maximum image side", [256, 384, 512, 768], index=2)

    st.caption("For Q1 experiments, use fixed seed and report all parameter values.")


# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <div>
        <span class="hero-badge">Q1 Experimental System</span>
        <span class="hero-badge">Batch Upload</span>
        <span class="hero-badge">IVPNSWA / IVPNSWG</span>
        <span class="hero-badge">BM3D + NLM + Bilateral</span>
    </div>
    <div class="hero-title">IVPNS-DenoiseLab Q1</div>
    <div class="hero-subtitle">
        A research-grade Streamlit system for batch evaluation of the proposed Interval-Valued
        Pythagorean Neutrosophic image denoising framework. The system supports up to 10 uploaded
        images, multiple noise models, strong baseline comparisons, ablation analysis, sensitivity
        analysis, and publication-ready result tables.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# EMPTY STATE
# ============================================================
if not uploaded_files:
    st.markdown('<div class="section-title">Start the Demonstration</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card">
            <div class="card-title">1. Upload Images</div>
            <div class="card-text">Upload one to ten grayscale or color images. The system converts them to grayscale automatically.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
            <div class="card-title">2. Run IVPNS Denoising</div>
            <div class="card-text">Tune λ, ε, η, κ, aggregation operator, truth-dominance and adaptive refinement parameters.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card">
            <div class="card-title">3. Export Section 7 Results</div>
            <div class="card-text">Generate comparison tables, CQI ranking, ablation results and sensitivity plots for manuscript reporting.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Method Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline">
        Image Input → Noise Simulation → Normalization → IVPNS Transformation → Local Aggregation → Truth-Dominance → Score Reconstruction → Adaptive Refinement → Evaluation
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Important Note</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="note-box">
        This upgraded version uses the normalized IVPNS score function:
        S = [ᾱ(1 − ηβ̄) + κ(1 − γ̄)] / (1 + κ), which keeps the reconstructed score in [0,1].
        This should match the revised manuscript.
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ============================================================
# PROCESS ALL UPLOADED IMAGES
# ============================================================
params = {
    "window_size": window_size,
    "bilateral_sigma_color": bilateral_sigma_color,
    "bilateral_sigma_spatial": bilateral_sigma_spatial,
    "bm3d_sigma": bm3d_sigma,
    "ivpns": {
        "lam": lam,
        "eps": eps,
        "window_size": window_size,
        "sigma": sigma,
        "eta": eta,
        "kappa": kappa,
        "delta1": delta1,
       "delta2": delta2,
        "delta3": delta3,
        "use_truth": use_truth,
        "use_refinement": use_refinement,
        "omega_min": omega_min,
        "omega_max": omega_max
    }
}

# Safety fallback if operator widget is missing or not executed
try:
    operator
except NameError:
    operator = "IVPNSWA"

base_params = {
    "operator": operator,
    "ivpns": params["ivpns"]
}

all_results = []
all_outputs = {}
image_records = []

global_start = time.time()

for idx, uploaded in enumerate(uploaded_files):
    original_pil, original_255 = load_image_as_gray(uploaded)
    if resize_images:
        original_255 = resize_if_large(original_255, max_side=max_side)

    noisy_255 = add_noise(
        original_255,
        noise_type=noise_type,
        gaussian_sigma=gaussian_sigma,
        speckle_sigma=speckle_sigma,
        salt_pepper_prob=salt_pepper_prob,
        seed=random_seed + idx
    )

    t0 = time.time()
    methods, metrics_df, details_wa, details_wg = evaluate_methods(
        original_255.astype(np.uint8),
        noisy_255.astype(np.uint8),
        params,
        include_bm3d=include_bm3d
    )
    elapsed = time.time() - t0

    metrics_df.insert(0, "Image", uploaded.name)
    metrics_df.insert(1, "Noise", noise_type)
    metrics_df.insert(2, "Noise Level", {
        "Gaussian": gaussian_sigma,
        "Speckle": speckle_sigma,
        "Salt & Pepper": salt_pepper_prob,
        "None": 0
    }[noise_type])
    metrics_df["Processing Time (s)"] = elapsed

    all_results.append(metrics_df)

    image_records.append({
        "name": uploaded.name,
        "original": original_255.astype(np.uint8),
        "noisy": noisy_255.astype(np.uint8),
        "methods": methods,
        "metrics": metrics_df,
        "details_wa": details_wa,
        "details_wg": details_wg,
        "elapsed": elapsed
    })

    for method_name, arr in methods.items():
        all_outputs[f"{uploaded.name}_{method_name}"] = arr

total_elapsed = time.time() - global_start
results_df = pd.concat(all_results, ignore_index=True)


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================
st.markdown('<div class="section-title">Executive Processing Summary</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Images Processed", f"{len(image_records)}")
k2.metric("Noise Model", noise_type)
k3.metric("Primary Operator", operator)
k4.metric("Total Time", f"{total_elapsed:.3f} s")

st.markdown(f"""
<div class="research-box">
    Batch processing completed for <b>{len(image_records)}</b> image(s) using noise model <b>{noise_type}</b>.
    The system compares the proposed IVPNS methods with classical and advanced denoising baselines:
    <b>Mean</b>, <b>Median</b>, <b>Bilateral</b>, <b>Non-Local Means</b>{", <b>BM3D</b>" if BM3D_AVAILABLE and include_bm3d else ""}.
    Evaluation metrics include <b>MSE</b>, <b>PSNR</b>, <b>SSIM</b>, and <b>CQI</b>.
</div>
""", unsafe_allow_html=True)

if include_bm3d and not BM3D_AVAILABLE:
    st.warning("BM3D is not installed. To include BM3D, run: pip install bm3d")


# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🖼️ Results Dashboard",
    "📊 Metrics & CQI",
    "🧠 IVPNS Components",
    "🧪 Ablation Study",
    "📈 Sensitivity Analysis",
    "🔍 Pixel Inspector",
    "📝 Manuscript Export"
])


# ============================================================
# TAB 1: RESULTS DASHBOARD
# ============================================================
with tab1:
    st.markdown('<div class="section-title">Visual Result Dashboard</div>', unsafe_allow_html=True)

    selected_name = st.selectbox("Select image", [r["name"] for r in image_records])
    rec = next(r for r in image_records if r["name"] == selected_name)

    st.markdown('<div class="section-subtitle">Direct visual comparison across baseline methods and proposed IVPNS outputs.</div>', unsafe_allow_html=True)

    top_methods = ["Noisy/Input", f"Mean Filter ({window_size}x{window_size})", f"Median Filter ({window_size}x{window_size})"]
    mid_methods = ["Bilateral Filter", "Non-Local Means"]
    if "BM3D" in rec["methods"]:
        mid_methods.append("BM3D")
    proposed_methods = ["Proposed IVPNSWA", "Proposed IVPNSWG"]

    c1, c2, c3 = st.columns(3)
    c1.image(rec["original"], caption="Original Reference", use_container_width=True)
    c2.image(rec["noisy"], caption=f"Noisy Input ({noise_type})", use_container_width=True)
    c3.image(rec["methods"][proposed_methods[0]], caption="Proposed IVPNSWA", use_container_width=True)

    st.markdown('<div class="section-title">Baseline and Proposed Comparison</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    display_list = top_methods + mid_methods + proposed_methods
    for i, method in enumerate(display_list):
        with cols[i % 3]:
            st.image(rec["methods"][method], caption=method, use_container_width=True)

    st.markdown('<div class="section-title">Histogram Analysis</div>', unsafe_allow_html=True)
    h1, h2 = st.columns(2)
    with h1:
        st.pyplot(plot_histogram(rec["noisy"], "Noisy Image Histogram"))
    with h2:
        best_method = rec["metrics"].sort_values("CQI Score", ascending=False).iloc[0]["Method"]
        st.pyplot(plot_histogram(rec["methods"][best_method], f"Best Output Histogram: {best_method}"))

    comparison_methods = ["Original", "Noisy/Input"] + [m for m in display_list if m in rec["methods"]]
    comparison_images = [rec["original"], rec["noisy"]] + [rec["methods"][m] for m in display_list if m in rec["methods"]]
    panel = create_comparison_canvas(comparison_images, comparison_methods)

    buf = io.BytesIO()
    panel.save(buf, format="PNG")
    st.download_button(
        "⬇️ Download Comparison Panel",
        data=buf.getvalue(),
        file_name=f"{selected_name}_comparison_panel.png",
        mime="image/png"
    )


# ============================================================
# TAB 2: METRICS & CQI
# ============================================================
with tab2:
    st.markdown('<div class="section-title">Quantitative Performance Evaluation</div>', unsafe_allow_html=True)

    st.dataframe(
        results_df.style.format({
            "MSE ↓": "{:.4f}",
            "PSNR (dB) ↑": "{:.4f}",
            "SSIM ↑": "{:.4f}",
            "MSE Index": "{:.4f}",
            "PSNR Index": "{:.4f}",
            "SSIM Index": "{:.4f}",
            "CQI Score": "{:.4f}",
            "Processing Time (s)": "{:.4f}"
        }),
        use_container_width=True
    )

    avg_df = results_df.groupby("Method", as_index=False)[
        ["MSE ↓", "PSNR (dB) ↑", "SSIM ↑", "CQI Score", "Processing Time (s)"]
    ].mean()

    avg_df["Average Rank"] = avg_df["CQI Score"].rank(ascending=False, method="min").astype(int)
    avg_df = avg_df.sort_values("Average Rank")

    st.markdown('<div class="section-title">Average Performance Across Uploaded Images</div>', unsafe_allow_html=True)
    st.dataframe(
        avg_df.style.format({
            "MSE ↓": "{:.4f}",
            "PSNR (dB) ↑": "{:.4f}",
            "SSIM ↑": "{:.4f}",
            "CQI Score": "{:.4f}",
            "Processing Time (s)": "{:.4f}"
        }),
        use_container_width=True
    )

    best = avg_df.iloc[0]
    st.markdown(f"""
    <div class="research-box">
        Based on the average CQI score across uploaded images, <b>{best["Method"]}</b>
        achieves the best overall performance with CQI = <b>{best["CQI Score"]:.4f}</b>.
        CQI combines normalized MSE, PSNR, and SSIM using weights 0.30, 0.30, and 0.40 respectively.
    </div>
    """, unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        st.pyplot(plot_bar_metric(avg_df, "MSE ↓", "Average MSE by Method"))
    with b2:
        st.pyplot(plot_bar_metric(avg_df, "PSNR (dB) ↑", "Average PSNR by Method"))
    with b3:
        st.pyplot(plot_bar_metric(avg_df, "SSIM ↑", "Average SSIM by Method"))

    st.download_button(
        "⬇️ Download Full Metrics CSV",
        data=dataframe_to_csv_bytes(results_df),
        file_name="ivpns_full_metrics.csv",
        mime="text/csv"
    )

    st.download_button(
        "⬇️ Download Average Metrics CSV",
        data=dataframe_to_csv_bytes(avg_df),
        file_name="ivpns_average_metrics.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 3: IVPNS COMPONENTS
# ============================================================
with tab3:
    st.markdown('<div class="section-title">IVPNS Component Visualization</div>', unsafe_allow_html=True)

    selected_name_comp = st.selectbox("Select image for component view", [r["name"] for r in image_records], key="comp_select")
    rec_comp = next(r for r in image_records if r["name"] == selected_name_comp)
    details = rec_comp["details_wa"] if operator == "IVPNSWA" else rec_comp["details_wg"]

    st.markdown("""
    <div class="pipeline">
        g(x,y) → α truth map, β indeterminacy map, γ falsity map → interval bounds → aggregation → truth-dominance → score map
    </div>
    """, unsafe_allow_html=True)

    sub1, sub2, sub3, sub4 = st.tabs(["Core Components", "Interval Bounds", "Aggregated Components", "Score and Refinement"])

    with sub1:
        c1, c2, c3 = st.columns(3)
        c1.image((details["components"]["alpha"] * 255).astype(np.uint8), caption="Truth α Map", use_container_width=True)
        c2.image((details["components"]["beta"] * 255).astype(np.uint8), caption="Indeterminacy β Map", use_container_width=True)
        c3.image((details["components"]["gamma"] * 255).astype(np.uint8), caption="Falsity γ Map", use_container_width=True)

    with sub2:
        a1, a2, a3 = st.tabs(["α interval", "β interval", "γ interval"])
        with a1:
            c1, c2 = st.columns(2)
            c1.image((details["components"]["alpha_L"] * 255).astype(np.uint8), caption="α Lower Bound", use_container_width=True)
            c2.image((details["components"]["alpha_U"] * 255).astype(np.uint8), caption="α Upper Bound", use_container_width=True)
        with a2:
            c1, c2 = st.columns(2)
            c1.image((details["components"]["beta_L"] * 255).astype(np.uint8), caption="β Lower Bound", use_container_width=True)
            c2.image((details["components"]["beta_U"] * 255).astype(np.uint8), caption="β Upper Bound", use_container_width=True)
        with a3:
            c1, c2 = st.columns(2)
            c1.image((details["components"]["gamma_L"] * 255).astype(np.uint8), caption="γ Lower Bound", use_container_width=True)
            c2.image((details["components"]["gamma_U"] * 255).astype(np.uint8), caption="γ Upper Bound", use_container_width=True)

    with sub3:
        c1, c2, c3 = st.columns(3)
        c1.image((details["alpha_m"] * 255).astype(np.uint8), caption="Midpoint Truth ᾱ", use_container_width=True)
        c2.image((details["beta_m"] * 255).astype(np.uint8), caption="Midpoint Indeterminacy β̄", use_container_width=True)
        c3.image((details["gamma_m"] * 255).astype(np.uint8), caption="Midpoint Falsity γ̄", use_container_width=True)

    with sub4:
        c1, c2, c3 = st.columns(3)
        c1.image((details["score"] * 255).astype(np.uint8), caption="Normalized IVPNS Score", use_container_width=True)
        c2.image((details["omega_map"] * 255).astype(np.uint8), caption="Adaptive ω Map", use_container_width=True)
        c3.image((details["refined_score"] * 255).astype(np.uint8), caption="Refined Score", use_container_width=True)

    st.markdown('<div class="section-title">Spatial Weight Kernel</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(np.round(details["kernel"], 5)), use_container_width=True)


# ============================================================
# TAB 4: ABLATION STUDY
# ============================================================
with tab4:
    st.markdown('<div class="section-title">Ablation Study</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">This analysis evaluates the effect of truth-dominance, adaptive refinement, and aggregation operator selection.</div>', unsafe_allow_html=True)

    selected_name_ab = st.selectbox("Select image for ablation", [r["name"] for r in image_records], key="ablation_select")
    rec_ab = next(r for r in image_records if r["name"] == selected_name_ab)

    ablation_df, ablation_outputs = run_ablation(rec_ab["original"], rec_ab["noisy"], base_params)

    st.dataframe(
        ablation_df.style.format({
            "MSE ↓": "{:.4f}",
            "PSNR (dB) ↑": "{:.4f}",
            "SSIM ↑": "{:.4f}",
            "MSE Index": "{:.4f}",
            "PSNR Index": "{:.4f}",
            "SSIM Index": "{:.4f}",
            "CQI Score": "{:.4f}"
        }),
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)
    ablation_items = list(ablation_outputs.items())
    for i, (name, img) in enumerate(ablation_items):
        with [c1, c2, c3][i % 3]:
            st.image(img, caption=name, use_container_width=True)

    st.download_button(
        "⬇️ Download Ablation CSV",
        data=dataframe_to_csv_bytes(ablation_df),
        file_name=f"{selected_name_ab}_ablation.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 5: SENSITIVITY ANALYSIS
# ============================================================
with tab5:
    st.markdown('<div class="section-title">Parameter Sensitivity Analysis</div>', unsafe_allow_html=True)

    selected_name_sens = st.selectbox("Select image for sensitivity analysis", [r["name"] for r in image_records], key="sens_select")
    rec_sens = next(r for r in image_records if r["name"] == selected_name_sens)

    sens_lam = sensitivity_lambda(rec_sens["original"], rec_sens["noisy"], base_params)
    sens_eps = sensitivity_epsilon(rec_sens["original"], rec_sens["noisy"], base_params)

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_sensitivity(sens_lam, "lambda", "PSNR (dB) ↑", "Sensitivity of PSNR to λ"))
        st.dataframe(sens_lam.style.format({"lambda": "{:.2f}", "MSE ↓": "{:.4f}", "PSNR (dB) ↑": "{:.4f}", "SSIM ↑": "{:.4f}"}), use_container_width=True)
    with c2:
        st.pyplot(plot_sensitivity(sens_eps, "epsilon", "PSNR (dB) ↑", "Sensitivity of PSNR to ε"))
        st.dataframe(sens_eps.style.format({"epsilon": "{:.2f}", "MSE ↓": "{:.4f}", "PSNR (dB) ↑": "{:.4f}", "SSIM ↑": "{:.4f}"}), use_container_width=True)

    st.download_button(
        "⬇️ Download Lambda Sensitivity CSV",
        data=dataframe_to_csv_bytes(sens_lam),
        file_name=f"{selected_name_sens}_lambda_sensitivity.csv",
        mime="text/csv"
    )
    st.download_button(
        "⬇️ Download Epsilon Sensitivity CSV",
        data=dataframe_to_csv_bytes(sens_eps),
        file_name=f"{selected_name_sens}_epsilon_sensitivity.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 6: PIXEL INSPECTOR
# ============================================================
with tab6:
    st.markdown('<div class="section-title">Pixel-Level IVPNS Inspection</div>', unsafe_allow_html=True)

    selected_name_pix = st.selectbox("Select image for pixel inspection", [r["name"] for r in image_records], key="pix_select")
    rec_pix = next(r for r in image_records if r["name"] == selected_name_pix)
    details_pix = rec_pix["details_wa"] if operator == "IVPNSWA" else rec_pix["details_wg"]

    row = st.slider("Select row y", 0, rec_pix["noisy"].shape[0] - 1, rec_pix["noisy"].shape[0] // 2)
    col = st.slider("Select column x", 0, rec_pix["noisy"].shape[1] - 1, rec_pix["noisy"].shape[1] // 2)

    pixel_info = {
        "Pixel coordinate": f"({row}, {col})",
        "Original intensity": float(rec_pix["original"][row, col]),
        "Noisy intensity": float(rec_pix["noisy"][row, col]),
        "Normalized g": float(details_pix["g"][row, col]),
        "Truth α": float(details_pix["components"]["alpha"][row, col]),
        "Indeterminacy β": float(details_pix["components"]["beta"][row, col]),
        "Falsity γ": float(details_pix["components"]["gamma"][row, col]),
        "α interval": f"[{details_pix['components']['alpha_L'][row, col]:.4f}, {details_pix['components']['alpha_U'][row, col]:.4f}]",
        "β interval": f"[{details_pix['components']['beta_L'][row, col]:.4f}, {details_pix['components']['beta_U'][row, col]:.4f}]",
        "γ interval": f"[{details_pix['components']['gamma_L'][row, col]:.4f}, {details_pix['components']['gamma_U'][row, col]:.4f}]",
        "Score S": float(details_pix["score"][row, col]),
        "Refined score S'": float(details_pix["refined_score"][row, col]),
        "Adaptive ω": float(details_pix["omega_map"][row, col])
    }

    st.table(pd.DataFrame(pixel_info.items(), columns=["Item", "Value"]))
    st.markdown('<div class="section-title">Normalized Image Matrix Preview</div>', unsafe_allow_html=True)
    st.dataframe(small_matrix_preview(details_pix["g"]), use_container_width=True)


# ============================================================
# TAB 7: MANUSCRIPT EXPORT
# ============================================================
with tab7:
    st.markdown('<div class="section-title">Manuscript Export and Section 7 Support</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">Suggested Section 7 Experimental Description</div>
        <div class="card-text">
        The upgraded system evaluates the proposed IVPNS denoising framework against classical and advanced
        baseline methods, including mean filtering, median filtering, bilateral filtering, Non-Local Means, and
        BM3D when available. Experiments are conducted using uploaded benchmark images under controlled
        noise simulations. Quantitative assessment is performed using MSE, PSNR, SSIM, and the Composite Quality
        Index (CQI), while ablation and sensitivity analyses are used to examine the influence of truth-dominance,
        adaptive refinement, aggregation operator selection, λ, and ε.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Current Parameter Configuration</div>', unsafe_allow_html=True)

    param_table = pd.DataFrame([
        ["Noise type", noise_type],
        ["Gaussian σ", gaussian_sigma],
        ["Speckle σ", speckle_sigma],
        ["Salt & Pepper probability", salt_pepper_prob],
        ["λ", lam],
        ["ε", eps],
        ["Window size", f"{window_size}x{window_size}"],
        ["Spatial σ", sigma],
        ["Primary operator", operator],
        ["η", eta],
        ["κ", kappa],
        ["δ1", delta1],
        ["δ2", delta2],
        ["δ3", delta3],
        ["Truth-dominance", use_truth],
        ["Adaptive refinement", use_refinement],
        ["ω minimum", omega_min],
        ["ω maximum", omega_max],
        ["BM3D available", BM3D_AVAILABLE]
    ], columns=["Parameter", "Value"])

    st.dataframe(param_table, use_container_width=True)

    st.download_button(
        "⬇️ Download Parameter CSV",
        data=dataframe_to_csv_bytes(param_table),
        file_name="ivpns_parameter_configuration.csv",
        mime="text/csv"
    )

    st.download_button(
        "⬇️ Download All Output Images ZIP",
        data=images_to_zip_bytes(all_outputs),
        file_name="ivpns_all_outputs.zip",
        mime="application/zip"
    )

    st.markdown("""
    <div class="note-box">
        For final Q1 submission, use the same fixed parameter setting for all images within each experiment,
        report the noise model and noise level explicitly, and include average performance across all uploaded images.
    </div>
    """, unsafe_allow_html=True)
