# ============================================================
# IVPNS-DenoiseLab: Interval-Valued Pythagorean Neutrosophic
# Image Denoising — interactive research prototype
#
# The denoising is performed by the IVPNS pipeline itself
# (variance-driven indeterminacy + bilateral-weighted IVPNSWA/IVPNSWG
# aggregation + consistency-preserving score + adaptive refinement).
# This app is faithful to the equations in the manuscript; there is
# no hidden mean/median post-filter.
#
# Run:
#   streamlit run streamlit_app.py
# Requirements:
#   pip install streamlit numpy pillow scipy scikit-image pandas matplotlib
# Optional:
#   pip install bm3d
# ============================================================

import io
import time
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter, median_filter
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_bilateral, denoise_nl_means
import matplotlib.pyplot as plt

# ---- IVPNS engine (manuscript-consistent v6) ----
from ivpns_v6 import process_ivpns, normalize_image, reconstruct_image

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
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #F8FAFC 0%, #EEF4FF 45%, #F7FBFF 100%); }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
.hero { padding: 34px 38px; border-radius: 28px;
    background: linear-gradient(135deg, #0B2447 0%, #19376D 48%, #2B5DAA 100%);
    box-shadow: 0 22px 50px rgba(11,36,71,0.22); color: white; margin-bottom: 24px; }
.hero-title { font-size: 42px; font-weight: 900; letter-spacing: -0.8px; margin-bottom: 8px; }
.hero-subtitle { font-size: 17px; color: #DDEBFF; line-height: 1.55; max-width: 1100px; }
.hero-badge { display: inline-block; padding: 7px 13px; border-radius: 999px;
    background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.25);
    color: #FFFFFF; font-size: 13px; font-weight: 700; margin-right: 8px; margin-bottom: 10px; }
.section-title { font-size: 25px; font-weight: 850; color: #102A43; margin-top: 15px; margin-bottom: 13px; }
.section-subtitle { font-size: 15px; color: #52616B; margin-top: -5px; margin-bottom: 18px; }
.card { background: rgba(255,255,255,0.88); border: 1px solid rgba(210,221,235,0.95);
    padding: 19px 20px; border-radius: 22px; box-shadow: 0 12px 32px rgba(15,23,42,0.065); margin-bottom: 14px; }
.card-title { font-size: 16px; font-weight: 800; color: #123C69; margin-bottom: 7px; }
.card-text { font-size: 14px; color: #4B5563; line-height: 1.45; }
.pipeline { padding: 15px 18px; border-radius: 18px;
    background: linear-gradient(90deg, #FFFFFF 0%, #EFF6FF 100%); border: 1px solid #D9E8FF;
    color: #0B2447; font-size: 15px; font-weight: 750; text-align: center;
    box-shadow: 0 7px 18px rgba(11,36,71,0.06); }
.research-box { padding: 18px 20px; border-radius: 20px;
    background: linear-gradient(135deg, #ECFDF5 0%, #F0FDFA 100%); border: 1px solid #A7F3D0;
    color: #064E3B; font-size: 15px; line-height: 1.55; box-shadow: 0 10px 25px rgba(6,78,59,0.08); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #FFFFFF 0%, #EEF4FF 100%); }
[data-testid="stMetricValue"] { font-size: 25px; font-weight: 850; color: #0B2447; }
[data-testid="stMetricLabel"] { font-size: 13px; color: #425466; }
.stDownloadButton button { border-radius: 14px; font-weight: 800; border: 1px solid #2B5DAA; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# IMAGE / NOISE HELPERS
# ============================================================
def load_image_as_gray(uploaded_file):
    image = Image.open(uploaded_file).convert("L")
    return image, np.array(image).astype(np.float64)


def resize_if_large(arr, max_side=512):
    h, w = arr.shape
    if max(h, w) <= max_side:
        return arr
    scale = max_side / max(h, w)
    img = Image.fromarray(arr.astype(np.uint8)).resize((int(w * scale), int(h * scale)))
    return np.array(img).astype(np.float64)


def image_to_download_bytes(arr_uint8):
    buffer = io.BytesIO()
    Image.fromarray(arr_uint8).save(buffer, format="PNG")
    return buffer.getvalue()


def add_noise(img_255, noise_type="Gaussian", gaussian_sigma=15,
              speckle_sigma=0.12, salt_pepper_prob=0.03, seed=42):
    rng = np.random.default_rng(seed)
    img = img_255.astype(np.float64)
    if noise_type == "None":
        return np.clip(img, 0, 255)
    if noise_type == "Gaussian":
        noisy = img + rng.normal(0, gaussian_sigma, img.shape)
    elif noise_type == "Speckle":
        g = img / 255.0
        noisy = (g + g * rng.normal(0, speckle_sigma, img.shape)) * 255.0
    elif noise_type == "Salt & Pepper":
        noisy = img.copy()
        rnd = rng.random(img.shape)
        noisy[rnd < salt_pepper_prob / 2] = 0
        noisy[(rnd >= salt_pepper_prob / 2) & (rnd < salt_pepper_prob)] = 255
    else:
        noisy = img.copy()
    return np.clip(noisy, 0, 255)


# ============================================================
# BASELINE FILTERS (for comparison only — NOT part of IVPNS)
# ============================================================
def mean_filter_img(img_255, window_size=3):
    return np.clip(uniform_filter(img_255, size=window_size), 0, 255).astype(np.uint8)


def median_filter_img(img_255, window_size=3):
    return np.clip(median_filter(img_255, size=window_size), 0, 255).astype(np.uint8)


def bilateral_filter_img(img_255, sigma_color=0.06, sigma_spatial=5):
    out = denoise_bilateral(img_255 / 255.0, sigma_color=sigma_color,
                            sigma_spatial=sigma_spatial, channel_axis=None)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def robust_noise_sigma(img01):
    local_mean = uniform_filter(img01.astype(np.float64), size=3)
    residual = img01 - local_mean
    mad = np.median(np.abs(residual - np.median(residual)))
    return float(np.clip(1.4826 * mad, 0.01, 0.20))


def nlm_filter_img(img_255, patch_size=5, patch_distance=6):
    img01 = np.clip(img_255 / 255.0, 0, 1)
    h = max(0.03, 1.15 * robust_noise_sigma(img01))
    out = denoise_nl_means(img01, h=h, patch_size=patch_size,
                           patch_distance=patch_distance, fast_mode=True, channel_axis=None)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def bm3d_filter_img(img_255, sigma_psd=0.08):
    if not BM3D_AVAILABLE:
        return None
    out = bm3d(img_255 / 255.0, sigma_psd=sigma_psd)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


# ============================================================
# METRICS
# ============================================================
def compute_metrics(reference, processed):
    ref, pro = reference.astype(np.float64), processed.astype(np.float64)
    mse = mean_squared_error(ref, pro)
    psnr = float("inf") if mse == 0 else peak_signal_noise_ratio(ref, pro, data_range=255)
    ssim = structural_similarity(ref, pro, data_range=255)
    return float(mse), float(psnr), float(ssim)


def evaluate_methods(reference, noisy, params, include_bm3d=True):
    methods = {"Noisy/Input": noisy.astype(np.uint8)}
    ws = params["window_size"]
    methods[f"Mean ({ws}x{ws})"] = mean_filter_img(noisy, ws)
    methods[f"Median ({ws}x{ws})"] = median_filter_img(noisy, ws)
    methods["Bilateral"] = bilateral_filter_img(noisy, params["bilateral_sigma_color"],
                                                params["bilateral_sigma_spatial"])
    methods["Non-Local Means"] = nlm_filter_img(noisy)
    if include_bm3d and BM3D_AVAILABLE:
        methods["BM3D"] = bm3d_filter_img(noisy, sigma_psd=params["bm3d_sigma"])

    # ---- proposed IVPNS (denoising done by the IVPNS pipeline) ----
    wa, details_wa = process_ivpns(noisy, operator="IVPNSWA", **params["ivpns"])
    wg, details_wg = process_ivpns(noisy, operator="IVPNSWG", **params["ivpns"])
    methods["Proposed IVPNSWA"] = wa
    methods["Proposed IVPNSWG"] = wg

    rows = []
    for name, output in methods.items():
        mse, psnr, ssim = compute_metrics(reference, output)
        rows.append([name, mse, psnr, ssim])
    metrics_df = pd.DataFrame(rows, columns=["Method", "MSE ↓", "PSNR (dB) ↑", "SSIM ↑"])
    metrics_df["Rank"] = metrics_df["PSNR (dB) ↑"].rank(ascending=False, method="min").astype(int)
    return methods, metrics_df, details_wa, details_wg


def plot_bar_metric(df, metric_col, title):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(df["Method"], df[metric_col])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_col); ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25); fig.tight_layout()
    return fig


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def images_to_zip_bytes(image_dict):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, arr in image_dict.items():
            safe = name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
            zf.writestr(f"{safe}.png", image_to_download_bytes(arr.astype(np.uint8)))
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🧠 IVPNS-DenoiseLab")
    st.caption("Interval-valued Pythagorean neutrosophic image denoising")

    uploaded_files = st.file_uploader(
        "Upload up to 10 images",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )
    if uploaded_files and len(uploaded_files) > 10:
        st.warning("Only the first 10 uploaded images will be processed.")
        uploaded_files = uploaded_files[:10]

    st.markdown("---")
    st.markdown("### 🧪 Noise Model")
    noise_type = st.selectbox("Noise type", ["Gaussian", "Speckle", "Salt & Pepper", "None"], index=0)
    gaussian_sigma = st.slider("Gaussian σ", 1, 60, 20, 1)
    speckle_sigma = st.slider("Speckle σ", 0.01, 0.50, 0.12, 0.01)
    salt_pepper_prob = st.slider("Salt & Pepper probability", 0.01, 0.30, 0.03, 0.01)
    random_seed = st.number_input("Random seed", 0, 9999, 42, 1)

    st.markdown("---")
    st.markdown("### ⚙️ IVPNS Parameters")
    lam = st.slider("λ — indeterminacy scale", 0.0, 1.0, 0.40, 0.05)
    eps = st.slider("ε — interval half-width", 0.00, 0.25, 0.05, 0.01)
    operator = st.selectbox("Aggregation operator", ["IVPNSWG", "IVPNSWA"], index=0)
    eta = st.slider("η — indeterminacy weight in score", 0.00, 1.00, 0.20, 0.05)
    kappa = st.slider("κ — falsity weight in score", 0.00, 1.00, 0.30, 0.05)
    rho = st.slider("ρ — refinement strength", 0.00, 1.00, 0.90, 0.05)

    st.markdown("### 🔧 Truth-Dominance (uncertainty-gated)")
    use_truth = st.checkbox("Use truth-dominance adjustment", value=True)
    d1 = st.slider("δ1 — truth enhancement", 0.00, 0.50, 0.20, 0.01)
    d2 = st.slider("δ2 — indeterminacy suppression", 0.00, 0.50, 0.30, 0.01)
    d3 = st.slider("δ3 — falsity suppression", 0.00, 0.50, 0.30, 0.01)
    use_refinement = st.checkbox("Use adaptive refinement", value=True)
    noise_adaptive = st.checkbox("Noise-adaptive window/range", value=True)

    st.markdown("---")
    st.markdown("### 🧪 Baseline Settings")
    bilateral_sigma_color = st.slider("Bilateral sigma color", 0.01, 0.30, 0.06, 0.01)
    bilateral_sigma_spatial = st.slider("Bilateral sigma spatial", 1, 15, 5, 1)
    bm3d_sigma = st.slider("BM3D sigma_psd", 0.01, 0.30, 0.08, 0.01)
    include_bm3d = st.checkbox("Include BM3D if installed", value=True)

    st.markdown("---")
    resize_images = st.checkbox("Resize large images for faster processing", value=True)
    max_side = st.selectbox("Maximum image side", [256, 384, 512, 768], index=2)
    st.caption("Use a fixed seed and report all parameter values for reproducibility.")


# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <div>
        <span class="hero-badge">IVPNSWA / IVPNSWG</span>
        <span class="hero-badge">Batch Upload</span>
        <span class="hero-badge">Uncertainty-Aware</span>
        <span class="hero-badge">BM3D + NLM + Bilateral</span>
    </div>
    <div class="hero-title">IVPNS-DenoiseLab</div>
    <div class="hero-subtitle">
        An interactive research prototype for the Interval-Valued Pythagorean Neutrosophic
        image denoising framework. The denoising is performed by the IVPNS pipeline itself —
        variance-driven indeterminacy, bilateral-weighted aggregation, a consistency-preserving
        score function, and adaptive refinement — with classical and non-local baselines shown
        for comparison and explicit per-pixel uncertainty maps.
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
        st.markdown("""<div class="card"><div class="card-title">1. Upload Images</div>
        <div class="card-text">Upload one to ten grayscale or color images. Colour images are converted to grayscale automatically.</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card"><div class="card-title">2. Run IVPNS Denoising</div>
        <div class="card-text">Tune λ, ε, η, κ, ρ, the aggregation operator, and the truth-dominance and refinement parameters.</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card"><div class="card-title">3. Export Results</div>
        <div class="card-text">Generate comparison tables and per-pixel uncertainty maps, and download the outputs for reporting.</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Method Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""<div class="pipeline">
        Image Input → Noise Simulation → Normalization → IVPNS Transformation → Local Aggregation → Truth-Dominance → Score Reconstruction → Adaptive Refinement → Evaluation
    </div>""", unsafe_allow_html=True)
    st.stop()


# ============================================================
# PROCESS ALL UPLOADED IMAGES
# ============================================================
params = {
    "window_size": 3,  # baseline filter window (baselines only)
    "bilateral_sigma_color": bilateral_sigma_color,
    "bilateral_sigma_spatial": bilateral_sigma_spatial,
    "bm3d_sigma": bm3d_sigma,
    "ivpns": {
        "lam": lam, "eps": eps, "eta": eta, "kappa": kappa, "rho": rho,
        "d1": d1, "d2": d2, "d3": d3,
        "use_truth": use_truth, "use_refinement": use_refinement,
        "noise_adaptive": noise_adaptive,
    },
}

all_results, all_outputs, image_records = [], {}, []
global_start = time.time()

for idx, uploaded in enumerate(uploaded_files):
    _, original_255 = load_image_as_gray(uploaded)
    if resize_images:
        original_255 = resize_if_large(original_255, max_side=max_side)

    noisy_255 = add_noise(original_255, noise_type, gaussian_sigma,
                          speckle_sigma, salt_pepper_prob, seed=random_seed + idx)

    t0 = time.time()
    methods, metrics_df, details_wa, details_wg = evaluate_methods(
        original_255.astype(np.uint8), noisy_255.astype(np.uint8), params, include_bm3d)
    elapsed = time.time() - t0

    metrics_df.insert(0, "Image", uploaded.name)
    metrics_df.insert(1, "Noise", noise_type)
    metrics_df["Time (s)"] = elapsed
    all_results.append(metrics_df)
    image_records.append({"name": uploaded.name, "original": original_255.astype(np.uint8),
                          "noisy": noisy_255.astype(np.uint8), "methods": methods,
                          "metrics": metrics_df, "details_wa": details_wa,
                          "details_wg": details_wg, "elapsed": elapsed})
    for method_name, arr in methods.items():
        all_outputs[f"{uploaded.name}_{method_name}"] = arr

total_elapsed = time.time() - global_start
results_df = pd.concat(all_results, ignore_index=True)


# ============================================================
# SUMMARY
# ============================================================
st.markdown('<div class="section-title">Processing Summary</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Images Processed", f"{len(image_records)}")
k2.metric("Noise Model", noise_type)
k3.metric("Primary Operator", operator)
k4.metric("Total Time", f"{total_elapsed:.3f} s")

st.markdown(f"""<div class="research-box">
    Batch processing completed for <b>{len(image_records)}</b> image(s) under noise model <b>{noise_type}</b>.
    The proposed IVPNS methods are compared with <b>Mean</b>, <b>Median</b>, <b>Bilateral</b>,
    <b>Non-Local Means</b>{", <b>BM3D</b>" if BM3D_AVAILABLE and include_bm3d else ""}.
    The denoising is produced by the IVPNS aggregation pipeline; metrics reported are MSE, PSNR, and SSIM.
</div>""", unsafe_allow_html=True)
if include_bm3d and not BM3D_AVAILABLE:
    st.warning("BM3D is not installed. To include BM3D, run: pip install bm3d")


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Results Dashboard", "📊 Metrics", "🧠 Uncertainty Maps", "📝 Export"])

with tab1:
    st.markdown('<div class="section-title">Visual Result Dashboard</div>', unsafe_allow_html=True)
    selected_name = st.selectbox("Select image", [r["name"] for r in image_records])
    rec = next(r for r in image_records if r["name"] == selected_name)
    c1, c2, c3 = st.columns(3)
    c1.image(rec["original"], caption="Original Reference", use_container_width=True)
    c2.image(rec["noisy"], caption=f"Noisy Input ({noise_type})", use_container_width=True)
    c3.image(rec["methods"]["Proposed IVPNSWG"], caption="Proposed IVPNSWG", use_container_width=True)

    st.markdown('<div class="section-title">Baseline and Proposed Comparison</div>', unsafe_allow_html=True)
    display_list = [m for m in rec["methods"].keys()]
    cols = st.columns(3)
    for i, method in enumerate(display_list):
        with cols[i % 3]:
            st.image(rec["methods"][method], caption=method, use_container_width=True)

with tab2:
    st.markdown('<div class="section-title">Quantitative Performance</div>', unsafe_allow_html=True)
    st.dataframe(results_df.style.format({"MSE ↓": "{:.4f}", "PSNR (dB) ↑": "{:.4f}",
                                          "SSIM ↑": "{:.4f}", "Time (s)": "{:.4f}"}),
                 use_container_width=True)
    avg_df = results_df.groupby("Method", as_index=False)[["MSE ↓", "PSNR (dB) ↑", "SSIM ↑"]].mean()
    avg_df["Rank"] = avg_df["PSNR (dB) ↑"].rank(ascending=False, method="min").astype(int)
    avg_df = avg_df.sort_values("Rank")
    st.markdown('<div class="section-title">Average Across Uploaded Images</div>', unsafe_allow_html=True)
    st.dataframe(avg_df.style.format({"MSE ↓": "{:.4f}", "PSNR (dB) ↑": "{:.4f}", "SSIM ↑": "{:.4f}"}),
                 use_container_width=True)
    b1, b2, b3 = st.columns(3)
    with b1: st.pyplot(plot_bar_metric(avg_df, "MSE ↓", "Average MSE"))
    with b2: st.pyplot(plot_bar_metric(avg_df, "PSNR (dB) ↑", "Average PSNR"))
    with b3: st.pyplot(plot_bar_metric(avg_df, "SSIM ↑", "Average SSIM"))
    st.download_button("⬇️ Download Metrics CSV", data=dataframe_to_csv_bytes(results_df),
                       file_name="ivpns_metrics.csv", mime="text/csv")

with tab3:
    st.markdown('<div class="section-title">Per-Pixel Uncertainty Maps</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">The IVPNS decomposition exposes truth, indeterminacy, and falsity, plus the adaptive refinement weight — the interpretability that classical and learned denoisers do not provide.</div>', unsafe_allow_html=True)
    sel = st.selectbox("Select image", [r["name"] for r in image_records], key="unc")
    r = next(x for x in image_records if x["name"] == sel)
    d = r["details_wa"] if operator == "IVPNSWA" else r["details_wg"]
    c1, c2, c3 = st.columns(3)
    c1.image((np.clip(d["alpha"], 0, 1) * 255).astype(np.uint8), caption="Truth ᾱ", use_container_width=True)
    c2.image((np.clip(d["beta"], 0, 1) * 255).astype(np.uint8), caption="Indeterminacy β̄", use_container_width=True)
    c3.image((np.clip(d["gamma"], 0, 1) * 255).astype(np.uint8), caption="Falsity γ̄", use_container_width=True)
    c4, c5 = st.columns(2)
    c4.image((np.clip(d["score"], 0, 1) * 255).astype(np.uint8), caption="IVPNS Score S", use_container_width=True)
    c5.image((np.clip(d["omega"], 0, 1) * 255).astype(np.uint8), caption="Adaptive weight ω", use_container_width=True)

with tab4:
    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
    param_table = pd.DataFrame([
        ["Noise type", noise_type], ["Gaussian σ", gaussian_sigma],
        ["λ", lam], ["ε", eps], ["Operator", operator],
        ["η", eta], ["κ", kappa], ["ρ", rho],
        ["δ1", d1], ["δ2", d2], ["δ3", d3],
        ["Truth-dominance", use_truth], ["Adaptive refinement", use_refinement],
        ["Noise-adaptive", noise_adaptive], ["Seed", random_seed],
        ["BM3D available", BM3D_AVAILABLE],
    ], columns=["Parameter", "Value"])
    st.dataframe(param_table, use_container_width=True)
    st.download_button("⬇️ Download Parameter CSV", data=dataframe_to_csv_bytes(param_table),
                       file_name="ivpns_parameters.csv", mime="text/csv")
    st.download_button("⬇️ Download All Output Images ZIP", data=images_to_zip_bytes(all_outputs),
                       file_name="ivpns_outputs.zip", mime="application/zip")
