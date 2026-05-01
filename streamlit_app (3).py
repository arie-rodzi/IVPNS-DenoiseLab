

import io
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.ndimage import uniform_filter, median_filter, convolve
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt


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
/* Main background */
.stApp {
    background: linear-gradient(135deg, #F8FAFC 0%, #EEF4FF 45%, #F7FBFF 100%);
}

/* Hide default menu/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hero section */
.hero {
    padding: 34px 38px;
    border-radius: 28px;
    background: linear-gradient(135deg, #0B2447 0%, #19376D 48%, #2B5DAA 100%);
    box-shadow: 0 22px 50px rgba(11,36,71,0.22);
    color: white;
    margin-bottom: 24px;
}
.hero-title {
    font-size: 44px;
    font-weight: 900;
    letter-spacing: -0.8px;
    margin-bottom: 8px;
}
.hero-subtitle {
    font-size: 18px;
    color: #DDEBFF;
    line-height: 1.55;
    max-width: 1050px;
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

/* Section title */
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

/* Cards */
.card {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(210, 221, 235, 0.95);
    padding: 19px 20px;
    border-radius: 22px;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.065);
    margin-bottom: 14px;
}
.card-compact {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(210, 221, 235, 0.95);
    padding: 13px 15px;
    border-radius: 18px;
    box-shadow: 0 7px 20px rgba(15, 23, 42, 0.045);
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

/* Pipeline */
.pipeline {
    padding: 15px 18px;
    border-radius: 18px;
    background: linear-gradient(90deg, #FFFFFF 0%, #EFF6FF 100%);
    border: 1px solid #D9E8FF;
    color: #0B2447;
    font-size: 16px;
    font-weight: 750;
    text-align: center;
    box-shadow: 0 7px 18px rgba(11,36,71,0.06);
}

/* Formula box */
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

/* Alert */
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

/* Warning note */
.note-box {
    padding: 16px 18px;
    border-radius: 18px;
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    color: #7C2D12;
    font-size: 14px;
    line-height: 1.50;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #EEF4FF 100%);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #0B2447;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: 26px;
    font-weight: 850;
    color: #0B2447;
}
[data-testid="stMetricLabel"] {
    font-size: 13px;
    color: #425466;
}

/* Buttons */
.stDownloadButton button {
    border-radius: 14px;
    font-weight: 800;
    border: 1px solid #2B5DAA;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 14px 14px 0 0;
    padding: 10px 16px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNCTIONS
# ============================================================
def load_image_as_gray(uploaded_file):
    image = Image.open(uploaded_file).convert("L")
    arr = np.array(image).astype(np.float64)
    return image, arr


def normalize_image(img_255):
    return np.clip(img_255 / 255.0, 0, 1)


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

    for k in out:
        out[k] = np.clip(out[k], 0, 1)

    return out, kernel


def defuzzify_score(agg):
    raw_score = (
        agg["alpha_L"] + agg["alpha_U"]
        - agg["beta_L"] - agg["beta_U"]
        - agg["gamma_L"] - agg["gamma_U"]
    ) / 3.0

    direct_clipped = np.clip(raw_score, 0, 1)
    normalized = np.clip((raw_score + 2.0 / 3.0) / (4.0 / 3.0), 0, 1)
    alpha_dominance = np.clip((agg["alpha_L"] + agg["alpha_U"]) / 2.0, 0, 1)

    return raw_score, direct_clipped, normalized, alpha_dominance


def reconstruct_image(score_01):
    return np.clip(255.0 * score_01, 0, 255).astype(np.uint8)


def add_noise(img_255, noise_type="None", gaussian_sigma=15, speckle_sigma=0.12, salt_pepper_prob=0.03):
    img = img_255.astype(np.float64)

    if noise_type == "None":
        return img.copy()

    if noise_type == "Gaussian":
        noisy = img + np.random.normal(0, gaussian_sigma, img.shape)

    elif noise_type == "Speckle":
        g = img / 255.0
        noisy = g + g * np.random.normal(0, speckle_sigma, img.shape)
        noisy = noisy * 255.0

    elif noise_type == "Salt & Pepper":
        noisy = img.copy()
        rnd = np.random.rand(*img.shape)
        noisy[rnd < salt_pepper_prob / 2] = 0
        noisy[(rnd >= salt_pepper_prob / 2) & (rnd < salt_pepper_prob)] = 255

    else:
        noisy = img.copy()

    return np.clip(noisy, 0, 255)


def compute_metrics(reference, processed):
    ref = reference.astype(np.float64)
    pro = processed.astype(np.float64)

    mse = mean_squared_error(ref, pro)

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = peak_signal_noise_ratio(ref, pro, data_range=255)

    ssim = structural_similarity(ref, pro, data_range=255)
    return mse, psnr, ssim


def image_to_download_bytes(arr_uint8):
    img = Image.fromarray(arr_uint8)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def plot_histogram(image_255, title):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(image_255.flatten(), bins=60)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25)
    return fig


def small_matrix_preview(arr, size=8):
    return pd.DataFrame(np.round(arr[:size, :size], 4))


def create_comparison_canvas(images, labels):
    pil_images = [Image.fromarray(img.astype(np.uint8)).convert("L") for img in images]
    widths, heights = zip(*(im.size for im in pil_images))
    w, h = max(widths), max(heights)

    canvas = Image.new("RGB", (w * len(pil_images), h + 42), "white")

    for i, im in enumerate(pil_images):
        im = im.resize((w, h))
        rgb = im.convert("RGB")
        canvas.paste(rgb, (i * w, 42))

    return canvas


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🧠 IVPNS-DenoiseLab")
    st.caption("Research-grade image denoising prototype")

    uploaded = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )

    st.markdown("---")
    st.markdown("### ⚙️ IVPNS Parameters")

    lam = st.slider("λ — indeterminacy control", 0.0, 1.0, 0.50, 0.05)
    eps = st.slider("ε — interval uncertainty width", 0.00, 0.25, 0.05, 0.01)
    window_size = st.selectbox("Neighborhood window size", [3, 5, 7], index=0)
    sigma = st.slider("Spatial weight σ", 0.3, 5.0, 1.0, 0.1)

    operator = st.radio(
        "Aggregation operator",
        ["IVPNSWA", "IVPNSWG"],
        captions=[
            "Weighted Averaging: stable smoothing",
            "Weighted Geometric: sensitive to local variation"
        ]
    )

    score_mode = st.radio(
        "Reconstruction mode",
        [
            "Normalized practical score",
            "Alpha-dominance score",
            "Direct clipped manuscript score"
        ],
        help="Normalized mode is recommended for stable visual reconstruction."
    )

    st.markdown("---")
    st.markdown("### 🧪 Noise Simulation")

    noise_type = st.selectbox(
        "Add artificial noise before processing?",
        ["Gaussian", "None", "Speckle", "Salt & Pepper"],
        index=0
    )

    gaussian_sigma = st.slider("Gaussian σ", 1, 60, 15, 1)
    speckle_sigma = st.slider("Speckle σ", 0.01, 0.50, 0.12, 0.01)
    salt_pepper_prob = st.slider("Salt & Pepper probability", 0.01, 0.30, 0.03, 0.01)

    compare_filters = st.checkbox("Compare with mean and median filters", value=True)

    st.markdown("---")
    st.caption("Recommended setting: Gaussian noise, λ=0.50, ε=0.05, 3×3 window, IVPNSWA.")


# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <div>
        <span class="hero-badge">Research Prototype</span>
        <span class="hero-badge">IVPNS</span>
        <span class="hero-badge">Image Denoising</span>
        <span class="hero-badge">Streamlit Demo</span>
    </div>
    <div class="hero-title">IVPNS-DenoiseLab</div>
    <div class="hero-subtitle">
        Intelligent neutrosophic image processing system using a pixel-level
        Interval-Valued Pythagorean Neutrosophic Aggregation Operator for
        uncertainty-aware noise reduction and visual quality enhancement.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# EMPTY STATE
# ============================================================
if uploaded is None:
    st.markdown('<div class="section-title">Start the Demonstration</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card">
            <div class="card-title">1. Upload Image</div>
            <div class="card-text">Upload PNG, JPG, BMP, TIF or TIFF image from the sidebar.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
            <div class="card-title">2. Tune IVPNS Parameters</div>
            <div class="card-text">Adjust λ, ε, aggregation operator and local window size.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card">
            <div class="card-title">3. Evaluate Result</div>
            <div class="card-text">View visual comparison, MSE, PSNR, SSIM and pixel-level details.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Method Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline">
        Image Input → Normalization → IVPNS Transformation → Interval Construction → Local Aggregation → Defuzzification → Reconstructed Output
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Research Highlights</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="research-box">
        ✔ Models pixel uncertainty using truth, indeterminacy and falsity components.<br>
        ✔ Uses interval-valued representation to capture imprecision in noisy regions.<br>
        ✔ Applies local IVPNS aggregation to suppress noise while preserving image structure.<br>
        ✔ Supports comparison with classical mean and median filters using MSE, PSNR and SSIM.
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ============================================================
# PROCESS IMAGE
# ============================================================
original_pil, original_255 = load_image_as_gray(uploaded)

np.random.seed(42)
input_255 = add_noise(
    original_255,
    noise_type=noise_type,
    gaussian_sigma=gaussian_sigma,
    speckle_sigma=speckle_sigma,
    salt_pepper_prob=salt_pepper_prob
)

start = time.time()
g = normalize_image(input_255)
components = ivpns_transform(g, lam=lam, eps=eps)
aggregated, kernel = aggregate_ivpns(
    components,
    operator=operator,
    window_size=window_size,
    sigma=sigma
)
raw_score, score_direct, score_norm, score_alpha = defuzzify_score(aggregated)

if score_mode == "Normalized practical score":
    score_used = score_norm
elif score_mode == "Alpha-dominance score":
    score_used = score_alpha
else:
    score_used = score_direct

output_255 = reconstruct_image(score_used)
elapsed = time.time() - start

mean_out = uniform_filter(input_255, size=window_size).astype(np.uint8)
median_out = median_filter(input_255, size=window_size).astype(np.uint8)

reference = original_255.astype(np.uint8)
input_uint8 = input_255.astype(np.uint8)

mse_in, psnr_in, ssim_in = compute_metrics(reference, input_uint8)
mse_ivpns, psnr_ivpns, ssim_ivpns = compute_metrics(reference, output_255)
mse_mean, psnr_mean, ssim_mean = compute_metrics(reference, mean_out)
mse_med, psnr_med, ssim_med = compute_metrics(reference, median_out)


# ============================================================
# KPI SUMMARY
# ============================================================
st.markdown('<div class="section-title">Executive Processing Summary</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Image Size", f"{original_255.shape[1]} × {original_255.shape[0]}")
k2.metric("Operator", operator)
k3.metric("Processing Time", f"{elapsed:.3f} s")
k4.metric("Noise Model", noise_type)

st.markdown(f"""
<div class="research-box">
    Processing completed using <b>{operator}</b> with window size <b>{window_size}×{window_size}</b>,
    λ=<b>{lam:.2f}</b>, ε=<b>{eps:.2f}</b> and σ=<b>{sigma:.2f}</b>.
    The system evaluates the output using <b>MSE</b>, <b>PSNR</b> and <b>SSIM</b>.
</div>
""", unsafe_allow_html=True)


# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🖼️ Results Dashboard",
    "📊 Metrics",
    "🧠 IVPNS Components",
    "⚙️ Step-by-Step Method",
    "🔍 Pixel Inspector",
    "📝 Paper Notes"
])


# ============================================================
# TAB 1: RESULTS DASHBOARD
# ============================================================
with tab1:
    st.markdown('<div class="section-title">Visual Result Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Direct comparison between the original image, degraded input and IVPNS reconstructed output.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(reference, caption="Original Reference", use_container_width=True)
    with c2:
        st.image(input_uint8, caption=f"Input Image ({noise_type})", use_container_width=True)
    with c3:
        st.image(output_255, caption=f"Proposed IVPNS Output ({score_mode})", use_container_width=True)

    st.markdown('<div class="section-title">Classical Filter Comparison</div>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        st.image(mean_out, caption=f"Mean Filter {window_size}×{window_size}", use_container_width=True)
    with c5:
        st.image(median_out, caption=f"Median Filter {window_size}×{window_size}", use_container_width=True)
    with c6:
        st.image(output_255, caption="Proposed IVPNS", use_container_width=True)

    d1, d2 = st.columns([1, 1])
    with d1:
        st.download_button(
            "⬇️ Download IVPNS Output",
            data=image_to_download_bytes(output_255),
            file_name="ivpns_denoised_output.png",
            mime="image/png"
        )
    with d2:
        comparison_png = create_comparison_canvas(
            [reference, input_uint8, mean_out, median_out, output_255],
            ["Original", "Input", "Mean", "Median", "IVPNS"]
        )
        buffer = io.BytesIO()
        comparison_png.save(buffer, format="PNG")
        st.download_button(
            "⬇️ Download Comparison Panel",
            data=buffer.getvalue(),
            file_name="ivpns_comparison_panel.png",
            mime="image/png"
        )

    st.markdown('<div class="section-title">Histogram Analysis</div>', unsafe_allow_html=True)
    h1, h2 = st.columns(2)
    with h1:
        st.pyplot(plot_histogram(input_255, "Input / Noisy Image Histogram"))
    with h2:
        st.pyplot(plot_histogram(output_255, "IVPNS Output Histogram"))


# ============================================================
# TAB 2: METRICS
# ============================================================
with tab2:
    st.markdown('<div class="section-title">Quantitative Performance Evaluation</div>', unsafe_allow_html=True)

    rows = [
        ["Noisy/Input Image", mse_in, psnr_in, ssim_in],
        [f"Mean Filter ({window_size}×{window_size})", mse_mean, psnr_mean, ssim_mean],
        [f"Median Filter ({window_size}×{window_size})", mse_med, psnr_med, ssim_med],
        ["Proposed IVPNS", mse_ivpns, psnr_ivpns, ssim_ivpns],
    ]

    metrics_df = pd.DataFrame(rows, columns=["Method", "MSE ↓", "PSNR (dB) ↑", "SSIM ↑"])
    metrics_df["Rank by SSIM"] = metrics_df["SSIM ↑"].rank(ascending=False, method="min").astype(int)
    metrics_df["Rank by MSE"] = metrics_df["MSE ↓"].rank(ascending=True, method="min").astype(int)

    st.dataframe(
        metrics_df.style.format({
            "MSE ↓": "{:.4f}",
            "PSNR (dB) ↑": "{:.4f}",
            "SSIM ↑": "{:.4f}"
        }),
        use_container_width=True
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Proposed IVPNS MSE", f"{mse_ivpns:.4f}", delta=f"{mse_in - mse_ivpns:.4f} vs input")
    m2.metric("Proposed IVPNS PSNR", f"{psnr_ivpns:.4f} dB", delta=f"{psnr_ivpns - psnr_in:.4f} dB vs input")
    m3.metric("Proposed IVPNS SSIM", f"{ssim_ivpns:.4f}", delta=f"{ssim_ivpns - ssim_in:.4f} vs input")

    st.markdown('<div class="section-title">Metric Interpretation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div class="card-text">
        <b>MSE</b> measures pixel-wise reconstruction error. Lower value is better.<br>
        <b>PSNR</b> measures reconstruction fidelity in decibel scale. Higher value is better.<br>
        <b>SSIM</b> measures structural similarity and perceptual quality. Higher value is better.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 3: IVPNS COMPONENTS
# ============================================================
with tab3:
    st.markdown('<div class="section-title">IVPNS Component Visualization</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="pipeline">
        g(x,y) → α truth map, β indeterminacy map, γ falsity map → interval-valued lower and upper bounds
    </div>
    """, unsafe_allow_html=True)

    sub1, sub2, sub3 = st.tabs(["Core Components", "Interval Bounds", "Aggregated Components"])

    with sub1:
        c1, c2, c3 = st.columns(3)
        c1.image((components["alpha"] * 255).astype(np.uint8), caption="Truth α Map", use_container_width=True)
        c2.image((components["beta"] * 255).astype(np.uint8), caption="Indeterminacy β Map", use_container_width=True)
        c3.image((components["gamma"] * 255).astype(np.uint8), caption="Falsity γ Map", use_container_width=True)

    with sub2:
        s1, s2, s3 = st.tabs(["α interval", "β interval", "γ interval"])
        with s1:
            c1, c2 = st.columns(2)
            c1.image((components["alpha_L"] * 255).astype(np.uint8), caption="α Lower Bound", use_container_width=True)
            c2.image((components["alpha_U"] * 255).astype(np.uint8), caption="α Upper Bound", use_container_width=True)
        with s2:
            c1, c2 = st.columns(2)
            c1.image((components["beta_L"] * 255).astype(np.uint8), caption="β Lower Bound", use_container_width=True)
            c2.image((components["beta_U"] * 255).astype(np.uint8), caption="β Upper Bound", use_container_width=True)
        with s3:
            c1, c2 = st.columns(2)
            c1.image((components["gamma_L"] * 255).astype(np.uint8), caption="γ Lower Bound", use_container_width=True)
            c2.image((components["gamma_U"] * 255).astype(np.uint8), caption="γ Upper Bound", use_container_width=True)

    with sub3:
        a1, a2, a3 = st.tabs(["Aggregated α", "Aggregated β", "Aggregated γ"])
        with a1:
            c1, c2 = st.columns(2)
            c1.image((aggregated["alpha_L"] * 255).astype(np.uint8), caption="Aggregated α Lower", use_container_width=True)
            c2.image((aggregated["alpha_U"] * 255).astype(np.uint8), caption="Aggregated α Upper", use_container_width=True)
        with a2:
            c1, c2 = st.columns(2)
            c1.image((aggregated["beta_L"] * 255).astype(np.uint8), caption="Aggregated β Lower", use_container_width=True)
            c2.image((aggregated["beta_U"] * 255).astype(np.uint8), caption="Aggregated β Upper", use_container_width=True)
        with a3:
            c1, c2 = st.columns(2)
            c1.image((aggregated["gamma_L"] * 255).astype(np.uint8), caption="Aggregated γ Lower", use_container_width=True)
            c2.image((aggregated["gamma_U"] * 255).astype(np.uint8), caption="Aggregated γ Upper", use_container_width=True)


# ============================================================
# TAB 4: STEP BY STEP METHOD
# ============================================================
with tab4:
    st.markdown('<div class="section-title">Proposed IVPNS Workflow</div>', unsafe_allow_html=True)

    with st.expander("Step 1 — Image Normalization", expanded=True):
        st.markdown('<div class="formula">g(x,y) = I(x,y) / 255</div>', unsafe_allow_html=True)
        st.dataframe(small_matrix_preview(g), use_container_width=True)

    with st.expander("Step 2 — IVPNS Representation", expanded=True):
        st.markdown('<div class="formula">α = g, &nbsp;&nbsp; γ = 1 − g, &nbsp;&nbsp; β = λ(1 − |2g − 1|)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <div class="card-text">
            The normalized pixel is converted into truth, indeterminacy and falsity components.
            This allows noisy pixels to be modeled with uncertainty instead of using only deterministic intensity values.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Step 3 — Interval-Valued Construction", expanded=True):
        st.markdown("""
        <div class="formula">
        αᴸ=max(0,α−ε), αᵁ=min(1,α+ε)<br>
        βᴸ=max(0,β−ε), βᵁ=min(1,β+ε)<br>
        γᴸ=max(0,γ−ε), γᵁ=min(1,γ+ε)
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Step 4 — Local IVPNS Aggregation", expanded=True):
        st.markdown('<div class="formula">wᵢ = (1/Z) exp(−dᵢ² / σ²)</div>', unsafe_allow_html=True)
        st.write("Spatial weight kernel:")
        st.dataframe(pd.DataFrame(np.round(kernel, 5)), use_container_width=True)

    with st.expander("Step 5 — Defuzzification and Reconstruction", expanded=True):
        st.markdown('<div class="formula">S = (αᴸ + αᵁ − βᴸ − βᵁ − γᴸ − γᵁ) / 3</div>', unsafe_allow_html=True)
        st.markdown('<div class="formula">I′(x,y) = 255 × S(x,y)</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.image(((np.clip(raw_score, -1, 1) + 1) / 2 * 255).astype(np.uint8), caption="Raw Score Visualization", use_container_width=True)
        c2.image((score_norm * 255).astype(np.uint8), caption="Normalized Score", use_container_width=True)
        c3.image(output_255, caption="Final Output", use_container_width=True)


# ============================================================
# TAB 5: PIXEL INSPECTOR
# ============================================================
with tab5:
    st.markdown('<div class="section-title">Pixel-Level IVPNS Inspection</div>', unsafe_allow_html=True)

    row = st.slider("Select row y", 0, input_255.shape[0] - 1, input_255.shape[0] // 2)
    col = st.slider("Select column x", 0, input_255.shape[1] - 1, input_255.shape[1] // 2)

    pixel_info = {
        "Pixel coordinate": f"({row}, {col})",
        "Original intensity": float(reference[row, col]),
        "Input intensity": float(input_255[row, col]),
        "Normalized g": float(g[row, col]),
        "Truth α": float(components["alpha"][row, col]),
        "Indeterminacy β": float(components["beta"][row, col]),
        "Falsity γ": float(components["gamma"][row, col]),
        "α interval": f"[{components['alpha_L'][row, col]:.4f}, {components['alpha_U'][row, col]:.4f}]",
        "β interval": f"[{components['beta_L'][row, col]:.4f}, {components['beta_U'][row, col]:.4f}]",
        "γ interval": f"[{components['gamma_L'][row, col]:.4f}, {components['gamma_U'][row, col]:.4f}]",
        "Raw score": float(raw_score[row, col]),
        "Output intensity": int(output_255[row, col])
    }

    st.table(pd.DataFrame(pixel_info.items(), columns=["Item", "Value"]))


# ============================================================
# TAB 6: PAPER NOTES
# ============================================================
with tab6:
    st.markdown('<div class="section-title">Implementation Notes for Manuscript</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">Suggested paper description</div>
        <div class="card-text">
        A Streamlit-based prototype was developed to demonstrate the practical applicability of the proposed
        IVPNS image denoising framework. The system allows users to upload grayscale images, simulate common
        noise models, configure IVPNS parameters and visualize each processing stage, including normalization,
        IVPNS transformation, interval construction, local aggregation, defuzzification and image reconstruction.
        Quantitative evaluation is performed using MSE, PSNR and SSIM, while visual comparison is provided against
        classical mean and median filtering methods.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="note-box">
        <b>Important technical note:</b><br>
        The direct manuscript score function may produce negative values in dark regions.
        Therefore, this app provides a normalized practical score mode to make reconstruction stable in [0,1].
        For final journal submission, the score function should be carefully defined so the output range is mathematically bounded.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Recommended GitHub Repository Name</div>', unsafe_allow_html=True)
    st.code("IVPNS-DenoiseLab", language="text")

    st.markdown('<div class="section-title">Recommended Citation Label</div>', unsafe_allow_html=True)
    st.code(
        "A Pixel-Level Interval-Valued Pythagorean Neutrosophic Aggregation Operator for Image Noise Reduction",
        language="text"
    )
'''

path = Path("/mnt/data/app_q1_premium.py")
path.write_text(code, encoding="utf-8")
print(f"Created: {path}")
print(f"Size: {path.stat().st_size/1024:.1f} KB")
