
# ============================================================
# Streamlit App: IVPNS Image Denoising Framework
# Proposed Method:
# Interval-Valued Pythagorean Neutrosophic Aggregation Operator
# for Image Noise Reduction
#
# Run:
#   streamlit run streamlit_app.py
#
# Required packages:
#   pip install streamlit numpy pillow scipy scikit-image pandas matplotlib
# ============================================================

import io
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.ndimage import uniform_filter, median_filter, convolve
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="IVPNS Image Denoising",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #123C69;
    margin-bottom: 0px;
}
.sub-title {
    font-size: 16px;
    color: #555;
    margin-top: 0px;
}
.metric-card {
    background: #F6F8FB;
    padding: 16px;
    border-radius: 14px;
    border: 1px solid #E5E7EB;
}
.step-box {
    background: #FFFFFF;
    padding: 16px;
    border-radius: 14px;
    border-left: 6px solid #1F77B4;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    margin-bottom: 14px;
}
.formula {
    background: #F4F7FC;
    padding: 10px 14px;
    border-radius: 10px;
    font-family: monospace;
    color: #123C69;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">IVPNS-Based Image Denoising System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Pixel-level Interval-Valued Pythagorean Neutrosophic Aggregation Operator for image noise reduction</div>',
    unsafe_allow_html=True
)

st.divider()


# -----------------------------
# Helper functions
# -----------------------------
def load_image_as_gray(uploaded_file):
    """Load uploaded image and convert to grayscale float image in [0, 255]."""
    image = Image.open(uploaded_file).convert("L")
    arr = np.array(image).astype(np.float64)
    return image, arr


def normalize_image(img_255):
    """Step 1: Normalize image into [0,1]."""
    return np.clip(img_255 / 255.0, 0, 1)


def ivpns_transform(g, lam=0.5, eps=0.05):
    """
    Step 2: Convert normalized image into IVPNS components.

    alpha = g
    gamma = 1 - g
    beta = lambda * (1 - |2g - 1|)

    Interval:
    alpha_L = max(0, alpha - eps), alpha_U = min(1, alpha + eps)
    beta_L  = max(0, beta  - eps), beta_U  = min(1, beta  + eps)
    gamma_L = max(0, gamma - eps), gamma_U = min(1, gamma + eps)
    """
    alpha = g
    gamma = 1.0 - g
    beta = lam * (1.0 - np.abs(2.0 * g - 1.0))

    alpha_L = np.clip(alpha - eps, 0, 1)
    alpha_U = np.clip(alpha + eps, 0, 1)
    beta_L = np.clip(beta - eps, 0, 1)
    beta_U = np.clip(beta + eps, 0, 1)
    gamma_L = np.clip(gamma - eps, 0, 1)
    gamma_U = np.clip(gamma + eps, 0, 1)

    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "alpha_L": alpha_L,
        "alpha_U": alpha_U,
        "beta_L": beta_L,
        "beta_U": beta_U,
        "gamma_L": gamma_L,
        "gamma_U": gamma_U,
    }


def spatial_kernel(window_size=3, sigma=1.0):
    """Create normalized spatial Gaussian-like weight kernel."""
    r = window_size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    d2 = x**2 + y**2
    kernel = np.exp(-d2 / (sigma**2 + 1e-12))
    kernel = kernel / np.sum(kernel)
    return kernel


def aggregate_ivpns(components, operator="IVPNSWA", window_size=3, sigma=1.0):
    """
    Step 3: Local IVPNS aggregation.

    IVPNSWA:
        alpha = weighted arithmetic mean
        beta  = weighted geometric mean
        gamma = weighted geometric mean

    IVPNSWG:
        alpha = weighted geometric mean
        beta  = weighted arithmetic mean
        gamma = weighted arithmetic mean
    """
    kernel = spatial_kernel(window_size, sigma)

    def weighted_mean(x):
        return convolve(x, kernel, mode="reflect")

    def weighted_geo(x):
        # product(x_i ^ w_i) = exp(sum(w_i log(x_i)))
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

    # Keep numerical values valid
    for k in out:
        out[k] = np.clip(out[k], 0, 1)

    return out, kernel


def defuzzify_score(agg):
    """
    Step 4: Defuzzification.

    The manuscript states:
        S = (alpha_L + alpha_U - beta_L - beta_U - gamma_L - gamma_U) / 3

    However, this can produce negative values for dark pixels.
    For practical reconstruction, we also provide a normalized score:
        S_norm = clip((S + 2) / 2, 0, 1)
    because theoretical S may lie approximately in [-2/3, 2/3].

    The app lets user choose either direct clipped score or normalized score.
    """
    raw_score = (
        agg["alpha_L"] + agg["alpha_U"]
        - agg["beta_L"] - agg["beta_U"]
        - agg["gamma_L"] - agg["gamma_U"]
    ) / 3.0

    direct_clipped = np.clip(raw_score, 0, 1)
    normalized = np.clip((raw_score + 2.0 / 3.0) / (4.0 / 3.0), 0, 1)

    return raw_score, direct_clipped, normalized


def reconstruct_image(score_01):
    """Step 5: Convert [0,1] score to [0,255]."""
    return np.clip(255.0 * score_01, 0, 255).astype(np.uint8)


def add_noise(img_255, noise_type="None", gaussian_sigma=15, speckle_sigma=0.12, salt_pepper_prob=0.03):
    """Optional demonstration noise generator."""
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
    """Compute MSE, PSNR and SSIM against reference image."""
    ref = reference.astype(np.float64)
    pro = processed.astype(np.float64)

    mse = mean_squared_error(ref, pro)
    psnr = peak_signal_noise_ratio(ref, pro, data_range=255)
    ssim = structural_similarity(ref, pro, data_range=255)

    return mse, psnr, ssim


def image_to_download_bytes(arr_uint8):
    """Convert numpy image to PNG bytes."""
    img = Image.fromarray(arr_uint8)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def plot_histogram(image_255, title):
    """Return matplotlib histogram figure."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(image_255.flatten(), bins=50)
    ax.set_title(title)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25)
    return fig


def small_matrix_preview(arr, size=8):
    """Display small top-left matrix values."""
    return pd.DataFrame(np.round(arr[:size, :size], 4))


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Processing Settings")

    uploaded = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )

    st.subheader("IVPNS Parameters")
    lam = st.slider("λ: indeterminacy control", 0.0, 1.0, 0.50, 0.05)
    eps = st.slider("ε: interval uncertainty width", 0.00, 0.25, 0.05, 0.01)
    window_size = st.selectbox("Neighborhood window size", [3, 5, 7], index=0)
    sigma = st.slider("Spatial weight σ", 0.3, 5.0, 1.0, 0.1)

    operator = st.radio(
        "Aggregation operator",
        ["IVPNSWA", "IVPNSWG"],
        captions=[
            "Weighted Averaging: stable smoothing",
            "Weighted Geometric: more sensitive to local variation"
        ]
    )

    score_mode = st.radio(
        "Reconstruction score mode",
        ["Normalized practical score", "Direct clipped manuscript score"],
        help="The direct manuscript score can become negative. Normalized practical score is safer for visual reconstruction."
    )

    st.subheader("Optional Noise Simulation")
    noise_type = st.selectbox("Add artificial noise before processing?", ["None", "Gaussian", "Speckle", "Salt & Pepper"])
    gaussian_sigma = st.slider("Gaussian σ", 1, 60, 15, 1)
    speckle_sigma = st.slider("Speckle σ", 0.01, 0.50, 0.12, 0.01)
    salt_pepper_prob = st.slider("Salt & Pepper probability", 0.01, 0.30, 0.03, 0.01)

    compare_filters = st.checkbox("Compare with mean and median filters", value=True)


# -----------------------------
# Main app
# -----------------------------
if uploaded is None:
    st.info("Upload one image from the sidebar to start the IVPNS denoising workflow.")
    st.markdown("""
    ### What this app will show

    1. **Input image**
    2. **Normalization** into `[0,1]`
    3. **IVPNS transformation** into truth, indeterminacy and falsity components
    4. **Interval bounds** for α, β and γ
    5. **Local IVPNS aggregation**
    6. **Defuzzification score**
    7. **Reconstructed denoised image**
    8. **MSE, PSNR and SSIM metrics**
    """)
    st.stop()


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

# Step 1
g = normalize_image(input_255)

# Step 2
components = ivpns_transform(g, lam=lam, eps=eps)

# Step 3
aggregated, kernel = aggregate_ivpns(
    components,
    operator=operator,
    window_size=window_size,
    sigma=sigma
)

# Step 4
raw_score, score_direct, score_norm = defuzzify_score(aggregated)
score_used = score_norm if score_mode == "Normalized practical score" else score_direct

# Step 5
output_255 = reconstruct_image(score_used)

elapsed = time.time() - start

# Baseline filters
mean_out = uniform_filter(input_255, size=window_size).astype(np.uint8)
median_out = median_filter(input_255, size=window_size).astype(np.uint8)


# -----------------------------
# Summary
# -----------------------------
st.success(f"Processing completed in {elapsed:.3f} seconds using {operator}, window={window_size}×{window_size}, λ={lam}, ε={eps}.")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Image size", f"{original_255.shape[1]} × {original_255.shape[0]}")
with c2:
    st.metric("Operator", operator)
with c3:
    st.metric("Processing time", f"{elapsed:.3f} s")


# -----------------------------
# Visual comparison
# -----------------------------
st.header("1. Image Input and Output")

col1, col2, col3 = st.columns(3)
with col1:
    st.image(original_255.astype(np.uint8), caption="Original image", use_container_width=True)
with col2:
    st.image(input_255.astype(np.uint8), caption=f"Input image {'' if noise_type == 'None' else '(' + noise_type + ' noise)'}", use_container_width=True)
with col3:
    st.image(output_255, caption="Proposed IVPNS output", use_container_width=True)

st.download_button(
    "⬇️ Download IVPNS processed image",
    data=image_to_download_bytes(output_255),
    file_name="ivpns_denoised_output.png",
    mime="image/png"
)


# -----------------------------
# Metrics
# -----------------------------
st.header("2. Quantitative Evaluation")

reference = original_255.astype(np.uint8)

metrics_rows = []

mse_in, psnr_in, ssim_in = compute_metrics(reference, input_255.astype(np.uint8))
metrics_rows.append(["Noisy/Input Image", mse_in, psnr_in, ssim_in])

if compare_filters:
    mse_mean, psnr_mean, ssim_mean = compute_metrics(reference, mean_out)
    mse_med, psnr_med, ssim_med = compute_metrics(reference, median_out)
    metrics_rows.append([f"Mean Filter ({window_size}×{window_size})", mse_mean, psnr_mean, ssim_mean])
    metrics_rows.append([f"Median Filter ({window_size}×{window_size})", mse_med, psnr_med, ssim_med])

mse_ivpns, psnr_ivpns, ssim_ivpns = compute_metrics(reference, output_255)
metrics_rows.append(["Proposed IVPNS", mse_ivpns, psnr_ivpns, ssim_ivpns])

metrics_df = pd.DataFrame(metrics_rows, columns=["Method", "MSE", "PSNR (dB)", "SSIM"])
st.dataframe(metrics_df.style.format({"MSE": "{:.4f}", "PSNR (dB)": "{:.4f}", "SSIM": "{:.4f}"}), use_container_width=True)

m1, m2, m3 = st.columns(3)
m1.metric("IVPNS MSE", f"{mse_ivpns:.4f}")
m2.metric("IVPNS PSNR", f"{psnr_ivpns:.4f} dB")
m3.metric("IVPNS SSIM", f"{ssim_ivpns:.4f}")


# -----------------------------
# Step-by-step explanation
# -----------------------------
st.header("3. Proposed IVPNS Workflow: Step-by-Step Details")

with st.expander("Step 1: Image normalization", expanded=True):
    st.markdown("""
    <div class="step-box">
    The uploaded image is converted into grayscale and normalized from the original pixel range [0,255] into [0,1].
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula">g(x,y) = I(x,y) / 255</div>', unsafe_allow_html=True)
    st.write("Preview of normalized pixel values:")
    st.dataframe(small_matrix_preview(g), use_container_width=True)

with st.expander("Step 2: IVPNS representation of pixels", expanded=True):
    st.markdown("""
    <div class="step-box">
    Each pixel is represented by three neutrosophic components: truth α, indeterminacy β and falsity γ.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula">α = g, &nbsp;&nbsp; γ = 1 − g, &nbsp;&nbsp; β = λ(1 − |2g − 1|)</div>', unsafe_allow_html=True)

    a, b, c = st.columns(3)
    with a:
        st.image((components["alpha"] * 255).astype(np.uint8), caption="Truth α map", use_container_width=True)
    with b:
        st.image((components["beta"] * 255).astype(np.uint8), caption="Indeterminacy β map", use_container_width=True)
    with c:
        st.image((components["gamma"] * 255).astype(np.uint8), caption="Falsity γ map", use_container_width=True)

with st.expander("Step 3: Interval-valued bounds", expanded=False):
    st.markdown("""
    <div class="step-box">
    The crisp α, β and γ values are expanded into lower and upper interval bounds using ε.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="formula">
    αᴸ=max(0,α−ε), αᵁ=min(1,α+ε)<br>
    βᴸ=max(0,β−ε), βᵁ=min(1,β+ε)<br>
    γᴸ=max(0,γ−ε), γᵁ=min(1,γ+ε)
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["α interval", "β interval", "γ interval"])
    with tabs[0]:
        col_l, col_u = st.columns(2)
        col_l.image((components["alpha_L"] * 255).astype(np.uint8), caption="α lower bound", use_container_width=True)
        col_u.image((components["alpha_U"] * 255).astype(np.uint8), caption="α upper bound", use_container_width=True)
    with tabs[1]:
        col_l, col_u = st.columns(2)
        col_l.image((components["beta_L"] * 255).astype(np.uint8), caption="β lower bound", use_container_width=True)
        col_u.image((components["beta_U"] * 255).astype(np.uint8), caption="β upper bound", use_container_width=True)
    with tabs[2]:
        col_l, col_u = st.columns(2)
        col_l.image((components["gamma_L"] * 255).astype(np.uint8), caption="γ lower bound", use_container_width=True)
        col_u.image((components["gamma_U"] * 255).astype(np.uint8), caption="γ upper bound", use_container_width=True)

with st.expander("Step 4: Local IVPNS aggregation", expanded=True):
    st.markdown("""
    <div class="step-box">
    A local neighborhood window is used around each pixel. Spatial weights are computed using distance from the centre pixel.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula">wᵢ = (1/Z) exp(−dᵢ² / σ²)</div>', unsafe_allow_html=True)

    st.write("Spatial weight kernel used in local aggregation:")
    st.dataframe(pd.DataFrame(np.round(kernel, 5)), use_container_width=True)

    tabs = st.tabs(["Aggregated α", "Aggregated β", "Aggregated γ"])
    with tabs[0]:
        col_l, col_u = st.columns(2)
        col_l.image((aggregated["alpha_L"] * 255).astype(np.uint8), caption="Aggregated α lower", use_container_width=True)
        col_u.image((aggregated["alpha_U"] * 255).astype(np.uint8), caption="Aggregated α upper", use_container_width=True)
    with tabs[1]:
        col_l, col_u = st.columns(2)
        col_l.image((aggregated["beta_L"] * 255).astype(np.uint8), caption="Aggregated β lower", use_container_width=True)
        col_u.image((aggregated["beta_U"] * 255).astype(np.uint8), caption="Aggregated β upper", use_container_width=True)
    with tabs[2]:
        col_l, col_u = st.columns(2)
        col_l.image((aggregated["gamma_L"] * 255).astype(np.uint8), caption="Aggregated γ lower", use_container_width=True)
        col_u.image((aggregated["gamma_U"] * 255).astype(np.uint8), caption="Aggregated γ upper", use_container_width=True)

with st.expander("Step 5: Defuzzification and reconstruction", expanded=True):
    st.markdown("""
    <div class="step-box">
    The aggregated IVPNS values are converted back into a crisp score, then reconstructed into the image intensity scale.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="formula">S = (αᴸ + αᵁ − βᴸ − βᵁ − γᴸ − γᵁ) / 3</div>', unsafe_allow_html=True)
    st.markdown('<div class="formula">I′(x,y) = 255 × S(x,y)</div>', unsafe_allow_html=True)

    col_raw, col_used = st.columns(2)
    with col_raw:
        st.image(((np.clip(raw_score, -1, 1) + 1) / 2 * 255).astype(np.uint8), caption="Raw score visualization", use_container_width=True)
    with col_used:
        st.image(output_255, caption=f"Reconstructed image using: {score_mode}", use_container_width=True)


# -----------------------------
# Baseline comparison
# -----------------------------
if compare_filters:
    st.header("4. Visual Comparison with Classical Filters")
    c1, c2, c3 = st.columns(3)
    c1.image(mean_out, caption=f"Mean Filter {window_size}×{window_size}", use_container_width=True)
    c2.image(median_out, caption=f"Median Filter {window_size}×{window_size}", use_container_width=True)
    c3.image(output_255, caption="Proposed IVPNS", use_container_width=True)


# -----------------------------
# Histogram analysis
# -----------------------------
st.header("5. Histogram Analysis")
h1, h2 = st.columns(2)
with h1:
    st.pyplot(plot_histogram(input_255, "Input / Noisy Image Histogram"))
with h2:
    st.pyplot(plot_histogram(output_255, "IVPNS Output Histogram"))


# -----------------------------
# Pixel-level inspection
# -----------------------------
st.header("6. Pixel-Level Inspection")

row = st.slider("Select row y", 0, input_255.shape[0] - 1, input_255.shape[0] // 2)
col = st.slider("Select column x", 0, input_255.shape[1] - 1, input_255.shape[1] // 2)

pixel_info = {
    "Pixel coordinate": f"({row}, {col})",
    "Input intensity": float(input_255[row, col]),
    "Normalized g": float(g[row, col]),
    "α": float(components["alpha"][row, col]),
    "β": float(components["beta"][row, col]),
    "γ": float(components["gamma"][row, col]),
    "α interval": f"[{components['alpha_L'][row, col]:.4f}, {components['alpha_U'][row, col]:.4f}]",
    "β interval": f"[{components['beta_L'][row, col]:.4f}, {components['beta_U'][row, col]:.4f}]",
    "γ interval": f"[{components['gamma_L'][row, col]:.4f}, {components['gamma_U'][row, col]:.4f}]",
    "Raw score": float(raw_score[row, col]),
    "Output intensity": int(output_255[row, col])
}

st.table(pd.DataFrame(pixel_info.items(), columns=["Item", "Value"]))


# -----------------------------
# Notes
# -----------------------------
st.header("7. Important Notes for Paper Demonstration")
st.warning(
    "The manuscript score function can produce negative values for dark regions. "
    "This app includes a normalized practical score mode for better visual reconstruction. "
    "For journal submission, consider revising the score function so the reconstructed value is naturally bounded in [0,1]."
)

st.markdown("""
### Suggested explanation for your paper

This Streamlit application demonstrates the proposed IVPNS denoising framework in an interactive environment.
Users can upload an image, configure λ, ε, window size and aggregation operator, then observe the complete processing workflow from normalization to IVPNS representation, interval construction, local aggregation, defuzzification and reconstruction.
The application also reports MSE, PSNR and SSIM to support quantitative comparison with classical mean and median filters.
""")
