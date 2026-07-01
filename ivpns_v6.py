"""
IVPNS-DenoiseLab — IVPNS image denoising (v6, manuscript-consistent).

This implementation is faithful to the equations in the manuscript:
  Step 1  Normalize to [0,1]
  Step 2  IVPNS representation:
            alpha = g
            gamma = 1 - g
            beta  = lambda * phi(local_variance)      (variance-driven indeterminacy)
          interval [+/- eps], with Pythagorean projection on alpha^U (Lemma 4.1)
  Step 3  Local aggregation (IVPNSWA / IVPNSWG) with Gaussian-AND-intensity
          (bilateral) spatial weights
  Step 4  Uncertainty-gated truth-dominance adjustment (delta1,delta2,delta3)
  Step 5  Consistency-preserving truth-dominant score (eta, kappa)
  Step 6  Adaptive refinement  S' = w*S + (1-w)*g,  w = 1 - rho*U
  Step 7  Reconstruct  I' = 255 * S'

The denoising is performed by the IVPNS pipeline itself (no median/mean
post-filter); the aggregation in Step 3 is the spatial denoiser.

Author: arie-rodzi/IVPNS-DenoiseLab — BSD 3-Clause
"""
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
try:
    from skimage.restoration import estimate_sigma
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


# ----------------------------------------------------------------------
def normalize_image(img_255):
    return np.clip(img_255.astype(float) / 255.0, 0.0, 1.0)

def reconstruct_image(score_01):
    return np.clip(255.0 * score_01, 0, 255).astype(np.uint8)

def _estimate_noise(g):
    if _HAVE_SK:
        try:
            return float(np.mean(estimate_sigma(g)))
        except Exception:
            pass
    # fallback: MAD of Laplacian
    lap = g - uniform_filter(g, 3)
    return float(1.4826 * np.median(np.abs(lap - np.median(lap))) + 1e-6)


# ----------------------------------------------------------------------
def ivpns_representation(g, lam=0.4, eps=0.05, var_sigma=1.0):
    """Step 2: variance-driven IVPNS encoding with Pythagorean projection."""
    g = np.clip(g, 0, 1)
    # local variance from a lightly smoothed image so noise does not fake structure
    gs = gaussian_filter(g, sigma=var_sigma)
    m  = uniform_filter(gs, 3)
    m2 = uniform_filter(gs * gs, 3)
    v  = np.maximum(0.0, m2 - m * m)
    v  = v / (v.max() + 1e-9)
    phi = np.sqrt(v)                       # phi(.) maps variance -> [0,1]

    alpha = g
    gamma = 1.0 - g
    beta  = lam * np.clip(phi, 0, 1)       # HIGH at edges/texture, LOW in flat areas

    aL = np.clip(alpha - eps, 0, 1); aU = np.clip(alpha + eps, 0, 1)
    bL = np.clip(beta  - eps, 0, 1); bU = np.clip(beta  + eps, 0, 1)
    cL = np.clip(gamma - eps, 0, 1); cU = np.clip(gamma + eps, 0, 1)
    # Pythagorean projection (Lemma 4.1): (alpha^U)^2 + (gamma^U)^2 <= 1
    aU = np.minimum(aU, np.sqrt(np.maximum(0.0, 1.0 - cU ** 2)))
    aL = np.minimum(aL, aU)
    return dict(aL=aL, aU=aU, bL=bL, bU=bU, cL=cL, cU=cU)


def aggregate_ivpns(comp, g, operator="IVPNSWG", k=5, sigma_s=2.5, sigma_r=0.4):
    """Step 3: bilateral-weighted IVPNS aggregation."""
    H, W = g.shape
    r = k // 2
    def neigh(a):
        P = np.pad(a, r, mode="reflect")
        return [P[r+dy:r+dy+H, r+dx:r+dx+W]
                for dy in range(-r, r+1) for dx in range(-r, r+1)]
    NB = neigh(g)
    NaL, NaU = neigh(comp["aL"]), neigh(comp["aU"])
    NbL, NbU = neigh(comp["bL"]), neigh(comp["bU"])
    NcL, NcU = neigh(comp["cL"]), neigh(comp["cU"])
    ys, xs = np.mgrid[-r:r+1, -r:r+1]
    wsp = np.exp(-(xs**2 + ys**2) / (sigma_s**2)).reshape(-1)

    wsum = np.zeros((H, W))
    # accumulators
    arith = {key: np.zeros((H, W)) for key in
             ("aL","aU","bL","bU","cL","cU")}
    geolog = {key: np.zeros((H, W)) for key in
              ("aL","aU","bL","bU","cL","cU")}
    packs = dict(aL=NaL, aU=NaU, bL=NbL, bU=NbU, cL=NcL, cU=NcU)
    for idx in range(k*k):
        wr = np.exp(-((NB[idx] - g) ** 2) / (sigma_r ** 2))
        w  = wsp[idx] * wr
        wsum += w
        for key in packs:
            s = np.clip(packs[key][idx], 1e-12, 1.0)
            arith[key]  += w * s
            geolog[key] += w * np.log(s)
    A = {key: arith[key] / wsum for key in arith}
    G = {key: np.exp(geolog[key] / wsum) for key in geolog}

    out = {}
    if operator.upper() == "IVPNSWA":
        # truth arithmetic; indeterminacy & falsity geometric
        out["aL"], out["aU"] = A["aL"], A["aU"]
        out["bL"], out["bU"] = G["bL"], G["bU"]
        out["cL"], out["cU"] = G["cL"], G["cU"]
    else:  # IVPNSWG
        out["aL"], out["aU"] = G["aL"], G["aU"]
        out["bL"], out["bU"] = A["bL"], A["bU"]
        out["cL"], out["cU"] = A["cL"], A["cU"]
    return out


def truth_dominance(agg, d1=0.2, d2=0.3, d3=0.3):
    """Step 4: uncertainty-gated adjustment (gate = pre-adjustment indeterminacy)."""
    gate = 0.5 * (agg["bL"] + agg["bU"])
    out = dict(agg)
    out["aL"] = np.minimum(1.0, agg["aL"] * (1 + d1 * gate))
    out["aU"] = np.minimum(1.0, agg["aU"] * (1 + d1 * gate))
    out["bL"] = agg["bL"] * (1 - d2 * gate)
    out["bU"] = agg["bU"] * (1 - d2 * gate)
    out["cL"] = agg["cL"] * (1 - d3 * gate)
    out["cU"] = agg["cU"] * (1 - d3 * gate)
    return out


def ivpns_score(agg, eta=0.2, kappa=0.3):
    """Step 5: consistency-preserving truth-dominant score."""
    a = 0.5 * (agg["aL"] + agg["aU"])
    b = 0.5 * (agg["bL"] + agg["bU"])
    c = 0.5 * (agg["cL"] + agg["cU"])
    T = 0.5 * ((1 + eta) * a + (1 - eta) * (1 - c))
    S = (1 - kappa * b) * T + kappa * b * a
    return np.clip(S, 0, 1), a, b, c


def adaptive_refinement(g, S, b, c, rho=0.9):
    """Step 6: uncertainty-driven blend of score and observation."""
    U = 0.5 * (b + c)
    w = 1.0 - rho * U
    return np.clip(w * S + (1 - w) * g, 0, 1), w


# ----------------------------------------------------------------------
def process_ivpns(input_255, operator="IVPNSWG", lam=0.4, eps=0.05,
                  d1=0.2, d2=0.3, d3=0.3, eta=0.2, kappa=0.3, rho=0.9,
                  noise_adaptive=True,
                  use_truth=True, use_aggregation=True, use_refinement=True):
    """Full IVPNS denoising pipeline. Returns (output_uint8, details)."""
    g = normalize_image(input_255)

    # noise-adaptive window / range (kept simple and documented)
    sig = _estimate_noise(g) if noise_adaptive else 0.08
    k = 5 if sig < 0.06 else (7 if sig < 0.12 else 9)
    sigma_s = k / 2.0
    sigma_r = max(0.4, 3.0 * sig)
    var_sigma = 1.0 + 4.0 * sig

    comp = ivpns_representation(g, lam=lam, eps=eps, var_sigma=var_sigma)

    if use_aggregation:
        agg = aggregate_ivpns(comp, g, operator=operator, k=k,
                              sigma_s=sigma_s, sigma_r=sigma_r)
    else:
        agg = dict(comp)  # no neighbourhood fusion (ablation)

    if use_truth:
        agg = truth_dominance(agg, d1=d1, d2=d2, d3=d3)

    S, a, b, cc = ivpns_score(agg, eta=eta, kappa=kappa)

    if use_refinement:
        Sp, omega = adaptive_refinement(g, S, b, cc, rho=rho)
    else:
        Sp, omega = S, np.ones_like(S)

    out = reconstruct_image(Sp)
    return out, dict(g=g, alpha=a, beta=b, gamma=cc, score=S, omega=omega)


# convenience wrapper returning a float image in [0,1]
def ivpns_denoise(g01, operator="IVPNSWG", **kw):
    out, _ = process_ivpns(np.clip(g01, 0, 1) * 255.0, operator=operator, **kw)
    return out.astype(float) / 255.0
