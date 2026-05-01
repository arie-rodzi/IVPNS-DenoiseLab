# # 🧠 IVPNS-DenoiseLab

A Streamlit-based implementation of an advanced image denoising framework using **Interval-Valued Pythagorean Neutrosophic Sets (IVPNS)**.

This project demonstrates a **pixel-level uncertainty modeling approach** for noise reduction, integrating neutrosophic theory with aggregation operators to preserve image structures while suppressing noise.

---

## 📌 Key Features

- 📷 Upload and process grayscale images  
- 🔄 Full IVPNS transformation (α, β, γ components)  
- 📊 Interval-valued uncertainty representation  
- ⚙️ Local aggregation:
  - IVPNS Weighted Averaging (IVPNSWA)
  - IVPNS Weighted Geometric (IVPNSWG)
- 📉 Noise reduction with edge preservation  
- 📈 Performance metrics:
  - MSE
  - PSNR
  - SSIM  
- 🧪 Step-by-step visualization  

---

## 🧠 Proposed Method

### 1. Normalization
g(x,y) = I(x,y) / 255  

### 2. IVPNS Transformation
α = g(x,y)  
γ = 1 − g(x,y)  
β = λ(1 − |2g − 1|)  

### 3. Interval Representation
α ∈ [α−ε, α+ε]  
β ∈ [β−ε, β+ε]  
γ ∈ [γ−ε, γ+ε]  

### 4. Aggregation
Apply IVPNS operator within neighborhood window  

### 5. Defuzzification
S = (αL + αU − βL − βU − γL − γU) / 3  

### 6. Reconstruction
I'(x,y) = 255 × S(x,y)  

---

## 🚀 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
1. **Normalization**
