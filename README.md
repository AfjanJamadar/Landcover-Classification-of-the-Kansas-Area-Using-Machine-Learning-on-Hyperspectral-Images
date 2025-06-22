# 🛰️ Landcover Classification of the Kansas Area Using Hyperspectral Images  


This project presents a complete deep learning pipeline for landcover classification using hyperspectral satellite imagery of the Kansas area. We use a combination of dimensionality reduction techniques and multiple Convolutional Neural Networks (CNNs) to compare the impact of spectral vs spatial features on classification performance.

---

## 📌 Objective

To build an end-to-end system that:
- Preprocesses raw hyperspectral satellite data using normalization and PCA
- Trains three different CNN models (1D, 2D, 3D)
- Compares model performance across architectures
- Visualizes the results in an interactive Streamlit dashboard

---

## 🗂️ Dataset

- **Source:** GF-5 Satellite (Kansas region)
- **Size:** 650×340 pixels  
- **Spectral Bands:** 308  
- **Classes:** 7 (urban, crops, water, roads, soil, meadows, residential)  
- **Format:** ENVI hyperspectral data  

---

## 🧠 Methodology

### 🔹 Preprocessing
- **Normalization**: Standardized each band to zero mean and unit variance.
- **Dimensionality Reduction**: PCA to compress 308 bands → 30 components (retaining >90% variance).

### 🔹 Models

- **1D CNN (Spectral-only)**  
  Input: PCA vector (30×1) per pixel  
  → Accuracy: ~95.7%
  ![Screenshot 2025-06-20 185633](https://github.com/user-attachments/assets/a8f4e8fa-5ddb-4b2c-ad3f-448fc0b89ba2)


- **2D CNN (Spatial + Spectral)**  
  Input: 5×5×30 patch  
  → Accuracy: ~98.6%
  ![Screenshot 2025-06-20 185803](https://github.com/user-attachments/assets/4db0dece-c4a1-4642-994b-0aed60010e6c)


- **3D CNN (Volumetric)**  
  Input: 5×5×30 cube  
  → Accuracy: ~98.5%
  ![Screenshot 2025-06-20 185853](https://github.com/user-attachments/assets/efb6eb77-04aa-44c6-841c-30be23319ce1)


Each CNN is trained using cross-entropy loss with accuracy, F1-score, and confusion matrix tracking.

---

## 📈 Results Summary

| Model  | Accuracy | Strengths |
|--------|----------|-----------|
| 1D CNN | 95.7%    | Fast, lightweight, no spatial awareness |
| 2D CNN | 98.6%    | Best tradeoff of accuracy + compute |
| 3D CNN | 98.5%    | Strong generalization, smoother maps |

- PCA was critical for reducing training time and preventing overfitting.  
- Adding spatial context (2D/3D) drastically improved classification smoothness.  
- 3D CNN showed slightly lower loss but required much higher computation.

---

## 🖼️ Visualizations

The integrated dashboard provides:

- PCA Band Visualizations  
- Confusion Matrices  
- Training Accuracy/Loss Curves  
- Ground Truth vs Predicted Maps  
- Class-Wise Metrics  
- Model Comparison Charts  

> ⚡ Built using [Streamlit](https://streamlit.io/) for real-time interactivity.

---

## 🛠️ Tech Stack

- Python, NumPy, Pandas  
- Spectral Python (SPy)  
- TensorFlow / Keras  
- Scikit-learn (PCA, metrics)  
- Streamlit (Dashboard UI)  
- Matplotlib / Seaborn (Plots)

---

## 🚀 How to Run

```bash
# Clone the repo
git clone
cd hyperspectral-landcover

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

> Make sure to place the Kansas AHSI dataset inside a `/data` directory in ENVI format before running.

---

## 👤 Authors

**Afjan Jamadar**  


---

## 📌 Key Takeaways

- Spatial context boosts landcover classification accuracy by >2%  
- PCA is effective for reducing HSI dimensionality with minimal loss  
- Streamlit dashboards help make research results more explainable and user-friendly  
- 2D CNN offers the best balance of performance and speed for practical use cases

---

## 📚 References

- Jia et al., 2016 – 3D CNN for hyperspectral classification  
- Zhong et al., 2016 – PCA + CNN for HSI  
- Chen et al., 2014 – Spectral–spatial deep learning  
- Wang et al., 2019 – Multi-scale CNNs  
