import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import pickle

# -------------------------------
# Helper functions to load precompiled outputs
# -------------------------------

@st.cache_data
def load_image(path):
    try:
        return Image.open(path)
    except Exception as e:
        st.error(f"Error loading image at {path}: {e}")
        return None

@st.cache_data
def load_pickle(path):
    try: 
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading pickle file at {path}: {e}")
        return None

# -------------------------------
# Sidebar: Navigation & Project Info
# -------------------------------
st.sidebar.title("Hyperspectral Landcover Classification")
st.sidebar.markdown("""
This dashboard presents the results of a project on **Landcover Classification using Hyperspectral Images**.  
It encompasses exploratory analysis, data preprocessing (Normalization & PCA), deep learning with CNN models (1D, 2D, and 3D) and a comparative evaluation.
""")
nav_options = [
    "Project Overview", 
    "Dataset Information", 
    "Exploratory Data Analysis", 
    "Preprocessing (Normalization & PCA)",
    "CNN Models & Training", 
    "Comparative Analysis",
    "System Architecture",
    "About"
]
selection = st.sidebar.selectbox("Section", nav_options)

# -------------------------------
# Page 1: Project Overview
# -------------------------------
if selection == "Project Overview":
    st.title("Project Overview")
    diff_img = load_image("./images/hyperspectral_vs_rgb.png")
    if diff_img:
        st.image(diff_img, caption="Difference: Hyperspectral vs RGB", use_container_width=True)
    else:
        st.info("Hyperspectral vs RGB image not available.")

    col1, col2 = st.columns(2)
    
    with col1:
        img_left = load_image("./images/rgb.png")
        if img_left:
            st.image(img_left, caption="RGB Image", use_container_width=True)
        else:
            st.info("Left image not available.")
    
    with col2:
        img_right = load_image("./images/rgb_vs_hp.webp")
        if img_right:
            st.image(img_right, caption="Hyperspectral Images", use_container_width=True)
        else:
            st.info("Right image not available.")

    st.markdown("""
    ### Landcover Classification using Hyperspectral Images
    **Objective:**  
    Develop an automated system to classify landcover using hyperspectral images by leveraging state‐of‐the‐art machine learning and deep learning techniques.
    
    **Approach:**  
    - **Data Ingestion & EDA:** Load the ENVI–formatted hyperspectral image and its ground truth, and perform an in-depth exploratory analysis.
    - **Preprocessing:** Apply normalization and use PCA for dimensionality reduction.
    - **Modeling:** Train three types of CNNs:
        - **1D CNN:** Based on per-pixel spectral vectors.
        - **2D CNN:** Based on spatial-spectral patches.
        - **3D CNN:** Based on spatial–spectral cubes.
    - **Comparison:** Compare the outputs, accuracies, losses, and other metrics of the three CNN models relative to the ground truth.
    
    The dashboard also provides details on the system architecture, dataset details, and implementation methodology.
    """)
    
    # Display a project overview image
    proj_img = load_image("./images/project_overview.png")
    if proj_img:
        st.image(proj_img, caption="Project Overview", use_container_width=True)
    else:
        st.info("Project overview image not available.")

# -------------------------------
# Page 2: Dataset Information
# -------------------------------
elif selection == "Dataset Information":
    st.title("Dataset Information")
    st.markdown("""
    ### Kansas AHSI Dataset
    - **Source:** Collected using the visible-SWIR AHSI onboard the GF-5 satellite.
       [Dataset Link](https://doi.org/10.21227/cdn8-db45)
    - **Spatial Resolution:** ~30 m  
    - **Spectral Bands:** 308 bands covering 400–2500 nm
        - 5 nm resolution in the VNIR region (400–1000 nm)
        - 10 nm resolution in the SWIR region (1000–2500 nm)
    - **Image Size:** 650 x 340 pixels.
    - **Ground Truth Classes:** 7 classes (Crop, Meadow, Business Area, Bare Soil, Residential Area, Water, Road)
    
    **Additional Details:**  
    The dataset is provided in ENVI formats, and the associated preprocessed images include ground truth and pseudo-color composites.  
    """)
    
    st.subheader("Raw Dataset Information")
    try:
        with open("data/dataset_info.txt", "r") as f:
            dataset_info = f.read()
        st.text_area("Dataset Info", dataset_info, height=300)
    except Exception:
        st.info("Dataset info text file not available.")
    
    st.subheader("Dataset Visual Examples")
    col1, col2 = st.columns(2)
    with col1:
        gt_img = load_image("./images/ground_truth.png")
        if gt_img:
            st.image(gt_img, caption="Ground Truth Image", use_container_width=True)
        else:
            st.info("Ground truth image not available.")
    with col2:
        run_img = load_image("./images/run_image.png")
        if run_img:
            st.image(run_img, caption="Image for CNN Processing", use_container_width=True)
        else:
            st.info("Image for CNN processing not available.")

# -------------------------------
# Page 3: Exploratory Data Analysis (EDA)
# -------------------------------
elif selection == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("""
    **Objective:**  
    To explore the hyperspectral data’s characteristics and obtain insights for model development.
    
    **Key Analyses:**
    - **Spectral Statistics:** Display of minimum, maximum, mean, and standard deviation values for selected bands.
    - **Visualizations:**  
      - *Pseudo-RGB Composite:* A combined view from selected bands.
      - *Ground Truth Map:* Visualization of class labels.
    
    **Interpretation:**  
    The spectral statistics help in understanding data distribution and variability. The pseudo-RGB composite provides a realistic view of the scene, while the ground truth map shows the available labeled classes.
    """)
    
    try:
        stats = load_pickle("data/spectral_stats.pkl")
        st.subheader("Spectral Statistics")
        st.table(pd.DataFrame(stats))
    except Exception:
        st.info("Spectral statistics not available.")
    
    st.subheader("Pseudo-RGB Composite")
    pseudo_rgb = load_image("./images/pseudo_rgb.png")
    if pseudo_rgb:
        st.image(pseudo_rgb, caption="Pseudo-RGB Composite (Bands 29, 19, 9)", use_container_width=True)
        st.markdown("**Interpretation:** The composite image is created by combining three selected bands. This representation helps visualize spatial patterns and feature differences in the spectral data.")
    else:
        st.info("Pseudo-RGB composite image not available.")
    
    st.subheader("Ground Truth Map")
    gt_disp = load_image("./images/ground_truth.png")
    if gt_disp:
        st.image(gt_disp, caption="Ground Truth", use_container_width=True)
        st.markdown("**Interpretation:** The ground truth map visualizes the labeled classes (e.g., Crop, Meadow, etc.), which are used to train and evaluate the CNN models.")
    else:
        st.info("Ground truth image not available.")

# -------------------------------
# Page 4: Preprocessing (Normalization & PCA)
# -------------------------------
elif selection == "Preprocessing (Normalization & PCA)":
    st.title("Preprocessing: Normalization & PCA")
    st.markdown("""
    **Normalization:**  
    The hyperspectral data is standardized to have zero mean and unit variance. This step ensures that all spectral bands contribute equally during model training.
    
    **PCA (Principal Component Analysis):**  
    PCA reduces the dimensionality from 308 bands to a smaller set (e.g., 30 principal components) while retaining most of the variance (over 90%).  
    The **cumulative explained variance** plot indicates how much information is preserved by the selected components.
                
    **Benefits of using PCA:**
                
    Reduction of Irrelevant Information:
    PCA filters out noise and less informative variability by focusing on the principal components that explain the majority of the variance. 
    This ensures that only the most discriminative features are retained.
                
    Less Overfitting:
    When using the full spectrum, there is a higher risk of overfitting due to the high dimensionality and the presence of noise. 
    PCA helps mitigate this risk by eliminating irrelevant spectral information, leading to better generalization on test data.
    
    **Potential Trade-Offs of using PCA:**
                
    Loss of Fine Spectral Information:
    Although PCA retains most of the overall variance, it is a linear transformation and might inadvertently discard some subtle yet important spectral details. 
    If too few components are retained, this could lead to a slight drop in classification performance.
                
    Determining the Optimal Number of Components:
    A key aspect is choosing the correct number of principal components. Using an explained variance plot allows us to determine an optimal balance—ensuring enough
    information is preserved while reducing dimensionality sufficiently to improve the model's learning efficiency.
    """)

    
    pca_plot = load_image("./images/pca_explained_variance.png")
    if pca_plot:
        st.image(pca_plot, caption="Cumulative Explained Variance from PCA", use_container_width=True)
        st.markdown("**Interpretation:** This plot shows the cumulative variance explained by the first N principal components. A steeper slope means that fewer components are needed to capture most of the variance.")
    else:
        st.info("PCA explained variance plot not available.")
    
   

# -------------------------------
# Page 5: CNN Models & Training
# -------------------------------
elif selection == "CNN Models & Training":
    st.title("CNN Models & Training")
    st.markdown("""
    **Models Implemented:**  
    - **1D CNN:** Processes per-pixel spectral vectors (after PCA).  
    - **2D CNN:** Uses spatial-spectral patches that include local spatial context.  
    - **3D CNN:** Exploits full spatial-spectral cubes with 3D convolutional layers.
    
    **Training Outcomes:**  
    For each CNN model, accuracy and loss curves are generated and classification reports are produced.
    
    **Interpretation:**  
    The training plots display the convergence behavior and generalization performance of each model. The classification reports provide per-class precision, recall, and F1-scores.
    """)
    
    st.markdown("#### Model Training Metrics")
    tab1, tab2, tab3 = st.tabs(["1D CNN", "2D CNN", "3D CNN"])
    
    with tab1:
        st.subheader("1D CNN")
        acc1 = load_image("./images/1d_accuracy.png")
        loss1 = load_image("./images/1d_loss.png")
        if acc1 and loss1:
            st.image(acc1, caption="1D CNN Accuracy", use_container_width=True)
            st.image(loss1, caption="1D CNN Loss", use_container_width=True)
        else:
            st.info("1D CNN training plots not available.")
        report1 = load_pickle("data/1d_classification_report.pkl")
        if report1:
            st.subheader("1D CNN Classification Report")
            st.text(report1)
        else:
            st.info("1D CNN classification report not available.")
    
    with tab2:
        st.subheader("2D CNN")
        acc2 = load_image("./images/2d_accuracy.png")
        loss2 = load_image("./images/2d_loss.png")
        if acc2 and loss2:
            st.image(acc2, caption="2D CNN Accuracy", use_container_width=True)
            st.image(loss2, caption="2D CNN Loss", use_container_width=True)
        else:
            st.info("2D CNN training plots not available.")
        report2 = load_pickle("data/2d_classification_report.pkl")
        if report2:
            st.subheader("2D CNN Classification Report")
            st.text(report2)
        else:
            st.info("2D CNN classification report not available.")
    
    with tab3:
        st.subheader("3D CNN")
        acc3 = load_image("./images/3d_accuracy.png")
        loss3 = load_image("./images/3d_loss.png")
        if acc3 and loss3:
            st.image(acc3, caption="3D CNN Accuracy", use_container_width=True)
            st.image(loss3, caption="3D CNN Loss", use_container_width=True)
        else:
            st.info("3D CNN training plots not available.")
        report3 = load_pickle("data/3d_classification_report.pkl")
        if report3:
            st.subheader("3D CNN Classification Report")
            st.text(report3)
        else:
            st.info("3D CNN classification report not available.")
    
    st.markdown("#### Comparison Table of Accuracy and Loss")
    comp_table = load_pickle("data/accuracy_loss_comparison.pkl")
    if comp_table:
        df_comp = pd.DataFrame(comp_table)
        st.table(df_comp)
    else:
        st.info("Comparison table for CNN models not available.")

# -------------------------------
# Page 6: Comparative Analysis
# -------------------------------
elif selection == "Comparative Analysis":
    st.title("Comparative Analysis of CNN Outputs")
    st.markdown("""
    **Objective:**  
    To compare the classification maps generated by the 1D, 2D, and 3D CNN models relative to the Ground Truth.
    
    **Interpretation:**  
    Visual comparison helps to assess the spatial coherence and class boundary definition. The accuracy and loss metrics from the previous section further aid in evaluating overall performance.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        gt_img = load_image("./images/ground_truth.png")
        if gt_img:
            st.image(gt_img, caption="Ground Truth", use_container_width=True)
        else:
            st.info("Ground truth image not available.")
    
    with col2:
        tabA, tabB, tabC = st.tabs(["1D CNN", "2D CNN", "3D CNN"])
        with tabA:
            map1 = load_image("./images/1d_classification_map.png")
            if map1:
                st.image(map1, caption="1D CNN Classification", use_container_width=True)
            else:
                st.info("1D CNN classification map not available.")
        with tabB:
            map2 = load_image("./images/2d_classification_map.png")
            if map2:
                st.image(map2, caption="2D CNN Classification", use_container_width=True)
            else:
                st.info("2D CNN classification map not available.")
        with tabC:
            map3 = load_image("./images/3d_classification_map.png")
            if map3:
                st.image(map3, caption="3D CNN Classification", use_container_width=True)
            else:
                st.info("3D CNN classification map not available.")
    
    st.subheader("Overall Accuracy Comparison")
    try:
        comp_metrics = load_pickle("data/accuracy_comparison.pkl")  # e.g., {"1D": 0.85, "2D": 0.88, "3D": 0.90}
        df_comp = pd.DataFrame(list(comp_metrics.items()), columns=["Model", "Overall Accuracy"])
        fig = px.bar(df_comp, x="Model", y="Overall Accuracy", color="Model", title="CNN Accuracy Comparison")
        st.plotly_chart(fig)
    except Exception:
        st.info("Comparative accuracy metrics not available.")

# -------------------------------
# Page 7: System Architecture
# -------------------------------
elif selection == "System Architecture":
    st.title("System Architecture")
    st.markdown("""
    ### Overview of the System Architecture
    The following diagram illustrates the overall system architecture for the landcover classification project:
    
    - **Data Acquisition:** Hyperspectral images are acquired using the GF-5 satellite.
    - **Preprocessing Pipeline:** Involves radiometric/geometric correction, normalization, and PCA for dimensionality reduction.
    - **Modeling:** Three separate CNN approaches (1D, 2D, and 3D) are employed:
        - **1D CNN:** Learns from per-pixel spectral vectors.
        - **2D CNN:** Utilizes spatial-spectral patches to incorporate local context.
        - **3D CNN:** Leverages full spatial-spectral cubes for richer feature learning.
    - **Evaluation & Comparison:** Each model’s output is compared visually and via detailed metrics.
    """)
    
    arch_img = load_image("./images/system_architecture.png")
    if arch_img:
        st.image(arch_img, caption="System Architecture Diagram", use_container_width=True)
    else:
        st.info("System architecture image not available.")
    
    st.markdown("**Interpretation:** This architecture shows the end-to-end pipeline from data collection to final classification. Each module is designed to ensure optimal performance and accuracy.")

# -------------------------------
# Page 8: About / Implementation Details
# -------------------------------
elif selection == "About":
    st.title("About the Project & Implementation")
    st.markdown("""
    **Project Details:**  
    - **Title:** Landcover Classification using Hyperspectral Images  
    - **Supervisor:** Dr. Ujwala Bharambe  
    - **Department:** Computer Engineering  
    - **Institution:** Thadomal Shahani Engineering College.
    
    
    **Implementation Overview:**  
    This project integrates advanced data preprocessing, dimensionality reduction, and multiple CNN architectures (1D, 2D, 3D) using Python, TensorFlow/Keras, and Spectral Python.  
    The dashboard is built with Streamlit for an interactive and professional presentation.
    
    **Steps Implemented:**  
    1. **Data Ingestion & EDA:** Comprehensive analysis and visualization of the hyperspectral dataset.  
    2. **Preprocessing:** Normalization and PCA reduce the high dimensionality while retaining critical spectral information.  
    3. **Modeling:** Three CNN pipelines are developed and compared.  
    4. **Evaluation & Comparison:** Detailed metrics and visualizations support model evaluation.
    
    **Note:** All outputs shown in this dashboard have been precompiled for faster performance.
    
    **Credits & References:**  
    - Hyperspectral Image Processing using [Spectral Python](https://www.spectralpython.net/)  
    - Deep Learning with [TensorFlow](https://www.tensorflow.org/)  
    - Dashboard built with [Streamlit](https://streamlit.io/)
    """)
    
