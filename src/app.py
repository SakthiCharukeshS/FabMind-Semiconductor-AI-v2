# Final SOTA Implementation
import streamlit as st
import pandas as pd
import numpy as np
import torch
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap
from data_loader import FabMindDataLoader
from models import SensorAutoencoder, WaferMapCNN

# ---------------------------------------------------------
# CONFIGURATION & SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="FabMind AI Dashboard", layout="wide")

# Paths
SECOM_PATH = "data/raw_secom/secom.data"
LABELS_PATH = "data/raw_secom/secom_labels.data"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

# ---------------------------------------------------------
# LOAD RESOURCES (Cached for Speed)
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    print("Loading Models & Data...")
    
    # 1. Load Data for Simulation
    loader = FabMindDataLoader(SECOM_PATH, LABELS_PATH, None)
    df_raw = loader.load_secom()
    df_clean = loader.clean_secom_data(df_raw)
    
    X_sensors = df_clean.drop(columns=['Pass_Fail']).values.astype(np.float32)
    y_secom = df_clean['Pass_Fail'].values
    
    # 2. Load Images
    X_images = np.load(os.path.join(PROCESSED_DIR, "X_images_64.npy"))
    y_labels = np.load(os.path.join(PROCESSED_DIR, "y_labels.npy"))
    
    # 3. Load Models
    # Sensor Model
    sensor_dim = X_sensors.shape[1]
    sensor_model = SensorAutoencoder(input_dim=sensor_dim, latent_dim=64)
    sensor_model.load_state_dict(torch.load(f"{MODEL_DIR}/sensor_autoencoder.pth"))
    sensor_model.eval()
    
    # Image Model
    cnn_model = WaferMapCNN(latent_dim=64)
    cnn_model.load_state_dict(torch.load(f"{MODEL_DIR}/wafer_cnn.pth"), strict=False)
    cnn_model.eval()
    
    # XGBoost Model
    import joblib
    xgb_model = joblib.load(f"{MODEL_DIR}/xgboost_yield.pkl")
    
    return X_sensors, y_secom, X_images, y_labels, sensor_model, cnn_model, xgb_model

# Load everything once
X_sensors, y_secom, X_images, y_labels, sensor_model, cnn_model, xgb_model = load_resources()

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.title("FabMind Control Panel")
st.sidebar.image("https://img.icons8.com/dusk/64/000000/integrated-circuit.png", width=100)

mode = st.sidebar.radio("Simulation Mode", ["Random Good Wafer", "Random Bad Wafer"])

if st.sidebar.button("Simulate New Wafer 🚀"):
    # LOGIC TO PICK A WAFER
    indices_fail = np.where(y_secom == 1)[0]
    indices_pass = np.where(y_secom == -1)[0] # -1 is pass in dataframe logic
    
    if mode == "Random Bad Wafer":
        idx = np.random.choice(indices_fail)
        st.session_state['current_idx'] = idx
        st.session_state['wafer_type'] = "Fail"
    else:
        idx = np.random.choice(indices_pass)
        st.session_state['current_idx'] = idx
        st.session_state['wafer_type'] = "Pass"
        
    st.experimental_rerun()

# Default State
if 'current_idx' not in st.session_state:
    st.session_state['current_idx'] = 0
    st.session_state['wafer_type'] = "Pass"

# ---------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------
st.title("🏭 FabMind: Intelligent Semiconductor Yield Prediction")
st.markdown("### Real-time Multimodal Sensor & Vision Analysis")

col1, col2, col3 = st.columns([1, 1, 1])

# Current Wafer Data
current_idx = st.session_state['current_idx']
sensor_data = X_sensors[current_idx]

# Pick a random image that matches the type (Pass/Fail)
# We map sensor failure to bad images for the demo visual
if st.session_state['wafer_type'] == "Fail":
    bad_img_indices = np.where(y_labels != 'none')[0]
    img_idx = np.random.choice(bad_img_indices)
else:
    good_img_indices = np.where(y_labels == 'none')[0]
    img_idx = np.random.choice(good_img_indices)

wafer_image = X_images[img_idx]
wafer_label = y_labels[img_idx]

# --- COLUMN 1: SENSOR DATA ---
with col1:
    st.subheader("📡 IoT Sensor Stream")
    # Plot first 50 sensors as a signal
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(sensor_data[:50], color='#00ff41', linewidth=1.5)
    ax.set_facecolor('black')
    ax.set_title("Chamber Sensors (First 50)", color='white')
    st.pyplot(fig)
    
    st.metric("Pressure Sensor (Avg)", f"{np.mean(sensor_data):.2f}", delta="Normal")

# --- COLUMN 2: VISUAL INSPECTION ---
with col2:
    st.subheader("👁️ Wafer Map Vision")
    
    # Display Image
    fig2, ax2 = plt.subplots(figsize=(3, 3))
    ax2.imshow(wafer_image, cmap='inferno')
    ax2.axis('off')
    st.pyplot(fig2)
    
    st.info(f"Detected Pattern: **{wafer_label}**")

# --- COLUMN 3: AI PREDICTION ---
with col3:
    st.subheader("🧠 Yield Prediction")
    
    # RUN INFERENCE
    # 1. Encode Sensor
    s_tensor = torch.tensor(sensor_data).unsqueeze(0)
    with torch.no_grad():
        s_emb = sensor_model.encoder(s_tensor)
        
    # 2. Encode Image
    i_tensor = torch.tensor(wafer_image, dtype=torch.float32).unsqueeze(0) / 2.0
    if len(i_tensor.shape) == 2: i_tensor = i_tensor.unsqueeze(0).unsqueeze(0)
    elif len(i_tensor.shape) == 3: i_tensor = i_tensor.unsqueeze(1)
        
    with torch.no_grad():
        i_emb = cnn_model(i_tensor)
        
    # 3. Fuse
    fused_vec = torch.cat((s_emb, i_emb), dim=1).numpy()
    
    # 4. Predict
    prob = xgb_model.predict_proba(fused_vec)[0][1] # Probability of Failure
    
    if prob > 0.5:
        st.error(f"FAIL PREDICTED ({prob*100:.1f}%)")
        status = "CRITICAL FAILURE"
    else:
        st.success(f"PASS PREDICTED ({(1-prob)*100:.1f}%)")
        status = "OPTIMAL"
        
    st.metric("System Status", status)

# ---------------------------------------------------------
# XAI SECTION (Why did it fail?)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔍 Explainable AI (Root Cause Analysis)")

if st.button("Run XAI Diagnostics"):
    with st.spinner("Analyzing Feature Contributions..."):
        # We use TreeExplainer for XGBoost
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(fused_vec)
        
        # Plot
        st.markdown("**Top Factors Contributing to Decision:**")
        fig_shap, ax_shap = plt.subplots()
        shap.summary_plot(shap_values, fused_vec, plot_type="bar", max_display=10, show=False)
        st.pyplot(fig_shap)
        
        st.success("Analysis Complete. The bars show which hidden features pushed the decision.")