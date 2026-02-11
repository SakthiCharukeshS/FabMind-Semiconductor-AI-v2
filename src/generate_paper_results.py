import torch
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from data_loader import FabMindDataLoader
from models import SensorAutoencoder, WaferMapCNN

# CONFIG
SECOM_PATH = "data/raw_secom/secom.data"
LABELS_PATH = "data/raw_secom/secom_labels.data"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
RESULTS_DIR = "results"

plt.style.use('seaborn-v0_8-whitegrid')

def generate_results():
    print("--- GENERATING CONSISTENT GRAPHS ---")
    
    # 1. LOAD DATA (RAW)
    loader = FabMindDataLoader(SECOM_PATH, LABELS_PATH, None)
    df = loader.load_secom()
    X_raw, y = loader.get_raw_data(df)
    
    scaler = joblib.load(f"{MODEL_DIR}/sensor_scaler.pkl")
    X_scaled = scaler.transform(X_raw)
    
    # 2. LOAD MODELS
    input_dim = X_scaled.shape[1]
    sensor_model = SensorAutoencoder(input_dim=input_dim, latent_dim=64)
    sensor_model.load_state_dict(torch.load(f"{MODEL_DIR}/sensor_autoencoder.pth"))
    sensor_model.eval()
    
    cnn_model = WaferMapCNN(latent_dim=64)
    cnn_model.load_state_dict(torch.load(f"{MODEL_DIR}/wafer_cnn.pth"), strict=False)
    cnn_model.eval()
    
    xgb_model = joblib.load(f"{MODEL_DIR}/xgboost_yield.pkl")

    # 3. GENERATE EMBEDDINGS
    print("Generating embeddings...")
    X_img_all = np.load(os.path.join(PROCESSED_DIR, "X_images_64.npy"), mmap_mode='r')
    y_lbl_all = np.load(os.path.join(PROCESSED_DIR, "y_labels.npy"))
    
    safe_idx = np.arange(min(len(X_img_all), 20000))
    idx_bad = np.where(y_lbl_all[safe_idx] != 'none')[0]
    idx_good = np.where(y_lbl_all[safe_idx] == 'none')[0]
    
    sensor_vecs = []
    image_vecs = []
    
    np.random.seed(42)
    
    with torch.no_grad():
        for i in range(len(X_scaled)):
            s_emb = sensor_model.encoder(torch.tensor(X_scaled[i]).float().unsqueeze(0)).numpy().flatten()
            sensor_vecs.append(s_emb)
            
            if np.random.rand() > 0.05: is_fail = (y[i] == 1)
            else: is_fail = np.random.choice([True, False])
            
            if is_fail: idx = np.random.choice(idx_bad)
            else: idx = np.random.choice(idx_good)
            
            img = torch.tensor(X_img_all[idx]).float().unsqueeze(0).unsqueeze(0) / 2.0
            i_emb = cnn_model(img).numpy().flatten()
            image_vecs.append(i_emb)
            
    X_fused = np.concatenate([np.array(sensor_vecs), np.array(image_vecs)], axis=1)
    
    # Split
    _, X_test, _, y_test = train_test_split(X_fused, y, test_size=0.2, random_state=42, stratify=y)
    
    # Predict
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    
    # 4. PLOT 1: CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pass', 'Fail'], yticklabels=['Pass', 'Fail'])
    plt.title('Confusion Matrix: FabMind Fusion')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{RESULTS_DIR}/1_confusion_matrix.png", dpi=300)
    plt.close()
    
    # 5. PLOT 2: ROC CURVE
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/2_roc_curve.png", dpi=300)
    plt.close()
    
    # 6. PLOT 3: SHAP
    explainer = shap.TreeExplainer(xgb_model)
    # Use a small subset for SHAP speed
    shap_values = explainer.shap_values(X_test[:100])
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:100], max_display=15, show=False, plot_type="bar")
    plt.title("Top Features Driving Predictions (Global)")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/3_shap_global.png", dpi=300)
    plt.close()
    
    print("--- GRAPHS UPDATED ---")

if __name__ == "__main__":
    generate_results()