import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import torch
import joblib
from sklearn.decomposition import PCA
from data_loader import FabMindDataLoader
from models import SensorAutoencoder

# CONFIG
RESULTS_DIR = "results"
MODEL_DIR = "models"
SECOM_PATH = "data/raw_secom/secom.data"
LABELS_PATH = "data/raw_secom/secom_labels.data"

plt.style.use('seaborn-v0_8-whitegrid')

def generate_assets():
    print("--- GENERATING SIGNAL & PCA ASSETS ---")
    
    loader = FabMindDataLoader(SECOM_PATH, LABELS_PATH, None)
    df = loader.load_secom()
    X_raw, y = loader.get_raw_data(df)
    
    scaler = joblib.load(f"{MODEL_DIR}/sensor_scaler.pkl")
    X_scaled = scaler.transform(X_raw)
    
    input_dim = X_scaled.shape[1]
    sensor_model = SensorAutoencoder(input_dim=input_dim, latent_dim=64)
    sensor_model.load_state_dict(torch.load(f"{MODEL_DIR}/sensor_autoencoder.pth"))
    sensor_model.eval()

    # 1. SIGNAL RECONSTRUCTION
    idx = np.random.randint(0, len(X_scaled))
    original = X_scaled[idx]
    with torch.no_grad():
        _, reconstructed = sensor_model(torch.tensor(original).float().unsqueeze(0))
    reconstructed = reconstructed.numpy().flatten()
    
    plt.figure(figsize=(12, 5))
    plt.plot(original[:100], label='Original', color='gray', alpha=0.7)
    plt.plot(reconstructed[:100], label='Reconstructed', color='blue', linewidth=2)
    plt.title(f"Sensor Reconstruction (Sample #{idx})")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/10_signal_reconstruction.png", dpi=300)
    plt.close()

    # 2. SENSOR CORRELATION
    subset_df = pd.DataFrame(X_scaled[:, :20])
    corr = subset_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', cbar=True)
    plt.title("Sensor Cross-Correlation")
    plt.savefig(f"{RESULTS_DIR}/13_sensor_correlation.png", dpi=300)
    plt.close()
    
    print("--- ASSETS UPDATED ---")

if __name__ == "__main__":
    generate_assets()