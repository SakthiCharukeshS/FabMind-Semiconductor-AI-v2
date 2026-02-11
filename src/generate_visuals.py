import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.manifold import TSNE
from data_loader import FabMindDataLoader
from models import SensorAutoencoder, WaferMapCNN

# CONFIG
SECOM_PATH = "data/raw_secom/secom.data"
LABELS_PATH = "data/raw_secom/secom_labels.data"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
RESULTS_DIR = "results"

def generate_visuals():
    print("--- GENERATING VISUALS (t-SNE & Gallery) ---")
    
    loader = FabMindDataLoader(SECOM_PATH, LABELS_PATH, None)
    df = loader.load_secom()
    X_raw, y = loader.get_raw_data(df)
    scaler = joblib.load(f"{MODEL_DIR}/sensor_scaler.pkl")
    X_scaled = scaler.transform(X_raw)
    
    input_dim = X_scaled.shape[1]
    sensor_model = SensorAutoencoder(input_dim=input_dim, latent_dim=64)
    sensor_model.load_state_dict(torch.load(f"{MODEL_DIR}/sensor_autoencoder.pth"))
    sensor_model.eval()
    
    cnn_model = WaferMapCNN(latent_dim=64)
    cnn_model.load_state_dict(torch.load(f"{MODEL_DIR}/wafer_cnn.pth"), strict=False)
    cnn_model.eval()
    
    xgb_model = joblib.load(f"{MODEL_DIR}/xgboost_yield.pkl")
    
    X_img_all = np.load(os.path.join(PROCESSED_DIR, "X_images_64.npy"), mmap_mode='r')
    y_lbl_all = np.load(os.path.join(PROCESSED_DIR, "y_labels.npy"))
    
    safe_idx = np.arange(min(len(X_img_all), 20000))
    idx_bad = np.where(y_lbl_all[safe_idx] != 'none')[0]
    idx_good = np.where(y_lbl_all[safe_idx] == 'none')[0]
    
    fused_vecs = []
    labels_used = []
    
    # Process 500 samples
    sample_size = 500
    with torch.no_grad():
        for i in range(sample_size):
            s_emb = sensor_model.encoder(torch.tensor(X_scaled[i]).float().unsqueeze(0)).numpy().flatten()
            
            if y[i] == 1: idx = np.random.choice(idx_bad)
            else: idx = np.random.choice(idx_good)
            
            img = torch.tensor(X_img_all[idx]).float().unsqueeze(0).unsqueeze(0) / 2.0
            i_emb = cnn_model(img).numpy().flatten()
            
            fused = np.concatenate([s_emb, i_emb])
            fused_vecs.append(fused)
            labels_used.append(y[i])

    X_embedded = np.array(fused_vecs)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_embedded)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[np.array(labels_used)==0, 0], X_tsne[np.array(labels_used)==0, 1], c='dodgerblue', label='Pass', alpha=0.6)
    plt.scatter(X_tsne[np.array(labels_used)==1, 0], X_tsne[np.array(labels_used)==1, 1], c='red', label='Fail', alpha=0.8, edgecolors='black')
    plt.title("FabMind Latent Space (t-SNE)")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/6_tsne_clusters.png", dpi=300)
    plt.close()
    
    print("--- VISUALS UPDATED ---")

if __name__ == "__main__":
    generate_visuals()