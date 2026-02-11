import torch
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import joblib
from data_loader import FabMindDataLoader
from models import SensorAutoencoder, WaferMapCNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Configuration
SECOM_PATH = "data/raw_secom/secom.data"
LABELS_PATH = "data/raw_secom/secom_labels.data"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

def train_fusion_model():
    print("--- Starting Phase 5: Multimodal Fusion & Yield Prediction ---")
    
    # 1. LOAD DATA & SCALER
    print("Loading Data...")
    loader = FabMindDataLoader(SECOM_PATH, LABELS_PATH, None)
    df = loader.load_secom()
    X_raw, y_secom = loader.get_raw_data(df)
    
    # Load the saved scaler (Ensure consistent scaling)
    if not os.path.exists(f"{MODEL_DIR}/sensor_scaler.pkl"):
        print("❌ Scaler not found. Please run train_sensors.py first.")
        return
        
    scaler = joblib.load(f"{MODEL_DIR}/sensor_scaler.pkl")
    X_scaled = scaler.transform(X_raw)
    
    # 2. LOAD PRE-TRAINED MODELS
    input_dim = X_scaled.shape[1]
    sensor_model = SensorAutoencoder(input_dim=input_dim, latent_dim=64)
    sensor_model.load_state_dict(torch.load(f"{MODEL_DIR}/sensor_autoencoder.pth"))
    sensor_model.eval()
    
    cnn_model = WaferMapCNN(latent_dim=64)
    cnn_model.load_state_dict(torch.load(f"{MODEL_DIR}/wafer_cnn.pth"), strict=False)
    cnn_model.eval()
    
    # Load Images
    X_images_all = np.load(os.path.join(PROCESSED_DIR, "X_images_64.npy"), mmap_mode='r')
    y_labels_all = np.load(os.path.join(PROCESSED_DIR, "y_labels.npy"))
    
    # Safe Indices for Image Loading
    safe_idx = np.arange(min(len(X_images_all), 50000))
    indices_good = np.where(y_labels_all[safe_idx] == 'none')[0]
    indices_bad = np.where(y_labels_all[safe_idx] != 'none')[0]
    
    # 3. GENERATE FUSED VECTORS
    print("Generating Fused Vectors...")
    fused_vectors = []
    final_labels = []
    
    with torch.no_grad():
        for i in range(len(X_scaled)):
            # A. Sensor Embedding
            s_emb = sensor_model.encoder(torch.tensor(X_scaled[i]).float().unsqueeze(0)).numpy().flatten()
            
            # B. Pick Matching Image
            is_fail = (y_secom[i] == 1)
            if is_fail: idx = np.random.choice(indices_bad)
            else: idx = np.random.choice(indices_good)
            
            img_data = X_images_all[idx]
            img_tensor = torch.tensor(img_data).float().unsqueeze(0).unsqueeze(0) / 2.0
            i_emb = cnn_model(img_tensor).numpy().flatten()
            
            # C. Fuse
            fused = np.concatenate([s_emb, i_emb])
            fused_vectors.append(fused)
            final_labels.append(is_fail)

    X_fused = np.array(fused_vectors)
    y_fused = np.array(final_labels)
    
    # 4. TRAIN XGBOOST
    print("Training XGBoost...")
    X_train, X_test, y_train, y_test = train_test_split(X_fused, y_fused, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=10, 
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # 5. EVALUATE
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nYield Prediction Accuracy: {acc*100:.2f}%")
    
    joblib.dump(model, f"{MODEL_DIR}/xgboost_yield.pkl")
    print(f"--- SUCCESS: Fusion Model Saved ---")

if __name__ == "__main__":
    train_fusion_model()