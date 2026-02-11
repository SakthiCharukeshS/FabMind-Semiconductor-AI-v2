import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import FabMindDataLoader
from models import SensorAutoencoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib
import numpy as np

# Configuration
SECOM_PATH = "data/raw_secom/secom.data"
LABELS_PATH = "data/raw_secom/secom_labels.data"
MODEL_SAVE_PATH = "models/sensor_autoencoder.pth"
SCALER_SAVE_PATH = "models/sensor_scaler.pkl"
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def train_sensor_model():
    print("--- Starting Phase 4A: Training Sensor Autoencoder (No Leakage) ---")
    
    # 1. Load Raw Data (Using get_raw_data to avoid pre-scaling)
    loader = FabMindDataLoader(SECOM_PATH, LABELS_PATH, None)
    df_raw = loader.load_secom()
    X, y = loader.get_raw_data(df_raw)
    
    # 2. SPLIT FIRST (Fixes Issue 5: Data Leakage)
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. SCALE ONLY ON TRAINING DATA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Save the scaler so we can use it on Test data later
    if not os.path.exists("models"): os.makedirs("models")
    joblib.dump(scaler, SCALER_SAVE_PATH)
    
    # Convert to Tensor
    tensor_x_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    
    # 4. Setup Model
    input_dim = X_train.shape[1]
    print(f"Input Features: {input_dim}")
    
    model = SensorAutoencoder(input_dim=input_dim, latent_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # 5. Training Loop
    model.train()
    print(f"Training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        encoded, decoded = model(tensor_x_train)
        loss = criterion(decoded, tensor_x_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"--- SUCCESS: Sensor Model Saved to {MODEL_SAVE_PATH} ---")

if __name__ == "__main__":
    train_sensor_model()