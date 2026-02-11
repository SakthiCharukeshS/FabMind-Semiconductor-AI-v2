import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from models import WaferMapCNN
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Configuration
PROCESSED_DIR = "data/processed"
MODEL_SAVE_PATH = "models/wafer_cnn.pth"
EPOCHS = 5  # Keep it low for speed (increase to 10-20 for final result)
BATCH_SIZE = 64
LEARNING_RATE = 0.001

def train_image_model():
    print("--- Starting Phase 4B: Training Wafer Map CNN ---")
    
    # 1. Load Processed Data
    print("Loading Image Data (This is fast)...")
    X_img = np.load(os.path.join(PROCESSED_DIR, "X_images_64.npy"))
    y_lbl = np.load(os.path.join(PROCESSED_DIR, "y_labels.npy"))
    
    # 2. Encode Labels (Text -> Numbers)
    # e.g., 'Scratch' -> 1, 'Donut' -> 2
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_lbl)
    
    # Save the label mapping for later (Dashboard needs this)
    np.save(os.path.join("models", "label_classes.npy"), encoder.classes_)
    print(f"Classes found: {encoder.classes_}")
    
    # 3. Split Train/Test
    # We use 80% for training, 20% for validation
    X_train, X_test, y_train, y_test = train_test_split(X_img, y_encoded, test_size=0.2, random_state=42)
    
    # 4. Convert to PyTorch Tensors
    # Images need to be Float [0-1] usually, but ours are [0,1,2]. 
    # Let's Normalize slightly by dividing by 2.0 to get range [0-1]
    tensor_x_train = torch.tensor(X_train, dtype=torch.float32) / 2.0
    tensor_y_train = torch.tensor(y_train, dtype=torch.long)
    
    # Create DataLoader (Batches)
    dataset = TensorDataset(tensor_x_train, tensor_y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 5. Setup Model
    num_classes = len(encoder.classes_)
    # We modify the model slightly to output classes instead of latent vector for training
    model = WaferMapCNN(latent_dim=64) 
    
    # HACK: We add a temporary classification head just for training
    # The base model outputs 64 features. We need to map 64 -> num_classes
    class TrainingWrapper(nn.Module):
        def __init__(self, base_model, n_classes):
            super().__init__()
            self.base = base_model
            self.head = nn.Linear(64, n_classes)
        def forward(self, x):
            features = self.base(x)
            return self.head(features)
            
    full_model = TrainingWrapper(model, num_classes)
    
    optimizer = optim.Adam(full_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 6. Training Loop
    full_model.train()
    print(f"Training on {len(X_train)} images for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (imgs, labels) in enumerate(dataloader):
            # Forward
            outputs = full_model(imgs)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"\rEpoch {epoch+1} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}", end="")
        
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")

    # 7. Save ONLY the base model (The Feature Extractor)
    # We discard the temporary classification head because for Fusion we only want the features
    torch.save(full_model.base.state_dict(), MODEL_SAVE_PATH)
    print(f"--- SUCCESS: CNN Model Saved to {MODEL_SAVE_PATH} ---")

if __name__ == "__main__":
    train_image_model()