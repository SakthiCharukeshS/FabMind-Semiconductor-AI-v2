import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------
# MODEL 1: Sensor Autoencoder (For SECOM Data)
# ----------------------------------------
class SensorAutoencoder(nn.Module):
    def __init__(self, input_dim=563, latent_dim=64):
        super(SensorAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim), # This is the "Embedding"
            nn.ReLU()
        )
        # Decoder (Only needed for training)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# ----------------------------------------
# MODEL 2: Wafer Map CNN (For WM-811K Images)
# ----------------------------------------
class WaferMapCNN(nn.Module):
    def __init__(self, latent_dim=64):
        super(WaferMapCNN, self).__init__()
        # Input is (1, 64, 64) -> Grayscale Image
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) # 64x64 -> 32x32
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Pool: 32x32 -> 16x16
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Pool: 16x16 -> 8x8
        
        # Flatten
        self.flatten_size = 64 * 8 * 8
        self.fc = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        # x shape: [batch, 64, 64] -> Needs [batch, 1, 64, 64]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, self.flatten_size) # Flatten
        x = self.fc(x) # Compress to latent_dim
        return x

# Note: Fusion is handled by XGBoost in train_fusion.py (Late Fusion)
# Unused PyTorch Fusion Architecture removed to fix Issue 7.

if __name__ == "__main__":
    print("Testing Architectures...")
    # Test Sensor Model
    sensor_model = SensorAutoencoder(input_dim=591)
    dummy_sensor = torch.randn(10, 591)
    enc, dec = sensor_model(dummy_sensor)
    print(f"Sensor Model: Encoded {enc.shape}")