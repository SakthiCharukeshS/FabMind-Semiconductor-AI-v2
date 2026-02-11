import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os

class FabMindDataLoader:
    def __init__(self, data_path_secom, data_path_labels, data_path_wm811k=None):
        self.secom_path = data_path_secom
        self.labels_path = data_path_labels
        self.wm811k_path = data_path_wm811k
        
    def load_secom(self):
        """Phase 1A: Load Raw Sensor Data"""
        print("Loading SECOM Data...")
        if not os.path.exists(self.secom_path):
            raise FileNotFoundError(f"Missing: {self.secom_path}")
            
        df_sensors = pd.read_csv(self.secom_path, sep=" ", header=None)
        
        # Load Labels
        df_labels = pd.read_csv(self.labels_path, sep=" ", header=None)
        df_labels.columns = ['Pass_Fail', 'Timestamp']
        
        # Merge
        df = pd.concat([df_labels, df_sensors], axis=1)
        # Handle timestamp parsing safely
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
        except:
            pass # Keep as string if format fails
            
        print(f"SECOM Data Loaded. Shape: {df.shape}")
        return df

    def get_raw_data(self, df):
        """
        Returns RAW X (features) and y (labels).
        NOTE: We do NOT scale here to prevent Data Leakage (Issue 4).
        Scaling must happen inside the training script after train_test_split.
        """
        # 1. Drop Metadata
        labels = df['Pass_Fail']
        # Convert -1 (Pass) to 0, 1 (Fail) to 1
        y = np.where(labels == -1, 0, 1)
        
        sensors = df.drop(['Pass_Fail', 'Timestamp'], axis=1)
        
        # 2. Drop High-Null Columns (>50% missing)
        threshold = len(sensors) * 0.5
        sensors_dropped = sensors.dropna(thresh=threshold, axis=1)
        
        # 3. Simple Imputation (Must happen here to handle NaNs)
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(sensors_dropped)
        
        return X, y

if __name__ == "__main__":
    # Test
    SECOM_PATH = "data/raw_secom/secom.data"
    LABELS_PATH = "data/raw_secom/secom_labels.data"
    
    loader = FabMindDataLoader(SECOM_PATH, LABELS_PATH)
    df = loader.load_secom()
    X, y = loader.get_raw_data(df)
    print(f"Raw Data Shape (No Leakage): {X.shape}")