import pandas as pd
import numpy as np
import cv2  # OpenCV for image resizing
import os
import pickle

# Configuration
RAW_WM_PATH = "data/raw_wm811k/LSWMD.pkl"
PROCESSED_DIR = "data/processed"
IMG_SIZE = 64  # Resize all wafer maps to 64x64

def process_wafer_maps():
    print("--- Starting Phase 2: Image Preprocessing ---")
    
    # 1. Load the Big File
    if not os.path.exists(RAW_WM_PATH):
        raise FileNotFoundError("LSWMD.pkl not found!")
    
    print("Loading raw pickle file (Wait ~20s)...")
    df = pd.read_pickle(RAW_WM_PATH)
    
    print(f"Original Count: {len(df)} wafers")
    
    # 2. Inspect & Clean Metadata
    # The dataset has extra dimensions in the list, we need to clean headers
    # Columns usually: waferMap, dieSize, lotName, waferIndex, trainTestLabel, failureType
    
    print("Filtering Data...")
    # Add a helper column to see if it has a failure label
    # failureType is usually a list like [['Loc']] or []
    df['has_label'] = df['failureType'].apply(lambda x: len(x) > 0)
    
    # FILTER: Keep ALL Failures + Sample of 5000 Good wafers
    df_failures = df[df['has_label'] == True]
    df_good = df[df['has_label'] == False].sample(n=5000, random_state=42)
    
    # Combine
    df_subset = pd.concat([df_failures, df_good])
    print(f"Filtered Subset: {len(df_subset)} wafers (Failures + Sampled Good)")
    
    # 3. Resize Images
    print(f"Resizing images to {IMG_SIZE}x{IMG_SIZE}...")
    
    resized_maps = []
    labels = []
    
    for index, row in df_subset.iterrows():
        # Get the wafer map (it's a numpy array of 0, 1, 2)
        wafer_map = row['waferMap']
        
        # Resize using Cubic interpolation (best for shrinking)
        # We need to ensure it's float for resizing, then back to int
        resized = cv2.resize(wafer_map.astype('float32'), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        resized_maps.append(resized)
        
        # Clean the label (convert [['Loc']] to 'Loc')
        raw_label = row['failureType']
        if len(raw_label) > 0:
            labels.append(raw_label[0][0])
        else:
            labels.append("none")

    # 4. Save Processed Data
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    # Convert to Numpy Arrays for AI
    X_images = np.array(resized_maps)
    y_labels = np.array(labels)
    
    print(f"Saving processed data to {PROCESSED_DIR}...")
    np.save(os.path.join(PROCESSED_DIR, "X_images_64.npy"), X_images)
    np.save(os.path.join(PROCESSED_DIR, "y_labels.npy"), y_labels)
    
    print("--- SUCCESS: Data Processed & Saved! ---")
    print(f"Final X Shape: {X_images.shape}")
    print(f"Final Y Shape: {y_labels.shape}")

if __name__ == "__main__":
    process_wafer_maps()