import os
import pandas as pd
import pydicom
import numpy as np
from PIL import Image

# Configuration
RAW_DIR = "data/raw/stage_2_train_images"
LABELS_CSV = "data/raw/stage_2_train_labels.csv"
OUTPUT_DIR = "data"
TOTAL_IMAGES = 3000  # Subset for 8GB RAM laptop
TARGET_SIZE = (224, 224)

def dicom_to_png(dicom_path, output_path):
    """Converts a DICOM file to a normalized 224x224 PNG."""
    try:
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array
        # Normalize pixel values to 0-255
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        img = Image.fromarray(img.astype(np.uint8)).resize(TARGET_SIZE)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error converting {dicom_path}: {e}")
        return False

def main():
    print("🔄 Loading labels and subsetting data...")
    df = pd.read_csv(LABELS_CSV)
    # Drop duplicates and reset index to ensure clean 0-to-N indexing
    df = df.drop_duplicates(subset=['patientId']).reset_index(drop=True)
    
    # Subset to 3000 images for fast CPU training
    df_subset = df.sample(n=TOTAL_IMAGES, random_state=42).reset_index(drop=True)

    print("🔄 Creating Non-IID splits (80/20 skew)...")
    # Separate positive (pneumonia) and negative (normal) cases
    pos_cases = df_subset[df_subset['Target'] == 1].copy()
    neg_cases = df_subset[df_subset['Target'] == 0].copy()

    # Initialize a 'hospital' column to track assignment safely
    df_subset['hospital'] = 'unassigned'

    # Hospital A: 80% Pneumonia, 20% Normal (Specialized Clinic)
    pos_a = pos_cases.sample(frac=0.60, random_state=1)
    neg_a = neg_cases.sample(frac=0.15, random_state=1)
    df_subset.loc[pos_a.index, 'hospital'] = 'hospital_a'
    df_subset.loc[neg_a.index, 'hospital'] = 'hospital_a'

    # Hospital B: 20% Pneumonia, 80% Normal (General Clinic)
    pos_remaining = pos_cases.drop(pos_a.index)
    neg_remaining = neg_cases.drop(neg_a.index)

    pos_b = pos_remaining.sample(frac=0.25, random_state=2)
    neg_b = neg_remaining.sample(frac=0.75, random_state=2)
    df_subset.loc[pos_b.index, 'hospital'] = 'hospital_b'
    df_subset.loc[neg_b.index, 'hospital'] = 'hospital_b'

    # Hospital C: 50/50 Balanced (Research Hospital)
    pos_c = pos_remaining.drop(pos_b.index)
    neg_c = neg_remaining.drop(neg_b.index)
    min_len = min(len(pos_c), len(neg_c))
    pos_c = pos_c.sample(n=min_len, random_state=3)
    neg_c = neg_c.sample(n=min_len, random_state=3)
    df_subset.loc[pos_c.index, 'hospital'] = 'hospital_c'
    df_subset.loc[neg_c.index, 'hospital'] = 'hospital_c'

    # Test Set: Take remaining unassigned data
    test_set = df_subset[df_subset['hospital'] == 'unassigned'].sample(frac=0.5, random_state=4)
    df_subset.loc[test_set.index, 'hospital'] = 'test_set'

    hospitals = {
        "hospital_a": df_subset[df_subset['hospital'] == 'hospital_a'],
        "hospital_b": df_subset[df_subset['hospital'] == 'hospital_b'],
        "hospital_c": df_subset[df_subset['hospital'] == 'hospital_c'],
        "test_set": df_subset[df_subset['hospital'] == 'test_set']
    }

    print("🔄 Converting DICOM to PNG and generating manifests...")
    for name, data_df in hospitals.items():
        print(f"\n🏥 Processing {name} ({len(data_df)} images)...")
        img_dir = os.path.join(OUTPUT_DIR, name, "images")
        os.makedirs(img_dir, exist_ok=True)
        
        manifest = []
        for idx, row in data_df.iterrows():
            patient_id = row['patientId']
            dicom_path = os.path.join(RAW_DIR, f"{patient_id}.dcm")
            png_path = os.path.join(img_dir, f"{patient_id}.png")
            
            # Convert and save
            if not os.path.exists(png_path):
                success = dicom_to_png(dicom_path, png_path)
                if not success:
                    continue
            
            manifest.append({"filename": f"{patient_id}.png", "label": int(row['Target'])})
        
        # Save manifest CSV
        manifest_df = pd.DataFrame(manifest)
        manifest_df.to_csv(os.path.join(OUTPUT_DIR, name, "manifest.csv"), index=False)
        
        # Print skew statistics
        pos_count = manifest_df['label'].sum()
        neg_count = len(manifest_df) - pos_count
        print(f"   ✅ Saved {len(manifest_df)} images. Skew: {pos_count} Pneumonia, {neg_count} Normal.")

    print("\n🎉 Level 2 Complete! Data partitioning and conversion finished.")
    print("Check the 'data/' folder for hospital_a, hospital_b, hospital_c, and test_set.")

if __name__ == "__main__":
    main()