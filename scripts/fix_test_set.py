import os
import pandas as pd
import shutil

# Configuration
DATA_DIR = "data"
HOSPITALS = ["hospital_a", "hospital_b"]
TEST_DIR = os.path.join(DATA_DIR, "test_set")
TEST_IMG_DIR = os.path.join(TEST_DIR, "images")
TEST_MANIFEST = os.path.join(TEST_DIR, "manifest.csv")

print("🔄 Analyzing current test set...")
test_df = pd.read_csv(TEST_MANIFEST)
print(f"Current test set: {len(test_df)} images, {test_df['label'].sum()} Pneumonia.")

pneumonia_to_move = []

# We will take 50 Pneumonia cases from Hospital A and 50 from Hospital B
for hosp in HOSPITALS:
    hosp_manifest = os.path.join(DATA_DIR, hosp, "manifest.csv")
    hosp_df = pd.read_csv(hosp_manifest)
    
    # Get pneumonia cases (label == 1)
    pos_cases = hosp_df[hosp_df['label'] == 1]
    
    # Take up to 50 from each hospital
    to_move = pos_cases.sample(n=min(50, len(pos_cases)), random_state=42)
    pneumonia_to_move.append(to_move)
    
    # Remove them from the hospital's manifest
    updated_hosp_df = hosp_df.drop(to_move.index)
    updated_hosp_df.to_csv(hosp_manifest, index=False)
    print(f"    Removed {len(to_move)} Pneumonia cases from {hosp}.")

# Combine the moved cases
moved_df = pd.concat(pneumonia_to_move)

print("🔄 Moving image files to test_set...")
# Move the actual PNG files
for idx, row in moved_df.iterrows():
    for hosp in HOSPITALS:
        src = os.path.join(DATA_DIR, hosp, "images", row['filename'])
        if os.path.exists(src):
            dst = os.path.join(TEST_IMG_DIR, row['filename'])
            shutil.move(src, dst)
            break

# Update the test set manifest
new_test_df = pd.concat([test_df, moved_df[['filename', 'label']]], ignore_index=True)
new_test_df.to_csv(TEST_MANIFEST, index=False)

print(f"\n✅ Test set fixed!")
print(f"New test set size: {len(new_test_df)} images.")
print(f"New test set skew: {new_test_df['label'].sum()} Pneumonia, {len(new_test_df) - new_test_df['label'].sum()} Normal.")