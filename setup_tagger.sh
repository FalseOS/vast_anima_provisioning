#!/bin/bash
# setup_tagger.sh (Upgraded Production Version)

# Force script to exit immediately if any command fails, or if an unset variable is used
set -euo pipefail

echo "========================================="
echo " Starting Robust WD Tagger V3 Provisioning"
echo "========================================="

### Configuration ###
# Use /workspace if it exists (standard for Vast.ai persistent storage), fallback to home directory
WORKSPACE_DIR="${WORKSPACE:-/workspace}"
TARGET_DIR="${WORKSPACE_DIR}/wd_tagger"
MODEL_PATH="${TARGET_DIR}/model.onnx"
TAGS_PATH="${TARGET_DIR}/selected_tags.csv"

# Error reporting helper
script_error() {
    local exit_code=$?
    local line_number=$1
    echo "❌ [ERROR] Provisioning failed at line $line_number with exit code $exit_code"
}
trap 'script_error $LINENO' ERR

# 1. System Preparation
echo "[1/4] Updating system packages..."
apt-get update && apt-get install -y git wget nano python3-pip

# 2. Dependency Installation
echo "[2/4] Verifying and installing Python dependencies..."
pip install --upgrade pip
pip install huggingface_hub onnxruntime-gpu pillow numpy pandas

# 3. Smart Model Fetching (Idempotent)
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "[3/4] Checking model assets..."
if [ -f "$MODEL_PATH" ]; then
    echo "✓ model.onnx already exists. Skipping download."
else
    echo "Downloading model.onnx from Hugging Face..."
    huggingface-cli download SmilingWolf/wd-vit-large-tagger-v3 model.onnx --local-dir . --quiet
fi

if [ -f "$TAGS_PATH" ]; then
    echo "✓ selected_tags.csv already exists. Skipping download."
else
    echo "Downloading selected_tags.csv from Hugging Face..."
    huggingface-cli download SmilingWolf/wd-vit-large-tagger-v3 selected_tags.csv --local-dir . --quiet
fi

# 4. Injecting the Python Script
echo "[4/4] Writing optimized batch-tagging script..."
cat << 'EOF' > tagger.py
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as ort

def preprocess_image(image_path, target_size=448):
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        new_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        new_img.paste(img, ((target_size - img.size[0]) // 2, (target_size - img.size[1]) // 2))
        img_array = np.array(new_img, dtype=np.float32)
        img_array = img_array[:, :, ::-1]  # RGB to BGR
        return img_array
    except Exception as e:
        print(f"Skipping {image_path} due to error: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tagger.py <path_to_folder> [batch_size]")
        sys.exit(1)
        
    folder_path = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    
    # Resolve relative paths inside the execution directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.onnx")
    tags_path = os.path.join(base_dir, "selected_tags.csv")
    threshold = 0.35
    
    cuda_options = {"cudnn_conv_algo_search": "HEURISTIC"}
    session = ort.InferenceSession(
        model_path, 
        providers=[('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
    )
    
    tags_df = pd.read_csv(tags_path)
    input_name = session.get_inputs()[0].name
    
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
    
    if not image_files:
        print(f"Error: No valid images found in folder '{folder_path}'")
        return

    print(f"Found {len(image_files)} images. Processing batches...")

    for i in range(0, len(image_files), batch_size):
        batch_paths = image_files[i:i + batch_size]
        batch_images = []
        actual_paths = []
        
        for p in batch_paths:
            img_data = preprocess_image(p)
            if img_data is not None:
                batch_images.append(img_data)
                actual_paths.append(p)
                
        if not batch_images:
            continue
            
        input_data = np.stack(batch_images, axis=0)
        outputs = session.run(None, {input_name: input_data})
        batch_probs = outputs[0]
        
        for idx, probs in enumerate(batch_probs):
            img_path = actual_paths[idx]
            img_tags_df = tags_df.copy()
            img_tags_df['probability'] = probs
            
            general_tags = img_tags_df[(img_tags_df['category'] == 0) & (img_tags_df['probability'] > threshold)]
            general_tags = general_tags.sort_values(by='probability', ascending=False)
            
            tag_list = [str(row['name']).replace('_', ' ') for _, row in general_tags.iterrows()]
            tag_string = ", ".join(tag_list)
            
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(tag_string)
                
        print(f"Progress: {min(i + batch_size, len(image_files))}/{len(image_files)} completed.")

    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
EOF

chmod +x tagger.py

echo "========================================="
echo " Setup Complete! Your environment is ready."
echo " Execution command:"
echo " python3 ${TARGET_DIR}/tagger.py /path/to/your/images"
echo "========================================="
