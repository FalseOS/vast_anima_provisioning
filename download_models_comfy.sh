#!/bin/bash

# ==============================================================================
# CONFIGURATION: Füge hier deine HuggingFace-Modell-URLs ein
# ==============================================================================

# Checkpoints
CHECKPOINTS=(
    "https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_raw_bf16.safetensors"
)

# VAEs
VAES=(
    "https://huggingface.co/Comfy-Org/Krea-2/blob/main/vae/qwen_image_vae.safetensors"
)

# Text Encoders
TEXT_ENCODERS=(
    "https://huggingface.co/Comfy-Org/Krea-2/blob/main/text_encoders/qwen3vl_4b_bf16.safetensors"
)

# Loras
LORAS=()

# ControlNets
CONTROLNETS=()

# ==============================================================================
# DOWNLOAD LOGIC
# ==============================================================================

BASE_DIR="/workspace/ComfyUI/models"

download_hf_file() {
    local url=$1
    local target_dir=$2

    # Extrahiert Repo, Revision und Dateipfad aus der URL
    if [[ $url =~ huggingface\.co/([^/]+/[^/]+)/(blob|resolve)/([^/]+)/(.*) ]]; then
        local repo="${BASH_REMATCH[1]}"
        local revision="${BASH_REMATCH[3]}"
        local file_path="${BASH_REMATCH[4]}"
        local filename=$(basename "$file_path")

        echo "🚀 [HF] Downloade via hf download: $filename"
        mkdir -p "$target_dir"

        # Nutzung der korrekten Optionen laut Log: --local-dir und --revision
        hf download "$repo" "$file_path" --revision "$revision" --local-dir "$target_dir"
        
        # Falls hf download den Pfadbaum (z.B. main/subfolder/file) erstellt, 
        # holen wir die Datei direkt in den Zielordner und löschen den leeren Rest.
        if [ -f "$target_dir/$file_path" ]; then
            mv "$target_dir/$file_path" "$target_dir/$filename"
            rm -rf "$target_dir/$(echo "$file_path" | cut -d'/' -f1)"
        fi
    else
        echo "⚠️ Ungültige HF-URL: $url"
    fi
}

main() {
    # 1. Checkpoints
    for url in "${CHECKPOINTS[@]}"; do
        download_hf_file "$url" "$BASE_DIR/checkpoints"
    done

    # 2. VAEs
    for url in "${VAES[@]}"; do
        download_hf_file "$url" "$BASE_DIR/vae"
    done

    # 3. Text Encoders
    for url in "${TEXT_ENCODERS[@]}"; do
        download_hf_file "$url" "$BASE_DIR/text_encoders"
    done

    # 4. Loras
    for url in "${LORAS[@]}"; do
        download_hf_file "$url" "$BASE_DIR/loras"
    done

    # 5. ControlNets
    for url in "${CONTROLNETS[@]}"; do
        download_hf_file "$url" "$BASE_DIR/controlnet"
    done

    echo "✅ Alle Downloads erfolgreich abgeschlossen!"
}

main
