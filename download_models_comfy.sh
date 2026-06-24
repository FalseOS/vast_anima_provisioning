#!/bin/bash

# ==============================================================================
# CONFIGURATION: Füge hier deine HuggingFace-Modell-URLs ein
# ==============================================================================

# Checkpoints (z.B. SDXL, Pony, Flux, SD 1.5)
CHECKPOINTS=(
    "https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_raw_bf16.safetensors"
)

# VAEs
VAES=(
    "https://huggingface.co/Comfy-Org/Krea-2/blob/main/vae/qwen_image_vae.safetensors"
)

# Text Encoders / CLIPs (z.B. für Flux oder SD3)
TEXT_ENCODERS=(
    "https://huggingface.co/Comfy-Org/Krea-2/blob/main/text_encoders/qwen3vl_4b_bf16.safetensors"
)

# Loras (Optional)
LORAS=(
    # "https://huggingface.co/user/repo/blob/main/lora.safetensors"
)

# ControlNet (Optional)
CONTROLNETS=(
    # "https://huggingface.co/user/repo/blob/main/controlnet.safetensors"
)

# ==============================================================================
# DOWNLOAD LOGIC VIA NEW 'hf' CLI
# ==============================================================================

BASE_DIR="/workspace/ComfyUI/models"

download_hf_file() {
    local url=$1
    local target_dir=$2

    # Parsing der neuen oder alten HF-URL Struktur (blob oder resolve)
    if [[ $url =~ huggingface\.co/([^/]+/[^/]+)/(blob|resolve)/([^/]+)/(.*) ]]; then
        local repo="${BASH_REMATCH[1]}"
        local revision="${BASH_REMATCH[3]}"
        local file_path="${BASH_REMATCH[4]}"
        local filename=$(basename "$file_path")

        echo "🚀 [HF] Downloade via native hf cli: $filename"
        
        # Sicherstellen, dass der Zielordner existiert
        mkdir -p "$target_dir"

        # Der neue 2026 Befehl: lädt rasend schnell und speichert es direkt an Ort und Stelle
        hf download "$repo" "$file_path" --revision "$revision" --to "$target_dir/$filename"
        
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

    echo "✅ Alle Downloads mit der neuen HF-CLI erfolgreich abgeschlossen!"
}

main
