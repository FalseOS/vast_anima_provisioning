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
# PARSING & DOWNLOAD LOGIC
# ==============================================================================

# Basis-Pfad zu ComfyUI (Vast.ai Standard)
BASE_DIR="/workspace/ComfyUI/models"

# Sicherstellen, dass das aktuellste huggingface_hub Paket installiert ist
ensure_hf_cli() {
    if ! command -v huggingface-cli &> /dev/null; then
        echo "🔄 Installiere/Aktualisiere huggingface_hub für High-Speed Downloads..."
        pip install -q -U "huggingface_hub[cli]"
    fi
}

# Funktion, die HuggingFace-URLs zerlegt und modern herunterlädt
download_hf_file() {
    local url=$1
    local target_dir=$2

    # Prüfen auf valide HuggingFace-URL
    if [[ $url =~ huggingface\.co/([^/]+/[^/]+)/blob/([^/]+)/(.*) ]]; then
        local repo="${BASH_REMATCH[1]}"
        local revision="${BASH_REMATCH[2]}"
        local file_path="${BASH_REMATCH[3]}"
        local filename=$(basename "$file_path")

        echo "🚀 Downloade: $filename via modern HF-CLI..."
        
        # Erstelle das Zielverzeichnis, falls es noch nicht existiert
        mkdir -p "$target_dir"

        # Der neue, saubere Befehl: Schreibt die Datei direkt als echte Datei 
        # in den Zielordner, behält die flache Struktur und nutzt maximale Bandbreite.
        huggingface-cli download "$repo" "$file_path" \
            --revision "$revision" \
            --local-dir "$target_dir" \
            --local-dir-use-symlinks False

        # Falls das Tool Unterordner miterstellt hat, ziehen wir die Datei eine Ebene höher
        if [ -f "$target_dir/$file_path" ]; then
            mv "$target_dir/$file_path" "$target_dir/$filename"
            # Alten, leeren Strukturbaum entfernen
            rm -rf "$target_dir/$(echo "$file_path" | cut -d'/' -f1)"
        fi
    else
        echo "⚠️ Ungültige HF-URL oder kein direkter Blob-Link: $url"
    fi
}

# Hauptprozess
main() {
    ensure_hf_cli

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

    echo "✅ Alle Downloads ohne Symlinks erfolgreich abgeschlossen!"
}

main
