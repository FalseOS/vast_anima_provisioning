#!/bin/bash

# ==========================================
# 1. KONFIGURATION
# ==========================================
# Passe den Pfad zu deinem ComfyUI Ordner an (ohne / am Ende)
COMFYUI_DIR="/workspace/ComfyUI"

# Füge hier deine Links ein. Das Format ist immer:
# "URL -> ZIELORDNER_IN_MODELS"
# (Der Pfeil "->" trennt den Link vom Zielordner)

read -r -d '' MODELS_LIST << EOM

# Checkpoints / Base Models
https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors -> DiffusionModels
https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors -> DiffusionModels

# VAEs
https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors -> VAE

# Text Encoder (CLIP)
https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors -> TextEncoders

# Ganze Repositories funktionieren auch (lädt alle Dateien im Repo herunter):
# https://huggingface.co/lcm-models/lcm-sdxl -> unet

EOM

# ==========================================
# 2. SCRIPT LOGIK (Ab hier nichts ändern)
# ==========================================
echo "Starte automatischen Model-Download..."

# Gehe jede Zeile der Liste durch
echo "$MODELS_LIST" | while IFS="->" read -r url target; do
    # Leerzeichen an den Rändern entfernen
    url=$(echo "$url" | xargs)
    target=$(echo "$target" | xargs)

    # Leere Zeilen oder Kommentare (die mit # beginnen) überspringen
    if [[ -z "$url" || "$url" == \#* ]]; then
        continue
    fi

    echo "----------------------------------------"
    echo "Verarbeite: $url"

    # URL zerlegen, um Repo-ID und Dateiname für HF CLI zu bekommen
    path=${url#*huggingface.co/}
    IFS='/' read -r user repo rest <<< "$path"
    repo_id="$user/$repo"

    # Dateiname extrahieren (alles nach resolve/main/ oder blob/main/)
    filename=$(echo "$rest" | sed -E 's/^(resolve|blob|tree)\/[^\/]+\///')

    # Zielordner zusammensetzen und erstellen (falls nicht existent)
    target_dir="$COMFYUI_DIR/models/$target"
    mkdir -p "$target_dir"

    # Herunterladen mit huggingface-cli
    if [ -z "$filename" ]; then
        echo "Lade komplettes Repository '$repo_id' nach '$target'..."
        hf download "$repo_id" --local-dir "$target_dir" --local-dir-use-symlinks False
    else
        echo "Lade Datei '$filename' nach '$target'..."
        hf download "$repo_id" "$filename" --local-dir "$target_dir" --local-dir-use-symlinks False
    fi
done

echo "----------------------------------------"
echo "✅ Alle Downloads erfolgreich abgeschlossen!"
