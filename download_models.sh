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
https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors -> diffusion_models
https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors -> diffusion_models

# VAEs
https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_video_vae_int8_convrot.safetensors -> vae
https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors -> vae

# Text Encoder (CLIP)
https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors -> text_encoders

# LoRa
https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors -> loras

EOM

# ==========================================
# 2. SCRIPT LOGIK (Ab hier nichts ändern)
# ==========================================
echo "Starte automatischen Model-Download mit 'hf'..."

# Gehe jede Zeile der Liste durch
echo "$MODELS_LIST" | while read -r line; do
    # Leerzeichen trimmen
    line=$(echo "$line" | xargs)

    # Leere Zeilen oder Kommentare überspringen
    if [[ -z "$line" || "$line" == \#* ]]; then
        continue
    fi

    # Überspringe Zeilen, die kein '->' enthalten (Sicherheitscheck)
    if [[ "$line" != *"->"* ]]; then
        continue
    fi

    echo "----------------------------------------"

    # URL und Target absolut sicher trennen (splittet exakt beim Pfeil)
    url="${line%%->*}"
    target="${line##*->}"

    # Nochmal trimmen, um Leerzeichen neben dem Pfeil zu entfernen
    url=$(echo "$url" | xargs)
    target=$(echo "$target" | xargs)
    
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

    # Herunterladen mit dem neuen hf-CLI (ohne veraltete Parameter)
    if [ -z "$filename" ]; then
        echo "Lade komplettes Repository '$repo_id' nach '$target_dir'..."
        hf download "$repo_id" --local-dir "$target_dir"
    else
        echo "Lade Datei '$filename' nach '$target_dir'..."
        hf download "$repo_id" "$filename" --local-dir "$target_dir"
    fi
done

echo "----------------------------------------"
echo "✅ Alle Downloads erfolgreich abgeschlossen!"
