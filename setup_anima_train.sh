#!/bin/bash

# ==========================================
# NOTBREMSE & FEHLER-LOGGING
# ==========================================
set -euo pipefail

script_error() {
    local exit_code=$?
    local line_number=$1
    echo "[ERROR] Provisioning Script fehlgeschlagen in Zeile $line_number mit Exit-Code $exit_code" | tee -a /workspace/provisioning_error.log
}
trap 'script_error $LINENO' ERR

echo "========================================"
echo "0. CONDA FÜR DIESES SKRIPT INITIALISIEREN"
echo "========================================"
# Suche nach der offiziellen Conda-Initialisierungsdatei (deckt alle gängigen Docker-Images ab)
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/root/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/usr/local/conda/etc/profile.d/conda.sh" ]; then
    source "/usr/local/conda/etc/profile.d/conda.sh"
elif [ -f "/opt/miniconda/etc/profile.d/conda.sh" ]; then
    source "/opt/miniconda/etc/profile.d/conda.sh"
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/miniconda3/etc/profile.d/conda.sh"
else
    echo "[FEHLER] Conda konnte auf diesem Image nicht gefunden werden!"
    exit 1
fi
echo "Conda erfolgreich initialisiert!"

echo "========================================"
echo "1. REPOSITORY KLONEN & ORDNER BAUEN"
echo "========================================"
cd /workspace/

# Nur klonen, wenn der Ordner nicht schon existiert
if [ ! -d "diffusion-pipe" ]; then
    git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe
fi

cd diffusion-pipe
mkdir -p project
mkdir -p model_weights

echo "========================================"
echo "2. CONDA UMGEBUNG ERSTELLEN"
echo "========================================"
conda create -n diffusion-pipe python=3.12 -y
conda activate diffusion-pipe

echo "========================================"
echo "3. CORE-ABHÄNGIGKEITEN INSTALLIEREN"
echo "========================================"
pip install torch torchvision
pip install -r requirements.txt
pip install "transformers<5.0"
pip install datasets requests pillow
# hf_transfer für maximale Download-Geschwindigkeiten installieren
pip install hf_transfer

echo "========================================"
echo "4. MODELLE DOWNLOADEN (Hugging Face CLI + HF Transfer)"
echo "========================================"
# Aktiviert den ultraschnellen Rust-Download-Modus
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p hf_temp

echo "Starte parallelen High-Speed-Download..."
huggingface-cli download circlestone-labs/Anima \
  split_files/diffusion_models/anima-base-v1.0.safetensors \
  split_files/text_encoders/qwen_3_06b_base.safetensors \
  split_files/vae/qwen_image_vae.safetensors \
  --local-dir hf_temp --local-dir-use-symlinks False

# Dateien flach in den model_weights Ordner verschieben
mv hf_temp/split_files/diffusion_models/anima-base-v1.0.safetensors model_weights/
mv hf_temp/split_files/text_encoders/qwen_3_06b_base.safetensors model_weights/
mv hf_temp/split_files/vae/qwen_image_vae.safetensors model_weights/

# Temporären Ordner aufräumen
rm -rf hf_temp

cd /workspace/diffusion-pipe/

echo "========================================"
echo "SETUP ABGESCHLOSSEN! BEREIT FÜRS TRAINING."
echo "========================================"

echo 'HINWEIS FÜR DAS TRAINING:'
echo 'Führe diese Befehle im Terminal aus, wenn du verbunden bist:'
echo 'conda activate diffusion-pipe'
echo 'NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config project/anima_train.toml --executable $(which python)'
