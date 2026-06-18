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
echo "0. CONDA ÜBER VENV/MAIN AKTIVIEREN"
echo "========================================"
# 1. Das offizielle Environment aktivieren
source /venv/main/bin/activate

# 2. Conda-Befehle für die non-interactive Shell verfügbar machen
if [ -f "/venv/etc/profile.d/conda.sh" ]; then
    source "/venv/etc/profile.d/conda.sh"
elif CONDA_SH=$(find /venv -name "conda.sh" -path "*/profile.d/conda.sh" 2>/dev/null | head -n 1); [ -n "$CONDA_SH" ]; then
    source "$CONDA_SH"
else
    export PATH="/venv/main/bin:/venv/bin:$PATH"
fi
echo "Conda erfolgreich geladen!"

echo "========================================"
echo "1. REPOSITORY KLONEN & CONFIGS LADEN"
echo "========================================"
cd /workspace/

# Nur klonen, wenn der Ordner nicht schon existiert
if [ ! -d "diffusion-pipe" ]; then
    git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe
fi

cd diffusion-pipe
mkdir -p project
mkdir -p model_weights

# HIER WERDEN DEINE CONFIGS AUS DEINEM REPO GEZOGEN
echo "Lade deine Config-Dateien herunter..."
wget -q -O project/anima_dataset.toml https://raw.githubusercontent.com/FalseOS/vast_anima_provisioning/refs/heads/main/anima_dataset.toml
wget -q -O project/anima_train.toml https://raw.githubusercontent.com/FalseOS/vast_anima_provisioning/refs/heads/main/anima_train.toml
echo "Configs erfolgreich in 'project/' abgelegt!"

echo "========================================"
echo "2. CONDA UMGEBUNG ERSTELLEN"
echo "========================================"
conda create -n diffusion-pipe python=3.12 -y
conda activate diffusion-pipe

echo "========================================"
echo "3. CORE-ABHÄNGIGKEITEN INSTALLIEREN"
echo "========================================"
# torchaudio hier direkt fest integriert!
pip install torch torchvision torchaudio 
pip install -r requirements.txt
pip install "transformers<5.0"
pip install datasets requests pillow
# hf_transfer für maximale Download-Geschwindigkeiten installieren
pip install hf_transfer

echo "========================================"
echo "4. MODELLE DOWNLOADEN (Hugging Face CLI + HF Transfer)"
echo "========================================"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p hf_temp

echo "Starte parallelen High-Speed-Download..."
hf download circlestone-labs/Anima \
  split_files/diffusion_models/anima-base-v1.0.safetensors \
  split_files/text_encoders/qwen_3_06b_base.safetensors \
  split_files/vae/qwen_image_vae.safetensors \
  --local-dir hf_temp --local-dir-use-symlinks False

# Dateien verschieben
mv hf_temp/split_files/diffusion_models/anima-base-v1.0.safetensors model_weights/
mv hf_temp/split_files/text_encoders/qwen_3_06b_base.safetensors model_weights/
mv hf_temp/split_files/vae/qwen_image_vae.safetensors model_weights/

# Aufräumen
rm -rf hf_temp

cd /workspace/diffusion-pipe/

echo "========================================"
echo "SETUP ABGESCHLOSSEN! BEREIT FÜRS TRAINING."
echo "========================================"

echo 'HINWEIS FÜR DAS TRAINING:'
echo 'Führe diese Befehle im Terminal aus, wenn du verbunden bist:'
echo 'conda activate diffusion-pipe'
echo 'NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" WANDB_MODE="online" deepspeed --num_gpus=1 train.py --deepspeed --config project/anima_train.toml'
echo 'Oder wenn du kein API key hinzugefügt hast:'
echo 'NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" WANDB_MODE="offline" deepspeed --num_gpus=1 train.py --deepspeed --config project/anima_train.toml'
