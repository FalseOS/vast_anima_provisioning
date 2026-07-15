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
source /venv/main/bin/activate

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

if [ ! -d "diffusion-pipe" ]; then
    git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe
fi

cd diffusion-pipe

# --- NEU: HOTFIX FÜR NEUERE PYTORCH VERSIONEN ---
echo "Patche diffusion-pipe für neueste PyTorch-Kompatibilität..."
sed -i 's/from torch._namedtensor_internals import check_serializing_named_tensor/check_serializing_named_tensor = lambda *args, **kwargs: None/g' utils/reduction.py
# ------------------------------------------------

mkdir -p project
mkdir -p model_weights

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
echo "3. CORE-ABHÄNGIGKEITEN & SMART PYTORCH INSTALL"
echo "========================================"
# Zuerst die requirements.txt installieren (installiert evtl. falsches Standard-Torch)
pip install -r requirements.txt
pip install "transformers<5.0"
pip install datasets requests pillow
pip install hf_transfer

# JETZT ERST überschreiben wir Torch gezielt mit der richtigen CUDA-Version.
# Dadurch hat pip keine Chance mehr, es heimlich zu überschreiben!
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)

echo "Gefundene GPU: $GPU_NAME"
echo "Installierter Treiber: $DRIVER_VER"

if [[ "$GPU_NAME" == *"5090"* ]] || [[ "$GPU_NAME" == *"5080"* ]] || [[ "$GPU_NAME" == *"5070"* ]]; then
    if (( $(echo "$DRIVER_VER < 570" | bc -l) )); then
        echo "🔥 Blackwell GPU mit älterem Treiber..."
        pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu126
    else
        echo "🔥 Blackwell GPU mit neuem Treiber..."
        # HINWEIS: Ältere PyTorch Versionen (wie 2.5.1) haben evtl. noch keine offiziellen cu130 Wheels. 
        # Weiche im Notfall hier auf cu124/cu126 aus.
        pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    fi
else
    echo "✅ Standard GPU erkannt..."
    pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
fi

# Der restliche Core-Stuff
pip install -r requirements.txt
pip install "transformers<5.0"
pip install datasets requests pillow
pip install hf_transfer

echo "========================================"
echo "4. MODELLE DOWNLOADEN (HF Transfer)"
echo "========================================"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p hf_temp

echo "Starte parallelen High-Speed-Download..."
hf download circlestone-labs/Anima \
  split_files/diffusion_models/anima-base-v1.0.safetensors \
  split_files/text_encoders/qwen_3_06b_base.safetensors \
  split_files/vae/qwen_image_vae.safetensors \
  --local-dir hf_temp

# Dateien in die richtigen Ordner flachen
mv hf_temp/split_files/diffusion_models/anima-base-v1.0.safetensors model_weights/
mv hf_temp/split_files/text_encoders/qwen_3_06b_base.safetensors model_weights/
mv hf_temp/split_files/vae/qwen_image_vae.safetensors model_weights/
rm -rf hf_temp

cd /workspace/diffusion-pipe/

echo "========================================"
echo "5. SIMPLES ALIAS-SETUP FÜR DEN START"
echo "========================================"
# Einfache Aliases ohne dynamische Parameter, die exakt das machen, was sie sollen.

echo "alias train-anima-offline='cd /workspace/diffusion-pipe && conda activate diffusion-pipe && NCCL_P2P_DISABLE=\"1\" NCCL_IB_DISABLE=\"1\" WANDB_MODE=\"offline\" deepspeed --num_gpus=1 train.py --deepspeed --config project/anima_train.toml'" >> ~/.bashrc

echo "alias train-anima-online='cd /workspace/diffusion-pipe && conda activate diffusion-pipe && NCCL_P2P_DISABLE=\"1\" NCCL_IB_DISABLE=\"1\" deepspeed --num_gpus=1 train.py --deepspeed --config project/anima_train.toml'" >> ~/.bashrc

echo "========================================"
echo "SETUP ABSOLUT KUGELSICHER ABGESCHLOSSEN!"
echo "========================================"
