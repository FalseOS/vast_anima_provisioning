#!/bin/bash

# Log file setup
LOG_FILE="installation_log.txt"
echo "Installation started at $(date)" > $LOG_FILE

echo "========================================"
echo "1. REPOSITORY KLONEN & ORDNER BAUEN"
echo "========================================"
cd /workspace/
git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe
cd diffusion-pipe
mkdir project
mkdir model_weights

echo "========================================"
echo "2. CONDA UMGEBUNG ERSTELLEN"
echo "========================================"
conda create -n diffusion-pipe python=3.12 -y
source activate diffusion-pipe

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

# Temporärer Ordner, da die CLI standardmäßig die Ordnerstruktur des Repos spiegelt
mkdir -p hf_temp

echo "Starte parallelen High-Speed-Download..."
huggingface-cli download circlestone-labs/Anima \
  split_files/diffusion_models/anima-base-v1.0.safetensors \
  split_files/text_encoders/qwen_3_06b_base.safetensors \
  split_files/vae/qwen_image_vae.safetensors \
  --local-dir hf_temp --local-dir-use-symlinks False

# Dateien flach in den model_weights Ordner verschieben (wie es vorher bei wget war)
mv hf_temp/split_files/diffusion_models/anima-base-v1.0.safetensors model_weights/
mv hf_temp/split_files/text_encoders/qwen_3_06b_base.safetensors model_weights/
mv hf_temp/split_files/vae/qwen_image_vae.safetensors model_weights/

# Temporären Ordner aufräumen
rm -rf hf_temp

cd /workspace/diffusion-pipe/

echo "========================================"
echo "SETUP ABGESCHLOSSEN! BEREIT FÜRS TRAINING."
echo "========================================"

# DEIN CHEAT-SHEET TRAININGSBEFEHL (Kopieren und manuell ausführen):
# NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config project/anima_train.toml --executable $(which python)
