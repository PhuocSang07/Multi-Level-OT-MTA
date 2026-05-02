#!/bin/bash
# Install all dependencies for Multi-Level-OT-MTA
# Requires Python 3.10+ and CUDA 12.x

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create virtual environment if it doesn't exist
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/.venv"
fi

source "$SCRIPT_DIR/.venv/bin/activate"
echo "[1/5] Virtual environment: $SCRIPT_DIR/.venv"

# Upgrade pip
pip install --upgrade pip --quiet

# PyTorch (CUDA 12.1) — change index URL if using a different CUDA version
echo "[2/5] Installing PyTorch 2.x (CUDA 12.1)..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121 --quiet

# Core training dependencies
echo "[3/5] Installing core training libraries..."
pip install \
    transformers==5.7.0 \
    peft==0.19.1 \
    datasets==4.8.5 \
    accelerate \
    tqdm \
    wandb \
    numpy \
    --quiet

# Evaluation / NLP utilities
echo "[4/5] Installing evaluation and NLP utilities..."
pip install \
    rouge-score \
    bert-score \
    evaluate \
    nltk \
    scikit-learn \
    pandas \
    pyarrow \
    --quiet

# spaCy + English model (needed for phrase-span extraction in MTA)
echo "[5/5] Installing spaCy and English model..."
pip install spacy==3.8.14 --quiet
python -m spacy download en_core_web_sm --quiet

echo ""
echo "Done! Activate with: source $SCRIPT_DIR/.venv/bin/activate"
echo "Run training with:   bash run_qwen1.5_1.8B_to_gpt2_120M_mta.sh"
