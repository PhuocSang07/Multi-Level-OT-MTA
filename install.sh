#!/bin/bash
# Install all dependencies for Multi-Level-OT-MTA
# Requires Python 3.10+ and a CUDA-capable GPU

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/.venv"
fi
source "$SCRIPT_DIR/.venv/bin/activate"
echo "[1/5] Virtual environment: $SCRIPT_DIR/.venv"
pip install --upgrade pip --quiet

# ── Detect CUDA version → pick PyTorch index URL ─────────────────────────────
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
if [ -z "$CUDA_VER" ]; then
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
fi

CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

if   [ "$CUDA_MAJOR" -ge 13 ]; then TORCH_CUDA="cu130"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then TORCH_CUDA="cu124"
elif [ "$CUDA_MAJOR" -eq 12 ]; then TORCH_CUDA="cu121"
elif [ "$CUDA_MAJOR" -eq 11 ]; then TORCH_CUDA="cu118"
else
    echo "WARNING: Could not detect CUDA version, installing CPU torch."
    TORCH_CUDA="cpu"
fi

TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"
echo "[2/5] Installing PyTorch (CUDA ${CUDA_VER:-unknown} → ${TORCH_CUDA})..."
pip install torch torchvision --index-url "$TORCH_INDEX" --quiet

# ── Core training dependencies ────────────────────────────────────────────────
echo "[3/5] Installing core training libraries..."
pip install \
    transformers \
    peft \
    datasets \
    accelerate \
    tqdm \
    wandb \
    numpy \
    --quiet

# ── Evaluation / NLP utilities ────────────────────────────────────────────────
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

# ── spaCy + English model (phrase-span extraction in MTA) ────────────────────
echo "[5/5] Installing spaCy and English model..."
pip install spacy --quiet
python -m spacy download en_core_web_sm --quiet

echo ""
echo "Done! Activate with: source $SCRIPT_DIR/.venv/bin/activate"
echo "Run training with:   bash run.sh"
