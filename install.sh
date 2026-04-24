#!/usr/bin/env bash
# Allma installer
set -e

ALLMA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ALLMA_DIR/venv"
BIN="$HOME/.local/bin"

echo ""
echo "  ██████╗     Allma installer"
echo "  ██╔══██╗    Local LLM Manager"
echo "  ███████║    "
echo "  ██╔══██║    "
echo "  ██║  ██║    "
echo "  ╚═╝  ╚═╝    "
echo ""

# ── Prerequisites ───────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python 3.10+ required (found $PY_VERSION)."
    exit 1
fi

echo "  Python $PY_VERSION — OK"

if ! command -v nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. Allma requires an NVIDIA GPU with CUDA drivers."
fi

# ── Virtual environment ─────────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv "$VENV"
else
    echo "  Virtual environment exists — skipping"
fi

echo "  Installing Python dependencies..."
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$ALLMA_DIR/requirements.txt"
echo "  Dependencies installed — OK"

# ── .env ────────────────────────────────────────────────────────────────────────
if [ ! -f "$ALLMA_DIR/.env" ]; then
    cp "$ALLMA_DIR/.env.example" "$ALLMA_DIR/.env"
    echo "  Created .env from .env.example — review and adjust if needed"
else
    echo "  .env exists — skipping"
fi

# ── allma command ───────────────────────────────────────────────────────────────
mkdir -p "$BIN"
ALLMA_BIN="$BIN/allma"

cat > "$ALLMA_BIN" <<EOF
#!/usr/bin/env bash
exec "$VENV/bin/python" "$ALLMA_DIR/allma_cli.py" "\$@"
EOF
chmod +x "$ALLMA_BIN"
echo "  Installed allma command → $ALLMA_BIN"

# Check PATH
if [[ ":$PATH:" != *":$BIN:"* ]]; then
    echo ""
    echo "  NOTE: $BIN is not in your PATH."
    echo "  Add this to your ~/.bashrc or ~/.zshrc:"
    echo ""
    echo '    export PATH="$HOME/.local/bin:$PATH"'
    echo ""
    echo "  Then run:  source ~/.bashrc"
fi

# ── Detect backends ─────────────────────────────────────────────────────────────
echo ""
echo "  Checking backends..."

LLAMA_OK=false
if command -v llama-server &>/dev/null; then
    echo "  llama-server — found on PATH ($(which llama-server))"
    LLAMA_OK=true
elif [ -f "$HOME/llama.cpp/build/bin/llama-server" ]; then
    echo "  llama-server — found at ~/llama.cpp/build/bin/llama-server"
    LLAMA_OK=true
elif [ -f "$HOME/AI/llama.cpp/build/bin/llama-server" ]; then
    echo "  llama-server — found at ~/AI/llama.cpp/build/bin/llama-server"
    LLAMA_OK=true
else
    echo "  llama-server — NOT found (see README for install instructions)"
fi

VLLM_OK=false
if "$VENV/bin/python" -c "import vllm" &>/dev/null 2>&1; then
    VLLM_VER=$("$VENV/bin/python" -c "import vllm; print(vllm.__version__)" 2>/dev/null)
    echo "  vllm $VLLM_VER — found in venv"
    VLLM_OK=true
elif command -v vllm &>/dev/null; then
    echo "  vllm — found on PATH ($(which vllm))"
    VLLM_OK=true
else
    echo "  vllm — NOT found (see README for install instructions)"
fi

# ── Summary ─────────────────────────────────────────────────────────────────────
echo ""
echo "  ─────────────────────────────────────────"
echo "  Installation complete!"
echo ""
echo "  Next steps:"
echo ""

if [ "$LLAMA_OK" = false ] || [ "$VLLM_OK" = false ]; then
    echo "  1. Install the backend(s) you want to use:"
    if [ "$VLLM_OK" = false ]; then
        echo "     vLLM:      $VENV/bin/pip install vllm"
    fi
    if [ "$LLAMA_OK" = false ]; then
        echo "     llama.cpp: see README — requires building from source"
    fi
    echo ""
    echo "  2. Create your base model configs:"
else
    echo "  1. Create your base model configs:"
fi

echo "     cp configs/base/Qwen3.6-27B-FP8.allm.example configs/base/MyModel.allm"
echo "     # Edit MyModel.allm and set the correct model path"
echo ""
echo "  Then start allma:"
echo "     allma serve"
echo "     allma list"
echo "     allma run <profile-name>"
echo ""
echo "  Full documentation: README.md"
echo "  ─────────────────────────────────────────"
echo ""
