#!/bin/bash
# Food Vision API — Mac Setup Script
# Run with: chmod +x setup_mac.sh && ./setup_mac.sh

set -e  # Exit on any error

echo ""
echo "🍽️  Food Vision API — Mac Setup"
echo "================================"
echo ""

# ── Step 1: Check for Homebrew ───────────────────────────────────────────────
echo "📦 Checking for Homebrew..."
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add brew to PATH for Apple Silicon Macs
    if [[ $(uname -m) == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo "✅ Homebrew already installed"
fi

# ── Step 2: Install Python 3.12 ─────────────────────────────────────────────
echo ""
echo "🐍 Installing Python 3.12..."
brew install python@3.12

# Make sure python3 and pip3 point to the right version
export PATH="$(brew --prefix python@3.12)/bin:$PATH"

echo "✅ Python version: $(python3.12 --version)"

# ── Step 3: Create a virtual environment ────────────────────────────────────
echo ""
echo "🔧 Creating virtual environment..."
python3.12 -m venv venv

echo "✅ Virtual environment created at ./venv"

# ── Step 4: Activate venv and install dependencies ──────────────────────────
echo ""
echo "📥 Installing dependencies (this may take a few minutes — TensorFlow is large)..."
source venv/bin/activate

pip install --upgrade pip --quiet
pip install -r requirements.txt

echo "✅ All dependencies installed"

# ── Step 5: Copy .env if it doesn't exist ───────────────────────────────────
echo ""
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "⚠️  ACTION REQUIRED: Add your Anthropic API key to .env"
    echo "   Open .env and set: ANTHROPIC_API_KEY=sk-ant-your-key-here"
    echo "   Get a key at: https://console.anthropic.com"
else
    echo "✅ .env already exists"
fi

# ── Step 6: Create storage directories ──────────────────────────────────────
mkdir -p storage/references storage/temp models
echo "✅ Storage directories created"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅  Setup complete!"
echo ""
echo "To start the API:"
echo ""
echo "  source venv/bin/activate"
echo "  uvicorn main:app --reload --port 8000"
echo ""
echo "API docs will be at: http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""