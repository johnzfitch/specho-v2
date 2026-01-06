#!/usr/bin/env bash
#
# SpecHO 45D Installation Script
# For omarchy / Arch Linux
#
# Usage:
#   ./install.sh              # Install to ~/.local
#   ./install.sh /opt/specho  # Install to custom path
#

set -e

INSTALL_DIR="${1:-$HOME/.local/share/specho}"
BIN_DIR="${2:-$HOME/.local/bin}"

echo "========================================"
echo "SpecHO 45D Installer"
echo "========================================"
echo "Install dir: $INSTALL_DIR"
echo "Binary dir:  $BIN_DIR"
echo ""

# Create directories
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"

# Copy package
echo "[1/4] Copying package..."
cp -r fingerprint_39d "$INSTALL_DIR/"
cp -r specHO_core "$INSTALL_DIR/"
cp specho_cli.py "$INSTALL_DIR/"
cp *.md "$INSTALL_DIR/" 2>/dev/null || true

# Create wrapper script
echo "[2/4] Creating CLI wrapper..."
cat > "$BIN_DIR/specho" << WRAPPER
#!/usr/bin/env bash
# SpecHO 45D CLI wrapper
exec python3 "$INSTALL_DIR/specho_cli.py" "\$@"
WRAPPER
chmod +x "$BIN_DIR/specho"

# Check dependencies
echo "[3/4] Checking dependencies..."
MISSING=""

python3 -c "import numpy" 2>/dev/null || MISSING="$MISSING numpy"
python3 -c "import sklearn" 2>/dev/null || MISSING="$MISSING scikit-learn"

if [ -n "$MISSING" ]; then
    echo "  Missing optional packages:$MISSING"
    echo "  Install with: pip install$MISSING --break-system-packages"
else
    echo "  Core dependencies: OK"
fi

# Optional heavy deps
echo ""
echo "  Optional (for full 45D):"
python3 -c "import spacy" 2>/dev/null && echo "    spacy: OK" || echo "    spacy: Not installed (Layer B fallback mode)"
python3 -c "import sentence_transformers" 2>/dev/null && echo "    sentence-transformers: OK" || echo "    sentence-transformers: Not installed (Layer A fallback mode)"

# Test
echo ""
echo "[4/4] Testing installation..."
"$BIN_DIR/specho" --help > /dev/null 2>&1 && echo "  Installation: SUCCESS" || echo "  Installation: FAILED"

echo ""
echo "========================================"
echo "Installation complete!"
echo ""
echo "Usage:"
echo "  specho 'Your text here'"
echo "  specho -f document.txt"
echo "  specho --analyze 'Deep analysis'"
echo "  specho --json 'JSON output'"
echo ""
echo "For full 45D features, install optional deps:"
echo "  pip install sentence-transformers spacy --break-system-packages"
echo "  python -m spacy download en_core_web_sm"
echo "========================================"
