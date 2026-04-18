#!/usr/bin/env bash
# latex_compile_check.sh
#
# Compiles a LaTeX file with pdflatex twice (to resolve references) and reports
# any undefined references or compilation errors.
#
# Usage:
#   bash latex_compile_check.sh path/to/05_formalism.tex
#
# Exit codes:
#   0: compilation succeeded with no undefined references
#   1: compilation failed
#   2: compiled but with undefined references or warnings worth surfacing

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <path-to-tex-file>" >&2
    exit 2
fi

TEX_FILE="$1"

if [ ! -f "$TEX_FILE" ]; then
    echo "error: $TEX_FILE not found" >&2
    exit 2
fi

if ! command -v pdflatex >/dev/null 2>&1; then
    echo "error: pdflatex not on PATH; install a TeX distribution" >&2
    exit 2
fi

DIR="$(dirname "$TEX_FILE")"
BASE="$(basename "$TEX_FILE" .tex)"

cd "$DIR"

# First pass
echo "=== pass 1 ==="
if ! pdflatex -interaction=nonstopmode -halt-on-error "$BASE.tex" >/dev/null 2>&1; then
    echo "compilation failed on pass 1; running again with output visible:" >&2
    pdflatex -interaction=nonstopmode "$BASE.tex" || true
    exit 1
fi

# Second pass for references
echo "=== pass 2 ==="
pdflatex -interaction=nonstopmode -halt-on-error "$BASE.tex" >/dev/null 2>&1 || {
    echo "compilation failed on pass 2" >&2
    exit 1
}

# Check log for undefined references or citations
LOG="$BASE.log"
UNDEF=$(grep -c "undefined" "$LOG" || true)
WARN=$(grep -c "Warning" "$LOG" || true)

if [ "$UNDEF" -gt 0 ]; then
    echo "warning: $UNDEF undefined reference(s) or citation(s):" >&2
    grep -n "undefined" "$LOG" >&2 || true
    exit 2
fi

if [ "$WARN" -gt 0 ]; then
    echo "note: $WARN LaTeX warning(s) in log (review $LOG if needed)"
fi

echo "compiled successfully: $DIR/$BASE.pdf"
exit 0
