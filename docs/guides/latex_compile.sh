#!/bin/bash

# LaTeX Compilation Script for Claude Code Documentation
# Compiles AGENTS_LIST.tex, COMMANDS_LIST.tex, and plugin-cheatsheet.tex to PDFs
# Version: 1.0.4

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if pdflatex is installed
check_latex() {
    if ! command -v pdflatex &> /dev/null; then
        print_error "pdflatex not found!"
        print_info "Please install a LaTeX distribution:"
        echo "  macOS: brew install --cask mactex"
        echo "  Ubuntu/Debian: sudo apt-get install texlive-full"
        echo "  Windows: Download MiKTeX from https://miktex.org/"
        exit 1
    fi
    print_success "pdflatex found: $(which pdflatex)"
}

# Compile a LaTeX file
compile_latex() {
    local texfile=$1
    local basename=${texfile%.tex}

    print_info "Compiling $texfile..."

    # First pass
    print_info "First pass (generating auxiliary files)..."
    pdflatex -interaction=nonstopmode -halt-on-error "$texfile" > /dev/null 2>&1 || {
        print_error "First compilation failed for $texfile"
        print_info "Check the log file: ${basename}.log"
        pdflatex -interaction=nonstopmode -halt-on-error "$texfile"
        exit 1
    }

    # Second pass (resolve references and TOC)
    print_info "Second pass (resolving references and TOC)..."
    pdflatex -interaction=nonstopmode -halt-on-error "$texfile" > /dev/null 2>&1 || {
        print_error "Second compilation failed for $texfile"
        print_info "Check the log file: ${basename}.log"
        pdflatex -interaction=nonstopmode -halt-on-error "$texfile"
        exit 1
    }

    print_success "Successfully compiled ${basename}.pdf"
}

# Clean auxiliary files
clean_aux() {
    print_info "Cleaning auxiliary files..."
    rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
    print_success "Auxiliary files cleaned"
}

# Main execution
main() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Claude Code Plugin Marketplace - LaTeX Compilation"
    echo "═══════════════════════════════════════════════════════════"
    echo ""

    # Check if LaTeX is installed
    check_latex
    echo ""

    # Compile AGENTS_LIST.tex
    if [ -f "AGENTS_LIST.tex" ]; then
        compile_latex "AGENTS_LIST.tex"
        echo ""
    else
        print_warning "AGENTS_LIST.tex not found, skipping..."
        echo ""
    fi

    # Compile COMMANDS_LIST.tex
    if [ -f "COMMANDS_LIST.tex" ]; then
        compile_latex "COMMANDS_LIST.tex"
        echo ""
    else
        print_warning "COMMANDS_LIST.tex not found, skipping..."
        echo ""
    fi

    # Compile plugin-cheatsheet.tex
    if [ -f "plugin-cheatsheet.tex" ]; then
        compile_latex "plugin-cheatsheet.tex"
        echo ""
    else
        print_warning "plugin-cheatsheet.tex not found, skipping..."
        echo ""
    fi

    # List generated PDFs
    print_info "Generated PDF files:"
    ls -lh *.pdf 2>/dev/null || print_warning "No PDF files found"
    echo ""

    # Ask if user wants to clean auxiliary files
    read -p "Clean auxiliary files? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        clean_aux
    else
        print_info "Keeping auxiliary files"
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════════"
    print_success "Compilation complete!"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
}

# Run main function
main
