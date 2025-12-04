# LaTeX Reference Documents

This directory contains professional LaTeX versions of the Claude Code Plugin Marketplace documentation.

## Files

| File | Size | Description |
|------|------|-------------|
| **AGENTS_LIST.tex** | ~27KB | Complete catalog of all 74 agents with professional typesetting |
| **COMMANDS_LIST.tex** | ~22KB | Complete catalog of all 48 commands with beautiful formatting |
| **plugin-cheatsheet.tex** | ~5KB | Quick reference cheatsheet (landscape, 3-column layout) |
| **latex_compile.sh** | ~4KB | Compilation script for generating PDFs |

## Features

**Professional Typography**
- Modern LaTeX styling with lmodern font package
- Color-coded sections and headers (blue for agents, magenta for commands)
- Professional table formatting with longtable for multi-page tables

**Beautiful Design**
- Professional title pages with key statistics
- Hyperlinked table of contents and cross-references
- Syntax-highlighted code listings
- Icon support with FontAwesome

**Comprehensive Layout**
- Proper sectioning and subsectioning
- Professional headers and footers
- Long tables that span multiple pages correctly

## Compilation

### Requirements

Install a LaTeX distribution:

**macOS:**
```bash
brew install --cask mactex
# or for minimal installation
brew install --cask basictex
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install texlive-full
# or minimal
sudo apt-get install texlive-latex-base texlive-fonts-recommended \
    texlive-latex-extra texlive-fonts-extra
```

**Windows:**
Download and install [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)

### Required LaTeX Packages

The following LaTeX packages are used (usually included in full distributions):
- `lmodern` - Modern fonts
- `geometry` - Page layout
- `xcolor` - Color support
- `longtable` - Multi-page tables
- `booktabs` - Professional tables
- `hyperref` - Hyperlinks
- `fancyhdr` - Headers/footers
- `titlesec` - Section formatting
- `listings` - Code listings
- `tcolorbox` - Colored boxes
- `fontawesome5` - Icons
- `pifont` - Symbols (checkmark, cross)
- `multicol` - Multi-column layouts (cheatsheet)

### Compile to PDF

**Option 1: Using the compilation script (recommended)**
```bash
chmod +x latex_compile.sh
./latex_compile.sh
```

**Option 2: Manual compilation**
```bash
# Compile all three documents
for file in AGENTS_LIST COMMANDS_LIST plugin-cheatsheet; do
    pdflatex $file.tex
    pdflatex $file.tex  # Run twice for TOC
done
```

**Option 3: Using latexmk**
```bash
latexmk -pdf AGENTS_LIST.tex COMMANDS_LIST.tex plugin-cheatsheet.tex
```

### Output Files

After compilation, you'll get:
- `AGENTS_LIST.pdf` - Beautiful PDF of all 74 agents (~12 pages)
- `COMMANDS_LIST.pdf` - Beautiful PDF of all 48 commands (~10 pages)
- `plugin-cheatsheet.pdf` - Quick reference cheatsheet (1-2 pages, landscape)

### Clean Auxiliary Files

```bash
rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
```

## Document Structure

### AGENTS_LIST.pdf
1. Title page with key statistics
2. Table of contents
3. Scientific Computing & Specialized (18 agents)
4. Development (24 agents)
5. AI/ML (6 agents)
6. DevOps & Infrastructure (11 agents)
7. Code Quality & Tools (12 agents)
8. Orchestration & Research (3 agents)
9. Quick reference table by plugin
10. Usage examples
11. Naming conventions
12. Additional resources

### COMMANDS_LIST.pdf
1. Title page with key statistics
2. Table of contents
3. Scientific Computing (4 commands)
4. Development (11 commands)
5. AI/ML (1 command)
6. DevOps & Infrastructure (8 commands)
7. Quality & Testing (10 commands)
8. Tools & Migration (9 commands)
9. Orchestration & AI Reasoning (5 commands)
10. Quick reference table by plugin
11. Usage examples with workflows
12. Naming conventions
13. Command categories summary
14. Additional resources

### plugin-cheatsheet.pdf
- Single-page landscape quick reference
- 3-column layout with all 31 plugins
- Key agents, commands, and skill counts per plugin
- Installation instructions

## Customization

### Change Colors

Edit the color definitions in the preamble:

```latex
% For AGENTS_LIST.tex (blue theme)
\definecolor{primarycolor}{RGB}{0,102,204}      % Blue
\definecolor{secondarycolor}{RGB}{51,51,51}     % Dark gray
\definecolor{accentcolor}{RGB}{204,0,102}       % Magenta

% For COMMANDS_LIST.tex (magenta theme)
\definecolor{primarycolor}{RGB}{204,0,102}      % Magenta
\definecolor{secondarycolor}{RGB}{51,51,51}     % Dark gray
\definecolor{accentcolor}{RGB}{0,153,153}       % Teal
```

### Adjust Page Margins

Edit the geometry package options:

```latex
\usepackage[margin=1in]{geometry}
% Change to \usepackage[margin=0.75in]{geometry} for narrower margins
```

### Change Font Size

Modify the document class:

```latex
\documentclass[11pt,a4paper]{article}
% Options: 10pt, 11pt (default), 12pt
```

## Troubleshooting

**Problem:** Missing package errors

**Solution:** Install the missing packages:
```bash
# For TeX Live (Linux/Mac)
tlmgr install package-name

# For MiKTeX (Windows)
mpm --install package-name
```

**Problem:** Fonts not found

**Solution:** Install lmodern package:
```bash
tlmgr install lmodern
```

**Problem:** Icons not showing

**Solution:** Install fontawesome5:
```bash
tlmgr install fontawesome5
```

## Version Information

| Item | Value |
|------|-------|
| LaTeX Documents Version | 1.0.4 |
| Marketplace Version | 1.0.4 |
| Last Updated | December 3, 2025 |
| Total Agents | 74 |
| Total Commands | 48 |
| Total Skills | 114 |
| Total Plugins | 31 |

## Links

- **GitHub:** https://github.com/imewei/MyClaude
- **Documentation:** https://myclaude.readthedocs.io/en/latest/
- **Full Agents List:** AGENTS_LIST.md (root)
- **Full Commands List:** COMMANDS_LIST.md (root)
- **Plugin Cheatsheet:** PLUGIN_CHEATSHEET.md (root)
