# Beautiful LaTeX Documentation

This directory contains professional LaTeX versions of the Claude Code Plugin Marketplace documentation.

## Files

- **AGENTS_LIST.tex** - Complete catalog of all 74 agents with professional typesetting
- **COMMANDS_LIST.tex** - Complete catalog of all 60+ commands with beautiful formatting
- **latex_compile.sh** - Compilation script for generating PDFs

## Features

✨ **Professional Typography**
- Modern LaTeX styling with lmodern font package
- Color-coded sections and headers
- Professional table formatting with longtable for multi-page tables

🎨 **Beautiful Design**
- Custom color scheme (blue for agents, magenta for commands)
- Professional title pages with key statistics
- Hyperlinked table of contents and cross-references
- Syntax-highlighted code listings

📖 **Comprehensive Layout**
- Proper sectioning and subsectioning
- Professional headers and footers
- Long tables that span multiple pages correctly
- Icon support with FontAwesome

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

### Compile to PDF

**Option 1: Using the compilation script**
```bash
chmod +x latex_compile.sh
./latex_compile.sh
```

**Option 2: Manual compilation**

For AGENTS_LIST.pdf:
```bash
pdflatex AGENTS_LIST.tex
pdflatex AGENTS_LIST.tex  # Run twice for TOC
```

For COMMANDS_LIST.pdf:
```bash
pdflatex COMMANDS_LIST.tex
pdflatex COMMANDS_LIST.tex  # Run twice for TOC
```

**Option 3: Using latexmk (recommended)**
```bash
latexmk -pdf AGENTS_LIST.tex
latexmk -pdf COMMANDS_LIST.tex
```

**Option 4: Compile both at once**
```bash
for file in AGENTS_LIST COMMANDS_LIST; do
    pdflatex $file.tex
    pdflatex $file.tex
done
```

### Output Files

After compilation, you'll get:
- `AGENTS_LIST.pdf` - Beautiful PDF of all agents (~20 pages)
- `COMMANDS_LIST.pdf` - Beautiful PDF of all commands (~15 pages)

### Clean Auxiliary Files

```bash
rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
```

## Document Structure

### AGENTS_LIST.pdf
1. Title page with key statistics
2. Table of contents
3. Scientific Computing & Specialized (17 agents)
4. Development (22 agents)
5. AI/ML (5 agents)
6. DevOps & Infrastructure (10 agents)
7. Code Quality & Tools (14 agents)
8. Orchestration & AI Reasoning (5 agents)
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

## Customization

### Change Colors

Edit the color definitions in the preamble:

```latex
% For AGENTS_LIST.tex
\definecolor{primarycolor}{RGB}{0,102,204}      % Blue
\definecolor{secondarycolor}{RGB}{51,51,51}     % Dark gray
\definecolor{accentcolor}{RGB}{204,0,102}       % Magenta

% For COMMANDS_LIST.tex
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
# For MiKTeX (Windows)
mpm --install package-name

# For TeX Live (Linux/Mac)
tlmgr install package-name
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

- **LaTeX Documents Version:** 1.0.1
- **Marketplace Version:** 1.0.1
- **Last Updated:** October 31, 2025
- **Total Agents:** 73
- **Total Commands:** 48
- **Total Plugins:** 31

## License

These LaTeX documents are part of the Claude Code Plugin Marketplace documentation.

## Links

- **GitHub:** https://github.com/imewei/MyClaude
- **Documentation:** https://myclaude.readthedocs.io/en/latest/
- **Plugin Validation Report:** PLUGIN_LINT_REPORT.md
