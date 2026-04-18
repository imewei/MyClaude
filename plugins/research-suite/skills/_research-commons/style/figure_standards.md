# Figure standards

Defaults for all figures produced in the research-spark stack: Stage 6 validation plots, Stage 7 plan figures, and any exploratory plots that may end up in a manuscript. Venue overrides take precedence when submitting.

## Fonts

- **Default font family:** sans-serif that matches body text. For LaTeX documents using Computer Modern, match with `\usepackage{cmbright}` or use Helvetica via `\usepackage{helvet}`.
- **Minimum font size in published figures:** 7 pt at final printed size. After figure resizing for a venue, verify that 7 pt is not violated.
- **Axis labels, tick labels, legend:** all at the same size or very close. Hierarchies of size are distracting in figures.
- **Font in software:** matplotlib default (DejaVu Sans) is acceptable; prefer `Arial` or `Helvetica` if the target venue specifies. Set via:

```python
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})
```

## Figure sizes

Target sizes match common journal column widths; set these at figure creation so the final figure is not resized disproportionately:

| Purpose | Width (inches) | Width (mm) |
|---|---|---|
| Single-column (PRL, most physics) | 3.375 | 85 |
| 1.5-column (some journals) | 4.75 | 120 |
| Double-column / full-page | 7.0 | 178 |
| Nature single-column | 3.5 | 89 |
| Nature double-column | 7.2 | 183 |

Height adjusts per figure content; common aspect ratios 3:2, 4:3, or 1:1.

```python
fig, ax = plt.subplots(figsize=(3.375, 2.25))  # single-column PRL
```

Do not create a figure at 10 inches wide and downscale on submission; font sizes become inconsistent.

## Resolution and format

- **Vector format preferred:** PDF, SVG, or EPS for everything that is not a photographic image or a high-density raster (e.g., a 1024×1024 image map).
- **Raster format when required:** PNG for web, TIFF for journal submission. Minimum 300 dpi at final size; 600 dpi for line art.
- **Never** submit JPEG for scientific figures unless the venue explicitly requires it. JPEG compression introduces artifacts in line plots.

Save from matplotlib:
```python
fig.savefig("fig_name.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("fig_name.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
```

## Colormaps

**Default: perceptually uniform and colorblind-safe.**

- **Sequential data:** `viridis` (default), `plasma`, `cividis` (especially colorblind-safe)
- **Diverging data:** `RdBu_r`, `coolwarm`, `PuOr`
- **Qualitative / categorical:** `tab10`, `Dark2`, or custom sets verified with a colorblind checker

**Avoid:**
- `jet`, `rainbow`, `hsv`: not perceptually uniform, fail in grayscale, mislead the reader about relative magnitudes
- Red-green diverging schemes for audiences that may be colorblind (roughly 8% of men have red-green deficiency)

For categorical comparisons with four or more categories, use `Dark2` or pick colors with pairwise contrast verified. For two categories, blue and orange from `tab10` is a strong default.

Grayscale check: every figure should be interpretable when printed in black and white. Use line style (solid, dashed, dotted) and marker shape (circle, square, triangle) to distinguish series, not color alone.

## Line widths and markers

- **Data lines:** 1.0-1.5 pt. Thin enough to reveal structure, thick enough to read at final size.
- **Axes and tick marks:** 0.5-0.8 pt. Thinner than data lines so axes recede visually.
- **Marker size:** 4-6 pt. Open markers (circle, square) read well for error-bar plots; filled markers read better for scatter plots with many points.
- **Error bars:** cap width matching marker size; line width matching data line.

## Axes and annotations

- **Linear or log?** Pick based on the data's natural scale. Power laws live on log-log; exponential decays on semi-log-y. Do not force linear axes on multi-decade data.
- **Tick direction:** inward. Minor ticks optional but consistent across panels.
- **Scientific notation:** use for values below 0.01 or above 1000. Prefer `1.2 × 10^{-3}` over `0.0012` for readability.
- **Units:** always on axis labels, in square brackets or parentheses: `Time [s]` or `Time (s)`. Consistent style within a paper.

## Subfigures

- **Panel labels:** uppercase (A, B, C) or lowercase (a, b, c) per venue, in bold sans-serif, placed at top-left of each panel with consistent offset.
- **Label size:** 2 pt larger than axis labels, bold.
- **Alignment:** left edges of panels aligned; baselines of x-axis labels aligned where possible.

## Figure caption style

- **First sentence:** a declarative summary of the figure's key finding or content. Not "Plot of X vs Y."
- **Rest of caption:** describes panels and any non-obvious details; defines symbols and colors; notes sample sizes, error-bar meanings, and statistical tests.
- **Length:** as long as needed for reproducibility; aim for under 150 words for most figures.

## Checklist before sharing a figure

- [ ] Font size at least 7 pt at final size
- [ ] Figure width matches target column width
- [ ] Vector format used for line art
- [ ] Colormap is perceptually uniform and colorblind-safe
- [ ] Grayscale version is interpretable
- [ ] Axes labeled with units in consistent style
- [ ] Error bars explained in the caption
- [ ] Panel labels and alignment consistent across subfigures

## Venue overrides

When a venue specifies exact standards (Nature journals in particular), the venue wins. Document the override in a comment at the top of the figure-generating script or notebook so future revisions use the same style.
