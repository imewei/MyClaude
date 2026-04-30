# Design: scientific-review --mode flag

**Date:** 2026-04-30
**Skill:** `research-suite:scientific-review`
**Goal:** Reduce token consumption by adding a `--mode` flag that gates how many reference files are loaded.

---

## Problem

The skill currently loads all three files on every invocation (~28,971 bytes / ~7,200 tokens), regardless of whether the user needs the full report template or the integrity checklist. Most reviews do not require both.

## Solution: Approach A — Mode gates reference file loading

Add a `--mode` flag. SKILL.md becomes a lean base (always loaded). Reference files are loaded conditionally based on mode.

| `--mode`      | Files loaded                                     | Use when                        |
|---------------|--------------------------------------------------|---------------------------------|
| `simple`      | SKILL.md only                                    | Quick triage, short papers      |
| `standard`    | SKILL.md + `references/review_structure.md`      | Typical review work (default)   |
| `comprehensive` | SKILL.md + both reference files               | Integrity-sensitive reviews     |

**Default:** `standard`

### Token estimates after changes

| Mode          | ~Bytes  | ~Tokens | vs. baseline |
|---------------|---------|---------|--------------|
| `simple`      | 4,200   | 1,050   | −85%         |
| `standard`    | 12,000  | 3,000   | −58%         |
| `comprehensive` | 17,500 | 4,375  | −39%         |
| current (always all) | 28,971 | 7,200 | baseline |

---

## Changes

### SKILL.md (10,489 → ~4,200 bytes)

1. **Add mode routing table** after frontmatter — maps `--mode` values to loaded files and use cases. Default is `standard`.

2. **Compress Phase 0** (reviewer responsibilities) — 3 bullet labels with one-line guidance each, replacing 3 full paragraphs.

3. **Compress Phase 1** (ingest) — 2-line cheatsheet: DOCX → `pandoc <path> -t markdown`; PDF → `pdftotext` then escalate to `pypdf`/`pymupdf`.

4. **Compress Phase 2** (journal adaptation) — 3-line checklist (structure / criteria / AI policy). Remove parenthetical journal examples.

5. **Compress Phase 3** (six lenses) — each lens: ~4 tight bullet points, removing narrative justification and sub-field examples. All six lenses still run in every mode.

6. **Compress Phase 4** (quantitative verification) — remove inline worked arithmetic example; keep the two verification modes (direct / flagging) as brief bullets.

7. **Compress Phase 5** (compose report) — 3 lines pointing to `review_structure.md` for the template. Move docx format spec (Arial 11pt, margins, metadata table) into `review_structure.md`.

8. **Compress Principles** — 7 bullets of ≤12 words each, replacing multi-sentence paragraphs.

### `references/review_structure.md` (11,711 → ~7,800 bytes)

1. **Receive docx format spec** moved from SKILL.md (Arial 11pt, margins, metadata table, color guidance).
2. **Cut tone examples** from 4 before/after pairs to 2. Keep: "Too harsh / Better" and "Too accusatory (integrity) / Better". Cut: "Too vague / Better" and "Too deferential / Better".
3. **Trim per-section guidance prose** ~20% — remove meta-commentary explaining why a section exists; keep the structural and content requirements.

### `references/integrity_checks.md` (6,771 → ~5,500 bytes)

1. **Remove final section** "How to write integrity concerns in the report" (lines 88–99) — restates guidance already in SKILL.md Phase 3 Lens 6 and Phase 5.
2. **Compress inline "how to flag" paragraphs** that end each category (plagiarism, data/image integrity, etc.) from full paragraphs to one-line each. General framing is covered in SKILL.md.

---

## Non-changes

- No files added or removed
- All six analysis lenses run in every mode
- Phase 0 (COI, confidentiality, expertise) runs in every mode
- Reference file architecture is unchanged; only prose within files is reduced
- `simple` mode uses SKILL.md as a complete self-contained guide (no detail is gated behind a mode that would cause silent omissions)
