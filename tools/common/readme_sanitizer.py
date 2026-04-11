"""README probe sanitizer for the /team-assemble codebase-detection path.

This module implements the S1 prompt-injection safeguards documented in
``plugins/agent-core/commands/team-assemble.md`` Step 2.6b. The prompt
instructions in that file describe the rules; this module is their
executable reference implementation.

The detection path currently lives entirely in prompt instructions that
Claude executes at runtime, so there is nothing to invoke the sanitizer
directly yet. The fixtures in ``tools/tests/test_readme_safeguards.py``
exercise the sanitizer end-to-end today, and any future automation of
the detection path (Python helper, MCP tool, hook) should import
``sanitize_readme_probe`` rather than reimplement the rules.

Keep this module and the Step 2.6b safeguard section in sync. The
regression test ``TestTeamAssembleSafeguardsPresent`` asserts the
prompt still contains the rule headings; a mismatch means one side
drifted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

LanguageHint = Literal["latin", "non-latin", "mixed", "empty"]
"""Coarse script-family classification of probe input.

- ``latin``: predominantly ASCII-range text (English, most Western European languages)
- ``non-latin``: predominantly non-ASCII code points (CJK, Arabic, Cyrillic, Devanagari, …)
- ``mixed``: a blend — e.g., an English intro followed by Chinese
- ``empty``: no characters to classify
"""

Confidence = Literal["standard", "low", "very_low"]
"""Confidence that the probe should be trusted for auto-fill.

- ``standard``: latin script, passes all safeguards. Suitable for inline
  substitution (still wrapped and surfaced in the metadata block).
- ``low``: mixed script, passes safeguards. User should double-check.
- ``very_low``: non-latin script. The English-primary refusal patterns
  may miss injection attempts in the target language, so the value is
  emitted but the user MUST review before pasting.
"""

# Patterns that indicate a probe value is untrustworthy. Any match causes
# the sanitizer to refuse the probe and downgrade the placeholder to
# ``[intent]``. These are intentionally broad; false positives are cheap
# (user re-runs with ``--var``) while false negatives allow prompt
# injection into the team prompt.
_REFUSAL_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bignore\s+(?:all\s+)?previous\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+the\s+above\b", re.IGNORECASE),
    re.compile(r"\bforget\s+everything\b", re.IGNORECASE),
    re.compile(r"\bnew\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(?:a|an)\b", re.IGNORECASE),
    re.compile(r"^\s*system\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"<\s*/?\s*(?:system|user|assistant)\s*>", re.IGNORECASE),
    re.compile(r"<\|(?:im_start|im_end|endoftext)\|>", re.IGNORECASE),
    re.compile(r"^\s*###\s+", re.MULTILINE),
)

# Characters that would break the ``<untrusted_readme_excerpt>`` wrapper
# or reopen a code fence inside the team prompt. Applied after the
# refusal scan so that suspicious content is rejected even if the
# attacker escapes the individual characters.
_DANGEROUS_CHARS: Final[dict[str, str]] = {
    "`": "",
    "<": "&lt;",
    ">": "&gt;",
}

# Hard cap on the raw text length that can be wrapped into a team prompt.
# Matches the 300-character cap documented in Step 1.5 Tier 4.
MAX_PROBE_LENGTH: Final[int] = 300

# A probe with >= this fraction of non-ASCII characters is classified
# as non-latin (strict threshold — CJK prose is effectively 100%).
_NON_LATIN_THRESHOLD: Final[float] = 0.5

# A probe with between _MIXED_LOW and _NON_LATIN_THRESHOLD non-ASCII
# fraction is classified as mixed.
_MIXED_LOW: Final[float] = 0.1


def _classify_language(text: str) -> LanguageHint:
    """Classify a probe paragraph by script family (ASCII-ratio heuristic).

    Uses a cheap character-by-character check rather than a real NLP
    language detector: the caller only needs to know whether to trust
    English-primary refusal patterns, not the actual language.
    """
    if not text:
        return "empty"
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ratio = non_ascii / len(text)
    if ratio >= _NON_LATIN_THRESHOLD:
        return "non-latin"
    if ratio >= _MIXED_LOW:
        return "mixed"
    return "latin"


def _confidence_for(language_hint: LanguageHint) -> Confidence:
    """Map a language hint to the auto-fill confidence tier."""
    if language_hint == "non-latin":
        return "very_low"
    if language_hint == "mixed":
        return "low"
    return "standard"


@dataclass(frozen=True)
class SanitizedProbe:
    """Result of sanitizing a README probe paragraph.

    Attributes:
        value: Final text, already wrapped in ``<untrusted_readme_excerpt>``
            tags and safe to substitute into a team prompt. Empty string
            if the probe was refused.
        safe: ``True`` if the probe passed all safeguard checks.
        refused: ``True`` if a refusal trigger fired. Mutually exclusive
            with ``safe``.
        refusal_reason: Human-readable description of which pattern
            matched, or ``None`` if the probe was accepted.
        raw_length: Length of the input text before any processing.
        truncated: ``True`` if the input exceeded :data:`MAX_PROBE_LENGTH`
            and was cut down.
        language_hint: Coarse script-family classification of the input.
            Always computed, regardless of refusal state.
        confidence: Auto-fill confidence tier derived from
            ``language_hint``. ``very_low`` for non-latin scripts —
            callers MUST surface these with an explicit review prompt
            since the English-primary refusal patterns may miss
            injection attempts in the target language.
    """

    value: str
    safe: bool
    refused: bool
    refusal_reason: str | None
    raw_length: int
    truncated: bool
    language_hint: LanguageHint
    confidence: Confidence


def sanitize_readme_probe(text: str) -> SanitizedProbe:
    """Apply the Step 2.6b prompt-injection safeguards to a probe paragraph.

    The order of operations matches the spec exactly:

    1. Measure the raw length and truncate to :data:`MAX_PROBE_LENGTH`.
    2. Scan the truncated text for any refusal pattern. If one matches,
       return an empty refused result — the caller must fall through to
       ``[intent]``.
    3. Neutralize dangerous characters (backticks, angle brackets).
    4. Wrap the neutralized text in ``<untrusted_readme_excerpt>`` tags.

    Args:
        text: The paragraph extracted from a README by Tier 4 of the
            detection step. Expected to already have markdown syntax
            stripped and badges removed.

    Returns:
        A :class:`SanitizedProbe` describing whether the probe is safe
        to substitute and the final wrapped value.
    """
    raw_length = len(text)
    truncated = raw_length > MAX_PROBE_LENGTH
    working = text[:MAX_PROBE_LENGTH]

    # Language classification runs on the *truncated* text so callers
    # see the same fraction the sanitizer used for its own decisions.
    language_hint = _classify_language(working)
    confidence = _confidence_for(language_hint)

    for pattern in _REFUSAL_PATTERNS:
        if pattern.search(working):
            return SanitizedProbe(
                value="",
                safe=False,
                refused=True,
                refusal_reason=f"matched refusal pattern: {pattern.pattern}",
                raw_length=raw_length,
                truncated=truncated,
                language_hint=language_hint,
                confidence=confidence,
            )

    neutralized = working
    for src, dst in _DANGEROUS_CHARS.items():
        neutralized = neutralized.replace(src, dst)

    wrapped = f"<untrusted_readme_excerpt>{neutralized}</untrusted_readme_excerpt>"

    return SanitizedProbe(
        value=wrapped,
        safe=True,
        refused=False,
        refusal_reason=None,
        raw_length=raw_length,
        truncated=truncated,
        language_hint=language_hint,
        confidence=confidence,
    )
