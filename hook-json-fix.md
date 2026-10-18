# Prompt-submit hook JSON fix

The dev-suite and science-suite `UserPromptSubmit` hooks emitted unsupported top-level `status` and `additionalContext` fields. Codex rejects unknown fields in this event's JSON output, producing `hook returned invalid user prompt submit JSON output`.

References: [official hook documentation](https://developers.openai.com/codex/hooks), [Codex 0.163.0 output schema](https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/hooks/schema/generated/user-prompt-submit.command.output.schema.json).

Both repository hooks and their installed copies under `~/.codex/plugins/cache/scientific-computing-workflows/{dev-suite,science-suite}/4.0.6/hooks/` now emit only nested `hookSpecificOutput` for routing hints, or `{}` when no hint is needed. Exceptions retain stderr diagnostics and emit `{}` to stdout.

Updated existing payload tests and added `tools/tests/test_prompt_hook_errors.py` to cover exception output. The updated tests reproduced 14 failures before the fix; all passed afterward.

Validation:

- Full repository suite: **450 passed**.
- Ruff on all changed Python files: passed.
- Four direct checks of installed hooks passed against Codex 0.163.0's allowed output fields and types, covering routing and unrelated prompts for both suites.
- Installed hook files exactly match the corrected repository sources.

No interactive Codex UI submission was performed as part of verification. Plugin reinstall or upgrade can overwrite the installed cache changes until the source fix is included in the distributed plugin version.
