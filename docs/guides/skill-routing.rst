Skill Loading and Routing
=========================

What loads when, and how to keep routing accurate and cheap.

What the model sees each session
--------------------------------

.. list-table::
   :header-rows: 1

   * - Item
     - In context
     - Loaded on demand
   * - Hub skill (42)
     - name and description
     - ``SKILL.md`` body when invoked
   * - Sub-skill (169)
     - nothing (``disable-model-invocation: true``)
     - read by path from a hub or agent
   * - Agent (27)
     - description
     - body on dispatch; no preloaded skills
   * - Command (21)
     - description
     - body when run

Claude Code discovers every ``skills/*/SKILL.md`` (the ``plugin.json`` ``skills`` list only adds paths) and
lists every model-invocable name against a shared budget of 1% of the context window. Sub-skills therefore opt
out, and hubs load them with Read on ``${CLAUDE_PLUGIN_ROOT}/skills/<name>/SKILL.md``. Agents name the skills
they use in a ``Related Skills`` section instead of preloading them, because a ``skills:`` preload injects the full
text into every new subagent instance.

The ``UserPromptSubmit`` hook of dev-suite and science-suite adds a one-line routing hint only when the prompt
matches a hub's domain pattern, and repeats it at most once per twelve prompts.

Writing descriptions that route
-------------------------------

- State the trigger (the user's words) and the boundary (``For X, use Y``). Look-alike entries must name each other.
- Never point a description at a hidden skill; name the hub that loads it.
- Keep each description under 1,536 characters; the listing drops descriptions first for the least-used skills.
- Agents state what they are *not* for and which peer takes it.

Checks
------

``make validate`` runs ``tools/validation/routing_checker.py`` (static rules above) and
``tools/validation/marketplace_validator.py``. Static checks cannot show that routing is good.
The 2026-10 prompt eval gave a router call the listing and 41 clear prompts: 100% correct with the 42-hub
listing versus 98% with all 211 skills visible, at 2.7x fewer listing tokens. A second set of 40 ambiguous
prompts scored 95% on Sonnet and 88% on Haiku with look-alike pointers, and 95% and 84% without them. Treat the
pointer benefit as unproven; the listing reduction is measured.
