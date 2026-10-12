# BlochSim — agent skills

`AGENTS.md` states the project's rules. The skills listed here are the
procedures for the recurring tasks those rules govern: what to run, in what
order, and what has to hold before the task is finished.

Each skill is a directory under `.claude/skills/` holding a `SKILL.md`. Agent
runtimes that read `.claude/skills/` discover them automatically; this page is
the index for those that do not.

| Skill | Use it when |
|---|---|
| [`build-and-test`](.claude/skills/build-and-test/SKILL.md) | Compiling the C++ kernels and running the suite, including the GPU kernels on a machine with no GPU. |
| [`build-the-docs`](.claude/skills/build-the-docs/SKILL.md) | Building the documentation, adding a figure to an explanation page, or adding a gallery example. |
| [`add-a-sequence`](.claude/skills/add-a-sequence/SKILL.md) | Writing a new simulator, signal model or sequence operator. |
| [`cut-a-release`](.claude/skills/cut-a-release/SKILL.md) | Publishing a release to PyPI, or diagnosing a wheel build. |

Where a skill and `AGENTS.md` disagree, `AGENTS.md` governs, and the skill is
wrong and should be corrected.
