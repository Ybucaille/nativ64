# nativ64

**Contributors:** `HANDOFF.md` and `docs/journal/` are the maintainer's local working notes and are not in the
repository. Work from `CONTRIBUTING.md`, the issue you took and the `docs/` page of the area you touch; ignore the rest
of this file.

Before resuming, read `HANDOFF.md` (state, rules, environment, next step): it is the reference.
At each finished step: **rewrite** `HANDOFF.md` (current state + next step, under ~150 lines; do not add a session
paragraph), write the session story at the end of `docs/journal/journal.md` (read only when needed), commit.
Agents: Sonnet = scoped implementation (spec + test); Opus = design, specs, review after each addition; escalate to
Opus only on failing tests or an obvious failure.
**Real validation at each milestone** (not each change): run the result and compare it with the project's oracle (see
`HANDOFF.md`, Environment), driving it yourself without waiting for the user to ask; tests are not enough.
