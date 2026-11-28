# Contributing

Thanks for helping. Everything happens on GitHub: issues for questions and claims, pull
requests for code.

## Review rules (read first)

nativ64 ships no game files and never will. Users own and dump their own copy. Maintainers
check **every** pull request for the points below (the legal ones first); if one is broken we will ask you to
remove the content and to squash it out of the branch history (a later commit that deletes
it is not enough) before merging. A pull request that keeps it is closed.

- **Never name the game, its publisher or its executable**, in code, docs, tests, commit
  messages, issues, pull requests or comments. Write "the title".
- **Never commit** game files, console system files, dumps, extracted shader blobs,
  textures, audio, logs, screenshots or crash reports that contain game data or game
  strings. This covers issue and pull request attachments too. The only exception is the
  aggregate coverage chart in `docs/`.
- **Clean room.** No code copied from proprietary SDKs or leaked sources, and none from GPL
  or LGPL projects, Wine included: this project is MIT OR Apache-2.0. Read them for facts
  (behaviour, layouts, error codes), then write your own code and tests. Code from
  permissively licensed projects is fine when credited in the file and in `THIRD_PARTY.md`.
- Reimplement from public documentation and observed behaviour.
- **Pure native only.** No Wine, Proton, DXVK, vkd3d or vkd3d-proton, dxil-spirv, or any
  shim or dependency built over them. Reference projects are read for facts only.
- **Memory discipline.** RAM and VRAM frugality from the start: every cache has a bound
  or a retirement, no CPU copy is kept once uploaded, staging memory is freed or reused,
  and GPU imports and images are released when the title frees their memory. A pull
  request that touches memory paths says how memory was measured.

## Pick something

1. Open the import map (`docs/import-map.md` explains how to draw it) or browse the
   issues labelled `good first issue`.
2. **Claim before implementing**: comment on the issue, or open one, naming the functions
   you take, so two people don't write the same thing. Unclaimed work may be taken by
   someone else after a few days of silence.
3. Prefer a coherent group (say, `api-ms-win-crt-string` `strncpy_s`/`strcpy_s`/`strcat_s`)
   over scattered functions.

## Write it

- One Rust module per Windows module, under `crates/nativ64/src/`. A function is
  declared and registered in one place (see `docs/library.md`).
- Match Windows, not glibc: Windows x64 calling convention (`extern "win64"`), `LONG` is
  32 bits, wide strings are UTF-16, failures set the last error. When Windows
  documentation and real behaviour disagree, follow real behaviour and say so in a
  comment with your source.
- A function that only pretends to work is registered as `Stub`, never `Native`.
- Every `Native` function has tests, including its error cases.
- Take expected behaviour from real Windows, not from memory. The best public record is
  Wine's conformance tests (`dlls/kernel32/tests`, `dlls/ucrtbase/tests`,
  `dlls/msvcrt/tests`...), which Wine runs on real Windows: read the relevant ones and
  write our own tests for the same behaviours (error codes, edge cases). Do not copy
  their code: it is LGPL, which is not compatible with this project's license.

## Sign off your commits (DCO)

Every commit in a pull request carries a `Signed-off-by:` line with your name and email:

```
Signed-off-by: Your Name <you@example.com>
```

`git commit -s` adds it (`git rebase --signoff main` adds it to commits you already made).
By signing off you certify the [Developer Certificate of Origin 1.1](https://developercertificate.org/):
you wrote the change, or you have the right to submit it under this project's license, and
you understand it is public and recorded. In practice: the code is yours (or an AI tool's
output you reviewed), or it comes from a project whose license allows it, credited as
above, and nothing in it was copied from GPL/LGPL, proprietary or leaked sources.
CI checks that every commit of a pull request is signed off.

## Before opening a pull request

```bash
scripts/fmt.sh --check   # rustfmt of the stable toolchain, as in CI
cargo clippy -- -D warnings
cargo test
```

CI runs the same checks.

## Other rules

- English for code, comments, commits and issues.
- AI tools are welcome (this project uses them too): just mention the tool in the pull
  request and how you tested the result.
- Bug reports: attach logs only after removing anything that comes from the game (paths
  inside its package, strings, resource names). When in doubt, describe it instead.

By contributing, you agree that your work is licensed under MIT or Apache-2.0, like the
rest of the project, and your sign-off records that agreement.
