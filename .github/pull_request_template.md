## What this adds

<!-- Module and functions, e.g. api-ms-win-crt-string: strcpy_s, strncpy_s, strcat_s -->

Closes #

## Windows behaviour

<!-- Anything non-obvious: where real Windows differs from its documentation or from glibc,
     and your source (docs page, test on Windows, Wine's tests...). -->

## AI assistance

<!-- None, or the tool you used (e.g. Claude Code, OpenCode, Copilot) and how you tested
     the result. -->

## Checklist

Maintainers check the legal items for every pull request and will ask you to remove and squash out anything that breaks them (see `CONTRIBUTING.md`).


- [ ] Each function is declared with `win32!`, under its normalized module name (see `docs/library.md`)
- [ ] `Native` only for functions that really work; anything partial is `Stub`
- [ ] Tests cover normal use **and** error cases (last error, return values, null pointers)
- [ ] Windows types and ABI: `extern "win64"`, 32-bit `LONG`/`long`, UTF-16 wide strings
- [ ] Formatted with the CI's toolchain; `cargo fmt --check`, `cargo clippy -- -D warnings` and `cargo test` pass locally
- [ ] No game files, console system files, dumps, shader blobs, logs or screenshots containing game data or strings, in the diff or in the history
- [ ] The game, its publisher and its executable are not named anywhere (code, comments, commits, this description)
- [ ] No code copied from proprietary SDKs or from GPL/LGPL projects (clean room: references read for facts only)
- [ ] Pure native: no Wine, Proton, DXVK, vkd3d(-proton), dxil-spirv or shim over them
- [ ] Memory: every cache is bounded or retired, no CPU copy kept after upload, staging freed or reused, GPU imports/images released with the title's memory; for memory paths, how it was measured is described above
- [ ] I claimed this work on an issue first (or opened one)
- [ ] AI use, if any, is mentioned above
- [ ] Code taken from another open-source project is credited and its license allows it
- [ ] Every commit is signed off (`git commit -s`): I certify the Developer Certificate of Origin (see `CONTRIBUTING.md`)
