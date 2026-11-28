# nativ64

[![CI](https://github.com/Ybucaille/nativ64/actions/workflows/ci.yml/badge.svg)](https://github.com/Ybucaille/nativ64/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![Good first issues](https://img.shields.io/github/issues/Ybucaille/nativ64/good%20first%20issue?label=good%20first%20issues&color=7057ff)](https://github.com/Ybucaille/nativ64/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

**Xbox Series games on Linux, natively: no Wine, no Proton, function by function.**

> **Legal, read first.** nativ64 ships no game files, no console system files and nothing
> extracted from them, and never will. You must own a legitimate copy of any title you run
> and dump it yourself. The project does not name any specific game or publisher, in the
> repository or in issues and pull requests, and asks contributors to do the same. See
> [Legal and contributions](#legal-and-contributions).

Compatibility layers and recompilers already run Xbox Series code on a Windows PC.
Underneath, they all lean on the same thing: hundreds of Windows APIs borrowed from the
host (`kernel32`, `ntdll`, the C runtime, threads, files, sockets...). nativ64 rewrites
those APIs natively on Linux, as a standalone library that any of these projects can
load through a thin adapter.

The library is not tied to one game: the functions it implements are the same for every
title. Priorities come from the import map, which reads a game's import tables and shows
what is done and what is left. The first title measured is a large open-world game; more
titles can be added to the map, and functions shared by many games come first.

![nativ64 coverage](docs/import-map-nativ64.svg)

<details>
<summary>For comparison, what the first base already covers natively</summary>

![Linux-native coverage](docs/import-map-linux.svg)

</details>

For that first title (executable and package DLLs), 697 system functions are needed from
Windows (excluding the 265 its own DLLs ship). 46 already have portable code in
[VIctory](https://github.com/Marvisak/VIctory), the compatibility layer used as the first
base, 69 have VIctory code that still calls Windows, and **580 are host Windows APIs a
native port must write**.

## Stability first

Other projects aim to get a title on screen as fast as possible, borrowing whatever the
host Windows (or Wine) already provides. nativ64 takes the slower road on purpose: a
**stable, native Linux runtime** that does not inherit behaviour from a layer it does not
control.

- **Written and tested one function at a time.** Each function has its own tests,
  error cases included (null pointers, too-small buffers, overflows, timeouts,
  contention), and CI runs formatting, lints and tests on every push.
- **Exact Windows behaviour, Linux primitives.** Structure layouts and calling
  conventions match Windows bit for bit; underneath, it is plain Linux (futexes,
  syscalls), with no translation layer in between.
- **Once done, done.** A finished function is not revisited when the title changes or
  the base above it moves on, and every base that adopts the library gets it for free.

## How it fits

```
 compatibility layer (the "base": VIctory today, any other tomorrow)
        │  the title calls CreateFileW, VirtualAlloc, AcquireSRWLockExclusive...
   ┌────▼─────┐
   │ adapter   │  small, the only part tied to one base
   └────┬─────┘
 nativ64: the Windows APIs, reimplemented on Linux (Rust, Windows x64 calling convention)
        │
      Linux (syscalls, futexes, Vulkan, PipeWire...)
```

- **Base-agnostic.** The library knows nothing about the base above it. Switching base
  means writing a new adapter, not redoing the work.
- **Usable from Rust and C/C++.** Functions are exported with the Windows x64 calling
  convention and looked up by `module!function`, from Rust or through a C entry point.
- **Testable without the game.** Each function is checked against documented Windows
  behaviour on its own.
- **Measured.** The import map reads the title's import tables and shows, cell by cell,
  what is done and what is left, in one tab per base and one for this library.

## Status

Early, and **not playable yet**. Where things stand (2026-11-27), for the first title
measured:

- **Boots natively.** A native PE loader (`nativ64-run`, [`docs/loader.md`](docs/loader.md))
  maps the executable and its package DLLs, resolves imports, sets up the TEB/PEB, TLS and
  guest threads, and handles structured exceptions ([`docs/seh.md`](docs/seh.md)).
- **Windows and Xbox API layer.** The `nativ64` library implements the host APIs the title
  imports (C runtime, synchronization, handles, threads, heap, time, modules, a first set of
  system calls, console service stubs). Import map coverage as of its last refresh: 476 of
  the 697 host functions the title needs are answered natively by nativ64, 68.3% ([`docs/import-map.md`](docs/import-map.md)).
- **GPU layer.** `nativ64-gpu` implements the console's Direct3D 12 interface as recorded
  commands replayed on **Vulkan 1.3** (own backend, no vkd3d), with a Wayland swapchain
  ([`docs/gpu.md`](docs/gpu.md)).
- **Own shader translator.** RDNA2 (GFX10) shader programs are decoded, turned into an IR and
  emitted as SPIR-V: vertex, pixel and, since the last milestone, compute programs
  ([`docs/shaders.md`](docs/shaders.md)).
- **On screen.** The setup and legal screens render, and the intro video decodes. The
  title does not reach gameplay.

Next: the memory peak while the title loads, keyboard input, cross-lane shader operations and
mesh shaders.

Host functions still to write for the first title (221 of 697):

| Area | To write |
|---|---|
| System | 85 |
| C/C++ runtime (`api-ms-win-crt-*`) | 18 |
| Network & crypto | 47 |
| Threads & sync | 23 |
| Files | 10 |
| Memory | 19 |
| Loader | 3 |
| Other (Xbox services, graphics, input, audio...) | 16 |

## Requirements

- Linux x86-64 and a recent stable Rust toolchain; Python 3 (no dependencies) for the tools.
- To run a title: a GPU and driver with **Vulkan 1.3**, `VK_EXT_external_memory_host` and
  the features the backend checks at start (`shaderInt64`, `depthClamp`, `independentBlend`,
  buffer device address, timeline semaphores, descriptor indexing with non-uniform
  indexing, dynamic rendering, synchronization2, demote to helper invocation, subgroup
  operations in compute and fragment shaders...). The exact list is in
  `crates/nativ64-gpu/src/backend/vk/` (`device.rs` and `subgroup.rs`); a device that lacks one is
  refused with the name of the first missing feature. A Wayland compositor gives a real
  window; without one, rendering is headless.
- The tests need no game and no GPU: CI runs them on Mesa's CPU driver (lavapipe).

## Getting started

```bash
git clone https://github.com/Ybucaille/nativ64.git
cd nativ64
cargo test                     # no game files needed
```

Running a title needs your own legally obtained, dumped copy of its files. Nothing here
helps you obtain one.

```bash
cargo run --release -p nativ64-loader --bin nativ64-run -- \
    --data /path/to/package/dir /path/to/your/dumped/executable
```

`nativ64-run --check <exe>` only maps the image and binds its imports. Other binaries:
`nativ64-vkdemo` (Vulkan swapchain demo), `nativ64-shader` (shader translator command line),
`nativ64-status` (library coverage). The environment variables that select the GPU
(`NATIV64_GPU`), the window (`NATIV64_WINDOW`), validation layers (`NATIV64_VK_VALIDATION`)
and the frame, draw and shader dumps for debugging are documented in
[`docs/gpu.md`](docs/gpu.md), [`docs/shaders.md`](docs/shaders.md) and
[`docs/loader.md`](docs/loader.md). Debug dumps can contain game data: never attach them to
an issue or pull request.

To draw the import map you need your own copy of a game's files and, for the base tabs, a
VIctory checkout next to this repository. See [`docs/import-map.md`](docs/import-map.md).

## Architecture

| Crate | Role | Documentation |
|---|---|---|
| `crates/nativ64` | Host Windows APIs on Linux (`extern "win64"`, libc as only dependency) | [`docs/library.md`](docs/library.md), [`docs/audio.md`](docs/audio.md) |
| `crates/nativ64-loader` | Native PE loader, TEB/PEB, TLS, threads, faults and SEH | [`docs/loader.md`](docs/loader.md), [`docs/seh.md`](docs/seh.md) |
| `crates/nativ64-gpu` | Console Direct3D 12 on Vulkan, texture layouts, RDNA2 shader translator | [`docs/gpu.md`](docs/gpu.md), [`docs/shaders.md`](docs/shaders.md), [`docs/textures.md`](docs/textures.md) |

## Community

Discord: https://discord.gg/CtX9AArtA2 (talk about the project in the community channels; to work on the
code, read [CONTRIBUTING.md](CONTRIBUTING.md) and ask for developer access in `#dev-access`). Bugs and
feature requests go to GitHub issues.

## Contributing

The map is the to-do list: pick a grey cell, say so on its issue, implement the function in
`nativ64` with its tests, open a pull request. [`CONTRIBUTING.md`](CONTRIBUTING.md) explains
how, and [`docs/library.md`](docs/library.md) covers the ABI rules (calling convention,
32-bit `LONG`, UTF-16 strings, `SetLastError`).

Good first contributions: C runtime string and math functions, timing and system-info
calls, synchronization primitives on futexes. Read the legal rules below before your first
pull request.

## Repository layout

| Path | What |
|---|---|
| `tools/` | Import map scripts and page template |
| `docs/` | Documentation |
| `game/` | Your local game files (not versioned) |
| `reports/` | Generated pages (not versioned: derived from game files) |

The crates are listed under [Architecture](#architecture).

## Related projects

Other projects run Xbox Series (GDK) titles on a PC. They target Windows and translate
the console's graphics to the host's Direct3D 12; nativ64 targets Linux, with its own
host APIs and a Vulkan path, and is meant to sit under any of them.

| Project | Host | Approach | License |
|---|---|---|---|
| [VIctory](https://github.com/Marvisak/VIctory) | Windows | Compatibility layer: the title's x64 code runs natively, console libraries reimplemented, RDNA2 shaders recompiled to DXIL | GPL-3.0 |
| [DexNom Runtime](https://github.com/DexNom/VIctory) | Windows | Fork of VIctory focused on Radeon rendering and GPU memory | GPL-3.0 |
| [XCL](https://github.com/xcl-dev/xcl) | Windows | Compatibility layer in the spirit of Wine/Proton: own console libraries, RDNA2 shaders recompiled through HLSL to DXIL | GPL-2.0 |
| **nativ64** | **Linux** | Windows host APIs rewritten natively, a PE loader, Xbox Direct3D 12 on Vulkan (in progress) | MIT OR Apache-2.0 |

nativ64 contains no code from these projects; they are read as references for how the
console's interfaces behave.

## Legal and contributions

- This project contains no game files, no console system files and nothing extracted from
  them, and never will. You supply your own legally obtained copy, which you dump yourself.
- It is not affiliated with Microsoft, any game publisher or any compatibility-layer
  project.
- **No game or publisher names**, and no executable names, anywhere in the repository,
  issues, pull requests or commit messages. Say "the title".
- **Never commit** game files, dumps, extracted shader blobs, textures, audio, logs,
  screenshots or crash reports containing game data or game strings. That includes
  attachments to issues and pull requests.
- **Clean room.** No code copied from proprietary SDKs or leaked sources, and none from
  GPL or LGPL projects (this project is MIT OR Apache-2.0). Other projects, Wine's tests and
  public documentation are read for facts only; the code is written from those facts.
- **Pure native only.** No Wine, Proton, DXVK, vkd3d or vkd3d-proton, dxil-spirv, or any
  shim or dependency built over them. Reference projects are read for facts only.
- **Memory discipline.** RAM and VRAM frugality from the start: every cache has a bound
  or a retirement, no CPU copy is kept once uploaded, staging memory is freed or reused,
  and GPU imports and images are released when the title frees their memory. A pull
  request that touches memory paths says how memory was measured.
- Maintainers check every pull request for these points. Content that breaks them is
  not merged: you will be asked to remove it and to squash it out of the history first,
  and a pull request that keeps it is closed.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option,
so any base can use the library, whatever its own license.
