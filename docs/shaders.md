# nativ64-gpu: the shader translator (G4)

The title ships its shaders compiled for the console GPU: RDNA2 (GFX10) machine code inside
a DXBC container. nativ64 has no DXIL to hand to a driver and no D3D12 to run it on, so each
program is translated at first use: GFX10 → our SSA IR → structured per-lane form → SPIR-V,
then a Vulkan pipeline. This file is the design; `docs/gpu.md` (Shaders bullet, G1
deviations, G4) places it in the GPU work.

Fact tags: **[seen]** checked on the title's own blobs (how is said once, in Sources);
**[llvm]** checked with `llvm-mc` (LLVM 23.1.5, `-mcpu=gfx1030` and `gfx1013` decode these blobs
identically); **[VIctory]** read in VIctory's `victory-shader` (GPL-3, facts only), not checked
here; **[spec]** from AMD/Khronos documents, not checked here; **[unverified]** a guess to
confirm.

## Sources

- **Blobs.** The executable embeds 10 DXBC containers (1 VS, 9 PS); the rest live in
  `shaders.rpf` (an `RPF9` archive, compressed or encrypted: no `DXBC` bytes visible), so they
  are only seen at pipeline creation. `pipeline.rs` keeps each stage's bytes (`Stage.bytes`)
  and the input layout. Chunk 1 adds a dump of every created pipeline's blobs; until then the
  10 embedded blobs are the corpus. Extraction was a throwaway script (scan for `DXBC`, check
  size and part table), never committed; game bytes never enter the repository or tests.
- **ISA.** AMD, *"RDNA 2" Instruction Set Architecture Reference Guide* (2020), public at
  `https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf`
  (the URL answers a plain `curl` with an empty page: read it in a browser). Machine-checkable
  oracle: `llvm-mc -arch=amdgcn -mcpu=gfx1030 -disassemble` (installed here, Apache-2.0 with
  LLVM exception; run as a tool, never linked). LLVM's `AMDGPUAsmGFX1030.rst` lists every form.
  Xbox2Windows calls the target gfx1013 **[unverified]**. The 10 embedded blobs decode the same
  under both LLVM targets, but a title run's 8634 blobs do not: they use `v_mad_f32`, `v_mac_f32`,
  `v_madak_f32` (gfx10.1 opcodes that gfx1030 lacks), and the same opcode can have another name
  (VOP3 0x140: `v_mad_legacy_f32` on gfx1013, `v_fma_legacy_f32` on gfx1030). **[llvm, seen]**
  The decoder's oracle is therefore `-mcpu=gfx1013`.
- **VIctory** (`../VIctory-mirror/crates/victory-shader`, GPL-3, targets DXIL): container
  offsets, stage data, root layout tables, the per-lane structuring argument. Facts only: no
  code, no module split, no names copied.
- **XCL** (`../xcl-mirror`, GPL-2, targets HLSL → DXIL): `docs/wave64-on-wave32.md` on running
  wave64 programs on 32-lane GPUs. Facts only.
- **Xbox2Windows** (`R0Y00/Xbox2Windows`): LICENSE is MIT (**[seen]**: raw LICENSE file, and
  the GitHub API reports `spdx_id: MIT`), so its code may be reused with its notice in
  `THIRD_PARTY.md`. It targets DXIL from Python/C++ tooling; it offers instruction-form
  coverage tables (`docs/coverage/rdna2.tsv`, `gfx1013.tsv`: 1968 forms, generated from LLVM's
  GFX10 asm docs), HLSL helpers (`wave64_emulate.hlsli`, `rgb9e5.hlsli`,
  `ffx_float_to_int.hlsli`) and hardware-measured fixtures (`tests/data/rdna_cvt_pk_u8_measured.tsv`,
  per-instruction `.s` tests). Use: its coverage table as our checklist, its measured data
  as test vectors (with notice). Its translator is HLSL-shaped; we keep our own.

## Container

```text
DXBC  +0x00 "DXBC", +0x04 digest[16], +0x14 u16 1, u16 0, +0x18 total size, +0x1c part count,
      +0x20 u32 part offsets; part = fourcc, u32 size, data                       [seen]
parts SFI0 ISG1 OSG1 PSV0 HASH DXIL PRIV, in that order on all 10 blobs            [seen]
DXIL  +0x00 u32 program version: low byte 0x66 (6.6), high half kind (0 PS, 1 VS)  [seen]
      +0x04 u32 size in dwords, +0x08 "DXIL", +0x0c u32 0x106, +0x10 bitcode offset, +0x14 size
      +0x18 "XDAT", u32 entry count (3), u32 0x24, then entries {u32 kind, u32 offset, u32 size}
            (offsets relative to the DXIL part data)                              [seen]
kind 0x12 16 bytes (hash?); kind 0x07 28-byte stub (stripped bitcode)             [seen]
kind 0x0d "XBCH", u32 version 3, u64 packed, u64 unpacked, then a zlib stream     [seen]
unpacked  "FSSM", u32 0x101, u64 payload size, payload                            [seen]
payload   +0x00 u64 6, +0x08..0x38 record offsets LS HS ES GS VS PS (absent:
          0xDEADBEEF_DEADBEEF), mesh at +0x48, compute at +0x60                    [seen VS/PS]
record    +0x00 u64 code offset, +0x08 u64 code size (payload-relative)            [seen]
          +0x14 u32 flags, bit 5 wave32 (clear on all 10: wave64)                  [VIctory]
          +0x28 u64 root layout table, +0x30 u64 used-resource table, +0x38/+0x3c counts [seen]
          +0x4c u32 LDS bytes, +0x50 u32 bits 0-9 VGPRs, bits 20-27 SGPRs          [seen]
layout    20 B: u32 root type, u32 table offset (dwords), u32 user SGPR count,
          u32 base register, u8 first user-data dword, u8 root slot, u16 0         [seen]
used      20 B: u8 kind (0 SRV, 1 sampler, 2 CBV, 3 UAV), u8 root type, u16 0,
          u32 table offset, u32 lower bound, u32 stride, u8 descriptor dwords,
          u8 first user-data dword, u8 root slot, u8 0                             [seen layout, VIctory kinds]
root type 0 constants, 1 root descriptor (64-bit address), 3 table (32-bit heap
          address), 4 static sampler, 5 resource heap, 6 sampler heap             [VIctory; 1 seen]
```

Stage data follows each record (E = record + 0x54):

```text
VS  E+0x24 32 inputs × 5 B {usage, signature row, channel mask, first VGPR, last channel},
    E+0xc4 count; E+0xc8 32 outputs × 6 B {usage, row, param n, row, slot, flags}, E+0x188 count
PS  E+0x24 32 inputs × 8 B {usage, row, mask, attr, default, 2nd row, param slot, flags:
    bit 0 flat, 1 linear, 3 fp16, 4 packed fp16}, E+0x124 count
    E+0x134 SPI_PS_INPUT_ADDR, E+0x138 DB_SHADER_CONTROL, E+0x148 patch size (dwords),
    E+0x14c 8 × {i32 code dword offset (-1 none), 10 variants × 10 dwords}
    variants: ZERO 32_R 32_GR 32_AR FP16 UNORM16 SNORM16 UINT16 SINT16 32_ABGR
```

**[seen]** on the corpus: VS output `{6, row 1, param 0}`; PS input `{6, row 1, mask 0xf,
attr 0, flags flat}`; `SPI_PS_INPUT_ADDR` 0x2; PS patch at dword 9 (size 6) whose 32_ABGR
variant is `s_nop ×4; exp mrt0 v2 done vm` in place of `v_cvt_pkrtz…; exp mrt0 … compr`.
Usage 6 = generic, 7 = clip distance **[VIctory]**. Merged stages (ES/GS, LS/HS, NGG) and mesh
keep other layouts **[VIctory: not decoded]**; the corpus has none.

**Chunk 1 check** (`shader/`, run on the 10 embedded blobs: all parse, 1 VS and 9 PS; VS and PS
fields above reproduced, `nativ64-shader` prints them). Precisions and corrections **[seen]**:

- The `u32 0x24` after the XDAT entry count is the offset of the entry table in the `DXIL` part
  (entries at part +0x24); the bitcode offset at `DXIL +0x10` counts from the `DXIL` magic (+8).
  The program version dword is `kind << 16 | major << 4 | minor` (0x66 = 6.6).
- `XBCH` packed size = bytes after the 24-byte header; `FSSM` payload size = unpacked − 16.
- Payload slot table: the first u64 (6) is read as a slot count; the pointers LS..PS sit at
  +0x08..+0x30, and +0x68 holds 2 on all blobs (meaning unknown, ignored).
- PS patch table: 8 entries of 404 bytes (`i32` code dword offset or -1, then 10 variants of
  10 dwords of which `patch size` are used). Only target 0 is patched on the corpus; the patch
  sits at dword 9, 10 or 12 depending on the program (not always 9), with size 6 everywhere.
  The 32_ABGR variant is four `s_nop` plus an `exp mrt0 … done vm` whose enable mask is the
  target's channel mask (0x1 or 0xf): `v2` only on 3 programs, 4 channels on the others.
- Used-resource kind 0 (`SRV`) names the VS's root descriptor (type 1) read as `s8:s9`, so
  `kind` does not separate buffers from views; tracing (Resources) decides.
- `ISG1`/`OSG1` elements are 32 bytes (stream, name offset, index, system value, component
  type, register, mask, read-write mask, min precision); names follow the table.
- VS inputs (stage data) are empty on the corpus: that layout is still unverified. Merged
  stages and compute records are not decoded (present slots are listed only).

**Code.** Starts with two 12-byte VOP3 words of opcode 0x180 (`v_nop` in VOP3 form with junk
operand fields and a literal: a marker or hash) and `s_inst_prefetch 3`; `llvm-mc` rejects the
two (bad operands for `v_nop`) **[seen, llvm]**. The program ends at `s_endpgm`, padded with
`s_code_end`; compilers may append constant data after it **[VIctory]**. So instructions are
found by following control flow from offset 0, never by a linear sweep, and `v_nop` decodes
whatever its operand bits say.

**Decisions.** One `Container` parser (DXBC) and one `NativeProgram` reader (XDAT → XBCH →
`miniz_oxide` inflate → FSSM → records), refusing any unknown version or an unpacked size over 16 MiB. A VS may carry two kind-0x0d
programs, compiled with and without a geometry stage **[VIctory]**: take the one with a VS
record. The PS export patch is applied with the 32_ABGR
variant (exports carry raw 32-bit values) and the render-target format is handled by Vulkan
(see Outputs). The digest is not checked (our checksum code is unverified).

## Decoding (GFX10)

Encodings, by the top bits of the first dword **[spec, llvm]**:

| Bits | Encoding | Size |
|---|---|---|
| `0` | VOP2 | 4 (+4 literal, +DPP/SDWA 4/8) |
| `0111111` / `0111110` | VOP1 / VOPC | 4 (+…) |
| `10` (not below) | SOP2 | 4 (+4) |
| `1011` | SOPK | 4 |
| `101111101` / `…110` / `…111` | SOP1 / SOPC / SOPP | 4 (+4) |
| `110010` | VINTRP | 4 |
| `110011` | VOP3P | 8 (+4) |
| `110101` | VOP3 (VOP3a/b; VOP1/2/C opcodes at fixed bases) | 8 (+4) |
| `110110` | DS | 8 |
| `110111` | FLAT / GLOBAL / SCRATCH | 8 |
| `111000` / `111010` | MUBUF / MTBUF | 8 |
| `111100` | MIMG (NSA adds dwords) | 8+ |
| `111101` | SMEM | 8 |
| `111110` | EXP | 8 |

Operand fields: SGPR 0-105, VCC 106-107, TTMP 108-123, M0 124, NULL 125, EXEC 126-127, inline
integers 128-208 (0..64, -1..-16), inline floats 240-248 (0.5, -0.5, 1, -1, 2, -2, 4, -4,
1/(2π)), `vccz`/`execz`/`scc` 251-253, literal 255, VGPRs 256-511 **[spec]**. One literal per
instruction, shared by every operand that names 255.

**Design.** A table-driven decoder: per encoding a `[Option<OpInfo>; N]` built at compile time
(`OpInfo` = name, operand kinds and widths, float/int class, side effects). `decode(code, at)
-> Instr { at, len, enc, op, operands, modifiers (abs/neg/clamp/omod/opsel), dpp/sdwa,
mem fields (offset, glc/slc/dlc, idxen/offen, dmask, dim, a16/d16) }`. Unknown opcode = error
`unknown GFX10 VOP3 opcode 0x1a3 at +0x40` (the "unknown slot = stop" rule). `disasm()`
prints LLVM syntax so tests diff it against `llvm-mc` output **[llvm]**. Flow following: from
offset 0, branches (`s_branch`, `s_cbranch_*`: target = next + 4·simm16) and fall-through;
`s_endpgm` stops a path; `s_setpc`/`s_swappc` are recorded as indirect jumps (`s_getpc`
alone is allowed: it addresses constant data). **[seen]** The whole corpus (8634 blobs, 20.25M
instructions) decodes with every encoding above, 0 diff against `llvm-mc` on the instructions
compared. Its indirect jumps are of two kinds only: 13 compute programs use the far jump
(`s_getpc_b64 s[n:n+1]`, `s_add_u32 sn, sn, literal`, carry, `s_setpc_b64 s[n:n+1]`), and 332
vertex programs open with `s_swappc_b64 s[4:5], s[4:5]`, the call to the fetch routine (Inputs
and outputs). No jump table.

## IR

SSA over 32-bit register dwords, one function per program:

- **Registers**: `S(n)`, `V(n)`, `VCC_LO/HI`, `EXEC_LO/HI`, `M0`, `SCC`. A 64-bit operand is a
  pair. Every vector instruction reads EXEC; a VGPR write also reads the old value (lanes
  outside EXEC keep it). SCC and carries are explicit results.
- **Blocks** split at branch targets and after branches; `s_nop`, `s_waitcnt*`,
  `s_inst_prefetch`, `s_clause`, `s_dcache_*`, `s_code_end`, `v_nop` vanish.
- **Instructions keep their decoded opcode** (no early raising); semantics live in one place,
  the emitter's per-opcode table. Phis are inserted by the usual dominance-frontier method.
- **Entry values**: every register live at entry is an `Arg(reg)`; I/O raising and resource
  tracing give them meaning (user SGPRs, system SGPRs, input VGPRs). A live-in that nothing
  explains is an error naming the register.
- **As built** (`ir.rs`, chunk 3). Registers are `S(n)`, `T(n)` (ttmp), `V(n)`, `VccLo/Hi`,
  `ExecLo/Hi`, `M0`, `Scc`; wave32 only has the `Lo` halves. An `Inst` keeps its decoded `Instr`
  plus `dsts` (explicit dwords, then EXEC for `*_saveexec` / `v_cmpx`, then SCC), `srcs` (one
  `Vec<Src>` per explicit source operand, one `Src` per dword) and `imps` (implicit reads:
  EXEC for every vector instruction except `v_readlane/writelane`, VCC for `v_div_fmas`, SCC
  for `s_cselect/addc/subb/cmov`, M0 for `v_interp*` / `s_sendmsg`, and the old value of every
  written VGPR, which is `undef` when nothing defined it). `s_cbranch_*` are `Term::Branch`
  with their condition registers as reads. A `Src` is a `Val`, a `Const`, `CodeLo(off)` /
  `CodeHi` (the program's own address: `s_getpc_b64` + offsets; the code is assumed aligned so
  the low add never carries, SCC from it folds to 0) or `Undef` (phi input of a path that never
  defined the register). Folded instructions leave no `Inst`; their SCC folds too. Phis go on
  the iterated dominance frontier of the registers that are read somewhere, then trivial and
  dead phis are removed. `Func` carries `idom` and `rpo` for chunks 4-5. Encodings the decoder
  does not read, and `s_setpc/swappc/rfe/call`, `s_movrel*`, `v_movrel*`, scalar atomics,
  MUBUF-to-LDS are typed `IrError`s.
  **Chunk 4b** (`ir.rs`): MIMG, MTBUF, DS, VOP3P and FLAT/GLOBAL/SCRATCH are lowered. Data dwords
  follow the decoder (dmask channels, halved by d16, plus one dword for tfe or lwe, which only
  loads accept); NSA address lists name each register; resource and sampler are SGPR quads
  (8 dwords, 4 for a BVH ray). Stores have no `dsts`; an atomic (MIMG, FLAT) defines its old
  value only with glc, half of the data for compare-and-swap. Every written VGPR keeps its old
  value as an implicit read. DS reads M0 only for the `gds` bit, `ds_ordered_count`,
  `ds_append`/`ds_consume` and `*_addtid` (LLVM: `ldsRequiresM0Init` is generation < 9); GWS
  opcodes, MIMG r128 (outside BVH) and tfe/lwe on a store or atomic, LDS loads (FLAT) are typed
  errors. The flat-scratch base of FLAT/SCRATCH is hidden state, not an IR register.
  **Chunk 4** (`ir.rs`, `native.rs`, `gfx10/flow.rs`): CS and mesh stage data are decoded like
  the VS/PS ones (`CsData`: `RSRC2` and group size, refused when a dimension is 0 or the group
  over 1024 threads; `MeshData`: outputs, lanes, vertex and primitive maxima, user SGPRs,
  topology). `EntryRegs::from_stage(record)` replaces the root-layout guess with the table of
  Inputs and outputs. `s_setpc_b64` of a `CodeLo` / `CodeHi` pair is a direct jump: `build`
  follows the code, folds, and when a `s_setpc_b64` folds to byte `t` follows `t` and builds
  again (`gfx10::follow_with`), so the far-jump targets need no separate analysis. A VS whose
  stage data lists attributes and whose first instruction is `s_swappc_b64 s[4:5], s[4:5]`
  gets that call as an `Inst` reading `s[4:5]` (entry values) and defining the attribute VGPRs
  (first VGPR + channel), then `s4`, `s5`; any other jump through a register, and the same call
  without attributes, is `IrErrorKind::Indirect`. Scalar compares of constants (`s_cmp_*`)
  fold their SCC. After SSA, `prune` turns an `s_cbranch_*` whose condition folded to a
  constant into a jump (a wave64 mask is decided when either half is a non-zero constant),
  drops the blocks only it reached, removes the dead phi inputs (the phis left with one value
  fold, which can decide more branches), recomputes `idom` / `rpo` and drops the arguments
  nothing reads any more. `nativ64-shader --ir --stats` reports, per stage, the programs that
  build and the errors by kind. `nativ64-shader --ir` prints the IR.
- **Analyses on SSA**: constant folding of scalar values (literals, `s_getpc` + offset, loads
  from appended constant data), lane-mask typing, resource tracing, uniform/lane classification.
  Emission does not need phis (below).

## Structuring: EXEC masks → per-lane structured flow

SPIR-V's `Shader` capability requires structured control flow; GFX10 code is a CFG whose
divergence is in EXEC. Each SPIR-V invocation runs one console lane, so EXEC becomes one
boolean per invocation.

1. **Mask typing.** A wave-sized SGPR value (64-bit pair in wave64, 32-bit in wave32) is a
   *mask* when it comes from `v_cmp*`, `v_cmpx*`, carry-outs, EXEC, VCC, `s_*_saveexec`,
   or bitwise ops (`and/or/xor/andn2/orn2/not/mov/cselect`) of masks and of constants 0/-1.
   A mask becomes a `bool` (this lane's bit). Uses allowed: EXEC write, `v_cndmask`, carry-in,
   `s_cbranch_vcc*/exec*`, SCC from a mask op (= the bit, see 2). Any other use (`s_bcnt1`,
   `s_ff1`, shifts, arithmetic, memory, `v_readlane` of a mask) is *cross-lane*: error until
   the wave chunk (7).
2. **Branches.** A wave-wide test of a mask (`s_cbranch_execz`, `vccz`, SCC of a mask op)
   becomes a test of the lane's own bit. Sound because compiled code skips only work the
   lanes outside the mask would not change, so each lane can decide alone **[VIctory]**. A
   branch on a scalar non-mask value is uniform: same in every invocation. A scalar value
   defined in a region some lanes skip and used after the join would diverge: error.
3. **Guards.** Each vector instruction is guarded by `active` (EXEC's bit). A path-sensitive
   pass removes guards it proves true (after `s_and_saveexec` + `s_cbranch_execz`, the usual
   if-pattern). What stays: ALU writes `v = active ? new : old` (`OpSelect`); stores, atomics,
   exports and `image_store` inside `if (active)`.
4. **Pixel discard.** `s_andn2 live, live, exec` then a branch when no lane is live, and
   `s_wqm` keeping helpers, become `OpDemoteToHelperInvocation` at that point: the lane keeps
   running as a helper (derivatives read it), its outputs and stores are dropped **[VIctory
   for the pattern; spec for demote]**. The final `exp … done` with the lane's bit clear is a
   demote too.
5. **Structured tree.** The CFG must be reducible (else error). A dominator-tree algorithm
   (Ramsey, *Beyond Relooper*, JFP 2022) turns it into a tree of `Block(label)`, `Loop(label)`,
   `If`, `Br(label)`, `Return`. Lowering to SPIR-V: `If` = `OpSelectionMerge`; `Loop` =
   `OpLoopMerge` with a continue block that branches back; `Block` = a one-trip loop (merge =
   label, continue block unreachable). A `Br` to the innermost loop's merge/continue is a
   plain branch; a `Br` out of k levels sets a function variable `target`, breaks, and each
   enclosing merge re-tests `target`. Always valid; `spirv-val` checks it in tests.
6. **Registers in emission** are `Function` `OpVariable`s (one per register dword the program
   touches, `bool` for masks), loaded and stored around each instruction: no phis to place
   after restructuring, and drivers promote them. SSA stays for analysis.

- **As built** (`structure.rs`, chunk 5). `structure(&Func)` returns `Structured`: `masks`
  (typing, `scc_of`, the cross-lane uses), `branches` (per block: `Uniform` or `Lane(LaneTest)`),
  `guards` (per instruction: `Scalar` / `Proved` / `Active`), `demotes` and `tree`. Errors are
  typed: `CrossLane { at, name }` (first in program order), `DivergentScalar`, `Irreducible`.
  Deviations from the steps above:
  1. Typing is per dword and optimistic (loop-carried mask phis type). A mask operation with a
     non-constant, non-mask input (`s_and_b32 s0, exec_lo, s5`) is reported as a cross-lane use
     by that instruction. Cross-lane by nature, whatever the operands: `v_readlane`,
     `v_readfirstlane`, `v_writelane`, `v_mbcnt*`, `v_permlane*`, `ds_swizzle/permute/bpermute`,
     any DPP instruction. Carry-in and `v_cndmask` take a mask in their last source only.
  2. A lane-wise branch whose mask is built only from one EXEC value and scalar selects
     (`vccz` of `exec & ~(scc ? exec : 0)`) and is not an EXEC value is not split-checked: every
     lane that runs on decides alike. `DivergentScalar` looks at the phis of the immediate
     post-dominator of each other lane-wise branch (a differing scalar, `undef` included). It
     is strict: it also stops the usual `v_cmpx` + `s_cbranch_execz` over scalar loads whose
     result is read after the join (corpus: 1321 CS, 683 PS, 50 VS, 8 mesh once cross-lane
     uses are set aside; resolved by speculation, below).
  3. Guards are proved from per-value facts (EXEC of the entry, the bit a branch tests on its
     edges, copies and and/or/andn2 of known values). The join after an if/else (`exec | saved`
     where `saved` came out of an xor) is not proved: it needs bit-set reasoning, so those
     guards stay `Active` (correct, only less sparse).
  4. The discard is marked at the `s_andn2 live, live, exec` (`Demote`) when its SCC branch
     reaches a block that only does scalar work and a `done` export to `null`. The final `exp
     done` with the lane's bit clear is not marked yet.
  5. The tree is Ramsey's, with exits inlined in the `Loop` when a loop exit has one
     predecessor. `Br { to, depth }` carries the number of constructs left; each `Block` /
     `Loop` lists in `forwards` the targets that leave through it (the `target` re-tests).
     Irreducible means a retreating edge whose target does not dominate its source.
  `nativ64-shader --structure --stats` reports per stage and per error kind, then the same
  ignoring cross-lane uses, and the cross-lane instructions by name.

**Decisions after chunk 5** (corpus of 8610 programs with IR: 2862 structure; cross-lane uses
stop 2023 CS, 1687 PS, all 1569 mesh, 270 VS; a divergent scalar stops the rest; none is
irreducible):

- **Speculated scalar regions.** Scalar instructions with no side effect (SALU, SMEM loads)
  inside a region that lanes skip are run by every invocation, unguarded. The wave runs that
  region when at least one lane enters it, so a scalar phi at the join selects the region's
  value when `OpGroupNonUniformAny(entered)` and the old value otherwise: exact, and it replaces
  the `DivergentScalar` error for these. SMEM loads are safe to speculate because every load is
  bounds-checked (Resources). A scalar store, atomic or `s_sendmsg` in such a region stays an
  error.
- **Scalar bits as lane masks.** A mask operation with a uniform non-mask scalar
  (`s_and_b32 exec_lo, exec_lo, s5`) is not cross-lane: lane n reads bit n of the scalar
  (`n` = the console lane index, below). Only reading a whole mask as data (`v_mov_b32 v, s_mask`,
  `s_bcnt1`, `s_ff1` of a mask) needs a ballot.
- **Wave ops are on the main path** (chunk 7, right after the SPIR-V builder). Revised by the
  wave survey below: no workgroup-memory emulation of the 64-lane wave for now.

**Chunk 7 design (Opus, after the wave survey).** `nativ64-shader --structure --waves` classifies
every cross-lane instruction of the corpus (8610 programs with IR). Programs with a cross-lane
use: cs 2023, ps 1687, vs 270, mesh 1569. With the survey's first, strict uniformity test, the
programs whose uses are all size-agnostic were cs 270, ps 1116, vs 226, mesh 0. What the uses
are:

- **SGPR spills into VGPR lanes**: `v_writelane v, s, k` / `v_readlane s, v, k` with a constant
  `k` on a VGPR that nothing else touches (cs: writelane in 775 programs, readlane at a constant
  lane < 32 in 1047, seen on a sample). Not a lane exchange at all.
- **`v_readfirstlane` of a value that is uniform over the active lanes** (cs 1691 programs, ps
  877, vs 232, mesh 426 by the strict test). The mesh "other" cases seen on a sample are uniform
  too, but the strict test misses them: a system entry VGPR, a `buffer_load` whose address is
  an SGPR, a `v_cndmask` on a mask `scc ? exec : 0`.
- **Waterfall loops** (`v_readfirstlane` → `v_cmp_eq` with the same VGPR; cs 1137, ps 135, vs
  25, mesh 34).
- **Lane index**: `v_mbcnt_lo/hi(-1, ·)` (all 1569 mesh, cs 270).
- **Whole-wave sections** (`s_or_saveexec -1` … restore): cs 923 programs, ps 284. They hold
  the DPP row ops, `v_permlanex16`, DPP8 and the final `v_readlane` of HLSL wave reductions and
  prefix sums: 37.6 k cross-lane instructions in cs, 4.7 k in ps.
- **Quad ops** in pixel shaders: `ds_swizzle` quad mode (916), DPP `quad_perm` (124).
- **Masks as data**: `s_ff1` (cs 787), `s_bcnt1`, a few moves and compares; `mbcnt` of a real
  mask (cs 100); `ds_bpermute` (cs 6). No `row_bcast`, `wave_*` DPP or `v_permlane16` anywhere.

Decision: **keep the per-lane model of chunk 5** (one invocation per console lane, structured
per-lane flow) and give each use a meaning that does not depend on the host's subgroup size.
XCL's split form (two wave32 halves meeting through groupshared, uniform flow per console wave)
would need a second translation model and costs 1.1–4× on its benchmark, while the survey shows
that almost every use has a size-independent reading:

1. **Uniformity** (`structure/uniform.rs`): a value is *uniform* when it is the same in every
   active lane. Constants, SGPRs (scalar code is uniform by construction once speculation, below,
   is in), `v_readfirstlane` results, VALU of uniform operands, `v_cndmask` whose mask is EXEC or
   0 chosen by a uniform condition (and whose values are uniform), vector loads (MUBUF, MTBUF,
   GLOBAL, MIMG without derivatives) whose every address and resource operand is uniform (not
   atomics, not after a store to the same resource in the same block: refused as not uniform),
   and system entry VGPRs that the stage table marks wave-uniform (mesh `v5` and the like, from
   Inputs and outputs; an unknown one is not uniform). `v_readfirstlane` / `v_readlane` of a
   uniform value is a copy: no subgroup op.
2. **Spill slots**: a VGPR whose every definition is `v_writelane` at a constant lane and whose
   every read is `v_readlane` at a constant lane becomes one scalar variable per lane used.
   Exact. A slot read before any write reads `undef`, as on the console.
3. **Waterfall**: `v_readfirstlane` in a waterfall loop takes this lane's own value; the
   compare is then true, the body runs once per lane, and the resource index it feeds is marked
   `NonUniform` (Resources). Exact per lane.
4. **Console lane index**: `v_mbcnt_lo(-1, x)` = `min(lane, 32) + x`, `v_mbcnt_hi(-1, x)` =
   `max(lane - 32, 0) + x`, where `lane` = the console lane index: in compute and mesh
   `LocalInvocationIndex % 64` (wave64; `% 32` in wave32), never the host's subgroup id. In PS
   and VS the console lane is not observable; lane-index uses there are refused by name.
5. **Whole-wave sections** are raised to one subgroup operation (VIctory's approach, read from
   `emit/wave.rs`, no code copied): simulate the section on 64 symbolic lanes; a value it
   leaves that turns out to be the reduction, inclusive or exclusive prefix of one associative
   operation over the lanes' masked inputs becomes `OpGroupNonUniform{IAdd,FAdd,UMin,...}`
   `Reduce` / `InclusiveScan` / `ExclusiveScan` over the host's active invocations. That is the
   HLSL meaning (`WaveActiveSum`, `WavePrefixSum`), which a size-agnostic program accepts; any
   step the simulation does not model leaves its values unknown and the program refused by
   name.
6. **Quad ops** (`ds_swizzle` quad mode, DPP `quad_perm`): `OpGroupNonUniformQuadBroadcast` /
   `QuadSwap`, or `Shuffle` to `(id & ~3) | sel` with `subgroupQuadOperationsInAllStages`
   unset. Exact on any host (quads are 4 lanes everywhere). Helper lanes stay active for them.
7. **Masks as data**: the ballot of the lane's bits over the host's active invocations
   (`OpGroupNonUniformBallot`). Allowed where the meaning is size-agnostic: `s_bcnt1` (count),
   `s_ff1` (first active lane, then only fed to `v_readlane`, which becomes `BroadcastFirst`),
   `mbcnt` of a mask (`BallotExclusiveBitCount`), comparing a mask with 0 or with EXEC. Lane
   numbers from a ballot are host lanes: they must never meet a console lane index (item 4)
   in the same program; refused by name if they do.
8. **The rest is refused by name**: `v_readlane` at a computed lane outside 7, `v_writelane`
   outside 2, `ds_bpermute`, DPP or `v_permlanex16` outside a whole-wave section. If the corpus
   count after items 1–7 shows such programs on the first screen's path, the split form (XCL) is
   the fallback, for compute and mesh only.

Requirements: SPIR-V `GroupNonUniform`, `…Ballot`, `…Arithmetic`, `…Quad`, `…Shuffle`, and
`SPV_KHR_maximal_reconvergence` (`MaximallyReconvergesKHR`) on every program with a subgroup
op, so the host's active set at the op is the set of lanes whose structured flow reached it
(NVIDIA, ANV and llvmpipe here all have `shaderMaximalReconvergence`). A subgroup op sits
inside `if (active)` where its guard is not `Proved`, so lanes whose EXEC bit is clear do not
take part.

Tests: the probe runs one lane; chunk 7 adds a **wave evaluator** that runs the `Cpu` backend
over 64 lanes (and over a host width of 8, 32 or 64, the subgroup ops computed over that host
width) so a test checks both "console-exact on a 64-wide host" and "same result on narrower
hosts" for size-agnostic programs. On a device (`NATIV64_SPIRV_RUN=1`) the probe gets a
`LocalSize` of 64 and compares per lane (llvmpipe: 8-wide subgroups, so only size-agnostic
cases run there).

Sub-chunks, each with its corpus count (`--structure --waves` and `--structure --stats`):
- **7a — IR and structure, no wave op**: `s_andn1/andn2_wrexec_*` write EXEC in the IR; far
  jumps resolved in one pass; speculated scalar regions (`OpGroupNonUniformAny` select at the
  join) replacing `DivergentScalar`; scalar bits as masks.
- **7b — uniformity, spill slots, waterfall, lane index** (items 1–4), with the wave evaluator.
- **7c — whole-wave sections, quad ops, ballots** (items 5–7), then the refusals of item 8 and
  the corpus count that decides on the split form.
- **Far jumps** resolve in one pass over the code before building (one program chains more than
  16: the per-build loop of chunk 4 is quadratic there).

**As built (7a)** (corpus of 8634 blobs; IR 8611 programs, was 8610):
- **`s_andn1/andn2_wrexec_*`**: the IR lists the SGPR then EXEC as destinations, reads EXEC, sets
  SCC; mask typing and the ALU (CPU and SPIR-V) treat them as the saveexec family. The semantics
  (`exec = sdst = ~src & exec`, `src & ~exec`; SCC = result != 0) are the ISA text, **[unverified]**.
  No corpus program uses them: no number moved. The bit-set facts of `s_andn2_saveexec` now
  combine as `src & ~exec` (they had `exec & ~src`).
- **Far jumps**: `resolve_jumps` runs a forward worklist over the instructions carrying the folded
  scalar values (`fold`), keeping at a join what every path agrees on; the build then runs once
  with every known target resolved. The per-build loop remains as the fallback for a target only
  the SSA form knows. The CS chaining more than 16 jumps now builds (IR 8611/8634; the 23
  refusals left are `v_movrels/movreld` indexed registers).
- **Speculated scalar regions**: deviation from the select at the join. The decision is on the
  branch, not on each phi: `BranchKind::Wave` is a lane-wise branch whose join has a scalar phi
  with differing inputs and whose region has no scalar store, atomic or `s_sendmsg`. Every
  invocation takes it on `OpGroupNonUniformAny(bit)` (capability `GroupNonUniformVote`), the
  region's scalar instructions run in all of them, its vector instructions stay `Active`, and
  no EXEC fact is drawn from its edges. This is the same value as the select (the region's when
  any lane entered, else the old one), with registers staying variables and no phi to place.
  The probe has one lane, so the vote is the lane's bit (the CPU evaluator reads `Wave` as
  `Lane`). Structure 2862 -> 3060 of 8611 (cs 693, ps 1607, vs 760, mesh 0); ignoring cross-lane
  uses 6549 -> 8603 of 8611; `divergent scalar` is gone (984 speculated branches), what is left
  is 8 irreducible cs, which it used to hide.
- **Scalar bits as lane masks**: a mask operation takes a non-mask scalar (or a constant other
  than 0 / -1) beside at least one mask; scalars alone still do not make a mask. The ALU reads
  bit 0 of the scalar's first dword (the probe's lane), for every dword of a wave64 mask. The
  mask operations (`s_and_b32`, `s_andn2_b32`, `s_or_b32`, `s_and_saveexec_*`, ...) leave the
  cross-lane list; the program counts do not move (the same programs still hold a readlane,
  mbcnt or `s_ff1`): cs 2024, ps 1687, mesh 1569, vs 270. ALU coverage 8525/8611 programs
  (3006/3060 structured).

**As built (7b)** (corpus of 19101 blobs; IR 19066 programs; `structure/uniform.rs`, the lane
facts in `Masks`, the table and emitter in `spirv/`):
- **Uniformity** (item 1). `Masks::uniform_vals` gives each value a *domain*: `Full` (every lane
  of the wave) or `Exec(e)` (the lanes of an EXEC value, `e` narrowing through `within`), or none.
  Optimistic fixpoint, lowered until stable. A VGPR instruction result has the domain of its own
  EXEC (`Full` under the entry EXEC or a constant all-ones one): its sources must cover that
  domain, so a write under a narrowed EXEC is uniform for the lanes that wrote, not for the ones
  that keep the old value. Rules: scalars, entry VGPRs the stage marks uniform
  (`EntryRegs::uniform_vgprs`, `Func::uniform_args`; mesh `v5`), VOP/VOP3 of uniform operands,
  `v_cndmask`/`v_addc`... whose mask is a constant or an EXEC-uniform mask (`Masks::uniform`)
  covering the instruction's lanes, pure loads (MUBUF, MTBUF, `global_load`, `image_load`,
  `get_resinfo` and samples with `_l`/`_lz`/`_d`, no `_b`/`_cl`) of uniform operands with no
  store or atomic earlier in the same block. A phi is uniform when its inputs are and its block
  is not in the region of a lane-wise branch (the blocks between the branch and its join, join
  included; every block of the region when the branch is in a loop, since its lanes leave at
  different iterations). **Deviation**: the survey's strict test took any phi of uniform inputs
  and ignored the old value of a written VGPR; both are unsound under lane-wise flow, which is
  why ps/vs structure counts dip slightly (below). `v_readfirstlane` and `v_readlane` of a
  value that covers their lanes are copies (`LaneUse::Copy`; `v_readlane` has no EXEC, so only
  a `Full` value qualifies). The table (`alu::exec`) runs both as a move.
- **Spill slots** (item 2). `slot_vgprs`: a VGPR that is not an entry register, whose every
  definition is `v_writelane` at a constant lane and every read `v_readlane` at a constant
  lane (lane masked to the wave size), nothing else naming it. Each `(vgpr, lane)` is a `Private`
  `u32` variable (zero at entry: a slot read before any write reads 0, the probe's `undef`);
  the emitter handles `LaneUse::SlotWrite/SlotRead` itself, ignoring EXEC (the ISA ignores it
  for both). The phis of a slot VGPR count as scalar phis for `divergent_scalar`, so a write in
  a lane-skipped region is speculated (`BranchKind::Wave`) or an error, like an SGPR.
  **Not supported**: a writelane whose scalar is a lane mask (a spilled EXEC/VCC; 23% of the
  sampled compute programs): the slot is refused and the instructions stay cross-lane (mask as
  data, item 7). The per-lane form would need a mask-typed slot lane and a mask-typed readlane
  result.
- **Waterfall** (item 3). A `v_readfirstlane` of a non-uniform value whose result and source
  meet in a `v_cmp_eq`/`v_cmpx_eq` anywhere in the program is `LaneUse::Waterfall`: the table runs
  it as a move (own lane value); the compare is true and the body runs once per lane.
  Resources need nothing new: every descriptor array element is already `NonUniform`
  (`resource.rs`, `non_uniform`).
- **Lane index** (item 4). `v_mbcnt_lo/hi(-1, x)` is `LaneUse::LaneIndex`; `Ctx::lane` carries
  the lane: the probe 0, compute `LocalInvocationIndex & 63` (`& 31` in wave32), a vertex or
  pixel program `None`, which is `Unsupported { chunk: 7 }` naming "the lane index of a vertex or
  pixel program". `mbcnt_lo = min(lane, 32) + x`, `mbcnt_hi = max(lane, 32) - 32 + x`. `mbcnt`
  of a real mask stays cross-lane (ballot, 7c). Not done: `scalar bits as lane masks` still read
  bit 0 of the scalar (7a) instead of bit `lane`.
- **Wave evaluator** (tests, `spirv/tests.rs`): `wave(f, s, lanes, inputs)` runs the CPU
  evaluator once per console lane with its lane index; lanes meet nowhere because no subgroup
  op is emitted yet, so the host width cannot matter and is not a parameter; 7c adds it with the
  first op that uses it (and a lock-step run for the `Wave` vote, still read as `Lane`). Tests:
  slots over 64 lanes, lane index lo/hi, waterfall own-value, uniform copies, narrowed-EXEC
  writes, loads before/after a store; each validated with `spirv-val`.
- **Corpus** (`--structure --stats`, `--spirv --stats`, release; before = 7a head, after = 7b):

  | | IR | structure before | structure after | SPIR-V valid before | after |
  |---|---|---|---|---|---|
  | all | 19066 | 5124 | **10328** (19058 ignoring cross-lane) | 2838 | **2843** |
  | cs | 2718 | 928 | 1086 | 268 | 275 |
  | mesh | 6681 | 0 | 5158 | 0 | 0 (all stop at `s_load`, chunk 9) |
  | ps | 8517 | 3128 | 2985 | 2026 | 2020 |
  | vs | 1150 | 1068 | 1099 | 544 | 548 |

  ps loses 143 structured programs and 6 valid ones against the unsound strict test (a
  readfirstlane after a lane-wise region, or of a value written under a narrowed EXEC). Cross-lane
  uses left, by programs: ps `ds_swizzle_b32` 4615, `v_readfirstlane` 2652 (cs 1447, mesh 1445:
  non-uniform source, no waterfall), DPP movs 2098, `v_readlane` 1392 (cs 1060: mask spills and
  whole-wave tails), `v_permlanex16` 1151 (cs 923), `s_ff1` cs 787, `v_writelane` cs 709.
  `--structure --stats` now prints the first failing blob of each error kind.

**As built (7c)** (same corpus; `structure/wave.rs`, `structure/uniform.rs`, the subgroup
operations in `spirv/emit.rs` and `spirv/emit/stage.rs`):
- **Whole-wave sections** (item 5). A section is the code of one block from an
  `s_or_saveexec` with all-ones to the `s_mov` that restores the saved EXEC. It is simulated on
  32 or 64 lanes; each lane holds a constant, or the combination by one operation of a set of
  lanes of one value. Modelled: `s_mov exec, const`, DPP16 (quad, row shifts / rotates /
  mirrors / share / xmask) and DPP8, `v_permlane(x)16`, `v_readlane` at a constant lane,
  `v_readfirstlane`, add / min / max (u32, i32, f32) and and / or / xor, and the
  inclusive-minus-own subtraction. A value read after the section must be a `Reduce`,
  `InclusiveScan` or `ExclusiveScan` of its input over all lanes: it becomes
  `OpGroupNonUniform*` (`LaneUse::WaveOut`) and the rest of the lane-crossing instructions are
  dropped (`WaveSkip`). The input must be defined inside the section, before its use (the
  `v_cndmask` that sets inactive lanes to the identity); the emitter keeps its value in a
  variable of its own. The old value of a VGPR that is written in every lane is not a read, so
  a later section may reuse the temporaries. Anything else refuses the whole section. The DPP
  row rotate direction is **[unverified]** (reductions and scans do not depend on it as far as
  the corpus shows). Float sums take the host's order.
- **Quad operations** (item 6). `ds_swizzle_b32` in quad mode (or bitmask mode that leaves the
  bits above the quad alone), DPP `quad_perm` and a `dpp8` that is the same in both quads:
  `QuadBroadcast`, `QuadSwap`, or a select over broadcasts on the host lane's low two bits.
  Helper invocations get no special handling beyond what the host's quad operations give
  **[unverified on a device]**.
- **Masks as data** (item 7). `s_bcnt1` = ballot bit count; `s_ff1` read only by `v_readlane` =
  `BallotFindLSB` then `Shuffle`; `v_mbcnt` of a mask = exclusive ballot count; `s_cmp_eq/lg`
  of a mask with zero or with a mask = `Any` / `All`. In a program that reads the console lane
  index, the three that number host lanes stay refused. A spill slot may hold a lane mask (the
  lane's bit); each `v_readlane` takes the kind of the writes it can see, so a compiler may
  reuse a lane for both kinds (a read that may see both refuses the VGPR).
- **Refusals** (item 8) are what the structurer already reports as cross-lane, by instruction
  name: `v_readlane` at a computed lane, `v_writelane` outside slots, `ds_bpermute`, DPP and
  `v_permlanex16` outside a modelled section, swizzles that leave the quad.
- Every program with a subgroup operation declares `MaximallyReconvergesKHR`.
- **Wave evaluator**: `interp_wave` runs the lanes in lock step with a host width (subgroup
  results computed per group of that width).
- **Not done**: the lock-step run still requires every lane to take the same path and reads the
  `Wave` vote as `Lane`; the device probe keeps `LocalSize` 1 (no per-lane comparison on a
  device); the backend does not yet require the subgroup features and
  `VK_KHR_shader_maximal_reconvergence` at device creation; scalar bits as masks still read
  bit 0; no product reductions; sections that span blocks.
- **Corpus** (before = 7b head, after = 7c):

  | | IR | structure before | structure after | SPIR-V valid before | after |
  |---|---|---|---|---|---|
  | all | 19066 | 10328 | **13321** (19058 ignoring cross-lane) | 2843 | **3492** |
  | cs | 2718 | 1086 | 1124 | 275 | 279 |
  | mesh | 6681 | 5158 | 5236 | 0 | 0 (all stop at `s_load`, chunk 9) |
  | ps | 8517 | 2985 | 5862 | 2020 | 2665 |
  | vs | 1150 | 1099 | 1099 | 548 | 548 |

  Cross-lane uses left, by programs: `v_readfirstlane` of a value not proved uniform and in no
  waterfall (ps 2652, cs 1447, mesh 1445), `v_mov_b32` reading a mask (cs 666, ps 578; seen:
  a scalar `s_cselect -1, 0` moved to a VGPR), `v_readlane` (ps 221, cs 201), unmodelled
  sections (`v_permlanex16` ps 148, cs 88), `s_ff1` read as data (cs 236, ps 79).
- **Split form**: not decided. The largest blocker is `v_readfirstlane`, which items 1-8 leave
  refused when uniformity is not proved; sampled cases are a loop counter or a scalar copied
  into a VGPR and merged under lane-wise flow, so a sharper uniformity (or a host-wide
  broadcast-first with the `WaveReadLaneFirst` meaning) should be measured before XCL's split
  form.

## Inputs and outputs

**User SGPRs.** A VS/PS/CS receives user-data dword i in `s[i]`; merged stages and mesh have 8
system SGPRs first (`s[i+8]`) **[VIctory]**. Seen: the VS's root descriptor (layout `user
data 8, 2 SGPRs`) is read as `s8:s9`; the PS gets the primitive mask in `s10`, right after
its user SGPRs (`s_mov_b32 m0, s10` before `v_interp`) **[seen]**.

**Entry SGPRs below s8, per stage** (first reads before any write, whole corpus **[seen]**;
meanings **[VIctory]** unless said):

| Stage | Reads in s0..s7 | Meaning |
|---|---|---|
| VS | `s[4:5]`, 332 of 1034, only as `s_swappc_b64 s[4:5], s[4:5]` at the entry | fetch routine address: the call is replaced by the attribute values (Vertex input) |
| PS | none | — |
| Mesh | `s3`, all 1569, by `s_bfe_u32` | merged wave info: wave index in the group, bits 24-27 |
| CS | `s4..s7` in most of the 2734 (`s[4:5]` as an SMEM base, `s6`/`s7` multiplied by group sizes), a few `s0..s3` | user SGPRs then group id x/y/z and group info, as `COMPUTE_PGM_RSRC2` (stage data E+0x24) enables them; scratch offset last |

So `EntryRegs` comes from the stage data, not from a guess: CS from `RSRC2` (user SGPR count
bits 5:1, group id x/y/z bits 7-9, group info bit 10, scratch bit 0) and the thread-group size
(E+0x28, 3 × u32); mesh from its record (E+0x448: group size in lanes, maximum vertices and
primitives, user SGPRs, topology; code at E+0x45c) plus `s3` and `v5` (group id × group size,
read through `v_readfirstlane`). A live-in still unexplained is an error naming the register.
That 332 VS read the fetch address while 702 pull their vertices themselves is seen; that the
call appears exactly when the stage data lists attributes is **[unverified]**.

**Root block.** Per draw the backend builds, from the recorded root arguments (`RootTable`,
`RootConstants`, `RootView` in `command.rs`) and the program's layout table, a *root block*:

```text
u32 user_data[32]   the values the console would load into user SGPRs (constants as given,
                    root descriptor = its 64-bit address, table = low 32 bits of its address)
u64 param_base[N]   per root slot: table → address of its first descriptor in heap memory,
                    root descriptor → its address; heaps (types 5, 6) → heap start
u64 view_table      address of the view table (Resources)
```

It reaches the shader as a push constant holding its device address (8 bytes; BDA is already
required by our backend). A user SGPR read is `user_data[i]`.

**Vertex input.** The console prefixes a fetch routine that loads attribute k's channels into
VGPR `first + k` **[VIctory]**. Ours: each VS input (stage data) → its `ISG1` row → semantic
name and index → the pipeline's `InputElement` with that name and index → a Vulkan vertex
attribute (location = input index, format from the DXGI format, binding = input slot, offset),
strides set at draw (`vkCmdBindVertexBuffers2`). The VGPRs start as those attribute values. The
corpus VS has no inputs and pulls its vertices itself through a raw buffer **[seen]**. System
VGPRs of a hardware VS: v0 vertex id, v3 instance id (v2 primitive id) **[VIctory]** →
`VertexIndex` / `InstanceIndex` (minus the bases, D3D ids start at 0).

**Pixel input.** `SPI_PS_INPUT_ADDR` enables input VGPR groups in a fixed order (persp sample,
center, centroid i/j pairs; pull model; linear sample/center/centroid; line stipple; pos
x/y/z/w; front face; ancillary; sample coverage; fixed pos) **[spec]**. Raising:
`v_interp_p1` + `v_interp_p2` of `attr n` with one barycentric pair = an `Input` variable at
that attribute's location: perspective center = default, centroid = `Centroid`, sample =
`Sample`, linear = `NoPerspective`; `v_interp_mov p0` = `Flat` (D3D and Vulkan both take the
first vertex as provoking); f16 forms read the half. Barycentric VGPRs used other than by
`v_interp` = error (or `VK_KHR_fragment_shader_barycentric`, later). Position → `FragCoord`,
front face → `FrontFacing`, ancillary → `SampleId`, coverage → `SampleMask`.

**Outputs.** VS `exp pos0` → `Position` (D3D clip space; the backend sets a negative viewport
height so y matches); `exp pos1..` with usage 7 → `ClipDistance`; `exp param n` → `Output` at
the location linked below. PS `exp mrt n` (after the 32_ABGR patch: raw 32-bit channels) →
`Output` location n, typed float/uint/int from the pipeline's `target_formats[n]` (Vulkan does
the UNORM/FP16 conversion); `exp mrtz` → `FragDepth` / stencil (`SPV_EXT_shader_stencil_export`)
/ `SampleMask` per its enable mask; `exp null` → nothing.

**Linking.** Pipelines are translated per VS+PS pair, so locations are assigned once: each VS
output row (`OSG1`) is matched by semantic name and index with a PS input row (`ISG1`), both
sides get the same location; an unmatched PS input reads its stage-data default.

## Resources

Our heaps do not hold RDNA2 descriptors: a view or sampler slot holds `"nVEW"` and a registry
index (`descriptor.rs`), slots 32 bytes (views) and 16 (samplers) (`info.rs`). A translated
shader must therefore never interpret the words it loads from a heap; it finds *which slot* a
descriptor operand comes from and looks the view up by its index.

**Tracing.** Every descriptor operand (SMEM `sbase`, MUBUF/MTBUF `srsrc`, MIMG `srsrc`/`ssamp`)
and every SMEM address is traced through SSA (copies, phis that agree, adds of constants or of
one computed value) to a `Source`:

| Source | Seen as | Translation |
|---|---|---|
| `Loaded { param, offset, dyn }` | `s_load_dwordx4/x8` from a table or heap param | slot address = `param_base + offset + dyn`; index = u32 at slot + 4 (BDA load); view = `view_table[index]` |
| `Built { addr, words }` | code ORs a root descriptor's address into a V# (`s_or_b32 s1, s9, 0x10000`, `s2 = -1`, `s3` constant) **[seen]** | raw buffer at that address, size from `num_records` (−1 = unbounded), stride from dword 1 |
| `Static(slot)` | root type 4 | sampler created at pipeline build from `RootSignature`'s static sampler |
| `Code(offset)` | `s_getpc` + offset | constant data, folded at translation |
| `Memory` | address loaded from memory or root constants | raw BDA at the computed address |

The layout and used tables say which root slot each user SGPR range is and how many dwords a
descriptor has; tracing cross-checks them and refuses mismatches. A loaded descriptor whose
words are used as numbers (bit extraction, compare) is an error first; fallback if a title
needs it: the view table also holds a synthesized RDNA2 descriptor per view.

**View table.** At submit the backend appends, for each registry entry not yet materialized,
a 32-byte record `{u32 kind, u32 format, u64 address, u32 size or width/height, u32
depth/layers/mips, u32 image slot, u32 sampler slot}` to a device buffer, and creates the
Vulkan object: `VkImageView` into `textures[]` (sampled) or `images[]` (storage), `VkSampler`
into `samplers[]`. One descriptor set, three `UPDATE_AFTER_BIND | PARTIALLY_BOUND` arrays sized
from device limits, bound once; slot = registry index (one namespace, holes allowed).
Buffers (CBV, raw/structured SRV/UAV) never need a descriptor: they are title memory, read by
`PhysicalStorageBuffer` pointers. Typed buffers (`buffer_load_format`, MTBUF) use the format
in the record and convert in the shader (chunk 9).

**Instructions.** SMEM `s_load*`/`s_buffer_load*` → BDA loads (bounds-checked against the
record size: out of range reads 0, as RDNA2 robust buffers do **[spec]**); MUBUF
`buffer_load/store_dword*` (`offen`/`idxen`, `inst_offset`, `soffset`) → BDA with the same
check; MIMG `image_sample*` → `OpImageSample*` on `OpSampledImage(textures[i], samplers[j])`
(`_l`, `_b`, `_d`, `_c`, `_o`, `_lz` variants), `image_load/store` → `OpImageRead/Write` on
`images[i]`, `image_get_resinfo` → `OpImageQuerySize(Lod)`; `dmask`, `dim`, `a16`, `d16`
shape the operands. FLAT/GLOBAL → BDA at the address. DS → a `Workgroup` u32 array of the
record's LDS size (compute later).

## SPIR-V emission

- **Our own builder** over the `spirv` crate (0.4.0+sdk-1.4.341.0, Apache-2.0 **[seen on
  crates.io]**: enums and constants only). A `Module` with sections (capabilities, extensions,
  memory model, entry points, decorations, types/constants deduplicated by key, functions),
  `finish() -> Vec<u32>`.
- **Target**: SPIR-V 1.5 (Vulkan 1.2, our floor), `PhysicalStorageBuffer64` addressing, GLSL450
  memory model; capabilities `Shader`, `PhysicalStorageBufferAddresses`, `RuntimeDescriptorArray`,
  `SampledImageArrayNonUniformIndexing` (`NonUniform` on indices from `dyn`), `Int64`,
  `DemoteToHelperInvocationEXT`, others on use (`Float16`, `ImageQuery`, `StorageImage…`).
- **Values are bits.** Registers are `u32`; each instruction bitcasts to its type (`f32`, `i32`,
  `f16x2`), applies modifiers (abs, neg, clamp, omod, opsel) and bitcasts back.
- **Exact semantics where they differ**: `v_mul_legacy_f32`/`v_mac_legacy` (0·x = 0),
  `v_min/max_f32` (IEEE mode: a NaN operand gives the other), `v_med3`, `v_cvt_pkrtz_f16_f32`
  (round toward zero: integer code), `v_rcp/rsq/sqrt` (host precision accepted), `v_frexp*`,
  `v_ldexp`, `v_bfe`/`v_bfi`/`v_alignbit`, 24-bit multiplies, carries of `v_add_co`/`v_addc`,
  `s_*` SCC rules, denorm/round mode from `s_setreg`/`s_round_mode`/`s_denorm_mode` (fixed
  per program: if it changes mid-program, error). One table entry per opcode, unit-tested on
  vectors (Xbox2Windows' measured data where it has some).
- **Stages**: hardware VS → `Vertex`, PS → `Fragment` (`OriginUpperLeft`, `DepthReplacing` if
  it writes depth, `EarlyFragmentTests` from `DB_SHADER_CONTROL`, later), CS → `GLCompute`
  (later). Wave64 and wave32 translate the same; the wave size only sizes masks.

## Validation and cache

- **Tests**: every translation test runs `spirv-val --target-env vulkan1.2` when it is in
  `PATH` (installed here: SPIRV-Tools v2026.4) and skips that step otherwise; CI installs it.
  Decoder tests diff `disasm()` with `llvm-mc` when present. Test inputs are hand-written GFX10
  sequences (assembled once with `llvm-mc -show-encoding`, checked in as hex with their
  source) and synthetic containers built by a test helper; never game bytes. A corpus test,
  `#[ignore]`, runs over `$NATIV64_SHADER_CORPUS` (the dump directory) and prints a coverage
  report: programs translated, first error per program grouped by message.
- **Runtime check**: `NATIV64_SHADER_VALIDATE=1` runs `spirv-val` on each new module and
  prints the error with the program's hash.
- **Cache**: `$XDG_CACHE_HOME/nativ64/shaders/<hash>.spv` (fallback `~/.cache`). Key = 64-bit
  FNV-1a hash (in-repo, no new crate) of {translator version, each stage blob, linking
  inputs: RT format classes, input layout}. The file starts with a header: magic, translator
  version, total input length and a second 64-bit hash of the same inputs (other seed),
  compared on load; a mismatch or a bad file is a miss, never an error. Bump the version on any
  translator change.
- **Dump**: `NATIV64_DUMP_SHADERS=dir` writes each pipeline's blobs at creation
  (`<hash>.<stage>.dxbc`) plus `pipelines.txt` (pipeline id, stage hashes, root signature,
  formats, `first_frame=N` and `draws=M` (draws and dispatches), rewritten at most twice a
  second and at presents); `NATIV64_TRACE_DRAWS=1` prints one line per presented frame (Vulkan: plus `drawn=` and `skips=` of that frame; frames
  1..120, then every 60th: draws by kind, clears, copies, pipeline ids); `nativ64-shader` (a bin
  of `nativ64-gpu`) prints a blob's parts, stage data, layout and disassembly, and translates it.

## Real validation

The milestone check (CLAUDE.md) is the title's first splash/loading screen, compared with what
the title shows on the console (Legal/Terms screens; frames by `NATIV64_DUMP_FRAMES`). The
order below makes the programs that screen draws with work first: chunk 1's dump names them.

## Chunk plan

One Sonnet agent per chunk, each with its tests (filtered output: failures and summary only);
Opus review after each. Module paths are in `crates/nativ64-gpu/src/shader/`. The corpus is
41% pixel, 32% compute, 18% mesh, 12% vertex: compute and mesh are part of the main path, not
later work. The order serves the first screen: each chunk lands for all four stages unless
it says otherwise.

1. **Container and dump** (done; `container.rs`, `native.rs`, `bin/nativ64-shader.rs`,
   `pipeline.rs` hook). `NATIV64_DUMP_SHADERS`.
2. **Decoder** (done; `gfx10/`). Every encoding of the table, flow following, `disasm()` in
   LLVM syntax, `llvm-mc` fixtures per encoding (`tools/gen_gfx10_ext.py` for the vector-memory
   and packed ones). The whole corpus decodes.
3. **IR** (done; `ir.rs`): blocks, SSA dwords, implicit operands, phis, entry `Arg`s, constant
   folding, `s_getpc` data.
4. **Entry and flow cleanup** (`ir.rs`, `native.rs`): CS and mesh stage data (`RSRC2`,
   group size, mesh record), `EntryRegs` from stage data for the four stages (table in Inputs
   and outputs), far jump → direct jump, VS fetch call → fetch point, then **constant-branch
   pruning**: an `s_cbranch_scc*/vcc*/exec*` whose condition folded to a constant becomes a
   jump, unreachable blocks go, phis are recomputed (dominators change). Pruning sits here,
   before structuring, so chunk 5 never sees dead edges (fewer `If`s, fewer irreducible false
   alarms). Acceptance: `--ir` on the corpus, every program builds or stops on a named
   unsupported opcode, none on an unexplained live-in or a register jump; the report gives the
   count per error.
4b. **IR lowering of the memory and packed encodings** (`ir.rs`): MIMG, MTBUF, DS, VOP3P,
   FLAT/GLOBAL/SCRATCH get their `dsts`, `srcs` and implicit reads (EXEC, M0 for DS, the old
   value of written VGPRs; stores and atomics as side effects). After chunk 4 they are the only
   reason 6947 of 8634 programs do not build (MIMG 4035, MTBUF 1881, DS 633, VOP3P 349, FLAT
   47). Acceptance: the corpus report shows only named unsupported opcodes, a handful at most.
5. **Structuring** (`structure.rs`): mask typing, per-lane branches, guards and their
   removal, discard → demote, Ramsey tree, multi-level break lowering plan. Tests: if, if/else,
   nested ifs, loop with break and continue, a 2-level break, an irreducible graph (error), a
   cross-lane use (error); corpus count of programs that structure.
6. **SPIR-V builder + ALU** (done; `spirv/`): module builder, types/constants, registers as
   variables, structured flow from chunk 5, scalar and VOP1/2/3/C ALU for f32/i32/u32 and the
   exact-semantics list, `v_cvt_pkrtz`, `v_fma_mix*` (2179 + 135 uses in the corpus: the only
   VOP3P seen). Tests: `spirv-val` on each test program; opcode vectors through a small CPU
   evaluator of our IR where a GPU is absent.
   Scope decisions (Opus, before chunk 6): I/O and resources come later, so chunk 6 emits a
   **probe entry**: `GLCompute`, `LocalSize 1 1 1`, one invocation = one lane; every entry
   register is a `Private` `u32` variable initialised from a `StorageBuffer` input array (one
   dword per live-in, index in `EntryRegs` order), and every register still live at
   `s_endpgm` is written to an output array, so tests run vectors on a GPU when one exists
   (lavapipe in CI) and on the CPU evaluator otherwise, with the same bytes. EXEC and masks
   are a `bool` per lane (the lane is always active in the probe); SCC/VCC a `bool`/`u32`
   per the mask typing of chunk 5. Memory, export, interpolation, `s_sendmsg` and wave
   instructions are not emitted: the translator stops on the first one with
   `Unsupported { chunk: 7|8|9, op }`. `nativ64-shader --spirv --stats` reports, over the
   structured programs, those that translate and validate, and the others by first
   unsupported op; `--spirv --alu-stats` counts programs whose every **ALU** instruction is
   supported (non-ALU ops ignored): the chunk's acceptance is ALU coverage, not full
   translation. `--spirv <blob> -o x.spv` writes a probe module.
   **As built** (`spirv/`: `builder.rs` module builder over the `spirv` crate, `alu.rs` the
   semantics, `emit.rs` the translation, `gpu.rs` the optional device runner of the tests).
   * One table, two backends. `alu::exec` is the per-opcode table, written once over a
     `Backend` of ~20 primitives on 32-bit bits (`ibin`, `fma`, `fcmp`, `select`, `pack_half`...);
     `Cpu` (the evaluator the unit tests run on) and the SPIR-V `Gen` both implement it, and
     `Dry` asks whether an opcode is supported (`--alu-stats`). Everything else (abs, neg,
     clamp, omod, opsel, SDWA selects, guards `v = active ? new : old`) is in the table.
   * Probe layout (`Probe`): bindings 0 / 1 of set 0, `inputs` = one dword per entry register
     sorted by `Reg` (EXEC's slot unused, EXEC starts true), `outputs` = one dword per
     register variable `(Reg, is_mask)`, written at each `s_endpgm`; a mask is 0 or 1. A wave64
     mask pair holds the lane's bit in both dwords (a pair is one bit, as the branch tests
     read it); the lane is console lane 0, so a scalar read as a mask is its bit 0.
   * Flow: `Block` and `Loop` are `OpLoopMerge` with a continue block that only goes back;
     a multi-level `Br` stores a `target` code (block * 2 + kind + 1) in a function variable,
     breaks, and the `forwards` re-tests after each merge (reset to 0 where it resolves).
     Phi dsts need no copies: they share the register variable with their inputs; only a
     phi input that is a folded constant is stored, at the end of its predecessor, after the
     branch condition is read. Demotes are ignored (the probe only honours EXEC).
   * ALU covered: SOP1/2/C/K (32-bit, `_b64` moves, shifts, mask ops, `saveexec`, SCC rules,
     `s_setreg` of anything but the f32 round / denormal bits of MODE, which is `Unsupported`),
     VOP1/2/3/C for f32 / i32 / u32 (+ i64/u64 compares and shifts, 16-bit shifts / compares /
     add), SDWA (select, sext, abs, neg, `dst_sel`, `dst_unused`), `v_cvt_pkrtz_f16_f32`
     (integer round-toward-zero code), `v_cvt_f16_f32` / `v_cvt_f32_f16` (`PackHalf2x16`),
     `v_pack_b32_f16`, `v_fma_mix*`, `v_cube*`, `v_cmp_class_f32`, `v_perm_b32`, `v_bfe/bfi`,
     `v_alignbit/byte`, carries, 24-bit multiplies. f16 arithmetic, `v_pk_*`, f64 and
     `v_swap_b32` are `Unsupported { chunk: 6 }` (plan item 11).
   * Tests (23 in `spirv/tests.rs`): `fixtures.txt` holds the `llvm-mc` hex of every
     instruction used (`[w64] ` prefix for wavefrontsize64); each case gives vectors, runs
     them on the evaluator, translates and validates with `spirv-val --target-env vulkan1.2`
     (skipped when absent). `NATIV64_SPIRV_RUN=1 NATIV64_SPIRV_DEVICE=llvmpipe cargo test -p
     nativ64-gpu spirv -- --test-threads 1` also runs every vector (and the integer sweeps)
     on the device and compares register by register: all 23 agree on llvmpipe.
   * Corpus (8610 programs with IR, 2862 structure): `--spirv --alu-stats`: **8524** programs
     have every ALU instruction supported (2810 of the 2862 structured ones; cs 2633/2717, ps
     3292/3294, vs 1030/1030, mesh 1569/1569); 2176 of 15.3 M ALU instructions are not (f16
     and `v_pk_*`: 38 programs at most per opcode, `v_swap_b32` 10, f64 5).
     `--spirv --stats`: 5 programs translate and validate (all cs); the other structured
     ones stop on their first non-ALU instruction: `s_load_dwordx4/x8/x16/dword` (chunk 9;
     1973 programs), `s_swappc_b64` (the VS fetch call, chunk 8; 324), `v_interp_*` (261),
     `exp` (295), `ds_write*` (4).
   * Two IR gaps found: `s_bitset0/1_b32` did not read their destination (fixed in `ir.rs`:
     it is now an old-value read like `s_cmov`), and `s_andn1/andn2_wrexec_b32/b64` (52 k uses)
     write EXEC but the IR lists only their SGPR destination: they are `Unsupported` chunk 7
     until `ir.rs` models the EXEC write (`exec = sdst = src & ~exec`, `andn1`: `~src & exec`,
     per the ISA text, unverified).
7. **Wave ops and speculation** (`structure.rs`, `structure/`, `spirv/`): "Chunk 7 design"
   above, in three sub-chunks (7a IR and speculation, 7b uniformity / spill slots / waterfall /
   lane index, 7c whole-wave sections / quad ops / ballots). Acceptance: `--structure --stats`
   near the IR count, the remaining refusals named by `--structure --waves`.
8. **I/O** (`io.rs`): user SGPRs and root block; VS system VGPRs and vertex attributes; PS
   input VGPRs and `v_interp` raising; exports and linking by semantic; CS (`GLCompute`,
   `LocalSize` from the group size, thread id VGPRs, group id SGPRs); mesh (`MeshEXT`, one
   invocation per lane, `exp pos/param` per vertex, `exp prim` → `PrimitiveTriangleIndicesEXT`
   and cull, per-primitive params, `s_sendmsg GS_ALLOC_REQ` → `SetMeshOutputsEXT`).
   Milestone: one program of each stage translates and validates.
   **As built (chunk 8, VS/PS)** (`io.rs` stage I/O and linking, `spirv/emit/stage.rs` the entry
   points; compute and mesh I/O are still to do). `io::translate_stage(&ParsedShader, targets,
   link)` builds the IR from the stage's code, structures it and emits a `Vertex` or `Fragment`
   module (`OriginUpperLeft`, `DepthReplacing` / `StencilRefReplacingEXT` on use); the probe
   entry stays for the ALU tests. `io::translate_pair` links first (`io::link`), then translates
   both.
   * Entry: every `Func.args` register gets a value. `s[n]` in the stage's user range is
     `user_data[n]` of the root block (push constant = its device address as `u64`,
     `OpConvertUToPtr` to a `PhysicalStorageBuffer` struct of 32 dwords, `Aligned 4` loads);
     VS `s4:s5` (fetch address) and PS `s[end]`, `s[end+1]` (primitive mask, ordered-section id,
     only read for `m0`) are 0; any other SGPR is `Unsupported { chunk: 8, "entry sN" }`. VS v0 =
     `VertexIndex - BaseVertex`, v3 = `InstanceIndex - BaseInstance` (capability `DrawParameters`),
     v2 (primitive id) = 0. PS VGPR groups follow `SPI_PS_INPUT_ADDR`: position → `FragCoord`,
     front face → `FrontFacing` as 0xffffffff / 0, ancillary → `SampleId << 8`, coverage →
     `SampleMask[0]`; the pull model, line stipple and fixed position are `Unsupported`.
   * Vertex input: the fetch call defines `first_vgpr + c` for each channel `c` of each stage
     input's mask from an `Input` vec4 at `location = input index`, typed by the `ISG1`
     component type (float, uint, sint). The channel placement (sparse, `first + c`, as
     `EntryRegs::from_stage` already had it) is **[unverified]**: the stage data's `last_channel`
     is not read.
   * Pixel input: a stage-data input is an `Input` vec4 at its location, read by channel.
     `v_interp_p1` (VINTRP and VOP3 forms) computes the whole value and `v_interp_p2` passes its
     old destination through; the decoration comes from the barycentric group of p1's VGPR
     (centroid, sample, linear = `NoPerspective`; the input's flat flag gives `Flat`);
     `v_interp_mov p0` is `Flat`. One mode per attribute (two is `Unsupported`), f16 forms,
     `mov p10/p20` and fp16 inputs are `Unsupported { chunk: 8 }`. A barycentric VGPR read other
     than by `v_interp` is refused; phis that only merge the entry value count as the entry value.
     An input with no vertex output of its semantic reads its default (`default & 3`: 0,0,0,0 /
     0,0,0,1 / 1,1,1,0 / 1,1,1,1, **[unverified]**, from the usual `SPI_PS_INPUT_CNTL` encoding).
   * Outputs: `exp pos0` → `Position`; `pos1/pos2` → `ClipDistance[8]` only when the stage data
     lists usage-7 outputs (other `pos1..3` uses are `Unsupported`); `exp param n` → `Output` at
     its linked location, dropped when no PS input reads it; `exp mrt n` → `Output` vec4 at
     location n, float / uint / sint from `dxgi_class(target_formats[n])` (float when unknown);
     `exp mrtz` channels 0, 1, 2 → `FragDepth`, `FragStencilRefEXT`, `SampleMask[0]` (channel 3
     is `Unsupported`); `exp null` nothing; compressed exports are `Unsupported`; an export under
     a lane guard is wrapped in the lane's condition.
   * Patches: the PS code is `io::stage_code`, with every export patch replaced by its `32_ABGR`
     variant, so every target exports four 32-bit channels (no `compr`).
   * Discard: the `s_andn2 live, live, exec` of a `Demote` becomes `if (exec bit) OpDemoteToHelperInvocation`
     (`SPV_EXT_demote_to_helper_invocation`); the branch to the exit is forced not taken, so the
     lane runs on as a helper.
   * Linking: `io::link` matches each VS parameter (`OSG1` row = register) with the PS input of
     the same upper-cased name and index (`ISG1` row = register); the shared location is the PS
     input's attribute number. A stage translated alone numbers its own (`param n`, `attr n`).
   * Memory and resource instructions (`s_load*`, `s_buffer_load*`, buffer / image ops) stay
     `Unsupported { chunk: 9 }`. `nativ64-shader --spirv --stats` uses the entries for the blob's
     own vs/ps record; `--spirv --pair <vs> <ps> -o out` writes `out.vert.spv` and `out.frag.spv`.
   * Tests (9, `spirv/tests.rs`, `mod stage`): synthetic VS / PS blobs from `fixtures.txt` and
     the container test helpers, each validated with `spirv-val`: fetch + `exp pos0` + `exp
     param`, `v_interp` pairs + `v_interp_mov` + `exp mrt0 done`, one decoration per barycentric
     group, barycentrics read elsewhere, linking by semantic, unmatched input, patch variant,
     chunk 9 still refused, discard.
   * Corpus (`--spirv --stats`): programs that translate and validate 365 of 3060 structured
     (was 5): vs 97 of 760, ps 263 of 1607, cs 5. Every other structured vs/ps stops on a
     chunk-9 op first (ps: `s_load_dwordx4` 707, `s_load_dwordx8` 636; vs: `s_load_dwordx4` 646,
     `s_load_dwordx8` 11, `s_buffer_load_dword` 6) except one ps (barycentrics read outside
     `v_interp`). The first-screen programs stop on `s_load_dwordx4` / `s_load_dwordx8` /
     `s_buffer_load_dword`, or translate (vs d2a2, ps 7ba0 and b40d).
9. **Resources and memory** (`resource.rs`): tracing table, BDA buffer loads with bounds
   (MUBUF, MTBUF typed formats, `global_load`), view table format, MIMG sample/load/resinfo/
   store, static samplers, DS → `Workgroup` LDS (compute, mesh). Tests: synthetic programs for
   each `Source`; one error test per refusal.
   **Chunk 9 design** (Opus, from the five first-screen blobs vs e5fcb71c, ps e2bcc91e, vs
   7fda1be8, ps c1d5a4b8, vs 28d887aa) **[seen]**:
   * A table user SGPR holds only the low 32 bits: the code sets the high SGPR to 0
     (`s_mov_b32 s9, 0`, then `s_load_dwordx4 s[4:7], s[8:9], off`). The layout rows of one
     user SGPR list several tables at `table dword` 0, 8, 16: the SGPR points at one block of
     32-byte slots and the SMEM offset (bytes) = `table dword × 4` picks the row; the used row
     with the same register says what it is (srv/cbv/sampler, 4 or 8 dwords).
   * Scope: **9a** (first screen) = SMEM `s_load*` / `s_buffer_load*`, MUBUF
     `buffer_load/store_dword*`, MIMG `image_sample` (+ `_l`, `_lz`); **9b** = MTBUF and `buffer_load_format` (typed),
     `global_*`, MIMG load/store/resinfo, DS/LDS, heap samplers from type-6 tables if unseen
     in 9a; each stays `Unsupported { chunk: 9 }` by name.
   * Root block grows: `u32 user_data[32]` (as chunk 8), then `u64 table_base[32]` (per user
     SGPR that the layout marks as a table: the full address its low 32 bits stand for, 0
     otherwise; filled by the backend, chunk 10), then `u64 view_table`. This replaces the
     per-root-slot `param_base` of Resources: the shader never needs root parameter numbers.
   * Sources, decided by tracing the descriptor operand's first dword through SSA: **Loaded**
     = SMEM load whose base low dword is a table user SGPR `n` (high dword folded to 0):
     slot address = `table_base[n] + offset (+ one dynamic value)`; registry index = u32 at
     slot + 4. **Built** = anything else (words computed by the program from user data or
     constants, e.g. a root SRV address ORed into a V#): decoded at run time as an RDNA2 V#
     (base = w0 | (w1 & 0xffff) << 32, stride = w1 bits 16-29, num_records = w2). S# that
     fold to constants (static samplers written by the code: `s_movk_i32 s0, 0x400`...) are
     **Code**: decoded at translation into a sampler description. A T# that is not Loaded, or
     a non-constant non-loaded S#, is `Unsupported`. A raw `s_load*` whose base is a 64-bit
     user SGPR pair (root descriptor) or computed is a BDA load at that address.
   * View table: 32-byte records indexed by registry index `{u32 kind (0 none, 1 buffer, 2
     image, 3 sampler), u32 dxgi format, u64 address, u32 size bytes (0xffffffff unbounded),
     u32 stride, u32 0, u32 0}`; only buffers read it. Images and samplers need no record:
     `textures[index]`, `samplers[index]` (set 0: binding 0 sampled images, binding 1 storage
     images, binding 2 samplers; one variable per (dim, sampled type) aliasing the binding).
     Static (Code) samplers: set 1, binding k = k-th distinct description, listed in the
     translation output for the backend to create as immutable samplers.
   * Bounds: `s_buffer_load*` and MUBUF reads past `size` (Loaded) or `num_records` (Built,
     stride 0 → bytes) give 0 per dword; stores past it are dropped.
   **As built (chunk 9a)** (`resource.rs` the data the backend gets, `spirv/emit/resource.rs`
   the tracing and the instructions; the design above stands, with what is said here).
   `io::translate_stage` returns `Translated { words, samplers }` (`translate_pair` two of
   them): `samplers[k]` is the static sampler at set 1, binding k.
   * Root block: `RootBlock { u32 user_data[32]; u64 table_base[32]; u64 view_table }`, offsets
     0, 128, 384, read in the entry block (the pointer dominates every later use).
     `StageIo::tables()` lists the user SGPRs the layout marks as tables or heaps (root types
     3, 5, 6, `first dword`); the layout's `used` rows are **not** cross-checked against the
     loads yet (a row's descriptor size and kind are not compared with what the code does).
   * Tracing (`analyze`, before emission): a `Loaded` descriptor is the result of an
     `s_load_dword*` whose base low dword is a table SGPR plus anything (arg, `s_mov`,
     `s_add_u32` of that and any value, phis that agree) and whose high dword folds to 0
     (constant, or `s_addc_u32` of zeros; another high dword is `Unsupported`). The load writes
     `index = u32 at table_base[n] + (low - user_data[n]) + imm + soffset + 4` to dword 0 of
     its result and 0 to the rest, so a dynamic offset is whatever the program added to the
     low dword; copies and phis of those dwords stay loaded. A loaded dword read other than as
     the resource operand of `s_buffer_load*`, MUBUF or MIMG (or copied) is `Unsupported {
     chunk: 9, "the words of a loaded descriptor used as numbers" }`.
   * Buffers: a `Loaded` buffer is `view_table + index * 32` (kind must be 1, else empty;
     size 0xffffffff is unbounded; stride from the record); anything else is `Built`: the four
     words read at run time as a V# (base = w0 | (w1 & 0xffff) << 32, stride = w1 bits 16-29,
     size = `num_records` bytes, times the stride when it is not 0; -1 unbounded). Every dword
     read is `OpConvertUToPtr` + `Aligned 4` load under `if (offset + 4k + 4 <= size)`, else 0
     (a `Function` temporary holds the result); a store under the same condition (and the
     lane's EXEC bit) is skipped. Offset = `soffset + inst offset + voffset (offen) + index *
     stride (idxen)`; SMEM: immediate + optional SGPR. A raw `s_load*` (base not a table)
     reads at `lo | hi << 32 + offset`, unbounded; so do root descriptors.
   * Images: `image_sample`, `_l`, `_lz`, dmask any, dims 1D, 2D, 3D, cube, 2D array. The T#
     must be `Loaded`: `textures[index]` (`OpTypeImage` f32 per dimension, `UniformConstant`
     runtime array, set 0 binding 0, one variable per dimension aliasing the binding, `NonUniform`
     on the index and the loaded objects; capabilities `RuntimeDescriptorArray`,
     `SampledImageArrayNonUniformIndexing`, `ShaderNonUniform`). The S# is four constants
     (decoded to a `SamplerDesc`, deduplicated, set 1 binding k) or `Loaded` (`samplers[index]`,
     set 0 binding 2). `OpImageSampleImplicitLod` for `image_sample` (refused in a vertex
     stage), `ExplicitLod` for `_l` (last address dword) and `_lz` (0.0). Refused: a16, d16,
     tfe, lwe, unorm, MSAA, 1D array, the other sample variants (`_b`, `_c`, `_d`, `_o`) by name.
     Cube: the hardware address `(s, t, face)` is turned back into a direction (`2 * (x - 1.5)`
     per face plane, the usual face order and signs) **[unverified]**: the `1.5` and the face
     numbering are from the ISA text, not from a captured cube sample.
   * Still `Unsupported { chunk: 9 }` by name: MTBUF, `buffer_load_format*`, `buffer_atomic*`,
     tfe / lds on MUBUF, `global_*` / `flat_*`, DS, `image_load/store/resinfo/atomic`,
     `s_store`, `s_dcache*`. A loaded S# is read from `slot + 4` like a view (**[unverified]**:
     no type-6 sampler table has been seen in a blob).
   * Guesses **[unverified]**: the S# field layout (words 0-3, as `SQ_IMG_SAMP_WORD0..3`;
     checked against another decoder in the reference mirror for filters, address modes,
     anisotropy, compare, LOD range and bias, border type) and the V# layout come from the
     public tables; a structured V# (stride > 0) is bounded by `num_records * stride` bytes
     instead of the hardware's index and offset checks; the out-of-range rule is applied per
     dword (a dword that straddles the end reads 0); helper invocations are not excluded from
     stores.
   * Tests (synthetic VS/PS from `fixtures.txt`, each validated with `spirv-val`): loaded
     CBV + `s_buffer_load`, dynamic table and buffer offsets, Built V# from a root descriptor
     with offen / idxen loads and a store, raw `s_load`, loaded T# + constant S# (and
     `SamplerDesc` equality), loaded S#, every sample form and dimension; one refusal each for
     a T# not loaded, a non-constant S#, descriptor words used as numbers, implicit LOD in a
     VS and four 9b operations by name; `SamplerDesc` decoding. The out-of-range rule has no
     execution test: the probe harness runs `GLCompute` with no root block, and no stage
     can be run on the CPU evaluator yet (chunk 10's llvmpipe run covers it).
   * First screen (`--spirv --stats`): the five blobs (vs e5fcb71c, ps e2bcc91e, vs 7fda1be8, ps
     c1d5a4b8, vs 28d887aa) all translate and validate; `--pair` on (7fda1be8, c1d5a4b8) and
     (e5fcb71c, 7ba0e3e2) writes valid modules. Corpus: programs that translate and validate
     1368 of 3060 structured (was 365): vs 436 of 760, ps 927 of 1607, cs 5. First unsupported
     op of the rest: cs `s_load_dwordx4` 548 (compute I/O is not built), vs `buffer_load_format_x`
     221, ps a sampler neither constant nor loaded 174, ps `image_gather4_lz` 148, cs
     `s_load_dwordx8` 126, ps `image_load` 112, ps and vs loaded descriptor words used as
     numbers 109 and 84, ps `image_get_lod` 64, ps `image_store` 50.
10. **Vulkan pipelines, draws and dispatches** (`backend/vk/pipeline.rs`): bindless set and view
   table upkeep at submit, root block per draw, graphics (VS+PS) and mesh (MS+PS,
   `VK_EXT_mesh_shader`) pipelines from translated stages + `GraphicsState`, compute pipelines,
   cache, `vkCmdDraw*` / `vkCmdDrawMeshTasks*` / `vkCmdDispatch*` replay. Real validation: the
   first screen against the expected frame.
   **Chunk 10 design** (Opus, from the first-screen trace and pipeline 56: vs e5fcb71c + ps
   7ba0e3e2) **[seen]**: vs e5fcb71c has one layout row (root type 1, first dword 8, 2 user
   SGPRs, root slot 0) and one used SRV row; its code builds a V# from `s8`, `s9 | 0x10000`,
   `-1`, `0x11014fac`, reads dwords 0 and 0xc of that buffer (low nibble = a kind, 3 / 4; bits
   7.. = offset, bits 3..8 = stride / 4), then loads position and colour at `offset + stride *
   v0`. No vertex buffer, no texture; the ps reads its interpolated TEXCOORD. So frame 2 needs
   memory, a root block, a graphics pipeline and `Draw`, nothing else.
   * Split: **10a** = GPU memory, root block, bindless set (empty), buffer view records,
     graphics pipelines (VS + PS), `Draw` / `DrawIndexed` into render targets, vertex and
     index buffers; dispatches, mesh, indirect and draws needing an image stay counted and
     skipped. **10b** = textures (title images detiled into `VkImage`s by `texture/`, views
     into `textures[]`, heap samplers into `samplers[]`), re-upload rule, then frame 4 and
     frame 565. **10c** = compute (after the cs I/O of chunk 8), mesh.
   * Vulkan floor becomes **1.3** (all three drivers here are 1.4): `dynamicRendering`,
     `extendedDynamicState` (topology, cull, front face, depth state, vertex strides
     dynamic), `synchronization2` if replay wants it; plus 1.2 `bufferDeviceAddress`,
     `descriptorIndexing` (update-after-bind, partially bound, runtime arrays, non-uniform
     indexing), `shaderDemoteToHelperInvocation` (1.3 core). `Gpu::open` refuses a device
     without them, naming the missing feature.
   * **GPU memory: title addresses are host addresses** (`xmem.rs`: graphics memory 4 GiB -
     1 TiB, `VirtualAlloc` above 1 TiB). Every address the backend hands a shader is
     translated **on the CPU** into a device address, never in the shader: root CBV / SRV /
     UAV pairs written into `user_data`, `table_base[n]`, view record addresses, vertex and
     index buffer bindings. `GpuMemory` (backend) keeps imported `HostBuffer`s by host range;
     `translate(va) -> Option<(vk::Buffer, offset, device address)>` finds the import holding
     `va`, else imports the committed run around it (`api_ms_win_core_memory::query`: the
     run of committed pages of that allocation, rounded to the import alignment), else
     fails (logged once per allocation; the draw is skipped). Memory outside the table that
     a shader reads: our own descriptor heaps, which become page-aligned and page-rounded
     allocations listed by `descriptor.rs` (`heap_ranges()`). Lifetime: the memory table
     and `xmem` (alias / unmap of physical pages) record every range they decommit,
     release or remap (`take_gpu_changes() -> Vec<(usize, usize)>`, cheap when empty); at the
     start of each `execute` the backend drops the imports overlapping a changed range (the
     queue is idle between submits: every `ExecuteCommandLists` waits). A device address
     must stay below 2^48 (a Built V# keeps 16 high bits): checked at import.
     Not covered: an address the program reads from memory (a pointer stored in a buffer),
     which reaches the GPU untranslated: `Translated` gains `memory_addresses: bool`, set when
     a raw SMEM load or a Built V# has a base that does not trace to user data; the backend
     skips such pipelines with a log.
   * **Root block per draw**: `{u32 user_data[32]; u64 table_base[32]; u64 view_table}` (392
     bytes, chunk 9a) written into a per-submit host-visible arena (256-byte aligned
     records), its device address in an 8-byte push constant (VS | FS). Filled from the
     list's current root signature and arguments (`RootSignature`, `RootTable`,
     `RootConstants`, `RootView`; graphics and compute kept apart) by the **stage's layout
     rows** (`first dword` = SGPR n, `user sgprs` = count, `slot` = root parameter): a
     constants parameter → its 32-bit values (from `table dword`); a root descriptor → the
     translated address as two dwords; a table → the low 32 bits of the translated handle
     address, the full one in `table_base[n]`. Rows of both stages are merged (same root
     signature). `view_table` = device address of the view buffer.
   * **Bindless and view table**: one descriptor set layout (set 0: binding 0 sampled
     images, 1 storage images, 2 samplers; `UPDATE_AFTER_BIND | PARTIALLY_BOUND`, sized by
     `min(device limit, 1 << 20)`), one pool, one set, bound for every draw; 10a writes no
     image. The view buffer (device-local + host-visible if available, else host-visible) has
     one 32-byte record per registry index (`descriptor.rs` `VIEWS`), grown by doubling;
     at each `execute` records for new indices are written: buffer SRV / UAV / CBV views get
     kind 1, the translated address, size in bytes (`0xffffffff` for none), stride (structured
     buffers; 0 raw); everything else kind 0 (10b fills images and samplers).
   * **Pipelines** (`backend/vk/pipeline.rs`), built at the first draw that uses them, cached
     by pipeline id with its outcome (`Built` or `Failed(reason)`, logged once): stages from
     `io::translate_pair` (VS + PS; a missing PS = depth only, VS alone), modules validated
     by `NATIV64_SHADER_VALIDATE`. Layout = set 0 + set 1 (the pair's static samplers as
     immutable samplers, deduplicated across both stages; the PS's numbering must be
     remapped or the pair shares one list: decide in `translate_pair`) + the push constant.
     Fixed state parsed from `GraphicsState.raw` (description and stream forms, same reader
     as the formats): blend per target (independent or not, logic op refused), write masks,
     alpha-to-coverage, sample mask, rasterizer (fill, cull, front CCW, depth bias / clamp /
     slope, depth clip → `depthClampEnable = !clip`, conservative refused), depth-stencil,
     IB strip cut. Formats → dynamic rendering formats (`format::lookup`). Vertex input from
     `input_layout`: each element matched with the VS input of the same semantic name and
     index (ISG1 row = location), binding = input slot, rate per class / step rate (step
     rate > 1 needs `vertexAttributeInstanceRateDivisor`: refuse for now), stride dynamic.
     Topology: the type's first member, real topology dynamic. Dynamic: viewport, scissor,
     blend constants, stencil reference, depth bounds, topology, vertex strides.
   * **Draws**: replay keeps the list's state (pipeline, render targets, viewports, scissors,
     IB, VBs, root arguments, heaps); at a draw it begins dynamic rendering on the current
     targets (the `TitleImage`s of the RTV / DSV records, views made per record and cached),
     ending it at the next non-draw command or target change. Viewport `y + height, -height`
     (the D3D y-down convention); front face with that flip: `FrontCounterClockwise` false →
     `CLOCKWISE` **[unverified]**. `Draw` → `vkCmdDraw`, `DrawIndexed` → `vkCmdBindIndexBuffer`
     (translated address; R16 / R32) + `vkCmdDrawIndexed`. v0 stays `VertexIndex -
     BaseVertex` (chunk 8): with a non-zero first vertex or base vertex log once, the run says
     whether the title relies on it. A draw whose pipeline failed, whose memory did not
     translate, or with no render target is skipped (counted in `ReplayStats` by reason).
   * Tests (lavapipe): GPU memory (import on demand, reuse, drop after a recorded change,
     refusal outside the table); root block filling from a synthetic root signature + layout
     rows; a real draw: a synthetic VS (vertex id → position read from a root SRV buffer,
     like vs e5fcb71c) + PS (colour from a root constant) into a cleared RGBA8 target, read
     back and compared per pixel; an indexed draw with a vertex buffer; a pipeline that fails
     translation is skipped and counted.
   * Real validation of 10a: the title's frame 2 (`NATIV64_DUMP_FRAMES`) shows pipeline 56's
     draw (not the clear colour), frames 4+ the draws of 50545 / 50546 that need no image;
     compare with the console's first screen.
   **As built (chunk 10a)** (`backend/vk/{gpumem,rootblock,bindless,pipeline,draw}.rs`,
   `d3d12/pipeline_state.rs`; the design above stands, with what is said here). Draws replay:
   `Draw` / `DrawIndexed` into the images of the bound render and depth views, with the VS+PS pair
   of the pipeline object.
   * Floor: `Gpu::open` wants Vulkan 1.3 and names the first missing feature (`shaderInt64`,
     `depthClamp`, `independentBlend`, `shaderDrawParameters`, buffer device address, timeline
     semaphores, descriptor indexing: runtime arrays, partially bound, update-after-bind and
     non-uniform indexing of sampled / storage images, `dynamicRendering`, `synchronization2`,
     demote to helper invocation). Optional ones are enabled when present (clip distance,
     sample shading, wireframe, depth bounds, anisotropy, `VK_EXT_shader_stencil_export`).
   * GPU memory (`gpumem.rs`): as designed. The memory table (`nativ64::api_ms_win_core_memory`)
     records decommitted, released, replaced and remapped ranges (`mark_mapped` covers the
     `xmem` alias / unmap, so `xmem.rs` is unchanged) once `watch_gpu_changes` was called, and
     `take_gpu_changes` hands them out; `committed_run(addr)` is the unit of an import. A run is
     imported whole, shrunk inwards to the import alignment (so a run smaller than it does not
     import), and a descriptor heap that owns its memory (now page-aligned, page-sized, listed
     by `heap_ranges()`, its drop recorded as a change) is imported like title memory. The
     device address plus length must stay below 2^48. A range that runs past its import does not
     translate (no merging of neighbouring imports).
   * `Translated` gained `images` (a texture, sampler or static sampler is read) and
     `memory_addresses` (a raw scalar load or a built V# with a word that is not scalar
     arithmetic on user data and constants, e.g. a pointer loaded from a buffer). Both are
     decided by the tracing pass; a pipeline with either is `Failed` and its draws skipped.
     Decision for the pair's static samplers: none are supported in 10a (they come with images
     in 10b), so the pipeline layout is set 0 and the push constant only; `translate_pair` is
     unchanged and 10b picks between a shared list and a renumbering.
   * Root block (`rootblock.rs`): as designed (392 bytes, 512-byte records in a per-submission
     host-visible arena, its device address pushed for VS | PS). Rows of both stages go into one
     block, so a user SGPR that means two different things in the two stages (different root
     type or parameter) refuses the draw (`state`) instead of letting one overwrite the other;
     real root signatures should not do this **[unverified]**. Heap rows (types 5, 6) are filled
     as tables from the start of the first bound views / samplers heap **[unverified]**: no
     blob with such a row has run. Compute arguments are not tracked.
   * Bindless set and view table (`bindless.rs`): set 0 with three runtime-sized bindings
     (`min(device limits, 1 << 20)`, update-after-bind, partially bound), one pool and set,
     bound at every draw, never written. The view buffer is host-visible (device-local when
     the device has such memory), doubled when the registry outgrows it. Records of new
     registry indices are written at the first draw of a submission; a buffer view whose
     memory does not translate stays kind 0 and is tried again at each submission; all are
     rewritten when a recorded memory change happened. A view's resource is read through
     `resource::info_of_checked` (the registry never forgets a view, so the resource may be
     released or, in tests, fake).
   * Pipelines (`pipeline.rs`), cached by a serial the pipeline object now carries (an object's
     address can be reused), not by `shader::note_pipeline`'s id. Fixed state comes from
     `pipeline_state::fixed_state` (both raw forms; absent subobjects default as in D3D12).
     Refused with a reason (logged once, draws skipped as `pipeline`): stages other than VS and
     PS, compute, a blend logic operation, dual-source factors, conservative rasterization,
     more than one sample, patch topology types, instance step rates above 1, vertex formats
     without a mapping, a stage that needs images or reads untranslated addresses. Static
     rather than dynamic: cull mode, front face (`FrontCounterClockwise` false is `CLOCKWISE`
     **[unverified]**: the test pipelines cull nothing), depth test, blend, depth bias (the
     clamp and the depth bounds test are ignored). Dynamic: viewport, scissor, blend constants,
     stencil reference, primitive topology (kept in the pipeline's class), vertex strides and
     primitive restart (on only for a strip topology when the pipeline has a strip cut).
     `NATIV64_SHADER_VALIDATE` runs each module through `spirv-val` (a failure is logged and the
     pipeline is still built). A VS attribute without a semantic or without an input element is
     logged and left unfed. Pipeline targets must match the bound images' formats (typeless
     views and sRGB reinterpretation are not handled): a mismatch is a `target` skip.
   * Draws (`draw.rs`): list state is reset per list; rendering stays open over draws and state
     changes on the same attachments and ends at any other command, with a global barrier;
     the viewport is `y + height, -height`; one viewport and scissor (the first); a target view
     of several layers draws into its first. `ReplayStats` gains `drawn` and `draw_skips`
     (`pipeline`, `memory`, `target`, `state`); `ExecuteIndirect` counts as a `state` skip,
     dispatches and mesh dispatches stay counted only. A non-zero first or base vertex is logged
     once.
   * Discrepancy found, not fixed: `d3d12/pipeline.rs` reads the input layout, formats and
     samples of the description form 8 bytes above the standard header layout (it counts the
     stream output as 40 bytes, the header's is 32), while `pipeline_state.rs` uses the standard
     offsets (blend 120, rasterizer 452, depth-stencil 496). Titles seen so far create pipelines
     from streams, so neither was exercised **[unverified]**.
   * Tests (each GPU test also run on lavapipe with `NATIV64_GPU=llvmpipe`): memory table
     records and committed runs; GPU memory (import on demand, reuse, drop after a recorded
     change, refusal outside the table, heap memory); the flags of `Translated`; both raw
     forms of the fixed state; root block filling and its refusals; view records of buffer
     views, constant buffer views, unreachable memory and textures; real draws read back per
     pixel: a VS pulling its position from a root SRV (as vs e5fcb71c) with a PS writing root
     constants, under a top-half viewport and a left-half scissor, then a second draw into the
     bottom half reusing the pipeline; an indexed draw with a vertex buffer and a base vertex; a
     depth-tested pair of draws; skips counted by reason (no pipeline, a PS that samples, a root
     descriptor outside GPU memory, no target, a topology of another class).
   * Skipped (still counted): compute (10c), mesh and amplification, indirect, draws that read
     images (10b), multisampling, stream output, predication, queries.
   * Real validation of the title's frame 2 (pipeline 56) against the console's first screen is
     **not done** in this step (the title was not run); the synthetic draws above are the
     evidence so far.
   **Chunk 10b design** (Opus, 2026-11-26, from the 10a run and the blobs) **[seen]**: the log
   names pipelines by serial, so serial 50546 (refused "topology type 19") and 50547 (refused
   "images") are trace pipelines 50545 and 50546. Trace 50546's ps c1d5a4b8 is a textured,
   vertex-coloured quad: `s_load_dwordx8` of a T# at table SGPR + 0x20 (SRV row: table dword 8,
   register 277), a constant S# (`0x400, 0xc00000, 0x18f00000, 0`), `image_sample` 2D, times
   COLOR0. 19 is not a topology type: it is the console's `D3D_PRIMITIVE_TOPOLOGY` **QUADS**
   (VIctory `crates/victory-gpu/src/draw.rs:2023-2026`: 17 rectangles, 18 line loop, 19 quads,
   20 quad strip); vs d2a23874 calls a fetch shader (`s_swappc_b64`).
   * **Images of title resources**: `VkBackend::images` stays the one map from resource to
     `TitleImage` (`replay.rs`). Each `TitleImage` gains an owner: **GPU** once it has been a
     render or depth attachment, a clear, copy or resolve destination, or an unordered-access
     target (its host memory is stale from then on and is never read again); **host**
     otherwise. Images get `MUTABLE_FORMAT` (views may reinterpret the format: typeless,
     sRGB) and `CUBE_COMPATIBLE` when 2D with a multiple of 6 layers; 3D resources get 3D
     images (1D refused by name). A host image is uploaded at the first submission that
     samples it, then again at a later one only if its content changed: an FNV/xx-style
     64-bit hash of its whole host memory (the console layout's size) is taken at the first
     sampled use in each submission and compared with the last upload's (counted: bytes
     hashed, uploads, in `ReplayStats`; VIctory tracks pages with write guards instead, its
     `watch.rs`: our upgrade path if the hash shows up in a profile). Upload: `texture::Layout`
     from the resource's description, `detile` of every mip and array slice of plane 0 into a
     staging buffer laid out linearly (BC formats by blocks), `vkCmdCopyBufferToImage` per
     subresource, then a global barrier; the image stays in `GENERAL`. A format or layout
     `texture/` does not handle (planar, depth planes, MSAA) is refused with a log and the
     view stays empty.
   * **Texture views into `textures[]`** (set 0 binding 0, `bindless.rs`): at the same point
     as buffer records, an SRV (`View::Shader`) of a texture gets a `VkImageView` written at its
     registry index: view type from the description's dimension (2D, 2D array, cube, cube
     array, 3D; none = the resource's own), format from the description (`format::lookup`),
     mip and layer ranges, `Shader4ComponentMapping` → `VkComponentMapping` (forced 0 / 1
     included). The view record stays kind 0 (shaders do not read records for images). Views
     cached by (image, description) and destroyed with the image. A `View::Placed` SRV
     (memory with a resource description, device slot 70) gets an image of its own keyed by
     (address, description words), host-owned; when its range overlaps a GPU-owned image log
     once (not merged). UAV images (binding 1) stay for 10c. **Holes**: every registry index
     a stage could read through binding 0 that holds no texture (never written, a buffer, a
     failed view) gets a 1x1 zero RGBA8 2D view, so no shader reads an unwritten descriptor
     (written for new indices at each update; an index read through a cube or 3D variable
     still mismatches: log the view type mismatches you can detect, nothing more).
   * **Samplers**: static (Code) samplers are immutable samplers of set 1; `translate_pair`
     keeps one deduplicated list for the pair (the PS's indices renumbered after the VS's) and
     the pipeline layout becomes set 0 + set 1 + push constant (set 1 layout cached by its
     sampler list). Heap samplers: `View::Sampler` (the 13-word D3D12 description) and
     `View::ConsoleSampler` (its first words in the same order, console tuning ignored, log
     the words once) become `VkSampler`s written at their registry index in binding 2, cached
     by description. `SamplerDesc` / D3D12 description → `VkSamplerCreateInfo`: filters, mip
     mode, address modes (mirror once needs 1.2 core `samplerMirrorClampToEdge`: check), LOD
     bias / min / max, anisotropy (only when the feature is on), compare op, border colour
     (transparent black, opaque black, opaque white; another colour → the nearest with a
     log). The `images` flag of `Translated` no longer fails a pipeline; a stage that needs
     what 10b does not build (storage images, a texture or sampler type `translate` refuses)
     still fails by name.
   * **Console topologies**: `topology()` (`backend/vk/pipeline.rs:182`) and the stream's
     topology subobject (`d3d12/pipeline.rs:441`): log the raw value of trace pipeline 50545
     (serial 50546) and the topology its list sets, to find where 19 comes from (a console
     topology *type*, or a `D3D_PRIMITIVE_TOPOLOGY` in that field). QUADS (19) draw as triangle
     lists through index expansion: `Draw(n, first)` → a generated index buffer of `n / 4`
     quads `(0 1 2, 0 2 3)` (host-visible, cached by count, `first` as vertex offset);
     `DrawIndexed` of quads → the indices read from title memory (translated), rewritten into a
     per-submission buffer. Rectangles (17), line loop, quad strip: refused by name (not seen).
   * **Loose ends from 10a**: (1) VUID-vkCmdDraw-pStrides-04913: a pipeline with no vertex
     input binding does not declare the dynamic stride state and binds no vertex buffer;
     (2) the GPU memory miss `0x10b00000 (+0xae00000) does not fit an import of
     0x10b00000..0x12600000`: find whether the run grew by a later commit (commits are not
     recorded as changes) or the range is past the allocation; if growth, a translate that
     runs past its import re-reads the committed run and re-imports it when bigger, the old
     import kept alive until the submission is done; (3) the description form of
     `d3d12/pipeline.rs` reads 8 bytes too far (stream output 32 bytes, not 40): fix and test
     against `pipeline_state.rs`' offsets.
   * Tests (lavapipe, `NATIV64_GPU=llvmpipe` too): detile + upload of a tiled RGBA8 texture
     (non-linear swizzle mode) with 2 mips, sampled by a synthetic PS (loaded T# + constant S#
     like ps c1d5a4b8) into a target, read back per pixel; the same with a BC1 texture; a host
     texture re-uploaded after its memory changes and not re-uploaded when it does not
     (counters); a render target drawn in one list and sampled in the next (GPU-owned, not
     overwritten from host); a heap sampler (`View::Sampler`) with clamp vs wrap seen in the
     read-back; a hole read as zero; a quad `Draw` and an indexed quad draw covering the
     target; the pair's static samplers deduplicated and renumbered.
   * Real validation of 10b: the title's frames 4+ (`NATIV64_DUMP_FRAMES`) should show trace
     pipeline 50546's textured quads; compare with the console's first screen.
   **As built (chunk 10b, part 1)** (`backend/vk/{textures,sampler}.rs`, `replay.rs`,
   `bindless.rs`, `pipeline.rs`, `shader/io.rs`; textures and samplers, not the quad draws nor the
   loose ends, which are part 2; the design above stands, with what is said here).
   * Ownership (`TitleImage`): host until a render or depth target, a clear or a copy destination
     (`VkBackend::own`), then GPU for good. Images are `MUTABLE_FORMAT` (and `CUBE_COMPATIBLE`
     when square with a multiple of 6 layers), 3D resources get 3D images (`Img::depth`; 3D
     copies are skipped), 1D and multisampled ones none (logged). BC1 to BC7 map to Vulkan
     (`textureCompressionBC`, enabled when present; copies into or out of buffers of BC images are
     skipped). `mip_levels` 0 means the whole chain (it was read as 1).
   * `sync_textures` runs at the start of every submission with a draw (not at the first sampled
     use: which texture a draw reads is not known, so every texture that has a view is looked at):
     registry entries since the last time are classified, each texture gets its image and, if
     the host owns it, one hash per submission (`ReplayStats::hashed`, 4-lane multiplicative, over
     `Layout::size` bytes, only when the memory is committed in the memory table) and an upload
     when it changed (`uploads`; per-image counter too): `detile` of every mip and slice of the
     one colour plane into one staging buffer, one copy per subresource, one global barrier.
     Depth and planar formats are not uploaded. A copy from a host image uploads it first
     (not in the design). Descriptors: binding 0 gets the view of each texture SRV (`texture_view`,
     cached on the image: 2D, array, cube, cube array, 3D, the mapping, mip and layer ranges, the
     view's format by `format::lookup`; a depth image is viewed as depth with its own format),
     a 1x1 zero RGBA8 view everywhere else; binding 2 gets a sampler per heap sampler view and a
     default (bilinear, repeat) one everywhere else. All are written again when an image is
     replaced. `View::Placed` SRVs make an image of their own (host-owned, same upload), the
     address read from view words 10 and 11 **[unverified]**; an overlap with a GPU-owned image
     is only logged.
   * Samplers (`sampler.rs`): `Spec` from `SamplerDesc` and from the D3D12 description (the first
     13 words of `View::ConsoleSampler`, logged once), one `VkSampler` per spec. Approximated and
     logged: min / max reduction (average), a border colour not one of three (nearest), the
     hardware's table border colour (transparent); the hardware's half-border and mirror-once
     address modes map to `MIRROR_CLAMP_TO_EDGE` or `CLAMP_TO_BORDER` **[unverified]**
     (mirror-once is clamp-to-edge when the device lacks it); anisotropy only with the feature;
     a comparison sampler only from a D3D12 description.
   * Static samplers: `translate_pair` returns one list for both stages (the PS's renumbered by
     rewriting the `Binding` decorations of set 1); `Bindless::layouts` makes, per list, a set
     layout of immutable samplers, one set from a pool and a pipeline layout (set 0, set 1, push
     constant; at most 16 samplers), cached by list. `Built` carries them and a draw binds set 1.
     The `images` refusal is gone from `pipeline.rs`.
   * Not in the design: a draw whose root signature has a descriptor table or heap argument
     the list never set now fails (`memory` skip) instead of leaving a null table, which the
     shader would read at address 0 (`rootblock::fill`).
   * Unverified: the console sampler's word order beyond the first 13; the sRGB and typeless
     view reinterpretations on real hardware; view type mismatches (a cube read through the 2D
     variable) cannot be detected, so none is logged; the cost of hashing every texture with a
     view in every submission has not been measured on the title.
   * Tests (lavapipe too): detile of a tiled 2-mip texture checked texel by texel through
     copies, upload counters (unchanged: none, changed: one); the view's mapping, mip range and a
     sampler view read as a texture (hole = zero); a render target drawn in one list, sampled in
     the next, not overwritten from host; heap sampler wrap vs clamp; a BC1 texture; a placed
     view; the pair's samplers deduplicated and renumbered (shader level); layouts of static
     samplers cached by list; BC formats; sampler specs. Run: all pass (a real GPU and
     `NATIV64_GPU=llvmpipe`). The real validation of 10b is not done.
   **As built (chunk 10b, part 2)** (`backend/vk/{draw,pipeline,gpumem}.rs`,
   `d3d12/{pipeline,pipeline_state}.rs`; the design above stands, with what is said here).
   * Console topologies: a pipeline topology type of 17 to 20 builds as a triangle list
     (`pipeline.rs::topology`), `Built` keeps the raw type. `d3d12/pipeline.rs` logs the first 8
     pipelines created with a type of 17 or more (stream or description, serial); a draw logs once
     the pipeline's raw type and the list topology when either is 17 or more. List topology 19
     draws as a triangle list through an index buffer: `Draw(n)` uses a buffer of `n / 4` quads
     `(0 1 2, 0 2 3)` as u32 (cached by quad count, `first` as vertex offset); `DrawIndexed` reads
     its indices from title memory on the CPU (u16 or u32, bounds-checked against the view) into a
     per-submission buffer. A leftover of fewer than 4 vertices is dropped; a draw with no whole
     quad does nothing. 17, 18 and 20 are skipped by name (`state` skip). Strip cut does not
     apply to quads. Index and vertex buffer usages were missing from the imported and host
     buffers and are added. Where 19 comes from stays **[unverified]**: read the log of the next
     run.
   * VUID-vkCmdDraw-pStrides-04913: the dynamic stride state is declared only when the vertex
     input has a binding (no binding: no vertex buffer bound either, as before).
   * GPU memory miss: a translate running past an import re-reads the committed run; if it grew
     (commits are not recorded as changes) the run is imported again whole, the old import goes
     to `retired` and is dropped at the next `sync`. The 10a miss (`0x10b00000 (+0xae00000)` in a
     run ending at `0x12600000`) asks for more than 174 MB where 27 MB were committed: if the
     next run still logs it, the range is past the allocation, not growth.
   * Description form: the graphics description is 0x290 bytes and its input layout, topology,
     formats, depth format and samples are read at 552, 572, 580, 612 and 616 (they were 8 bytes
     higher); the test uses the header's offsets as literals.
   * Tests (lavapipe too): a quad `Draw` (leftover vertex, cache reused), an indexed quad draw
     with u16 and u32 indices read from `first_index`, the three console topologies refused by
     name, a run that grew and its retired import, the description offsets. The 04913 fix has no
     test (no validation layer in the tests).

   **Chunk 10b, part 3: rectangle lists** (Opus, 2026-11-26, from the first 10b run)
   **[seen]**: the topology the title draws with is not quads: pipeline serials 50546 / 50547
   (trace 50545 / 50546) have stream topology *type* 19 and list topology **17, rectangles**
   (every frame from 3 on, both skipped as `state`). A rectangle is three corners in any order;
   the hardware adds the fourth opposite the right angle, from the **vertex shader's outputs**.
   Trace 50546's vs reads its vertices from an SRV and 50545's through a fetch shader, so an
   expansion of the vertex inputs on the CPU (VIctory's fallback) does not apply: the fourth
   corner is made on the GPU by a generated geometry shader, as VIctory does when it can
   (facts: VIctory `crates/victory-shader/src/emit/rect.rs`).
   * Device: `geometryShader` becomes required when present and optional otherwise; without it
     rectangle draws are skipped by name (`pipeline`).
   * Pipeline: a pipeline whose type is 19 (or a list topology of 17 at its first draw: build
     both variants lazily, cached by (serial, rectangles)) gets a third stage, `rect_gs`, emitted
     by the shader crate from the VS's output interface (every `Location` output plus
     `Position`, same types, same decorations, flat ones kept flat): input `triangles`.
     Corner c = the right angle, opposite the longest side
     in post-divide x, y (ties or NaN → corner 1); `va` = the corner before c, `vb` the one
     after; emit `(v0, v1, v2)` then `(va, vb, v3)` as two separate triangles (output
     `triangle_strip`, two `EndPrimitive`, `max_vertices 6`) so both turn the same way, with v3 = va + vb
     - vc for every float output and `Position`, and va's value for integer / flat outputs.
     `PrimitiveId` and clip distances: pass as for any output (clip distances interpolated like
     floats).
   * Draws: list topology 17 binds as `TRIANGLE_LIST` (3 vertices per rectangle) and needs the
     rectangle pipeline variant; a rectangle draw on a pipeline without it is a `pipeline`
     skip. The quad path of part 2 stays for 19; 18 and 20 stay refused.
   * Tests: `right_angle` on the corners in all six orders and a degenerate case (CPU twin of
     the GS rule); a lavapipe draw of one rectangle (three corners, the right angle in the
     middle of the list) with a colour output, read back: the full axis-aligned rectangle is
     covered, nothing outside; the same with a flat integer output; the generated GS passes
     `spirv-val`.
   * Real validation: frames 4+ show trace 50545's and 50546's rectangles over pipeline 56's
     result.
   **As built (chunk 10b, part 3)** (`shader/spirv/rect.rs`, `shader/io.rs::rect_interface`,
   `backend/vk/{pipeline,draw,device}.rs`; the design above stands, with what is said here).
   * Device: `geometryShader` is enabled when the device has it (`Caps::geometry`), never
     required; without it a rectangle draw is a `pipeline` skip with the reason logged.
   * `rect::rectangle_gs(&RectInterface)` writes the geometry shader directly with the module
     builder (not through the DXBC-style `Shader` of the VIctory facts). The interface is the VS's
     `Position`, `ClipDistance` (when the VS exports clip distances) and every linked `Location`
     (always `vec4` of `f32`: the VS outputs are never integer; a location is flat when the
     pixel shader reads it flat, decorated `Flat` on the GS output only). The right angle is
     chosen on x/w, y/w; `right_angle` is its CPU twin. Output is `max_vertices 6`. `PrimitiveId`
     is not passed (no vertex shader writes it).
   * `Pipelines::get(.., rect)` keys by `(serial, rect)`; the variant is built at the first
     rectangle draw (list topology 17), only on a triangle-class pipeline, with the GS stage
     between the vertex and the pixel stage. Draws bind it as `TRIANGLE_LIST`.
   * Tests (lavapipe too): `right_angle` in the six orders and its degenerate cases; the GS
     passes `spirv-val` (with and without clip distances and flat outputs); a rectangle draw
     (right angle in the middle of the list) read back per pixel, the interpolated position
     colour and the flat output included, and the same for the six orders of the corners. The
     refusal test no longer lists 17. Unverified: the flat value of the second triangle is
     `va`'s, the first's is `v0`'s (provoking vertices); blends and culling over the rectangle
     as the hardware does them; the real validation (frames 4+ of the title) is not done.
   **As built (chunk 10b, part 4)** (`backend/vk/watch.rs`, `textures.rs::upload`). A texture
   whose pages were not written since its last look is no longer hashed (`ReplayStats::watch_skips`
   counts them; `hashed` still counts the bytes that are).
   * Mechanism: one process-wide `userfaultfd` (`UFFD_USER_MODE_ONLY`, so unprivileged) with
     `WP_ASYNC | WP_UNPOPULATED`, `UFFDIO_REGISTER_MODE_WP` on the whole committed run holding the
     texture, and `PAGEMAP_SCAN` on `/proc/self/pagemap` (`PM_SCAN_WP_MATCHING |
     PM_SCAN_CHECK_WPASYNC`, `PAGE_IS_WRITTEN`) that reports and re-protects the written pages.
     Nothing faults into the process; writes by the kernel (`read(2)`) are tracked (tested).
   * A scan clears the state of its pages, and textures share pages: the written regions are
     kept in a log with the epoch of the scan, each `TitleImage` holds a `Cursor`, and a look
     answers for the writes since the cursor's own previous look. A trimmed log, or a run
     registered again, makes older cursors answer written.
   * Fallback: any refusal (no `userfaultfd`, no `PAGEMAP_SCAN`, memory unmapped and mapped
     again at the same address: the scan fails, the run is registered again) answers None,
     taken as written, so the texture is hashed as before. The first look of a texture, and a
     texture never uploaded, are hashed. Logged once when the watch is unavailable.
   * Cost of a look: `table::committed_run` walks the page array of the whole allocation (a
     linear scan of up to millions of pages, twice per texture), which at 700 textures per
     submission cost ~40 s per 60 frames from frame 180 on. A look now checks only the pages of
     the texture (`table::is_committed`) and `watch::is_watched` (a map lookup); the committed
     run is computed only to register a run that is not watched yet. The 120 s run then
     passes frame 2300 (about 1 s per 60 frames once the scenes are loaded).
   * `NATIV64_PROFILE=1` prints, every 60 presented frames, the seconds spent per phase of
     the replay (sync, watch, hash, upload, record, dump, image, descr, host/placed uploads) and
     counters (textures looked at and skipped, scan calls and failures, log size, MB hashed,
     uploads, descriptors written) to stderr (`backend/vk/profile.rs`).
   * Tests (no GPU): first look written, second not, one byte dirties only its page, a pipe
     `read` into a watched page is reported, two lookers of one page, unmap and map again.
   * Unverified: the title's real memory (the run size, the cost of the registrations on the
     VMAs of the table); pages Vulkan imports as host pointers (`gpumem`) while watched; the
     interaction with `mprotect` / `VirtualProtect` by the title on a registered run;
     a `fork`ed child (the registration is not inherited as a watch).
11. **Widening, driven by the corpus report**: f16 and `v_pk_*`, remaining VOP3, MIMG
    gather/compare/derivatives, FLAT/SCRATCH stores and atomics, ray queries, merged stages if a
    title uses them (none in this corpus: its stages are VS, PS, CS, mesh only).

   **As built (chunk 10b, part 5)** (`backend/vk/{format,pipeline,draw,replay}.rs`; first setup screens).
   * `A8_UNORM` (DXGI 65, the small-text glyph atlas) is an `R8_UNORM` image whose views read it
     as alpha: the view's own mapping is composed with `(0, 0, 0, r)` (`a8_swizzle`).
   * A pipeline is keyed by its target formats too: when the bound target is the sRGB (or UNORM)
     twin of the format the pipeline was made for, a variant for the bound format is built
     instead of skipping the draw.
   * `CopyMemory` and `FillMemory` are replayed as CPU copies at record time on committed title
     memory (`table::is_committed`), like `CopyBufferRegion`; the title builds glyph data with them.
   * Still without a Vulkan format: DXGI 3, 13, 17, 30, 36, 42, 50, 51, 57, 62, 67, 86, 115, 118
     and a view in format 21 (depth of D32S8). No test yet for the sRGB pipeline variant.

   **Chunk 10c design: compute** (Opus, 2026-11-27, from the first-screen trace and cs
   1e32f251 of pipeline 51456) **[seen]**: from frame 565 the title dispatches cs 1e32f251
   once per frame. It is a 3D colour-grading table: RSRC2 `0x138c` (6 user SGPRs, group id
   x/y/z in s6/s7/s8, no group info, no scratch, 3 thread-id VGPRs), group 8x8x1, LDS 0, one
   table at user dword 5 read as `s[2:3] = (s5, 0)`; used rows: SRV at table dword 0
   (`image_load` 2D, `a16`, two texels), UAV at table dword 8 (`image_store` 3D, NSA address
   `[v0, v1, v6]`, dmask 0xf, at the end), CBVs at table dwords 16 and 24 (`s_buffer_load*`).
   Its first refusal is `s_load_dwordx4 (chunk 9)` only because a compute blob has no
   `StageIo` and falls to the probe entry. Corpus: 2718 cs build IR, 693 structure, all 693
   stop on resources (s_load x4 548, x8 126, other SMEM 9, DS 5). Mesh moves to **10d** (the
   first screen has none).
   * **Part 1, translator** (`io.rs`, `spirv/emit/{stage,resource}.rs`):
     `StageIo::Compute(CsIo)` with `user = 0..user_sgprs()` (from s0: not `ir::user_range`,
     whose floor of s8 is the VS/PS rule), `tables`, `group`, the SGPR of each enabled group
     id and the thread-id VGPR count. Entry `GLCompute`, `LocalSize` = group; user SGPR i =
     `user_data[i]` as for VS/PS; group id x/y/z = `WorkgroupId`; v0..v2 =
     `LocalInvocationId`; the console lane = `LocalInvocationIndex % 64` (wave64; read the 7a
     lane model in "Structuring" before wiring it). Refused by name: group info SGPR (bit
     10), scratch (bit 0), wave32. Resources use the 9a tracing unchanged; the **kind** of a
     `Loaded` image descriptor is the used row whose table dword × 4 equals the load offset
     (srv / uav), none = refusal. `image_load` / `image_load_mip`: SRV → `OpImageFetch` on
     `textures[]` of that dim (Lod 0 or the mip operand), UAV → `OpImageRead` on `images[]`;
     `image_store` / `image_store_mip` on a UAV → `OpImageWrite` on `images[]`, dmask 0xf only
     (other masks refused by name). `images[]` (set 0 binding 1) variables per dim, sampled
     type f32, format `Unknown`, capabilities `StorageImageReadWithoutFormat` /
     `WriteWithoutFormat`. `a16` on load/store: each address dword holds two u16 coordinates
     (low half first), zero-extended; `unorm`, `glc`, `slc`, `dlc` ignored on load/store; `d16`,
     `tfe`, `lwe` refused. NSA addresses as given. DS (`ds_read*`, `ds_write*`, `ds_write2*`,
     `ds_read2*` b32/b64) → a `Workgroup` u32 array of the record's LDS size (address in bytes
     + offset, / 4; out of range: write dropped, read 0); `s_barrier` →
     `OpControlBarrier(Workgroup, Workgroup, AcquireRelease | WorkgroupMemory)`, refused by
     name under a per-lane branch. Tests: `spirv-val` and real dispatches on lavapipe (the
     `spirv/gpu.rs` harness): ids written to a root buffer, a 3D `image_store` read back, an
     `a16` `image_load` of a 2D SRV, LDS through a barrier; cs 1e32f251 translates and
     validates; corpus report before and after.
   * **Part 2, backend** (`backend/vk/{pipeline,rootblock,replay,textures,residency}.rs`):
     compute pipelines from the pipeline's CS (layout = set 0 + set 1 of its static samplers +
     push constant; cached like graphics); the compute root arguments (`compute: true`) kept
     apart and turned into the dispatch's root block from the CS's layout table; `Dispatch`
     binds, pushes, `vkCmdDispatch`, then the global barrier (simple and correct; finer later
     if a profile asks); skips counted by reason. **UAV views** (`View::Unordered` of a texture
     resource) → `VkImageView` in binding 1 at the registry index: type from the UAV
     description (2D, 2D array, 3D with its W range), one mip, format from the view if it has
     `STORAGE_IMAGE` (an sRGB view takes its UNORM twin, images are `MUTABLE_FORMAT`), else
     refused with a log; integer-class formats refused too (the shader's variables are f32;
     upgrade: pick the class per pipeline from the bound views). Images whose resource flags
     allow unordered access (D3D12 flag 0x4) get `STORAGE` usage when the format supports it.
     Binding 1 holes: a 1x1 RGBA8 2D storage view. Residency scans dispatches as draws, both
     bindings. A resource with a UAV view in the ranges a dispatch's CS reads becomes
     GPU-owned (never re-uploaded). Buffers written by a CS need nothing: title memory through
     imported host pointers. `Gpu::open` refuses a device without the two formatless storage
     features (all three drivers here have them). Real validation: the windowed scripted run,
     frames 565-700 dumped with targets: the 3D table written (not zero), then the intro
     against what the user saw.
   **As built (chunk 10c, part 1)** **[seen]** for the translation, **[unverified]** where said:
   `io::stage_io` / `translate_stage(&cs, &[], None)` give a `GLCompute` entry (`StageIo::Compute(CsIo)`:
   user SGPRs `0..user_sgprs()`, tables, group, group-id SGPRs, thread-id VGPR count, the used rows,
   LDS size); `LocalSize` = group, the push constant is the root block as for VS/PS, group ids =
   `WorkgroupId`, v0..v2 = `LocalInvocationId`. `io::refuse_entry` refuses by name the group info
   SGPR, scratch and wave32 (126 programs of the corpus are wave32). `image_load[_mip]` /
   `image_store[_mip]` (dims 1D, 2D, 3D, 2D array; `a16` packs two u16 per dword; NSA as given)
   find the slot kind through `Analysis::origin` (the constant byte offset a loaded descriptor
   came from, through copies and agreeing phis) and the used row at that offset: SRV =
   `OpImageFetch` on `textures[]`, UAV = `OpImageRead` / `OpImageWrite` on `images[]`
   (`images[]` is set 0 binding 1, sampled type f32, format `Unknown`, with the two formatless
   capabilities). `Translated::storage_images` says binding 1 is used (`images` is then set too).
   DS: `ds_read` / `ds_write` b32..b128 and the `2` / `2st64` pairs on a `Workgroup` u32 array of
   the record's LDS size (out of range: read 0, write dropped); `s_barrier` = `OpControlBarrier`
   (Workgroup, AcquireRelease | WorkgroupMemory). Refused by name (chunk 9 entries of the report):
   store with a dmask other than 0xf, `_mip` on a UAV, cube / MSAA / 1D-array dims, d16 / tfe /
   lwe, a slot no used row names or loaded at a non-constant offset, a store to an SRV,
   `s_barrier` outside compute or under a per-lane `If`, LDS access with no LDS, GDS, and
   **`image_sample` with implicit derivatives** (Vulkan wants `DerivativeGroupLinearKHR` in
   compute; 7 programs failed `spirv-val` on it before the refusal; lane-linear quads, the HLSL
   meaning, are the likely fix). Not wired: the console lane index (`LocalInvocationIndex % 64`)
   for `v_mbcnt` and the other wave ops stay refused as chunk 7. Tests: spirv-val'd synthetic
   blobs, and (with `NATIV64_SPIRV_RUN=1`, lavapipe) real dispatches through `Gpu::run_stage`
   (ids, an LDS mirror through a barrier, a 2D `a16` fetch stored to a 3D image). cs 1e32f251
   translates and `spirv-val`s (114k words). Corpus (2718 cs with IR, 693 structured):
   translated 5 -> 228, all 228 validated (first run: 235 translated, 7 invalid, the implicit
   derivatives above). Left: wave32 126, loaded-descriptor words as numbers 49, typed buffers
   (`buffer_*_format*`, `tbuffer_*`) ~150, own address 38, `image_gather4` 30, DS atomics ~35.
   **[unverified]**: lanes of one wave exchanging through LDS without `s_barrier` (the console's
   lockstep is not reproduced; our invocations are free-running), a barrier after a lane-driven
   loop break, f32 sampled/storage image types against integer-format views (backend picks).

   * **As built (chunk 10c, part 2)**: `backend/vk/compute.rs` (new), `pipeline.rs`,
     `rootblock.rs`, `replay.rs`, `textures.rs`, `residency.rs`, `bindless.rs`, `device.rs`.
     * `Gpu::open` refuses a device without `shaderStorageImageReadWithoutFormat` or
       `shaderStorageImageWriteWithoutFormat` (the refusal names the missing one); both are
       enabled. The push constant range is VS | PS | CS.
     * Compute pipelines: `build` falls to `build_compute` for a pipeline with no graphics
       state: its CS through `io::translate_stage(&cs, &[], None)` (the translator's error is the
       `Failed` reason, so until part 1 covers a program its dispatches are skipped as `pipeline`),
       the layout of its static samplers, `vkCreateComputePipelines`; cached by serial like the
       graphics ones. `HeapReads` now includes the UAV rows (kind 3) of the CS.
     * Root arguments: `RootArgs::compute()` folds the `compute: true` commands (a new compute
       root signature drops only them); `ListState.croot` beside `root`. `Dispatch` fills the
       block from the CS's layout rows exactly as a draw does (an unset table or a root address
       that is not GPU memory skips it), writes it in the arena (sized for draws plus
       dispatches), binds set 0 / set 1, pushes the address, `vkCmdDispatch`, then the global
       barrier. A group count of 0 does nothing; above 65535 (the Vulkan minimum limit) skips.
     * Skips are counted in `ReplayStats::dispatch_skips` (`DispatchSkips`: `pipeline` = none
       bound, graphics pipeline, translator or Vulkan refusal; `memory` = root block address;
       `state` = no bindless set, no arena room, too many groups), `dispatched` counts the
       recorded ones, `dispatches` still counts every `Dispatch*` seen. Mesh dispatches stay
       counted only. Each skip is logged once as `dispatch skipped: ...`.
     * UAV views: `View::Unordered` of a 2D or 3D texture is an `Entry::Storage`. Its view goes
       to binding 1 at the registry index (`storage_view`: type from the description, 2D, 2D array
       or 3D, one mip; the format of the view, an sRGB one taking its UNORM twin; refused with a
       log when the format has no storage support or has no Vulkan mapping, which covers the
       integer classes since `format::lookup` only maps float, unorm and snorm); binding 0 holds
       the sampled hole there. Every other index holds a 1x1 RGBA8 storage hole of its own
       (separate from binding 0's, shaders may write to it). Views are cached on the image with
       the top bit of the first key word set and die with it. Images of a resource with the
       unordered flag (0x4) get `STORAGE` usage when the format (or the UNORM twin of an sRGB one,
       with `EXTENDED_USAGE`) supports it (`TitleImage::storage`).
     * Residency: `residency::scan` reads a `Dispatch` like a draw from the compute signature's
       tables, so the UAV entries of its ranges are wanted; `sync_textures` makes their images
       (uploading the host content first) and then marks them GPU-owned, so they are never
       uploaded again nor evicted. A submission with only dispatches syncs textures too.
     * Deviations: a 3D UAV view is the whole mip (its range of W slices is logged, not applied);
       integer UAV formats are refused (no f32 variable would match); a UAV on a 1D texture, a
       placed UAV or a depth image is a hole. Counter resources are ignored.
     * Tests (lavapipe, `NATIV64_GPU=llvmpipe`, `compute::tests`): a fixture module compiled
       with glslc (injected through `pipeline::COMPUTE_MODULES`, test only, since the translator
       does not cover compute yet) writes a root buffer and an 8x2 storage image through the
       push-constant root block; the image is read back (255, 128, 0, 255), is `STORAGE` and
       GPU-owned; an image without the flag reads the hole while the buffer is still written;
       the skip reasons are counted; compute and graphics root arguments are separate.
     * Not verified: the real cs 1e32f251 (needs part 1), real hardware (only llvmpipe), the
       windowed scripted run of frames 565-700.

## Open questions

- Unverified ALU semantics (no hardware data, the table follows the ISA text / LLVM): sign of
  the zero of `v_mul_legacy_f32` and `v_mad_legacy_f32` (the product's sign here), `v_med3_f32`
  with a NaN (the minimum of the others), `v_cube*` (LLVM's description), `v_frexp_*` of
  denormals (treated like zero), `v_cvt_pk_u8_f32` (truncates, saturates to 255), `v_perm_b32`
  selectors 8..11 (sign bits 15, 31, 47, 63), `v_bfe_i32` with offset + width past 32 (the
  arithmetic shift fills the field), 16-bit VOP2 ops zeroing the high half of the destination.
- f32 denormals and rounding: the emitted code uses the host's (SPIR-V has no per-instruction
  mode; `DenormPreserve` / `RoundingModeRTE` need `SPV_KHR_float_controls`, core in 1.4).
  A program that writes the f32 bits of MODE is refused; the default MODE of a stage is not
  decoded yet. `PackHalf2x16` rounds as the host does (llvmpipe: to nearest even);
  `v_mad_f32` / `v_mac_f32` are `OpFMul` + `OpFAdd` without `NoContraction`, so a driver may
  fuse them.
- `s_wqm` is a move in the probe; in a pixel shader it is a quad-wise OR (chunk 8).
- Wave32 bit (record +0x14 bit 5) and the VS system VGPR layout are VIctory's readings, not
  confirmed by the corpus (all wave64, VS reads only v0).
- Root-layout and used-resource tables have up to a few hundred entries (the chunk 1 cap of 64
  refused 2486 vertex/pixel blobs; now 512, bounds-checked against the payload).
- VOP3 interpolation (`v_interp_*_e64`) decodes; the f16 forms print `high`, not decoded.
- The VS fetch routine: its address arrives in `s[4:5]`, its code is not in the blob. That it
  only loads attributes into the stage data's VGPRs (and keeps every other register) is
  VIctory's reading; chunk 4 checks that the stage data of each of the 332 lists attributes.
- Compute programs reading `s0..s3` (about 50 `s_mov` / `s_and`): user data, or system SGPRs
  before it when `RSRC2` says few user SGPRs? Chunk 4 answers it by decoding `RSRC2`.
- Kind 0x12 XDAT entry, `PRIV` part (12 bytes), program-record bytes not listed: meaning unknown,
  ignored.
- Bindless limits, `VK_EXT_shader_demote_to_helper_invocation` and `VK_EXT_mesh_shader` on
  lavapipe and the Intel driver: to check with `nativ64-vkcheck` before chunk 10 (mesh is 18%
  of the programs: without the extension those pipelines cannot be built at all).
