//! Prints the parsed container of shader blobs (the files `NATIV64_DUMP_SHADERS` writes):
//! `nativ64-shader [--disasm | --ir] <blob>...`. With `--disasm` each program's code is followed
//! from its entry and printed in LLVM syntax; with `--ir` it is printed as the SSA IR. `nativ64-shader --hex <file>` disassembles one
//! instruction per line of hex bytes (for diffing against `llvm-mc`). `--stats` counts the
//! programs that decode and why the others do not; `--ir --stats` counts the programs whose IR
//! builds, per stage, and the errors by kind; `--structure --stats` does the same for the
//! structuring (cross-lane uses by instruction name included); `--structure --waves` classifies
//! every cross-lane use (lane id, uniform or waterfall `readfirstlane`, DPP by control kind,
//! swizzles, masks read as data...), per stage in instructions and programs, counts the
//! programs whose uses are all size-agnostic and the ones in a whole-wave section. `--spirv --stats` translates the
//! structured programs to the probe module (chunk 6) and validates them with `spirv-val` when it
//! is in PATH, listing the others by first unsupported op; `--spirv --alu-stats` counts the
//! programs whose every ALU instruction translates; `--spirv <blob> -o out.spv` writes one
//! module (the real entry point of a vertex or pixel blob, the probe of the others);
//! `--spirv --pair <vs> <ps> -o out` writes both stages with their locations linked by
//! semantic. Exit status 1 if any blob is refused or any code fails to decode.

use nativ64_gpu::shader::{self, gfx10};
use std::process::ExitCode;

/// The widest wave: the subgroup size under which no program is refused for its wave size.
const WHOLE_WAVE: u32 = 64;

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let disasm = args.first().is_some_and(|a| a == "--disasm");
    let ir = args.first().is_some_and(|a| a == "--ir");
    let structure = args.first().is_some_and(|a| a == "--structure");
    if args.first().is_some_and(|a| a == "--spirv") {
        args.remove(0);
        return match args.first().map(String::as_str) {
            Some("--stats") => match args.get(1).map(String::as_str) {
                Some("--subgroup") => match args.get(2).and_then(|n| n.parse().ok()) {
                    Some(n) => spirv_stats(&args[3..], false, n),
                    None => {
                        eprintln!("--subgroup needs a number of invocations");
                        ExitCode::from(2)
                    }
                },
                _ => spirv_stats(&args[1..], false, WHOLE_WAVE),
            },
            Some("--alu-stats") => spirv_stats(&args[1..], true, WHOLE_WAVE),
            Some("--pair") if args.len() == 5 && args[3] == "-o" => {
                spirv_pair(&args[1], &args[2], &args[4])
            }
            Some(_) if args.len() == 3 && args[1] == "-o" => spirv_write(&args[0], &args[2]),
            _ => {
                eprintln!(
                    "usage: nativ64-shader --spirv (--stats [--subgroup <n>] | --alu-stats) <blob.dxbc>...\n       \
                     nativ64-shader --spirv <blob.dxbc> -o <out.spv>\n       \
                     nativ64-shader --spirv --pair <vs.dxbc> <ps.dxbc> -o <out>\n       \
                     (a vertex or pixel blob is written as its stage's entry point; --pair \
                     writes <out>.vert.spv and <out>.frag.spv with shared locations)"
                );
                ExitCode::from(2)
            }
        };
    }
    if disasm || ir {
        args.remove(0);
    }
    if structure {
        args.remove(0);
        if args.first().is_some_and(|a| a == "--stats") {
            return structure_stats(&args[1..]);
        }
        if args.first().is_some_and(|a| a == "--waves") {
            return wave_survey(&args[1..]);
        }
        eprintln!("usage: nativ64-shader --structure (--stats | --waves) <blob.dxbc>...");
        return ExitCode::from(2);
    }
    if args.first().is_some_and(|a| a == "--stats") {
        return if ir {
            ir_stats(&args[1..])
        } else {
            stats(&args[1..])
        };
    }
    if args.first().is_some_and(|a| a == "--hex") {
        return match args.get(1) {
            Some(f) => hex_lines(f),
            None => ExitCode::from(2),
        };
    }
    if args.is_empty() {
        eprintln!("usage: nativ64-shader [--disasm | --ir] <blob.dxbc>... | --hex <file>");
        return ExitCode::from(2);
    }
    let mut failed = false;
    for f in &args {
        let bytes = match std::fs::read(f) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{f}: {e}");
                failed = true;
                continue;
            }
        };
        println!(
            "== {f} ({} bytes, hash {:016x})",
            bytes.len(),
            shader::fnv1a64(&bytes)
        );
        match shader::parse(&bytes) {
            Ok(p) => {
                print!("{p}");
                if ir {
                    failed |= !print_ir(&p);
                } else if disasm {
                    failed |= !print_code(&p);
                }
            }
            Err(e) => {
                println!("refused: {e}");
                failed = true;
            }
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// Prints the SSA IR of the program's own stage record.
fn print_ir(p: &shader::ParsedShader) -> bool {
    let Some(rec) = p.record() else {
        return true;
    };
    let code = p.native.code(rec);
    let regs = shader::ir::EntryRegs::from_stage(rec);
    match shader::ir::build(code, gfx10::ENTRY, rec.wave32(), &regs) {
        Ok(f) => {
            print!("{f}");
            true
        }
        Err(e) => {
            println!("  IR error: {e}");
            false
        }
    }
}

/// Prints the flow-followed code of the program's VS or PS record. Stages without a decoded
/// record (compute, mesh) print each slot's code, read as wave64 like `--stats` does.
fn print_code(p: &shader::ParsedShader) -> bool {
    let Some(rec) = p.record() else {
        let mut ok = true;
        for (_, code) in p.native.slot_codes() {
            ok &= print_follow(code, false);
        }
        return ok;
    };
    print_follow(p.native.code(rec), rec.wave32())
}

fn print_follow(code: &[u8], wave32: bool) -> bool {
    match gfx10::follow(code, gfx10::ENTRY, wave32) {
        Ok(prog) => {
            for i in &prog.instrs {
                println!("  +{:#05x}: {}", i.at, gfx10::disasm_instr(i));
            }
            for at in &prog.indirect {
                println!("  indirect jump at +{at:#x}");
            }
            true
        }
        Err(e) => {
            println!("  decode error: {e}");
            false
        }
    }
}

/// Decodes every program of every blob (all stages) and reports how many decode entirely and
/// which opcodes / encodings stop the rest, by count.
fn stats(files: &[String]) -> ExitCode {
    use std::collections::BTreeMap;
    let mut by_stage: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    let mut errs: BTreeMap<String, u32> = BTreeMap::new();
    let mut refused = 0;
    let mut instrs = 0u64;
    for f in files {
        let Ok(bytes) = std::fs::read(f) else {
            continue;
        };
        let Ok(p) = shader::parse(&bytes) else {
            refused += 1;
            continue;
        };
        let stage = format!("{:?}", p.kind);
        for (_, code) in p.native.slot_codes() {
            let e = by_stage.entry(stage.clone()).or_default();
            e.0 += 1;
            match gfx10::follow(code, gfx10::ENTRY, false) {
                Ok(prog) => {
                    instrs += prog.instrs.len() as u64;
                    e.1 += 1;
                }
                Err(err) => {
                    let key = match err.kind {
                        gfx10::ErrorKind::UnknownOpcode { enc, opcode } => {
                            format!("{} opcode {opcode:#x}", enc.name())
                        }
                        gfx10::ErrorKind::UnknownEncoding { name, .. } => {
                            format!("encoding {name}")
                        }
                        k => format!("{k:?}"),
                    };
                    *errs.entry(key).or_default() += 1;
                }
            }
        }
    }
    println!("blobs refused by the container parser: {refused}; instructions decoded: {instrs}");
    for (s, (n, ok)) in &by_stage {
        println!("{s}: {ok}/{n} programs decode entirely");
    }
    let mut v: Vec<_> = errs.into_iter().collect();
    v.sort_by_key(|a| std::cmp::Reverse(a.1));
    for (k, n) in v.iter().take(40) {
        println!("{n:6}  first failure: {k}");
    }
    ExitCode::SUCCESS
}

/// Builds the IR of every program with a decoded record (vertex, pixel, mesh, compute) and
/// reports, per stage, how many build and the errors by kind. A read of a register the entry
/// table does not explain is listed with the first reading instruction and a sample blob.
fn ir_stats(files: &[String]) -> ExitCode {
    use shader::ir::{self, IrErrorKind};
    use std::collections::BTreeMap;
    let mut stage: BTreeMap<&str, (u32, u32)> = BTreeMap::new();
    let mut errs: BTreeMap<(String, String), u32> = BTreeMap::new();
    let mut live_ins: BTreeMap<(String, String), (u64, String)> = BTreeMap::new();
    // vertex programs: (has attributes, opens with the fetch call) -> count
    let mut fetch: BTreeMap<(bool, bool), u32> = BTreeMap::new();
    let (mut refused, mut blocks, mut insts, mut phis) = (0, 0usize, 0usize, 0usize);
    for f in files {
        let Ok(bytes) = std::fs::read(f) else {
            continue;
        };
        let Ok(p) = shader::parse(&bytes) else {
            refused += 1;
            continue;
        };
        for (name, rec) in p.native.records() {
            let code = p.native.code(rec);
            let regs = ir::EntryRegs::from_stage(rec);
            let e = stage.entry(name).or_default();
            e.0 += 1;
            if name == "vs" {
                let call = gfx10::decode_wave(code, gfx10::ENTRY, rec.wave32())
                    .is_ok_and(|i| i.name() == "s_swappc_b64");
                *fetch.entry((regs.fetch.is_some(), call)).or_default() += 1;
            }
            match ir::build(code, gfx10::ENTRY, rec.wave32(), &regs) {
                Ok(func) => {
                    e.1 += 1;
                    blocks += func.blocks.len();
                    insts += func.blocks.iter().map(|b| b.insts.len()).sum::<usize>();
                    phis += func.blocks.iter().map(|b| b.phis.len()).sum::<usize>();
                }
                Err(err) => {
                    let kind = match &err.kind {
                        IrErrorKind::Decode(d) => match d.kind {
                            gfx10::ErrorKind::UnknownOpcode { enc, opcode } => {
                                format!("decode: {} opcode {opcode:#x}", enc.name())
                            }
                            gfx10::ErrorKind::UnknownEncoding { name, .. } => {
                                format!("decode: encoding {name}")
                            }
                            ref k => format!("decode: {k:?}"),
                        },
                        IrErrorKind::Indirect => "register jump".to_string(),
                        IrErrorKind::UnsupportedEncoding(n) => format!("encoding {n}"),
                        IrErrorKind::Unsupported(s) => format!("unsupported: {s}"),
                        IrErrorKind::UndefinedRead(r) => {
                            let at = gfx10::decode_wave(code, err.at, rec.wave32())
                                .map(|i| gfx10::disasm_instr(&i))
                                .unwrap_or_default();
                            live_ins
                                .entry((name.to_string(), r.to_string()))
                                .or_insert((
                                    shader::fnv1a64(&bytes),
                                    format!("+{:#x}: {at}", err.at),
                                ));
                            format!("live-in {r} not explained")
                        }
                        IrErrorKind::BadOperand(s) => format!("bad operand: {s}"),
                    };
                    *errs.entry((name.to_string(), kind)).or_default() += 1;
                }
            }
        }
    }
    println!("blobs refused by the container parser: {refused}");
    let (programs, built) = stage.values().fold((0, 0), |a, v| (a.0 + v.0, a.1 + v.1));
    println!("IR built: {built}/{programs} programs ({blocks} blocks, {insts} instructions, {phis} phis)");
    for (s, (n, ok)) in &stage {
        println!("  {s}: {ok}/{n} build");
    }
    let mut v: Vec<_> = errs.into_iter().collect();
    v.sort_by_key(|a| std::cmp::Reverse(a.1));
    for ((s, k), n) in &v {
        println!("{n:6}  {s}: {k}");
    }
    for ((s, r), (hash, at)) in &live_ins {
        println!("live-in {s} {r}: first read {at} (blob hash {hash:016x})");
    }
    for ((attrs, call), n) in &fetch {
        println!(
            "vs {n}: {} attributes in the stage data, {} fetch call",
            if *attrs { "with" } else { "no" },
            if *call { "opens with a" } else { "no" }
        );
    }
    ExitCode::SUCCESS
}

/// Structures the IR of every program that builds and reports, per stage, how many structure
/// and the errors by kind; then the same ignoring cross-lane uses (what else stops them), and
/// the cross-lane instructions by name with the number of programs that have each.
fn structure_stats(files: &[String]) -> ExitCode {
    use shader::ir;
    use shader::structure::{self, StructError};
    use std::collections::{BTreeMap, BTreeSet};
    let kind = |e: &StructError| match e {
        StructError::CrossLane(_) => "cross-lane use".to_string(),
        StructError::DivergentScalar { .. } => "divergent scalar".to_string(),
        StructError::Irreducible { .. } => "irreducible".to_string(),
        StructError::Malformed(s) => format!("malformed: {s}"),
    };
    let mut stage: BTreeMap<String, (u32, u32, u32)> = BTreeMap::new();
    let mut errs: BTreeMap<(String, String), u32> = BTreeMap::new();
    let mut errs_free: BTreeMap<(String, String), u32> = BTreeMap::new();
    let mut names: BTreeMap<(String, String), u32> = BTreeMap::new();
    let mut samples: BTreeMap<(String, String), (u64, String)> = BTreeMap::new();
    let mut first_err: BTreeMap<(String, String), (u64, String)> = BTreeMap::new();
    let (mut no_ir, mut demotes, mut lane_br, mut uni_br, mut wave_br) =
        (0u32, 0u32, 0usize, 0usize, 0usize);
    let (mut proved, mut active) = (0usize, 0usize);
    for f in files {
        let Ok(bytes) = std::fs::read(f) else {
            continue;
        };
        let Ok(p) = shader::parse(&bytes) else {
            continue;
        };
        for (name, rec) in p.native.records() {
            let code = p.native.code(rec);
            let regs = ir::EntryRegs::from_stage(rec);
            let Ok(func) = ir::build(code, gfx10::ENTRY, rec.wave32(), &regs) else {
                no_ir += 1;
                continue;
            };
            let e = stage.entry(name.to_string()).or_default();
            e.0 += 1;
            let masks = structure::type_masks(&func);
            let distinct: BTreeSet<&str> =
                masks.cross_lane.iter().map(|c| c.name.as_str()).collect();
            for n in distinct {
                *names.entry((name.to_string(), n.to_string())).or_default() += 1;
            }
            match structure::structure(&func) {
                Ok(s) => {
                    e.1 += 1;
                    demotes += u32::from(!s.demotes.is_empty());
                    for b in s.branches.iter().flatten() {
                        match b {
                            structure::BranchKind::Lane(_) => lane_br += 1,
                            structure::BranchKind::Wave(_) => wave_br += 1,
                            structure::BranchKind::Uniform => uni_br += 1,
                        }
                    }
                    for g in s.guards.iter().flatten() {
                        match g {
                            structure::Guard::Proved => proved += 1,
                            structure::Guard::Active => active += 1,
                            structure::Guard::Scalar => {}
                        }
                    }
                }
                Err(err) => {
                    let key = (name.to_string(), kind(&err));
                    // the first blob of each kind, for `--structure --ir`-style digging
                    first_err
                        .entry(key.clone())
                        .or_insert((shader::fnv1a64(&bytes), err.to_string()));
                    *errs.entry(key).or_default() += 1;
                }
            }
            match structure::structure_with(&func, true) {
                Ok(_) => e.2 += 1,
                Err(err) => {
                    let key = (name.to_string(), kind(&err));
                    samples
                        .entry(key.clone())
                        .or_insert((shader::fnv1a64(&bytes), err.to_string()));
                    *errs_free.entry(key).or_default() += 1;
                }
            }
        }
    }
    println!("programs without IR (not counted): {no_ir}");
    let total = stage
        .values()
        .fold((0, 0, 0), |a, v| (a.0 + v.0, a.1 + v.1, a.2 + v.2));
    println!(
        "structured: {}/{} programs; ignoring cross-lane uses: {}/{}",
        total.1, total.0, total.2, total.0
    );
    for (s, (n, ok, free)) in &stage {
        println!("  {s}: {ok}/{n} structure ({free} ignoring cross-lane uses)");
    }
    println!(
        "branches in structured programs: {lane_br} per-lane, {wave_br} speculated, {uni_br} uniform"
    );
    println!("vector instructions in structured programs: {proved} guards proved, {active} kept");
    println!("programs with a pixel discard (demote): {demotes}");
    let sorted = |m: BTreeMap<(String, String), u32>| {
        let mut v: Vec<_> = m.into_iter().collect();
        v.sort_by_key(|a| std::cmp::Reverse(a.1));
        v
    };
    println!("errors:");
    for ((s, k), n) in sorted(errs) {
        println!("{n:6}  {s}: {k}");
    }
    println!("errors ignoring cross-lane uses:");
    for ((s, k), n) in sorted(errs_free) {
        println!("{n:6}  {s}: {k}");
    }
    for ((s, k), (hash, text)) in &samples {
        println!("sample {s}: {k}: {text} (blob hash {hash:016x})");
    }
    for ((s, k), (hash, text)) in &first_err {
        println!("first {s}: {k}: {text} (blob hash {hash:016x})");
    }
    println!("cross-lane instructions (programs that have one):");
    for ((s, k), n) in sorted(names) {
        println!("{n:6}  {s}: {k}");
    }
    ExitCode::SUCCESS
}

/// The category of one cross-lane use (`wave_survey`).
fn wave_category(
    m: &shader::structure::Masks,
    inst: &shader::ir::Inst,
    eq_cmps: &[Vec<shader::ir::Src>],
) -> String {
    use shader::ir::Src;
    let i = &inst.instr;
    let n = i.name();
    if let Some(d) = &i.dpp {
        let c = d.ctrl;
        return match d.kind {
            gfx10::DppKind::Dpp8 | gfx10::DppKind::Dpp8Fi => "dpp8",
            _ => match c {
                0..=0xff => "dpp_quad_perm",
                0x101..=0x12f | 0x140 | 0x141 | 0x150..=0x16f => "dpp_row",
                0x142 | 0x143 => "dpp_row_bcast",
                0x130..=0x13f => "dpp_wave",
                _ => "other:dpp_ctrl",
            },
        }
        .to_string();
    }
    let src = |k: usize| inst.srcs.get(k).and_then(|v| v.first()).copied();
    if n.starts_with("v_mbcnt") {
        return match src(0) {
            Some(Src::Const(u32::MAX)) => "mbcnt_lane_id",
            _ => "mbcnt_mask",
        }
        .to_string();
    }
    if n.starts_with("v_readfirstlane") {
        let Some(Src::Val(v)) = src(0) else {
            return "readfirstlane_uniform".into();
        };
        if m.is_uniform(v) {
            return "readfirstlane_uniform".into();
        }
        let res = inst.dsts.first().map(|d| d.1);
        let waterfall = eq_cmps.iter().any(|srcs| {
            srcs.contains(&Src::Val(v)) && res.is_some_and(|r| srcs.contains(&Src::Val(r)))
        });
        return if waterfall {
            "readfirstlane_waterfall"
        } else {
            "readfirstlane_other"
        }
        .to_string();
    }
    for (p, base) in [("v_readlane", "readlane"), ("v_writelane", "writelane")] {
        if n.starts_with(p) {
            return match src(1) {
                Some(Src::Const(c)) if c < 32 => format!("{base}_const_lo"),
                Some(Src::Const(_)) => format!("{base}_const_hi"),
                _ => format!("{base}_sgpr"),
            };
        }
    }
    if n.starts_with("v_permlanex16") {
        return "permlanex16".into();
    }
    if n.starts_with("v_permlane16") {
        return "permlane16".into();
    }
    if n.starts_with("ds_swizzle") {
        return if i.mem.offset & 0x8000 != 0 {
            "ds_swizzle_quad"
        } else {
            "ds_swizzle_bitmask"
        }
        .to_string();
    }
    if n.starts_with("ds_permute") || n.starts_with("ds_bpermute") {
        return n.to_string();
    }
    if shader::structure::cross_lane_by_nature(i) {
        return format!("other:{n}");
    }
    let scalar_mask_op = [
        "s_and",
        "s_or",
        "s_xor",
        "s_not",
        "s_nand",
        "s_nor",
        "s_xnor",
        "s_mov",
        "s_cselect",
        "s_wqm",
    ]
    .iter()
    .any(|p| n.starts_with(p));
    if scalar_mask_op {
        "mask_with_scalar".into()
    } else {
        format!("mask_data:{n}")
    }
}

/// The survey behind the wave-operations decision (chunk 7): classifies every cross-lane use of
/// the programs whose IR builds. Per stage, per category, the instructions and the programs;
/// then the programs whose uses are all size-agnostic (lane id, uniform or waterfall
/// `readfirstlane`, quad-perm DPP, quad swizzle, mask with a uniform scalar) against those that
/// need a 64-wide wave. A whole-wave section starts at `s_or_saveexec x, -1` (or
/// `s_nand_saveexec x, 0`) and runs to the next EXEC write: the rest of its block, then the
/// blocks it dominates (depth first from its successors) up to their first EXEC write.
fn wave_survey(files: &[String]) -> ExitCode {
    use shader::ir::{self, Reg, Src};
    use shader::structure;
    use std::collections::{BTreeMap, BTreeSet};
    const AGNOSTIC: [&str; 6] = [
        "mbcnt_lane_id",
        "readfirstlane_uniform",
        "readfirstlane_waterfall",
        "dpp_quad_perm",
        "ds_swizzle_quad",
        "mask_with_scalar",
    ];
    #[derive(Default)]
    struct Stage {
        progs: u32,
        any: u32,
        agnostic: u32,
        whole: u32,
        whole_uses: u32,
        whole_cross_progs: u32,
        cats: BTreeMap<String, (u32, u32)>,
    }
    let writes_exec = |i: &ir::Inst| {
        i.dsts
            .iter()
            .any(|(r, _)| matches!(r, Reg::ExecLo | Reg::ExecHi))
    };
    let whole_wave = |i: &ir::Inst| {
        let n = i.instr.name();
        let s = i.srcs.first().and_then(|v| v.first());
        (n.starts_with("s_or_saveexec") && matches!(s, Some(Src::Const(u32::MAX))))
            || (n.starts_with("s_nand_saveexec") && matches!(s, Some(Src::Const(0))))
    };
    let mut stages: BTreeMap<String, Stage> = BTreeMap::new();
    for file in files {
        let Ok(bytes) = std::fs::read(file) else {
            continue;
        };
        let Ok(p) = shader::parse(&bytes) else {
            continue;
        };
        for (name, rec) in p.native.records() {
            let code = p.native.code(rec);
            let regs = ir::EntryRegs::from_stage(rec);
            let Ok(f) = ir::build(code, gfx10::ENTRY, rec.wave32(), &regs) else {
                continue;
            };
            let m = structure::type_masks(&f);
            let eq_cmps: Vec<Vec<Src>> = f
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .filter(|u| {
                    let n = u.instr.name();
                    n.starts_with("v_cmp_eq") || n.starts_with("v_cmpx_eq")
                })
                .map(|u| u.srcs.iter().flatten().copied().collect())
                .collect();
            let st = stages.entry(name.to_string()).or_default();
            st.progs += 1;
            let mut counts: BTreeMap<String, u32> = BTreeMap::new();
            let mut inside: BTreeSet<usize> = BTreeSet::new();
            let mut has_whole = false;
            for (b, blk) in f.blocks.iter().enumerate() {
                for (k, inst) in blk.insts.iter().enumerate() {
                    if structure::cross_lane_use(&m, inst).is_some() {
                        *counts.entry(wave_category(&m, inst, &eq_cmps)).or_default() += 1;
                    }
                    if !whole_wave(inst) {
                        continue;
                    }
                    has_whole = true;
                    let mut stop = false;
                    for x in &blk.insts[k + 1..] {
                        if writes_exec(x) {
                            stop = true;
                            break;
                        }
                        inside.insert(x.instr.at);
                    }
                    if stop {
                        continue;
                    }
                    let dominated = |mut c: usize| loop {
                        if c == b {
                            return true;
                        }
                        match f.idom[c] {
                            Some(d) => c = d,
                            None => return false,
                        }
                    };
                    let mut seen = BTreeSet::new();
                    let mut work = blk.succs();
                    while let Some(c) = work.pop() {
                        if c == b || !dominated(c) || !seen.insert(c) {
                            continue;
                        }
                        let mut stop = false;
                        for x in &f.blocks[c].insts {
                            if writes_exec(x) {
                                stop = true;
                                break;
                            }
                            inside.insert(x.instr.at);
                        }
                        if !stop {
                            work.extend(f.blocks[c].succs());
                        }
                    }
                }
            }
            let mut in_whole = 0;
            for blk in &f.blocks {
                for inst in &blk.insts {
                    if structure::cross_lane_use(&m, inst).is_some()
                        && inside.contains(&inst.instr.at)
                    {
                        in_whole += 1;
                    }
                }
            }
            st.whole += u32::from(has_whole);
            st.whole_uses += in_whole;
            st.whole_cross_progs += u32::from(in_whole > 0);
            if counts.is_empty() {
                continue;
            }
            st.any += 1;
            if counts.keys().all(|c| AGNOSTIC.contains(&c.as_str())) {
                st.agnostic += 1;
            }
            for (c, n) in counts {
                let e = st.cats.entry(c).or_default();
                e.0 += n;
                e.1 += 1;
            }
        }
    }
    for (s, st) in &stages {
        println!("== {s}: {} programs with IR", st.progs);
        println!("{:>8} {:>8}  category", "insts", "programs");
        let mut v: Vec<_> = st.cats.iter().collect();
        v.sort_by_key(|(_, c)| std::cmp::Reverse(c.1));
        for (c, (n, pr)) in v {
            println!("{n:8} {pr:8}  {c}");
        }
        println!(
            "any cross-lane use: {}; size-agnostic only: {}; needing a 64-wide wave: {}",
            st.any,
            st.agnostic,
            st.any - st.agnostic
        );
        println!(
            "whole-wave sections: {} programs; cross-lane instructions inside: {} in {} programs",
            st.whole, st.whole_uses, st.whole_cross_progs
        );
    }
    ExitCode::SUCCESS
}

/// One instruction per input line (hex bytes, any separators); prints `text` or `ERR message`.
fn hex_lines(file: &str) -> ExitCode {
    let Ok(text) = std::fs::read_to_string(file) else {
        eprintln!("{file}: unreadable");
        return ExitCode::from(2);
    };
    for line in text.lines() {
        let hex: String = line.chars().filter(char::is_ascii_hexdigit).collect();
        let bytes: Vec<u8> = (0..hex.len() / 2)
            .map(|k| u8::from_str_radix(&hex[2 * k..2 * k + 2], 16).unwrap_or(0))
            .collect();
        match gfx10::decode_wave(&bytes, 0, false) {
            Ok(i) if i.len == bytes.len() => println!("{}", gfx10::disasm_instr(&i)),
            Ok(i) => println!("ERR length {} != {}", i.len, bytes.len()),
            Err(e) => println!("ERR {e}"),
        }
    }
    ExitCode::SUCCESS
}

/// The probe module of a blob's program, written to `out`.
fn spirv_write(blob: &str, out: &str) -> ExitCode {
    use shader::{ir, spirv, structure};
    let Ok(bytes) = std::fs::read(blob) else {
        eprintln!("{blob}: unreadable");
        return ExitCode::from(2);
    };
    let p = match shader::parse(&bytes) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{blob}: refused: {e}");
            return ExitCode::FAILURE;
        }
    };
    let Some(rec) = p.record() else {
        eprintln!("{blob}: no stage record to translate");
        return ExitCode::FAILURE;
    };
    if let Some(io) = shader::io::stage_io(&p, &[], None) {
        return match shader::io::translate_stage(&p, &[], None, WHOLE_WAVE) {
            Ok(t) => write_words(&t.words, out, &io),
            Err(e) => {
                eprintln!("{blob}: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let regs = ir::EntryRegs::from_stage(rec);
    let func = match ir::build(p.native.code(rec), gfx10::ENTRY, rec.wave32(), &regs) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{blob}: IR error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let st = match structure::structure(&func) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{blob}: structuring: {e}");
            return ExitCode::FAILURE;
        }
    };
    match spirv::translate(&func, &st) {
        Ok(probe) => {
            let bytes: Vec<u8> = probe.words.iter().flat_map(|w| w.to_le_bytes()).collect();
            if let Err(e) = std::fs::write(out, bytes) {
                eprintln!("{out}: {e}");
                return ExitCode::FAILURE;
            }
            println!(
                "{out}: {} words, {} inputs, {} outputs",
                probe.words.len(),
                probe.inputs.len(),
                probe.outputs.len()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{blob}: {e}");
            ExitCode::FAILURE
        }
    }
}

fn write_words(words: &[u32], out: &str, io: &shader::io::StageIo) -> ExitCode {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    if let Err(e) = std::fs::write(out, bytes) {
        eprintln!("{out}: {e}");
        return ExitCode::FAILURE;
    }
    let stage = match io {
        shader::io::StageIo::Vertex(_) => "Vertex",
        shader::io::StageIo::Pixel(_) => "Fragment",
        shader::io::StageIo::Compute(_) => "GLCompute",
    };
    println!("{out}: {stage} entry, {} words", words.len());
    ExitCode::SUCCESS
}

/// `--spirv --pair`: both stages, with the locations linked by semantic.
fn spirv_pair(vs: &str, ps: &str, out: &str) -> ExitCode {
    let parse = |f: &str| match std::fs::read(f) {
        Ok(b) => shader::parse(&b).map_err(|e| format!("{f}: refused: {e}")),
        Err(e) => Err(format!("{f}: {e}")),
    };
    let (vs, ps) = match (parse(vs), parse(ps)) {
        (Ok(v), Ok(p)) => (v, p),
        (v, p) => {
            for e in [v.err(), p.err()].into_iter().flatten() {
                eprintln!("{e}");
            }
            return ExitCode::FAILURE;
        }
    };
    match shader::io::translate_pair(&vs, &ps, &[], WHOLE_WAVE) {
        Ok((v, p)) => {
            let l = shader::io::link(&vs, &ps);
            println!("link: {:?} / {:?}", l.params, l.attrs);
            let (Some(a), Some(b)) = (
                shader::io::stage_io(&vs, &[], Some(&l)),
                shader::io::stage_io(&ps, &[], Some(&l)),
            ) else {
                return ExitCode::FAILURE;
            };
            let r = write_words(&v.words, &format!("{out}.vert.spv"), &a);
            if r != ExitCode::SUCCESS {
                return r;
            }
            write_words(&p.words, &format!("{out}.frag.spv"), &b)
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

/// True when `spirv-val` accepts the module (`None` when it is not installed).
fn spirv_val(words: &[u32], scratch: &std::path::Path) -> Option<Result<(), String>> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(scratch, bytes).ok()?;
    let r = std::process::Command::new("spirv-val")
        .args(["--target-env", "vulkan1.2"])
        .arg(scratch)
        .output()
        .ok()?;
    Some(if r.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&r.stderr).trim().to_string())
    })
}

/// `--spirv --stats`: per stage, the programs that structure, translate and validate, and the
/// others by first unsupported op. With `--subgroup <n>` the programs that need a wider
/// subgroup than `n` (`io::refuse_wave`) count as refused, and each stage also reports the
/// structured programs that depend on their wave size. `--spirv --alu-stats`: the programs whose every ALU
/// instruction translates, and the unsupported ALU ops by number of programs.
fn spirv_stats(files: &[String], alu_only: bool, subgroup: u32) -> ExitCode {
    use shader::spirv::{alu, SpirvError};
    use shader::{ir, spirv, structure};
    use std::collections::{BTreeMap, BTreeSet};
    let scratch =
        std::env::temp_dir().join(format!("nativ64-spirv-stats-{}.spv", std::process::id()));
    // per stage: programs with IR, structured, translated, validated
    let mut stage: BTreeMap<String, [u32; 4]> = BTreeMap::new();
    // per stage: structured programs refused for their wave size, those of them that would
    // translate and validate on a whole wave, and the dependence found
    let mut narrow: BTreeMap<String, [u32; 2]> = BTreeMap::new();
    let mut why: BTreeMap<(String, &str), u32> = BTreeMap::new();
    let mut first: BTreeMap<(String, u8, String), u32> = BTreeMap::new();
    let mut other: BTreeMap<String, u32> = BTreeMap::new();
    let mut invalid: Vec<(u64, String)> = Vec::new();
    let mut no_val = false;
    // ALU coverage: programs, with every ALU instruction supported, per stage
    let mut alu_ok: BTreeMap<String, [u32; 4]> = BTreeMap::new();
    let mut alu_bad: BTreeMap<(u8, String), u32> = BTreeMap::new();
    let mut alu_insts = (0u64, 0u64);
    for f in files {
        let Ok(bytes) = std::fs::read(f) else {
            continue;
        };
        let Ok(p) = shader::parse(&bytes) else {
            continue;
        };
        for (name, rec) in p.native.records() {
            // a vertex or pixel program of the blob's own stage gets its real entry point
            let own = p.record().is_some_and(|r| std::ptr::eq(r, rec));
            let io = if own && !alu_only {
                shader::io::stage_io(&p, &[], None)
            } else {
                None
            };
            let code = match io {
                Some(_) => shader::io::stage_code(&p, rec),
                None => p.native.code(rec).to_vec(),
            };
            let regs = ir::EntryRegs::from_stage(rec);
            let Ok(func) = ir::build(&code, gfx10::ENTRY, rec.wave32(), &regs) else {
                continue;
            };
            let st = structure::structure(&func);
            if alu_only {
                let e = alu_ok.entry(name.to_string()).or_default();
                e[0] += 1;
                e[1] += u32::from(st.is_ok());
                let mut bad: BTreeSet<(u8, String)> = BTreeSet::new();
                for blk in &func.blocks {
                    for i in &blk.insts {
                        if alu::non_alu_chunk(&i.instr).is_some() {
                            continue;
                        }
                        alu_insts.0 += 1;
                        if let Err(u) = alu::supported(i) {
                            alu_insts.1 += 1;
                            bad.insert((u.chunk, u.op));
                        }
                    }
                }
                e[2] += u32::from(bad.is_empty());
                e[3] += u32::from(bad.is_empty() && st.is_ok());
                for b in bad {
                    *alu_bad.entry(b).or_default() += 1;
                }
                continue;
            }
            let e = stage.entry(name.to_string()).or_default();
            e[0] += 1;
            let Ok(st) = st else {
                continue;
            };
            e[1] += 1;
            let refused = shader::io::refuse_wave(&func, &st, subgroup).is_err();
            if refused {
                narrow.entry(name.to_string()).or_default()[0] += 1;
                if let Some(d) = structure::wave_dependence(&func, &st) {
                    *why.entry((name.to_string(), d.what)).or_default() += 1;
                }
            }
            let built = match &io {
                Some(io) => shader::io::refuse_entry(rec)
                    .and_then(|()| spirv::translate_io(&func, &st, io))
                    .map(|t| t.words),
                None => spirv::translate(&func, &st).map(|p| p.words),
            };
            match built {
                Ok(words) if refused => {
                    if matches!(spirv_val(&words, &scratch), Some(Ok(()))) {
                        narrow.entry(name.to_string()).or_default()[1] += 1;
                    }
                }
                Ok(words) => {
                    e[2] += 1;
                    match spirv_val(&words, &scratch) {
                        Some(Ok(())) => e[3] += 1,
                        Some(Err(msg)) => invalid.push((
                            shader::fnv1a64(&bytes),
                            msg.lines().next().unwrap_or("").to_string(),
                        )),
                        None => no_val = true,
                    }
                }
                // refused for the wave size anyway: counted above
                Err(_) if refused => {}
                Err(SpirvError::Unsupported(u)) => {
                    *first.entry((name.to_string(), u.chunk, u.op)).or_default() += 1;
                }
                Err(SpirvError::Malformed(m)) => *other.entry(m).or_default() += 1,
            }
        }
    }
    let _ = std::fs::remove_file(&scratch);
    let top = |mut v: Vec<(String, u32)>| {
        v.sort_by_key(|a| std::cmp::Reverse(a.1));
        v
    };
    if alu_only {
        let sum = alu_ok.values().fold([0; 4], |a, v| {
            [a[0] + v[0], a[1] + v[1], a[2] + v[2], a[3] + v[3]]
        });
        println!(
            "ALU coverage: {}/{} programs with IR have every ALU instruction supported \
             ({}/{} of the structured ones)",
            sum[2], sum[0], sum[3], sum[1]
        );
        println!(
            "ALU instructions: {} of {} unsupported",
            alu_insts.1, alu_insts.0
        );
        for (s, v) in &alu_ok {
            println!("  {s}: {}/{} ({}/{} structured)", v[2], v[0], v[3], v[1]);
        }
        println!("unsupported ALU ops (programs that have one):");
        let v = alu_bad
            .into_iter()
            .map(|((c, o), n)| (format!("{o} (chunk {c})"), n))
            .collect();
        for (o, n) in top(v) {
            println!("{n:6}  {o}");
        }
        return ExitCode::SUCCESS;
    }
    let sum = stage.values().fold([0; 4], |a, v| {
        [a[0] + v[0], a[1] + v[1], a[2] + v[2], a[3] + v[3]]
    });
    println!(
        "programs with IR: {}; structured: {}; translated: {}; validated: {}{}",
        sum[0],
        sum[1],
        sum[2],
        sum[3],
        if no_val { " (spirv-val not found)" } else { "" }
    );
    for (s, v) in &stage {
        println!(
            "  {s}: {} IR, {} structured, {} translated, {} validated",
            v[0], v[1], v[2], v[3]
        );
    }
    if subgroup < WHOLE_WAVE {
        println!("refused for their wave size on subgroups of {subgroup}:");
        for (s, v) in &narrow {
            println!(
                "  {s}: {} structured, {} of them valid on a whole wave",
                v[0], v[1]
            );
        }
        println!("first dependence of those:");
        for ((s, what), n) in &why {
            println!("{n:6}  {s}: {what}");
        }
    }
    println!("first unsupported op of the structured programs:");
    let v = first
        .into_iter()
        .map(|((s, c, o), n)| (format!("{s}: {o} (chunk {c})"), n))
        .collect();
    for (o, n) in top(v) {
        println!("{n:6}  {o}");
    }
    for (m, n) in top(other.into_iter().collect()) {
        println!("{n:6}  malformed: {m}");
    }
    for (hash, msg) in invalid.iter().take(10) {
        println!("INVALID {hash:016x}: {msg}");
    }
    ExitCode::SUCCESS
}
