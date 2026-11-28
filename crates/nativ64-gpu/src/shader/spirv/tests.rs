use std::collections::{BTreeMap, HashMap};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use super::alu::{exec, Cpu, Ctx, Opd, Unsupported, Wave};
use super::emit::{translate, Probe, SpirvError};
use super::gpu::{self, Gpu, Run};
use crate::shader::ir::{build, EntryRegs, Func, Reg, Src, Term, Val};
use crate::shader::structure::{structure, BranchKind, Guard, LaneUse, Scan, Structured, WaveOp};

// Test programs are `fixtures.txt`: `hex<TAB>text` pairs, the text being what
// `llvm-mc -arch=amdgcn -mcpu=gfx1013 -mattr=+wavefrontsize32 -show-encoding` assembled to that
// hex (wavefrontsize64 for the lines that start with `[w64] `). A case is a few of these lines
// plus `s_endpgm`; its vectors give registers on entry and the registers expected at the end.

const FIXTURES: &str = include_str!("fixtures.txt");

fn encoding(text: &str) -> Vec<u8> {
    for l in FIXTURES.lines() {
        if let Some((hex, t)) = l.split_once('\t') {
            if t == text {
                return (0..hex.len() / 2)
                    .map(|k| u8::from_str_radix(&hex[2 * k..2 * k + 2], 16).unwrap())
                    .collect();
            }
        }
    }
    panic!("no fixture for `{text}`");
}

fn code(lines: &[&str]) -> Vec<u8> {
    let mut c: Vec<u8> = lines.iter().flat_map(|l| encoding(l)).collect();
    c.extend([0x00, 0x00, 0x81, 0xbf]); // s_endpgm
    c
}

fn func(lines: &[&str]) -> Func {
    let wave32 = !lines.iter().any(|l| l.starts_with("[w64]"));
    let regs = EntryRegs::new().with_sgprs(0..16).with_vgprs(0..8);
    build(&code(lines), 0, wave32, &regs).unwrap()
}

fn words(w: &[u32]) -> Func {
    let code: Vec<u8> = w.iter().flat_map(|w| w.to_le_bytes()).collect();
    let regs = EntryRegs::new().with_sgprs(8..12).with_vgprs(0..4);
    build(&code, 0, false, &regs).unwrap()
}

// ---------------------------------------------------------------------------------------
// The evaluator: the program on the CPU backend, by its SSA values

/// The last value written to each register variable (a mask as 0 or 1).
type Out = BTreeMap<(Reg, bool), u32>;

const NAN: u32 = 0x7fc0_0000;

fn rd(env: &HashMap<Val, Opd<Cpu>>, s: Src) -> Result<Opd<Cpu>, SpirvError> {
    Ok(match s {
        Src::Val(v) => env[&v],
        Src::Const(c) => Opd::V(c),
        Src::Undef => Opd::V(0),
        _ => {
            return Err(SpirvError::Unsupported(Unsupported {
                chunk: 9,
                op: "code address".into(),
            }))
        }
    })
}

fn as_u32(o: Opd<Cpu>) -> u32 {
    match o {
        Opd::V(v) => v,
        Opd::B(b) => u32::from(b),
    }
}

fn nonzero(o: Opd<Cpu>) -> bool {
    match o {
        Opd::V(v) => v != 0,
        Opd::B(b) => b,
    }
}

fn interp(f: &Func, s: &Structured, ins: &[(Reg, u32)]) -> Result<Out, SpirvError> {
    interp_lane(f, s, ins, 0)
}

/// One console lane of the program, alone: `lane` is its lane index (what `v_mbcnt(-1, .)`
/// counts). Programs without a subgroup operation can run each lane of a wave this way (a lane
/// may then branch its own way); `interp_wave` runs the lanes together.
fn interp_lane(f: &Func, s: &Structured, ins: &[(Reg, u32)], lane: u32) -> Result<Out, SpirvError> {
    Ok(interp_wave(f, s, &[ins.to_vec()], &[lane], 1)?.remove(0))
}

/// The state of one lane of the evaluator.
struct LaneSt {
    env: HashMap<Val, Opd<Cpu>>,
    slots: HashMap<(u16, u8), u32>,
    out: Out,
    lane: u32,
}

/// What a subgroup instruction made of one lane's inputs.
#[derive(Clone, Copy)]
enum Swap {
    /// The result of the instruction.
    Result(u32),
    /// The source operand of a DPP instruction, already swapped.
    Source(u32),
}

fn lane_inst(
    f: &Func,
    s: &Structured,
    (cb, k): (usize, usize),
    st: &mut LaneSt,
    swap: Option<Swap>,
) -> Result<(), SpirvError> {
    let i = &f.blocks[cb].insts[k];
    match s.masks.lanes.get(&i.instr.at) {
        Some(&LaneUse::SlotWrite { vgpr, lane }) => {
            st.slots
                .insert((vgpr, lane), as_u32(rd(&st.env, i.srcs[0][0])?));
            return Ok(());
        }
        Some(&LaneUse::SlotRead { vgpr, lane }) => {
            let x = st.slots.get(&(vgpr, lane)).copied().unwrap_or(0);
            let mask = s.masks.is_mask(i.dsts[0].1);
            st.env.insert(
                i.dsts[0].1,
                if mask { Opd::B(x & 1 != 0) } else { Opd::V(x) },
            );
            st.out.insert((i.dsts[0].0, mask), x);
            return Ok(());
        }
        _ => {}
    }
    let mut srcs = i
        .srcs
        .iter()
        .map(|v| {
            v.iter()
                .map(|&d| rd(&st.env, d))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let imps = i
        .imps
        .iter()
        .map(|&(r, d)| Ok((r, rd(&st.env, d)?)))
        .collect::<Result<Vec<_>, SpirvError>>()?;
    let mut wave = Wave::None;
    match swap {
        Some(Swap::Result(x)) => wave = Wave::Result(Opd::V(x)),
        Some(Swap::Source(x)) => {
            srcs[0][0] = Opd::V(x);
            wave = Wave::Source;
        }
        None => {}
    }
    let active = if s.guards[cb][k] == Guard::Active {
        i.imps
            .iter()
            .find(|(r, _)| *r == Reg::ExecLo)
            .map(|&(_, d)| {
                nonzero(
                    st.env[&match d {
                        Src::Val(v) => v,
                        _ => panic!("EXEC is a value"),
                    }],
                )
            })
    } else {
        None
    };
    let ctx = Ctx {
        inst: i,
        srcs,
        imps,
        dst_mask: i.dsts.iter().map(|(_, v)| s.masks.is_mask(*v)).collect(),
        active,
        lane: Some(st.lane),
        wave,
    };
    let res = exec(&mut Cpu, &ctx)?;
    for ((reg, val), o) in i.dsts.iter().zip(res) {
        st.env.insert(*val, o);
        st.out.insert((*reg, s.masks.is_mask(*val)), as_u32(o));
    }
    Ok(())
}

fn identity(op: WaveOp) -> u32 {
    match op {
        WaveOp::IAdd | WaveOp::FAdd | WaveOp::UMax | WaveOp::Or | WaveOp::Xor => 0,
        WaveOp::UMin | WaveOp::And => u32::MAX,
        WaveOp::SMin => i32::MAX as u32,
        WaveOp::SMax => i32::MIN as u32,
        WaveOp::FMin => f32::INFINITY.to_bits(),
        WaveOp::FMax => f32::NEG_INFINITY.to_bits(),
    }
}

/// One step of a host's group operation.
fn group_step(op: WaveOp, a: u32, b: u32) -> u32 {
    let float = |f: fn(f32, f32) -> f32| f(f32::from_bits(a), f32::from_bits(b)).to_bits();
    match op {
        WaveOp::IAdd => a.wrapping_add(b),
        WaveOp::FAdd => float(|x, y| x + y),
        WaveOp::UMin => a.min(b),
        WaveOp::UMax => a.max(b),
        WaveOp::SMin => (a as i32).min(b as i32) as u32,
        WaveOp::SMax => (a as i32).max(b as i32) as u32,
        WaveOp::FMin => float(f32::min),
        WaveOp::FMax => float(f32::max),
        WaveOp::And => a & b,
        WaveOp::Or => a | b,
        WaveOp::Xor => a ^ b,
    }
}

/// The program on every lane of a wave at once, in lock step (every lane takes the same path:
/// the subgroup instructions need all of them). `lanes` are the console lane indices, `host`
/// the width of the host's subgroups: the lanes are cut in groups of `host` that the subgroup
/// operations (reductions, ballots) work in, quads being 4 lanes on any host.
fn interp_wave(
    f: &Func,
    s: &Structured,
    ins: &[Vec<(Reg, u32)>],
    lanes: &[u32],
    host: usize,
) -> Result<Vec<Out>, SpirvError> {
    let mut sts: Vec<LaneSt> = Vec::new();
    for (n, &lane) in lanes.iter().enumerate() {
        let mut st = LaneSt {
            env: HashMap::new(),
            slots: HashMap::new(),
            out: Out::new(),
            lane,
        };
        for &(reg, val) in &f.args {
            let o = if s.masks.is_mask(val) {
                Opd::B(true)
            } else {
                Opd::V(ins[n].iter().find(|(r, _)| *r == reg).map_or(0, |x| x.1))
            };
            st.env.insert(val, o);
        }
        sts.push(st);
    }
    let mut cb = f.entry;
    let mut prev: Option<usize> = None;
    for _ in 0..100_000 {
        let blk = &f.blocks[cb];
        for st in &mut sts {
            let mut phis = Vec::new();
            for p in &blk.phis {
                let (_, src) = p.args.iter().find(|(b, _)| Some(*b) == prev).unwrap();
                phis.push((p.reg, p.dst, rd(&st.env, *src)?));
            }
            for (reg, dst, o) in phis {
                st.env.insert(dst, o);
                st.out.insert((reg, s.masks.is_mask(dst)), as_u32(o));
            }
        }
        for (k, i) in blk.insts.iter().enumerate() {
            let at = (cb, k);
            let use_ = s.masks.lanes.get(&i.instr.at).copied();
            match use_ {
                // only the section's results are computed; the rest holds nothing
                Some(LaneUse::WaveSkip) => {
                    for st in &mut sts {
                        for &(_, v) in &i.dsts {
                            st.env.insert(v, Opd::V(0));
                        }
                    }
                }
                Some(LaneUse::WaveOut { op, scan, leaf }) => {
                    let xs: Vec<u32> = sts
                        .iter()
                        .map(|st| rd(&st.env, Src::Val(leaf)).map(as_u32))
                        .collect::<Result<_, _>>()?;
                    let (reg, val) = i.dsts[0];
                    for (n, st) in sts.iter_mut().enumerate() {
                        let g0 = n / host * host;
                        let end = match scan {
                            Scan::Reduce => (g0 + host).min(xs.len()),
                            Scan::Inclusive => n + 1,
                            Scan::Exclusive => n,
                        };
                        let r = xs[g0..end]
                            .iter()
                            .fold(identity(op), |a, &b| group_step(op, a, b));
                        st.env.insert(val, Opd::V(r));
                        st.out.insert((reg, false), r);
                    }
                }
                Some(LaneUse::Quad { sel }) => {
                    let xs: Vec<u32> = sts
                        .iter()
                        .map(|st| rd(&st.env, i.srcs[0][0]).map(as_u32))
                        .collect::<Result<_, _>>()?;
                    let dpp = i.instr.dpp.is_some();
                    for (n, st) in sts.iter_mut().enumerate() {
                        let x = xs[(n & !3) | sel[n & 3] as usize];
                        lane_inst(
                            f,
                            s,
                            at,
                            st,
                            Some(if dpp {
                                Swap::Source(x)
                            } else {
                                Swap::Result(x)
                            }),
                        )?;
                    }
                }
                Some(u @ (LaneUse::MaskZero { eq } | LaneUse::MasksEqual { eq })) => {
                    // the two operands' bits per lane, and the group's answer
                    let bit = |st: &LaneSt, k: usize| {
                        i.srcs[k].iter().try_fold(false, |a, &d| {
                            rd(&st.env, d).map(|o| a | (as_u32(o) & 1 != 0))
                        })
                    };
                    let (mut a, mut b) = (Vec::new(), Vec::new());
                    for st in &sts {
                        a.push(bit(st, 0)?);
                        b.push(bit(st, 1)?);
                    }
                    for n in 0..sts.len() {
                        let g0 = n / host * host;
                        let group = g0..(g0 + host).min(sts.len());
                        let same = if matches!(u, LaneUse::MaskZero { .. }) {
                            !group.clone().any(|k| a[k] | b[k])
                        } else {
                            group.clone().all(|k| a[k] == b[k])
                        };
                        let flag = same == eq;
                        lane_inst(f, s, at, &mut sts[n], Some(Swap::Result(u32::from(flag))))?;
                    }
                }
                Some(
                    u @ (LaneUse::BallotCount
                    | LaneUse::FirstSet
                    | LaneUse::Rank { .. }
                    | LaneUse::ReadHostLane),
                ) => {
                    // the lane's bit of the mask operand, and its lane operand / count source
                    let bits: Vec<bool> = sts
                        .iter()
                        .map(|st| {
                            i.srcs[0].iter().try_fold(false, |a, &d| {
                                rd(&st.env, d).map(|o| a | (as_u32(o) & 1 != 0))
                            })
                        })
                        .collect::<Result<_, _>>()?;
                    let src = |st: &LaneSt, k: usize| rd(&st.env, i.srcs[k][0]).map(as_u32);
                    let vals: Vec<u32> = sts
                        .iter()
                        .map(|st| match u {
                            LaneUse::ReadHostLane | LaneUse::Rank { .. } => {
                                src(st, if u == LaneUse::ReadHostLane { 0 } else { 1 })
                            }
                            _ => Ok(0),
                        })
                        .collect::<Result<_, _>>()?;
                    let mut res = Vec::new();
                    for n in 0..sts.len() {
                        let g0 = n / host * host;
                        let group = g0..(g0 + host).min(sts.len());
                        res.push(match u {
                            LaneUse::BallotCount => {
                                bits[group].iter().filter(|b| **b).count() as u32
                            }
                            LaneUse::FirstSet => bits[group]
                                .iter()
                                .position(|b| *b)
                                .map_or(u32::MAX, |p| p as u32),
                            LaneUse::Rank { hi: false } => {
                                (g0..n).filter(|&k| bits[k]).count() as u32 + vals[n]
                            }
                            LaneUse::Rank { hi: true } => vals[n],
                            _ => {
                                // the lane operand is a host lane of the group
                                let lane = src(&sts[n], 1)? as usize;
                                vals[g0 + lane]
                            }
                        });
                    }
                    for (st, x) in sts.iter_mut().zip(res) {
                        lane_inst(f, s, at, st, Some(Swap::Result(x)))?;
                    }
                }
                _ => {
                    for st in &mut sts {
                        lane_inst(f, s, at, st, None)?;
                    }
                }
            }
        }
        let next = match &blk.term {
            Term::End => return Ok(sts.into_iter().map(|st| st.out).collect()),
            Term::Jump(t) => *t,
            Term::Branch {
                op,
                cond,
                taken,
                fall,
            } => {
                let mut gos = Vec::new();
                for st in &sts {
                    let go = match &s.branches[cb] {
                        Some(BranchKind::Lane(t) | BranchKind::Wave(t)) => {
                            let bit = nonzero(st.env[&t.vals[0]]);
                            bit == t.taken_when_set
                        }
                        _ => {
                            let nz = cond
                                .iter()
                                .map(|(_, c)| rd(&st.env, *c).map(nonzero))
                                .collect::<Result<Vec<_>, _>>()?
                                .into_iter()
                                .any(|x| x);
                            let set = matches!(
                                *op,
                                "s_cbranch_scc1" | "s_cbranch_vccnz" | "s_cbranch_execnz"
                            );
                            !cond.is_empty() && nz == set
                        }
                    };
                    gos.push(go);
                }
                assert!(
                    gos.iter().all(|g| *g == gos[0]) || sts.len() == 1,
                    "the lanes of a lock-step wave take different paths"
                );
                if gos[0] {
                    *taken
                } else {
                    *fall
                }
            }
        };
        prev = Some(cb);
        cb = next;
    }
    panic!("the program does not end");
}

fn get(out: &Out, r: Reg) -> u32 {
    out.get(&(r, true))
        .or_else(|| out.get(&(r, false)))
        .copied()
        .unwrap_or_else(|| panic!("{r} was not written: {out:?}"))
}

// ---------------------------------------------------------------------------------------
// Validation

pub(crate) fn validate(words: &[u32]) {
    static N: AtomicUsize = AtomicUsize::new(0);
    if Command::new("spirv-val").arg("--version").output().is_err() {
        eprintln!("spirv-val is not in PATH: validation skipped");
        return;
    }
    let path = std::env::temp_dir().join(format!(
        "nativ64-spirv-{}-{}.spv",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&path, bytes).unwrap();
    let r = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.2"])
        .arg(&path)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(
        r.status.success(),
        "spirv-val: {}",
        String::from_utf8_lossy(&r.stderr)
    );
}

fn module_of(f: &Func) -> Result<Vec<u32>, SpirvError> {
    let s = structure(f).unwrap_or_else(|e| panic!("{e}"));
    let p = translate(f, &s)?;
    validate(&p.words);
    Ok(p.words)
}

// ---------------------------------------------------------------------------------------
// Optional: the same vectors on a Vulkan device (`NATIV64_SPIRV_RUN=1`), compared with the
// evaluator register by register

static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn runner(p: &Probe) -> Option<Run<'static>> {
    if !gpu::enabled() {
        return None;
    }
    let g = GPU
        .get_or_init(|| {
            Some(Gpu::new().unwrap_or_else(|e| panic!("NATIV64_SPIRV_RUN is set: {e}")))
        })
        .as_ref()?;
    Some(
        g.load(&p.words, p.inputs.len(), p.outputs.len())
            .unwrap_or_else(|e| panic!("the device refuses the module: {e}")),
    )
}

/// Float results of the transcendental opcodes differ in the last bits between hosts.
fn loose(asm: &[&str]) -> bool {
    asm.iter().any(|a| {
        [
            "v_sin_f32",
            "v_cos_f32",
            "v_exp_f32",
            "v_log_f32",
            "v_rcp_f32",
            "v_rsq_f32",
            "v_sqrt_f32",
        ]
        .iter()
        .any(|n| a.contains(n))
    })
}

fn on_gpu(run: &Option<Run<'static>>, p: &Probe, asm: &[&str], ins: &[(Reg, u32)], want: &Out) {
    let Some(run) = run else {
        return;
    };
    let words: Vec<u32> = p
        .inputs
        .iter()
        .map(|r| ins.iter().find(|(x, _)| x == r).map_or(0, |x| x.1))
        .collect();
    let got = {
        let _l = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        run.go(&words).unwrap_or_else(|e| panic!("{e}"))
    };
    for (k, key) in p.outputs.iter().enumerate() {
        let Some(&w) = want.get(key) else {
            continue;
        };
        let (a, b) = (got[k], w);
        let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
        let ok = a == b
            || (fa.is_nan() && fb.is_nan())
            || (loose(asm) && (fa - fb).abs() <= 1e-5 * fa.abs().max(fb.abs()).max(1e-30));
        assert!(
            ok,
            "{asm:?} {ins:x?}: {} on the device {a:#010x} ({fa}), evaluator {b:#010x} ({fb})",
            key.0
        );
    }
}

// ---------------------------------------------------------------------------------------
// Vectors

type Io<'a> = &'a [(Reg, u32)];

const V0: Reg = Reg::V(0);
const V1: Reg = Reg::V(1);
const V2: Reg = Reg::V(2);
const V3: Reg = Reg::V(3);
const V4: Reg = Reg::V(4);
const S0: Reg = Reg::S(0);
const S1: Reg = Reg::S(1);
const S2: Reg = Reg::S(2);
const S3: Reg = Reg::S(3);
const S4: Reg = Reg::S(4);
const S6: Reg = Reg::S(6);
const S7: Reg = Reg::S(7);
const SCC: Reg = Reg::Scc;
const VCC: Reg = Reg::VccLo;
const EXEC: Reg = Reg::ExecLo;

const fn f(x: f32) -> u32 {
    x.to_bits()
}

const INF: f32 = f32::INFINITY;

/// Runs a case: the program translates and validates, and every vector gives its outputs.
fn check(asm: &[&str], vecs: &[(Io, Io)]) {
    let fun = func(asm);
    let s = structure(&fun).unwrap_or_else(|e| panic!("{asm:?}: {e}"));
    let p = translate(&fun, &s).unwrap_or_else(|e| panic!("{asm:?}: {e}"));
    validate(&p.words);
    let run = runner(&p);
    for (k, (ins, outs)) in vecs.iter().enumerate() {
        let out = interp(&fun, &s, ins).unwrap_or_else(|e| panic!("{asm:?}: {e}"));
        on_gpu(&run, &p, asm, ins, &out);
        for &(r, want) in *outs {
            let got = get(&out, r);
            let ok = got == want || (want == NAN && f32::from_bits(got).is_nan());
            assert!(
                ok,
                "{asm:?} vector {k} {ins:x?}: {r} = {got:#010x} ({}), want {want:#010x} ({})",
                f32::from_bits(got),
                f32::from_bits(want)
            );
        }
    }
}

fn near(asm: &[&str], x: u32, want: f32, tol: f32) {
    let fun = func(asm);
    let s = structure(&fun).unwrap();
    let p = translate(&fun, &s).unwrap();
    validate(&p.words);
    let out = interp(&fun, &s, &[(V0, x)]).unwrap();
    on_gpu(&runner(&p), &p, asm, &[(V0, x)], &out);
    let got = f32::from_bits(get(&out, V3));
    assert!(
        got == want || (got - want).abs() <= tol,
        "{asm:?} {}: {got}, want {want}",
        f32::from_bits(x)
    );
}

#[test]
fn float_arithmetic() {
    let (n, z, nz) = (NAN, 0.0f32, -0.0f32);
    let two = |a: f32, b: f32| -> Io { Box::leak(Box::new([(V0, f(a)), (V1, f(b))])) };
    let one = |r: u32| -> Io { Box::leak(Box::new([(V3, r)])) };
    check(
        &["v_add_f32 v3, v0, v1"],
        &[
            (two(1.0, 2.0), one(f(3.0))),
            (two(INF, -INF), one(n)),
            (two(nz, nz), one(f(nz))),
        ],
    );
    check(&["v_sub_f32 v3, v0, v1"], &[(two(3.0, 1.0), one(f(2.0)))]);
    check(
        &["v_subrev_f32 v3, v0, v1"],
        &[(two(3.0, 1.0), one(f(-2.0)))],
    );
    check(
        &["v_mul_f32 v3, v0, v1"],
        &[(two(2.0, 4.0), one(f(8.0))), (two(0.0, INF), one(n))],
    );
    // the legacy multiply: 0 * x = 0, even for infinities and NaNs
    check(
        &["v_mul_legacy_f32 v3, v0, v1"],
        &[
            (two(2.0, 3.0), one(f(6.0))),
            (two(0.0, INF), one(f(0.0))),
            (two(INF, 0.0), one(f(0.0))),
            (&[(V0, n), (V1, f(0.0))], one(f(0.0))),
            (two(INF, 2.0), one(f(INF))),
        ],
    );
    // IEEE-mode min / max: a NaN operand gives the other, -0 is below +0
    let min = [
        (two(1.0, 2.0), one(f(1.0))),
        (&[(V0, n), (V1, f(2.0))] as Io, one(f(2.0))),
        (&[(V0, f(2.0)), (V1, n)], one(f(2.0))),
        (&[(V0, n), (V1, n)], one(n)),
        (two(z, nz), one(f(nz))),
        (two(nz, z), one(f(nz))),
    ];
    check(&["v_min_f32 v3, v0, v1"], &min);
    let max = [
        (two(1.0, 2.0), one(f(2.0))),
        (&[(V0, n), (V1, f(2.0))] as Io, one(f(2.0))),
        (&[(V0, f(2.0)), (V1, n)], one(f(2.0))),
        (two(z, nz), one(f(z))),
        (two(nz, z), one(f(z))),
    ];
    check(&["v_max_f32 v3, v0, v1"], &max);
    let three = |a: f32, b: f32, c: f32| -> Io {
        Box::leak(Box::new([(V0, f(a)), (V1, f(b)), (V2, f(c))]))
    };
    check(
        &["v_fma_f32 v3, v0, v1, v2"],
        &[(three(2.0, 3.0, 4.0), one(f(10.0)))],
    );
    check(
        &["v_mad_f32 v3, v0, v1, v2"],
        &[(three(2.0, 3.0, 4.0), one(f(10.0)))],
    );
    check(
        &["v_mad_legacy_f32 v3, v0, v1, v2"],
        &[
            (three(2.0, 3.0, 4.0), one(f(10.0))),
            (three(0.0, INF, 5.0), one(f(5.0))),
        ],
    );
    // the destination is read: v3 starts at 1.0
    let acc =
        |a: f32, b: f32| -> Io { Box::leak(Box::new([(V0, f(a)), (V1, f(b)), (V3, f(1.0))])) };
    check(&["v_fmac_f32 v3, v0, v1"], &[(acc(2.0, 3.0), one(f(7.0)))]);
    check(&["v_mac_f32 v3, v0, v1"], &[(acc(2.0, 3.0), one(f(7.0)))]);
    // literal forms: 0x40400000 = 3.0
    for t in [
        "v_madak_f32 v3, v0, v1, 0x40400000",
        "v_fmaak_f32 v3, v0, v1, 0x40400000",
    ] {
        check(&[t], &[(two(2.0, 4.0), one(f(11.0)))]);
    }
    for t in [
        "v_madmk_f32 v3, v0, 0x40400000, v1",
        "v_fmamk_f32 v3, v0, 0x40400000, v1",
    ] {
        check(&[t], &[(two(2.0, 4.0), one(f(10.0)))]);
    }
    check(
        &["v_med3_f32 v3, v0, v1, v2"],
        &[
            (three(1.0, 5.0, 3.0), one(f(3.0))),
            (three(5.0, 1.0, 3.0), one(f(3.0))),
            (three(3.0, 3.0, 1.0), one(f(3.0))),
            (three(-2.0, -7.0, 4.0), one(f(-2.0))),
            // a NaN operand: the minimum of the others (the table's reading, unverified)
            (&[(V0, n), (V1, f(1.0)), (V2, f(2.0))], one(f(1.0))),
        ],
    );
    check(
        &["v_min3_f32 v3, v0, v1, v2"],
        &[(three(3.0, 1.0, 2.0), one(f(1.0)))],
    );
    check(
        &["v_max3_f32 v3, v0, v1, v2"],
        &[(three(3.0, 1.0, 2.0), one(f(3.0)))],
    );
}

#[test]
fn float_modifiers() {
    let two = |a: f32, b: f32| -> Io { Box::leak(Box::new([(V0, f(a)), (V1, f(b))])) };
    let one = |r: u32| -> Io { Box::leak(Box::new([(V3, r)])) };
    // -|v0| + v1, clamped to [0, 1], NaN to 0
    check(
        &["v_add_f32_e64 v3, -|v0|, v1 clamp"],
        &[
            (two(-2.0, 3.0), one(f(1.0))),
            (two(1.0, 0.25), one(f(0.0))),
            (two(-0.25, 2.0), one(f(1.0))),
            (two(-0.5, 1.0), one(f(0.5))),
            (&[(V0, NAN), (V1, f(1.0))], one(f(0.0))),
        ],
    );
    check(
        &["v_add_f32_e64 v3, |v0|, -v1"],
        &[(two(-2.0, -3.0), one(f(5.0)))],
    );
    check(
        &["v_mul_f32_e64 v3, v0, v1 mul:4"],
        &[(two(1.5, 2.0), one(f(12.0)))],
    );
    check(
        &["v_mul_f32_e64 v3, v0, v1 div:2"],
        &[(two(3.0, 2.0), one(f(3.0)))],
    );
}

#[test]
fn float_unary_and_conversions() {
    let x = |a: u32| -> Io { Box::leak(Box::new([(V0, a)])) };
    let one = |r: u32| -> Io { Box::leak(Box::new([(V3, r)])) };
    check(
        &["v_floor_f32 v3, v0"],
        &[(x(f(-1.5)), one(f(-2.0))), (x(f(2.5)), one(f(2.0)))],
    );
    check(
        &["v_ceil_f32 v3, v0"],
        &[(x(f(-1.5)), one(f(-1.0))), (x(f(1.25)), one(f(2.0)))],
    );
    check(
        &["v_trunc_f32 v3, v0"],
        &[(x(f(-1.5)), one(f(-1.0))), (x(f(1.75)), one(f(1.0)))],
    );
    check(
        &["v_rndne_f32 v3, v0"],
        &[
            (x(f(2.5)), one(f(2.0))),
            (x(f(3.5)), one(f(4.0))),
            (x(f(-0.5)), one(f(-0.0))),
        ],
    );
    check(
        &["v_fract_f32 v3, v0"],
        &[(x(f(1.25)), one(f(0.25))), (x(f(-0.25)), one(f(0.75)))],
    );
    check(
        &["v_rcp_f32 v3, v0"],
        &[(x(f(4.0)), one(f(0.25))), (x(f(0.0)), one(f(INF)))],
    );
    check(&["v_rsq_f32 v3, v0"], &[(x(f(4.0)), one(f(0.5)))]);
    check(
        &["v_sqrt_f32 v3, v0"],
        &[(x(f(16.0)), one(f(4.0))), (x(f(-1.0)), one(NAN))],
    );
    check(
        &["v_exp_f32 v3, v0"],
        &[(x(f(3.0)), one(f(8.0))), (x(f(-INF)), one(f(0.0)))],
    );
    check(
        &["v_log_f32 v3, v0"],
        &[(x(f(8.0)), one(f(3.0))), (x(f(0.0)), one(f(-INF)))],
    );
    // the sine takes revolutions: host precision accepted
    near(&["v_sin_f32 v3, v0"], f(0.25), 1.0, 1e-6);
    near(&["v_sin_f32 v3, v0"], f(0.0), 0.0, 1e-6);
    near(&["v_cos_f32 v3, v0"], f(0.5), -1.0, 1e-6);
    near(&["v_cos_f32 v3, v0"], f(0.0), 1.0, 1e-6);

    check(
        &["v_cvt_f32_u32 v3, v0"],
        &[
            (x(7), one(f(7.0))),
            (x(0xffff_ffff), one(f(4_294_967_296.0))),
        ],
    );
    check(
        &["v_cvt_f32_i32 v3, v0"],
        &[(x(-3i32 as u32), one(f(-3.0)))],
    );
    check(
        &["v_cvt_u32_f32 v3, v0"],
        &[
            (x(f(-1.0)), one(0)),
            (x(f(3.7)), one(3)),
            (x(NAN), one(0)),
            (x(f(5.0e9)), one(0xffff_ffff)),
            (x(f(4.0e9)), one(4_000_000_000)),
        ],
    );
    check(
        &["v_cvt_i32_f32 v3, v0"],
        &[
            (x(f(-3.7)), one(-3i32 as u32)),
            (x(f(3.0e9)), one(0x7fff_ffff)),
            (x(f(-3.0e9)), one(0x8000_0000)),
            (x(NAN), one(0)),
        ],
    );
    check(
        &["v_cvt_flr_i32_f32 v3, v0"],
        &[(x(f(-1.5)), one(-2i32 as u32))],
    );
    check(
        &["v_cvt_rpi_i32_f32 v3, v0"],
        &[(x(f(-1.5)), one(-1i32 as u32)), (x(f(2.5)), one(3))],
    );
    for (k, t) in [
        "v_cvt_f32_ubyte0 v3, v0",
        "v_cvt_f32_ubyte1 v3, v0",
        "v_cvt_f32_ubyte2 v3, v0",
        "v_cvt_f32_ubyte3 v3, v0",
    ]
    .iter()
    .enumerate()
    {
        check(&[t], &[(x(0x0403_0201), one(f(k as f32 + 1.0)))]);
    }
    check(
        &["v_frexp_exp_i32_f32 v3, v0"],
        &[
            (x(f(8.0)), one(4)),
            (x(f(-12.0)), one(4)),
            (x(f(0.5)), one(0)),
            (x(f(0.0)), one(0)),
            (x(f(INF)), one(0)),
        ],
    );
    check(
        &["v_frexp_mant_f32 v3, v0"],
        &[
            (x(f(8.0)), one(f(0.5))),
            (x(f(-12.0)), one(f(-0.75))),
            (x(f(0.0)), one(f(0.0))),
            (x(f(INF)), one(f(INF))),
        ],
    );
    let two = |a: u32, b: u32| -> Io { Box::leak(Box::new([(V0, a), (V1, b)])) };
    check(
        &["v_ldexp_f32 v3, v0, v1"],
        &[
            (two(f(1.5), 3), one(f(12.0))),
            (two(f(1.0), -2i32 as u32), one(f(0.25))),
            (two(f(1.0), 1000), one(f(INF))),
            (two(f(1.0), -1000i32 as u32), one(f(0.0))),
        ],
    );
}

#[test]
fn half_precision_conversions() {
    let two = |a: u32, b: u32| -> Io { Box::leak(Box::new([(V0, a), (V1, b)])) };
    let one = |r: u32| -> Io { Box::leak(Box::new([(V3, r)])) };
    // 1 + 3 * 2^-12 is 3/4 of a half-precision step above 1: round-to-zero keeps 0x3c00
    let above = 1.0 + 3.0 / 4096.0;
    check(
        &["v_cvt_pkrtz_f16_f32 v3, v0, v1"],
        &[
            (two(f(1.0), f(-2.0)), one(0xc000_3c00)),
            (two(f(above), f(65520.0)), one(0x7bff_3c00)),
            (two(f(1.0e-8), f(3.0e-6)), one(0x0032_0000)),
            (two(NAN, f(INF)), one(0x7c00_7e00)),
            (two(f(-INF), f(-65536.0)), one(0xfbff_fc00)),
            (two(f(-1.0e-8), f(6.0e-8)), one(0x0001_8000)),
        ],
    );
    // round to nearest even: 0x3c01
    check(
        &["v_cvt_f16_f32 v3, v0"],
        &[
            (&[(V0, f(1.0))], one(0x3c00)),
            (&[(V0, f(above))], one(0x3c01)),
            (&[(V0, f(65520.0))], one(0x7c00)),
            (&[(V0, f(-2.0))], one(0xc000)),
        ],
    );
    check(
        &["v_cvt_f32_f16 v3, v0"],
        &[
            (&[(V0, 0x3c00)], one(f(1.0))),
            (&[(V0, 0xfc00)], one(f(-INF))),
            (&[(V0, 0x0001)], one(0x3380_0000)),
            (&[(V0, 0x7e00)], one(NAN)),
        ],
    );
    check(
        &["v_pack_b32_f16 v3, v0, v1"],
        &[(two(0x1111_aaaa, 0x2222_bbbb), one(0xbbbb_aaaa))],
    );
    check(
        &["v_pack_b32_f16 v3, v0, v1 op_sel:[1,1]"],
        &[(two(0x1111_aaaa, 0x2222_bbbb), one(0x2222_1111))],
    );
    let three = |a: u32, b: u32, c: u32| -> Io { Box::leak(Box::new([(V0, a), (V1, b), (V2, c)])) };
    check(
        &["v_fma_mix_f32 v3, v0, v1, v2"],
        &[(three(f(2.0), f(3.0), f(4.0)), one(f(10.0)))],
    );
    // sources 0 and 2 are the low halves of f16 values: 1.0 * 3.0 + 2.0
    check(
        &["v_fma_mix_f32 v3, v0, v1, v2 op_sel_hi:[1,0,1]"],
        &[(three(0x3c00, f(3.0), 0x4000), one(f(5.0)))],
    );
    // source 0 is the high half
    check(
        &["v_fma_mix_f32 v3, v0, v1, v2 op_sel:[1,0,0] op_sel_hi:[1,0,0]"],
        &[(three(0x4000_3c00, f(3.0), f(4.0)), one(f(10.0)))],
    );
    // the low or high half of the destination takes the f16 result; the other half stays
    let old = |v: u32| -> Io {
        Box::leak(Box::new([
            (V0, f(2.0)),
            (V1, f(3.0)),
            (V2, f(4.0)),
            (V3, v),
        ]))
    };
    check(
        &["v_fma_mixlo_f16 v3, v0, v1, v2"],
        &[(old(0xabcd_0000), one(0xabcd_4900))],
    );
    check(
        &["v_fma_mixhi_f16 v3, v0, v1, v2"],
        &[(old(0x0000_1234), one(0x4900_1234))],
    );
}

#[test]
fn cube_map_ops() {
    let xyz = |x: f32, y: f32, z: f32| -> Io {
        Box::leak(Box::new([(V0, f(x)), (V1, f(y)), (V2, f(z))]))
    };
    let one = |r: f32| -> Io { Box::leak(Box::new([(V3, f(r))])) };
    // (x, y, z): the major axis and its sign pick the face
    let cases = [
        ((2.0, 1.0, 0.0), 0.0),
        ((-3.0, 1.0, 0.0), 1.0),
        ((0.0, 4.0, 1.0), 2.0),
        ((0.0, -4.0, 1.0), 3.0),
        ((0.5, 1.0, 2.0), 4.0),
        ((0.0, 0.0, -1.0), 5.0),
    ];
    let vecs: Vec<(Io, Io)> = cases
        .iter()
        .map(|&((x, y, z), id)| (xyz(x, y, z), one(id)))
        .collect();
    check(&["v_cubeid_f32 v3, v0, v1, v2"], &vecs);
    check(
        &["v_cubema_f32 v3, v0, v1, v2"],
        &[
            (xyz(2.0, 1.0, 0.0), one(4.0)),
            (xyz(0.0, -4.0, 1.0), one(-8.0)),
            (xyz(0.5, 1.0, 2.0), one(4.0)),
        ],
    );
    check(
        &["v_cubesc_f32 v3, v0, v1, v2"],
        &[
            (xyz(4.0, 1.0, 2.0), one(-2.0)),
            (xyz(-4.0, 1.0, 2.0), one(2.0)),
            (xyz(0.0, -4.0, 1.0), one(0.0)),
            (xyz(0.5, 1.0, 2.0), one(0.5)),
            (xyz(1.0, 0.5, -2.0), one(-1.0)),
        ],
    );
    check(
        &["v_cubetc_f32 v3, v0, v1, v2"],
        &[
            (xyz(4.0, 1.0, 2.0), one(-1.0)),
            (xyz(0.0, -4.0, 1.0), one(-1.0)),
            (xyz(0.0, 4.0, 1.0), one(1.0)),
            (xyz(0.5, 1.0, 2.0), one(-1.0)),
            (xyz(1.0, 0.5, -2.0), one(-0.5)),
        ],
    );
}

const INTS: [u32; 9] = [
    0,
    1,
    2,
    31,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_ffff,
    0x1234_5678,
    0xdead_beef,
];

/// Runs `asm` (inputs v0, v1, v2, result v3) over every triple of the interesting values.
fn sweep(asm: &str, want: impl Fn(u32, u32, u32) -> u32) {
    let fun = func(&[asm]);
    let s = structure(&fun).unwrap();
    let p = translate(&fun, &s).unwrap();
    validate(&p.words);
    let run = runner(&p);
    for a in INTS {
        for b in INTS {
            for c in [0, 5, 0x8000_0000, 0xffff_ffff] {
                let out = interp(&fun, &s, &[(V0, a), (V1, b), (V2, c)]).unwrap();
                on_gpu(&run, &p, &[asm], &[(V0, a), (V1, b), (V2, c)], &out);
                assert_eq!(
                    get(&out, V3),
                    want(a, b, c),
                    "{asm}: {a:#x}, {b:#x}, {c:#x}"
                );
            }
        }
    }
}

#[test]
fn integer_vector_ops() {
    let sh = |x: u32| x & 31;
    sweep("v_add_nc_u32 v3, v0, v1", |a, b, _| a.wrapping_add(b));
    sweep("v_sub_nc_u32 v3, v0, v1", |a, b, _| a.wrapping_sub(b));
    sweep("v_subrev_nc_u32 v3, v0, v1", |a, b, _| b.wrapping_sub(a));
    sweep("v_and_b32 v3, v0, v1", |a, b, _| a & b);
    sweep("v_or_b32 v3, v0, v1", |a, b, _| a | b);
    sweep("v_xor_b32 v3, v0, v1", |a, b, _| a ^ b);
    sweep("v_xnor_b32 v3, v0, v1", |a, b, _| !(a ^ b));
    sweep("v_not_b32 v3, v0", |a, _, _| !a);
    sweep("v_bfrev_b32 v3, v0", |a, _, _| a.reverse_bits());
    sweep("v_lshlrev_b32 v3, v0, v1", move |a, b, _| b << sh(a));
    sweep("v_lshrrev_b32 v3, v0, v1", move |a, b, _| b >> sh(a));
    sweep("v_ashrrev_i32 v3, v0, v1", move |a, b, _| {
        ((b as i32) >> sh(a)) as u32
    });
    sweep("v_min_i32 v3, v0, v1", |a, b, _| {
        (a as i32).min(b as i32) as u32
    });
    sweep("v_max_i32 v3, v0, v1", |a, b, _| {
        (a as i32).max(b as i32) as u32
    });
    sweep("v_min_u32 v3, v0, v1", |a, b, _| a.min(b));
    sweep("v_max_u32 v3, v0, v1", |a, b, _| a.max(b));
    sweep("v_mul_lo_u32 v3, v0, v1", |a, b, _| a.wrapping_mul(b));
    sweep("v_mul_hi_u32 v3, v0, v1", |a, b, _| {
        ((u64::from(a) * u64::from(b)) >> 32) as u32
    });
    sweep("v_mul_hi_i32 v3, v0, v1", |a, b, _| {
        ((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32
    });
    let u24 = |x: u32| u64::from(x & 0xff_ffff);
    let i24 = |x: u32| i64::from(((x << 8) as i32) >> 8);
    sweep("v_mul_u32_u24 v3, v0, v1", move |a, b, _| {
        (u24(a) * u24(b)) as u32
    });
    sweep("v_mul_hi_u32_u24 v3, v0, v1", move |a, b, _| {
        ((u24(a) * u24(b)) >> 32) as u32
    });
    sweep("v_mul_i32_i24 v3, v0, v1", move |a, b, _| {
        (i24(a) * i24(b)) as u32
    });
    sweep("v_mul_hi_i32_i24 v3, v0, v1", move |a, b, _| {
        ((i24(a) * i24(b)) >> 32) as u32
    });
    sweep("v_mad_u32_u24 v3, v0, v1, v2", move |a, b, c| {
        ((u24(a) * u24(b)) as u32).wrapping_add(c)
    });
    sweep("v_mad_i32_i24 v3, v0, v1, v2", move |a, b, c| {
        ((i24(a) * i24(b)) as u32).wrapping_add(c)
    });
    sweep("v_bfi_b32 v3, v0, v1, v2", |m, a, b| (m & a) | (!m & b));
    sweep("v_alignbit_b32 v3, v0, v1, v2", move |a, b, c| {
        ((u64::from(a) << 32 | u64::from(b)) >> sh(c)) as u32
    });
    sweep("v_alignbyte_b32 v3, v0, v1, v2", |a, b, c| {
        ((u64::from(a) << 32 | u64::from(b)) >> (8 * (c & 3))) as u32
    });
    sweep("v_med3_i32 v3, v0, v1, v2", |a, b, c| {
        let mut v = [a as i32, b as i32, c as i32];
        v.sort_unstable();
        v[1] as u32
    });
    sweep("v_med3_u32 v3, v0, v1, v2", |a, b, c| {
        let mut v = [a, b, c];
        v.sort_unstable();
        v[1]
    });
    sweep("v_min3_u32 v3, v0, v1, v2", |a, b, c| a.min(b).min(c));
    sweep("v_max3_i32 v3, v0, v1, v2", |a, b, c| {
        (a as i32).max(b as i32).max(c as i32) as u32
    });
    sweep("v_add3_u32 v3, v0, v1, v2", |a, b, c| {
        a.wrapping_add(b).wrapping_add(c)
    });
    sweep("v_lshl_add_u32 v3, v0, v1, v2", move |a, b, c| {
        (a << sh(b)).wrapping_add(c)
    });
    sweep("v_add_lshl_u32 v3, v0, v1, v2", move |a, b, c| {
        a.wrapping_add(b) << sh(c)
    });
    sweep("v_lshl_or_b32 v3, v0, v1, v2", move |a, b, c| {
        (a << sh(b)) | c
    });
    sweep("v_and_or_b32 v3, v0, v1, v2", |a, b, c| (a & b) | c);
    sweep("v_or3_b32 v3, v0, v1, v2", |a, b, c| a | b | c);
    sweep("v_xad_u32 v3, v0, v1, v2", |a, b, c| {
        (a ^ b).wrapping_add(c)
    });
    sweep("v_bcnt_u32_b32 v3, v0, v1", |a, b, _| {
        a.count_ones().wrapping_add(b)
    });
    sweep("v_bfm_b32 v3, v0, v1", move |a, b, _| {
        (((1u64 << sh(a)) - 1) as u32) << sh(b)
    });
    sweep("v_ffbh_u32 v3, v0", |a, _, _| {
        if a == 0 {
            u32::MAX
        } else {
            a.leading_zeros()
        }
    });
    sweep("v_ffbl_b32 v3, v0", |a, _, _| {
        if a == 0 {
            u32::MAX
        } else {
            a.trailing_zeros()
        }
    });
    sweep("v_ffbh_i32 v3, v0", |a, _, _| {
        if a == 0 || a == u32::MAX {
            u32::MAX
        } else if (a as i32) < 0 {
            (!a).leading_zeros()
        } else {
            a.leading_zeros()
        }
    });
    // field extraction: offset and width are 5 bits; width 0 gives 0
    sweep("v_bfe_u32 v3, v0, v1, v2", move |a, b, c| {
        let w = c & 31;
        if w == 0 {
            0
        } else {
            (a >> sh(b)) & (((1u64 << w) - 1) as u32)
        }
    });
    sweep("v_bfe_i32 v3, v0, v1, v2", move |a, b, c| {
        let w = c & 31;
        if w == 0 {
            0
        } else {
            let x = (((a as i32) >> sh(b)) as u32) & (((1u64 << w) - 1) as u32);
            (((x << (32 - w)) as i32) >> (32 - w)) as u32
        }
    });
}

#[test]
fn byte_permute() {
    let abs = |a: u32, b: u32, c: u32| -> Io { Box::leak(Box::new([(V0, a), (V1, b), (V2, c)])) };
    let one = |r: u32| -> Io { Box::leak(Box::new([(V3, r)])) };
    // bytes 0..=3 are the second source, 4..=7 the first
    check(
        &["v_perm_b32 v3, v0, v1, v2"],
        &[
            (abs(0x0f0e_0d0c, 0x0302_0100, 0x0706_0504), one(0x0f0e_0d0c)),
            (abs(0x0f0e_0d0c, 0x0302_0100, 0x0001_0203), one(0x0001_0203)),
            (abs(0x0f0e_0d0c, 0x0302_0100, 0x0c0c_0c0c), one(0)),
            (abs(0, 0, 0xffff_ffff), one(0xffff_ffff)),
            // 8 and 9: 0xff when bit 15 / 31 of the second source is set; 10, 11: the first
            (abs(0, 0x0000_8000, 0x0c0c_0c08), one(0x0000_00ff)),
            (abs(0, 0x8000_0000, 0x0c0c_0c09), one(0x0000_00ff)),
            (abs(0x0000_8000, 0, 0x0c0c_0c0a), one(0x0000_00ff)),
            (abs(0x0000_8000, 0, 0x0c0c_0c0b), one(0)),
        ],
    );
}

#[test]
fn integer_vector_64_bit_shifts() {
    let ins =
        |lo: u32, hi: u32, n: u32| -> Io { Box::leak(Box::new([(V0, n), (V1, lo), (V2, hi)])) };
    let out = |lo: u32, hi: u32| -> Io { Box::leak(Box::new([(V3, lo), (V4, hi)])) };
    check(
        &["v_lshlrev_b64 v[3:4], v0, v[1:2]"],
        &[
            (ins(0xf000_0000, 1, 4), out(0, 0x1f)),
            (ins(1, 0, 36), out(0, 0x10)),
            (ins(0x1234, 0x5678, 0), out(0x1234, 0x5678)),
            (ins(1, 0, 64), out(1, 0)),
        ],
    );
    check(
        &["v_lshrrev_b64 v[3:4], v0, v[1:2]"],
        &[
            (ins(0, 0x1f, 4), out(0xf000_0000, 1)),
            (ins(0, 0x10, 36), out(1, 0)),
            (ins(0x1234, 0x5678, 0), out(0x1234, 0x5678)),
        ],
    );
    check(
        &["v_ashrrev_i64 v[3:4], v0, v[1:2]"],
        &[
            (ins(0, 0x8000_0000, 36), out(0xf800_0000, 0xffff_ffff)),
            (ins(0, 0x8000_0000, 63), out(0xffff_ffff, 0xffff_ffff)),
            (ins(0, 0x4000_0000, 62), out(1, 0)),
        ],
    );
}

#[test]
fn carries_and_selects() {
    let ab = |a: u32, b: u32| -> Io { Box::leak(Box::new([(V0, a), (V1, b)])) };
    let r = |v: u32, c: u32| -> Io { Box::leak(Box::new([(V3, v), (S2, c)])) };
    check(
        &["v_add_co_u32 v3, s2, v0, v1"],
        &[(ab(0xffff_ffff, 1), r(0, 1)), (ab(1, 2), r(3, 0))],
    );
    check(
        &["v_sub_co_u32 v3, s2, v0, v1"],
        &[(ab(1, 2), r(0xffff_ffff, 1)), (ab(2, 1), r(1, 0))],
    );
    check(
        &["v_subrev_co_u32 v3, s2, v0, v1"],
        &[(ab(2, 1), r(0xffff_ffff, 1)), (ab(1, 2), r(1, 0))],
    );
    // the carry in comes out of a compare: v2 < v4
    let cin = |a: u32, b: u32, c: bool| -> Io {
        // v2 < v4 exactly when `c`
        Box::leak(Box::new([(V0, a), (V1, b), (V2, 0), (V4, u32::from(c))]))
    };
    let rv = |v: u32, c: u32| -> Io { Box::leak(Box::new([(V3, v), (VCC, c)])) };
    let lt = "v_cmp_lt_u32 vcc_lo, v2, v4";
    check(
        &[lt, "v_add_co_ci_u32 v3, vcc_lo, v0, v1, vcc_lo"],
        &[
            (cin(0xffff_fffe, 1, true), rv(0, 1)),
            (cin(0xffff_ffff, 0, true), rv(0, 1)),
            (cin(0xffff_ffff, 0, false), rv(0xffff_ffff, 0)),
            (cin(1, 2, true), rv(4, 0)),
        ],
    );
    check(
        &[lt, "v_sub_co_ci_u32 v3, vcc_lo, v0, v1, vcc_lo"],
        &[
            (cin(0, 0, true), rv(0xffff_ffff, 1)),
            (cin(5, 3, true), rv(1, 0)),
            (cin(3, 3, true), rv(0xffff_ffff, 1)),
            (cin(3, 3, false), rv(0, 0)),
        ],
    );
    check(
        &[lt, "v_subrev_co_ci_u32 v3, vcc_lo, v0, v1, vcc_lo"],
        &[
            (cin(3, 5, true), rv(1, 0)),
            (cin(5, 3, true), rv(0xffff_fffd, 1)),
        ],
    );
    // select by a compare, and by a scalar whose bit 0 is the lane's bit
    let sel = |a: u32, b: u32, m: u32| -> Io { Box::leak(Box::new([(V0, a), (V1, b), (S2, m)])) };
    let o = |v: u32| -> Io { Box::leak(Box::new([(V3, v)])) };
    check(
        &["v_cndmask_b32_e64 v3, v0, v1, s2"],
        &[(sel(10, 20, 0), o(10)), (sel(10, 20, 1), o(20))],
    );
    check(
        &[
            "v_cmp_lt_u32 vcc_lo, v0, v1",
            "v_cndmask_b32 v3, v0, v1, vcc_lo",
        ],
        &[
            (sel(10, 20, 0), o(20)),
            (sel(20, 10, 0), o(20)),
            (sel(7, 7, 0), o(7)),
        ],
    );
}

const FPAIRS: [(f32, f32); 5] = [
    (1.0, 2.0),
    (2.0, 1.0),
    (1.0, 1.0),
    (f32::NAN, 1.0),
    (1.0, f32::NAN),
];

#[test]
fn float_compares() {
    // (opcode, truth on each pair of FPAIRS): ordered tests are false on NaN, `n*` true
    let table: [(&str, [u32; 5]); 16] = [
        ("v_cmp_f_f32 vcc_lo, v0, v1", [0, 0, 0, 0, 0]),
        ("v_cmp_lt_f32 vcc_lo, v0, v1", [1, 0, 0, 0, 0]),
        ("v_cmp_eq_f32 vcc_lo, v0, v1", [0, 0, 1, 0, 0]),
        ("v_cmp_le_f32 vcc_lo, v0, v1", [1, 0, 1, 0, 0]),
        ("v_cmp_gt_f32 vcc_lo, v0, v1", [0, 1, 0, 0, 0]),
        ("v_cmp_lg_f32 vcc_lo, v0, v1", [1, 1, 0, 0, 0]),
        ("v_cmp_ge_f32 vcc_lo, v0, v1", [0, 1, 1, 0, 0]),
        ("v_cmp_o_f32 vcc_lo, v0, v1", [1, 1, 1, 0, 0]),
        ("v_cmp_u_f32 vcc_lo, v0, v1", [0, 0, 0, 1, 1]),
        ("v_cmp_nge_f32 vcc_lo, v0, v1", [1, 0, 0, 1, 1]),
        ("v_cmp_nlg_f32 vcc_lo, v0, v1", [0, 0, 1, 1, 1]),
        ("v_cmp_ngt_f32 vcc_lo, v0, v1", [1, 0, 1, 1, 1]),
        ("v_cmp_nle_f32 vcc_lo, v0, v1", [0, 1, 0, 1, 1]),
        ("v_cmp_neq_f32 vcc_lo, v0, v1", [1, 1, 0, 1, 1]),
        ("v_cmp_nlt_f32 vcc_lo, v0, v1", [0, 1, 1, 1, 1]),
        ("v_cmp_tru_f32 vcc_lo, v0, v1", [1, 1, 1, 1, 1]),
    ];
    for (asm, truth) in table {
        let fun = func(&[asm]);
        let s = structure(&fun).unwrap();
        let p = translate(&fun, &s).unwrap();
        validate(&p.words);
        let run = runner(&p);
        for (k, (a, b)) in FPAIRS.iter().enumerate() {
            let out = interp(&fun, &s, &[(V0, f(*a)), (V1, f(*b))]).unwrap();
            on_gpu(&run, &p, &[asm], &[(V0, f(*a)), (V1, f(*b))], &out);
            assert_eq!(get(&out, VCC), truth[k], "{asm}: {a}, {b}");
        }
    }
    // the VOP3 form: source modifiers, the result in an SGPR
    check(
        &["v_cmp_lt_f32_e64 s2, -v0, |v1|"],
        &[
            (&[(V0, f(-2.0)), (V1, f(-3.0))], &[(S2, 1)]),
            (&[(V0, f(2.0)), (V1, f(3.0))], &[(S2, 1)]),
            (&[(V0, f(-2.0)), (V1, f(1.0))], &[(S2, 0)]),
        ],
    );
}

#[test]
fn integer_compares_and_class() {
    let pairs: [(u32, u32); 5] = [(1, 2), (2, 1), (5, 5), (0xffff_ffff, 1), (1, 0xffff_ffff)];
    type Cmp = fn(u32, u32) -> bool;
    let ops: [(&str, Cmp, Cmp); 6] = [
        ("eq", |a, b| a == b, |a, b| a == b),
        ("ne", |a, b| a != b, |a, b| a != b),
        ("lt", |a, b| (a as i32) < b as i32, |a, b| a < b),
        ("le", |a, b| (a as i32) <= b as i32, |a, b| a <= b),
        ("gt", |a, b| (a as i32) > b as i32, |a, b| a > b),
        ("ge", |a, b| (a as i32) >= b as i32, |a, b| a >= b),
    ];
    for (op, si, un) in ops {
        for (ty, g) in [("i32", si), ("u32", un)] {
            let asm = format!("v_cmp_{op}_{ty} vcc_lo, v0, v1");
            let fun = func(&[asm.as_str()]);
            let s = structure(&fun).unwrap();
            let p = translate(&fun, &s).unwrap();
            validate(&p.words);
            let run = runner(&p);
            for (a, b) in pairs {
                let out = interp(&fun, &s, &[(V0, a), (V1, b)]).unwrap();
                on_gpu(&run, &p, &[asm.as_str()], &[(V0, a), (V1, b)], &out);
                assert_eq!(get(&out, VCC), u32::from(g(a, b)), "{asm}: {a:#x} {b:#x}");
            }
        }
    }
    // 64-bit compares
    let wide = |lo0, hi0, lo1, hi1| -> Io {
        Box::leak(Box::new([(V0, lo0), (V1, hi0), (V2, lo1), (V3, hi1)]))
    };
    let b = |x: u32| -> Io { Box::leak(Box::new([(VCC, x)])) };
    check(
        &["v_cmp_lt_u64 vcc_lo, v[0:1], v[2:3]"],
        &[
            (wide(5, 1, 4, 2), b(1)),
            (wide(5, 2, 4, 1), b(0)),
            (wide(4, 1, 5, 1), b(1)),
            (wide(5, 1, 5, 1), b(0)),
            (wide(0, 0x8000_0000, 0, 1), b(0)),
        ],
    );
    check(
        &["v_cmp_lt_i64 vcc_lo, v[0:1], v[2:3]"],
        &[
            (wide(0, 0x8000_0000, 0, 1), b(1)),
            (wide(0, 1, 0, 0x8000_0000), b(0)),
        ],
    );
    check(
        &["v_cmp_eq_u64 vcc_lo, v[0:1], v[2:3]"],
        &[
            (wide(1, 2, 1, 2), b(1)),
            (wide(1, 2, 1, 3), b(0)),
            (wide(1, 2, 0, 2), b(0)),
        ],
    );
    // class: the mask bits are sNaN, qNaN, -inf, -normal, -denormal, -0, +0, +denormal,
    // +normal, +inf
    let class = |x: u32, mask: u32, want: u32| {
        check(
            &["v_cmp_class_f32 vcc_lo, v0, v1"],
            &[(&[(V0, x), (V1, mask)][..], &[(VCC, want)][..])],
        );
    };
    class(f(1.0), 0x100, 1);
    class(f(1.0), 0x0ff, 0);
    class(f(-1.0), 0x008, 1);
    class(f(-0.0), 0x020, 1);
    class(f(0.0), 0x020, 0);
    class(f(0.0), 0x040, 1);
    class(f(INF), 0x200, 1);
    class(f(-INF), 0x004, 1);
    class(0x7fc0_0000, 0x003, 1);
    class(0x7fc0_0000, 0x001, 0);
    class(0x7fa0_0000, 0x001, 1);
    class(0x7fa0_0000, 0x002, 0);
    class(0x0000_0001, 0x080, 1);
    class(0x8000_0001, 0x010, 1);
    class(0x8000_0001, 0x080, 0);
    // v_cmpx writes EXEC
    check(
        &["v_cmpx_lt_f32 v0, v1"],
        &[
            (&[(V0, f(1.0)), (V1, f(2.0))], &[(EXEC, 1)]),
            (&[(V0, f(2.0)), (V1, f(1.0))], &[(EXEC, 0)]),
        ],
    );
}

#[test]
fn scalar_arithmetic_and_scc() {
    let ab = |a: u32, b: u32| -> Io { Box::leak(Box::new([(S0, a), (S1, b)])) };
    let r = |v: u32, scc: u32| -> Io { Box::leak(Box::new([(S2, v), (SCC, scc)])) };
    let m = 0xffff_ffffu32;
    check(
        &["s_add_u32 s2, s0, s1"],
        &[(ab(m, 1), r(0, 1)), (ab(1, 2), r(3, 0))],
    );
    check(
        &["s_sub_u32 s2, s0, s1"],
        &[(ab(1, 2), r(m, 1)), (ab(2, 1), r(1, 0))],
    );
    // signed overflow
    check(
        &["s_add_i32 s2, s0, s1"],
        &[
            (ab(0x7fff_ffff, 1), r(0x8000_0000, 1)),
            (ab(1, 2), r(3, 0)),
            (ab(m, m), r(m - 1, 0)),
            (ab(0x8000_0000, 0x8000_0000), r(0, 1)),
        ],
    );
    check(
        &["s_sub_i32 s2, s0, s1"],
        &[
            (ab(0x8000_0000, 1), r(0x7fff_ffff, 1)),
            (ab(0, 1), r(m, 0)),
            (ab(0x7fff_ffff, m), r(0x8000_0000, 1)),
        ],
    );
    // the carry in is SCC of an earlier compare: s6 == s7
    let cin = |a: u32, b: u32, c: bool| -> Io {
        Box::leak(Box::new([(S0, a), (S1, b), (S6, 1), (S7, u32::from(c))]))
    };
    let eq = "s_cmp_eq_u32 s6, s7";
    check(
        &[eq, "s_addc_u32 s2, s0, s1"],
        &[
            (cin(m, 0, true), r(0, 1)),
            (cin(1, 1, true), r(3, 0)),
            (cin(m, 0, false), r(m, 0)),
            (cin(m, m, true), r(m, 1)),
        ],
    );
    check(
        &[eq, "s_subb_u32 s2, s0, s1"],
        &[
            (cin(0, 0, true), r(m, 1)),
            (cin(5, 3, true), r(1, 0)),
            (cin(3, 3, true), r(m, 1)),
            (cin(3, 3, false), r(0, 0)),
        ],
    );
    // the selected operand sets SCC
    check(&["s_min_i32 s2, s0, s1"], &[(ab(m, 1), r(m, 1))]);
    check(&["s_min_u32 s2, s0, s1"], &[(ab(m, 1), r(1, 0))]);
    check(&["s_max_i32 s2, s0, s1"], &[(ab(m, 1), r(1, 0))]);
    check(&["s_max_u32 s2, s0, s1"], &[(ab(m, 1), r(m, 1))]);
    check(
        &["s_mul_i32 s2, s0, s1"],
        &[
            (ab(0x10000, 0x10000), &[(S2, 0)][..]),
            (ab(m, 3), &[(S2, m - 2)][..]),
        ],
    );
    check(
        &["s_mul_hi_u32 s2, s0, s1"],
        &[(ab(m, m), &[(S2, m - 1)][..])],
    );
    check(&["s_mul_hi_i32 s2, s0, s1"], &[(ab(m, m), &[(S2, 0)][..])]);
    check(
        &["s_lshl_b32 s2, s0, s1"],
        &[(ab(1, 33), r(2, 1)), (ab(0x8000_0000, 1), r(0, 0))],
    );
    check(
        &["s_lshr_b32 s2, s0, s1"],
        &[(ab(0x8000_0000, 31), r(1, 1))],
    );
    check(
        &["s_ashr_i32 s2, s0, s1"],
        &[(ab(0x8000_0000, 31), r(m, 1))],
    );
    let w = |lo: u32, hi: u32, n: u32| -> Io { Box::leak(Box::new([(S0, lo), (S1, hi), (S4, n)])) };
    let w2 = |lo: u32, hi: u32, scc: u32| -> Io {
        Box::leak(Box::new([(S2, lo), (S3, hi), (SCC, scc)]))
    };
    check(
        &["s_lshl_b64 s[2:3], s[0:1], s4"],
        &[
            (w(0x8000_0000, 0, 1), w2(0, 1, 1)),
            (w(1, 0, 64), w2(1, 0, 1)),
            (w(0, 0, 3), w2(0, 0, 0)),
        ],
    );
    check(
        &["s_lshr_b64 s[2:3], s[0:1], s4"],
        &[
            (w(0, 1, 1), w2(0x8000_0000, 0, 1)),
            (w(0, 1, 32), w2(1, 0, 1)),
        ],
    );
    check(
        &["s_ashr_i64 s[2:3], s[0:1], s4"],
        &[
            (w(0, 0x8000_0000, 63), w2(m, m, 1)),
            (w(0, 0x8000_0000, 32), w2(0x8000_0000, m, 1)),
        ],
    );
    check(
        &["s_bfm_b32 s2, s0, s1"],
        &[
            (ab(4, 8), &[(S2, 0xf00)][..]),
            (ab(0, 8), &[(S2, 0)][..]),
            (ab(32, 8), &[(S2, 0)][..]),
        ],
    );
    // offset in bits 4:0, width in bits 22:16 of the second operand
    check(
        &["s_bfe_u32 s2, s0, s1"],
        &[
            (ab(0xabcd_1234, 8 << 16 | 8), r(0x12, 1)),
            (ab(0xabcd_1234, 8), r(0, 0)),
            (ab(0xabcd_1234, 32 << 16), r(0xabcd_1234, 1)),
        ],
    );
    check(
        &["s_bfe_i32 s2, s0, s1"],
        &[
            (ab(0x80, 8 << 16), r(0xffff_ff80, 1)),
            (ab(0x7f, 8 << 16), r(0x7f, 1)),
        ],
    );
    check(
        &["s_absdiff_i32 s2, s0, s1"],
        &[
            (ab(3, 10), r(7, 1)),
            (ab(10, 3), r(7, 1)),
            (ab(5, 5), r(0, 0)),
        ],
    );
    check(
        &["s_lshl2_add_u32 s2, s0, s1"],
        &[(ab(1, 2), r(6, 0)), (ab(0x8000_0000, 0), r(0, 1))],
    );
    check(
        &["s_pack_ll_b32_b16 s2, s0, s1"],
        &[(ab(0x1234_abcd, 0x5678_ef01), &[(S2, 0xef01_abcd)][..])],
    );
    check(
        &["s_pack_hh_b32_b16 s2, s0, s1"],
        &[(ab(0x1234_abcd, 0x5678_ef01), &[(S2, 0x5678_1234)][..])],
    );
}

#[test]
fn scalar_bit_ops_and_compares() {
    let ab = |a: u32, b: u32| -> Io { Box::leak(Box::new([(S0, a), (S1, b)])) };
    let r = |v: u32, scc: u32| -> Io { Box::leak(Box::new([(S2, v), (SCC, scc)])) };
    let m = 0xffff_ffffu32;
    let nz = |v: u32| u32::from(v != 0);
    let bit = |asm: &str, g: fn(u32, u32) -> u32| {
        let vecs: Vec<(Io, Io)> = [(0xf0f0_f0f0, 0xff00_ff00), (1, 2), (m, m), (0, 5)]
            .iter()
            .map(|&(a, b)| (ab(a, b), r(g(a, b), nz(g(a, b)))))
            .collect();
        check(&[asm], &vecs);
    };
    bit("s_and_b32 s2, s0, s1", |a, b| a & b);
    bit("s_or_b32 s2, s0, s1", |a, b| a | b);
    bit("s_xor_b32 s2, s0, s1", |a, b| a ^ b);
    bit("s_andn2_b32 s2, s0, s1", |a, b| a & !b);
    bit("s_orn2_b32 s2, s0, s1", |a, b| a | !b);
    bit("s_nand_b32 s2, s0, s1", |a, b| !(a & b));
    bit("s_nor_b32 s2, s0, s1", |a, b| !(a | b));
    bit("s_xnor_b32 s2, s0, s1", |a, b| !(a ^ b));
    check(
        &["s_not_b32 s2, s0"],
        &[(ab(0, 0), r(m, 1)), (ab(m, 0), r(0, 0))],
    );
    check(&["s_mov_b32 s2, s0"], &[(ab(7, 0), &[(S2, 7)][..])]);
    check(
        &["s_brev_b32 s2, s0"],
        &[(ab(1, 0), &[(S2, 0x8000_0000)][..])],
    );
    check(
        &["s_bcnt0_i32_b32 s2, s0"],
        &[(ab(0xffff_0000, 0), r(16, 1)), (ab(m, 0), r(0, 0))],
    );
    check(
        &["s_bcnt1_i32_b32 s2, s0"],
        &[(ab(0xffff_0000, 0), r(16, 1)), (ab(0, 0), r(0, 0))],
    );
    check(
        &["s_ff1_i32_b32 s2, s0"],
        &[(ab(8, 0), &[(S2, 3)][..]), (ab(0, 0), &[(S2, m)][..])],
    );
    check(
        &["s_ff0_i32_b32 s2, s0"],
        &[(ab(0xf, 0), &[(S2, 4)][..]), (ab(m, 0), &[(S2, m)][..])],
    );
    check(
        &["s_flbit_i32_b32 s2, s0"],
        &[
            (ab(1, 0), &[(S2, 31)][..]),
            (ab(0, 0), &[(S2, m)][..]),
            (ab(0x8000_0000, 0), &[(S2, 0)][..]),
        ],
    );
    check(
        &["s_flbit_i32 s2, s0"],
        &[
            (ab(m, 0), &[(S2, m)][..]),
            (ab(0xffff_fff0, 0), &[(S2, 28)][..]),
            (ab(1, 0), &[(S2, 31)][..]),
        ],
    );
    check(
        &["s_sext_i32_i8 s2, s0"],
        &[(ab(0x80, 0), &[(S2, 0xffff_ff80)][..])],
    );
    check(
        &["s_sext_i32_i16 s2, s0"],
        &[(ab(0x8000, 0), &[(S2, 0xffff_8000)][..])],
    );
    check(
        &["s_abs_i32 s2, s0"],
        &[
            (ab(-5i32 as u32, 0), r(5, 1)),
            (ab(0, 0), r(0, 0)),
            (ab(0x8000_0000, 0), r(0x8000_0000, 1)),
        ],
    );
    // compares: SCC only
    let c = |asm: &str, want: [(u32, u32, u32); 3]| {
        let vecs: Vec<(Io, Io)> = want
            .iter()
            .map(|&(a, b, s)| {
                (
                    ab(a, b),
                    &*Box::leak(Box::new([(SCC, s)]) as Box<[(Reg, u32); 1]>) as Io,
                )
            })
            .collect();
        check(&[asm], &vecs);
    };
    c("s_cmp_eq_u32 s0, s1", [(1, 1, 1), (1, 2, 0), (m, m, 1)]);
    c("s_cmp_lg_u32 s0, s1", [(1, 1, 0), (1, 2, 1), (m, 0, 1)]);
    c("s_cmp_gt_i32 s0, s1", [(m, 1, 0), (2, 1, 1), (1, 1, 0)]);
    c("s_cmp_lt_u32 s0, s1", [(1, m, 1), (m, 1, 0), (1, 1, 0)]);
    c("s_cmp_ge_i32 s0, s1", [(m, 1, 0), (1, 1, 1), (2, 1, 1)]);
    c("s_cmp_le_u32 s0, s1", [(m, 1, 0), (1, 1, 1), (0, 1, 1)]);
    c(
        "s_bitcmp1_b32 s0, s1",
        [(4, 2, 1), (4, 1, 0), (0x8000_0000, 31, 1)],
    );
    check(
        &["s_cmp_eq_u64 s[0:1], s[2:3]"],
        &[
            (&[(S0, 1), (S1, 2), (S2, 1), (S3, 2)], &[(SCC, 1)][..]),
            (&[(S0, 1), (S1, 2), (S2, 1), (S3, 3)], &[(SCC, 0)][..]),
        ],
    );
    check(
        &["s_cmpk_lt_i32 s0, 0xffff"],
        &[
            (&[(S0, -5i32 as u32)], &[(SCC, 1)][..]),
            (&[(S0, 0)], &[(SCC, 0)][..]),
        ],
    );
    check(
        &["s_cmpk_eq_u32 s0, 0xffff"],
        &[
            (&[(S0, 0xffff)], &[(SCC, 1)][..]),
            (&[(S0, m)], &[(SCC, 0)][..]),
        ],
    );
    // SCC selects
    let sel = |a: u32, b: u32, c: bool| -> Io {
        Box::leak(Box::new([
            (S0, a),
            (S1, b),
            (S2, 9),
            (S6, 1),
            (S7, u32::from(c)),
        ]))
    };
    check(
        &["s_cmp_eq_u32 s6, s7", "s_cselect_b32 s2, s0, s1"],
        &[
            (sel(10, 20, true), &[(S2, 10)][..]),
            (sel(10, 20, false), &[(S2, 20)][..]),
        ],
    );
    check(
        &["s_cmp_eq_u32 s6, s7", "s_cmov_b32 s2, s0"],
        &[
            (sel(10, 20, true), &[(S2, 10)][..]),
            (sel(10, 20, false), &[(S2, 9)][..]),
        ],
    );
    // the 16-bit immediates
    // the constant folds away: only its use is left
    check(
        &["s_movk_i32 s2, 0xfffe", "s_add_u32 s3, s2, s0"],
        &[(&[(S0, 5)], &[(S3, 3)][..])],
    );
    check(
        &["s_addk_i32 s2, 0x7fff"],
        &[
            (&[(S2, 1)], &[(S2, 0x8000), (SCC, 0)][..]),
            (&[(S2, 0x7fff_ffff)], &[(S2, 0x8000_7ffe), (SCC, 1)][..]),
        ],
    );
    check(
        &["s_mulk_i32 s2, 0xfffd"],
        &[(&[(S2, 5)], &[(S2, -15i32 as u32)][..])],
    );
}

#[test]
fn sdwa_selects_and_destination_modes() {
    let o = |v: u32| -> Io { Box::leak(Box::new([(V3, v)])) };
    let ins =
        |a: u32, b: u32, old: u32| -> Io { Box::leak(Box::new([(V0, a), (V1, b), (V3, old)])) };
    // word 1 of v0 plus byte 2 of v1, into byte 1 of v3 (the rest of v3 kept)
    check(
        &["v_add_nc_u32_sdwa v3, v0, v1 dst_sel:BYTE_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:BYTE_2"],
        &[(ins(0x0003_0000, 0x0005_0000, 0xaabb_ccdd), o(0xaabb_08dd))],
    );
    check(
        &["v_cvt_f32_u32_sdwa v3, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1"],
        &[(ins(0x0000_1200, 0, 0), o(f(18.0)))],
    );
    check(
        &["v_add_nc_u32_sdwa v3, sext(v0), v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD"],
        &[(ins(0xff, 5, 0), o(4)), (ins(0x7f, 5, 0), o(0x84))],
    );
    check(
        &["v_add_nc_u32_sdwa v3, v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_SEXT src0_sel:DWORD src1_sel:DWORD"],
        &[(ins(0x7fff, 1, 0x1234_5678), o(0xffff_8000)), (ins(1, 1, 0x1234_5678), o(2))],
    );
    check(
        &["v_add_nc_u32_sdwa v3, v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD"],
        &[(ins(0x7fff, 1, 0x1234_5678), o(0x8000))],
    );
    check(
        &["v_cmp_eq_u32_sdwa vcc_lo, v0, v1 src0_sel:BYTE_0 src1_sel:BYTE_1"],
        &[
            (ins(0x1234, 0x3400, 0), &[(VCC, 1)][..]),
            (ins(0x1234, 0x3500, 0), &[(VCC, 0)][..]),
        ],
    );
    check(
        &["v_add_f32_sdwa v3, |v0|, -v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD"],
        &[(ins(f(-2.0), f(-3.0), 0), o(f(5.0)))],
    );
}

#[test]
fn sixteen_bit_integer_ops() {
    let ab = |a: u32, b: u32| -> Io { Box::leak(Box::new([(V0, a), (V1, b)])) };
    let o = |v: u32| -> Io { Box::leak(Box::new([(V3, v)])) };
    check(
        &["v_lshrrev_b16 v3, v0, v1"],
        &[(ab(4, 0xabcd_1234), o(0x123))],
    );
    check(
        &["v_lshlrev_b16 v3, v0, v1"],
        &[(ab(4, 0xabcd_1234), o(0x2340))],
    );
    check(&["v_ashrrev_i16 v3, v0, v1"], &[(ab(4, 0x8000), o(0xf800))]);
    check(
        &["v_cmp_lt_i16 vcc_lo, v0, v1"],
        &[
            (ab(0x8000, 1), &[(VCC, 1)][..]),
            (ab(1, 0x8000), &[(VCC, 0)][..]),
        ],
    );
    check(
        &["v_cmp_eq_u16 vcc_lo, v0, v1"],
        &[
            (ab(0x1_0005, 5), &[(VCC, 1)][..]),
            (ab(6, 5), &[(VCC, 0)][..]),
        ],
    );
}

#[test]
fn bit_set_and_clear_read_the_destination() {
    let r = |a: u32, old: u32| -> Io { Box::leak(Box::new([(S0, a), (S2, old)])) };
    check(
        &["s_bitset0_b32 s2, s0"],
        &[
            (r(4, 0xffff_ffff), &[(S2, 0xffff_ffef)][..]),
            (r(36, 0xff), &[(S2, 0xef)][..]),
        ],
    );
    check(
        &["s_bitset1_b32 s2, s0"],
        &[(r(31, 1), &[(S2, 0x8000_0001)][..])],
    );
}

#[test]
fn hardware_register_writes() {
    // the f32 round / denormal bits of MODE are not modelled: refused. Other fields and
    // FLAT_SCRATCH (hidden state) are not ALU state.
    let ok = |asm: &str| {
        let fun = func(&[asm]);
        module_of(&fun).unwrap();
    };
    ok("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3");
    ok("s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_LO), 0");
    let fun = func(&["s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 2), 3"]);
    let s = structure(&fun).unwrap();
    assert!(matches!(
        translate(&fun, &s),
        Err(SpirvError::Unsupported(Unsupported { chunk: 6, .. }))
    ));
}

#[test]
fn mask_ops_in_wave32_and_wave64() {
    let ab = |a: u32, b: u32| -> Io { Box::leak(Box::new([(V0, a), (V1, b)])) };
    let f1 = f(1.0);
    let f2 = f(2.0);
    // v_cmp then s_and_saveexec: s2 gets the old EXEC, EXEC the compare
    check(
        &[
            "v_cmp_lt_f32 vcc_lo, v0, v1",
            "s_and_saveexec_b32 s2, vcc_lo",
        ],
        &[
            (ab(f1, f2), &[(S2, 1), (EXEC, 1), (SCC, 1)][..]),
            (ab(f2, f1), &[(S2, 1), (EXEC, 0), (SCC, 0)][..]),
        ],
    );
    // a wave64 mask is two dwords holding the lane's bit twice
    check(
        &[
            "[w64] v_cmp_lt_f32 vcc, v0, v1",
            "[w64] s_and_saveexec_b64 s[2:3], vcc",
        ],
        &[
            (
                ab(f1, f2),
                &[
                    (VCC, 1),
                    (Reg::VccHi, 1),
                    (S2, 1),
                    (S3, 1),
                    (EXEC, 1),
                    (Reg::ExecHi, 1),
                ][..],
            ),
            (
                ab(f2, f1),
                &[
                    (VCC, 0),
                    (Reg::VccHi, 0),
                    (EXEC, 0),
                    (Reg::ExecHi, 0),
                    (SCC, 0),
                ][..],
            ),
        ],
    );
    // s_andn1_wrexec: exec = sdst = ~src & exec (ISA text); SCC = result != 0
    check(
        &[
            "v_cmp_lt_f32 vcc_lo, v0, v1",
            "s_andn1_wrexec_b32 s2, vcc_lo",
        ],
        &[
            (ab(f1, f2), &[(S2, 0), (EXEC, 0), (SCC, 0)][..]),
            (ab(f2, f1), &[(S2, 1), (EXEC, 1), (SCC, 1)][..]),
        ],
    );
    // s_andn2_wrexec: exec = sdst = src & ~exec
    check(
        &[
            "v_cmp_lt_f32 vcc_lo, v0, v1",
            "s_and_saveexec_b32 s3, vcc_lo",
            "s_andn2_wrexec_b32 s2, s3",
        ],
        &[
            (ab(f1, f2), &[(S2, 0), (EXEC, 0), (SCC, 0)][..]),
            (ab(f2, f1), &[(S2, 1), (EXEC, 1), (SCC, 1)][..]),
        ],
    );
    check(
        &[
            "[w64] v_cmp_lt_f32 vcc, v0, v1",
            "[w64] s_andn1_wrexec_b64 s[2:3], vcc",
        ],
        &[(
            ab(f2, f1),
            &[(S2, 1), (S3, 1), (EXEC, 1), (Reg::ExecHi, 1), (SCC, 1)][..],
        )],
    );
    // a scalar beside a mask: the lane's bit is bit 0 (the probe's lane), in every dword
    let bits = |s0: u32, s1: u32| -> Io { Box::leak(Box::new([(S0, s0), (S1, s1)])) };
    check(
        &["s_and_b32 exec_lo, exec_lo, s0"],
        &[
            (bits(0b101, 0), &[(EXEC, 1), (SCC, 1)][..]),
            (bits(0b110, 0), &[(EXEC, 0), (SCC, 0)][..]),
        ],
    );
    check(
        &["[w64] s_and_b64 exec, exec, s[0:1]"],
        &[
            (bits(1, 2), &[(EXEC, 1), (Reg::ExecHi, 1)][..]),
            (bits(2, 1), &[(EXEC, 0), (Reg::ExecHi, 0)][..]),
        ],
    );
    let cmp = "[w64] v_cmp_lt_f32 vcc, v0, v1";
    check(
        &[cmp, "[w64] s_and_b64 s[2:3], exec, vcc"],
        &[
            (ab(f1, f2), &[(S2, 1), (S3, 1), (SCC, 1)][..]),
            (ab(f2, f1), &[(S2, 0), (S3, 0), (SCC, 0)][..]),
        ],
    );
    check(
        &[cmp, "[w64] s_andn2_b64 s[2:3], exec, vcc"],
        &[
            (ab(f1, f2), &[(S2, 0), (SCC, 0)][..]),
            (ab(f2, f1), &[(S2, 1), (SCC, 1)][..]),
        ],
    );
    check(
        &[cmp, "[w64] s_not_b64 s[2:3], vcc"],
        &[(ab(f1, f2), &[(S2, 0)][..]), (ab(f2, f1), &[(S2, 1)][..])],
    );
    check(
        &[cmp, "[w64] s_mov_b64 s[2:3], vcc"],
        &[(ab(f1, f2), &[(S2, 1), (S3, 1)][..])],
    );
    check(
        &[cmp, "[w64] v_cndmask_b32 v3, v0, v1, vcc"],
        &[(ab(f1, f2), &[(V3, f2)][..]), (ab(f2, f1), &[(V3, f2)][..])],
    );
    check(
        &[
            cmp,
            "[w64] s_cmp_eq_u32 s6, s7",
            "[w64] s_cselect_b64 s[2:3], vcc, exec",
        ],
        &[(
            &[(V0, f1), (V1, f2), (S6, 1), (S7, 1)],
            &[(S2, 1), (S3, 1)][..],
        )],
    );
    check(
        &["[w64] v_add_co_u32 v3, s[2:3], v0, v1"],
        &[
            (ab(m(), 1), &[(V3, 0), (S2, 1), (S3, 1)][..]),
            (ab(1, 1), &[(V3, 2), (S2, 0)][..]),
        ],
    );
    check(
        &[cmp, "[w64] v_add_co_ci_u32 v3, vcc, v0, v1, vcc"],
        &[(ab(f1, f2), &[(V3, f1.wrapping_add(f2).wrapping_add(1))][..])],
    );
    check(
        &["[w64] v_cmpx_lt_f32 v0, v1"],
        &[
            (ab(f1, f2), &[(EXEC, 1), (Reg::ExecHi, 1)][..]),
            (ab(f2, f1), &[(EXEC, 0)][..]),
        ],
    );
    // 64-bit scalar values that are data, not masks
    let w = |lo: u32, hi: u32| -> Io { Box::leak(Box::new([(S0, lo), (S1, hi)])) };
    check(
        &["[w64] s_brev_b64 s[2:3], s[0:1]"],
        &[(w(1, 0), &[(S2, 0), (S3, 0x8000_0000)][..])],
    );
    check(
        &["[w64] s_bcnt1_i32_b64 s2, s[0:1]"],
        &[
            (w(0xff, 0xf), &[(S2, 12), (SCC, 1)][..]),
            (w(0, 0), &[(S2, 0), (SCC, 0)][..]),
        ],
    );
    check(
        &["[w64] s_ff1_i32_b64 s2, s[0:1]"],
        &[
            (w(0, 8), &[(S2, 35)][..]),
            (w(4, 8), &[(S2, 2)][..]),
            (w(0, 0), &[(S2, m())][..]),
        ],
    );
    check(
        &["[w64] s_flbit_i32_b64 s2, s[0:1]"],
        &[
            (w(1, 0), &[(S2, 63)][..]),
            (w(0, 1), &[(S2, 31)][..]),
            (w(0, 0), &[(S2, m())][..]),
        ],
    );
}

fn m() -> u32 {
    u32::MAX
}

// ---------------------------------------------------------------------------------------
// Structured flow

// v_cmp_lt_f32 vcc, v0, v1 ; if (lane) v2 = 1.0 ; v3 = v2 + v2   (wave64)
const IF: [u32; 7] = [
    0x7c020300, 0xbe82246a, 0xbf880001, 0x7e0402f2, 0xbefe0402, 0x06060502, 0xbf810000,
];

// the same with an else that sets v2 = 0
const IF_ELSE: [u32; 11] = [
    0x7c020300, 0xbe82246a, 0x8982027e, 0xbf880001, 0x7e0402f2, 0xbe822702, 0xbf880001, 0x7e040280,
    0x88fe027e, 0x06060502, 0xbf810000,
];

const NESTED: [u32; 13] = [
    0x7c020300, 0xbe82246a, 0xbf880007, 0x7e0402f2, 0x7c080302, 0xbe84246a, 0xbf880001, 0x7e0602f4,
    0x88fe047e, 0x06040503, 0x88fe027e, 0x06060502, 0xbf810000,
];

// s0 = 0 ; LH: s0 += 1 ; if s0 == s8 goto LCONT ; if s0 == s9 goto LEXIT ; v0 += v1 ;
// LCONT: if s0 < 8 goto LH ; LEXIT: v1 = v0 ; s_endpgm
const LOOP: [u32; 11] = [
    0xbe800380, 0x80008100, 0xbf060800, 0xbf850003, 0xbf060900, 0xbf850003, 0x06000300, 0xbf0a8800,
    0xbf85fff8, 0x7e020300, 0xbf810000,
];

// an inner loop exit that leaves two levels
const BREAK2: [u32; 16] = [
    0xbf068009, 0xbf85000c, 0xbe800380, 0x80008100, 0xbe810380, 0x80018101, 0xbf060801, 0xbf850006,
    0x06000300, 0xbf0a8401, 0xbf85fffa, 0x10020301, 0xbf0a8400, 0xbf85fff5, 0x7e040300, 0xbf810000,
];

fn run_words(w: &[u32], ins: &[(Reg, u32)]) -> Out {
    let fun = words(w);
    let s = structure(&fun).unwrap();
    let p = translate(&fun, &s).unwrap_or_else(|e| panic!("{e}"));
    validate(&p.words);
    let out = interp(&fun, &s, ins).unwrap();
    on_gpu(&runner(&p), &p, &["flow"], ins, &out);
    out
}

#[test]
fn if_patterns_translate_and_run() {
    let (f1, f2) = (f(1.0), f(2.0));
    // v0 < v1: v2 = 1.0, v3 = 2.0; else v2 keeps its entry value (0 here)
    let o = run_words(&IF, &[(V0, f1), (V1, f2), (V2, f(5.0))]);
    assert_eq!((get(&o, V2), get(&o, V3)), (f1, f2));
    let o = run_words(&IF, &[(V0, f2), (V1, f1), (V2, f(5.0))]);
    assert_eq!((get(&o, V2), get(&o, V3)), (f(5.0), f(10.0)));
    let o = run_words(&IF_ELSE, &[(V0, f1), (V1, f2), (V2, f(5.0))]);
    assert_eq!((get(&o, V2), get(&o, V3)), (f1, f2));
    let o = run_words(&IF_ELSE, &[(V0, f2), (V1, f1), (V2, f(5.0))]);
    assert_eq!((get(&o, V2), get(&o, V3)), (0, 0));
    let o = run_words(&NESTED, &[(V0, f1), (V1, f2)]);
    assert_eq!((get(&o, V2), get(&o, V3)), (f1, f2));
}

// v_cmp_lt_f32 ; s_and_saveexec ; s_cbranch_execz LJ ; s_add_u32 s9, s9, 1 ; LJ: restore EXEC ;
// s_add_u32 s10, s9, 1 ; v_mov_b32 v2, s10: a scalar defined in a region lanes skip (wave64)
const SPECULATED: [u32; 8] = [
    0x7c020300, 0xbe82246a, 0xbf880001, 0x80098109, 0x88fe027e, 0x800a8109, 0x7e04020a, 0xbf810000,
];

#[test]
fn speculated_scalar_region_runs_when_any_lane_enters() {
    let (f1, f2) = (f(1.0), f(2.0));
    let fun = words(&SPECULATED);
    let s = structure(&fun).unwrap();
    assert!(s
        .branches
        .iter()
        .flatten()
        .any(|b| matches!(b, BranchKind::Wave(_))));
    let p = translate(&fun, &s).unwrap();
    // the vote needs its capability
    assert!(p
        .words
        .windows(2)
        .any(|w| w[0] & 0xffff == 17 && w[1] == 62));
    // the lane enters: s9 + 1 is read after the join
    let o = run_words(&SPECULATED, &[(V0, f1), (V1, f2), (Reg::S(9), 10)]);
    assert_eq!(get(&o, V2), 12);
    // no lane enters: the old s9
    let o = run_words(&SPECULATED, &[(V0, f2), (V1, f1), (Reg::S(9), 10)]);
    assert_eq!(get(&o, V2), 11);
}

#[test]
fn loops_with_continue_break_and_deep_exit() {
    // s8 = 3: iteration 3 skips the add, so v0 gains 7 * v1 over the 8 iterations
    let ins = |s9: u32| [(V0, f(1.0)), (V1, f(2.0)), (Reg::S(8), 3), (Reg::S(9), s9)];
    let o = run_words(&LOOP, &ins(100));
    assert_eq!(get(&o, V1), f(1.0 + 7.0 * 2.0));
    // s9 = 5: leaves at iteration 5, after the adds of iterations 1, 2 and 4
    let o = run_words(&LOOP, &ins(5));
    assert_eq!(get(&o, V1), f(1.0 + 3.0 * 2.0));
    // the two-level exit translates and validates
    let fun = words(&BREAK2);
    module_of(&fun).unwrap();
}

// ---------------------------------------------------------------------------------------
// Chunk 7b: uniformity, spill slots, waterfall, console lane index (the wave evaluator)

/// The wave evaluator: the program on each console lane of a wave. Lanes meet nowhere yet (the
/// subgroup operations of chunk 7c are not emitted), so the host's subgroup width cannot change a
/// result and is not a parameter; chunk 7c adds it with the first op that uses it.
fn wave(f: &Func, s: &Structured, lanes: u32, ins: impl Fn(u32) -> Vec<(Reg, u32)>) -> Vec<Out> {
    (0..lanes)
        .map(|l| interp_lane(f, s, &ins(l), l).unwrap())
        .collect()
}

fn wave_ok(lines: &[&str]) -> (Func, Structured) {
    let fun = func(lines);
    let s = structure(&fun).unwrap_or_else(|e| panic!("{e}"));
    assert!(s.masks.cross_lane.is_empty(), "{:?}", s.masks.cross_lane);
    (fun, s)
}

fn first_cross_lane(lines: &[&str]) -> String {
    let fun = func(lines);
    let m = crate::shader::structure::type_masks(&fun);
    m.cross_lane
        .first()
        .map(|c| c.name.clone())
        .unwrap_or_default()
}

#[test]
fn spill_slots_are_one_scalar_per_lane_used() {
    let lines = [
        "[w64] v_writelane_b32 v8, s0, 3",
        "[w64] v_writelane_b32 v8, s1, 40",
        "[w64] v_readlane_b32 s2, v8, 40",
        "[w64] v_readlane_b32 s3, v8, 3",
        "[w64] v_readlane_b32 s4, v8, 7",
    ];
    let (fun, s) = wave_ok(&lines);
    assert_eq!(s.masks.slots.iter().copied().collect::<Vec<_>>(), [8]);
    let p = translate(&fun, &s).unwrap();
    validate(&p.words);
    // a slot never written reads 0; every lane sees the same slots
    for o in wave(&fun, &s, 64, |l| vec![(S0, 100 + l), (S1, 200)]) {
        assert_eq!(get(&o, S2), 200);
        assert_eq!(get(&o, S3), o[&(S3, false)]);
        assert_eq!(get(&o, S4), 0);
    }
    let outs = wave(&fun, &s, 4, |l| vec![(S0, 100 + l), (S1, 200)]);
    assert_eq!(
        outs.iter().map(|o| get(o, S3)).collect::<Vec<_>>(),
        [100, 101, 102, 103]
    );
}

#[test]
fn a_vgpr_used_for_more_than_spills_is_not_a_slot() {
    // read as a vector: still a lane exchange
    assert_eq!(
        first_cross_lane(&[
            "[w64] v_writelane_b32 v8, s0, 3",
            "[w64] v_mov_b32 v3, v8",
            "[w64] v_readlane_b32 s3, v8, 3",
        ]),
        "v_writelane_b32"
    );
    // read at a lane that is not a constant
    assert_eq!(
        first_cross_lane(&[
            "[w64] v_writelane_b32 v8, s0, 3",
            "[w64] v_readlane_b32 s4, v8, s5",
        ]),
        "v_writelane_b32"
    );
}

#[test]
fn the_console_lane_index_is_not_the_hosts() {
    let lines = [
        "[w64] v_mbcnt_lo_u32_b32 v9, -1, 5",
        "[w64] v_mbcnt_hi_u32_b32 v9, -1, v9",
    ];
    let (fun, s) = wave_ok(&lines);
    let p = translate(&fun, &s).unwrap();
    validate(&p.words);
    // lo = min(lane, 32) + x, hi adds max(lane, 32) - 32: together lane + x
    for (l, o) in wave(&fun, &s, 64, |_| vec![]).iter().enumerate() {
        assert_eq!(get(o, Reg::V(9)), l as u32 + 5, "lane {l}");
    }
    let (fun, s) = wave_ok(&["[w64] v_mbcnt_lo_u32_b32 v9, -1, 0"]);
    let v: Vec<u32> = wave(&fun, &s, 64, |_| vec![])
        .iter()
        .map(|o| get(o, Reg::V(9)))
        .collect();
    assert_eq!(v[31], 31);
    assert_eq!(v[32], 32);
    assert_eq!(v[63], 32);
    // the count of a real mask is a ballot (7c)
    assert_eq!(
        first_cross_lane(&["[w64] v_mbcnt_lo_u32_b32 v9, s0, 0"]),
        "v_mbcnt_lo_u32_b32"
    );
}

#[test]
fn a_probe_has_no_lane_index_in_a_vertex_or_pixel_program() {
    // the ALU table asks the stage for the lane: none gives the unsupported error by name
    let fun = func(&["[w64] v_mbcnt_lo_u32_b32 v9, -1, 0"]);
    let i = &fun.blocks[0].insts[0];
    let ctx = Ctx {
        inst: i,
        srcs: vec![vec![Opd::V(u32::MAX)], vec![Opd::V(0)]],
        imps: vec![],
        dst_mask: vec![false],
        active: None,
        lane: None,
        wave: Wave::None,
    };
    match exec(&mut Cpu, &ctx) {
        Err(Unsupported { chunk: 7, op }) => assert!(op.contains("lane index"), "{op}"),
        r => panic!("{:?}", r.map(|_| ())),
    }
}

#[test]
fn readfirstlane_and_readlane_of_a_uniform_value_are_copies() {
    let lines = [
        "[w64] v_mov_b32 v9, s0",
        "[w64] v_readfirstlane_b32 s5, v9",
        "[w64] v_readlane_b32 s6, v9, 5",
    ];
    let (fun, s) = wave_ok(&lines);
    for o in wave(&fun, &s, 64, |l| vec![(S0, 77), (V0, l)]) {
        assert_eq!((get(&o, Reg::S(5)), get(&o, S6)), (77, 77));
    }
    // a per-lane value (the thread id) is a lane exchange
    assert_eq!(
        first_cross_lane(&["[w64] v_readfirstlane_b32 s5, v0"]),
        "v_readfirstlane_b32"
    );
    assert_eq!(
        first_cross_lane(&["[w64] v_readlane_b32 s5, v0, 5"]),
        "v_readlane_b32"
    );
}

#[test]
fn a_write_under_a_narrowed_exec_keeps_the_old_lanes_values() {
    // v1 holds a per-lane value on entry; the mov under `v_cmp` + `s_and_saveexec` changes only
    // some lanes, so after the restore v1 is not uniform
    let lines = [
        "[w64] v_cmp_lt_u32 vcc, v0, v1",
        "[w64] s_and_saveexec_b64 s[2:3], vcc",
        "[w64] v_mov_b32 v1, s0",
        "[w64] s_mov_b64 exec, s[2:3]",
        "[w64] v_readfirstlane_b32 s5, v1",
    ];
    assert_eq!(first_cross_lane(&lines), "v_readfirstlane_b32");
    // under the entry EXEC every lane takes the new value
    let (fun, s) = wave_ok(&["[w64] v_mov_b32 v1, s0", "[w64] v_readfirstlane_b32 s5, v1"]);
    for o in wave(&fun, &s, 8, |l| vec![(S0, 9), (V1, l)]) {
        assert_eq!(get(&o, Reg::S(5)), 9);
    }
}

#[test]
fn a_waterfall_readfirstlane_takes_the_lanes_own_value() {
    let lines = [
        "[w64] v_readfirstlane_b32 s5, v2",
        "[w64] v_cmp_eq_u32 vcc, s5, v2",
    ];
    let (fun, s) = wave_ok(&lines);
    assert!(s
        .masks
        .lanes
        .values()
        .any(|u| *u == crate::shader::structure::LaneUse::Waterfall));
    let p = translate(&fun, &s).unwrap();
    validate(&p.words);
    // every lane's compare is true, so the loop body runs once per lane with its own index
    for (l, o) in wave(&fun, &s, 64, |l| vec![(V2, 1000 + l / 3)])
        .iter()
        .enumerate()
    {
        assert_eq!(get(o, Reg::S(5)), 1000 + l as u32 / 3);
        assert_eq!(get(o, VCC), 1);
    }
    // the same readfirstlane without the compare is a lane exchange
    assert_eq!(
        first_cross_lane(&["[w64] v_readfirstlane_b32 s5, v2"]),
        "v_readfirstlane_b32"
    );
}

#[test]
fn a_load_from_a_uniform_address_is_uniform_until_a_store() {
    let m = |lines: &[&str]| {
        let fun = func(lines);
        crate::shader::structure::type_masks(&fun).cross_lane.len()
    };
    let load = [
        "[w64] v_mov_b32 v9, s1",
        "[w64] buffer_load_dword v10, v9, s[8:11], 0 offen",
        "[w64] v_readfirstlane_b32 s5, v10",
    ];
    assert_eq!(m(&load), 0);
    // a per-lane address
    assert_eq!(
        m(&[
            "[w64] buffer_load_dword v10, v2, s[8:11], 0 offen",
            "[w64] v_readfirstlane_b32 s5, v10",
        ]),
        1
    );
    // another lane may have stored to the address first
    assert_eq!(
        m(&[
            "[w64] v_mov_b32 v9, s1",
            "[w64] buffer_store_dword v0, v9, s[8:11], 0 offen",
            "[w64] buffer_load_dword v10, v9, s[8:11], 0 offen",
            "[w64] v_readfirstlane_b32 s5, v10",
        ]),
        1
    );
}

#[test]
fn a_cndmask_on_a_per_lane_mask_is_not_uniform() {
    // vcc compares two per-lane values: the select is per lane
    let lanes = [
        "[w64] v_mov_b32 v9, s0",
        "[w64] v_cmp_lt_u32 vcc, v0, v1",
        "[w64] v_cndmask_b32 v10, v9, v1, vcc",
        "[w64] v_readfirstlane_b32 s5, v10",
    ];
    assert_eq!(first_cross_lane(&lanes), "v_readfirstlane_b32");
}

#[test]
fn non_alu_instructions_are_unsupported_by_chunk() {
    let fun = func(&["s_load_dwordx4 s[0:3], s[4:5], 0x0"]);
    let s = structure(&fun).unwrap();
    match translate(&fun, &s) {
        Err(SpirvError::Unsupported(u)) => {
            assert_eq!((u.chunk, u.op.as_str()), (9, "s_load_dwordx4"))
        }
        r => panic!("{:?}", r.map(|p| p.words.len())),
    }
}

#[test]
fn module_layout_is_declared() {
    let fun = func(&["v_add_f32 v3, v0, v1"]);
    let s = structure(&fun).unwrap();
    let p = translate(&fun, &s).unwrap();
    assert_eq!(p.words[0], 0x0723_0203);
    assert_eq!(p.words[1], 0x0001_0500);
    assert!(p.inputs.contains(&V0) && p.inputs.contains(&V1) && p.inputs.contains(&EXEC));
    assert!(p.outputs.contains(&(V3, false)));
}

#[test]
fn f16_rounding_helpers_agree_with_hardware_values() {
    use super::alu::{f16_to_f32, f32_to_f16_rte};
    for h in 0..=0xffffu32 {
        let x = f16_to_f32(h);
        if f32::from_bits(x).is_nan() {
            assert!(f32::from_bits(f16_to_f32(f32_to_f16_rte(x))).is_nan());
        } else {
            assert_eq!(f32_to_f16_rte(x), h, "{h:#x}");
        }
    }
    // ties go to even
    assert_eq!(f32_to_f16_rte(f(1.0 + 1.0 / 2048.0)), 0x3c00);
    assert_eq!(f32_to_f16_rte(f(1.0 + 3.0 / 2048.0)), 0x3c02);
}

// ---------------------------------------------------------------------------------------
// Stage entry points (chunk 8): synthetic VS and PS blobs, never game bytes

pub(crate) mod stage {
    use super::super::gpu::StageImage;
    use super::*;
    use crate::shader::container::tests::{container, dxil, sig};
    use crate::shader::container::XDAT_NATIVE;
    use crate::shader::io::{self, Link};
    use crate::shader::native::tests::{payload, xbch};
    use crate::shader::resource::{SamplerDesc, Translated};
    use crate::shader::{parse, ParsedShader};

    const VS_AT: usize = 0x300;
    const PS_AT: usize = 0x700;
    /// Where the record's stage data starts, and a free area of the payload for the code.
    const STAGE: usize = 0x54;
    const CODE_AT: usize = 0x1c00;

    fn put32(p: &mut [u8], at: usize, v: u32) {
        p[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    /// The 24 bytes before the entry point (`gfx10::ENTRY`), then the program.
    fn prog(lines: &[&str]) -> Vec<u8> {
        let mut c = vec![0; 24];
        c.extend(code(lines));
        c
    }

    /// A one-stage blob around `code`; `tweak` edits the payload (stage data at `e`).
    fn blob(
        vs: bool,
        code: &[u8],
        isg: Vec<u8>,
        osg: Vec<u8>,
        tweak: impl FnOnce(&mut [u8], usize),
    ) -> ParsedShader {
        parse(&blob_bytes(vs, code, isg, osg, tweak)).unwrap()
    }

    /// The bytes of [`blob`].
    fn blob_bytes(
        vs: bool,
        code: &[u8],
        isg: Vec<u8>,
        osg: Vec<u8>,
        tweak: impl FnOnce(&mut [u8], usize),
    ) -> Vec<u8> {
        let mut p = payload(vs, !vs);
        let at = if vs { VS_AT } else { PS_AT };
        p[CODE_AT..CODE_AT + code.len()].copy_from_slice(code);
        p[at..at + 8].copy_from_slice(&(CODE_AT as u64).to_le_bytes());
        p[at + 8..at + 16].copy_from_slice(&(code.len() as u64).to_le_bytes());
        tweak(&mut p, at + STAGE);
        let entries = [
            (0x12, vec![0; 16]),
            (XDAT_NATIVE, xbch(&p)),
            (0x07, vec![0; 28]),
        ];
        container(&[
            (b"SFI0", vec![0; 8]),
            (b"ISG1", isg),
            (b"OSG1", osg),
            (b"PSV0", vec![0; 20]),
            (b"DXIL", dxil(u16::from(vs), &entries)),
            (b"PRIV", vec![0; 12]),
        ])
    }

    // ---- Blobs for the backend's real draws (chunk 10a) ----

    /// A vertex program with no vertex input that pulls vertex `v0`'s position (a `float4`) from
    /// the buffer the root descriptor at parameter 0 (SGPRs s8, s9) names, as vs e5fcb71c pulls
    /// its vertices: a V# built from the user data, `buffer_load_dwordx4` at `16 * v0`.
    pub(crate) fn draw_vs_pulling_from_a_root_buffer() -> Vec<u8> {
        let lines = [
            "s_mov_b32 s0, s8",
            "s_or_b32 s1, s9, 0x10000",
            "s_mov_b32 s2, -1",
            "s_mov_b32 s3, 0x11014fac",
            "v_lshlrev_b32 v1, 4, v0",
            "buffer_load_dwordx4 v[4:7], v1, s[0:3], 0 offen",
            "exp pos0, v4, v5, v6, v7 done",
        ];
        blob_bytes(
            true,
            &prog(&lines),
            sig(&[("POSITION", 0, 0, 1, 0xf)]),
            sigs(),
            |p, e| {
                p[0x221] = 0; // the root descriptor is root parameter 0
                put32(p, e + 0xc4, 0); // no vertex input
                put32(p, e + 0x188, 0); // no parameter output
            },
        )
    }

    /// As [`draw_vs_pulling_from_a_root_buffer`], with two parameter outputs: TEXCOORD0 is the
    /// position mapped to 0..1 (x, y, 0, 1) and COLOR0 is the constant (0, 0, 1, 1).
    pub(crate) fn draw_vs_pulling_with_outputs() -> Vec<u8> {
        let lines = [
            "s_mov_b32 s0, s8",
            "s_or_b32 s1, s9, 0x10000",
            "s_mov_b32 s2, -1",
            "s_mov_b32 s3, 0x11014fac",
            "v_lshlrev_b32 v1, 4, v0",
            "buffer_load_dwordx4 v[4:7], v1, s[0:3], 0 offen",
            "v_mul_f32 v8, 0.5, v4",
            "v_mul_f32 v9, 0.5, v5",
            "v_add_f32 v8, 0.5, v8",
            "v_add_f32 v9, 0.5, v9",
            "v_mov_b32 v10, 1.0",
            "v_mov_b32 v11, 0",
            "exp pos0, v4, v5, v6, v7",
            "exp param0, v8, v9, v11, v10",
            "exp param1, v11, v11, v10, v10 done",
        ];
        blob_bytes(
            true,
            &prog(&lines),
            sig(&[("POSITION", 0, 0, 1, 0xf)]),
            sigs(),
            |p, e| {
                p[0x221] = 0; // the root descriptor is root parameter 0
                put32(p, e + 0xc4, 0); // no vertex input
                for (k, row) in [1u8, 2].into_iter().enumerate() {
                    p[e + 0xc8 + 6 * k..e + 0xc8 + 6 * k + 6]
                        .copy_from_slice(&[6, row, k as u8, 0, 0, 0]);
                }
                put32(p, e + 0x188, 2);
            },
        )
    }

    /// A pixel program that writes (TEXCOORD0.x, TEXCOORD0.y, flat COLOR0.z, 1) to target 0.
    pub(crate) fn draw_ps_reading_the_outputs() -> Vec<u8> {
        let lines = [
            "s_mov_b32 m0, s16", // the parameter offset: right after the four user SGPRs
            "v_interp_p1_f32 v2, v0, attr0.x",
            "v_interp_p2_f32 v2, v1, attr0.x",
            "v_interp_p1_f32 v3, v0, attr0.y",
            "v_interp_p2_f32 v3, v1, attr0.y",
            "v_interp_mov_f32 v4, p0, attr1.z",
            "v_mov_b32 v5, 1.0",
            "exp mrt0, v2, v3, v4, v5 done",
        ];
        blob_bytes(
            false,
            &prog(&lines),
            sigs(),
            sig(&[("SV_Target", 0, 64, 0, 0xf)]),
            |p, e| {
                put32(p, 0x210, 0); // root constants, as in the other draw programs
                put32(p, 0x214, 0);
                put32(p, 0x218, 4);
                p[0x220] = 12;
                p[0x221] = 1;
                let rows = [(1u8, 0u8, 0u8), (2, 1, 1)];
                for (k, (row, attr, flags)) in rows.into_iter().enumerate() {
                    p[e + 0x24 + 8 * k..e + 0x24 + 8 * k + 8]
                        .copy_from_slice(&[6, row, 0xf, attr, 0, 0, 1, flags]);
                }
                put32(p, e + 0x124, 2);
                put32(p, e + 0x134, 0x2);
                put32(p, e + 0x14c, u32::MAX); // no export patch
            },
        )
    }

    /// A vertex program with one input, `POSITION` as a `float4` in v4..v7, exported as is.
    pub(crate) fn draw_vs_with_a_position_input() -> Vec<u8> {
        let lines = [
            "s_swappc_b64 s[4:5], s[4:5]",
            "exp pos0, v4, v5, v6, v7 done",
        ];
        blob_bytes(
            true,
            &prog(&lines),
            sig(&[("POSITION", 0, 0, 1, 0xf)]),
            sigs(),
            |p, e| {
                p[e + 0x24..e + 0x29].copy_from_slice(&[1, 1, 0xf, 4, 3]);
                put32(p, e + 0xc4, 1);
                put32(p, e + 0x188, 0);
            },
        )
    }

    /// A pixel program with no inputs that writes root constants 0..3 (user SGPRs s12..s15, root
    /// parameter 1) to render target 0. (Both stages read the draw's one root block, so their user
    /// SGPRs must not overlap: the vertex programs above use s8 and s9.)
    pub(crate) fn draw_ps_writing_root_constants() -> Vec<u8> {
        let lines = [
            "v_mov_b32 v2, s12",
            "v_mov_b32 v3, s13",
            "v_mov_b32 v4, s14",
            "v_mov_b32 v5, s15",
            "exp mrt0, v2, v3, v4, v5 done",
        ];
        blob_bytes(
            false,
            &prog(&lines),
            sigs(),
            sig(&[("SV_Target", 0, 64, 0, 0xf)]),
            |p, e| {
                put32(p, 0x210, 0); // root constants
                put32(p, 0x214, 0); // from constant 0
                put32(p, 0x218, 4); // four user SGPRs
                p[0x220] = 12;
                p[0x221] = 1; // root parameter 1
                put32(p, e + 0x124, 0); // no inputs
                put32(p, e + 0x134, 0);
                put32(p, e + 0x14c, u32::MAX); // no export patch
            },
        )
    }

    /// The T# from a table in user SGPR 12 (+ 0x20) and the S# as four constants in s16..s19.
    const TABLE_12_TEX_CONST_SAMPLER: [&str; 6] = [
        "s_mov_b32 s13, 0",
        "s_load_dwordx8 s[0:7], s[12:13], 0x20",
        "s_movk_i32 s16, 0x400",
        "s_mov_b32 s17, 0xc00000",
        "s_mov_b32 s18, 0x18f00000",
        "s_mov_b32 s19, 0",
    ];

    /// As above, the S# loaded from the same table (+ 0x40).
    const TABLE_12_TEX_LOADED_SAMPLER: [&str; 3] = [
        "s_mov_b32 s13, 0",
        "s_load_dwordx8 s[0:7], s[12:13], 0x20",
        "s_load_dwordx4 s[16:19], s[12:13], 0x40",
    ];

    const SAMPLE_2D_S16: &str =
        "image_sample v[4:7], v[2:3], s[0:7], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D";

    fn draw_ps_sampling(pre: &[&'static str]) -> Vec<u8> {
        let lines = ps_after(pre, &[SAMPLE_2D_S16]);
        blob_bytes(
            false,
            &prog(&lines),
            sigs(),
            sig(&[("SV_Target", 0, 64, 0, 0xf)]),
            |p, e| {
                put32(p, 0x210, 3); // a descriptor table
                put32(p, 0x218, 1);
                p[0x220] = 12; // in user SGPR 12 (the vertex programs above use s8 and s9)
                p[0x221] = 1; // root parameter 1
                put32(p, e + 0x124, 0);
                put32(p, e + 0x134, 0);
                put32(p, e + 0x14c, u32::MAX);
            },
        )
    }

    /// A pixel program that samples a texture at (1.0, 0.5): its T# is at table + 0x20 and its
    /// S# is a constant (trilinear, 4x anisotropic, wrap), as in ps c1d5a4b8. The table is
    /// root parameter 1, user SGPR 12.
    pub(crate) fn draw_ps_sampling_a_texture() -> Vec<u8> {
        draw_ps_sampling(&TABLE_12_TEX_CONST_SAMPLER)
    }

    /// As [`draw_ps_sampling_a_texture`], the S# a heap sampler at table + 0x40.
    pub(crate) fn draw_ps_sampling_a_heap_sampler() -> Vec<u8> {
        draw_ps_sampling(&TABLE_12_TEX_LOADED_SAMPLER)
    }

    const VS_CODE: [&str; 6] = [
        "s_swappc_b64 s[4:5], s[4:5]",
        "v_cvt_f32_u32 v1, v0",
        "v_add_f32 v8, v4, v1",
        "v_add_f32 v9, s8, v5",
        "exp pos0, v8, v9, v8, v9",
        "exp param0, v4, v5, v4, v5 done",
    ];

    const PS_CODE: [&str; 5] = [
        "s_mov_b32 m0, s10",
        "v_interp_p1_f32 v2, v0, attr0.x",
        "v_interp_p2_f32 v2, v1, attr0.x",
        "v_interp_mov_f32 v3, p0, attr1.y",
        "exp mrt0, v2, v3, v2, v3 done",
    ];

    /// TEXCOORD0 at register 1, COLOR0 at 2, NORMAL0 at 3, FOG0 at 4 (all float4).
    fn sigs() -> Vec<u8> {
        sig(&[
            ("SV_Position", 0, 1, 0, 0xf),
            ("TEXCOORD", 0, 0, 1, 0xf),
            ("COLOR", 0, 0, 2, 0xf),
            ("NORMAL", 0, 0, 3, 0xf),
            ("FOG", 0, 0, 4, 0xf),
        ])
    }

    fn vs_blob() -> ParsedShader {
        blob(
            true,
            &prog(&VS_CODE),
            sig(&[("POSITION", 0, 0, 1, 0xf)]),
            sigs(),
            |p, e| {
                // one input (row 1, v4 and v5); params 0, 1 and 2 are rows 2, 1 and 3
                p[e + 0x24..e + 0x29].copy_from_slice(&[1, 1, 3, 4, 1]);
                for (k, row) in [2u8, 1, 3].into_iter().enumerate() {
                    p[e + 0xc8 + 6 * k..e + 0xc8 + 6 * k + 6]
                        .copy_from_slice(&[6, row, k as u8, 0, 0, 0]);
                }
                put32(p, e + 0x188, 3);
            },
        )
    }

    /// A PS whose inputs are attr 0 (TEXCOORD0, row 1), attr 1 (COLOR0, row 2, flat) and
    /// attr 2 (FOG0, row 4); `spi` enables its input VGPRs.
    fn ps_blob(spi: u32, lines: &[&str]) -> ParsedShader {
        ps_blob_with(spi, lines, |_| {})
    }

    fn ps_blob_with(spi: u32, lines: &[&str], extra: impl FnOnce(&mut [u8])) -> ParsedShader {
        blob(
            false,
            &prog(lines),
            sigs(),
            sig(&[("SV_Target", 0, 64, 0, 0xf)]),
            |p, e| {
                let rows = [(1u8, 0u8, 0u8), (2, 1, 1), (4, 2, 0)];
                for (k, (row, attr, flags)) in rows.into_iter().enumerate() {
                    p[e + 0x24 + 8 * k..e + 0x24 + 8 * k + 8]
                        .copy_from_slice(&[6, row, 0xf, attr, 0, 0, 1, flags]);
                }
                put32(p, e + 0x124, 3);
                put32(p, e + 0x134, spi);
                put32(p, e + 0x14c, u32::MAX); // no export patch: the code is as written
                extra(p);
            },
        )
    }

    /// `(opcode, operands)` of every instruction of a module.
    fn insts(w: &[u32]) -> Vec<(u32, Vec<u32>)> {
        let mut at = 5;
        let mut v = Vec::new();
        while at < w.len() {
            let n = (w[at] >> 16) as usize;
            v.push((w[at] & 0xffff, w[at + 1..at + n].to_vec()));
            at += n;
        }
        v
    }

    /// The `OpDecorate`s: `(target, decoration, operands)`.
    fn decorations(w: &[u32]) -> Vec<(u32, u32, Vec<u32>)> {
        insts(w)
            .into_iter()
            .filter(|(op, _)| *op == 71)
            .map(|(_, o)| (o[0], o[1], o[2..].to_vec()))
            .collect()
    }

    fn count(w: &[u32], deco: u32, args: &[u32]) -> usize {
        decorations(w)
            .iter()
            .filter(|d| d.1 == deco && d.2 == args)
            .count()
    }

    fn entry_model(w: &[u32]) -> u32 {
        insts(w).iter().find(|i| i.0 == 15).unwrap().1[0]
    }

    fn translate(p: &ParsedShader, link: Option<&Link>) -> Vec<u32> {
        translate_full(p, link).words
    }

    fn translate_full(p: &ParsedShader, link: Option<&Link>) -> Translated {
        let t = io::translate_stage(p, &[], link).unwrap_or_else(|e| panic!("{e}"));
        validate(&t.words);
        t
    }

    const LOCATION: u32 = 30;
    const BUILT_IN: u32 = 11;
    const FLAT: u32 = 14;
    const NO_PERSPECTIVE: u32 = 13;
    const CENTROID: u32 = 16;
    const SAMPLE: u32 = 17;

    #[test]
    fn vertex_entry_reads_attributes_and_exports_position_and_parameters() {
        let w = translate(&vs_blob(), None);
        assert_eq!(entry_model(&w), 0); // Vertex
        assert_eq!(count(&w, BUILT_IN, &[0]), 1); // Position
        assert_eq!(count(&w, BUILT_IN, &[42]), 1); // VertexIndex
        assert_eq!(count(&w, BUILT_IN, &[4424]), 1); // BaseVertex
                                                     // the attribute at location 0, and param 0 at location 0 (its own number, unlinked)
        assert_eq!(count(&w, LOCATION, &[0]), 2);
        // the root block is reached through a push constant (storage class 9)
        assert!(insts(&w)
            .iter()
            .any(|(op, o)| *op == 59 && o.get(2) == Some(&9)));
    }

    #[test]
    fn pixel_entry_raises_interpolation_to_inputs() {
        let w = translate(&ps_blob(0x2, &PS_CODE), None);
        assert_eq!(entry_model(&w), 4); // Fragment
                                        // OriginUpperLeft
        assert!(insts(&w).iter().any(|(op, o)| *op == 16 && o[1] == 7));
        // attr 0 perspective (no decoration), attr 1 flat; the target at location 0
        assert_eq!(count(&w, FLAT, &[]), 1);
        assert_eq!(count(&w, NO_PERSPECTIVE, &[]), 0);
        assert_eq!(count(&w, LOCATION, &[0]), 2);
        assert_eq!(count(&w, LOCATION, &[1]), 1);
    }

    #[test]
    fn barycentric_group_picks_the_decoration() {
        // SPI_PS_INPUT_ADDR bit: 0 persp sample, 1 center, 2 centroid, 4/5/6 linear
        for (spi, deco) in [
            (0x1, vec![(SAMPLE, true)]),
            (0x4, vec![(CENTROID, true)]),
            (0x20, vec![(NO_PERSPECTIVE, true)]),
            (0x40, vec![(NO_PERSPECTIVE, true), (CENTROID, true)]),
            (0x10, vec![(NO_PERSPECTIVE, true), (SAMPLE, true)]),
        ] {
            let w = translate(&ps_blob(spi, &PS_CODE), None);
            for (d, on) in deco {
                assert_eq!(count(&w, d, &[]) == 1, on, "spi {spi:#x} decoration {d}");
            }
        }
    }

    #[test]
    fn barycentrics_used_other_than_by_interp_are_refused() {
        let p = ps_blob(0x2, &["v_add_f32 v8, v0, v1"]);
        match io::translate_stage(&p, &[], None) {
            Err(io::StageError::Spirv(SpirvError::Unsupported(u))) => assert_eq!(u.chunk, 8),
            r => panic!("{:?}", r.map(|t| t.words.len())),
        }
    }

    #[test]
    fn linking_matches_semantics_not_numbers() {
        let (vs, ps) = (vs_blob(), ps_blob(0x2, &PS_CODE));
        let l = io::link(&vs, &ps);
        // the VS's param 0 is COLOR0 (the PS's attr 1), param 1 TEXCOORD0 (attr 0); param 2
        // (NORMAL0) feeds nothing; the PS's attr 2 (FOG0) has no VS output
        assert_eq!(l.params, [(0, 1), (1, 0)].into_iter().collect());
        assert_eq!(
            l.attrs,
            [(0, Some(0)), (1, Some(1)), (2, None)]
                .into_iter()
                .collect()
        );
        let (v, p) = io::translate_pair(&vs, &ps, &[]).unwrap();
        let (v, p) = (v.words, p.words);
        validate(&v);
        validate(&p);
        // both stages carry the shared locations: the VS exports param 0 (v4, v5) at 1
        assert_eq!(count(&v, LOCATION, &[1]), 1);
        assert_eq!(count(&p, LOCATION, &[1]), 1);
    }

    #[test]
    fn an_unmatched_pixel_input_reads_its_default() {
        let ps = ps_blob(0x2, &PS_CODE);
        // no vertex output has COLOR0: attr 1 is not an input any more
        let l = Link {
            params: Default::default(),
            attrs: [(0, Some(0)), (1, None)].into_iter().collect(),
        };
        let w = translate(&ps, Some(&l));
        assert_eq!(count(&w, FLAT, &[]), 0);
        assert_eq!(count(&w, LOCATION, &[1]), 0);
        // attr 0 and the target
        assert_eq!(count(&w, LOCATION, &[0]), 2);
    }

    #[test]
    fn export_patch_selects_the_raw_32_bit_variant() {
        // the default payload patches dword 9 with six distinctive dwords in variant 9
        let p = blob(false, &[0; 80], sigs(), sigs(), |_, _| {});
        let rec = p.record().unwrap();
        let c = io::stage_code(&p, rec);
        assert_eq!(c[36..40], 0x100u32.to_le_bytes());
        assert_eq!(c[56..60], 0x105u32.to_le_bytes());
        assert_eq!(c[32..36], [0; 4]);
    }

    // -----------------------------------------------------------------------------------
    // Resources (chunk 9a)

    /// A VS around `lines`, which export position from v8 / v9; `tweak` edits the payload.
    fn vs_with(lines: &[&str], tweak: impl FnOnce(&mut [u8])) -> ParsedShader {
        blob(
            true,
            &prog(lines),
            sig(&[("POSITION", 0, 0, 1, 0xf)]),
            sigs(),
            |p, _| tweak(p),
        )
    }

    /// The layout of the synthetic record: one table in user SGPR s8 (instead of the default
    /// root descriptor pair s8, s9).
    fn one_table(p: &mut [u8]) {
        put32(p, 0x210, 3);
        put32(p, 0x218, 1);
    }

    fn ps_tables(lines: &[&str]) -> ParsedShader {
        ps_blob_with(0, lines, one_table)
    }

    /// The `(chunk, op)` a stage is refused with.
    fn refused(p: &ParsedShader) -> (u8, String) {
        match io::translate_stage(p, &[], None) {
            Err(io::StageError::Spirv(SpirvError::Unsupported(u))) => (u.chunk, u.op),
            r => panic!("{:?}", r.map(|t| t.words.len())),
        }
    }

    /// Instructions `op` of a module.
    fn count_op(w: &[u32], op: u32) -> usize {
        insts(w).iter().filter(|i| i.0 == op).count()
    }

    /// `OpLoad`s with `Aligned n`: the 64-bit ones read the root block's addresses.
    fn aligned_loads(w: &[u32], n: u32) -> usize {
        insts(w)
            .iter()
            .filter(|(op, o)| *op == 61 && o.len() == 5 && o[3] == 2 && o[4] == n)
            .count()
    }

    const OP_SAMPLED_IMAGE: u32 = 86;
    const OP_SAMPLE_IMPLICIT: u32 = 87;
    const OP_SAMPLE_EXPLICIT: u32 = 88;
    const OP_STORE: u32 = 62;
    const DESCRIPTOR_SET: u32 = 34;
    const BINDING: u32 = 33;

    const VS_EXPORT: [&str; 3] = [
        "v_mov_b32 v8, 1.0",
        "v_mov_b32 v9, 0.5",
        "exp pos0, v8, v9, v8, v9 done",
    ];

    fn vs_lines(body: &[&'static str]) -> Vec<&'static str> {
        body.iter().chain(VS_EXPORT.iter()).copied().collect()
    }

    #[test]
    fn constant_buffer_from_a_table_is_read_through_the_view_table() {
        let lines = vs_lines(&[
            "s_mov_b32 s9, 0",
            "s_load_dwordx4 s[0:3], s[8:9], 0x0",
            "s_buffer_load_dword s4, s[0:3], 0x10",
            "s_buffer_load_dwordx4 s[4:7], s[0:3], 0x20",
        ]);
        let w = translate(&vs_with(&lines, one_table), None);
        assert_eq!(entry_model(&w), 0);
        // table_base[8] once for the descriptor, view_table once per buffer load
        assert_eq!(aligned_loads(&w, 8), 3);
        // the root block: user_data[32], table_base[32], view_table
        assert!(insts(&w).iter().any(|(op, o)| *op == 30 && o.len() == 4));
    }

    #[test]
    fn dynamic_offsets_of_a_table_and_of_a_buffer() {
        // the table address gets a value computed from memory; the buffer offset is an SGPR
        let lines = vs_lines(&[
            "s_mov_b32 s9, 0",
            "s_load_dwordx4 s[0:3], s[8:9], 0x0",
            "s_buffer_load_dword s4, s[0:3], 0x10",
            "s_lshl_b32 s10, s4, 5",
            "s_add_u32 s8, s8, s10",
            "s_load_dwordx4 s[12:15], s[8:9], 0x40",
            "s_buffer_load_dword s5, s[12:15], 0x10",
            "s_buffer_load_dword s5, s[0:3], s10",
        ]);
        let w = translate(&vs_with(&lines, one_table), None);
        // two table slots, three buffer reads
        assert_eq!(aligned_loads(&w, 8), 2 + 3);
    }

    #[test]
    fn buffer_built_from_a_root_descriptor_is_decoded_at_run_time() {
        // `s0 = s8; s1 = s9 | 0x10000; s2 = -1; s3 = flags`: base from s8 / s9, stride 1
        let lines = vs_lines(&[
            "s_mov_b32 s0, s8",
            "s_or_b32 s1, s9, 0x10000",
            "s_mov_b32 s2, -1",
            "s_mov_b32 s3, 0x11014fac",
            "s_mov_b32 s4, 0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "buffer_load_dwordx4 v[4:7], v1, s[0:3], 0 offen",
            "buffer_load_dword v4, v1, s[0:3], 0 idxen offset:8",
            "buffer_load_dword v4, v[1:2], s[0:3], s4 idxen offen",
            "buffer_store_dword v4, v1, s[0:3], 0 offen",
        ]);
        let w = translate(&vs_with(&lines, |_| {}), None);
        // the root block is only read for the user dwords: no table, no view table
        assert_eq!(aligned_loads(&w, 8), 0);
        // the stores of the guarded loads' result, and the one store the program does
        assert!(count_op(&w, OP_STORE) >= 6);
    }

    #[test]
    fn addresses_read_from_memory_are_flagged() {
        // a root descriptor's address is user data: translated by the backend
        let user = vs_lines(&["s_load_dwordx4 s[0:3], s[8:9], 0x0"]);
        let t = translate_full(&vs_with(&user, |_| {}), None);
        assert!(!t.memory_addresses && !t.images);
        let built = vs_lines(&[
            "s_mov_b32 s0, s8",
            "s_or_b32 s1, s9, 0x10000",
            "s_mov_b32 s2, -1",
            "s_mov_b32 s3, 0x11014fac",
            "v_mov_b32 v1, 0",
            "buffer_load_dwordx4 v[4:7], v1, s[0:3], 0 offen",
        ]);
        assert!(!translate_full(&vs_with(&built, |_| {}), None).memory_addresses);
        // a pointer stored in a buffer reaches the GPU as it was stored
        let chased = vs_lines(&[
            "s_load_dwordx2 s[4:5], s[8:9], 0x0",
            "s_load_dwordx4 s[12:15], s[4:5], 0x0",
        ]);
        assert!(translate_full(&vs_with(&chased, |_| {}), None).memory_addresses);
    }

    #[test]
    fn raw_scalar_load_reads_a_root_descriptor_address() {
        let lines = vs_lines(&["s_load_dwordx4 s[0:3], s[8:9], 0x0"]);
        let w = translate(&vs_with(&lines, |_| {}), None);
        assert_eq!(aligned_loads(&w, 8), 0);
        // four dwords loaded from the address the two user dwords make, and the root block
        assert_eq!(count_op(&w, 120), 4 + 1); // OpConvertUToPtr
    }

    /// The vector registers a sample reads and the exports read.
    const VREGS: [&str; 7] = [
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 1.0",
        "v_mov_b32 v3, 0.5",
        "v_mov_b32 v4, 0",
        "v_mov_b32 v5, 0",
        "v_mov_b32 v6, 0",
        "v_mov_b32 v7, 0",
    ];

    /// The T# from a table slot (s8 + 0x20) and the S# as four constants.
    const TEX_CONST_SAMPLER: [&str; 6] = [
        "s_mov_b32 s9, 0",
        "s_load_dwordx8 s[0:7], s[8:9], 0x20",
        "s_movk_i32 s12, 0x400",
        "s_mov_b32 s13, 0xc00000",
        "s_mov_b32 s14, 0x18f00000",
        "s_mov_b32 s15, 0",
    ];

    const SAMPLE_2D: &str =
        "image_sample v[4:7], v[2:3], s[0:7], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D";

    /// `pre`, the vector registers, `then`, and a render target export.
    fn ps_after(pre: &[&'static str], then: &[&'static str]) -> Vec<&'static str> {
        let mut l = pre.to_vec();
        l.extend(VREGS);
        l.extend(then);
        l.push("exp mrt0, v4, v5, v6, v7 done");
        l
    }

    #[test]
    fn image_sample_with_a_loaded_texture_and_a_constant_sampler() {
        let t = translate_full(
            &ps_tables(&ps_after(&TEX_CONST_SAMPLER, &[SAMPLE_2D])),
            None,
        );
        let w = &t.words;
        assert_eq!(count_op(w, OP_SAMPLE_IMPLICIT), 1);
        assert_eq!(count_op(w, OP_SAMPLED_IMAGE), 1);
        // the sampled images at set 0, binding 0; the static sampler at set 1, binding 0
        assert_eq!(count(w, DESCRIPTOR_SET, &[0]), 1);
        assert_eq!(count(w, DESCRIPTOR_SET, &[1]), 1);
        assert_eq!(count(w, BINDING, &[0]), 2);
        assert_eq!(
            t.samplers,
            [SamplerDesc::from_words([0x400, 0xc0_0000, 0x18f0_0000, 0])]
        );
        assert_eq!(t.samplers[0].max_anisotropy(), 4);
    }

    #[test]
    fn sample_forms_and_dimensions() {
        for (sample, explicit) in [
            (
                "image_sample v[4:5], v[2:3], s[0:7], s[12:15] dmask:0x5 dim:SQ_RSRC_IMG_2D",
                false,
            ),
            (
                "image_sample_l v[4:7], v[2:4], s[0:7], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
                true,
            ),
            (
                "image_sample_lz v[4:7], v[2:3], s[0:7], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
                true,
            ),
            (
                "image_sample_lz v[4:7], v[1:3], s[0:7], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_CUBE",
                true,
            ),
            (
                "image_sample v[4:7], v[1:3], s[0:7], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY",
                false,
            ),
        ] {
            let w = translate(&ps_tables(&ps_after(&TEX_CONST_SAMPLER, &[sample])), None);
            assert_eq!(count_op(&w, OP_SAMPLE_EXPLICIT) == 1, explicit, "{sample}");
            assert_eq!(count_op(&w, OP_SAMPLE_IMPLICIT) == 1, !explicit, "{sample}");
        }
    }

    #[test]
    fn a_loaded_sampler_indexes_the_sampler_array() {
        let lines = ps_after(
            &[
                "s_mov_b32 s9, 0",
                "s_load_dwordx8 s[0:7], s[8:9], 0x20",
                "s_load_dwordx4 s[12:15], s[8:9], 0x40",
            ],
            &[SAMPLE_2D],
        );
        let t = translate_full(&ps_tables(&lines), None);
        // no static sampler; the array is at set 0, binding 2
        assert!(t.samplers.is_empty());
        assert_eq!(count(&t.words, BINDING, &[2]), 1);
        assert_eq!(count(&t.words, DESCRIPTOR_SET, &[1]), 0);
    }

    #[test]
    fn a_texture_that_is_not_loaded_is_refused() {
        // with the default layout s8 / s9 is a root descriptor: the T# is raw memory
        let mut pre = TEX_CONST_SAMPLER.to_vec();
        pre[0] = "s_mov_b32 s9, 0";
        let p = ps_blob_with(0, &ps_after(&pre, &[SAMPLE_2D]), |_| {});
        let (chunk, op) = refused(&p);
        assert_eq!(chunk, 9);
        assert!(op.contains("not loaded from a table"), "{op}");
    }

    #[test]
    fn a_sampler_neither_constant_nor_loaded_is_refused() {
        let mut pre = TEX_CONST_SAMPLER.to_vec();
        pre[2] = "s_mov_b32 s12, s8"; // a word of user data
        let (chunk, op) = refused(&ps_tables(&ps_after(&pre, &[SAMPLE_2D])));
        assert_eq!(chunk, 9);
        assert!(op.contains("neither constant nor loaded"), "{op}");
    }

    #[test]
    fn descriptor_words_used_as_numbers_are_refused() {
        let lines = vs_lines(&[
            "s_mov_b32 s9, 0",
            "s_load_dwordx4 s[0:3], s[8:9], 0x0",
            "s_and_b32 s5, s2, 0xff",
        ]);
        let (chunk, op) = refused(&vs_with(&lines, one_table));
        assert_eq!(chunk, 9);
        assert!(op.contains("used as numbers"), "{op}");
    }

    #[test]
    fn implicit_lod_sampling_is_refused_in_a_vertex_stage() {
        let mut lines = TEX_CONST_SAMPLER.to_vec();
        lines.extend(VREGS);
        lines.push(SAMPLE_2D);
        lines.extend(VS_EXPORT);
        let (chunk, op) = refused(&vs_with(&lines, one_table));
        assert_eq!(chunk, 9);
        assert!(op.contains("vertex stage"), "{op}");
    }

    #[test]
    fn a_pair_shares_one_deduplicated_list_of_static_samplers() {
        let lz = "image_sample_lz v[4:7], v[2:3], s[0:7], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D";
        let vs_with_sampler = |s12: &'static str| {
            let mut l = TEX_CONST_SAMPLER.to_vec();
            l[2] = s12;
            l.extend(VREGS);
            l.push(lz);
            l.extend(VS_EXPORT);
            vs_with(&l, one_table)
        };
        let ps = ps_tables(&ps_after(&TEX_CONST_SAMPLER, &[SAMPLE_2D]));
        // different samplers: the VS's is binding 0, the PS's is renumbered to 1
        let vs = vs_with_sampler("s_movk_i32 s12, 0x500");
        let (v, p) = io::translate_pair(&vs, &ps, &[]).unwrap();
        assert_eq!(v.samplers.len(), 2);
        assert_eq!(p.samplers, v.samplers);
        assert_eq!(
            v.samplers[1],
            SamplerDesc::from_words([0x400, 0xc0_0000, 0x18f0_0000, 0])
        );
        validate(&v.words);
        validate(&p.words);
        assert_eq!(count(&v.words, BINDING, &[0]), 2);
        assert_eq!(count(&p.words, BINDING, &[0]), 1, "only the textures");
        assert_eq!(
            count(&p.words, BINDING, &[1]),
            1,
            "the PS's sampler is renumbered"
        );
        // the same sampler in both: one entry, no renumbering
        let vs = vs_with_sampler("s_movk_i32 s12, 0x400");
        let (v, p) = io::translate_pair(&vs, &ps, &[]).unwrap();
        assert_eq!(v.samplers.len(), 1);
        assert_eq!(p.samplers, v.samplers);
        assert_eq!(count(&p.words, BINDING, &[0]), 2);
    }

    #[test]
    fn chunk_9b_operations_stay_unsupported_by_name() {
        for (line, name) in [
            (
                "buffer_load_format_x v4, v1, s[0:3], 0 offen",
                "buffer_load_format_x",
            ),
            ("global_load_dword v4, v[2:3], off", "global_load_dword"),
            (
                "image_sample_b v[4:7], v[2:4], s[0:7], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
                "image_sample_b",
            ),
            (
                "image_sample v[4:5], v[2:3], s[0:7], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D d16",
                "d16",
            ),
        ] {
            let pre = [
                "s_mov_b32 s9, 0",
                "s_load_dwordx8 s[0:7], s[8:9], 0x20",
                "s_load_dwordx4 s[12:15], s[8:9], 0x40",
            ];
            let lines = ps_after(&pre, &[line]);
            let (chunk, op) = refused(&ps_tables(&lines));
            assert_eq!(chunk, 9, "{line}");
            assert!(op.contains(name), "{line}: {op}");
        }
    }

    #[test]
    fn pixel_discard_demotes_the_lane() {
        // the structure test's discard: v_cmpx ; s_andn2 live, live, exec ; branch to an
        // `exp null done` exit; FragCoord.x / .y stand in for v0 / v1
        let words = [
            0xbe820a7eu32,
            0x7c220300,
            0x8a827e02,
            0xbf840003,
            0xbefe0a02,
            0x7e0402f2,
            0xbf810000,
            0xf8000890,
            0x00000000,
            0xbf810000,
        ];
        let mut bytes = vec![0; 24];
        bytes.extend(words.iter().flat_map(|w| w.to_le_bytes()));
        let p = blob(false, &bytes, sigs(), sigs(), |p, e| {
            put32(p, e + 0x124, 0);
            put32(p, e + 0x134, 0x300);
            put32(p, e + 0x14c, u32::MAX);
        });
        let w = translate(&p, None);
        // OpDemoteToHelperInvocation, and no OpKill
        assert!(insts(&w).iter().any(|(op, _)| *op == 5380));
        assert!(!insts(&w).iter().any(|(op, _)| *op == 252));
    }

    // ---- Compute (chunk 10c): synthetic compute blobs, never game bytes ----

    const CS_AT: usize = 0x1500;

    /// A compute record: `rsrc2`, the group size, the LDS size, the used rows (kind, table
    /// dword, user SGPR) and the user SGPR the one descriptor table sits in (`None`: a root
    /// descriptor pair at s0).
    struct CsSpec {
        rsrc2: u32,
        group: [u32; 3],
        lds: u32,
        flags: u32,
        used: Vec<(u8, u32, u8)>,
        table: Option<u8>,
    }

    impl CsSpec {
        fn new(rsrc2: u32, group: [u32; 3]) -> Self {
            Self {
                rsrc2,
                group,
                lds: 0,
                flags: 0xa0000,
                used: Vec::new(),
                table: None,
            }
        }
    }

    fn cs_blob(lines: &[&str], c: &CsSpec) -> ParsedShader {
        let code = prog(lines);
        let mut p = crate::shader::native::tests::payload_all(false, false, true, false);
        p[CODE_AT..CODE_AT + code.len()].copy_from_slice(&code);
        p[CS_AT..CS_AT + 8].copy_from_slice(&(CODE_AT as u64).to_le_bytes());
        p[CS_AT + 8..CS_AT + 16].copy_from_slice(&(code.len() as u64).to_le_bytes());
        put32(&mut p, CS_AT + 0x14, c.flags);
        put32(&mut p, CS_AT + 0x4c, c.lds);
        let e = CS_AT + STAGE;
        put32(&mut p, e + 0x24, c.rsrc2);
        for (k, g) in c.group.iter().enumerate() {
            put32(&mut p, e + 0x28 + 4 * k, *g);
        }
        match c.table {
            // a table in user SGPR s (slot 1)
            Some(s) => {
                put32(&mut p, 0x210, 3);
                put32(&mut p, 0x218, 1);
                p[0x220] = s;
                p[0x221] = 1;
            }
            // the root descriptor pair at s0
            None => p[0x220] = 0,
        }
        put32(&mut p, CS_AT + 0x3c, c.used.len() as u32);
        for (k, &(kind, dword, sgpr)) in c.used.iter().enumerate() {
            let at = 0x230 + 20 * k;
            p[at..at + 20].fill(0);
            p[at] = kind;
            p[at + 1] = 3;
            put32(&mut p, at + 4, dword);
            p[at + 16] = 8;
            p[at + 17] = sgpr;
            p[at + 18] = 1;
        }
        let entries = [
            (0x12, vec![0; 16]),
            (XDAT_NATIVE, xbch(&p)),
            (0x07, vec![0; 28]),
        ];
        parse(&container(&[
            (b"SFI0", vec![0; 8]),
            (b"ISG1", sig(&[])),
            (b"OSG1", sig(&[])),
            (b"PSV0", vec![0; 20]),
            (b"DXIL", dxil(5, &entries)),
            (b"PRIV", vec![0; 12]),
        ]))
        .unwrap()
    }

    /// The buffer descriptor a program writes through: s[12:15] built from the root descriptor.
    const CS_BUFFER: [&str; 4] = [
        "s_mov_b32 s12, s0",
        "s_or_b32 s13, s1, 0x10000",
        "s_mov_b32 s14, -1",
        "s_mov_b32 s15, 0x11014fac",
    ];

    /// 8x4 groups with x and y group ids in s4, s5 and two thread-id VGPRs: stores
    /// `x | y << 8` at index `y * 16 + x` of the root buffer.
    fn cs_ids() -> ParsedShader {
        let mut lines: Vec<&str> = CS_BUFFER.to_vec();
        lines.extend([
            "v_lshl_add_u32 v2, s4, 3, v0",
            "v_lshl_add_u32 v3, s5, 2, v1",
            "v_lshl_add_u32 v5, v3, 4, v2",
            "v_lshlrev_b32 v5, 2, v5",
            "v_lshl_or_b32 v6, v3, 8, v2",
            "buffer_store_dword v6, v5, s[12:15], 0 offen",
        ]);
        cs_blob(
            &lines,
            &CsSpec::new(4 << 1 | 1 << 7 | 1 << 8 | 1 << 11, [8, 4, 1]),
        )
    }

    /// The workgroup's lanes write `100 + lane` to LDS, meet at a barrier and read the mirror
    /// lane's: `out[lane] = 100 + 63 - lane`.
    fn cs_lds_mirror() -> ParsedShader {
        let mut lines: Vec<&str> = CS_BUFFER.to_vec();
        lines.extend([
            "v_add_nc_u32 v2, 100, v0",
            "v_sub_nc_u32 v3, 63, v0",
            "v_lshlrev_b32 v1, 2, v0",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "s_barrier",
            "v_lshlrev_b32 v3, 2, v3",
            "ds_read_b32 v4, v3",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v4, v1, s[12:15], 0 offen",
        ]);
        let mut c = CsSpec::new(4 << 1, [64, 1, 1]);
        c.lds = 256;
        cs_blob(&lines, &c)
    }

    /// 4x4 threads: load texel (1, 1) of the 2D SRV at table dword 0 (an `a16` address) and
    /// store it, as four channels, at (x, y, 0) of the 3D UAV at table dword 8. The table is
    /// in s5.
    fn cs_image_copy(lines_after: &[&'static str]) -> ParsedShader {
        let mut lines = vec![
            "s_mov_b32 s2, s5",
            "s_mov_b32 s3, 0",
            "s_load_dwordx8 s[16:23], s[2:3], 0x20",
            "s_load_dwordx8 s[24:31], s[2:3], null",
            "v_mov_b32 v5, 0x10001",
            "s_waitcnt lgkmcnt(0)",
            "image_load v[8:11], v5, s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D unorm a16",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32 v6, 0",
            "image_store v[8:11], [v0, v1, v6], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_3D unorm",
        ];
        lines.extend(lines_after);
        let mut c = CsSpec::new(6 << 1 | 1 << 11, [4, 4, 1]);
        c.table = Some(5);
        c.used = vec![(0, 0, 5), (3, 8, 5)];
        cs_blob(&lines, &c)
    }

    fn refused_cs(p: &ParsedShader) -> (u8, String) {
        match io::translate_stage(p, &[], None) {
            Err(io::StageError::Spirv(SpirvError::Unsupported(u))) => (u.chunk, u.op),
            r => panic!("{:?}", r.map(|t| t.words.len())),
        }
    }

    const OP_IMAGE_FETCH: u32 = 95;
    const OP_IMAGE_READ: u32 = 98;
    const OP_IMAGE_WRITE: u32 = 99;
    const OP_CONTROL_BARRIER: u32 = 224;

    #[test]
    fn compute_entry_has_a_local_size_and_the_group_and_thread_ids() {
        let t = translate_full(&cs_ids(), None);
        let w = &t.words;
        assert_eq!(entry_model(w), 5); // GLCompute
                                       // OpExecutionMode LocalSize 8 4 1
        assert!(insts(w)
            .iter()
            .any(|(op, o)| *op == 16 && o[1] == 17 && o[2..] == [8, 4, 1]));
        assert_eq!(count(w, BUILT_IN, &[26]), 1); // WorkgroupId
        assert_eq!(count(w, BUILT_IN, &[27]), 1); // LocalInvocationId
                                                  // the root block comes from the push constant, as for the other stages
        assert!(insts(w)
            .iter()
            .any(|(op, o)| *op == 59 && o.get(2) == Some(&9)));
        assert!(!t.images && !t.storage_images);
    }

    #[test]
    fn image_loads_and_stores_use_the_slot_kind() {
        let t = translate_full(&cs_image_copy(&[]), None);
        let w = &t.words;
        assert!(t.images && t.storage_images);
        assert_eq!(count_op(w, OP_IMAGE_FETCH), 1); // the SRV
        assert_eq!(count_op(w, OP_IMAGE_WRITE), 1); // the UAV
        assert_eq!(count_op(w, OP_IMAGE_READ), 0);
        // the sampled array is binding 0, the storage array binding 1, both in set 0
        assert_eq!(count(w, BINDING, &[0]), 1);
        assert_eq!(count(w, BINDING, &[1]), 1);
        assert_eq!(count(w, DESCRIPTOR_SET, &[0]), 2);
        // StorageImageReadWithoutFormat and WriteWithoutFormat
        let caps: Vec<u32> = insts(w)
            .iter()
            .filter(|i| i.0 == 17)
            .map(|i| i.1[0])
            .collect();
        assert!(caps.contains(&55) && caps.contains(&56));
    }

    #[test]
    fn a_uav_is_read_and_a_srv_fetched_at_a_mip() {
        let t = translate_full(
            &cs_image_copy(&[
                "v_mov_b32 v7, 0",
                "image_load v[8:11], [v0, v1, v6], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_3D unorm",
                "image_load_mip v[8:11], [v0, v1, v6, v7], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY unorm",
            ]),
            None,
        );
        assert_eq!(count_op(&t.words, OP_IMAGE_READ), 1);
        assert_eq!(count_op(&t.words, OP_IMAGE_FETCH), 2);
    }

    #[test]
    fn unsupported_image_forms_are_refused_by_name() {
        // a store of two channels
        let (chunk, op) = refused_cs(&cs_image_copy(&[
            "image_store v[8:9], [v0, v1, v6], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_3D unorm",
        ]));
        assert_eq!(chunk, 9);
        assert!(op.contains("dmask"), "{op}");
        // a store to a slot the used rows call an SRV
        let mut lines = vec![
            "s_mov_b32 s2, s5",
            "s_mov_b32 s3, 0",
            "s_load_dwordx8 s[24:31], s[2:3], null",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32 v6, 0",
            "v_mov_b32 v8, 0",
            "v_mov_b32 v9, 0",
            "v_mov_b32 v10, 0",
            "v_mov_b32 v11, 0",
            "image_store v[8:11], [v0, v1, v6], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_3D unorm",
        ];
        let mut c = CsSpec::new(6 << 1 | 1 << 11, [4, 4, 1]);
        c.table = Some(5);
        c.used = vec![(0, 0, 5), (3, 8, 5)];
        let (_, op) = refused_cs(&cs_blob(&lines, &c));
        assert!(op.contains("image_store") && op.contains("kind 0"), "{op}");
        // a slot no used row names
        c.used = vec![(3, 8, 5)];
        lines[9] = "image_load v[8:11], v[0:1], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D unorm";
        let (_, op) = refused_cs(&cs_blob(&lines, &c));
        assert!(op.contains("no used-resource row"), "{op}");
    }

    #[test]
    fn compute_entries_the_translation_does_not_provide_are_refused() {
        let lines = ["s_mov_b32 s12, s0"];
        let base = 4 << 1 | 1 << 11;
        for (rsrc2, flags, what) in [
            (base | 0x400, 0xa0000, "thread-group info"),
            (base | 1, 0xa0000, "scratch"),
            (base, 0xa0020, "wave32"),
        ] {
            let mut c = CsSpec::new(rsrc2, [8, 8, 1]);
            c.flags = flags;
            let (chunk, op) = refused_cs(&cs_blob(&lines, &c));
            assert_eq!(chunk, 10);
            assert!(op.contains(what), "{op}");
        }
    }

    #[test]
    fn workgroup_memory_and_the_barrier_translate() {
        let t = translate_full(&cs_lds_mirror(), None);
        assert_eq!(count_op(&t.words, OP_CONTROL_BARRIER), 1);
        // a Workgroup (storage class 4) variable
        assert!(insts(&t.words)
            .iter()
            .any(|(op, o)| *op == 59 && o.get(2) == Some(&4)));
        // the paired and 64-bit forms
        let mut c = CsSpec::new(4 << 1, [64, 1, 1]);
        c.lds = 512;
        let lines = [
            "v_lshlrev_b32 v1, 2, v0",
            "v_add_nc_u32 v2, 100, v0",
            "v_sub_nc_u32 v3, 63, v0",
            "ds_write2_b32 v1, v2, v3 offset0:1 offset1:2",
            "ds_read2_b32 v[4:5], v1 offset0:1 offset1:2",
            "ds_write_b64 v1, v[2:3] offset:8",
            "ds_read_b64 v[4:5], v1 offset:8",
        ];
        translate_full(&cs_blob(&lines, &c), None);
        // no LDS in the record: refused
        c.lds = 0;
        let (_, op) = refused_cs(&cs_blob(&lines, &c));
        assert!(op.contains("no LDS"), "{op}");
    }

    #[test]
    fn a_barrier_outside_compute_or_under_a_lane_branch_is_refused() {
        let vs = vs_with(&["s_barrier"], |_| {});
        let (_, op) = refused_cs(&vs);
        assert!(op.contains("outside a compute"), "{op}");
        let c = CsSpec::new(4 << 1, [64, 1, 1]);
        let lines = ["v_cmp_eq_u32 vcc, 0, v0", "s_cbranch_vccz 1", "s_barrier"];
        let (_, op) = refused_cs(&cs_blob(&lines, &c));
        assert!(op.contains("per-lane branch"), "{op}");
    }

    // ---- On a device (`NATIV64_SPIRV_RUN=1`) ----

    fn device() -> Option<&'static Gpu> {
        if !gpu::enabled() {
            return None;
        }
        super::GPU
            .get_or_init(|| {
                Some(Gpu::new().unwrap_or_else(|e| panic!("NATIV64_SPIRV_RUN is set: {e}")))
            })
            .as_ref()
    }

    #[test]
    fn dispatch_writes_the_group_and_thread_ids() {
        let Some(g) = device() else { return };
        let _lock = super::GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let t = translate_full(&cs_ids(), None);
        let (out, _) = g
            .run_stage(&t.words, [2, 2, 1], &[], &[], &[], 16 * 8)
            .unwrap();
        for y in 0..8u32 {
            for x in 0..16u32 {
                assert_eq!(out[(y * 16 + x) as usize], x | y << 8, "({x}, {y})");
            }
        }
    }

    #[test]
    fn dispatch_exchanges_values_through_workgroup_memory() {
        let Some(g) = device() else { return };
        let _lock = super::GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let t = translate_full(&cs_lds_mirror(), None);
        let (out, _) = g.run_stage(&t.words, [1, 1, 1], &[], &[], &[], 64).unwrap();
        for lane in 0..64u32 {
            assert_eq!(out[lane as usize], 100 + 63 - lane, "lane {lane}");
        }
    }

    #[test]
    fn dispatch_loads_a_2d_texel_and_stores_it_to_a_3d_image() {
        let Some(g) = device() else { return };
        let _lock = super::GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let t = translate_full(&cs_image_copy(&[]), None);
        // the table: the SRV (registry index 2) at dword 0, the UAV (index 3) at dword 8
        let magic = u32::from_le_bytes(*b"nVEW");
        let mut table = vec![0u32; 16];
        table[0] = magic;
        table[1] = 2;
        table[8] = magic;
        table[9] = 3;
        let src: Vec<f32> = (0..16).map(|k| k as f32 / 16.0).collect();
        let images = [
            StageImage {
                index: 2,
                storage: false,
                three_d: false,
                extent: [2, 2, 1],
                texels: src.clone(),
            },
            StageImage {
                index: 3,
                storage: true,
                three_d: true,
                extent: [4, 4, 2],
                texels: vec![-1.0; 4 * 4 * 4 * 2],
            },
        ];
        let (_, back) = g
            .run_stage(&t.words, [1, 1, 1], &[], &[(5, table)], &images, 1)
            .unwrap();
        // texel (1, 1) of the 2x2 source is its fourth
        let want = &src[12..16];
        for texel in 0..16 {
            assert_eq!(&back[1][4 * texel..4 * texel + 4], want, "texel {texel}");
        }
        // the second slice is untouched
        assert!(back[1][64..].iter().all(|&v| v == -1.0));
    }
}

// ---------------------------------------------------------------------------------------
// Chunk 7c: quad operations

/// The lock-step wave evaluator over `lanes` console lanes; `v0` is the lane's own input.
fn lock(
    f: &Func,
    s: &Structured,
    lanes: u32,
    host: usize,
    ins: impl Fn(u32) -> Vec<(Reg, u32)>,
) -> Vec<Out> {
    let ids: Vec<u32> = (0..lanes).collect();
    let all: Vec<Vec<(Reg, u32)>> = ids.iter().map(|&l| ins(l)).collect();
    interp_wave(f, s, &all, &ids, host).unwrap()
}

fn quad_expect(sel: [usize; 4], lanes: u32, v: impl Fn(u32) -> u32) -> Vec<u32> {
    (0..lanes)
        .map(|n| v((n & !3) + sel[(n & 3) as usize] as u32))
        .collect()
}

#[test]
fn ds_swizzle_quad_modes_permute_inside_the_quad() {
    let cases: [(&str, [usize; 4]); 7] = [
        (
            "ds_swizzle_b32 v1, v0 offset:swizzle(QUAD_PERM,1,1,1,1)",
            [1; 4],
        ),
        (
            "ds_swizzle_b32 v1, v0 offset:swizzle(QUAD_PERM,1,0,3,2)",
            [1, 0, 3, 2],
        ),
        (
            "ds_swizzle_b32 v1, v0 offset:swizzle(QUAD_PERM,0,1,0,1)",
            [0, 1, 0, 1],
        ),
        (
            "ds_swizzle_b32 v1, v0 offset:swizzle(QUAD_PERM,3,2,1,0)",
            [3, 2, 1, 0],
        ),
        (
            "ds_swizzle_b32 v1, v0 offset:swizzle(QUAD_PERM,0,0,2,2)",
            [0, 0, 2, 2],
        ),
        // bitmask mode that leaves the bits above the quad alone
        (
            "ds_swizzle_b32 v1, v0 offset:swizzle(BITMASK_PERM,\"pppip\")",
            [2, 3, 0, 1],
        ),
        (
            "ds_swizzle_b32 v1, v0 offset:swizzle(BITMASK_PERM,\"ppp01\")",
            [1; 4],
        ),
    ];
    for (asm, sel) in cases {
        let (fun, s) = wave_ok(&[asm]);
        module_of(&fun).unwrap();
        for host in [4, 8, 32] {
            let out = lock(&fun, &s, 32, host, |l| vec![(Reg::V(0), 1000 + l * 7)]);
            let got: Vec<u32> = out.iter().map(|o| get(o, Reg::V(1))).collect();
            assert_eq!(
                got,
                quad_expect(sel, 32, |n| 1000 + n * 7),
                "{asm} host {host}"
            );
        }
    }
}

#[test]
fn dpp_quad_perm_swaps_the_source_then_runs_the_instruction() {
    // v1 = v0[quad lane 3 - k] + v2, floats; the old v1 is not read
    let (fun, s) =
        wave_ok(&["v_add_f32_dpp v1, v0, v2 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf"]);
    module_of(&fun).unwrap();
    let x = |n: u32| (n as f32 + 0.5).to_bits();
    let out = lock(&fun, &s, 16, 8, |l| {
        vec![(Reg::V(0), x(l)), (Reg::V(2), 100f32.to_bits())]
    });
    for (n, o) in out.iter().enumerate() {
        let src = (n & !3) + 3 - (n & 3);
        assert_eq!(f32::from_bits(get(o, Reg::V(1))), src as f32 + 0.5 + 100.0);
    }
    let (fun, s) =
        wave_ok(&["v_mov_b32_dpp v1, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"]);
    module_of(&fun).unwrap();
    let out = lock(&fun, &s, 8, 4, |l| vec![(Reg::V(0), 10 + l)]);
    let got: Vec<u32> = out.iter().map(|o| get(o, Reg::V(1))).collect();
    assert_eq!(got, [11, 10, 13, 12, 15, 14, 17, 16]);
    // dpp8 that stays in the quad, the same in both quads of its group of 8
    let (fun, s) = wave_ok(&["v_mov_b32_dpp v1, v0 dpp8:[1,0,3,2,5,4,7,6]"]);
    module_of(&fun).unwrap();
    let out = lock(&fun, &s, 8, 8, |l| vec![(Reg::V(0), 10 + l)]);
    let got: Vec<u32> = out.iter().map(|o| get(o, Reg::V(1))).collect();
    assert_eq!(got, [11, 10, 13, 12, 15, 14, 17, 16]);
}

#[test]
fn what_leaves_the_quad_stays_refused_by_name() {
    // a row shift, a partial row mask and a bitmask swizzle across quads are not quad ops
    assert_eq!(
        first_cross_lane(&["v_mov_b32_dpp v1, v0 row_shl:1 row_mask:0xf bank_mask:0xf"]),
        "v_mov_b32 (dpp)"
    );
    assert_eq!(
        first_cross_lane(&["v_mov_b32_dpp v1, v0 quad_perm:[1,0,3,2] row_mask:0x5 bank_mask:0xf"]),
        "v_mov_b32 (dpp)"
    );
    assert_eq!(
        first_cross_lane(&["ds_swizzle_b32 v1, v0 offset:swizzle(BITMASK_PERM,\"11110\")"]),
        "ds_swizzle_b32"
    );
}

#[test]
fn a_spilled_lane_mask_comes_back_as_the_same_mask() {
    // VCC goes into a lane of a slot, is clobbered, and comes back through `v_readlane` as the
    // lane's own bit: per lane the spill is the identity
    let lines = [
        "v_cmp_lt_u32_e32 vcc_lo, 3, v0",
        "v_writelane_b32 v8, vcc_lo, 5",
        "s_mov_b32 vcc_lo, 0",
        "v_readlane_b32 s3, v8, 5",
        "v_cndmask_b32_e64 v1, 0, 1, s3",
    ];
    let (fun, s) = wave_ok(&lines);
    assert!(s.masks.slot_masks.contains(&(8, 5)));
    module_of(&fun).unwrap();
    let out = lock(&fun, &s, 8, 8, |l| vec![(Reg::V(0), l)]);
    let got: Vec<u32> = out.iter().map(|o| get(o, Reg::V(1))).collect();
    assert_eq!(got, [0, 0, 0, 0, 1, 1, 1, 1]);
}

fn ballot_expect(lanes: u32, host: u32, set: impl Fn(u32) -> bool) -> Vec<(u32, u32, u32)> {
    // per lane: the count of set bits in its group, the first set host lane, the set bits
    // before it
    (0..lanes)
        .map(|n| {
            let g0 = n / host * host;
            let count = (g0..g0 + host).filter(|&k| set(k)).count() as u32;
            let first = (g0..g0 + host)
                .position(&set)
                .map_or(u32::MAX, |p| p as u32);
            let rank = (g0..n).filter(|&k| set(k)).count() as u32;
            (count, first, rank)
        })
        .collect()
}

#[test]
fn ballots_count_find_and_rank_the_set_lanes_of_the_host_group() {
    for (w64, lines) in [
        (
            false,
            vec![
                "v_cmp_lt_u32_e32 vcc_lo, 3, v0",
                "s_bcnt1_i32_b32 s4, vcc_lo",
                "s_ff1_i32_b32 s5, vcc_lo",
                "v_readlane_b32 s6, v2, s5",
                "v_mbcnt_lo_u32_b32 v3, vcc_lo, 0",
                "v_mov_b32_e32 v4, s6",
            ],
        ),
        (
            true,
            vec![
                "[w64] v_cmp_lt_u32_e32 vcc, 3, v0",
                "[w64] s_bcnt1_i32_b64 s4, vcc",
                "[w64] s_ff1_i32_b64 s5, vcc",
                "[w64] v_readlane_b32 s6, v2, s5",
                "[w64] v_mbcnt_lo_u32_b32 v3, vcc_lo, 0",
                "[w64] v_mbcnt_hi_u32_b32 v3, vcc_hi, v3",
                "v_mov_b32_e32 v4, s6",
            ],
        ),
    ] {
        let (fun, s) = wave_ok(&lines);
        module_of(&fun).unwrap();
        for host in [8u32, 16, 32] {
            let out = lock(&fun, &s, 32, host as usize, |l| {
                vec![(Reg::V(0), l), (Reg::V(2), 500 + 3 * l)]
            });
            // the compare is on v0 = the lane: the bits are lane > 3
            let want2 = ballot_expect(32, host, |k| k > 3);
            for (n, o) in out.iter().enumerate() {
                let (count, first, rank) = want2[n];
                assert_eq!(get(o, Reg::S(4)), count, "w64 {w64} host {host} lane {n}");
                assert_eq!(get(o, Reg::V(3)), rank, "w64 {w64} host {host} lane {n}");
                let g0 = n as u32 / host * host;
                if first != u32::MAX {
                    assert_eq!(
                        get(o, Reg::V(4)),
                        500 + 3 * (g0 + first),
                        "host {host} lane {n}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_ballot_lane_number_that_is_read_as_data_is_refused() {
    // the first set lane is a host lane: shifting it into a mask would mix it with console lanes
    assert_eq!(
        first_cross_lane(&[
            "v_cmp_lt_u32_e32 vcc_lo, 3, v0",
            "s_ff1_i32_b32 s5, vcc_lo",
            "s_lshl_b32 s6, s5, 1",
        ]),
        "s_ff1_i32_b32"
    );
}

#[test]
fn a_mask_compared_with_zero_or_another_mask_is_a_vote() {
    let flag = |lines: &[&str], lanes: u32, host: usize, f: fn(u32) -> u32| -> Vec<u32> {
        let (fun, s) = wave_ok(lines);
        module_of(&fun).unwrap();
        let out = lock(&fun, &s, lanes, host, |l| vec![(Reg::V(0), f(l))]);
        out.iter().map(|o| get(o, Reg::S(7))).collect()
    };
    let cmp = "v_cmp_lt_u32_e32 vcc_lo, 3, v0";
    // lg: some lane has its bit set; eq: none does
    let lg = [cmp, "s_cmp_lg_u32 vcc_lo, 0", "s_cselect_b32 s7, 1, 0"];
    let eq = [cmp, "s_cmp_eq_u32 vcc_lo, 0", "s_cselect_b32 s7, 1, 0"];
    // 16 lanes, host groups of 8: the first group (lanes 0..8) has bits 4..8 set, a group of
    // lanes below 4 has none
    assert_eq!(flag(&lg, 16, 8, |l| l), vec![1; 16]);
    assert_eq!(flag(&eq, 16, 8, |l| l), vec![0; 16]);
    assert_eq!(flag(&lg, 16, 4, |l| l), [vec![0; 4], vec![1; 12]].concat());
    // vcc against EXEC (all ones): equal when every lane of the group has its bit set
    let same = [
        cmp,
        "s_cmp_eq_u32 vcc_lo, exec_lo",
        "s_cselect_b32 s7, 1, 0",
    ];
    assert_eq!(flag(&same, 8, 4, |_| 9), vec![1; 8]);
    assert_eq!(flag(&same, 8, 4, |l| l), [vec![0; 4], vec![1; 4]].concat());
}

/// A wave32 sum the way a compiler lowers it: set the inactive lanes to 0, prefix inside the
/// rows of 16 by DPP shifts, carry row 0's total into row 1 by `v_permlanex16`, read lane 31.
const SUM_SECTION: [&str; 18] = [
    "v_cmp_lt_u32_e32 vcc_lo, 3, v0",
    "s_and_saveexec_b32 s3, vcc_lo",
    "s_or_saveexec_b32 s4, -1",
    "v_cndmask_b32_e64 v1, 0, v0, s4",
    "v_mov_b32_dpp v2, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1",
    "v_add_nc_u32_e32 v1, v1, v2",
    "v_mov_b32_dpp v2, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1",
    "v_add_nc_u32_e32 v1, v1, v2",
    "v_mov_b32_dpp v2, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1",
    "v_add_nc_u32_e32 v1, v1, v2",
    "v_mov_b32_dpp v2, v1 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1",
    "v_add_nc_u32_e32 v1, v1, v2",
    "v_permlanex16_b32 v2, v1, -1, -1",
    "s_mov_b32 exec_lo, 0xffff0000",
    "v_add_nc_u32_e32 v1, v1, v2",
    "s_mov_b32 exec_lo, -1",
    "v_readlane_b32 s6, v1, 31",
    "s_mov_b32 exec_lo, s4",
];

#[test]
fn a_whole_wave_section_becomes_a_scan_and_a_reduction() {
    let mut lines = SUM_SECTION.to_vec();
    lines.extend(["v_mov_b32_e32 v3, v1", "v_mov_b32_e32 v4, s6"]);
    let (fun, s) = wave_ok(&lines);
    let outs: Vec<Scan> = s
        .masks
        .lanes
        .values()
        .filter_map(|u| match u {
            LaneUse::WaveOut { op, scan, .. } => {
                assert_eq!(*op, WaveOp::IAdd);
                Some(*scan)
            }
            _ => None,
        })
        .collect();
    assert_eq!(outs.len(), 2, "{outs:?}");
    assert!(outs.contains(&Scan::Inclusive) && outs.contains(&Scan::Reduce));
    module_of(&fun).unwrap();
    // the lanes up to 3 are inactive: they add nothing and keep their v3
    let value = |k: u32| if k > 3 { 100 + k } else { 0 };
    for host in [8usize, 16, 32] {
        let out = lock(&fun, &s, 32, host, |l| {
            vec![(Reg::V(0), if l > 3 { 100 + l } else { l })]
        });
        for (n, o) in out.iter().enumerate().skip(4) {
            let n = n as u32;
            let g0 = n / host as u32 * host as u32;
            let prefix: u32 = (g0..=n).map(value).sum();
            let total: u32 = (g0..g0 + host as u32).map(value).sum();
            assert_eq!(get(o, Reg::V(3)), prefix, "host {host} lane {n}");
            assert_eq!(get(o, Reg::V(4)), total, "host {host} lane {n}");
        }
    }
}

#[test]
fn a_whole_wave_section_the_simulation_does_not_model_is_refused() {
    // a product of two lane-crossing values is no reduction the simulation knows
    let mut lines = SUM_SECTION.to_vec();
    lines.insert(13, "v_mul_lo_u32 v2, v1, v2");
    assert_eq!(first_cross_lane(&lines), "v_mov_b32 (dpp)");
}
