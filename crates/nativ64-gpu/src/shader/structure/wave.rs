//! Chunk 7c (`docs/shaders.md`, Chunk 7 design, item 5): whole-wave sections. A compiler lowers
//! `WaveActiveSum`, `WaveActiveMin`, `WavePrefixSum`... to a section that sets EXEC to all ones
//! (`s_or_saveexec_b32 sN, -1`), combines the lanes with DPP, `v_permlanex16` and `v_readlane`,
//! and restores EXEC (`s_mov_b32 exec_lo, sN`). The section is simulated on the wave's lanes
//! with symbolic values: what each lane holds is the combination, by one operation, of a set of
//! lanes of one value (the masked input the section starts from). A value the section leaves
//! that turns out to be the reduction, the inclusive prefix, or the exclusive prefix of one
//! operation over all the lanes becomes one subgroup operation over the host's active
//! invocations (`WaveOut`); every other instruction of the section that touches a lane-crossing
//! value is skipped. Any step the simulation does not model refuses the section: its
//! instructions stay cross-lane uses, refused by name.

use std::collections::HashMap;

use super::uniform::LaneUse;
use crate::shader::gfx10::{Dpp, DppKind, Enc};
use crate::shader::ir::{BlockId, Func, Inst, Reg, Src, Term, Val, ValDef};

/// The associative operation of a reduction or scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveOp {
    IAdd,
    FAdd,
    UMin,
    UMax,
    SMin,
    SMax,
    FMin,
    FMax,
    And,
    Or,
    Xor,
}

impl WaveOp {
    fn of(name: &str) -> Option<Self> {
        Some(match name {
            "v_add_nc_u32" | "v_add_u32" => Self::IAdd,
            "v_add_f32" => Self::FAdd,
            "v_min_u32" => Self::UMin,
            "v_max_u32" => Self::UMax,
            "v_min_i32" => Self::SMin,
            "v_max_i32" => Self::SMax,
            "v_min_f32" => Self::FMin,
            "v_max_f32" => Self::FMax,
            "v_and_b32" => Self::And,
            "v_or_b32" => Self::Or,
            "v_xor_b32" => Self::Xor,
            _ => return None,
        })
    }

    /// Combining a lane with itself changes nothing.
    fn idempotent(self) -> bool {
        !matches!(self, Self::IAdd | Self::FAdd | Self::Xor)
    }

    /// The operand that changes nothing, and what a lane the DPP reads out of range gives
    /// with `bound_ctrl` (0).
    fn identity(self) -> u32 {
        match self {
            Self::IAdd | Self::FAdd | Self::UMax | Self::Or | Self::Xor => 0,
            Self::UMin | Self::And => u32::MAX,
            Self::SMin => i32::MAX as u32,
            Self::SMax => i32::MIN as u32,
            Self::FMin => 0x7f80_0000,
            Self::FMax => 0xff80_0000,
        }
    }
}

/// Which lanes a result is the combination of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scan {
    /// Every lane: the same value in all of them.
    Reduce,
    /// Lane k: lanes 0..=k.
    Inclusive,
    /// Lane k: lanes 0..k (the operation's identity for lane 0).
    Exclusive,
}

/// What one lane holds during the simulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sym {
    /// The combination by `op` of the values of `v` in the lanes of `set`; `op` is `None` for
    /// a single lane's own value.
    Acc {
        op: Option<WaveOp>,
        v: Val,
        set: u64,
    },
    Const(u32),
    Unknown,
}

fn combine(op: WaveOp, a: Sym, b: Sym) -> Sym {
    match (a, b) {
        (Sym::Const(c), x) | (x, Sym::Const(c)) if c == op.identity() => match x {
            Sym::Unknown => Sym::Unknown,
            x => x,
        },
        (
            Sym::Acc {
                op: oa,
                v: va,
                set: sa,
            },
            Sym::Acc {
                op: ob,
                v: vb,
                set: sb,
            },
        ) if va == vb
            && oa.is_none_or(|o| o == op)
            && ob.is_none_or(|o| o == op)
            && (op.idempotent() || sa & sb == 0) =>
        {
            Sym::Acc {
                op: Some(op),
                v: va,
                set: sa | sb,
            }
        }
        _ => Sym::Unknown,
    }
}

type State = HashMap<Val, Vec<Sym>>;

struct Sim<'a> {
    f: &'a Func,
    n: usize,
    st: State,
    /// The constant dword each EXEC value holds inside the section.
    exec: HashMap<Val, u32>,
}

impl Sim<'_> {
    fn sym(&self, s: Src, k: usize) -> Sym {
        match s {
            Src::Val(v) => match self.st.get(&v) {
                Some(x) => x[k],
                None => Sym::Acc {
                    op: None,
                    v,
                    set: 1 << k,
                },
            },
            Src::Const(c) => Sym::Const(c),
            _ => Sym::Unknown,
        }
    }

    fn touches_state(&self, i: &Inst) -> bool {
        i.srcs
            .iter()
            .flatten()
            .any(|s| matches!(s, Src::Val(v) if self.st.contains_key(v)))
    }

    /// The EXEC mask an instruction runs under, as lane bits.
    fn exec_of(&self, i: &Inst) -> Option<u64> {
        let get = |r: Reg| match i.imps.iter().find(|(x, _)| *x == r)?.1 {
            Src::Val(v) => self.exec.get(&v).copied(),
            Src::Const(c) => Some(c),
            _ => None,
        };
        let lo = u64::from(get(Reg::ExecLo)?);
        let hi = if self.f.wave32 {
            0
        } else {
            u64::from(get(Reg::ExecHi)?)
        };
        Some(lo | hi << 32)
    }

    fn full(&self) -> u64 {
        if self.n == 64 {
            u64::MAX
        } else {
            (1 << self.n) - 1
        }
    }
}

/// The lane a DPP16 / DPP8 destination lane `i` reads: `Err` for a control the simulation does
/// not model, `Ok(None)` for a lane that has no source.
fn dpp_source(d: &Dpp, i: usize) -> Result<Option<usize>, ()> {
    let (row, lane) = (i & !15, i & 15);
    if d.kind != DppKind::Dpp16 {
        return Ok(Some((i & !7) + usize::from(d.sel[i & 7])));
    }
    let c = d.ctrl as usize;
    Ok(match c {
        0..=0xff => Some((i & !3) | (c >> (2 * (i & 3)) & 3)),
        0x101..=0x10f => (lane + (c & 15) < 16).then(|| row + lane + (c & 15)),
        0x111..=0x11f => (lane >= c & 15).then(|| row + lane - (c & 15)),
        0x121..=0x12f => Some(row + (lane + 16 - (c & 15)) % 16),
        0x140 => Some(row + 15 - lane),
        0x141 => Some((i & !7) + 7 - (i & 7)),
        0x150..=0x15f => Some(row + (c & 15)),
        0x160..=0x16f => Some(row + (lane ^ (c & 15))),
        _ => return Err(()),
    })
}

/// DPP16 only writes the banks and rows its masks name.
fn dpp_enabled(d: &Dpp, i: usize) -> bool {
    d.kind != DppKind::Dpp16
        || (d.row_mask >> ((i / 16) & 3)) & 1 != 0 && (d.bank_mask >> ((i / 4) & 3)) & 1 != 0
}

fn plain_mods(i: &Inst) -> bool {
    let m = i.instr.mods;
    m.abs == 0 && m.neg == 0 && !m.clamp && m.omod == 0 && i.instr.sdwa.is_none()
}

/// What the section leaves in one value, as one subgroup operation, when it is one.
fn classify(x: &[Sym], full: u64) -> Option<(WaveOp, Scan, Val)> {
    let mut found = None;
    for s in x {
        if let Sym::Acc {
            op: Some(op), v, ..
        } = s
        {
            found = Some((*op, *v));
            break;
        }
    }
    let (op, v) = found?;
    let set_of = |k: usize, s: &Sym| match *s {
        // a lane that only combined identities still holds its own value alone
        Sym::Acc { op: o, v: w, set } if o.is_none_or(|o| o == op) && w == v => Some(set),
        Sym::Const(c) if c == op.identity() && k == 0 => Some(0),
        _ => None,
    };
    let sets: Option<Vec<u64>> = x.iter().enumerate().map(|(k, s)| set_of(k, s)).collect();
    let sets = sets?;
    let upto = |k: usize| if k >= 64 { u64::MAX } else { (1u64 << k) - 1 };
    if sets.iter().all(|&s| s == full) {
        return Some((op, Scan::Reduce, v));
    }
    if sets
        .iter()
        .enumerate()
        .all(|(k, &s)| s == upto(k + 1) & full)
    {
        return Some((op, Scan::Inclusive, v));
    }
    if sets.iter().enumerate().all(|(k, &s)| s == upto(k)) {
        return Some((op, Scan::Exclusive, v));
    }
    None
}

/// Where each value is observed: the block and the instruction (`usize::MAX`: a phi or the
/// terminator) of every read that can matter.
type Reads = HashMap<Val, Vec<(BlockId, usize)>>;

/// The constant a dword of EXEC holds, when the instruction that defines it says so.
fn exec_const(f: &Func, s: Src) -> Option<u32> {
    let v = match s {
        Src::Const(c) => return Some(c),
        Src::Val(v) => v,
        _ => return None,
    };
    let ValDef::Inst { block, index } = f.vals[v.0 as usize].def else {
        return None;
    };
    let i = &f.blocks[block].insts[index];
    let k = i
        .dsts
        .iter()
        .filter(|(r, _)| matches!(r, Reg::ExecLo | Reg::ExecHi))
        .position(|(_, d)| *d == v)?;
    let n = i.instr.name();
    match i.srcs.first()?.get(k)? {
        Src::Const(c) if n.starts_with("s_mov_b") => Some(*c),
        Src::Const(u32::MAX) if n.starts_with("s_or_saveexec_b") => Some(u32::MAX),
        _ => None,
    }
}

/// True when the instruction writes every lane of its destination, so the old value it names
/// is never seen again.
fn writes_every_lane(f: &Func, i: &Inst) -> bool {
    let all = |r: Reg| {
        i.imps
            .iter()
            .any(|&(x, s)| x == r && exec_const(f, s) == Some(u32::MAX))
    };
    let partial_dpp = i.instr.dpp.is_some_and(|d| {
        let shifts = d.kind == DppKind::Dpp16 && (0x101..=0x13f).contains(&d.ctrl);
        (0..64).any(|k| !dpp_enabled(&d, k)) || (shifts && !d.bound_ctrl)
    });
    all(Reg::ExecLo) && (f.wave32 || all(Reg::ExecHi)) && !partial_dpp
}

/// The reads of every value. The old value of a VGPR that an instruction overwrites in every
/// lane is not one (the temporaries of a section are reused that way), nor is the input of a
/// phi nothing reads.
fn reads_of(f: &Func) -> Reads {
    let mut out = Reads::new();
    let mut note = |s: &Src, at: (BlockId, usize)| {
        if let Src::Val(v) = s {
            out.entry(*v).or_default().push(at);
        }
    };
    for (b, blk) in f.blocks.iter().enumerate() {
        for (k, inst) in blk.insts.iter().enumerate() {
            inst.srcs.iter().flatten().for_each(|s| note(s, (b, k)));
            let full = writes_every_lane(f, inst);
            for (r, s) in &inst.imps {
                if !(full && matches!(r, Reg::V(_)) && inst.dsts.iter().any(|d| d.0 == *r)) {
                    note(s, (b, k));
                }
            }
        }
        if let Term::Branch { cond, .. } = &blk.term {
            cond.iter().for_each(|(_, s)| note(s, (b, usize::MAX)));
        }
    }
    let mut pending: Vec<(BlockId, usize)> = f
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(b, blk)| (0..blk.phis.len()).map(move |k| (b, k)))
        .collect();
    loop {
        let before = pending.len();
        pending.retain(|&(b, k)| {
            let phi = &f.blocks[b].phis[k];
            if !out.contains_key(&phi.dst) {
                return true;
            }
            for (_, s) in &phi.args {
                if let Src::Val(v) = s {
                    out.entry(*v).or_default().push((b, usize::MAX));
                }
            }
            false
        });
        if pending.len() == before {
            return out;
        }
    }
}

/// Simulates the section of block `b` between instruction `a` (the `s_or_saveexec ..., -1`)
/// and `end` (the restore), and says what each instruction of it becomes.
fn section(
    f: &Func,
    reads: &Reads,
    (b, a, end): (BlockId, usize, usize),
) -> Option<HashMap<usize, LaneUse>> {
    let n = if f.wave32 { 32 } else { 64 };
    let mut sim = Sim {
        f,
        n,
        st: HashMap::new(),
        exec: HashMap::new(),
    };
    let insts = &f.blocks[b].insts;
    for (r, v) in &insts[a].dsts {
        if matches!(r, Reg::ExecLo | Reg::ExecHi) {
            sim.exec.insert(*v, u32::MAX);
        }
    }
    let mut skipped: Vec<usize> = Vec::new();
    for (idx, i) in insts.iter().enumerate().take(end).skip(a + 1) {
        let name = i.instr.name();
        // EXEC changes: constants only
        if i.dsts
            .iter()
            .any(|(r, _)| matches!(r, Reg::ExecLo | Reg::ExecHi))
        {
            if name != "s_mov_b32" && name != "s_mov_b64" {
                return None;
            }
            for (k, (r, v)) in i
                .dsts
                .iter()
                .filter(|(r, _)| matches!(r, Reg::ExecLo | Reg::ExecHi))
                .enumerate()
            {
                let _ = r;
                match i.srcs.first().and_then(|s| s.get(k)) {
                    Some(Src::Const(c)) => sim.exec.insert(*v, *c),
                    _ => return None,
                };
            }
            continue;
        }
        let cross = i.instr.dpp.is_some()
            || name.starts_with("v_permlane")
            || name.starts_with("v_readlane")
            || name.starts_with("v_readfirstlane")
            || name.starts_with("v_writelane");
        let touches = sim.touches_state(i);
        if !cross && !touches {
            // lane-local on lane-local values; under a partial EXEC the lanes it skips keep an
            // old value that may not be
            if let Some((Reg::V(_), _)) = i.dsts.first() {
                let old_state = i.imps.iter().any(
                    |(r, s)| matches!((r, s), (Reg::V(_), Src::Val(v)) if sim.st.contains_key(v)),
                );
                if old_state && sim.exec_of(i)? != sim.full() {
                    return None;
                }
            }
            continue;
        }
        let exec = sim
            .exec_of(i)
            .or_else(|| (!i.imps.iter().any(|(r, _)| *r == Reg::ExecLo)).then(|| sim.full()))?;
        let dst = i.dsts.first()?;
        let old = i.imps.iter().find_map(|(r, s)| (*r == dst.0).then_some(*s));
        let at = |sim: &Sim, s: Src, k: usize| sim.sym(s, k);
        let mut out = vec![Sym::Unknown; n];
        if name.starts_with("v_readlane") || name.starts_with("v_readfirstlane") {
            if !matches!(dst.0, Reg::S(_)) {
                return None;
            }
            let lane = if name.starts_with("v_readfirstlane") {
                let e = exec & sim.full();
                if e == 0 {
                    return None;
                }
                e.trailing_zeros() as usize
            } else {
                match i.srcs.get(1)?.first()? {
                    Src::Const(c) => (*c as usize) & (n - 1),
                    _ => return None,
                }
            };
            let s = at(&sim, *i.srcs.first()?.first()?, lane);
            out = vec![s; n];
        } else if name == "v_permlanex16_b32" || name == "v_permlane16_b32" {
            let (Src::Const(c1), Src::Const(c2)) =
                (*i.srcs.get(1)?.first()?, *i.srcs.get(2)?.first()?)
            else {
                return None;
            };
            let sel = |l: usize| {
                let c = if l & 15 < 8 { c1 } else { c2 };
                (c >> (4 * (l & 7)) & 15) as usize
            };
            for (k, o) in out.iter_mut().enumerate() {
                let row = if name == "v_permlanex16_b32" {
                    (k >> 4) ^ 1
                } else {
                    k >> 4
                };
                let j = row * 16 + sel(k);
                let old_k = old.map_or(Sym::Unknown, |s| at(&sim, s, k));
                *o = if j >= n || exec >> k & 1 == 0 || exec >> j & 1 == 0 {
                    old_k
                } else {
                    at(&sim, *i.srcs.first()?.first()?, j)
                };
            }
        } else if name.starts_with("v_writelane") {
            return None;
        } else {
            // a VALU: DPP on source 0, or a combination of lane-crossing values
            if !plain_mods(i) || !matches!(dst.0, Reg::V(_)) {
                return None;
            }
            if (name == "v_sub_nc_u32" || name == "v_subrev_nc_u32") && i.instr.dpp.is_none() {
                // inclusive prefix minus the lane's own value: the exclusive prefix
                let (x, y) = (*i.srcs.first()?.first()?, *i.srcs.get(1)?.first()?);
                let (x, y) = if name == "v_sub_nc_u32" {
                    (x, y)
                } else {
                    (y, x)
                };
                for (k, o) in out.iter_mut().enumerate() {
                    *o = match (at(&sim, x, k), at(&sim, y, k)) {
                        (
                            Sym::Acc {
                                op: Some(WaveOp::IAdd),
                                v,
                                set,
                            },
                            Sym::Acc {
                                op: None,
                                v: w,
                                set: one,
                            },
                        ) if v == w && set & one == one && exec >> k & 1 == 1 => {
                            if set & !one == 0 {
                                Sym::Const(0)
                            } else {
                                Sym::Acc {
                                    op: Some(WaveOp::IAdd),
                                    v,
                                    set: set & !one,
                                }
                            }
                        }
                        _ => Sym::Unknown,
                    };
                }
                sim.st.insert(dst.1, out);
                skipped.push(idx);
                continue;
            }
            let op = if name == "v_mov_b32" || name == "v_mov_b32_e32" {
                None
            } else {
                Some(WaveOp::of(name)?)
            };
            let s0 = *i.srcs.first()?.first()?;
            let s1 = i.srcs.get(1).and_then(|s| s.first()).copied();
            let d = i.instr.dpp;
            if d.is_some_and(|d| d.src0_neg || d.src0_abs || d.src1_neg || d.src1_abs) {
                return None;
            }
            for (k, o) in out.iter_mut().enumerate() {
                let old_k = old.map_or(Sym::Unknown, |s| at(&sim, s, k));
                if exec >> k & 1 == 0 {
                    *o = old_k;
                    continue;
                }
                let a = match d {
                    None => at(&sim, s0, k),
                    Some(d) => {
                        if !dpp_enabled(&d, k) {
                            *o = old_k;
                            continue;
                        }
                        match dpp_source(&d, k).ok()? {
                            Some(j) if j < n && exec >> j & 1 == 1 => at(&sim, s0, j),
                            _ if d.bound_ctrl => Sym::Const(0),
                            _ => {
                                *o = old_k;
                                continue;
                            }
                        }
                    }
                };
                *o = match (op, s1) {
                    (None, _) => a,
                    (Some(op), Some(s1)) => combine(op, a, at(&sim, s1, k)),
                    (Some(_), None) => Sym::Unknown,
                };
            }
        }
        sim.st.insert(dst.1, out);
        skipped.push(idx);
    }
    // what the rest of the program reads of the section
    let inside = |&(bi, k): &(BlockId, usize)| bi == b && k > a && k < end;
    let mut uses: HashMap<usize, LaneUse> = HashMap::new();
    for &idx in &skipped {
        let i = &insts[idx];
        let v = i.dsts.first()?.1;
        let u = if reads.get(&v).is_some_and(|r| !r.iter().all(inside)) {
            let (op, scan, leaf) = classify(&sim.st[&v], sim.full())?;
            // the emitter keeps the leaf's value from its definition: it has to be in the
            // section, before this instruction
            let here = |d: &Inst| d.dsts.iter().any(|(_, x)| *x == leaf);
            if !insts[a + 1..idx].iter().any(here) {
                return None;
            }
            LaneUse::WaveOut { op, scan, leaf }
        } else {
            LaneUse::WaveSkip
        };
        uses.insert(i.instr.at, u);
    }
    Some(uses)
}

/// The sections of `f` the simulation understands, as the `LaneUse` of each instruction in them
/// that crosses lanes.
pub(super) fn sections(f: &Func) -> HashMap<usize, LaneUse> {
    let mut out = HashMap::new();
    let mut reads = None;
    for (b, blk) in f.blocks.iter().enumerate() {
        for (a, i) in blk.insts.iter().enumerate() {
            let n = i.instr.name();
            let all_ones = i
                .srcs
                .first()
                .is_some_and(|s| !s.is_empty() && s.iter().all(|x| *x == Src::Const(u32::MAX)));
            if !(n.starts_with("s_or_saveexec_b") && all_ones && i.instr.enc == Enc::Sop1) {
                continue;
            }
            // the register the old EXEC went to, and where it comes back
            let saved: Vec<Val> = i
                .dsts
                .iter()
                .filter(|(r, _)| !matches!(r, Reg::ExecLo | Reg::ExecHi | Reg::Scc))
                .map(|(_, v)| *v)
                .collect();
            let end = blk.insts.iter().enumerate().skip(a + 1).find_map(|(k, j)| {
                let restores = j.dsts.iter().any(|(r, _)| *r == Reg::ExecLo)
                    && j.instr.name().starts_with("s_mov_b")
                    && j.srcs.first().is_some_and(|s| {
                        !s.is_empty()
                            && s.iter()
                                .all(|x| matches!(x, Src::Val(v) if saved.contains(v)))
                    });
                restores.then_some(k)
            });
            let Some(end) = end else { continue };
            let reads = reads.get_or_insert_with(|| reads_of(f));
            if let Some(u) = section(f, reads, (b, a, end)) {
                out.extend(u);
            }
        }
    }
    out
}
