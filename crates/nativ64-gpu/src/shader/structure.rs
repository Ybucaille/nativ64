//! Structuring (`docs/shaders.md`, Structuring; chunk 5 of the G4 plan): turns the SSA `Func`
//! of a program, whose divergence lives in EXEC, into what a per-lane structured program
//! needs. Each SPIR-V invocation runs one console lane, so a wave-sized mask becomes one
//! boolean per invocation, and no SPIR-V is written here: this module only computes the
//! facts the emitter reads.
//!
//! 1. `type_masks`: which values are masks, and every use that crosses lanes.
//! 2. `Structured::branches`: a wave-wide test of a mask is a test of the lane's own bit;
//!    any other branch is uniform. A scalar a lane-skipped region leaves different on the two
//!    paths makes that branch a wave-wide one (`BranchKind::Wave`: the region runs when any lane
//!    enters it), unless the region stores, does an atomic or sends a message: then it is a
//!    `DivergentScalar` error.
//! 3. `Structured::guards`: every vector instruction is guarded by `active`; the guards a
//!    path-sensitive pass proves true (after `s_and_saveexec` + `s_cbranch_execz`) are
//!    `Guard::Proved`.
//! 4. `Structured::demotes`: the pixel discard pattern, marked at its `s_andn2 live, live, exec`.
//! 5. `Structured::tree`: the reducible CFG as Ramsey's tree (*Beyond Relooper*, JFP 2022).
//!    `Br` out of several `Block` / `Loop` levels is data (`depth`, and `forwards` on each
//!    construct it leaves): the emitter sets the `target` variable and each merge re-tests it.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

mod uniform;
use uniform::narrows;
pub use uniform::{Dom, LaneUse};

use super::gfx10::{Enc, Instr, Operand};
use super::ir::{dominators, BlockId, Func, Inst, Reg, Src, Term, Val, ValDef};

/// An instruction that reads across lanes (a use of a mask other than the allowed ones, or an
/// instruction that is cross-lane by nature: `v_readlane`, `v_mbcnt`, DPP...).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrossLane {
    /// Byte offset of the instruction.
    pub at: usize,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StructError {
    /// Needs the wave-operations chunk.
    CrossLane(CrossLane),
    /// A scalar the paths of a lane-wise branch define differently, read after their join
    /// (block at byte offset `at`).
    DivergentScalar {
        at: usize,
        reg: Reg,
        /// Byte offset of the first instruction of the block of the lane-wise branch.
        branch: usize,
    },
    /// The CFG has a loop with two entries; `at` is the block a retreating edge goes to.
    Irreducible { at: usize },
    /// The tree builder met a state its invariants rule out.
    Malformed(&'static str),
}

impl fmt::Display for StructError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CrossLane(c) => write!(f, "cross-lane use by {} at +{:#x}", c.name, c.at),
            Self::DivergentScalar { at, reg, branch } => write!(
                f,
                "scalar {reg} differs between the paths of the lane-wise branch of the block at \
                 +{branch:#x}, read at +{at:#x}"
            ),
            Self::Irreducible { at } => write!(f, "irreducible flow at +{at:#x}"),
            Self::Malformed(s) => write!(f, "malformed structure: {s}"),
        }
    }
}

impl std::error::Error for StructError {}

// ---------------------------------------------------------------------------------------
// 1. Mask typing

/// The bitwise operations on masks (and `s_wqm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bit {
    And,
    Or,
    Xor,
    Andn2,
    Andn1,
    Orn2,
    Nand,
    Nor,
    Xnor,
    Not,
    Mov,
    Cselect,
    Wqm,
}

#[derive(Clone, Copy)]
struct MaskOp {
    bit: Bit,
    /// `s_*_saveexec` and `s_*_wrexec`: reads EXEC and writes it.
    save: bool,
    /// `s_*_wrexec`: the destinations are the new EXEC twice (the SGPR, then EXEC), not the
    /// old EXEC then the new one.
    wr: bool,
}

/// The wave-sized scalar operation `i` is, when it is one of the mask operations.
fn mask_op(i: &Instr) -> Option<MaskOp> {
    if !matches!(i.enc, Enc::Sop1 | Enc::Sop2) {
        return None;
    }
    let n = i
        .name()
        .strip_suffix(if i.wave32 { "_b32" } else { "_b64" })?;
    let (n, save, wr) = match (n.strip_suffix("_saveexec"), n.strip_suffix("_wrexec")) {
        (Some(b), _) => (b, true, false),
        (_, Some(b)) => (b, true, true),
        _ => (n, false, false),
    };
    let bit = match n {
        "s_and" => Bit::And,
        "s_or" => Bit::Or,
        "s_xor" => Bit::Xor,
        "s_andn2" => Bit::Andn2,
        "s_andn1" if save => Bit::Andn1,
        "s_orn2" => Bit::Orn2,
        "s_nand" => Bit::Nand,
        "s_nor" => Bit::Nor,
        "s_xnor" => Bit::Xnor,
        "s_not" => Bit::Not,
        "s_mov" => Bit::Mov,
        "s_cselect" => Bit::Cselect,
        "s_wqm" => Bit::Wqm,
        _ => return None,
    };
    if save && matches!(bit, Bit::Not | Bit::Mov | Bit::Cselect | Bit::Wqm) {
        return None;
    }
    if wr && !matches!(bit, Bit::Andn1 | Bit::Andn2) {
        return None;
    }
    Some(MaskOp { bit, save, wr })
}

fn is_exec(r: Reg) -> bool {
    matches!(r, Reg::ExecLo | Reg::ExecHi)
}

fn is_vector(i: &Instr) -> bool {
    !matches!(
        i.enc,
        Enc::Sop1 | Enc::Sop2 | Enc::Sopc | Enc::Sopk | Enc::Sopp | Enc::Smem
    )
}

/// The destinations a mask operation defines (its SCC is not one of them).
fn mask_dsts(inst: &Inst) -> impl Iterator<Item = Val> + '_ {
    inst.dsts
        .iter()
        .filter(|(r, _)| *r != Reg::Scc)
        .map(|(_, v)| *v)
}

/// The sources a mask operation combines: its operands, and EXEC for `saveexec`.
fn mask_srcs<'a>(inst: &'a Inst, op: MaskOp) -> impl Iterator<Item = Src> + 'a {
    let imps = inst
        .imps
        .iter()
        .filter(move |(r, _)| op.save && is_exec(*r))
        .map(|(_, s)| *s);
    inst.srcs.iter().flatten().copied().chain(imps)
}

/// True for the vector instructions that read a mask per lane in their last source (or VCC).
fn takes_lane_mask(name: &str) -> bool {
    ["v_cndmask", "v_addc", "v_subb", "v_subbrev", "v_div_fmas"]
        .iter()
        .any(|p| name.starts_with(p))
        || (name.contains("_co_ci_"))
}

/// Instructions whose result depends on other lanes, whatever their operands.
pub fn cross_lane_by_nature(i: &Instr) -> bool {
    let n = i.name();
    i.dpp.is_some()
        || n.starts_with("v_readlane")
        || n.starts_with("v_readfirstlane")
        || n.starts_with("v_writelane")
        || n.starts_with("v_mbcnt")
        || n.starts_with("v_permlane")
        || n.starts_with("ds_swizzle")
        || n.starts_with("ds_permute")
        || n.starts_with("ds_bpermute")
}

/// The typing of the wave-sized values.
#[derive(Clone, Debug, Default)]
pub struct Masks {
    /// The mask dwords: each is the lane's own bit (a wave64 pair is two entries).
    pub masks: HashSet<Val>,
    /// An SCC defined by a mask operation holds "the result is not zero", which per lane is
    /// the bit of that result: SCC value -> the result dwords.
    pub scc_of: HashMap<Val, Vec<Val>>,
    /// Masks that are uniform over the lanes an EXEC value holds: the bit is the same in
    /// every lane where that EXEC (the entry here) is set. EXEC values are uniform over
    /// themselves, constants over any (`None`), and bitwise operations (and `s_cselect` on a
    /// scalar condition) of masks uniform over the same EXEC are too. A branch on one that is
    /// not an EXEC value (`vccz` of `exec & !(scc ? exec : 0)`) is wave-wide in the program
    /// and does not split the lanes that run on.
    pub uniform: HashMap<Val, Option<Val>>,
    /// An EXEC value that only narrows an earlier one (`v_cmpx`, `s_and_saveexec`, `and`):
    /// its lanes are among those of the earlier value.
    pub within: HashMap<Val, Val>,
    /// Every use that crosses lanes, in program order.
    pub cross_lane: Vec<CrossLane>,
    /// By `Val`: the lanes the value is the same in (everything that is not a mask or a VGPR
    /// counts as uniform over all of them).
    pub uniform_vals: Vec<Option<Dom>>,
    /// VGPRs that only spill SGPRs into lanes (`v_writelane` / `v_readlane` at constant lanes).
    pub slots: HashSet<u16>,
    /// The wave instructions items 1-4 of the chunk 7 design give a meaning to, by the byte
    /// offset of the instruction. They are not in `cross_lane`.
    pub lanes: HashMap<usize, LaneUse>,
}

impl Masks {
    pub fn is_mask(&self, v: Val) -> bool {
        self.masks.contains(&v)
    }

    /// The value is the same in every active lane.
    pub fn is_uniform(&self, v: Val) -> bool {
        self.uniform_vals
            .get(v.0 as usize)
            .is_some_and(Option::is_some)
    }

    /// A mask, or one of the constants 0 / -1.
    fn src_ok(&self, s: &Src) -> bool {
        match s {
            Src::Const(c) => *c == 0 || *c == u32::MAX,
            Src::Val(v) => self.masks.contains(v),
            Src::Undef => true,
            _ => false,
        }
    }

    /// What a mask operation may combine: `src_ok`, or a scalar that is not a mask, whose bit
    /// n is the bit of lane n (`s_and_b32 exec_lo, exec_lo, s5`).
    fn bit_src_ok(&self, s: &Src) -> bool {
        matches!(s, Src::Const(_) | Src::Val(_) | Src::Undef)
    }

    fn is_mask_src(&self, s: &Src) -> bool {
        matches!(s, Src::Val(v) if self.masks.contains(v))
    }
}

/// Types the masks of `f` (step 1).
///
/// A value is a mask when it is an entry EXEC, a destination of a vector instruction that is
/// not a VGPR (`v_cmp*`, carry-outs), or the result of a bitwise operation (`and`, `or`, `xor`,
/// `andn2`, `orn2`, `nand`, `nor`, `xnor`, `not`, `mov`, `cselect`, `wqm`, and their
/// `saveexec` forms) whose inputs are masks or the constants 0 / -1, or a phi of such values.
/// Typing starts optimistic and drops what its inputs refute, so loop-carried masks type.
pub fn type_masks(f: &Func) -> Masks {
    let mut m = Masks::default();
    for &(r, v) in &f.args {
        if is_exec(r) {
            m.masks.insert(v);
        }
    }
    enum Cand {
        Op(BlockId, usize, MaskOp),
        Phi(BlockId, usize),
    }
    let mut cands = Vec::new();
    for (b, blk) in f.blocks.iter().enumerate() {
        for (k, p) in blk.phis.iter().enumerate() {
            if !matches!(p.reg, Reg::V(_) | Reg::Scc | Reg::M0) {
                m.masks.insert(p.dst);
                cands.push(Cand::Phi(b, k));
            }
        }
        for (k, inst) in blk.insts.iter().enumerate() {
            if is_vector(&inst.instr) {
                if !cross_lane_by_nature(&inst.instr) {
                    for &(r, v) in &inst.dsts {
                        if !matches!(r, Reg::V(_)) {
                            m.masks.insert(v);
                        }
                    }
                }
            } else if let Some(op) = mask_op(&inst.instr) {
                m.masks.extend(mask_dsts(inst));
                cands.push(Cand::Op(b, k, op));
            }
        }
    }
    loop {
        let mut changed = false;
        for c in &cands {
            let (ok, dsts): (bool, Vec<Val>) = match *c {
                Cand::Phi(b, k) => {
                    let p = &f.blocks[b].phis[k];
                    let ok = p.args.iter().all(|(_, s)| m.src_ok(s))
                        && p.args.iter().any(|(_, s)| m.is_mask_src(s));
                    (ok, vec![p.dst])
                }
                Cand::Op(b, k, op) => {
                    let inst = &f.blocks[b].insts[k];
                    // a mask operation reads at least one mask; scalar bits are allowed beside it
                    let ok = mask_srcs(inst, op).all(|s| m.bit_src_ok(&s))
                        && (mask_srcs(inst, op).any(|s| m.is_mask_src(&s))
                            || mask_srcs(inst, op).all(|s| m.src_ok(&s)));
                    (ok, mask_dsts(inst).collect())
                }
            };
            if !ok {
                for d in dsts {
                    changed |= m.masks.remove(&d);
                }
            }
        }
        if !changed {
            break;
        }
    }
    for blk in &f.blocks {
        for inst in &blk.insts {
            if let Some(op) = mask_op(&inst.instr) {
                let dsts: Vec<Val> = mask_dsts(inst).collect();
                if let Some(&(_, scc)) = inst.dsts.iter().find(|(r, _)| *r == Reg::Scc) {
                    if dsts.iter().all(|d| m.masks.contains(d)) && !dsts.is_empty() {
                        // saveexec: the SCC is about the new EXEC, the dwords after the old one
                        let from = if op.save { dsts.len() / 2 } else { 0 };
                        m.scc_of.insert(scc, dsts[from..].to_vec());
                    }
                }
            }
        }
    }
    uniform_masks(f, &mut m);
    m.uniform_vals = uniform::uniform_vals(f, &m);
    m.slots = uniform::slot_vgprs(f, &m);
    m.lanes = uniform::lane_uses(f, &m, &m.slots);
    for blk in &f.blocks {
        for inst in &blk.insts {
            if let Some(name) = cross_lane_use(&m, inst) {
                m.cross_lane.push(CrossLane {
                    at: inst.instr.at,
                    name,
                });
            }
        }
    }
    m
}

/// Fills `m.uniform`. Definitions dominate uses, so one pass in reverse post-order suffices;
/// a phi is never uniform (the lanes may reach it by different paths).
fn uniform_masks(f: &Func, m: &mut Masks) {
    for &(r, v) in &f.args {
        if is_exec(r) {
            m.uniform.insert(v, Some(v));
        }
    }
    for &b in &f.rpo {
        for inst in &f.blocks[b].insts {
            let execs: Vec<Src> = inst
                .imps
                .iter()
                .filter(|(r, _)| is_exec(*r))
                .map(|(_, s)| *s)
                .collect();
            let name = inst.instr.name();
            for &(r, v) in &inst.dsts {
                if is_exec(r) && m.masks.contains(&v) {
                    m.uniform.insert(v, Some(v));
                    let k = usize::from(r == Reg::ExecHi);
                    let old = if name.starts_with("v_cmpx")
                        || (name.contains("saveexec")
                            && (name.starts_with("s_and_") || name.starts_with("s_andn2_")))
                    {
                        execs.get(k).copied()
                    } else if name.starts_with("s_and_") || name.starts_with("s_andn2_") {
                        inst.srcs.first().and_then(|o| o.get(k)).copied()
                    } else {
                        None
                    };
                    if let Some(Src::Val(o)) = old {
                        if matches!(m.uniform.get(&o), Some(Some(r)) if *r == o) {
                            m.within.insert(v, o);
                        }
                    }
                }
            }
            let Some(op) = mask_op(&inst.instr) else {
                continue;
            };
            let dsts: Vec<Val> = mask_dsts(inst).collect();
            // a select on SCC is uniform only when the SCC is: a scalar compare, not a mask
            let scc_ok = !inst.imps.iter().any(|(r, s)| {
                *r == Reg::Scc && matches!(s, Src::Val(v) if m.scc_of.contains_key(v))
            });
            let n = if op.save { execs.len() } else { dsts.len() };
            // `wrexec`: no destination is the old EXEC; the new one is handled above
            for k in 0..if op.wr { 0 } else { n } {
                let d = dsts[k];
                if !m.masks.contains(&d) || m.uniform.contains_key(&d) || !scc_ok {
                    continue;
                }
                // dword k of the inputs; the EXEC every non-constant one is uniform over
                let ins: Vec<Src> = if op.save {
                    vec![execs[k]]
                } else {
                    inst.srcs.iter().filter_map(|o| o.get(k).copied()).collect()
                };
                let mut over: Option<Val> = None;
                let mut ok = true;
                for s in ins {
                    match s {
                        Src::Const(_) | Src::Undef => {}
                        Src::Val(v) => match m.uniform.get(&v) {
                            Some(None) => {}
                            Some(Some(r)) => match over {
                                Some(o) if narrows(m, o, *r) => {}
                                Some(o) if !narrows(m, *r, o) => ok = false,
                                _ => over = Some(*r),
                            },
                            _ => ok = false,
                        },
                        _ => ok = false,
                    }
                }
                if ok {
                    m.uniform.insert(d, over);
                }
            }
        }
    }
}

/// The name under which `inst` crosses lanes, when it does.
pub fn cross_lane_use(m: &Masks, inst: &Inst) -> Option<String> {
    let i = &inst.instr;
    if m.lanes.contains_key(&i.at) {
        return None;
    }
    if i.dpp.is_some() {
        return Some(format!("{} (dpp)", i.name()));
    }
    if cross_lane_by_nature(i) {
        return Some(i.name().to_string());
    }
    let reads_mask = inst.srcs.iter().flatten().any(|s| m.is_mask_src(s));
    if !reads_mask {
        return None;
    }
    if let Some(op) = mask_op(i) {
        // fine when it types: every input is a mask, a constant or a scalar bit
        return if mask_srcs(inst, op).all(|s| m.bit_src_ok(&s)) {
            None
        } else {
            Some(i.name().to_string())
        };
    }
    if is_vector(i) && takes_lane_mask(i.name()) {
        let n = inst.srcs.len();
        let others = inst.srcs[..n.saturating_sub(1)]
            .iter()
            .flatten()
            .any(|s| m.is_mask_src(s));
        if !others {
            return None;
        }
    }
    Some(i.name().to_string())
}

// ---------------------------------------------------------------------------------------
// 2. Branches

/// A wave-wide test of a mask, as a test of the lane's own bit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneTest {
    /// The mask dwords tested (a wave64 pair is one bit).
    pub vals: Vec<Val>,
    /// The branch is taken when the bit is set (`execnz`, `vccnz`, SCC 1) or clear.
    pub taken_when_set: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BranchKind {
    /// Same in every invocation.
    Uniform,
    Lane(LaneTest),
    /// A lane-wise test whose region defines scalars read after its join (a speculated scalar
    /// region): the wave runs the region when any lane enters it, so every invocation takes
    /// the branch on `any(bit)` and the region's scalar instructions run in all of them. Vector
    /// instructions inside stay guarded by `active`.
    Wave(LaneTest),
}

fn classify_branch(m: &Masks, op: &str, cond: &[(Reg, Src)]) -> BranchKind {
    let set = match op {
        "s_cbranch_execnz" | "s_cbranch_vccnz" | "s_cbranch_scc1" => true,
        "s_cbranch_execz" | "s_cbranch_vccz" | "s_cbranch_scc0" => false,
        _ => return BranchKind::Uniform,
    };
    let vals: Vec<Val> = if op.contains("scc") {
        match cond {
            [(_, Src::Val(v))] => m.scc_of.get(v).cloned().unwrap_or_default(),
            _ => vec![],
        }
    } else {
        let v: Vec<Val> = cond
            .iter()
            .filter_map(|(_, s)| match s {
                Src::Val(v) if m.is_mask(*v) => Some(*v),
                _ => None,
            })
            .collect();
        if v.len() == cond.len() {
            v
        } else {
            vec![]
        }
    };
    if vals.is_empty() {
        BranchKind::Uniform
    } else {
        BranchKind::Lane(LaneTest {
            vals,
            taken_when_set: set,
        })
    }
}

/// Immediate post-dominators (`None` when no path reaches an exit).
fn post_idoms(f: &Func) -> Vec<Option<BlockId>> {
    let n = f.blocks.len();
    let mut rsuccs: Vec<Vec<usize>> = vec![vec![]; n + 1];
    let mut rpreds: Vec<Vec<usize>> = vec![vec![]; n + 1];
    for (b, blk) in f.blocks.iter().enumerate() {
        let s = blk.succs();
        if s.is_empty() {
            rsuccs[n].push(b);
            rpreds[b].push(n);
        }
        for t in s {
            rsuccs[t].push(b);
            rpreds[b].push(t);
        }
    }
    let (_, idom) = dominators(n + 1, n, &rsuccs, &rpreds);
    (0..n).map(|b| idom[b]).collect()
}

/// The error for a scalar that the paths of a lane-wise branch leave different at their join.
///
/// A lane-wise branch whose region only has scalar instructions without side effects is
/// speculated instead (`BranchKind::Wave`).
fn divergent_scalar(
    f: &Func,
    m: &Masks,
    branches: &mut [Option<BranchKind>],
) -> Result<(), StructError> {
    let pdom = post_idoms(f);
    for d in 0..branches.len() {
        let Some(BranchKind::Lane(t)) = &branches[d] else {
            continue;
        };
        let t = t.clone();
        let exec_vals = |v: &Val| matches!(f.vals[v.0 as usize].reg, Reg::ExecLo | Reg::ExecHi);
        if t.vals
            .iter()
            .all(|v| m.uniform.contains_key(v) && !exec_vals(v))
        {
            continue;
        }
        let Some(j) = pdom[d].filter(|&j| j < f.blocks.len()) else {
            continue;
        };
        for p in &f.blocks[j].phis {
            let slot = matches!(p.reg, Reg::V(k) if m.slots.contains(&k));
            if (matches!(p.reg, Reg::V(_)) && !slot) || m.is_mask(p.dst) {
                continue;
            }
            let first = p.args.first().map(|(_, s)| *s);
            let differs = p.args.iter().any(|(_, s)| Some(*s) != first);
            let scc_bit = p.reg == Reg::Scc
                && p.args.iter().all(|(_, s)| match s {
                    Src::Val(v) => m.scc_of.contains_key(v),
                    Src::Const(_) | Src::Undef => true,
                    _ => false,
                });
            if differs && !scc_bit {
                if !region_is_speculable(f, d, j) {
                    return Err(StructError::DivergentScalar {
                        at: f.blocks[j].at,
                        reg: p.reg,
                        branch: f.blocks[d].at,
                    });
                }
                branches[d] = Some(BranchKind::Wave(t));
                break;
            }
        }
    }
    Ok(())
}

/// True when the scalar instructions between the branch of block `d` and its join `j` can run
/// in invocations that did not enter the region: no scalar store, atomic or message.
fn region_is_speculable(f: &Func, d: BlockId, j: BlockId) -> bool {
    let mut seen = HashSet::new();
    let mut stack = f.blocks[d].succs();
    while let Some(b) = stack.pop() {
        if b == j || b == d || !seen.insert(b) {
            continue;
        }
        let effect = f.blocks[b].insts.iter().any(|i| {
            let n = i.instr.name();
            ["s_store", "s_buffer_store", "s_atomic", "s_buffer_atomic"]
                .iter()
                .any(|p| n.starts_with(p))
                || matches!(n, "s_sendmsg" | "s_sendmsghalt" | "s_trap" | "s_dcache_wb")
        });
        if effect {
            return false;
        }
        stack.extend(f.blocks[b].succs());
    }
    true
}

// ---------------------------------------------------------------------------------------
// 3. Guards

/// How the emitter guards one instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Guard {
    /// Not a vector instruction: runs in every invocation.
    Scalar,
    /// A vector instruction EXEC is proved set for in this lane: no guard.
    Proved,
    /// A vector instruction: `if (active)` around its effect (ALU writes select the old value).
    Active,
}

type Facts = HashMap<Val, bool>;

fn truth(f: &Facts, s: &Src) -> Option<bool> {
    match s {
        Src::Const(0) => Some(false),
        Src::Const(u32::MAX) => Some(true),
        Src::Val(v) => f.get(v).copied(),
        _ => None,
    }
}

fn bit_fn(bit: Bit, a: Option<bool>, b: Option<bool>) -> Option<bool> {
    let not = |x: Option<bool>| x.map(|x| !x);
    let and = |a: Option<bool>, b: Option<bool>| match (a, b) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    };
    let or = |a: Option<bool>, b: Option<bool>| not(and(not(a), not(b)));
    match bit {
        Bit::And => and(a, b),
        Bit::Or => or(a, b),
        Bit::Andn2 => and(a, not(b)),
        Bit::Andn1 => and(not(a), b),
        Bit::Orn2 => or(a, not(b)),
        Bit::Nand => not(and(a, b)),
        Bit::Nor => not(or(a, b)),
        Bit::Xor => a.zip(b).map(|(a, b)| a ^ b),
        Bit::Xnor => a.zip(b).map(|(a, b)| !(a ^ b)),
        Bit::Not => not(a),
        Bit::Mov => a,
        Bit::Wqm => a.filter(|x| *x),
        Bit::Cselect => {
            if a == b {
                a
            } else {
                None
            }
        }
    }
}

/// What a mask operation proves about its results, given the facts before it.
fn transfer(inst: &Inst, facts: &mut Facts, m: &Masks) {
    let Some(op) = mask_op(&inst.instr) else {
        return;
    };
    let dsts: Vec<Val> = mask_dsts(inst).filter(|d| m.is_mask(*d)).collect();
    let src = |o: usize, k: usize| inst.srcs.get(o).and_then(|s| s.get(k));
    let mut out: Vec<(Val, Option<bool>)> = Vec::new();
    if op.save {
        let execs: Vec<Src> = inst
            .imps
            .iter()
            .filter(|(r, _)| is_exec(*r))
            .map(|(_, s)| *s)
            .collect();
        let n = execs.len();
        if dsts.len() != 2 * n {
            return;
        }
        for k in 0..n {
            let old = truth(facts, &execs[k]);
            let new = bit_fn(op.bit, src(0, k).and_then(|s| truth(facts, s)), old);
            if op.wr {
                out.push((dsts[k], new));
            } else {
                out.push((dsts[k], old));
            }
            out.push((dsts[n + k], new));
        }
    } else {
        for (k, d) in dsts.iter().enumerate() {
            let a = src(0, k).and_then(|s| truth(facts, s));
            let b = src(1, k).and_then(|s| truth(facts, s));
            out.push((*d, bit_fn(op.bit, a, b)));
        }
    }
    for (d, t) in out {
        if let Some(t) = t {
            facts.insert(d, t);
        }
    }
}

/// What the branch of `p` tells about the bit it tests on the edge to `to`.
fn edge_facts(
    f: &Func,
    branches: &[Option<BranchKind>],
    p: BlockId,
    to: BlockId,
) -> Vec<(Val, bool)> {
    let (Some(BranchKind::Lane(t)), Term::Branch { taken, fall, .. }) =
        (&branches[p], &f.blocks[p].term)
    else {
        return vec![];
    };
    if taken == fall {
        return vec![];
    }
    let bit = if to == *taken {
        t.taken_when_set
    } else {
        !t.taken_when_set
    };
    t.vals.iter().map(|v| (*v, bit)).collect()
}

fn guards(f: &Func, m: &Masks, branches: &[Option<BranchKind>]) -> Vec<Vec<Guard>> {
    let n = f.blocks.len();
    let mut start = Facts::new();
    for &(r, v) in &f.args {
        if is_exec(r) {
            start.insert(v, true);
        }
    }
    let mut out: Vec<Option<Facts>> = vec![None; n];
    let mut record: Vec<Vec<Guard>> = vec![vec![]; n];
    for _ in 0..1000 {
        let mut changed = false;
        for &b in &f.rpo {
            let blk = &f.blocks[b];
            // the facts on entry: what every computed predecessor edge agrees on
            let mut cur: Option<Facts> = (b == f.entry).then(|| start.clone());
            for &p in &blk.preds {
                let Some(po) = &out[p] else { continue };
                let mut e = po.clone();
                e.extend(edge_facts(f, branches, p, b));
                cur = Some(match cur {
                    None => e,
                    Some(mut c) => {
                        c.retain(|k, v| e.get(k) == Some(v));
                        c
                    }
                });
            }
            let Some(mut facts) = cur else { continue };
            for p in &blk.phis {
                if !m.is_mask(p.dst) {
                    continue;
                }
                let mut agreed: Option<Option<bool>> = None;
                for (pb, s) in &p.args {
                    let Some(po) = &out[*pb] else { continue };
                    let mut e = po.clone();
                    e.extend(edge_facts(f, branches, *pb, b));
                    let t = truth(&e, s);
                    agreed = Some(match agreed {
                        None => t,
                        Some(a) if a == t => a,
                        Some(_) => None,
                    });
                }
                if let Some(Some(t)) = agreed {
                    facts.insert(p.dst, t);
                }
            }
            let mut gs = Vec::with_capacity(blk.insts.len());
            for inst in &blk.insts {
                gs.push(if !is_vector(&inst.instr) {
                    Guard::Scalar
                } else if inst
                    .imps
                    .iter()
                    .filter(|(r, _)| is_exec(*r))
                    .all(|(_, s)| truth(&facts, s) == Some(true))
                {
                    Guard::Proved
                } else {
                    Guard::Active
                });
                transfer(inst, &mut facts, m);
            }
            record[b] = gs;
            if out[b].as_ref() != Some(&facts) {
                out[b] = Some(facts);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    record
}

// ---------------------------------------------------------------------------------------
// 4. Pixel discard

/// A pixel discard: `s_andn2 live, live, exec` whose SCC decides a branch to the exit of the
/// whole wave. The emitter demotes the lane at this instruction (`OpDemoteToHelperInvocation`)
/// and the exit becomes a no-op: the lane keeps running as a helper.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Demote {
    pub block: BlockId,
    /// Index of the `s_andn2` in `block.insts`.
    pub inst: usize,
    /// The block of the branch that tests the SCC.
    pub branch: BlockId,
    /// The successor that leaves the program without doing any work.
    pub exit: BlockId,
}

/// True when entering `b` only runs scalar work and a `done` export to `null`, then ends.
fn kill_exit(f: &Func, mut b: BlockId) -> bool {
    for _ in 0..8 {
        let blk = &f.blocks[b];
        let mut null_done = false;
        for inst in &blk.insts {
            let i = &inst.instr;
            if i.enc == Enc::Exp {
                if i.exp.target == 9 && i.exp.done {
                    null_done = true;
                } else {
                    return false;
                }
            } else if is_vector(i) {
                return false;
            }
        }
        match blk.term {
            Term::End => return null_done,
            Term::Jump(next) if !null_done => b = next,
            _ => return false,
        }
    }
    false
}

fn demotes(f: &Func) -> Vec<Demote> {
    let mut v = Vec::new();
    for (b, blk) in f.blocks.iter().enumerate() {
        let Term::Branch {
            op,
            cond,
            taken,
            fall,
        } = &blk.term
        else {
            continue;
        };
        if !op.starts_with("s_cbranch_scc") {
            continue;
        }
        let [(_, Src::Val(scc))] = &cond[..] else {
            continue;
        };
        let ValDef::Inst { block, index } = f.vals[scc.0 as usize].def else {
            continue;
        };
        let i = &f.blocks[block].insts[index].instr;
        if !matches!(i.name(), "s_andn2_b32" | "s_andn2_b64") {
            continue;
        }
        let ops = &i.operands;
        let same_exec = matches!(
            (ops.first(), ops.get(1), ops.get(2)),
            (
                Some(Operand::Reg { code: d, .. }),
                Some(Operand::Reg { code: s, .. }),
                Some(Operand::Reg { code: e, .. }),
            ) if d == s && (*e == 126 || *e == 127)
        );
        if !same_exec {
            continue;
        }
        let exit = [*taken, *fall].into_iter().find(|&s| kill_exit(f, s));
        if let Some(exit) = exit {
            v.push(Demote {
                block,
                inst: index,
                branch: b,
                exit,
            });
        }
    }
    v
}

// ---------------------------------------------------------------------------------------
// 5. The structured tree

/// Where a `Br` goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Target {
    pub block: BlockId,
    pub kind: BrKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BrKind {
    /// Leave the `Block` labelled `block`: continue with that block's code.
    Exit,
    /// Go back to the header of the `Loop` labelled `block`.
    Continue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    /// The instructions of a block (its terminator is the node that follows).
    Code(BlockId),
    /// Forward merge: `Br` to `label` lands after this node, where `label`'s code follows.
    /// `forwards`: the targets of the multi-level `Br`s that leave through this merge, each
    /// re-tested (`target == t`) right after it.
    Block {
        label: BlockId,
        body: Vec<Node>,
        forwards: Vec<Target>,
    },
    Loop {
        label: BlockId,
        body: Vec<Node>,
        forwards: Vec<Target>,
    },
    /// The branch ending block `head`.
    If {
        head: BlockId,
        taken: Vec<Node>,
        fall: Vec<Node>,
    },
    /// `depth`: how many `Block` / `Loop` constructs the jump leaves before the one it
    /// targets (0: the innermost, a plain branch; k > 0: the emitter sets `target`, breaks
    /// out of the innermost and each of the k constructs re-tests it).
    Br {
        to: Target,
        depth: usize,
    },
    Return,
}

/// The nodes as text, for tests and dumps: `b3` code, `block 5 {..}`, `loop 1 {..}`,
/// `if {taken | fall}`, `br 5` / `continue 1` (`^k` when it leaves k levels), `ret`.
pub fn render(nodes: &[Node]) -> String {
    fn fw(f: &[Target]) -> String {
        if f.is_empty() {
            return String::new();
        }
        let v: Vec<String> = f
            .iter()
            .map(|t| match t.kind {
                BrKind::Exit => format!("br {}", t.block),
                BrKind::Continue => format!("continue {}", t.block),
            })
            .collect();
        format!(" fwd[{}]", v.join(","))
    }
    nodes
        .iter()
        .map(|n| match n {
            Node::Code(b) => format!("b{b}"),
            Node::Block {
                label,
                body,
                forwards,
            } => format!("block {label} {{{}}}{}", render(body), fw(forwards)),
            Node::Loop {
                label,
                body,
                forwards,
            } => format!("loop {label} {{{}}}{}", render(body), fw(forwards)),
            Node::If { taken, fall, .. } => {
                format!("if {{{} | {}}}", render(taken), render(fall))
            }
            Node::Br { to, depth } => {
                let w = match to.kind {
                    BrKind::Exit => "br",
                    BrKind::Continue => "continue",
                };
                if *depth == 0 {
                    format!("{w} {}", to.block)
                } else {
                    format!("{w} {}^{depth}", to.block)
                }
            }
            Node::Return => "ret".to_string(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Clone, Copy)]
struct Frame {
    id: usize,
    target: Target,
}

struct Tree<'a> {
    f: &'a Func,
    pos: Vec<usize>,
    /// Merge nodes among the children of each node of the dominator tree, outermost first.
    merge_children: Vec<Vec<BlockId>>,
    merge: Vec<bool>,
    header: Vec<bool>,
    ctx: Vec<Frame>,
    next_id: usize,
    forwards: HashMap<usize, BTreeSet<Target>>,
}

impl Tree<'_> {
    fn do_tree(&mut self, x: BlockId) -> Result<Vec<Node>, StructError> {
        let ys = self.merge_children[x].clone();
        if self.header[x] {
            let id = self.fresh();
            self.ctx.push(Frame {
                id,
                target: Target {
                    block: x,
                    kind: BrKind::Continue,
                },
            });
            let body = self.node_within(x, &ys);
            self.ctx.pop();
            let forwards = self.take(id);
            Ok(vec![Node::Loop {
                label: x,
                body: body?,
                forwards,
            }])
        } else {
            self.node_within(x, &ys)
        }
    }

    fn fresh(&mut self) -> usize {
        self.next_id += 1;
        self.next_id
    }

    fn take(&mut self, id: usize) -> Vec<Target> {
        self.forwards
            .remove(&id)
            .map(|s| s.into_iter().collect())
            .unwrap_or_default()
    }

    fn node_within(&mut self, x: BlockId, ys: &[BlockId]) -> Result<Vec<Node>, StructError> {
        let Some((&y, rest)) = ys.split_first() else {
            return self.code(x);
        };
        let id = self.fresh();
        self.ctx.push(Frame {
            id,
            target: Target {
                block: y,
                kind: BrKind::Exit,
            },
        });
        let inner = self.node_within(x, rest);
        self.ctx.pop();
        let forwards = self.take(id);
        let mut out = vec![Node::Block {
            label: y,
            body: inner?,
            forwards,
        }];
        out.extend(self.do_tree(y)?);
        Ok(out)
    }

    fn code(&mut self, x: BlockId) -> Result<Vec<Node>, StructError> {
        let mut out = vec![Node::Code(x)];
        match &self.f.blocks[x].term {
            Term::End => out.push(Node::Return),
            Term::Jump(t) => out.extend(self.do_branch(x, *t)?),
            Term::Branch { taken, fall, .. } => {
                let (taken, fall) = (*taken, *fall);
                if taken == fall {
                    out.extend(self.do_branch(x, taken)?);
                } else {
                    out.push(Node::If {
                        head: x,
                        taken: self.do_branch(x, taken)?,
                        fall: self.do_branch(x, fall)?,
                    });
                }
            }
        }
        Ok(out)
    }

    fn do_branch(&mut self, src: BlockId, tgt: BlockId) -> Result<Vec<Node>, StructError> {
        let kind = if self.pos[tgt] <= self.pos[src] {
            BrKind::Continue
        } else if self.merge[tgt] {
            BrKind::Exit
        } else {
            return self.do_tree(tgt);
        };
        let to = Target { block: tgt, kind };
        let Some(depth) = self.ctx.iter().rev().position(|fr| fr.target == to) else {
            return Err(StructError::Malformed(
                "branch target has no enclosing label",
            ));
        };
        let crossed: Vec<usize> = self.ctx.iter().rev().take(depth).map(|fr| fr.id).collect();
        for id in crossed {
            self.forwards.entry(id).or_default().insert(to);
        }
        Ok(vec![Node::Br { to, depth }])
    }
}

/// Checks that every retreating edge goes to a block that dominates its source.
fn check_reducible(f: &Func, pos: &[usize]) -> Result<(), StructError> {
    for (u, blk) in f.blocks.iter().enumerate() {
        for v in blk.succs() {
            if pos[v] <= pos[u] {
                let mut d = u;
                while d != v {
                    match f.idom[d] {
                        Some(p) => d = p,
                        None => return Err(StructError::Irreducible { at: f.blocks[v].at }),
                    }
                }
            }
        }
    }
    Ok(())
}

fn build_tree(f: &Func) -> Result<Vec<Node>, StructError> {
    let n = f.blocks.len();
    let mut pos = vec![usize::MAX; n];
    for (k, &b) in f.rpo.iter().enumerate() {
        pos[b] = k;
    }
    check_reducible(f, &pos)?;
    let mut forward_preds = vec![0usize; n];
    let mut header = vec![false; n];
    for (u, blk) in f.blocks.iter().enumerate() {
        let mut seen = BTreeSet::new();
        for v in blk.succs() {
            if !seen.insert(v) {
                continue;
            }
            if pos[v] <= pos[u] {
                header[v] = true;
            } else {
                forward_preds[v] += 1;
            }
        }
    }
    let merge: Vec<bool> = forward_preds.iter().map(|&c| c >= 2).collect();
    let mut merge_children = vec![Vec::new(); n];
    for (b, d) in f.idom.iter().enumerate() {
        if let Some(d) = d {
            if merge[b] {
                merge_children[*d].push(b);
            }
        }
    }
    for c in &mut merge_children {
        c.sort_by_key(|&b| std::cmp::Reverse(pos[b]));
    }
    let mut t = Tree {
        f,
        pos,
        merge_children,
        merge,
        header,
        ctx: Vec::new(),
        next_id: 0,
        forwards: HashMap::new(),
    };
    t.do_tree(f.entry)
}

// ---------------------------------------------------------------------------------------

/// The structured form of a program.
#[derive(Clone, Debug)]
pub struct Structured {
    pub masks: Masks,
    /// By block: how its `s_cbranch_*` is tested (`None` for jumps and `s_endpgm`).
    pub branches: Vec<Option<BranchKind>>,
    /// By block, by instruction.
    pub guards: Vec<Vec<Guard>>,
    pub demotes: Vec<Demote>,
    pub tree: Vec<Node>,
}

/// Structures `f`: the first error is a cross-lane use (until the wave chunk), then a
/// divergent scalar, then irreducible flow.
pub fn structure(f: &Func) -> Result<Structured, StructError> {
    structure_with(f, false)
}

/// Like `structure`; `allow_cross_lane` skips the cross-lane check (to see what else stops a
/// program, for the corpus report). The result is not usable for emission then.
pub fn structure_with(f: &Func, allow_cross_lane: bool) -> Result<Structured, StructError> {
    let masks = type_masks(f);
    if !allow_cross_lane {
        if let Some(c) = masks.cross_lane.first() {
            return Err(StructError::CrossLane(c.clone()));
        }
    }
    let mut branches: Vec<Option<BranchKind>> = f
        .blocks
        .iter()
        .map(|b| match &b.term {
            Term::Branch { op, cond, .. } => Some(classify_branch(&masks, op, cond)),
            _ => None,
        })
        .collect();
    divergent_scalar(f, &masks, &mut branches)?;
    let guards = guards(f, &masks, &branches);
    let demotes = demotes(f);
    let tree = build_tree(f)?;
    Ok(Structured {
        masks,
        branches,
        guards,
        demotes,
        tree,
    })
}

#[cfg(test)]
mod tests;
