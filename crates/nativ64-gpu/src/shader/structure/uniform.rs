//! Chunk 7b (`docs/shaders.md`, Chunk 7 design, items 1-4): which values are the same in every
//! active lane, the VGPRs that only spill SGPRs into lanes, the waterfall `v_readfirstlane`,
//! and the console lane index. Every use this module vets leaves the cross-lane list and gets a
//! `LaneUse` the emitter reads.

use std::collections::{HashMap, HashSet};

use super::{classify_branch, is_exec, post_idoms, takes_lane_mask, BranchKind, Masks};
use crate::shader::gfx10::Enc;
use crate::shader::ir::{BlockId, Func, Inst, Reg, Src, Term, Val, ValDef};

/// What the emitter does with a wave instruction the structurer accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneUse {
    /// `v_readfirstlane` / `v_readlane` of a value every lane holds: a copy.
    Copy,
    /// `v_readfirstlane` in a waterfall loop: the lane's own value (the compare it feeds is
    /// then true, so the loop body runs once per lane). The resource index it feeds is
    /// `NonUniform`.
    Waterfall,
    /// `v_writelane v, s, lane` on a spill slot: a store to the scalar variable of
    /// `(vgpr, lane)`.
    SlotWrite { vgpr: u16, lane: u8 },
    /// `v_readlane s, v, lane` on a spill slot.
    SlotRead { vgpr: u16, lane: u8 },
    /// `v_mbcnt_lo(-1, x)` (`hi: false`) / `v_mbcnt_hi(-1, x)`: the console lane index.
    LaneIndex { hi: bool },
    /// `ds_swizzle_b32` in quad mode, or a DPP `quad_perm` / `dpp8` that stays in the quad of
    /// 4 lanes: lane `k` of each quad reads the source of lane `sel[k]`.
    Quad { sel: [u8; 4] },
    /// `s_bcnt1_i32` of a lane mask: the number of invocations (of the host's active set)
    /// whose bit is set.
    BallotCount,
    /// `s_ff1_i32` of a lane mask: the host lane of the first invocation whose bit is set. A
    /// host lane number, so it is only ever read as the lane of a `ReadHostLane`.
    FirstSet,
    /// `v_readlane` of the lane a `FirstSet` found: that invocation's value, in every one.
    ReadHostLane,
    /// `v_mbcnt_lo(mask, x)` is `x` plus the number of invocations before this one whose bit
    /// is set; `v_mbcnt_hi(mask, x)` right after it adds nothing (the count already covers the
    /// whole wave).
    Rank { hi: bool },
    /// `s_cmp_eq/lg` of a lane mask with zero: SCC is "no invocation has its bit set"
    /// (`eq`) or "some invocation does".
    MaskZero { eq: bool },
    /// `s_cmp_eq/lg` of two lane masks: SCC is "every invocation has the same two bits" or
    /// its negation.
    MasksEqual { eq: bool },
}

/// How many times each value is read, and how many of those reads are the lane operand of a
/// `v_readlane`.
#[derive(Default)]
struct Reads {
    all: HashMap<Val, usize>,
    as_lane: HashMap<Val, usize>,
}

fn count_reads(f: &Func) -> Reads {
    let mut r = Reads::default();
    let mut add = |s: &Src| {
        if let Src::Val(v) = s {
            *r.all.entry(*v).or_default() += 1;
        }
    };
    for blk in &f.blocks {
        for p in &blk.phis {
            p.args.iter().for_each(|(_, s)| add(s));
        }
        for inst in &blk.insts {
            inst.srcs.iter().flatten().for_each(&mut add);
            inst.imps.iter().for_each(|(_, s)| add(s));
        }
        if let Term::Branch { cond, .. } = &blk.term {
            cond.iter().for_each(|(_, s)| add(s));
        }
    }
    for blk in &f.blocks {
        for inst in &blk.insts {
            if inst.instr.name() == "v_readlane_b32" {
                if let Some(Src::Val(v)) = inst.srcs.get(1).and_then(|s| s.first()) {
                    *r.as_lane.entry(*v).or_default() += 1;
                }
            }
        }
    }
    r
}

/// Operand `k` of `inst` is made of lane mask dwords (and zeros): what the ballot reads.
fn mask_operand(m: &Masks, inst: &Inst, k: usize) -> bool {
    let Some(d) = inst.srcs.get(k) else {
        return false;
    };
    let is_mask = |s: &Src| matches!(s, Src::Val(x) if m.masks.contains(x));
    d.iter().any(is_mask)
        && d.iter()
            .all(|s| is_mask(s) || matches!(s, Src::Const(0) | Src::Undef))
}

/// The quad permutation of a `ds_swizzle_b32` or a DPP instruction, when it stays inside the
/// quad (the same on every quad, whatever the console lane of the quad).
fn quad_sel(i: &crate::shader::gfx10::Instr) -> Option<[u8; 4]> {
    use crate::shader::gfx10::DppKind;
    let pick = |ctrl: u32| [0, 1, 2, 3].map(|k| (ctrl >> (2 * k) & 3) as u8);
    if i.name() == "ds_swizzle_b32" {
        let off = i.mem.offset as u32 & 0xffff;
        if off & 0x8000 != 0 {
            return Some(pick(off));
        }
        // bitmask mode within groups of 32: quad-local when the bits above the quad pass
        // through untouched
        let (and, or, xor) = (off & 31, off >> 5 & 31, off >> 10 & 31);
        if and & 0x1c != 0x1c || (or | xor) & 0x1c != 0 {
            return None;
        }
        return Some([0, 1, 2, 3].map(|k| (((k & and) | or) ^ xor) as u8 & 3));
    }
    let d = i.dpp.as_ref()?;
    if !matches!(i.enc, Enc::Vop1 | Enc::Vop2) {
        return None;
    }
    match d.kind {
        DppKind::Dpp16 => {
            // no per-row / per-bank disabling, no source modifiers
            let plain = d.row_mask == 0xf
                && d.bank_mask == 0xf
                && !(d.src0_neg || d.src0_abs || d.src1_neg || d.src1_abs);
            (plain && d.ctrl < 0x100).then(|| pick(u32::from(d.ctrl)))
        }
        DppKind::Dpp8 | DppKind::Dpp8Fi => {
            // each quad of the group of 8 reads inside itself, the same way
            let q = |h: usize| [0, 1, 2, 3].map(|k| d.sel[4 * h + k].wrapping_sub(4 * h as u8));
            (q(0) == q(1) && q(0).iter().all(|&s| s < 4)).then(|| q(0))
        }
    }
}

/// `a` is `b` or narrows it (`Masks::within`).
pub(super) fn narrows(m: &Masks, mut a: Val, b: Val) -> bool {
    loop {
        if a == b {
            return true;
        }
        match m.within.get(&a) {
            Some(&n) => a = n,
            None => return false,
        }
    }
}

/// The blocks whose phis may merge values the lanes of a lane-wise branch chose differently
/// (`.0`: the region of the branch up to and including its join), and the blocks whose
/// instructions may hold different values per lane when read after the region (`.1`: the
/// region of a branch inside a loop, whose lanes leave at different iterations).
fn divergent_blocks(f: &Func, m: &Masks) -> (HashSet<BlockId>, HashSet<BlockId>) {
    let (mut phis, mut all) = (HashSet::new(), HashSet::new());
    let pdom = post_idoms(f);
    for (d, blk) in f.blocks.iter().enumerate() {
        let Term::Branch { op, cond, .. } = &blk.term else {
            continue;
        };
        let BranchKind::Lane(t) = classify_branch(m, op, cond) else {
            continue;
        };
        let exec_val = |v: &Val| matches!(f.vals[v.0 as usize].reg, Reg::ExecLo | Reg::ExecHi);
        if t.vals
            .iter()
            .all(|v| m.uniform.contains_key(v) && !exec_val(v))
        {
            continue;
        }
        let join = pdom[d].filter(|&j| j < f.blocks.len());
        let mut seen = HashSet::new();
        let mut stack = blk.succs();
        while let Some(b) = stack.pop() {
            if Some(b) == join || !seen.insert(b) {
                continue;
            }
            stack.extend(f.blocks[b].succs());
        }
        let cyclic = seen.contains(&d);
        phis.extend(seen.iter().copied());
        phis.extend(join);
        if cyclic {
            all.extend(seen);
            all.insert(d);
            all.extend(join);
        }
    }
    (phis, all)
}

fn is_store(inst: &Inst) -> bool {
    let n = inst.instr.name();
    matches!(
        inst.instr.enc,
        Enc::Mubuf | Enc::Mtbuf | Enc::Mimg | Enc::Flat | Enc::Ds
    ) && (n.contains("store") || n.contains("atomic") || n.starts_with("image_store"))
}

/// A vector load whose result is the same for every lane that gives it the same address and
/// resource: buffer and global loads, and image loads and fetches without derivatives.
fn is_pure_load(inst: &Inst) -> bool {
    let n = inst.instr.name();
    match inst.instr.enc {
        Enc::Mubuf => n.starts_with("buffer_load"),
        Enc::Mtbuf => n.starts_with("tbuffer_load"),
        Enc::Flat => n.starts_with("global_load"),
        Enc::Mimg => {
            if n.starts_with("image_load") || n == "image_get_resinfo" {
                return true;
            }
            // a sample is derivative-free with an explicit level (`_l`, `_lz`) or gradients
            // (`_d`) and no implicit-derivative modifier (`_b`, `_cl`)
            let Some(rest) = n
                .strip_prefix("image_sample_")
                .or_else(|| n.strip_prefix("image_gather4_"))
            else {
                return false;
            };
            let toks: Vec<&str> = rest.split('_').collect();
            let explicit = toks.iter().any(|t| matches!(*t, "l" | "lz" | "d"));
            explicit && !toks.iter().any(|t| matches!(*t, "b" | "cl" | "bias"))
        }
        _ => false,
    }
}

/// Where a value is guaranteed the same in every lane: all lanes of the wave, or the lanes an
/// EXEC value holds (what is left of the lanes outside it is an older value).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dom {
    Full,
    Exec(Val),
}

/// The lanes of `e` are all among those of `d`.
pub(super) fn covers(m: &Masks, d: Dom, e: Dom) -> bool {
    match (d, e) {
        (Dom::Full, _) => true,
        (Dom::Exec(s), Dom::Exec(x)) => narrows(m, x, s),
        (Dom::Exec(_), Dom::Full) => false,
    }
}

/// The smaller of two domains when one holds the other.
fn meet(m: &Masks, a: Dom, b: Dom) -> Option<Dom> {
    if covers(m, a, b) {
        Some(b)
    } else if covers(m, b, a) {
        Some(a)
    } else {
        None
    }
}

/// The lanes an instruction runs for: all of them under the entry EXEC or a constant all-ones
/// one, else the lanes of its EXEC value. `None` when it has no EXEC (or a partial constant).
pub(super) fn exec_dom(f: &Func, inst: &Inst) -> Option<Dom> {
    match inst.imps.iter().find(|(r, _)| *r == Reg::ExecLo)?.1 {
        Src::Const(c) => (c == u32::MAX).then_some(Dom::Full),
        Src::Val(x) => Some(match f.vals[x.0 as usize].def {
            ValDef::Arg => Dom::Full,
            _ => Dom::Exec(x),
        }),
        _ => None,
    }
}

/// The mask `s` is the same in every lane of `e`.
fn mask_uniform_over(f: &Func, m: &Masks, e: Dom, s: &Src) -> bool {
    match s {
        Src::Const(c) => *c == 0 || *c == u32::MAX,
        Src::Undef => true,
        Src::Val(x) => match m.uniform.get(x) {
            Some(None) => true,
            Some(Some(r)) => {
                let d = match f.vals[r.0 as usize].def {
                    ValDef::Arg => Dom::Full,
                    _ => Dom::Exec(*r),
                };
                covers(m, d, e)
            }
            None => false,
        },
        _ => false,
    }
}

/// Where each VGPR value is the same in every lane (by `Val`; `None`: not uniform): the
/// optimistic fixpoint of the rules of design item 1. A value that is not a VGPR is uniform
/// (scalar code is, once speculation is in) unless it is a mask.
pub(super) fn uniform_vals(f: &Func, m: &Masks) -> Vec<Option<Dom>> {
    let (div_phis, div_all) = divergent_blocks(f, m);
    let mut uni: Vec<Option<Dom>> = f
        .vals
        .iter()
        .enumerate()
        .map(|(n, info)| {
            (!m.masks.contains(&Val(n as u32)) && info.reg != Reg::Null).then_some(Dom::Full)
        })
        .collect();
    // stores before each load, per block
    let mut store_before: HashSet<(BlockId, usize)> = HashSet::new();
    for (b, blk) in f.blocks.iter().enumerate() {
        let mut stored = false;
        for (k, inst) in blk.insts.iter().enumerate() {
            if stored {
                store_before.insert((b, k));
            }
            stored |= is_store(inst);
        }
    }
    loop {
        let mut changed = false;
        for (n, info) in f.vals.iter().enumerate() {
            let Some(cur) = uni[n] else { continue };
            if !matches!(info.reg, Reg::V(_)) {
                continue;
            }
            let new = match info.def {
                ValDef::Arg => f.uniform_args.contains(&info.reg).then_some(Dom::Full),
                ValDef::Phi { block, index } => {
                    if div_phis.contains(&block) || div_all.contains(&block) {
                        None
                    } else {
                        let mut acc = Some(Dom::Full);
                        for (_, s) in &f.blocks[block].phis[index].args {
                            if let Src::Val(x) = s {
                                acc = match (acc, uni[x.0 as usize]) {
                                    (Some(a), Some(d)) => meet(m, a, d),
                                    _ => None,
                                };
                            }
                        }
                        acc
                    }
                }
                ValDef::Inst { block, index } => {
                    let inst = &f.blocks[block].insts[index];
                    inst_uniform(
                        f,
                        m,
                        inst,
                        &uni,
                        !div_all.contains(&block),
                        store_before.contains(&(block, index)),
                    )
                }
            };
            let new = new.and_then(|d| meet(m, cur, d));
            if new != Some(cur) {
                uni[n] = new;
                changed = true;
            }
        }
        if !changed {
            return uni;
        }
    }
}

fn inst_uniform(
    f: &Func,
    m: &Masks,
    inst: &Inst,
    uni: &[Option<Dom>],
    region_ok: bool,
    after_store: bool,
) -> Option<Dom> {
    let i = &inst.instr;
    let n = i.name();
    if !region_ok || i.dpp.is_some() {
        return None;
    }
    let alu = matches!(
        i.enc,
        Enc::Vop1 | Enc::Vop2 | Enc::Vop3 | Enc::Vop3p | Enc::Vopc
    );
    let load = is_pure_load(inst);
    if (load && after_store)
        || !(alu || load)
        || n.starts_with("v_mbcnt")
        || n.starts_with("v_permlane")
        || n.starts_with("v_writelane")
        || n.starts_with("v_readlane")
    {
        return None;
    }
    // the result holds on the lanes the instruction runs for, whatever the others keep
    let e = exec_dom(f, inst)?;
    let mask_pos = takes_lane_mask(n).then(|| inst.srcs.len().saturating_sub(1));
    for (k, s) in inst.srcs.iter().enumerate() {
        for d in s {
            let Src::Val(x) = d else { continue };
            if m.masks.contains(x) {
                if mask_pos != Some(k) || !mask_uniform_over(f, m, e, d) {
                    return None;
                }
            } else if !uni[x.0 as usize].is_some_and(|d| covers(m, d, e)) {
                return None;
            }
        }
    }
    for (r, s) in &inst.imps {
        if let Src::Val(x) = s {
            if m.masks.contains(x) && !is_exec(*r) && !mask_uniform_over(f, m, e, s) {
                return None;
            }
        }
    }
    Some(e)
}

/// The VGPRs that only spill SGPRs: every definition is `v_writelane` at a constant lane and
/// every read is `v_readlane` at a constant lane (and nothing else names them). Also the slots
/// (`(vgpr, lane)`) that hold a lane mask: spilled EXEC and VCC dwords, which come back as masks
/// (a write of a mask and of a plain dword to one slot is refused).
pub(super) fn slot_vgprs(f: &Func, m: &Masks) -> (HashSet<u16>, HashSet<(u16, u8)>) {
    let (mut good, mut bad) = (HashSet::new(), HashSet::new());
    let mut kinds: HashMap<(u16, u8), bool> = HashMap::new();
    let vgpr = |r: Reg| match r {
        Reg::V(k) => Some(k),
        _ => None,
    };
    let reg_of = |x: &Val| f.vals[x.0 as usize].reg;
    let lane_of = |inst: &Inst, k: usize| match inst.srcs.get(k).and_then(|s| s.first()) {
        Some(Src::Const(c)) => Some((c & if f.wave32 { 31 } else { 63 }) as u8),
        _ => None,
    };
    for &(r, _) in &f.args {
        bad.extend(vgpr(r));
    }
    for blk in &f.blocks {
        for inst in &blk.insts {
            let n = inst.instr.name();
            if n == "v_writelane_b32" && lane_of(inst, 1).is_some() && inst.dsts.len() == 1 {
                // the scalar spilled is a plain dword or a lane mask, not a VGPR
                let src = inst.srcs[0].first();
                let is_vgpr = matches!(src, Some(Src::Val(x)) if vgpr(reg_of(x)).is_some());
                let is_mask = matches!(src, Some(Src::Val(x)) if m.masks.contains(x));
                let k = vgpr(inst.dsts[0].0);
                match (k, is_vgpr) {
                    (Some(k), false) => {
                        good.insert(k);
                        let lane = lane_of(inst, 1).unwrap_or(0);
                        if *kinds.entry((k, lane)).or_insert(is_mask) != is_mask {
                            bad.insert(k);
                        }
                    }
                    _ => bad.extend(k),
                }
                // the old value of the slot is an implicit read of the same register
                for (r, _) in &inst.imps {
                    if vgpr(*r) != k {
                        bad.extend(vgpr(*r));
                    }
                }
                continue;
            }
            if n == "v_readlane_b32" && lane_of(inst, 1).is_some() {
                if let Some(Src::Val(x)) = inst.srcs[0].first() {
                    good.extend(vgpr(reg_of(x)));
                }
                bad.extend(inst.imps.iter().filter_map(|(r, _)| vgpr(*r)));
                bad.extend(inst.dsts.iter().filter_map(|d| vgpr(d.0)));
                continue;
            }
            bad.extend(inst.dsts.iter().filter_map(|d| vgpr(d.0)));
            bad.extend(inst.imps.iter().filter_map(|d| vgpr(d.0)));
            for s in inst.srcs.iter().flatten() {
                if let Src::Val(x) = s {
                    bad.extend(vgpr(reg_of(x)));
                }
            }
        }
    }
    let slots: HashSet<u16> = good.difference(&bad).copied().collect();
    let masks = kinds
        .into_iter()
        .filter(|((k, _), mask)| *mask && slots.contains(k))
        .map(|(key, _)| key)
        .collect();
    (slots, masks)
}

/// The destinations of the `v_readlane` instructions that read a mask slot: lane masks too.
pub(super) fn mask_slot_reads(f: &Func, m: &Masks) -> HashSet<Val> {
    let mut out = HashSet::new();
    for blk in &f.blocks {
        for inst in &blk.insts {
            if let Some(LaneUse::SlotRead { vgpr, lane }) = m.lanes.get(&inst.instr.at) {
                if m.slot_masks.contains(&(*vgpr, *lane)) {
                    out.extend(inst.dsts.first().map(|d| d.1));
                }
            }
        }
    }
    out
}

/// The meaning of a cross-lane-by-nature instruction under items 1-4, when it has one.
pub(super) fn lane_use(
    f: &Func,
    m: &Masks,
    slots: &HashSet<u16>,
    inst: &Inst,
    eq_cmps: &[Vec<Src>],
    first_set: &HashSet<Val>,
) -> Option<LaneUse> {
    let i = &inst.instr;
    let n = i.name();
    if let Some(sel) = quad_sel(i) {
        return Some(LaneUse::Quad { sel });
    }
    if i.dpp.is_some() {
        return None;
    }
    let src = |k: usize| inst.srcs.get(k).and_then(|v| v.first()).copied();
    let lane_of = |s: Option<Src>| match s {
        Some(Src::Const(c)) => Some((c & if f.wave32 { 31 } else { 63 }) as u8),
        _ => None,
    };
    // the value is the same in every lane this instruction runs for
    let uniform = |s: Option<Src>| match s {
        // `v_readlane` has no EXEC: it reads the lane whatever is active, so every lane must
        // hold the value
        Some(Src::Val(v)) => match (m.uniform_vals[v.0 as usize], exec_dom(f, inst)) {
            (Some(d), Some(e)) => covers(m, d, e),
            (Some(Dom::Full), None) => true,
            _ => false,
        },
        Some(_) => true,
        None => false,
    };
    if n == "v_writelane_b32" {
        let k = match inst.dsts.first() {
            Some((Reg::V(k), _)) if slots.contains(k) => *k,
            _ => return None,
        };
        return lane_of(src(1)).map(|lane| LaneUse::SlotWrite { vgpr: k, lane });
    }
    if n == "v_readlane_b32" {
        if let Some(Src::Val(v)) = src(0) {
            if let Reg::V(k) = f.vals[v.0 as usize].reg {
                if slots.contains(&k) {
                    return lane_of(src(1)).map(|lane| LaneUse::SlotRead { vgpr: k, lane });
                }
            }
        }
        if uniform(src(0)) {
            return Some(LaneUse::Copy);
        }
        return matches!(src(1), Some(Src::Val(v)) if first_set.contains(&v))
            .then_some(LaneUse::ReadHostLane);
    }
    if matches!(
        n,
        "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_u64" | "s_cmp_lg_u64"
    ) {
        let eq = n.contains("_eq_");
        let zero = |k: usize| {
            inst.srcs
                .get(k)
                .is_some_and(|d| d.iter().all(|s| matches!(s, Src::Const(0))))
        };
        if mask_operand(m, inst, 0) && mask_operand(m, inst, 1) {
            return Some(LaneUse::MasksEqual { eq });
        }
        if (mask_operand(m, inst, 0) && zero(1)) || (zero(0) && mask_operand(m, inst, 1)) {
            return Some(LaneUse::MaskZero { eq });
        }
        return None;
    }
    if n.starts_with("s_bcnt1_i32_b") && mask_operand(m, inst, 0) {
        return Some(LaneUse::BallotCount);
    }
    if n.starts_with("s_ff1_i32_b") && mask_operand(m, inst, 0) {
        return inst
            .dsts
            .first()
            .is_some_and(|(_, v)| first_set.contains(v))
            .then_some(LaneUse::FirstSet);
    }
    if (n == "v_mbcnt_lo_u32_b32" || n == "v_mbcnt_hi_u32_b32") && mask_operand(m, inst, 0) {
        let hi = n == "v_mbcnt_hi_u32_b32";
        // the high half only follows the low one of the same count
        let after_lo = match src(1) {
            Some(Src::Val(x)) => match f.vals[x.0 as usize].def {
                ValDef::Inst { block, index } => {
                    let d = &f.blocks[block].insts[index];
                    d.instr.name() == "v_mbcnt_lo_u32_b32" && mask_operand(m, d, 0)
                }
                _ => false,
            },
            _ => false,
        };
        return (!hi || after_lo).then_some(LaneUse::Rank { hi });
    }
    if n.starts_with("v_readfirstlane") {
        if uniform(src(0)) {
            return Some(LaneUse::Copy);
        }
        let (Some(Src::Val(v)), Some((_, res))) = (src(0), inst.dsts.first()) else {
            return None;
        };
        let waterfall = eq_cmps
            .iter()
            .any(|s| s.contains(&Src::Val(v)) && s.contains(&Src::Val(*res)));
        return waterfall.then_some(LaneUse::Waterfall);
    }
    if n == "v_mbcnt_lo_u32_b32" || n == "v_mbcnt_hi_u32_b32" {
        return (src(0) == Some(Src::Const(u32::MAX))).then_some(LaneUse::LaneIndex {
            hi: n == "v_mbcnt_hi_u32_b32",
        });
    }
    None
}

/// The operand lists of every `v_cmp_eq` / `v_cmpx_eq` (waterfall detection).
pub(super) fn eq_compares(f: &Func) -> Vec<Vec<Src>> {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|u| {
            let n = u.instr.name();
            n.starts_with("v_cmp_eq") || n.starts_with("v_cmpx_eq")
        })
        .map(|u| u.srcs.iter().flatten().copied().collect())
        .collect()
}

/// Items 1-4 for every instruction that is cross-lane by nature: the `LaneUse`s found, by the
/// byte offset of the instruction.
pub(super) fn lane_uses(f: &Func, m: &Masks, slots: &HashSet<u16>) -> HashMap<usize, LaneUse> {
    let eq = eq_compares(f);
    // `s_ff1` of a mask whose result is only ever the lane of a `v_readlane`
    let reads = count_reads(f);
    let mut first_set = HashSet::new();
    for blk in &f.blocks {
        for inst in &blk.insts {
            if inst.instr.name().starts_with("s_ff1_i32_b") && mask_operand(m, inst, 0) {
                if let Some(&(_, v)) = inst.dsts.first() {
                    let n = reads.all.get(&v).copied().unwrap_or(0);
                    if n == reads.as_lane.get(&v).copied().unwrap_or(0) {
                        first_set.insert(v);
                    }
                }
            }
        }
    }
    let mut out = HashMap::new();
    for blk in &f.blocks {
        for inst in &blk.insts {
            if let Some(u) = lane_use(f, m, slots, inst, &eq, &first_set) {
                out.insert(inst.instr.at, u);
            }
        }
    }
    out
}
