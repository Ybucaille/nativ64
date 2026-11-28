//! Wave-size dependence (`docs/shaders.md`, Wave64 on narrower subgroups): a subgroup
//! operation covers the invocations of one host subgroup, so on a host whose subgroups are
//! narrower than the program's wave, an operation whose result depends on the other lanes of
//! the wave does not give the console's result.

use super::uniform::{Dom, LaneUse};
use super::wave::Scan;
use super::{BranchKind, Masks, Structured};
use crate::shader::ir::{Func, Inst, Src, Term, Val, ValDef};

/// The first operation of a program whose result depends on lanes a narrower subgroup does
/// not hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaveDependence {
    /// Byte offset of the instruction (of the block, for a branch).
    pub at: usize,
    /// What it does with the other lanes.
    pub what: &'static str,
}

/// The EXEC value `e` holds every lane of the wave: the entry one, or the all-ones one an
/// `s_or_saveexec ..., -1` sets (a whole-wave section).
fn whole_exec(f: &Func, e: Val) -> bool {
    match f.vals[e.0 as usize].def {
        ValDef::Arg => true,
        ValDef::Inst { block, index } => {
            let i = &f.blocks[block].insts[index];
            i.instr.name().starts_with("s_or_saveexec_b")
                && i.srcs
                    .first()
                    .is_some_and(|d| d.iter().all(|s| *s == Src::Const(u32::MAX)))
        }
        ValDef::Phi { .. } => false,
    }
}

/// The mask dword `v` has the same bit in every lane of the wave.
fn mask_wave_uniform(f: &Func, m: &Masks, v: Val) -> bool {
    match m.uniform.get(&v) {
        Some(None) => true,
        // uniform over the lanes of an EXEC value: that has to hold all of them
        Some(Some(e)) => whole_exec(f, *e),
        None => false,
    }
}

fn src_wave_uniform(f: &Func, m: &Masks, s: &Src) -> bool {
    match s {
        Src::Const(c) => *c == 0 || *c == u32::MAX,
        Src::Undef => true,
        Src::Val(v) => mask_wave_uniform(f, m, *v),
        _ => false,
    }
}

/// What `u` does with other lanes, unless its result is the same on every subgroup size.
/// Exhaustive, so that a new meaning has to be classified here.
fn needs_the_wave(f: &Func, m: &Masks, inst: &Inst, u: LaneUse) -> Option<&'static str> {
    match u {
        // the lane's own value, its own variables, its console index, or its quad (four
        // lanes of one subgroup on every host); `Copy` was proved uniform
        LaneUse::Copy
        | LaneUse::Waterfall
        | LaneUse::SlotWrite { .. }
        | LaneUse::SlotRead { .. }
        | LaneUse::LaneIndex { .. }
        | LaneUse::Quad { .. } => None,
        // only feeds the section's results, which are classified below
        LaneUse::WaveSkip => None,
        LaneUse::BallotCount => Some("a lane count of a mask"),
        LaneUse::FirstSet => Some("the first lane of a mask"),
        LaneUse::ReadHostLane => Some("a read of another lane"),
        LaneUse::Rank { .. } => Some("a rank in a mask"),
        LaneUse::MaskZero { .. } | LaneUse::MasksEqual { .. } => {
            let uniform = inst
                .srcs
                .iter()
                .flatten()
                .all(|s| src_wave_uniform(f, m, s));
            (!uniform).then_some("a vote on a mask")
        }
        LaneUse::WaveOut { op, scan, leaf } => {
            // a minimum, maximum, and, or of one value every lane holds is that value on any
            // number of lanes (an exclusive scan is not: its first lane gets the identity)
            let same = op.idempotent()
                && scan != Scan::Exclusive
                && match m.uniform_vals[leaf.0 as usize] {
                    Some(Dom::Full) => true,
                    Some(Dom::Exec(e)) => whole_exec(f, e),
                    None => false,
                };
            (!same).then_some(match scan {
                Scan::Reduce => "a reduction over the wave",
                Scan::Inclusive | Scan::Exclusive => "a prefix scan over the wave",
            })
        }
    }
}

/// The first operation of `f` that needs the whole wave in one subgroup, in block order.
pub fn wave_dependence(f: &Func, s: &Structured) -> Option<WaveDependence> {
    let m = &s.masks;
    for (b, blk) in f.blocks.iter().enumerate() {
        for inst in &blk.insts {
            let at = inst.instr.at;
            let found = m
                .lanes
                .get(&at)
                .and_then(|&u| needs_the_wave(f, m, inst, u));
            if let Some(what) = found {
                return Some(WaveDependence { at, what });
            }
        }
        // the 7a vote: the wave enters a region when any of its lanes does
        if let (Term::Branch { .. }, Some(Some(BranchKind::Wave(t)))) =
            (&blk.term, s.branches.get(b))
        {
            if !t.vals.iter().all(|&v| mask_wave_uniform(f, m, v)) {
                return Some(WaveDependence {
                    at: blk.at,
                    what: "a wave-wide branch",
                });
            }
        }
    }
    None
}
