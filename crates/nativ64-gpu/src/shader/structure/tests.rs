use super::*;
use crate::shader::ir::{build, EntryRegs};

// Programs are assembled with `llvm-mc -mcpu=gfx1013 -mattr=+wavefrontsize64`; branch targets
// are labels in the source, shown here as comments.

fn func(words: &[u32]) -> Func {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let regs = EntryRegs::new().with_sgprs(8..12).with_vgprs(0..4);
    build(&code, 0, false, &regs).unwrap()
}

fn st(words: &[u32]) -> Result<Structured, StructError> {
    structure(&func(words))
}

/// The tree must hold every block once.
fn assert_each_block_once(f: &Func, tree: &[Node]) {
    fn walk(n: &[Node], seen: &mut Vec<BlockId>) {
        for x in n {
            match x {
                Node::Code(b) => seen.push(*b),
                Node::Block { body, .. } | Node::Loop { body, .. } => walk(body, seen),
                Node::If { taken, fall, .. } => {
                    walk(taken, seen);
                    walk(fall, seen);
                }
                _ => {}
            }
        }
    }
    let mut seen = Vec::new();
    walk(tree, &mut seen);
    seen.sort_unstable();
    assert_eq!(seen, (0..f.blocks.len()).collect::<Vec<_>>());
}

fn tree(words: &[u32]) -> String {
    let f = func(words);
    let s = structure(&f).unwrap();
    assert_each_block_once(&f, &s.tree);
    render(&s.tree)
}

const IF: [u32; 7] = [
    0x7c020300, // v_cmp_lt_f32 vcc, v0, v1
    0xbe82246a, // s_and_saveexec_b64 s[2:3], vcc
    0xbf880001, // s_cbranch_execz LJ
    0x7e0402f2, // v_mov_b32 v2, 1.0
    0xbefe0402, // LJ: s_mov_b64 exec, s[2:3]
    0x06060502, // v_add_f32 v3, v2, v2
    0xbf810000, // s_endpgm
];

const IF_ELSE: [u32; 11] = [
    0x7c020300, // v_cmp_lt_f32 vcc, v0, v1
    0xbe82246a, // s_and_saveexec_b64 s[2:3], vcc
    0x8982027e, // s_xor_b64 s[2:3], exec, s[2:3]
    0xbf880001, // s_cbranch_execz LE
    0x7e0402f2, // v_mov_b32 v2, 1.0
    0xbe822702, // LE: s_andn2_saveexec_b64 s[2:3], s[2:3]
    0xbf880001, // s_cbranch_execz LJ
    0x7e040280, // v_mov_b32 v2, 0
    0x88fe027e, // LJ: s_or_b64 exec, exec, s[2:3]
    0x06060502, // v_add_f32 v3, v2, v2
    0xbf810000, // s_endpgm
];

const NESTED: [u32; 13] = [
    0x7c020300, // v_cmp_lt_f32 vcc, v0, v1
    0xbe82246a, // s_and_saveexec_b64 s[2:3], vcc
    0xbf880007, // s_cbranch_execz LJ1
    0x7e0402f2, // v_mov_b32 v2, 1.0
    0x7c080302, // v_cmp_gt_f32 vcc, v2, v1
    0xbe84246a, // s_and_saveexec_b64 s[4:5], vcc
    0xbf880001, // s_cbranch_execz LJ2
    0x7e0602f4, // v_mov_b32 v3, 2.0
    0x88fe047e, // LJ2: s_or_b64 exec, exec, s[4:5]
    0x06040503, // v_add_f32 v2, v3, v2
    0x88fe027e, // LJ1: s_or_b64 exec, exec, s[2:3]
    0x06060502, // v_add_f32 v3, v2, v2
    0xbf810000, // s_endpgm
];

// s0 = 0 ; LH: s0 += 1 ; if s0 == s8 goto LCONT ; if s0 == s9 goto LEXIT ; v0 += v1 ;
// LCONT: if s0 < 8 goto LH ; LEXIT: v1 = v0 ; s_endpgm
const LOOP: [u32; 11] = [
    0xbe800380, 0x80008100, 0xbf060800, 0xbf850003, 0xbf060900, 0xbf850003, 0x06000300, 0xbf0a8800,
    0xbf85fff8, 0x7e020300, 0xbf810000,
];

// the same, with the code after both loops also reachable from before them: that merge
// is outside both loops, and the inner exit leaves two levels
const BREAK2: [u32; 16] = [
    0xbf068009, 0xbf85000c, 0xbe800380, 0x80008100, 0xbe810380, 0x80018101, 0xbf060801, 0xbf850006,
    0x06000300, 0xbf0a8401, 0xbf85fffa, 0x10020301, 0xbf0a8400, 0xbf85fff5, 0x7e040300, 0xbf810000,
];

// a loop entered at either of its two blocks
const IRREDUCIBLE: [u32; 10] = [
    0xbf068008, 0xbf850004, 0x80098109, 0xbf0a8509, 0xbf850001, 0xbf810000, 0x800a810a, 0xbf0a850a,
    0xbf85fff9, 0xbf810000,
];

#[test]
fn if_pattern_tree_and_lane_branch() {
    let f = func(&IF);
    let s = structure(&f).unwrap();
    assert_each_block_once(&f, &s.tree);
    assert_eq!(render(&s.tree), "block 2 {b0 if {br 2 | b1 br 2}} b2 ret");
    // execz tests the lane's own bit of the new EXEC, taken when it is clear
    let Some(BranchKind::Lane(t)) = &s.branches[0] else {
        panic!("{:?}", s.branches)
    };
    assert!(!t.taken_when_set);
    assert_eq!(t.vals.len(), 2);
}

#[test]
fn if_else_tree() {
    assert_eq!(
        tree(&IF_ELSE),
        "block 2 {b0 if {br 2 | b1 br 2}} block 4 {b2 if {br 4 | b3 br 4}} b4 ret"
    );
}

#[test]
fn nested_ifs_tree() {
    assert_eq!(
        tree(&NESTED),
        "block 4 {b0 if {br 4 | block 3 {b1 if {br 3 | b2 br 3}} b3 br 4}} b4 ret"
    );
}

#[test]
fn loop_with_break_and_continue() {
    let t = tree(&LOOP);
    assert!(t.contains("loop"), "{t}");
    // the `continue` of the first test and the back edge go to the loop; the second test breaks
    assert!(t.contains("continue 1"), "{t}");
    assert!(t.contains("br "), "{t}");
}

#[test]
fn two_level_break_is_data() {
    let f = func(&BREAK2);
    let s = structure(&f).unwrap();
    assert_each_block_once(&f, &s.tree);
    let mut deep = Vec::new();
    fn walk(n: &[Node], deep: &mut Vec<(Target, usize)>) {
        for x in n {
            match x {
                Node::Block { body, .. } | Node::Loop { body, .. } => walk(body, deep),
                Node::If { taken, fall, .. } => {
                    walk(taken, deep);
                    walk(fall, deep);
                }
                Node::Br { to, depth } if *depth > 0 => deep.push((*to, *depth)),
                _ => {}
            }
        }
    }
    walk(&s.tree, &mut deep);
    assert!(
        deep.iter().any(|(t, d)| t.kind == BrKind::Exit && *d >= 1),
        "{}",
        render(&s.tree)
    );
    // the construct a deep break leaves lists its target for the re-test after it
    fn forwards(n: &[Node], out: &mut Vec<Target>) {
        for x in n {
            match x {
                Node::Block {
                    body, forwards: f, ..
                }
                | Node::Loop {
                    body, forwards: f, ..
                } => {
                    out.extend(f);
                    forwards(body, out);
                }
                Node::If { taken, fall, .. } => {
                    forwards(taken, out);
                    forwards(fall, out);
                }
                _ => {}
            }
        }
    }
    let mut fw = Vec::new();
    forwards(&s.tree, &mut fw);
    assert!(!fw.is_empty(), "{}", render(&s.tree));
}

#[test]
fn irreducible_graph_is_an_error() {
    assert!(matches!(
        st(&IRREDUCIBLE),
        Err(StructError::Irreducible { .. })
    ));
}

#[test]
fn cross_lane_use_names_the_instruction() {
    // v_cmp_lt_f32 vcc ; s_bcnt1_i32_b64 s0, vcc ; v_readfirstlane_b32 s1, v0 ; s_endpgm
    // (the count of a mask is a ballot; the readfirstlane of a lane-wise value is not)
    let words = [0x7c020300, 0xbe80106a, 0x7e020500, 0xbf810000];
    let f = func(&words);
    let m = type_masks(&f);
    let names: Vec<_> = m
        .cross_lane
        .iter()
        .map(|c| (c.at, c.name.as_str()))
        .collect();
    assert_eq!(names, vec![(8, "v_readfirstlane_b32")]);
    match structure(&f) {
        Err(StructError::CrossLane(c)) => {
            assert_eq!((c.at, c.name.as_str()), (8, "v_readfirstlane_b32"))
        }
        r => panic!("{r:?}"),
    }
}

#[test]
fn mask_typing_follows_the_allowed_uses() {
    let words = [
        0x7c020300, // v_cmp_lt_f32 vcc, v0, v1
        0x87827e6a, // s_and_b64 s[2:3], vcc, exec
        0x88848002, // s_or_b64 s[4:5], s[2:3], 0
        0xbe860804, // s_not_b64 s[6:7], s[4:5]
        0xd5010002, 0x001a0300, // v_cndmask_b32 v2, v0, v1, s[6:7]
        0x50060300, // v_addc_co_u32 v3, vcc, v0, v1, vcc
        0x8780087e, // s_and_b64 s[0:1], exec, s[8:9]: a scalar beside a mask: its bit n for lane n
        0xbf810000, // s_endpgm
    ];
    let f = func(&words);
    let m = type_masks(&f);
    let named = |reg: Reg| {
        f.vals
            .iter()
            .enumerate()
            .filter(|(_, i)| i.reg == reg && matches!(i.def, ValDef::Inst { .. }))
            .map(|(k, _)| Val(k as u32))
            .collect::<Vec<_>>()
    };
    for r in [
        Reg::S(2),
        Reg::S(3),
        Reg::S(4),
        Reg::S(5),
        Reg::S(6),
        Reg::S(7),
    ] {
        assert!(
            named(r).iter().all(|v| m.is_mask(*v)),
            "{r} should be a mask"
        );
    }
    assert!(named(Reg::V(2)).iter().all(|v| !m.is_mask(*v)));
    // the cndmask, the carry-in and the scalar bit are allowed uses
    assert!(m.cross_lane.is_empty(), "{:?}", m.cross_lane);
    for r in [Reg::S(0), Reg::S(1)] {
        assert!(
            named(r).iter().all(|v| m.is_mask(*v)),
            "{r} should be a mask"
        );
    }
}

#[test]
fn scalars_alone_do_not_make_a_mask() {
    // s_and_b64 s[0:1], s[8:9], s[10:11]: no mask input
    let f = func(&[0x8780_0a08, 0xbf810000]);
    let m = type_masks(&f);
    assert!(m.cross_lane.is_empty());
    for (k, i) in f.vals.iter().enumerate() {
        if matches!(i.def, ValDef::Inst { .. }) {
            assert!(!m.is_mask(Val(k as u32)), "{} should not be a mask", i.reg);
        }
    }
}

#[test]
fn wrexec_types_like_the_saveexec_family() {
    let words = [
        0x7c020300, // v_cmp_lt_f32 vcc, v0, v1
        0xbe823a6a, // s_andn2_wrexec_b64 s[2:3], vcc
        0xbe84396a, // s_andn1_wrexec_b64 s[4:5], vcc
        0xbf810000, // s_endpgm
    ];
    let f = func(&words);
    let m = type_masks(&f);
    assert!(m.cross_lane.is_empty(), "{:?}", m.cross_lane);
    for (k, i) in f.vals.iter().enumerate() {
        let sgpr = matches!(i.reg, Reg::S(2..=5) | Reg::ExecLo | Reg::ExecHi);
        if sgpr && matches!(i.def, ValDef::Inst { .. }) {
            assert!(m.is_mask(Val(k as u32)), "{} should be a mask", i.reg);
        }
    }
}

#[test]
fn loop_carried_masks_type_optimistically() {
    // s[2:3] = 0 ; L: v_cmp_lt_f32 vcc ; s_or_b64 s[2:3], s[2:3], vcc ; s_cbranch_vccnz L
    let words = [
        0xbe820480, // s_mov_b64 s[2:3], 0
        0x7c020300, // L: v_cmp_lt_f32 vcc, v0, v1
        0x8882026a, // s_or_b64 s[2:3], s[2:3], vcc
        0xbf87fffd, // s_cbranch_vccnz L
        0xbf810000,
    ];
    let m = type_masks(&func(&words));
    assert!(m.cross_lane.is_empty(), "{:?}", m.cross_lane);
}

#[test]
fn scalar_defined_in_a_lane_skipped_region_is_speculated() {
    // v_cmp ; s_and_saveexec ; s_cbranch_execz LJ ; s_add_u32 s9, s9, 1 ; LJ: ... s_add s10, s9, 1
    let words = [
        0x7c020300, 0xbe82246a, 0xbf880001, 0x80098109, 0x88fe027e, 0x800a8109, 0x7e04020a,
        0xbf810000,
    ];
    let s = st(&words).unwrap();
    assert!(matches!(s.branches[0], Some(BranchKind::Wave(_))));
}

#[test]
fn scalar_message_in_a_lane_skipped_region_is_an_error() {
    // the same with an s_sendmsg in the region
    let words = [
        0xbefc0380, 0x7c020300, 0xbe82246a, 0xbf880002, 0x80098109, 0xbf900001, 0x88fe027e,
        0x800a8109, 0x7e04020a, 0xbf810000,
    ];
    assert!(matches!(
        st(&words),
        Err(StructError::DivergentScalar { reg: Reg::S(9), .. })
    ));
}

#[test]
fn uniform_branch_over_the_same_scalar_is_fine() {
    // s_cmp_eq_u32 s8, 0 ; s_cbranch_scc1 LJ ; s_add_u32 s9, s9, 1 ; LJ: s_add s10, s9, 1
    let words = [
        0xbf068008, 0xbf850001, 0x80098109, 0x800a8109, 0x7e04020a, 0xbf810000,
    ];
    let s = st(&words).unwrap();
    assert_eq!(s.branches[0], Some(BranchKind::Uniform));
}

fn guards_of(s: &Structured) -> Vec<Guard> {
    s.guards
        .iter()
        .flatten()
        .copied()
        .filter(|g| *g != Guard::Scalar)
        .collect()
}

#[test]
fn guards_are_removed_on_the_if_pattern() {
    let s = st(&IF).unwrap();
    // v_cmp (entry EXEC), v_mov (after execz), v_add (EXEC restored from the saved copy)
    assert_eq!(guards_of(&s), vec![Guard::Proved; 3]);
}

#[test]
fn guards_are_removed_through_nesting() {
    let s = st(&NESTED).unwrap();
    assert_eq!(guards_of(&s), vec![Guard::Proved; 6]);
}

#[test]
fn guards_after_an_else_join_stay() {
    // The else region and its body are proved by the branches' bits, but the join's EXEC is
    // `exec | saved` where `saved` came out of an xor: that needs set reasoning on bits, not
    // facts about values, so the add after the join keeps its guard (always correct).
    let s = st(&IF_ELSE).unwrap();
    assert_eq!(
        guards_of(&s),
        vec![Guard::Proved, Guard::Proved, Guard::Proved, Guard::Active]
    );
}

#[test]
fn guards_stay_when_exec_is_narrowed_without_a_branch() {
    // v_cmpx_lt_f32 v0, v1 ; v_mov_b32 v2, 1.0: EXEC now depends on the compare
    let words = [0x7c220300, 0x7e0402f2, 0xbf810000];
    let s = st(&words).unwrap();
    assert_eq!(guards_of(&s), vec![Guard::Proved, Guard::Active]);
}

#[test]
fn pixel_discard_is_recognised() {
    // s_wqm_b64 s[2:3], exec ; v_cmpx_lt_f32 v0, v1 ; s_andn2_b64 s[2:3], s[2:3], exec ;
    // s_cbranch_scc0 LX ; s_wqm_b64 exec, s[2:3] ; v_mov_b32 v2, 1.0 ; s_endpgm ;
    // LX: exp null off, off, off, off done ; s_endpgm
    let words = [
        0xbe820a7e, 0x7c220300, 0x8a827e02, 0xbf840003, 0xbefe0a02, 0x7e0402f2, 0xbf810000,
        0xf8000890, 0x00000000, 0xbf810000,
    ];
    let s = st(&words).unwrap();
    assert_eq!(s.demotes.len(), 1, "{:?}", s.demotes);
    let d = &s.demotes[0];
    let i = &func(&words).blocks[d.block].insts[d.inst].instr;
    assert_eq!(i.name(), "s_andn2_b64");
    // the branch tests the SCC of a mask operation: per lane, its own live bit
    assert!(matches!(s.branches[d.branch], Some(BranchKind::Lane(_))));
}

#[test]
fn an_andn2_that_feeds_a_normal_exit_is_no_discard() {
    // the exit does real work (a vector instruction), so this is an ordinary test
    let words = [
        0xbe820a7e, 0x7c220300, 0x8a827e02, 0xbf840003, 0xbefe0a02, 0x7e0402f2, 0xbf810000,
        0x7e040280, 0xbf810000,
    ];
    assert!(st(&words).unwrap().demotes.is_empty());
}

#[test]
fn readfirstlane_of_a_uniform_value_is_a_copy() {
    // v_mov_b32 v0, s8 ; v_readfirstlane_b32 s1, v0 ; s_endpgm
    let words = [0x7e000208, 0x7e020500, 0xbf810000];
    let f = func(&words);
    assert!(type_masks(&f).cross_lane.is_empty());
    assert!(structure(&f).is_ok());
}
