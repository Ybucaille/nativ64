//! The shader IR (`docs/shaders.md`, IR; chunk 3 of the G4 plan): the decoded GFX10 program as
//! SSA over 32-bit register dwords, one function per program.
//!
//! * Registers are dwords (`Reg`): a 64-bit operand or a wider range is split into its dwords.
//!   EXEC, VCC, SCC and M0 are ordinary registers, so the implicit operands of every opcode are
//!   explicit uses and defs here (chunk 4 types the masks from them).
//! * An instruction keeps its decoded form (`Inst::instr`: opcode, modifiers, literal, memory
//!   fields); the IR only adds the SSA values it reads and writes. No semantics are lowered.
//! * Blocks split at branch targets and after branches; branches become terminators. Phis are
//!   placed on the iterated dominance frontier of each register's definitions, then pruned.
//! * A register read with no definition on the path is an `Arg` when the entry layout
//!   (`EntryRegs`) names it, else an error. The old value a vector write preserves (lanes
//!   outside EXEC) reads `undef` when nothing defined it.
//! * Scalar constants fold while building: literals, inline constants, `s_getpc_b64` plus an
//!   offset (`Src::CodeLo` / `Src::CodeHi`: the program's own address) and `s_load_*` from
//!   constant data inside the code slice. A folded instruction leaves no `Inst`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::{self, Write as _};
use std::ops::Range;

use super::gfx10::ops::Slot;
use super::gfx10::{self, DecodeError, Enc, Instr, Operand, Resolved};

/// One 32-bit register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reg {
    S(u16),
    /// Trap temporaries `ttmp0..ttmp15`.
    T(u8),
    V(u16),
    VccLo,
    VccHi,
    ExecLo,
    ExecHi,
    M0,
    Scc,
    /// The `null` destination (code 125). Never appears in a built `Func`.
    Null,
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::S(n) => write!(f, "s{n}"),
            Self::T(n) => write!(f, "ttmp{n}"),
            Self::V(n) => write!(f, "v{n}"),
            Self::VccLo => f.write_str("vcc_lo"),
            Self::VccHi => f.write_str("vcc_hi"),
            Self::ExecLo => f.write_str("exec_lo"),
            Self::ExecHi => f.write_str("exec_hi"),
            Self::M0 => f.write_str("m0"),
            Self::Scc => f.write_str("scc"),
            Self::Null => f.write_str("null"),
        }
    }
}

pub type BlockId = usize;

/// An SSA value (one register dword).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Val(pub u32);

/// What an instruction reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Src {
    Val(Val),
    Const(u32),
    /// Low dword of the address of byte `n` of the program's code (`s_getpc_b64` + offsets;
    /// the code is assumed aligned so that adding an offset never carries into the high dword).
    CodeLo(u32),
    /// High dword of the code address.
    CodeHi,
    /// A phi input from a path that never defined the register.
    Undef,
}

impl fmt::Display for Src {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Val(v) => write!(f, "%{}", v.0),
            Self::Const(c) => write!(f, "#{c:#x}"),
            Self::CodeLo(o) => write!(f, "code+{o:#x}"),
            Self::CodeHi => f.write_str("code.hi"),
            Self::Undef => f.write_str("undef"),
        }
    }
}

/// Where a value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValDef {
    /// The register's value at entry.
    Arg,
    Phi {
        block: BlockId,
        index: usize,
    },
    Inst {
        block: BlockId,
        index: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValInfo {
    pub reg: Reg,
    pub def: ValDef,
}

#[derive(Clone, Debug)]
pub struct Phi {
    pub reg: Reg,
    pub dst: Val,
    /// One input per predecessor, in `Block::preds` order.
    pub args: Vec<(BlockId, Src)>,
}

/// One instruction with its SSA operands.
#[derive(Clone, Debug)]
pub struct Inst {
    pub instr: Instr,
    /// Written dwords: the explicit destinations in operand order, then EXEC and SCC when the
    /// opcode writes them.
    pub dsts: Vec<(Reg, Val)>,
    /// One entry per explicit source operand (in print order), one `Src` per dword.
    pub srcs: Vec<Vec<Src>>,
    /// Implicit reads: EXEC, VCC, SCC, M0, and the old value of every written VGPR / of a
    /// conditionally written SGPR (named by the register they preserve).
    pub imps: Vec<(Reg, Src)>,
}

/// How a block ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Term {
    Jump(BlockId),
    /// `s_cbranch_*`: reads `cond`, goes to `taken` or falls to `fall`.
    Branch {
        op: &'static str,
        cond: Vec<(Reg, Src)>,
        taken: BlockId,
        fall: BlockId,
    },
    /// `s_endpgm`.
    End,
}

#[derive(Clone, Debug)]
pub struct Block {
    /// Byte offset of the first instruction.
    pub at: usize,
    pub phis: Vec<Phi>,
    pub insts: Vec<Inst>,
    pub term: Term,
    pub preds: Vec<BlockId>,
}

impl Block {
    pub fn succs(&self) -> Vec<BlockId> {
        match &self.term {
            Term::Jump(b) => vec![*b],
            Term::Branch { taken, fall, .. } => {
                if taken == fall {
                    vec![*taken]
                } else {
                    vec![*taken, *fall]
                }
            }
            Term::End => vec![],
        }
    }
}

/// A translated program.
#[derive(Clone, Debug)]
pub struct Func {
    pub wave32: bool,
    pub blocks: Vec<Block>,
    pub entry: BlockId,
    /// Indexed by `Val.0`.
    pub vals: Vec<ValInfo>,
    /// The entry values, by register.
    pub args: Vec<(Reg, Val)>,
    /// The entry registers whose value is the same in every lane (`EntryRegs::uniform_vgprs`).
    pub uniform_args: Vec<Reg>,
    /// Immediate dominators (`None` for the entry).
    pub idom: Vec<Option<BlockId>>,
    /// Reverse post-order from the entry.
    pub rpo: Vec<BlockId>,
}

impl Func {
    /// The constant a source holds, when it is one.
    pub fn konst(&self, s: Src) -> Option<u32> {
        match s {
            Src::Const(c) => Some(c),
            _ => None,
        }
    }
}

/// The registers that hold something when a program starts (everything else is undefined).
/// EXEC is always one.
#[derive(Clone, Debug, Default)]
pub struct EntryRegs {
    pub sgprs: Vec<Range<u16>>,
    pub vgprs: Vec<Range<u16>>,
    /// A vertex program's attribute VGPRs. When the program opens with the call to the fetch
    /// routine (`s_swappc_b64 s[4:5], s[4:5]`) the call becomes an instruction that defines
    /// them; otherwise they are not entry registers.
    pub fetch: Option<Vec<u16>>,
    /// Entry VGPRs the stage table marks wave-uniform (mesh `v5`): the same in every lane.
    pub uniform_vgprs: Vec<Range<u16>>,
}

/// Dwords each `SPI_PS_INPUT_ADDR` bit takes, in bit order: the barycentric pairs (the pull
/// model's takes three), then the single-dword system values.
pub const PS_INPUT_DWORDS: [u16; 16] = [2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1];

/// The SGPRs that hold user data for a stage: from the lowest `first_user_dword` of its root
/// layout to the end of the last entry, at least s8 (`docs/shaders.md`, Inputs and outputs).
pub fn user_range(rec: &super::native::Record) -> Range<u16> {
    let lo = rec
        .layout
        .iter()
        .map(|l| u16::from(l.first_user_dword))
        .min()
        .unwrap_or(8);
    let hi = rec
        .layout
        .iter()
        .map(|l| u16::from(l.first_user_dword) + l.user_sgprs as u16)
        .max()
        .unwrap_or(0);
    lo..hi.max(8)
}

impl EntryRegs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_sgprs(mut self, r: Range<u16>) -> Self {
        self.sgprs.push(r);
        self
    }

    pub fn with_vgprs(mut self, r: Range<u16>) -> Self {
        self.vgprs.push(r);
        self
    }

    /// Marks entry VGPRs as holding the same value in every lane.
    pub fn with_uniform_vgprs(mut self, r: Range<u16>) -> Self {
        self.uniform_vgprs.push(r);
        self
    }

    pub fn with_fetch(mut self, vgprs: Vec<u16>) -> Self {
        self.fetch = Some(vgprs);
        self
    }

    /// The entry state of a stage, from its stage data (`docs/shaders.md`, Inputs and outputs).
    ///
    /// * VS: the user SGPRs its root layout names (from s8); v0 vertex id, v2 primitive id, v3
    ///   instance id; the attribute VGPRs of its inputs come from the fetch call.
    /// * PS: the user SGPRs, then the primitive mask and the ordered-section id; the VGPRs
    ///   `SPI_PS_INPUT_ADDR` enables, in bit order.
    /// * CS: user SGPRs from s0, then the enabled group id and info SGPRs and the scratch
    ///   offset (`COMPUTE_PGM_RSRC2`); thread ids in v0...
    /// * Mesh: s3 (merged wave info), the user SGPRs from s8, v5 (group id times group size).
    pub fn from_stage(rec: &super::native::Record) -> Self {
        use super::native::StageData;
        let user = || user_range(rec);
        let mut r = Self::new();
        match &rec.data {
            StageData::Vertex(d) => {
                r = r.with_sgprs(user()).with_vgprs(0..1).with_vgprs(2..4);
                let mut attrs: Vec<u16> = d
                    .inputs
                    .iter()
                    .flat_map(|i| {
                        (0..4)
                            .filter(move |c| i.mask >> c & 1 != 0)
                            .map(move |c| u16::from(i.first_vgpr) + c)
                    })
                    .collect();
                attrs.sort_unstable();
                attrs.dedup();
                if !attrs.is_empty() {
                    r = r.with_fetch(attrs);
                }
            }
            StageData::Pixel(d) => {
                let u = user();
                r = r.with_sgprs(u.start..u.end + 2);
                let mut next = 0;
                for (bit, n) in PS_INPUT_DWORDS.iter().enumerate() {
                    if d.spi_ps_input_addr >> bit & 1 != 0 {
                        r = r.with_vgprs(next..next + n);
                        next += n;
                    }
                }
            }
            StageData::Compute(d) => {
                let top = d
                    .system_sgprs()
                    .last()
                    .map_or(d.user_sgprs(), |&l| l + 1)
                    .max(d.user_sgprs());
                if top > 0 {
                    r = r.with_sgprs(0..top);
                }
                r = r.with_vgprs(0..d.thread_id_vgprs());
            }
            StageData::Mesh(d) => {
                r = r
                    .with_sgprs(3..4)
                    .with_sgprs(8..8 + u16::from(d.user_sgprs))
                    .with_vgprs(5..6)
                    .with_uniform_vgprs(5..6);
            }
        }
        r
    }

    fn has(&self, reg: Reg) -> bool {
        match reg {
            Reg::ExecLo | Reg::ExecHi => true,
            Reg::S(n) => self.sgprs.iter().any(|r| r.contains(&n)),
            Reg::V(n) => self.vgprs.iter().any(|r| r.contains(&n)),
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrError {
    /// Byte offset of the instruction.
    pub at: usize,
    pub kind: IrErrorKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IrErrorKind {
    Decode(DecodeError),
    /// `s_setpc_b64` / `s_swappc_b64` / `s_rfe_b64` / `s_call_b64`.
    Indirect,
    /// An encoding the IR does not model (its register reads and writes are not known).
    UnsupportedEncoding(&'static str),
    /// An opcode or operand form the IR does not model.
    Unsupported(String),
    /// A read of a register nothing defines and the entry layout does not provide.
    UndefinedRead(Reg),
    /// A code the operand fields cannot use here (constant destination, `vccz` value...).
    BadOperand(&'static str),
}

impl fmt::Display for IrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let at = self.at;
        match &self.kind {
            IrErrorKind::Decode(e) => write!(f, "{e}"),
            IrErrorKind::Indirect => write!(f, "indirect jump at +{at:#x}"),
            IrErrorKind::UnsupportedEncoding(e) => {
                write!(f, "{e} instruction at +{at:#x} is not modelled by the IR")
            }
            IrErrorKind::Unsupported(s) => write!(f, "{s} at +{at:#x}"),
            IrErrorKind::UndefinedRead(r) => write!(
                f,
                "read of {r} at +{at:#x}: not an entry register and nothing defines it"
            ),
            IrErrorKind::BadOperand(s) => write!(f, "bad operand ({s}) at +{at:#x}"),
        }
    }
}

impl std::error::Error for IrError {}

impl From<DecodeError> for IrError {
    fn from(e: DecodeError) -> Self {
        Self {
            at: e.at,
            kind: IrErrorKind::Decode(e),
        }
    }
}

// ---------------------------------------------------------------------------------------
// Lowering one instruction to register reads and writes.

/// A source before renaming.
#[derive(Clone, Copy, Debug)]
enum Rd {
    Reg(Reg),
    Const(u32),
}

#[derive(Default)]
struct Lowered {
    dsts: Vec<Reg>,
    srcs: Vec<Vec<Rd>>,
    imps: Vec<Reg>,
    olds: Vec<Reg>,
}

fn bad(at: usize, kind: IrErrorKind) -> IrError {
    IrError { at, kind }
}

/// The inline constants 240..=248 as f32 bits.
fn inline_float(code: u16) -> u32 {
    match code {
        240 => 0.5f32.to_bits(),
        241 => (-0.5f32).to_bits(),
        242 => 1.0f32.to_bits(),
        243 => (-1.0f32).to_bits(),
        244 => 2.0f32.to_bits(),
        245 => (-2.0f32).to_bits(),
        246 => 4.0f32.to_bits(),
        247 => (-4.0f32).to_bits(),
        _ => 0x3e22_f983,
    }
}

/// Dword `k` of the operand with source code `code`. `f64` selects the double reading of an
/// inline float constant.
fn operand_dword(code: u16, k: u16, f64: bool, at: usize) -> Result<Rd, IrError> {
    let err = |s| Err(bad(at, IrErrorKind::BadOperand(s)));
    let reg = |r| Ok(Rd::Reg(r));
    match code {
        0..=105 => {
            if code + k > 105 {
                return err("SGPR range past s105");
            }
            reg(Reg::S(code + k))
        }
        106..=123 => match code + k {
            106 => reg(Reg::VccLo),
            107 => reg(Reg::VccHi),
            a @ 108..=123 => reg(Reg::T((a - 108) as u8)),
            _ => err("register range past ttmp15"),
        },
        124 => {
            if k == 0 {
                reg(Reg::M0)
            } else {
                err("m0 in a register range")
            }
        }
        125 => reg(Reg::Null),
        126 | 127 => match code - 126 + k {
            0 => reg(Reg::ExecLo),
            1 => reg(Reg::ExecHi),
            _ => err("exec range"),
        },
        128..=208 | 240..=248 => {
            let int = if code <= 192 {
                i64::from(code) - 128
            } else if code <= 208 {
                -(i64::from(code) - 192)
            } else {
                0
            };
            if code >= 240 {
                if f64 {
                    let bits = if code == 248 {
                        (1.0 / (2.0 * std::f64::consts::PI)).to_bits()
                    } else {
                        f64::from(f32::from_bits(inline_float(code))).to_bits()
                    };
                    Ok(Rd::Const(match k {
                        0 => bits as u32,
                        1 => (bits >> 32) as u32,
                        _ => 0,
                    }))
                } else {
                    Ok(Rd::Const(if k == 0 { inline_float(code) } else { 0 }))
                }
            } else {
                Ok(Rd::Const(match k {
                    0 => int as u32,
                    1 => (int >> 32) as u32,
                    _ => (int >> 32) as u32,
                }))
            }
        }
        251 | 252 => err("vccz / execz used as a value"),
        253 => {
            if k == 0 {
                reg(Reg::Scc)
            } else {
                err("scc in a register range")
            }
        }
        256..=511 => reg(Reg::V(code - 256 + k)),
        _ => err("operand code"),
    }
}

fn slot_width(s: &Slot, wave32: bool) -> Option<u8> {
    let c = match s {
        Slot::SDst(c)
        | Slot::SSrc0(c)
        | Slot::SSrc1(c)
        | Slot::VDst(c)
        | Slot::VDstS(c)
        | Slot::Src0(c)
        | Slot::VSrc1(c)
        | Slot::Src1(c)
        | Slot::Src2(c)
        | Slot::SDstB(c)
        | Slot::SBase(c)
        | Slot::SData(c)
        | Slot::Vdata(c) => *c,
        _ => return None,
    };
    Some(if c == 0 {
        if wave32 {
            1
        } else {
            2
        }
    } else {
        c
    })
}

fn mask_regs(wave32: bool) -> Vec<Reg> {
    if wave32 {
        vec![Reg::ExecLo]
    } else {
        vec![Reg::ExecLo, Reg::ExecHi]
    }
}

fn vcc_regs(wave32: bool) -> Vec<Reg> {
    if wave32 {
        vec![Reg::VccLo]
    } else {
        vec![Reg::VccLo, Reg::VccHi]
    }
}

/// The source dwords of one operand.
fn src_dwords(op: &Operand, width: u8, i: &Instr) -> Result<Vec<Rd>, IrError> {
    let name = i.name();
    let f64 = name.contains("f64");
    let sext = name.contains("i64");
    Ok(match op {
        Operand::Reg { code, .. } => (0..u16::from(width))
            .map(|k| {
                operand_dword(*code, k, f64, i.at).map(|r| match r {
                    Rd::Reg(Reg::Null) => Rd::Const(0),
                    r => r,
                })
            })
            .collect::<Result<_, _>>()?,
        Operand::Literal(v) | Operand::Imm(v) => {
            let mut d = vec![Rd::Const(0); usize::from(width)];
            if f64 && width == 2 {
                d[1] = Rd::Const(*v);
            } else {
                d[0] = Rd::Const(*v);
                if sext && width == 2 {
                    d[1] = Rd::Const(((*v as i32) >> 31) as u32);
                }
            }
            d
        }
        Operand::Fixed(_) => vcc_regs(i.wave32).into_iter().map(Rd::Reg).collect(),
    })
}

fn dst_dwords(op: &Operand, width: u8, i: &Instr) -> Result<Vec<Reg>, IrError> {
    match op {
        Operand::Reg { code, .. } => (0..u16::from(width))
            .map(|k| match operand_dword(*code, k, false, i.at)? {
                Rd::Reg(r) => Ok(r),
                Rd::Const(_) => Err(bad(
                    i.at,
                    IrErrorKind::BadOperand("constant used as a destination"),
                )),
            })
            .collect(),
        Operand::Fixed(_) => Ok(vcc_regs(i.wave32)),
        _ => Err(bad(
            i.at,
            IrErrorKind::BadOperand("immediate used as a destination"),
        )),
    }
}

/// True for the instructions that carry no register effect the IR keeps.
fn vanishes(i: &Instr) -> bool {
    let n = i.name();
    matches!(
        n,
        "s_nop"
            | "v_nop"
            | "s_inst_prefetch"
            | "s_clause"
            | "s_code_end"
            | "s_icache_inv"
            | "s_gl1_inv"
            | "s_setprio"
            | "s_sleep"
            | "s_wait_idle"
            | "s_ttracedata"
            | "s_ttracedata_imm"
    ) || n.starts_with("s_waitcnt")
        || n.starts_with("s_dcache_")
        || n.starts_with("buffer_gl")
}

fn writes_scc(i: &Instr) -> bool {
    let n = i.name();
    match i.enc {
        Enc::Sop2 => {
            !(n.starts_with("s_cselect")
                || n.starts_with("s_mul_")
                || n.starts_with("s_bfm")
                || n.starts_with("s_pack"))
        }
        Enc::Sop1 => {
            n.starts_with("s_not_")
                || n.starts_with("s_wqm_")
                || n.starts_with("s_bcnt")
                || n.starts_with("s_quadmask")
                || n.starts_with("s_abs_")
                || n.contains("saveexec")
                || n.contains("wrexec")
        }
        Enc::Sopc => true,
        Enc::Sopk => n.starts_with("s_cmpk") || n == "s_addk_i32",
        _ => false,
    }
}

fn reads_scc(i: &Instr) -> bool {
    let n = i.name();
    match i.enc {
        Enc::Sop2 => matches!(n, "s_addc_u32" | "s_subb_u32") || n.starts_with("s_cselect"),
        Enc::Sop1 => n.starts_with("s_cmov"),
        Enc::Sopk => n.starts_with("s_cmovk"),
        _ => false,
    }
}

/// Reads and writes of `i`, or `None` when it vanishes.
#[allow(unreachable_patterns)] // the decoder grows encodings; they must stay a typed error
fn lower(i: &Instr) -> Result<Option<Lowered>, IrError> {
    if vanishes(i) {
        return Ok(None);
    }
    let name = i.name();
    match i.enc {
        Enc::Sop1
        | Enc::Sop2
        | Enc::Sopc
        | Enc::Sopk
        | Enc::Sopp
        | Enc::Smem
        | Enc::Vop1
        | Enc::Vop2
        | Enc::Vopc
        | Enc::Vop3
        | Enc::Vintrp
        | Enc::Exp
        | Enc::Mubuf
        | Enc::Mtbuf
        | Enc::Mimg
        | Enc::Ds
        | Enc::Vop3p
        | Enc::Flat => {}
        e => return Err(bad(i.at, IrErrorKind::UnsupportedEncoding(e.name()))),
    }
    if name.contains("movrel") || name.starts_with("s_set_gpr_idx") {
        return Err(bad(
            i.at,
            IrErrorKind::Unsupported(format!("{name}: indexed register access")),
        ));
    }
    let vector = !matches!(
        i.enc,
        Enc::Sop1 | Enc::Sop2 | Enc::Sopc | Enc::Sopk | Enc::Sopp | Enc::Smem
    );
    let mut lo = Lowered::default();
    let mut seen_src = false;
    for (slot, op) in i.info.slots.iter().zip(&i.operands) {
        let Some(w) = slot_width(slot, i.wave32) else {
            if let (Slot::Lit, Operand::Literal(v)) = (slot, op) {
                lo.srcs.push(vec![Rd::Const(*v)]);
            } else if let (Slot::Hex16, Operand::Imm(v), Enc::Sopk) = (slot, op, i.enc) {
                // the 16-bit immediate: unsigned for the `_u32` compares, else sign-extended
                let v = if name.ends_with("_u32") {
                    *v & 0xffff
                } else {
                    i32::from(*v as u16 as i16) as u32
                };
                lo.srcs.push(vec![Rd::Const(v)]);
            } else if let Slot::Fixed(_) = slot {
                fixed_slot(i, op, seen_src, &mut lo)?;
            }
            continue;
        };
        match slot {
            Slot::SDst(_) => {
                let regs = dst_dwords(op, w, i)?;
                if i.enc == Enc::Sopk
                    && (name.starts_with("s_cmpk") || name.starts_with("s_setreg"))
                {
                    lo.srcs
                        .push(regs.into_iter().map(Rd::Reg).collect::<Vec<_>>());
                } else {
                    if i.enc == Enc::Sopk && matches!(name, "s_addk_i32" | "s_mulk_i32") {
                        lo.srcs
                            .push(regs.iter().copied().map(Rd::Reg).collect::<Vec<_>>());
                    }
                    if name.starts_with("s_cmov")
                        || name.starts_with("s_cmovk")
                        || name.starts_with("s_bitset")
                    {
                        lo.olds.extend(&regs);
                    }
                    lo.dsts.extend(regs);
                }
            }
            Slot::SDstB(_) | Slot::VDstS(_) => lo.dsts.extend(dst_dwords(op, w, i)?),
            // an atomic returns its old value only with glc
            Slot::VDst(_) if i.enc == Enc::Flat && is_flat_atomic(i) && !i.mem.glc => {}
            Slot::VDst(_) => {
                let regs = dst_dwords(op, w, i)?;
                lo.olds.extend(&regs);
                lo.dsts.extend(regs);
            }
            Slot::SData(_) => {
                if name.contains("atomic") {
                    return Err(bad(
                        i.at,
                        IrErrorKind::Unsupported(format!("{name}: scalar atomic")),
                    ));
                }
                if name.contains("store") {
                    seen_src = true;
                    lo.srcs.push(src_dwords(op, w, i)?);
                } else {
                    lo.dsts.extend(dst_dwords(op, w, i)?);
                }
            }
            Slot::Vdata(_) => {}
            _ => {
                seen_src = true;
                lo.srcs.push(src_dwords(op, w, i)?);
            }
        }
    }
    match i.enc {
        Enc::Smem => {
            if i.info.slots.iter().any(|s| matches!(s, Slot::SBase(_))) && i.mem.soffset != 125 {
                lo.srcs.push(vec![
                    operand_dword(i.mem.soffset, 0, false, i.at)?.into_src()
                ]);
            }
        }
        Enc::Vintrp => {
            let t = i.intrp;
            lo.dsts.push(Reg::V(u16::from(t.vdst)));
            lo.olds.push(Reg::V(u16::from(t.vdst)));
            if name != "v_interp_mov_f32" {
                lo.srcs.push(vec![Rd::Reg(Reg::V(u16::from(t.vsrc)))]);
            }
            lo.imps.push(Reg::M0);
        }
        Enc::Exp => {
            let e = i.exp;
            if e.compr {
                for k in 0..2 {
                    if e.en >> (2 * k) & 3 != 0 {
                        lo.srcs
                            .push(vec![Rd::Reg(Reg::V(u16::from(e.vsrc[k as usize])))]);
                    }
                }
            } else {
                for k in 0..4 {
                    if e.en >> k & 1 != 0 {
                        lo.srcs
                            .push(vec![Rd::Reg(Reg::V(u16::from(e.vsrc[k as usize])))]);
                    }
                }
            }
        }
        Enc::Mubuf | Enc::Mtbuf => lower_mubuf(i, &mut lo)?,
        Enc::Mimg => lower_mimg(i, &mut lo)?,
        Enc::Ds => lower_ds(i, &mut lo)?,
        Enc::Flat => lower_flat(i, &mut lo)?,
        _ => {}
    }
    if vector && !matches!(name, "v_readlane_b32" | "v_writelane_b32") {
        lo.imps.extend(mask_regs(i.wave32));
    }
    if name.starts_with("v_cmpx_") {
        lo.dsts.extend(mask_regs(i.wave32));
    }
    if name.starts_with("v_div_fmas") {
        lo.imps.extend(vcc_regs(i.wave32));
    }
    if i.enc == Enc::Vop3 && name.starts_with("v_interp") {
        lo.imps.push(Reg::M0);
    }
    if name.starts_with("s_sendmsg") {
        lo.imps.push(Reg::M0);
    }
    if name.contains("saveexec") || name.contains("wrexec") {
        lo.imps.extend(mask_regs(i.wave32));
        lo.dsts.extend(mask_regs(i.wave32));
    }
    if reads_scc(i) {
        lo.imps.push(Reg::Scc);
    }
    if writes_scc(i) {
        lo.dsts.push(Reg::Scc);
    }
    Ok(Some(lo))
}

impl Rd {
    fn into_src(self) -> Rd {
        match self {
            Rd::Reg(Reg::Null) => Rd::Const(0),
            r => r,
        }
    }
}

/// An implicit `vcc` operand of a VOP: a destination before the first source, a source after
/// (`v_add_co_ci_u32` has both); a register standing in for it (SDWA compare) is a destination.
fn fixed_slot(i: &Instr, op: &Operand, seen_src: bool, lo: &mut Lowered) -> Result<(), IrError> {
    let w = if i.wave32 { 1 } else { 2 };
    if matches!(op, Operand::Reg { .. }) || !seen_src {
        lo.dsts.extend(dst_dwords(op, w, i)?);
    } else {
        lo.srcs.push(src_dwords(op, w, i)?);
    }
    Ok(())
}

fn lower_mubuf(i: &Instr, lo: &mut Lowered) -> Result<(), IrError> {
    let name = i.name();
    let m = i.mem;
    if m.lds {
        return Err(bad(
            i.at,
            IrErrorKind::Unsupported(format!("{name}: load to LDS")),
        ));
    }
    let Some((Slot::Vdata(_), Operand::Reg { code, count })) =
        i.info.slots.first().zip(i.operands.first())
    else {
        return Ok(());
    };
    let first = *code - 256;
    let n = u16::from(*count);
    let data: Vec<Reg> = (0..n).map(|k| Reg::V(first + k)).collect();
    // address, then resource, then offset
    let mut addr = Vec::new();
    let mut a = u16::from(m.vaddr);
    if m.idxen {
        addr.push(Rd::Reg(Reg::V(a)));
        a += 1;
    }
    if m.offen {
        addr.push(Rd::Reg(Reg::V(a)));
    }
    if name.contains("load") {
        lo.olds.extend(&data);
        lo.dsts.extend(data);
        if !addr.is_empty() {
            lo.srcs.push(addr);
        }
    } else if name.contains("store") {
        lo.srcs.push(data.into_iter().map(Rd::Reg).collect());
        if !addr.is_empty() {
            lo.srcs.push(addr);
        }
    } else if name.contains("atomic") {
        lo.srcs.push(data.iter().copied().map(Rd::Reg).collect());
        if m.glc {
            let keep = if name.contains("cmpswap") {
                data.len() / 2
            } else {
                data.len()
            };
            lo.olds.extend(&data[..keep]);
            lo.dsts.extend(&data[..keep]);
        }
        if !addr.is_empty() {
            lo.srcs.push(addr);
        }
    } else {
        return Err(bad(
            i.at,
            IrErrorKind::Unsupported(format!("{name}: buffer opcode")),
        ));
    }
    let r = u16::from(m.srsrc);
    lo.srcs
        .push((0..4).map(|k| Rd::Reg(Reg::S(r + k))).collect());
    lo.srcs
        .push(vec![operand_dword(m.soffset, 0, false, i.at)?.into_src()]);
    Ok(())
}

fn is_flat_atomic(i: &Instr) -> bool {
    i.info.flags & gfx10::ops::FLAG_FLAT_ATOMIC != 0
}

fn vgprs(first: u16, n: u16, at: usize) -> Result<Vec<Reg>, IrError> {
    if first + n > 256 {
        return Err(bad(at, IrErrorKind::BadOperand("VGPR range past v255")));
    }
    Ok((first..first + n).map(Reg::V).collect())
}

fn sgprs(first: u16, n: u16, at: usize) -> Result<Vec<Rd>, IrError> {
    (0..n)
        .map(|k| operand_dword(first + k, 0, false, at))
        .collect()
}

fn vec_rd(regs: &[Reg]) -> Vec<Rd> {
    regs.iter().copied().map(Rd::Reg).collect()
}

/// The register count of a `Reg` operand, or `None` for `off`.
fn reg_count(op: &Operand) -> Option<u8> {
    match op {
        Operand::Reg { count, .. } => Some(*count),
        _ => None,
    }
}

/// The operand printed for the first slot `f` accepts, if the opcode has one.
fn slot_operand(i: &Instr, f: impl Fn(&Slot) -> bool) -> Option<&Operand> {
    i.info
        .slots
        .iter()
        .zip(&i.operands)
        .find_map(|(s, o)| f(s).then_some(o))
}

/// MIMG. Data dwords come from the decoder (`ndata`: dmask channels, halved by d16, plus the
/// tfe dword); `lwe` adds the same extra dword (LLVM sizes the data as `tfe || lwe`). A load
/// or sample writes them; a store reads them; an atomic reads them and, with glc, writes its
/// old value (half of the operands for compare-and-swap).
fn lower_mimg(i: &Instr, lo: &mut Lowered) -> Result<(), IrError> {
    let name = i.name();
    let m = i.img;
    let flags = i.info.flags;
    let atomic = flags & gfx10::ops::FLAG_IMG_ATOMIC != 0;
    let store = name.starts_with("image_store");
    let bvh = flags & gfx10::ops::FLAG_IMG_BVH != 0;
    if m.r128 && !bvh {
        return Err(bad(
            i.at,
            IrErrorKind::Unsupported(format!("{name}: r128 resource")),
        ));
    }
    if (store || atomic) && (m.tfe || m.lwe) {
        return Err(bad(
            i.at,
            IrErrorKind::Unsupported(format!("{name}: tfe / lwe on a store or atomic")),
        ));
    }
    let ndata = u16::from(m.ndata) + u16::from(m.lwe && !m.tfe);
    let data = vgprs(u16::from(m.vdata), ndata, i.at)?;
    let naddr = u16::from(m.naddr);
    let addr = if m.nsa == 0 {
        vgprs(u16::from(m.vaddr[0]), naddr, i.at)?
    } else {
        m.vaddr[..usize::from(m.naddr)]
            .iter()
            .map(|v| Reg::V(u16::from(*v)))
            .collect()
    };
    if store {
        lo.srcs.push(vec_rd(&data));
    } else if atomic {
        lo.srcs.push(vec_rd(&data));
        if m.glc {
            let keep = if flags & gfx10::ops::FLAG_IMG_CMPSWAP != 0 {
                data.len() / 2
            } else {
                data.len()
            };
            lo.olds.extend(&data[..keep]);
            lo.dsts.extend(&data[..keep]);
        }
    } else {
        lo.olds.extend(&data);
        lo.dsts.extend(data);
    }
    if !addr.is_empty() {
        lo.srcs.push(vec_rd(&addr));
    }
    let nrsrc = if bvh { 4 } else { 8 };
    lo.srcs.push(sgprs(u16::from(m.srsrc), nrsrc, i.at)?);
    if i.info.slots.contains(&Slot::ImgSamp) {
        lo.srcs.push(sgprs(u16::from(m.ssamp), 4, i.at)?);
    }
    Ok(())
}

/// DS. On GFX9 and later the LDS needs no M0 (LLVM: `ldsRequiresM0Init` is generation < 9);
/// M0 is still read by the GDS forms (the `gds` bit), `ds_ordered_count`, `ds_append` /
/// `ds_consume` (M0 is their address) and the `*_addtid` forms (M0 is their base). The GWS
/// opcodes take their operand from `data0`, which the decoder prints as the address: not
/// modelled.
fn lower_ds(i: &Instr, lo: &mut Lowered) -> Result<(), IrError> {
    let name = i.name();
    if name.starts_with("ds_gws_") {
        return Err(bad(
            i.at,
            IrErrorKind::Unsupported(format!("{name}: global wave sync")),
        ));
    }
    for (s, o) in i.info.slots.iter().zip(&i.operands) {
        if matches!(s, Slot::DsAddr | Slot::DsData0(_) | Slot::DsData1(_)) {
            lo.srcs.push(src_dwords(o, reg_count(o).unwrap_or(1), i)?);
        }
    }
    if i.mem.gds
        || matches!(name, "ds_ordered_count" | "ds_append" | "ds_consume")
        || name.ends_with("_addtid_b32")
    {
        lo.imps.push(Reg::M0);
    }
    Ok(())
}

/// FLAT / GLOBAL / SCRATCH: address (a pair, a word, or `off`), data, then the scalar base.
/// The hidden flat-scratch base of the flat and scratch segments is not a register of the IR.
fn lower_flat(i: &Instr, lo: &mut Lowered) -> Result<(), IrError> {
    if i.mem.lds {
        return Err(bad(
            i.at,
            IrErrorKind::Unsupported(format!("{}: load to LDS", i.name())),
        ));
    }
    for want in [0, 1, 2] {
        let o = slot_operand(i, |s| match want {
            0 => matches!(s, Slot::FlatAddr),
            1 => matches!(s, Slot::Vdata(_)),
            _ => matches!(s, Slot::FlatSaddr),
        });
        if let Some(n) = o.and_then(reg_count) {
            lo.srcs.push(src_dwords(o.unwrap(), n, i)?);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Constant folding.

struct Folded {
    vals: Vec<Src>,
    scc: Option<Src>,
}

fn is_known(s: &Src) -> bool {
    matches!(s, Src::Const(_) | Src::CodeLo(_) | Src::CodeHi)
}

fn bool_src(b: bool) -> Option<Src> {
    Some(Src::Const(u32::from(b)))
}

/// Folds `i` when every input is known. `scc_in` is the resolved SCC read, if the opcode has one.
fn fold(i: &Instr, srcs: &[Vec<Src>], scc_in: Option<Src>, code: &[u8]) -> Option<Folded> {
    let name = i.name();
    let one = |s: Src| Folded {
        vals: vec![s],
        scc: None,
    };
    match name {
        "s_getpc_b64" => {
            return Some(Folded {
                vals: vec![Src::CodeLo((i.at + i.len) as u32), Src::CodeHi],
                scc: None,
            })
        }
        "s_mov_b32" | "s_movk_i32" if srcs.first().is_some_and(|s| is_known(&s[0])) => {
            return Some(one(srcs[0][0]))
        }
        "s_mov_b64" if srcs.first().is_some_and(|s| s.iter().all(is_known)) => {
            return Some(Folded {
                vals: srcs[0].clone(),
                scc: None,
            })
        }
        // the carry (or borrow) of the low half is 0 (the code is aligned), so the high half
        // keeps its value: far jumps back are `s_sub_u32` + `s_subb_u32`
        "s_addc_u32" | "s_subb_u32" => {
            if let ([Src::CodeHi], [Src::Const(0)], Some(Src::Const(0))) =
                (&srcs[0][..], &srcs[1][..], scc_in)
            {
                return Some(Folded {
                    vals: vec![Src::CodeHi],
                    scc: bool_src(false),
                });
            }
            return None;
        }
        _ => {}
    }
    if name.starts_with("s_load_dword") {
        return fold_load(i, srcs, code);
    }
    if i.enc == Enc::Sopc {
        return fold_compare(name, srcs);
    }
    if i.enc != Enc::Sop1 && i.enc != Enc::Sop2 {
        return None;
    }
    // address + constant offset
    if matches!(name, "s_add_u32" | "s_add_i32" | "s_sub_u32" | "s_sub_i32") {
        let (a, b) = (srcs[0][0], srcs[1][0]);
        let add = name.starts_with("s_add");
        let moved = match (a, b) {
            (Src::CodeLo(o), Src::Const(k)) => Some((o, k, add)),
            (Src::Const(k), Src::CodeLo(o)) if add => Some((o, k, true)),
            _ => None,
        };
        if let Some((o, k, add)) = moved {
            let d = i64::from(k as i32);
            let n = i64::from(o) + if add { d } else { -d };
            return Some(Folded {
                vals: vec![Src::CodeLo(u32::try_from(n).ok()?)],
                scc: bool_src(false),
            });
        }
    }
    let wide = name.ends_with("_b64");
    let all: Option<Vec<u32>> = srcs
        .iter()
        .flatten()
        .map(|s| match s {
            Src::Const(c) => Some(*c),
            _ => None,
        })
        .collect();
    let all = all?;
    let fold1 = |r: u32, scc: bool| Folded {
        vals: vec![Src::Const(r)],
        scc: bool_src(scc),
    };
    let base = name
        .strip_suffix("_b64")
        .or_else(|| name.strip_suffix("_b32"))
        .or_else(|| name.strip_suffix("_u32"))
        .or_else(|| name.strip_suffix("_i32"));
    match (name, base) {
        ("s_add_u32", _) => {
            let (r, c) = all[0].overflowing_add(all[1]);
            Some(fold1(r, c))
        }
        ("s_sub_u32", _) => {
            let (r, c) = all[0].overflowing_sub(all[1]);
            Some(fold1(r, c))
        }
        ("s_add_i32", _) => {
            let (r, c) = (all[0] as i32).overflowing_add(all[1] as i32);
            Some(fold1(r as u32, c))
        }
        ("s_sub_i32", _) => {
            let (r, c) = (all[0] as i32).overflowing_sub(all[1] as i32);
            Some(fold1(r as u32, c))
        }
        ("s_lshl_b32", _) => {
            let r = all[0] << (all[1] & 31);
            Some(fold1(r, r != 0))
        }
        ("s_lshr_b32", _) => {
            let r = all[0] >> (all[1] & 31);
            Some(fold1(r, r != 0))
        }
        ("s_ashr_i32", _) => {
            let r = ((all[0] as i32) >> (all[1] & 31)) as u32;
            Some(fold1(r, r != 0))
        }
        ("s_mul_i32", _) => Some(Folded {
            vals: vec![Src::Const(all[0].wrapping_mul(all[1]))],
            scc: None,
        }),
        (_, Some(b)) => {
            let n = if wide { 2 } else { 1 };
            let f: fn(u32, u32) -> u32 = match b {
                "s_and" => |a, b| a & b,
                "s_or" => |a, b| a | b,
                "s_xor" => |a, b| a ^ b,
                "s_andn2" => |a, b| a & !b,
                "s_orn2" => |a, b| a | !b,
                "s_nand" => |a, b| !(a & b),
                "s_nor" => |a, b| !(a | b),
                "s_xnor" => |a, b| !(a ^ b),
                "s_not" => |a, _| !a,
                _ => return None,
            };
            let unary = b == "s_not";
            if all.len() != if unary { n } else { 2 * n } {
                return None;
            }
            let vals: Vec<u32> = (0..n)
                .map(|k| f(all[k], if unary { 0 } else { all[n + k] }))
                .collect();
            let nz = vals.iter().any(|v| *v != 0);
            Some(Folded {
                vals: vals.into_iter().map(Src::Const).collect(),
                scc: bool_src(nz),
            })
        }
        _ => None,
    }
}

/// The SCC of a scalar compare of constants.
fn fold_compare(name: &str, srcs: &[Vec<Src>]) -> Option<Folded> {
    let (a, b) = (srcs.first()?, srcs.get(1)?);
    let konst = |s: &[Src]| -> Option<Vec<u32>> {
        s.iter()
            .map(|x| match x {
                Src::Const(c) => Some(*c),
                _ => None,
            })
            .collect()
    };
    let (a, b) = (konst(a)?, konst(b)?);
    let rest = name.strip_prefix("s_cmp_")?;
    let (op, ty) = rest.rsplit_once('_')?;
    let scc = match (ty, a.len(), b.len()) {
        ("u32", 1, 1) | ("i32", 1, 1) => {
            let (x, y) = (a[0], b[0]);
            let ord = if ty == "i32" {
                (x as i32).cmp(&(y as i32))
            } else {
                x.cmp(&y)
            };
            match op {
                "eq" => ord.is_eq(),
                "lg" => ord.is_ne(),
                "gt" => ord.is_gt(),
                "ge" => ord.is_ge(),
                "lt" => ord.is_lt(),
                "le" => ord.is_le(),
                _ => return None,
            }
        }
        ("u64", 2, 2) => match op {
            "eq" => a == b,
            "lg" => a != b,
            _ => return None,
        },
        _ => return None,
    };
    Some(Folded {
        vals: vec![],
        scc: bool_src(scc),
    })
}

/// `s_load_dword*` from `code` itself (constant data after the instructions).
fn fold_load(i: &Instr, srcs: &[Vec<Src>], code: &[u8]) -> Option<Folded> {
    let base = srcs.first()?;
    let Src::CodeLo(o) = *base.first()? else {
        return None;
    };
    if base.get(1) != Some(&Src::CodeHi) {
        return None;
    }
    let mut off = i64::from(o) + i64::from(i.mem.offset);
    if let Some(s) = srcs.get(1) {
        let Src::Const(k) = s[0] else { return None };
        off += i64::from(k);
    }
    let n = match i.name() {
        "s_load_dword" => 1,
        "s_load_dwordx2" => 2,
        "s_load_dwordx4" => 4,
        "s_load_dwordx8" => 8,
        "s_load_dwordx16" => 16,
        _ => return None,
    };
    let off = usize::try_from(off).ok()?;
    let bytes = code.get(off..off.checked_add(4 * n)?)?;
    Some(Folded {
        vals: (0..n)
            .map(|k| {
                let b = &bytes[4 * k..4 * k + 4];
                Src::Const(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            })
            .collect(),
        scc: None,
    })
}

// ---------------------------------------------------------------------------------------
// Building.

enum RawTerm {
    Jump(BlockId),
    Branch {
        op: &'static str,
        regs: Vec<Reg>,
        taken: BlockId,
        fall: BlockId,
    },
    /// `s_setpc_b64` of `regs`; `target` once a build found where it goes.
    SetPc {
        at: usize,
        regs: [Reg; 2],
        target: Option<BlockId>,
    },
    End,
}

struct RawBlock {
    at: usize,
    insts: Vec<(Instr, Lowered)>,
    term: RawTerm,
}

/// Most `s_setpc_b64` targets resolved per program (the corpus has one).
const MAX_JUMPS: usize = 16;

/// What one build attempt found.
enum Step {
    Done(Func),
    /// An `s_setpc_b64` at `at` whose address folded to byte `target`: follow it and build again.
    Jump {
        at: usize,
        target: usize,
    },
}

/// True when the instruction at `entry` is the call to the vertex fetch routine,
/// `s_swappc_b64 s[4:5], s[4:5]`.
fn opens_with_fetch_call(code: &[u8], entry: usize, wave32: bool) -> bool {
    let Ok(i) = gfx10::decode_wave(code, entry, wave32) else {
        return false;
    };
    i.name() == "s_swappc_b64"
        && matches!(
            &i.operands[..],
            [
                Operand::Reg { code: 4, count: 2 },
                Operand::Reg { code: 4, count: 2 }
            ]
        )
}

/// The scalar registers known to hold a constant or a code address at one instruction.
type Known = BTreeMap<Reg, Src>;

/// Resolves the jumps through a folded `CodeLo` / `CodeHi` pair of everything reachable from
/// `entry` in one forward pass: a worklist over instructions that carries the known scalar
/// values (the folds of `fold`) and keeps, at a join, the values every path agrees on. This
/// replaces one full rebuild per jump; a jump whose target is not known here is left to the
/// per-build loop of `build`.
fn resolve_jumps(
    code: &[u8],
    entry: usize,
    wave32: bool,
    resolved: &mut BTreeMap<usize, Resolved>,
) {
    let mut ins: HashMap<usize, Known> = HashMap::new();
    ins.insert(entry, Known::new());
    let mut work = vec![entry];
    let mut jumps: BTreeMap<usize, usize> = BTreeMap::new();
    while let Some(at) = work.pop() {
        let Some(mut st) = ins.get(&at).cloned() else {
            continue;
        };
        let Ok(i) = gfx10::decode_wave(code, at, wave32) else {
            continue;
        };
        let next = at + i.len;
        let mut succs: Vec<usize> = Vec::new();
        let target = i
            .branch_target()
            .filter(|t| *t >= 0 && (*t as usize) < code.len() && t % 4 == 0)
            .map(|t| t as usize);
        if i.is_endpgm() {
        } else if i.is_branch() {
            succs.extend(target);
        } else if i.is_cbranch() {
            succs.extend(target);
            succs.push(next);
        } else if i.is_indirect() {
            if resolved.get(&at) == Some(&Resolved::Call) {
                st.clear();
                succs.push(next);
            } else if i.name() == "s_setpc_b64" {
                // the last visit has the final state: an earlier one may have known more
                jumps.remove(&at);
                let pair = match &i.operands[..] {
                    [Operand::Reg { code, count: 2 }] => {
                        match (
                            operand_dword(*code, 0, false, at),
                            operand_dword(*code, 1, false, at),
                        ) {
                            (Ok(Rd::Reg(lo)), Ok(Rd::Reg(hi))) => Some([lo, hi]),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                if let Some([lo, hi]) = pair {
                    if let (Some(Src::CodeLo(t)), Some(Src::CodeHi)) = (st.get(&lo), st.get(&hi)) {
                        let t = *t as usize;
                        if t < code.len() && t.is_multiple_of(4) {
                            jumps.insert(at, t);
                            succs.push(t);
                        }
                    }
                }
            }
        } else {
            match lower(&i) {
                Ok(Some(l)) => {
                    apply_known(&i, &l, &mut st, code);
                    succs.push(next);
                }
                Ok(None) => succs.push(next),
                Err(_) => {}
            }
        }
        for t in succs {
            let changed = match ins.get_mut(&t) {
                None => {
                    ins.insert(t, st.clone());
                    true
                }
                Some(old) => {
                    let n = old.len();
                    old.retain(|r, v| st.get(r) == Some(v));
                    old.len() != n
                }
            };
            if changed {
                work.push(t);
            }
        }
    }
    for (at, t) in jumps {
        resolved.entry(at).or_insert(Resolved::Jump(t));
    }
}

/// The effect of `i` on the known values: a fold, else its destinations become unknown.
fn apply_known(i: &Instr, l: &Lowered, st: &mut Known, code: &[u8]) {
    if l.dsts.is_empty() {
        return;
    }
    let srcs: Vec<Vec<Src>> = l
        .srcs
        .iter()
        .map(|s| {
            s.iter()
                .map(|r| match r {
                    Rd::Const(c) => Src::Const(*c),
                    Rd::Reg(Reg::Null) => Src::Const(0),
                    Rd::Reg(r) => st.get(r).copied().unwrap_or(Src::Undef),
                })
                .collect()
        })
        .collect();
    let scc_in = l
        .imps
        .contains(&Reg::Scc)
        .then(|| st.get(&Reg::Scc).copied().unwrap_or(Src::Undef));
    if let Some(fd) = fold(i, &srcs, scc_in, code) {
        let explicit: Vec<Reg> = l.dsts.iter().copied().filter(|r| *r != Reg::Scc).collect();
        let wants_scc = l.dsts.contains(&Reg::Scc);
        if fd.vals.len() == explicit.len() && (!wants_scc || fd.scc.is_some()) {
            for (r, v) in explicit.iter().zip(&fd.vals) {
                if *r != Reg::Null {
                    st.insert(*r, *v);
                }
            }
            if let Some(s) = fd.scc.filter(|_| wants_scc) {
                st.insert(Reg::Scc, s);
            }
            return;
        }
    }
    for r in &l.dsts {
        st.remove(r);
    }
}

/// Follows the code from `entry` and builds its SSA IR.
///
/// `s_setpc_b64` of a folded `CodeLo` / `CodeHi` pair is a direct jump, followed as such. When
/// `regs.fetch` is set and the program opens with the fetch call, the call is an instruction
/// defining the attribute VGPRs. Any other jump through a register is `IrErrorKind::Indirect`.
/// Branches whose condition folded to a constant are then pruned (`prune`).
pub fn build(code: &[u8], entry: usize, wave32: bool, regs: &EntryRegs) -> Result<Func, IrError> {
    let mut resolved: BTreeMap<usize, Resolved> = BTreeMap::new();
    if regs.fetch.is_some() && opens_with_fetch_call(code, entry, wave32) {
        resolved.insert(entry, Resolved::Call);
    }
    resolve_jumps(code, entry, wave32, &mut resolved);
    for _ in 0..MAX_JUMPS {
        let prog = gfx10::follow_with(code, entry, wave32, &resolved)?;
        for &at in &prog.indirect {
            let setpc = prog
                .instrs
                .binary_search_by_key(&at, |i| i.at)
                .is_ok_and(|k| prog.instrs[k].name() == "s_setpc_b64");
            if !setpc {
                return Err(bad(at, IrErrorKind::Indirect));
            }
        }
        let raw = split_blocks(&prog, &resolved, regs)?;
        let mut b = Builder {
            code,
            wave32,
            regs,
            fetch_at: resolved
                .iter()
                .find_map(|(at, r)| (*r == Resolved::Call).then_some(*at)),
            vals: Vec::new(),
            args: BTreeMap::new(),
        };
        match b.run(raw, entry)? {
            Step::Done(f) => return Ok(f),
            Step::Jump { at, target } => {
                resolved.insert(at, Resolved::Jump(target));
            }
        }
    }
    let at = resolved.keys().next_back().copied().unwrap_or(entry);
    Err(bad(at, IrErrorKind::Indirect))
}

fn split_blocks(
    prog: &gfx10::Program,
    resolved: &BTreeMap<usize, Resolved>,
    regs: &EntryRegs,
) -> Result<Vec<RawBlock>, IrError> {
    let ids: HashMap<usize, BlockId> = prog
        .leaders
        .iter()
        .enumerate()
        .map(|(n, &at)| (at, n))
        .collect();
    let mut blocks: Vec<RawBlock> = Vec::new();
    let mut cur: Option<RawBlock> = None;
    let n = prog.instrs.len();
    for (k, i) in prog.instrs.iter().enumerate() {
        let b = cur.get_or_insert_with(|| RawBlock {
            at: i.at,
            insts: Vec::new(),
            term: RawTerm::End,
        });
        let next = prog.instrs.get(k + 1);
        let falls = next.filter(|x| x.at == i.at + i.len);
        let target = |t: i64| ids[&(t as usize)];
        let fetch_call = i.is_indirect() && resolved.get(&i.at) == Some(&Resolved::Call);
        let jump_through_register = i.is_indirect() && !fetch_call;
        let mut term = None;
        if i.is_endpgm() {
            term = Some(RawTerm::End);
        } else if i.is_branch() {
            term = Some(RawTerm::Jump(target(i.branch_target().unwrap_or(0))));
        } else if i.is_cbranch() {
            let regs = match i.name() {
                "s_cbranch_scc0" | "s_cbranch_scc1" => vec![Reg::Scc],
                "s_cbranch_vccz" | "s_cbranch_vccnz" => vcc_regs(i.wave32),
                "s_cbranch_execz" | "s_cbranch_execnz" => mask_regs(i.wave32),
                n => {
                    return Err(bad(
                        i.at,
                        IrErrorKind::Unsupported(format!("{n}: debug branch")),
                    ))
                }
            };
            let fall = falls.ok_or_else(|| {
                bad(
                    i.at,
                    IrErrorKind::Decode(DecodeError {
                        at: i.at,
                        kind: gfx10::ErrorKind::Truncated,
                    }),
                )
            })?;
            term = Some(RawTerm::Branch {
                op: i.info.name,
                regs,
                taken: target(i.branch_target().unwrap_or(0)),
                fall: ids[&fall.at],
            });
        } else if jump_through_register {
            let pair = match &i.operands[..] {
                [Operand::Reg { code, count: 2 }] => {
                    match (
                        operand_dword(*code, 0, false, i.at)?,
                        operand_dword(*code, 1, false, i.at)?,
                    ) {
                        (Rd::Reg(lo), Rd::Reg(hi)) => Some([lo, hi]),
                        _ => None,
                    }
                }
                _ => None,
            };
            let Some(regs) = pair else {
                return Err(bad(i.at, IrErrorKind::Indirect));
            };
            let target = match resolved.get(&i.at) {
                Some(Resolved::Jump(t)) => Some(ids[t]),
                _ => None,
            };
            term = Some(RawTerm::SetPc {
                at: i.at,
                regs,
                target,
            });
        } else if let Some(nx) = falls.filter(|x| ids.contains_key(&x.at)) {
            term = Some(RawTerm::Jump(ids[&nx.at]));
        } else if k + 1 == n || falls.is_none() {
            return Err(bad(
                i.at,
                IrErrorKind::Decode(DecodeError {
                    at: i.at,
                    kind: gfx10::ErrorKind::Truncated,
                }),
            ));
        }
        if fetch_call {
            b.insts.push((i.clone(), fetch_lowered(i, regs)));
        } else if !(i.is_endpgm() || i.is_branch() || i.is_cbranch() || jump_through_register) {
            if let Some(l) = lower(i)? {
                b.insts.push((i.clone(), l));
            }
        }
        if let Some(t) = term {
            let mut done = cur.take().unwrap_or(RawBlock {
                at: 0,
                insts: vec![],
                term: RawTerm::End,
            });
            done.term = t;
            blocks.push(done);
        }
    }
    Ok(blocks)
}

/// The fetch call as an instruction: it reads the routine's address (`s[4:5]`), writes the
/// return address back there and the attribute VGPRs, under EXEC.
fn fetch_lowered(i: &Instr, regs: &EntryRegs) -> Lowered {
    let attrs = regs.fetch.as_deref().unwrap_or_default();
    Lowered {
        dsts: attrs
            .iter()
            .map(|&v| Reg::V(v))
            .chain([Reg::S(4), Reg::S(5)])
            .collect(),
        srcs: vec![vec![Rd::Reg(Reg::S(4)), Rd::Reg(Reg::S(5))]],
        imps: mask_regs(i.wave32),
        olds: Vec::new(),
    }
}

struct Builder<'a> {
    code: &'a [u8],
    wave32: bool,
    regs: &'a EntryRegs,
    /// Offset of the fetch call, whose `s[4:5]` read is an entry value.
    fetch_at: Option<usize>,
    vals: Vec<ValInfo>,
    args: BTreeMap<Reg, Val>,
}

type State = HashMap<Reg, Src>;

impl Builder<'_> {
    fn new_val(&mut self, reg: Reg, def: ValDef) -> Val {
        self.vals.push(ValInfo { reg, def });
        Val(self.vals.len() as u32 - 1)
    }

    fn arg(&mut self, reg: Reg) -> Src {
        if let Some(v) = self.args.get(&reg) {
            return Src::Val(*v);
        }
        let v = self.new_val(reg, ValDef::Arg);
        self.args.insert(reg, v);
        Src::Val(v)
    }

    /// The value of `reg` now. `strict`: an undefined register is an error, else `undef`.
    fn read(&mut self, st: &State, reg: Reg, at: usize, strict: bool) -> Result<Src, IrError> {
        if let Some(s) = st.get(&reg) {
            return Ok(*s);
        }
        if self.regs.has(reg) {
            return Ok(self.arg(reg));
        }
        if strict {
            Err(bad(at, IrErrorKind::UndefinedRead(reg)))
        } else {
            Ok(Src::Undef)
        }
    }

    fn run(&mut self, raw: Vec<RawBlock>, entry: usize) -> Result<Step, IrError> {
        let n = raw.len();
        let entry_id = raw.iter().position(|b| b.at == entry).unwrap_or(0);
        let succs: Vec<Vec<BlockId>> = raw
            .iter()
            .map(|b| match &b.term {
                RawTerm::Jump(t) => vec![*t],
                RawTerm::Branch { taken, fall, .. } => {
                    if taken == fall {
                        vec![*taken]
                    } else {
                        vec![*taken, *fall]
                    }
                }
                RawTerm::SetPc { target, .. } => target.iter().copied().collect(),
                RawTerm::End => vec![],
            })
            .collect();
        let mut preds = vec![Vec::new(); n];
        for (b, s) in succs.iter().enumerate() {
            for &t in s {
                preds[t].push(b);
            }
        }
        let (rpo, idom) = dominators(n, entry_id, &succs, &preds);

        // registers read somewhere, and the blocks defining each
        let mut used: HashSet<Reg> = HashSet::new();
        let mut defs: BTreeMap<Reg, BTreeSet<BlockId>> = BTreeMap::new();
        for (b, rb) in raw.iter().enumerate() {
            for (_, l) in &rb.insts {
                used.extend(l.srcs.iter().flatten().filter_map(|r| match r {
                    Rd::Reg(r) => Some(*r),
                    Rd::Const(_) => None,
                }));
                used.extend(&l.imps);
                used.extend(&l.olds);
                for d in &l.dsts {
                    defs.entry(*d).or_default().insert(b);
                }
            }
            match &rb.term {
                RawTerm::Branch { regs, .. } => used.extend(regs),
                RawTerm::SetPc { regs, .. } => used.extend(regs),
                _ => {}
            }
        }
        let df = dominance_frontier(n, entry_id, &idom, &preds);
        let mut phi_regs: Vec<Vec<Reg>> = vec![Vec::new(); n];
        for (reg, blocks) in &defs {
            if !used.contains(reg) || *reg == Reg::Null {
                continue;
            }
            let mut has = HashSet::new();
            let mut work: Vec<BlockId> = blocks.iter().copied().collect();
            let mut queued: HashSet<BlockId> = work.iter().copied().collect();
            while let Some(b) = work.pop() {
                for &f in &df[b] {
                    if has.insert(f) {
                        phi_regs[f].push(*reg);
                        if queued.insert(f) {
                            work.push(f);
                        }
                    }
                }
            }
        }

        let mut blocks: Vec<Block> = raw
            .iter()
            .enumerate()
            .map(|(k, b)| Block {
                at: b.at,
                phis: Vec::new(),
                insts: Vec::new(),
                term: Term::End,
                preds: preds[k].clone(),
            })
            .collect();
        let mut out: Vec<State> = vec![State::new(); n];
        for &b in &rpo {
            let mut st = match idom[b] {
                Some(d) if b != entry_id => out[d].clone(),
                _ => State::new(),
            };
            for &reg in &phi_regs[b] {
                let index = blocks[b].phis.len();
                let dst = self.new_val(reg, ValDef::Phi { block: b, index });
                blocks[b].phis.push(Phi {
                    reg,
                    dst,
                    args: Vec::new(),
                });
                st.insert(reg, Src::Val(dst));
            }
            for (instr, l) in &raw[b].insts {
                self.rename(instr, l, &mut st, &mut blocks[b], b)?;
            }
            blocks[b].term = match &raw[b].term {
                RawTerm::Jump(t) => Term::Jump(*t),
                RawTerm::End => Term::End,
                RawTerm::SetPc { at, regs, target } => {
                    let lo = self.read(&st, regs[0], *at, false)?;
                    let hi = self.read(&st, regs[1], *at, false)?;
                    match (lo, hi, target) {
                        (Src::CodeLo(t), Src::CodeHi, Some(tb)) if raw[*tb].at == t as usize => {
                            Term::Jump(*tb)
                        }
                        (Src::CodeLo(t), Src::CodeHi, None) => {
                            return Ok(Step::Jump {
                                at: *at,
                                target: t as usize,
                            })
                        }
                        _ => return Err(bad(*at, IrErrorKind::Indirect)),
                    }
                }
                RawTerm::Branch {
                    op,
                    regs,
                    taken,
                    fall,
                } => {
                    let at = raw[b].insts.last().map_or(raw[b].at, |(i, _)| i.at);
                    let mut cond = Vec::new();
                    for &r in regs {
                        cond.push((r, self.read(&st, r, at, true)?));
                    }
                    Term::Branch {
                        op,
                        cond,
                        taken: *taken,
                        fall: *fall,
                    }
                }
            };
            out[b] = st;
        }
        // phi inputs
        for blk in &mut blocks {
            let preds = blk.preds.clone();
            for phi in &mut blk.phis {
                let reg = phi.reg;
                let mut args = Vec::new();
                for &pr in &preds {
                    let s = match out[pr].get(&reg) {
                        Some(s) => *s,
                        None if self.regs.has(reg) => self.arg(reg),
                        None => Src::Undef,
                    };
                    args.push((pr, s));
                }
                phi.args = args;
            }
        }
        let mut f = Func {
            wave32: self.wave32,
            blocks,
            entry: entry_id,
            vals: std::mem::take(&mut self.vals),
            args: self.args.iter().map(|(r, v)| (*r, *v)).collect(),
            uniform_args: self
                .args
                .keys()
                .copied()
                .filter(|r| {
                    matches!(r, Reg::V(n) if self.regs.uniform_vgprs.iter().any(|u| u.contains(n)))
                })
                .collect(),
            idom: (0..n)
                .map(|b| if b == entry_id { None } else { idom[b] })
                .collect(),
            rpo,
        };
        simplify_phis(&mut f);
        prune(&mut f);
        renumber(&mut f);
        Ok(Step::Done(f))
    }

    fn rename(
        &mut self,
        instr: &Instr,
        l: &Lowered,
        st: &mut State,
        block: &mut Block,
        bid: BlockId,
    ) -> Result<(), IrError> {
        let at = instr.at;
        if self.fetch_at == Some(at) {
            for r in [Reg::S(4), Reg::S(5)] {
                let a = self.arg(r);
                st.entry(r).or_insert(a);
            }
        }
        let mut srcs = Vec::with_capacity(l.srcs.len());
        for s in &l.srcs {
            let mut v = Vec::with_capacity(s.len());
            for r in s {
                v.push(match r {
                    Rd::Const(c) => Src::Const(*c),
                    Rd::Reg(r) => self.read(st, *r, at, true)?,
                });
            }
            srcs.push(v);
        }
        let mut imps = Vec::new();
        for &r in &l.imps {
            imps.push((r, self.read(st, r, at, true)?));
        }
        for &r in &l.olds {
            if r != Reg::Null {
                imps.push((r, self.read(st, r, at, false)?));
            }
        }
        let scc_in = imps.iter().find(|(r, _)| *r == Reg::Scc).map(|(_, s)| *s);
        if let Some(fd) = fold(instr, &srcs, scc_in, self.code) {
            let explicit: Vec<Reg> = l.dsts.iter().copied().filter(|r| *r != Reg::Scc).collect();
            let wants_scc = l.dsts.contains(&Reg::Scc);
            if fd.vals.len() == explicit.len() && (!wants_scc || fd.scc.is_some()) {
                for (r, v) in explicit.iter().zip(&fd.vals) {
                    if *r != Reg::Null {
                        st.insert(*r, *v);
                    }
                }
                if let Some(s) = fd.scc.filter(|_| wants_scc) {
                    st.insert(Reg::Scc, s);
                }
                return Ok(());
            }
        }
        let index = block.insts.len();
        let mut dsts = Vec::new();
        for &r in &l.dsts {
            if r == Reg::Null {
                continue;
            }
            let v = self.new_val(r, ValDef::Inst { block: bid, index });
            dsts.push((r, v));
        }
        for (r, v) in &dsts {
            st.insert(*r, Src::Val(*v));
        }
        block.insts.push(Inst {
            instr: instr.clone(),
            dsts,
            srcs,
            imps,
        });
        Ok(())
    }
}

/// Reverse post-order and immediate dominators (Cooper, Harvey, Kennedy). The entry's own
/// `idom` entry is `None`.
pub(super) fn dominators(
    n: usize,
    entry: usize,
    succs: &[Vec<usize>],
    preds: &[Vec<usize>],
) -> (Vec<usize>, Vec<Option<usize>>) {
    let mut seen = vec![false; n];
    let mut post = Vec::with_capacity(n);
    let mut stack = vec![(entry, 0usize)];
    seen[entry] = true;
    while let Some(&mut (b, ref mut k)) = stack.last_mut() {
        if let Some(&s) = succs[b].get(*k) {
            *k += 1;
            if !seen[s] {
                seen[s] = true;
                stack.push((s, 0));
            }
        } else {
            post.push(b);
            stack.pop();
        }
    }
    let rpo: Vec<usize> = post.into_iter().rev().collect();
    let mut index = vec![usize::MAX; n];
    for (k, &b) in rpo.iter().enumerate() {
        index[b] = k;
    }
    let mut idom: Vec<Option<usize>> = vec![None; n];
    idom[entry] = Some(entry);
    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().skip(1) {
            let mut new: Option<usize> = None;
            for &p in &preds[b] {
                if idom[p].is_none() {
                    continue;
                }
                new = Some(match new {
                    None => p,
                    Some(mut a) => {
                        let mut c = p;
                        while a != c {
                            while index[a] > index[c] {
                                a = idom[a].unwrap_or(entry);
                            }
                            while index[c] > index[a] {
                                c = idom[c].unwrap_or(entry);
                            }
                        }
                        a
                    }
                });
            }
            if new != idom[b] {
                idom[b] = new;
                changed = true;
            }
        }
    }
    (rpo, idom)
}

fn dominance_frontier(
    n: usize,
    entry: usize,
    idom: &[Option<usize>],
    preds: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    let mut df: Vec<Vec<usize>> = vec![Vec::new(); n];
    for b in 0..n {
        if preds[b].len() < 2 && b != entry {
            continue;
        }
        // the entry also has the program start as a predecessor: no stop above it
        let stop = if b == entry { None } else { idom[b] };
        for &p in &preds[b] {
            if idom[p].is_none() {
                continue;
            }
            let mut r = p;
            loop {
                if Some(r) == stop {
                    break;
                }
                if !df[r].contains(&b) {
                    df[r].push(b);
                }
                if r == entry {
                    break;
                }
                match idom[r] {
                    Some(d) => r = d,
                    None => break,
                }
            }
        }
    }
    df
}

/// Removes phis whose inputs are all one value (or themselves), and phis nothing reads.
fn simplify_phis(f: &mut Func) {
    loop {
        let mut repl: HashMap<Val, Src> = HashMap::new();
        for b in &f.blocks {
            for p in &b.phis {
                let mut same: Option<Src> = None;
                let mut ok = true;
                for (_, s) in &p.args {
                    if *s == Src::Val(p.dst) {
                        continue;
                    }
                    match same {
                        None => same = Some(*s),
                        Some(x) if x == *s => {}
                        _ => ok = false,
                    }
                }
                if let (true, Some(s)) = (ok, same) {
                    if s != Src::Undef {
                        repl.insert(p.dst, s);
                    }
                }
            }
        }
        // resolve chains
        let resolve = |mut s: Src| {
            let mut guard = 0;
            while let Src::Val(v) = s {
                match repl.get(&v) {
                    Some(n) if guard < 1000 => {
                        s = *n;
                        guard += 1;
                    }
                    _ => break,
                }
            }
            s
        };
        let mut changed = !repl.is_empty();
        if changed {
            for b in &mut f.blocks {
                for p in &mut b.phis {
                    for a in &mut p.args {
                        a.1 = resolve(a.1);
                    }
                }
                for i in &mut b.insts {
                    for s in i.srcs.iter_mut().flatten() {
                        *s = resolve(*s);
                    }
                    for (_, s) in &mut i.imps {
                        *s = resolve(*s);
                    }
                }
                if let Term::Branch { cond, .. } = &mut b.term {
                    for (_, s) in cond {
                        *s = resolve(*s);
                    }
                }
                b.phis.retain(|p| !repl.contains_key(&p.dst));
            }
        }
        // dead phis
        let mut uses: HashMap<Val, usize> = HashMap::new();
        let mut count = |s: &Src| {
            if let Src::Val(v) = s {
                *uses.entry(*v).or_default() += 1;
            }
        };
        for b in &f.blocks {
            for p in &b.phis {
                for (_, s) in &p.args {
                    if *s != Src::Val(p.dst) {
                        count(s);
                    }
                }
            }
            for i in &b.insts {
                i.srcs.iter().flatten().for_each(&mut count);
                i.imps.iter().for_each(|(_, s)| count(s));
            }
            if let Term::Branch { cond, .. } = &b.term {
                cond.iter().for_each(|(_, s)| count(s));
            }
        }
        for b in &mut f.blocks {
            let before = b.phis.len();
            b.phis.retain(|p| uses.contains_key(&p.dst));
            changed |= b.phis.len() != before;
        }
        if !changed {
            break;
        }
    }
}

/// The outcome of a branch whose condition registers all folded to constants: `Some(true)` when
/// it is taken. A condition is "zero" when every register is the constant 0 and "non-zero" when
/// one is a non-zero constant (the other half of a wave64 mask may be unknown). A branch with no
/// condition register (`s_cbranch_cdbg*`) is never constant.
fn constant_branch(op: &str, cond: &[(Reg, Src)]) -> Option<bool> {
    if cond.is_empty() {
        return None;
    }
    let konst = |s: &Src| match s {
        Src::Const(c) => Some(*c),
        _ => None,
    };
    let nonzero = cond.iter().any(|(_, s)| konst(s).is_some_and(|c| c != 0));
    let zero = cond.iter().all(|(_, s)| konst(s) == Some(0));
    let want_zero = matches!(op, "s_cbranch_scc0" | "s_cbranch_vccz" | "s_cbranch_execz");
    match (nonzero, zero) {
        (true, _) => Some(!want_zero),
        (_, true) => Some(want_zero),
        _ => None,
    }
}

/// Constant-branch pruning: an `s_cbranch_*` whose condition folded to a constant becomes a
/// jump, the blocks only it reached go, phis lose their inputs from the dead edges (and the
/// ones left with one value fold away, which may fold more branches), then the dominators
/// and the reverse post-order are recomputed. Arguments nothing reads any more are dropped.
fn prune(f: &mut Func) {
    let mut any = false;
    loop {
        let mut cut: Vec<(BlockId, BlockId)> = Vec::new();
        for (b, blk) in f.blocks.iter_mut().enumerate() {
            if let Term::Branch {
                op,
                cond,
                taken,
                fall,
            } = &blk.term
            {
                if let Some(t) = constant_branch(op, cond) {
                    let (go, dead) = if t { (*taken, *fall) } else { (*fall, *taken) };
                    if go != dead {
                        cut.push((b, dead));
                    }
                    blk.term = Term::Jump(go);
                    any = true;
                }
            }
        }
        if cut.is_empty() {
            // Same-target branches were turned into jumps: nothing else changes.
            break;
        }
        for (b, dead) in cut {
            let blk = &mut f.blocks[dead];
            blk.preds.retain(|&p| p != b);
            for phi in &mut blk.phis {
                phi.args.retain(|(p, _)| *p != b);
            }
        }
        remove_unreachable(f);
        simplify_phis(f);
    }
    if !any {
        return;
    }
    let n = f.blocks.len();
    let succs: Vec<Vec<BlockId>> = f.blocks.iter().map(Block::succs).collect();
    let preds: Vec<Vec<BlockId>> = f.blocks.iter().map(|b| b.preds.clone()).collect();
    let (rpo, idom) = dominators(n, f.entry, &succs, &preds);
    f.idom = (0..n)
        .map(|b| if b == f.entry { None } else { idom[b] })
        .collect();
    f.rpo = rpo;
    let mut used: HashSet<Val> = HashSet::new();
    let mut note = |s: &Src| {
        if let Src::Val(v) = s {
            used.insert(*v);
        }
    };
    for b in &f.blocks {
        for p in &b.phis {
            p.args.iter().for_each(|(_, s)| note(s));
        }
        for i in &b.insts {
            i.srcs.iter().flatten().for_each(&mut note);
            i.imps.iter().for_each(|(_, s)| note(s));
        }
        if let Term::Branch { cond, .. } = &b.term {
            cond.iter().for_each(|(_, s)| note(s));
        }
    }
    f.args.retain(|(_, v)| used.contains(v));
}

/// Drops the blocks the entry cannot reach, renumbering the others in order.
fn remove_unreachable(f: &mut Func) {
    let n = f.blocks.len();
    let mut seen = vec![false; n];
    let mut work = vec![f.entry];
    seen[f.entry] = true;
    while let Some(b) = work.pop() {
        for s in f.blocks[b].succs() {
            if !seen[s] {
                seen[s] = true;
                work.push(s);
            }
        }
    }
    if seen.iter().all(|&s| s) {
        return;
    }
    let mut map = vec![usize::MAX; n];
    let mut next = 0;
    for (b, &s) in seen.iter().enumerate() {
        if s {
            map[b] = next;
            next += 1;
        }
    }
    let old = std::mem::take(&mut f.blocks);
    for (b, mut blk) in old.into_iter().enumerate() {
        if !seen[b] {
            continue;
        }
        blk.preds = blk
            .preds
            .iter()
            .filter(|&&p| seen[p])
            .map(|&p| map[p])
            .collect();
        for phi in &mut blk.phis {
            phi.args = phi
                .args
                .iter()
                .filter(|(p, _)| seen[*p])
                .map(|&(p, s)| (map[p], s))
                .collect();
        }
        blk.term = match blk.term {
            Term::Jump(t) => Term::Jump(map[t]),
            Term::Branch {
                op,
                cond,
                taken,
                fall,
            } => Term::Branch {
                op,
                cond,
                taken: map[taken],
                fall: map[fall],
            },
            Term::End => Term::End,
        };
        f.blocks.push(blk);
    }
    f.entry = map[f.entry];
}

/// Numbers the values in print order: arguments, then each block's phis and instructions.
fn renumber(f: &mut Func) {
    let mut map: HashMap<Val, Val> = HashMap::new();
    let mut vals: Vec<ValInfo> = Vec::new();
    let mut take = |old: Val, info: ValInfo, vals: &mut Vec<ValInfo>| {
        vals.push(info);
        let new = Val(vals.len() as u32 - 1);
        map.insert(old, new);
        new
    };
    for (_, v) in &mut f.args {
        let info = f.vals[v.0 as usize];
        *v = take(*v, info, &mut vals);
    }
    for (bid, b) in f.blocks.iter_mut().enumerate() {
        for (index, p) in b.phis.iter_mut().enumerate() {
            let reg = p.reg;
            p.dst = take(
                p.dst,
                ValInfo {
                    reg,
                    def: ValDef::Phi { block: bid, index },
                },
                &mut vals,
            );
        }
        for (index, i) in b.insts.iter_mut().enumerate() {
            for (reg, v) in &mut i.dsts {
                let info = ValInfo {
                    reg: *reg,
                    def: ValDef::Inst { block: bid, index },
                };
                *v = take(*v, info, &mut vals);
            }
        }
    }
    let fix = |s: &mut Src| {
        if let Src::Val(v) = s {
            *v = map[v];
        }
    };
    for b in &mut f.blocks {
        for p in &mut b.phis {
            p.args.iter_mut().for_each(|a| fix(&mut a.1));
        }
        for i in &mut b.insts {
            i.srcs.iter_mut().flatten().for_each(fix);
            i.imps.iter_mut().for_each(|(_, s)| fix(s));
        }
        if let Term::Branch { cond, .. } = &mut b.term {
            cond.iter_mut().for_each(|(_, s)| fix(s));
        }
    }
    f.vals = vals;
}

// ---------------------------------------------------------------------------------------
// Text.

impl fmt::Display for Func {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "wave{}", if self.wave32 { 32 } else { 64 })?;
        for (r, v) in &self.args {
            writeln!(f, "%{}:{r} = arg", v.0)?;
        }
        for (id, b) in self.blocks.iter().enumerate() {
            let preds: Vec<String> = b.preds.iter().map(|p| format!("b{p}")).collect();
            writeln!(f, "b{id}: ; +{:#x} preds [{}]", b.at, preds.join(" "))?;
            for p in &b.phis {
                let args: Vec<String> = p.args.iter().map(|(b, s)| format!("b{b}: {s}")).collect();
                writeln!(f, "  %{}:{} = phi {}", p.dst.0, p.reg, args.join(", "))?;
            }
            for i in &b.insts {
                writeln!(f, "  {}", inst_text(i))?;
            }
            match &b.term {
                Term::Jump(t) => writeln!(f, "  jump b{t}")?,
                Term::End => writeln!(f, "  s_endpgm")?,
                Term::Branch {
                    op,
                    cond,
                    taken,
                    fall,
                } => {
                    let c: Vec<String> = cond.iter().map(|(r, s)| format!("{r}={s}")).collect();
                    writeln!(f, "  {op} {} -> b{taken} else b{fall}", c.join(", "))?
                }
            }
        }
        Ok(())
    }
}

/// One instruction as text: `%5:v0 = v_add_f32 %1, #0x3f800000 ; exec_lo=%0, v0=%2`.
pub fn inst_text(i: &Inst) -> String {
    let mut s = String::new();
    let d: Vec<String> = i
        .dsts
        .iter()
        .map(|(r, v)| format!("%{}:{r}", v.0))
        .collect();
    if !d.is_empty() {
        let _ = write!(s, "{} = ", d.join(", "));
    }
    s.push_str(i.instr.name());
    if i.instr.enc == Enc::Vop3 {
        s.push_str("_e64");
    }
    let ops: Vec<String> = i
        .srcs
        .iter()
        .map(|o| {
            if let [one] = &o[..] {
                one.to_string()
            } else {
                let v: Vec<String> = o.iter().map(ToString::to_string).collect();
                format!("{{{}}}", v.join(", "))
            }
        })
        .collect();
    if !ops.is_empty() {
        let _ = write!(s, " {}", ops.join(", "));
    }
    if matches!(i.instr.enc, Enc::Smem | Enc::Mubuf) && i.instr.mem.offset != 0 {
        let _ = write!(s, " offset:{}", i.instr.mem.offset);
    }
    if !i.imps.is_empty() {
        let v: Vec<String> = i.imps.iter().map(|(r, x)| format!("{r}={x}")).collect();
        let _ = write!(s, " ; {}", v.join(", "));
    }
    s
}

#[cfg(test)]
mod tests;
