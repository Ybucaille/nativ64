//! The semantics of the ALU opcodes, written once over a `Backend` of 32-bit primitives.
//!
//! Registers are bits (`V`: a `u32`), lane masks are one boolean per invocation (`B`). The
//! function `exec` holds the per-opcode table: it combines the primitives into the result of
//! one `Inst`. Two backends instantiate it: `Cpu` (this file; the evaluator the unit tests run
//! on, and what a GPU run is compared with) and the SPIR-V emitter (`emit.rs`), so the vectors
//! tested on the CPU are the semantics of the emitted code. `Dry` runs the table with no
//! values, to ask whether an opcode is supported.

use crate::shader::gfx10::{Enc, Instr, Mods, Sdwa};
use crate::shader::ir::{Inst, Reg, Src};

/// An instruction the translator does not emit yet: `chunk` is the plan chunk that adds it
/// (6 for an ALU opcode this chunk leaves out, 7 wave operations, 8 I/O, 9 memory).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unsupported {
    pub chunk: u8,
    pub op: String,
}

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (chunk {})", self.op, self.chunk)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IBin {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    /// Shifts take the amount as is (the table masks it to 5 bits first).
    Shl,
    Shr,
    Sar,
    UMin,
    UMax,
    SMin,
    SMax,
    UMulHi,
    SMulHi,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IUn {
    Not,
    BitCount,
    /// Index of the lowest set bit, `u32::MAX` for 0.
    FindLsb,
    /// Index of the highest set bit, `u32::MAX` for 0.
    FindUMsb,
    /// Index of the highest bit that differs from the sign, `u32::MAX` for 0 and -1.
    FindSMsb,
    BitRev,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ICmp {
    Eq,
    Ne,
    ULt,
    ULe,
    UGt,
    UGe,
    SLt,
    SLe,
    SGt,
    SGe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FBin {
    Add,
    Sub,
    Mul,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FUn {
    Sqrt,
    Rsq,
    Rcp,
    Exp2,
    Log2,
    /// Radians (the table scales the revolutions of `v_sin_f32`).
    Sin,
    Cos,
    Floor,
    Ceil,
    Trunc,
    RoundEven,
}

/// Float compares: `O*` are ordered (false on NaN), `U*` unordered (true on NaN).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FCmp {
    OLt,
    OEq,
    OLe,
    OGt,
    ONe,
    OGe,
    Ord,
    Uno,
    ULt,
    UEq,
    ULe,
    UGt,
    UNe,
    UGe,
}

/// Raw conversions, defined only in range (the table clamps and handles NaN).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cvt {
    FToU,
    FToI,
    UToF,
    IToF,
}

/// The 32-bit primitives the table is written with.
pub trait Backend {
    type V: Copy;
    type B: Copy;
    fn k(&mut self, x: u32) -> Self::V;
    fn kb(&mut self, x: bool) -> Self::B;
    fn ibin(&mut self, op: IBin, a: Self::V, b: Self::V) -> Self::V;
    fn iun(&mut self, op: IUn, a: Self::V) -> Self::V;
    fn icmp(&mut self, op: ICmp, a: Self::V, b: Self::V) -> Self::B;
    fn fbin(&mut self, op: FBin, a: Self::V, b: Self::V) -> Self::V;
    fn fma(&mut self, a: Self::V, b: Self::V, c: Self::V) -> Self::V;
    fn fun(&mut self, op: FUn, a: Self::V) -> Self::V;
    fn fcmp(&mut self, op: FCmp, a: Self::V, b: Self::V) -> Self::B;
    fn cvt(&mut self, op: Cvt, a: Self::V) -> Self::V;
    /// `a * 2^e` (`e` a signed count).
    fn ldexp(&mut self, a: Self::V, e: Self::V) -> Self::V;
    /// The f32 as f16 bits (round to nearest even) in the low half, 0 above.
    fn pack_half(&mut self, a: Self::V) -> Self::V;
    /// The f16 in the low half as f32 bits.
    fn unpack_half(&mut self, a: Self::V) -> Self::V;
    fn select(&mut self, c: Self::B, a: Self::V, b: Self::V) -> Self::V;
    fn bnot(&mut self, a: Self::B) -> Self::B;
    fn band(&mut self, a: Self::B, b: Self::B) -> Self::B;
    fn bor(&mut self, a: Self::B, b: Self::B) -> Self::B;
    fn bxor(&mut self, a: Self::B, b: Self::B) -> Self::B;
}

/// A value of an operand or result: bits, or a lane mask.
pub enum Opd<K: Backend> {
    V(K::V),
    B(K::B),
}

impl<K: Backend> Clone for Opd<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K: Backend> Copy for Opd<K> {}

/// What `exec` needs to run one instruction.
pub struct Ctx<'a, K: Backend> {
    pub inst: &'a Inst,
    /// The values of `inst.srcs`, dword by dword.
    pub srcs: Vec<Vec<Opd<K>>>,
    /// The values of `inst.imps`.
    pub imps: Vec<(Reg, Opd<K>)>,
    /// Per destination of `inst.dsts`: the result is stored as a lane mask.
    pub dst_mask: Vec<bool>,
    /// A vector instruction EXEC is not proved set for: this lane's EXEC bit. `None` runs it
    /// unconditionally.
    pub active: Option<K::B>,
    /// The console lane index of this invocation (`LocalInvocationIndex` modulo the wave
    /// size; 0 in the probe). `None` where a stage cannot tell it (vertex, pixel).
    pub lane: Option<K::V>,
}

/// The plan chunk an instruction that is not ALU belongs to, `None` for the ALU ones.
pub fn non_alu_chunk(i: &Instr) -> Option<u8> {
    let n = i.name();
    if i.dpp.is_some()
        || n.starts_with("v_readlane")
        || n.starts_with("v_readfirstlane")
        || n.starts_with("v_writelane")
        || n.starts_with("v_mbcnt")
        || n.starts_with("v_permlane")
        || n.starts_with("ds_swizzle")
        || n.starts_with("ds_permute")
        || n.starts_with("ds_bpermute")
    {
        return Some(7);
    }
    if matches!(
        n,
        "s_swappc_b64" | "s_setpc_b64" | "s_getpc_b64" | "s_rfe_b64"
    ) {
        return Some(8);
    }
    match i.enc {
        Enc::Exp | Enc::Vintrp => Some(8),
        Enc::Smem | Enc::Mubuf | Enc::Mtbuf | Enc::Mimg | Enc::Ds | Enc::Flat => Some(9),
        Enc::Sopp => Some(8),
        _ if n.starts_with("v_interp") => Some(8),
        _ => None,
    }
}

/// The wave instructions the structurer vets and `exec` runs per lane: a `v_readfirstlane` /
/// `v_readlane` of a uniform value or a waterfall index is a copy, `v_mbcnt` of -1 is the
/// console lane index. (Spill slots are handled by the emitter, before the table.)
fn is_wave_exempt(name: &str) -> bool {
    name.starts_with("v_readfirstlane")
        || name.starts_with("v_readlane")
        || name.starts_with("v_mbcnt")
}

// ---------------------------------------------------------------------------------------
// Backends

/// A backend with no values: `exec` against it only tells whether an opcode is supported.
pub struct Dry;

impl Backend for Dry {
    type V = ();
    type B = ();
    fn k(&mut self, _: u32) {}
    fn kb(&mut self, _: bool) {}
    fn ibin(&mut self, _: IBin, _: (), _: ()) {}
    fn iun(&mut self, _: IUn, _: ()) {}
    fn icmp(&mut self, _: ICmp, _: (), _: ()) {}
    fn fbin(&mut self, _: FBin, _: (), _: ()) {}
    fn fma(&mut self, _: (), _: (), _: ()) {}
    fn fun(&mut self, _: FUn, _: ()) {}
    fn fcmp(&mut self, _: FCmp, _: (), _: ()) {}
    fn cvt(&mut self, _: Cvt, _: ()) {}
    fn ldexp(&mut self, _: (), _: ()) {}
    fn pack_half(&mut self, _: ()) {}
    fn unpack_half(&mut self, _: ()) {}
    fn select(&mut self, _: (), _: (), _: ()) {}
    fn bnot(&mut self, _: ()) {}
    fn band(&mut self, _: (), _: ()) {}
    fn bor(&mut self, _: (), _: ()) {}
    fn bxor(&mut self, _: (), _: ()) {}
}

/// Runs the table on host numbers.
pub struct Cpu;

fn f(a: u32) -> f32 {
    f32::from_bits(a)
}

/// f32 bits to f16 bits, round to nearest even.
pub fn f32_to_f16_rte(bits: u32) -> u32 {
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if man != 0 { 0x200 | (man >> 13) } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let up = rem > mid || (rem == mid && half & 1 == 1);
        return sign | (half + u32::from(up));
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let up = rem > 0x1000 || (rem == 0x1000 && half & 1 == 1);
    sign | (half + u32::from(up))
}

/// f16 bits (low half) to f32 bits.
pub fn f16_to_f32(h: u32) -> u32 {
    let h = h & 0xffff;
    let sign = (h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let man = h & 0x3ff;
    match exp {
        0 if man == 0 => sign,
        0 => {
            let v = man as f32 * (1.0 / 16_777_216.0);
            sign | v.to_bits()
        }
        31 => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    }
}

impl Backend for Cpu {
    type V = u32;
    type B = bool;
    fn k(&mut self, x: u32) -> u32 {
        x
    }
    fn kb(&mut self, x: bool) -> bool {
        x
    }
    fn ibin(&mut self, op: IBin, a: u32, b: u32) -> u32 {
        match op {
            IBin::Add => a.wrapping_add(b),
            IBin::Sub => a.wrapping_sub(b),
            IBin::Mul => a.wrapping_mul(b),
            IBin::And => a & b,
            IBin::Or => a | b,
            IBin::Xor => a ^ b,
            IBin::Shl => a.checked_shl(b).unwrap_or(0),
            IBin::Shr => a.checked_shr(b).unwrap_or(0),
            IBin::Sar => ((a as i32) >> b.min(31)) as u32,
            IBin::UMin => a.min(b),
            IBin::UMax => a.max(b),
            IBin::SMin => (a as i32).min(b as i32) as u32,
            IBin::SMax => (a as i32).max(b as i32) as u32,
            IBin::UMulHi => ((u64::from(a) * u64::from(b)) >> 32) as u32,
            IBin::SMulHi => ((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32,
        }
    }
    fn iun(&mut self, op: IUn, a: u32) -> u32 {
        match op {
            IUn::Not => !a,
            IUn::BitCount => a.count_ones(),
            IUn::FindLsb => {
                if a == 0 {
                    u32::MAX
                } else {
                    a.trailing_zeros()
                }
            }
            IUn::FindUMsb => {
                if a == 0 {
                    u32::MAX
                } else {
                    31 - a.leading_zeros()
                }
            }
            IUn::FindSMsb => {
                if a == 0 || a == u32::MAX {
                    u32::MAX
                } else {
                    let x = if (a as i32) < 0 { !a } else { a };
                    31 - x.leading_zeros()
                }
            }
            IUn::BitRev => a.reverse_bits(),
        }
    }
    fn icmp(&mut self, op: ICmp, a: u32, b: u32) -> bool {
        match op {
            ICmp::Eq => a == b,
            ICmp::Ne => a != b,
            ICmp::ULt => a < b,
            ICmp::ULe => a <= b,
            ICmp::UGt => a > b,
            ICmp::UGe => a >= b,
            ICmp::SLt => (a as i32) < b as i32,
            ICmp::SLe => (a as i32) <= b as i32,
            ICmp::SGt => (a as i32) > b as i32,
            ICmp::SGe => (a as i32) >= b as i32,
        }
    }
    fn fbin(&mut self, op: FBin, a: u32, b: u32) -> u32 {
        let (a, b) = (f(a), f(b));
        match op {
            FBin::Add => a + b,
            FBin::Sub => a - b,
            FBin::Mul => a * b,
        }
        .to_bits()
    }
    fn fma(&mut self, a: u32, b: u32, c: u32) -> u32 {
        f(a).mul_add(f(b), f(c)).to_bits()
    }
    fn fun(&mut self, op: FUn, a: u32) -> u32 {
        let a = f(a);
        match op {
            FUn::Sqrt => a.sqrt(),
            FUn::Rsq => 1.0 / a.sqrt(),
            FUn::Rcp => 1.0 / a,
            FUn::Exp2 => a.exp2(),
            FUn::Log2 => a.log2(),
            FUn::Sin => a.sin(),
            FUn::Cos => a.cos(),
            FUn::Floor => a.floor(),
            FUn::Ceil => a.ceil(),
            FUn::Trunc => a.trunc(),
            FUn::RoundEven => a.round_ties_even(),
        }
        .to_bits()
    }
    fn fcmp(&mut self, op: FCmp, a: u32, b: u32) -> bool {
        let (a, b) = (f(a), f(b));
        let un = a.is_nan() || b.is_nan();
        match op {
            FCmp::OLt => a < b,
            FCmp::OEq => a == b,
            FCmp::OLe => a <= b,
            FCmp::OGt => a > b,
            FCmp::ONe => !un && a != b,
            FCmp::OGe => a >= b,
            FCmp::Ord => !un,
            FCmp::Uno => un,
            FCmp::ULt => un || a < b,
            FCmp::UEq => un || a == b,
            FCmp::ULe => un || a <= b,
            FCmp::UGt => un || a > b,
            FCmp::UNe => un || a != b,
            FCmp::UGe => un || a >= b,
        }
    }
    fn cvt(&mut self, op: Cvt, a: u32) -> u32 {
        match op {
            Cvt::FToU => f(a) as u32,
            Cvt::FToI => f(a) as i32 as u32,
            Cvt::UToF => (a as f32).to_bits(),
            Cvt::IToF => (a as i32 as f32).to_bits(),
        }
    }
    fn ldexp(&mut self, a: u32, e: u32) -> u32 {
        let e = (e as i32).clamp(-400, 400);
        ((f64::from(f(a)) * 2f64.powi(e)) as f32).to_bits()
    }
    fn pack_half(&mut self, a: u32) -> u32 {
        f32_to_f16_rte(a)
    }
    fn unpack_half(&mut self, a: u32) -> u32 {
        f16_to_f32(a)
    }
    fn select(&mut self, c: bool, a: u32, b: u32) -> u32 {
        if c {
            a
        } else {
            b
        }
    }
    fn bnot(&mut self, a: bool) -> bool {
        !a
    }
    fn band(&mut self, a: bool, b: bool) -> bool {
        a && b
    }
    fn bor(&mut self, a: bool, b: bool) -> bool {
        a || b
    }
    fn bxor(&mut self, a: bool, b: bool) -> bool {
        a ^ b
    }
}

// ---------------------------------------------------------------------------------------
// The table

type R<T> = Result<T, Unsupported>;

/// What an opcode computes: its data destinations (in `Inst::dsts` order, SCC apart; a shorter
/// list repeats its last value, which is how a lane mask fills a wave64 pair) and its SCC.
struct Res<K: Backend> {
    data: Vec<Opd<K>>,
    scc: Option<K::V>,
}

struct X<'a, K: Backend> {
    b: &'a mut K,
    c: &'a Ctx<'a, K>,
    i: &'a Instr,
    name: &'static str,
    /// The source and result modifiers: the VOP3 fields, or the SDWA ones.
    mods: Mods,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Bit {
    And,
    Or,
    Xor,
    Andn2,
    Orn2,
    Nand,
    Nor,
    Xnor,
    Andn1,
    Orn1,
    Not,
    Mov,
}

fn bit_named(n: &str) -> Option<Bit> {
    Some(match n {
        "and" => Bit::And,
        "or" => Bit::Or,
        "xor" => Bit::Xor,
        "andn2" => Bit::Andn2,
        "orn2" => Bit::Orn2,
        "nand" => Bit::Nand,
        "nor" => Bit::Nor,
        "xnor" => Bit::Xnor,
        "andn1" => Bit::Andn1,
        "orn1" => Bit::Orn1,
        "not" => Bit::Not,
        "mov" | "wqm" => Bit::Mov,
        _ => return None,
    })
}

const SIGN: u32 = 0x8000_0000;
const ABS: u32 = 0x7fff_ffff;

/// Runs one ALU instruction. The values come back in `inst.dsts` order, as a lane mask where
/// `ctx.dst_mask` says so. Every opcode that is not translated here is an `Unsupported`.
pub fn exec<K: Backend>(b: &mut K, c: &Ctx<K>) -> Result<Vec<Opd<K>>, Unsupported> {
    let i = &c.inst.instr;
    let name = i.name();
    // The structurer only lets a `v_readfirstlane` through when its source is uniform: a copy.
    if let Some(chunk) = non_alu_chunk(i).filter(|_| !is_wave_exempt(name)) {
        return Err(Unsupported {
            chunk,
            op: name.to_string(),
        });
    }
    let mods = match i.sdwa {
        Some(sd) => Mods {
            abs: u8::from(sd.src0_abs) | u8::from(sd.src1_abs) << 1,
            neg: u8::from(sd.src0_neg) | u8::from(sd.src1_neg) << 1,
            clamp: sd.clamp,
            omod: sd.omod,
            ..Mods::default()
        },
        None => i.mods,
    };
    let mut x = X {
        b,
        c,
        i,
        name,
        mods,
    };
    let res = x.run()?;
    x.finish(res)
}

impl<K: Backend> X<'_, K> {
    fn un(&self) -> Unsupported {
        Unsupported {
            chunk: 6,
            op: self.name.to_string(),
        }
    }

    fn k(&mut self, v: u32) -> K::V {
        self.b.k(v)
    }

    fn src(&self, i: usize, k: usize) -> R<Opd<K>> {
        self.c
            .srcs
            .get(i)
            .and_then(|s| s.get(k))
            .copied()
            .ok_or_else(|| self.un())
    }

    fn as_v(&mut self, o: Opd<K>) -> K::V {
        match o {
            Opd::V(v) => v,
            Opd::B(c) => {
                let (one, zero) = (self.k(1), self.k(0));
                self.b.select(c, one, zero)
            }
        }
    }

    /// The lane's own bit of a value read as a mask: bit 0 (the probe's lane is 0).
    fn as_b(&mut self, o: Opd<K>) -> K::B {
        match o {
            Opd::B(c) => c,
            Opd::V(v) => {
                let one = self.k(1);
                let zero = self.k(0);
                let t = self.b.ibin(IBin::And, v, one);
                self.b.icmp(ICmp::Ne, t, zero)
            }
        }
    }

    /// True when the value is not zero.
    fn nz(&mut self, o: Opd<K>) -> K::B {
        match o {
            Opd::B(c) => c,
            Opd::V(v) => {
                let zero = self.k(0);
                self.b.icmp(ICmp::Ne, v, zero)
            }
        }
    }

    fn bv(&mut self, c: K::B) -> K::V {
        self.as_v(Opd::B(c))
    }

    fn v(&mut self, i: usize) -> R<K::V> {
        let o = self.src(i, 0)?;
        let v = self.as_v(o);
        match self.i.sdwa {
            Some(sd) if i < 2 => Ok(self.sdwa_src(v, sd, i)),
            _ => Ok(v),
        }
    }

    /// An SDWA source: the selected byte or word, zero or sign extended.
    fn sdwa_src(&mut self, v: K::V, sd: Sdwa, i: usize) -> K::V {
        let (sel, sext) = if i == 0 {
            (sd.src0_sel, sd.src0_sext)
        } else {
            (sd.src1_sel, sd.src1_sext)
        };
        let (pos, width) = match sel {
            0..=3 => (8 * u32::from(sel), 8),
            4 | 5 => (16 * u32::from(sel - 4), 16),
            _ => return v,
        };
        let t = self.shl_k(v, 32 - pos - width);
        let sh = 32 - width;
        self.ibk(if sext { IBin::Sar } else { IBin::Shr }, t, sh)
    }

    /// An SDWA result goes to the selected byte or word of the destination; the rest is
    /// zero, the sign, or the old value (`dst_unused`).
    fn sdwa_dst(&mut self, r: K::V, sd: Sdwa) -> R<K::V> {
        let (pos, width) = match sd.dst_sel {
            0..=3 => (8 * u32::from(sd.dst_sel), 8u32),
            4 | 5 => (16 * u32::from(sd.dst_sel - 4), 16),
            _ => return Ok(r),
        };
        let low = if width == 8 { 0xff } else { 0xffff };
        let field = self.and_k(r, low);
        let bits = self.shl_k(field, pos);
        let mask = low << pos;
        Ok(match sd.dst_unused {
            0 => bits,
            1 => {
                let sign = self.and_k(r, 1 << (width - 1));
                let neg = self.ick(ICmp::Ne, sign, 0);
                let fill = self.k(!mask);
                let zero = self.k(0);
                let f = self.sel(neg, fill, zero);
                self.ib(IBin::Or, bits, f)
            }
            _ => {
                let old = self.old_dst()?;
                let keep = self.and_k(old, !mask);
                self.ib(IBin::Or, bits, keep)
            }
        })
    }

    fn vk(&mut self, i: usize, k: usize) -> R<K::V> {
        let o = self.src(i, k)?;
        Ok(self.as_v(o))
    }

    fn m(&mut self, i: usize) -> R<K::B> {
        let o = self.src(i, 0)?;
        Ok(self.as_b(o))
    }

    /// Source `i` as a float: abs then neg applied.
    fn fsrc(&mut self, i: usize) -> R<K::V> {
        let mut a = self.v(i)?;
        let mods = self.mods;
        if mods.abs >> i & 1 != 0 {
            let m = self.k(ABS);
            a = self.b.ibin(IBin::And, a, m);
        }
        if mods.neg >> i & 1 != 0 {
            let m = self.k(SIGN);
            a = self.b.ibin(IBin::Xor, a, m);
        }
        Ok(a)
    }

    fn imp(&self, r: Reg) -> Option<Opd<K>> {
        self.c.imps.iter().find(|(x, _)| *x == r).map(|(_, v)| *v)
    }

    fn imp_scc(&mut self) -> R<K::B> {
        let o = self.imp(Reg::Scc).ok_or_else(|| self.un())?;
        Ok(self.nz(o))
    }

    fn old_dst(&mut self) -> R<K::V> {
        let reg = self.c.inst.dsts.first().ok_or_else(|| self.un())?.0;
        let o = self.imp(reg).ok_or_else(|| self.un())?;
        Ok(self.as_v(o))
    }

    // -- small builders

    fn ib(&mut self, op: IBin, a: K::V, b: K::V) -> K::V {
        self.b.ibin(op, a, b)
    }
    fn ibk(&mut self, op: IBin, a: K::V, k: u32) -> K::V {
        let k = self.k(k);
        self.b.ibin(op, a, k)
    }
    fn ic(&mut self, op: ICmp, a: K::V, b: K::V) -> K::B {
        self.b.icmp(op, a, b)
    }
    fn ick(&mut self, op: ICmp, a: K::V, k: u32) -> K::B {
        let k = self.k(k);
        self.b.icmp(op, a, k)
    }
    fn fb(&mut self, op: FBin, a: K::V, b: K::V) -> K::V {
        self.b.fbin(op, a, b)
    }
    fn fc(&mut self, op: FCmp, a: K::V, b: K::V) -> K::B {
        self.b.fcmp(op, a, b)
    }
    fn sel(&mut self, c: K::B, a: K::V, b: K::V) -> K::V {
        self.b.select(c, a, b)
    }
    fn selk(&mut self, c: K::B, a: u32, b: K::V) -> K::V {
        let a = self.k(a);
        self.b.select(c, a, b)
    }
    fn fk(&mut self, x: f32) -> K::V {
        self.k(x.to_bits())
    }
    fn and_k(&mut self, a: K::V, k: u32) -> K::V {
        self.ibk(IBin::And, a, k)
    }
    fn shl_k(&mut self, a: K::V, k: u32) -> K::V {
        self.ibk(IBin::Shl, a, k)
    }
    fn shr_k(&mut self, a: K::V, k: u32) -> K::V {
        self.ibk(IBin::Shr, a, k)
    }
    fn is_nan(&mut self, a: K::V) -> K::B {
        self.fc(FCmp::Uno, a, a)
    }

    /// `a << (n & 31)` and friends: the hardware takes the low 5 bits of the amount.
    fn shift(&mut self, op: IBin, a: K::V, n: K::V) -> K::V {
        let n = self.and_k(n, 31);
        self.ib(op, a, n)
    }

    fn fneg(&mut self, a: K::V) -> K::V {
        self.ibk(IBin::Xor, a, SIGN)
    }

    /// `v_min_f32` / `v_max_f32` (IEEE mode): a NaN operand gives the other one, and -0 is
    /// below +0.
    fn fminmax(&mut self, a: K::V, b: K::V, max: bool) -> K::V {
        let (lt, eq) = if max {
            (self.fc(FCmp::OGt, a, b), self.fc(FCmp::OEq, a, b))
        } else {
            (self.fc(FCmp::OLt, a, b), self.fc(FCmp::OEq, a, b))
        };
        // equal values: the zero of the right sign wins
        let sa = self.and_k(a, SIGN);
        let zero = self.k(0);
        let a_neg = self.ic(ICmp::Ne, sa, zero);
        let a_wins_tie = if max { self.b.bnot(a_neg) } else { a_neg };
        let tie = self.b.band(eq, a_wins_tie);
        let take_a = self.b.bor(lt, tie);
        let r = self.sel(take_a, a, b);
        let na = self.is_nan(a);
        let nb = self.is_nan(b);
        let r = self.sel(nb, a, r);
        self.sel(na, b, r)
    }

    fn fmed3(&mut self, a: K::V, b: K::V, c: K::V) -> K::V {
        let na = self.is_nan(a);
        let nb = self.is_nan(b);
        let nc = self.is_nan(c);
        let any = self.b.bor(na, nb);
        let any = self.b.bor(any, nc);
        let lo = self.fminmax(a, b, false);
        let lo3 = self.fminmax(lo, c, false);
        let hi = self.fminmax(a, b, true);
        let mid = self.fminmax(hi, c, false);
        let med = self.fminmax(lo, mid, true);
        self.sel(any, lo3, med)
    }

    /// DX9 multiply: zero times anything is a zero (of the product's sign).
    fn mul_legacy(&mut self, a: K::V, b: K::V) -> K::V {
        let p = self.fb(FBin::Mul, a, b);
        let ma = self.and_k(a, ABS);
        let mb = self.and_k(b, ABS);
        let z = self.k(0);
        let za = self.ic(ICmp::Eq, ma, z);
        let zb = self.ic(ICmp::Eq, mb, z);
        let any = self.b.bor(za, zb);
        let x = self.ib(IBin::Xor, a, b);
        let sign = self.and_k(x, SIGN);
        self.sel(any, sign, p)
    }

    fn f2u(&mut self, a: K::V) -> K::V {
        let raw = self.b.cvt(Cvt::FToU, a);
        let big = self.fk(4_294_967_296.0);
        let hi = self.fc(FCmp::OGe, a, big);
        let r = self.selk(hi, u32::MAX, raw);
        let zero = self.k(0);
        let pos = self.fc(FCmp::OGt, a, zero);
        let neg = self.b.bnot(pos);
        self.sel(neg, zero, r)
    }

    fn f2i(&mut self, a: K::V) -> K::V {
        let raw = self.b.cvt(Cvt::FToI, a);
        let hi_f = self.fk(2_147_483_648.0);
        let lo_f = self.fk(-2_147_483_648.0);
        let hi = self.fc(FCmp::OGe, a, hi_f);
        let lo = self.fc(FCmp::OLe, a, lo_f);
        let r = self.selk(hi, 0x7fff_ffff, raw);
        let r = self.selk(lo, SIGN, r);
        let nan = self.is_nan(a);
        let zero = self.k(0);
        self.sel(nan, zero, r)
    }

    /// Clamp to [0, 1], NaN to 0.
    fn fclamp(&mut self, a: K::V) -> K::V {
        let zero = self.fk(0.0);
        let one = self.fk(1.0);
        let gt1 = self.fc(FCmp::OGt, a, one);
        let r = self.sel(gt1, one, a);
        let pos = self.fc(FCmp::OGt, a, zero);
        let neg = self.b.bnot(pos);
        self.sel(neg, zero, r)
    }

    /// The output modifiers of a float result.
    fn fout(&mut self, r: K::V) -> R<Res<K>> {
        let m = self.mods;
        let mut r = r;
        if m.omod != 0 {
            let s = [2.0, 4.0, 0.5][usize::from(m.omod) - 1];
            let s = self.fk(s);
            r = self.fb(FBin::Mul, r, s);
        }
        if m.clamp {
            r = self.fclamp(r);
        }
        Ok(Res::v(r))
    }

    fn bfe(&mut self, a: K::V, off: K::V, width: K::V, signed: bool) -> K::V {
        let off = self.and_k(off, 31);
        let zero = self.k(0);
        let w0 = self.ic(ICmp::Eq, width, zero);
        if !signed {
            let sh = self.ib(IBin::Shr, a, off);
            let one = self.k(1);
            let m = self.ib(IBin::Shl, one, width);
            let m = self.ibk(IBin::Sub, m, 1);
            let wide = self.ick(ICmp::UGe, width, 32);
            let m = self.selk(wide, u32::MAX, m);
            let r = self.ib(IBin::And, sh, m);
            return self.sel(w0, zero, r);
        }
        let sh = self.ib(IBin::Sar, a, off);
        let k32 = self.k(32);
        let back = self.ib(IBin::Sub, k32, width);
        let back = self.and_k(back, 31);
        let t = self.ib(IBin::Shl, sh, back);
        let t = self.ib(IBin::Sar, t, back);
        let wide = self.ick(ICmp::UGe, width, 32);
        let t = self.sel(wide, sh, t);
        self.sel(w0, zero, t)
    }

    /// `{a, b} >> s` keeping the low dword (`s` already in 0..=31).
    fn funnel(&mut self, a: K::V, b: K::V, s: K::V) -> K::V {
        let lo = self.ib(IBin::Shr, b, s);
        // a << (32 - s), 0 for s = 0
        let a1 = self.shl_k(a, 1);
        let k31 = self.k(31);
        let inv = self.ib(IBin::Sub, k31, s);
        let hi = self.ib(IBin::Shl, a1, inv);
        self.ib(IBin::Or, lo, hi)
    }

    /// A 64-bit shift of `(lo, hi)` by `amt & 63`.
    fn shift64(&mut self, kind: IBin, lo: K::V, hi: K::V, amt: K::V) -> (K::V, K::V) {
        let n = self.and_k(amt, 63);
        let s = self.and_k(n, 31);
        let big = self.ick(ICmp::UGe, n, 32);
        let zero = self.k(0);
        match kind {
            IBin::Shl => {
                let part = self.ib(IBin::Shl, lo, s);
                let k31 = self.k(31);
                let inv = self.ib(IBin::Sub, k31, s);
                let l1 = self.shr_k(lo, 1);
                let spill = self.ib(IBin::Shr, l1, inv);
                let h = self.ib(IBin::Shl, hi, s);
                let h = self.ib(IBin::Or, h, spill);
                let rl = self.sel(big, zero, part);
                let rh = self.sel(big, part, h);
                (rl, rh)
            }
            _ => {
                let sar = kind == IBin::Sar;
                let l = self.funnel(hi, lo, s);
                let h = self.ib(kind, hi, s);
                let fill = if sar {
                    self.ibk(IBin::Sar, hi, 31)
                } else {
                    zero
                };
                let l_big = self.ib(kind, hi, s);
                let rl = self.sel(big, l_big, l);
                let rh = self.sel(big, fill, h);
                (rl, rh)
            }
        }
    }

    /// f32 bits of one operand of `v_fma_mix*`: f16 half or f32, with abs and neg.
    fn mix_src(&mut self, i: usize) -> R<K::V> {
        let m = self.mods;
        let raw = self.v(i)?;
        let mut a = if m.opsel_hi >> i & 1 != 0 {
            let h = if m.opsel >> i & 1 != 0 {
                self.shr_k(raw, 16)
            } else {
                raw
            };
            self.b.unpack_half(h)
        } else {
            raw
        };
        if m.abs >> i & 1 != 0 {
            a = self.and_k(a, ABS);
        }
        if m.neg >> i & 1 != 0 {
            a = self.ibk(IBin::Xor, a, SIGN);
        }
        Ok(a)
    }

    /// `v_cvt_pkrtz_f16_f32`: one f32 to f16, rounding toward zero (integer code).
    fn f16_rtz(&mut self, a: K::V) -> K::V {
        let sign = self.shr_k(a, 16);
        let sign = self.and_k(sign, 0x8000);
        let e = self.shr_k(a, 23);
        let e = self.and_k(e, 0xff);
        let m = self.and_k(a, 0x7f_ffff);
        // normal range: f32 exponent 113..=142 maps to 1..=30
        let k112 = self.k(112);
        let he = self.ib(IBin::Sub, e, k112);
        let he10 = self.shl_k(he, 10);
        let m13 = self.shr_k(m, 13);
        let normal = self.ib(IBin::Or, he10, m13);
        // subnormal range: 103..=112, shift (126 - e)
        let full = self.ibk(IBin::Or, m, 0x80_0000);
        let k126 = self.k(126);
        let sh = self.ib(IBin::Sub, k126, e);
        let sh = self.and_k(sh, 31);
        let sub = self.ib(IBin::Shr, full, sh);
        let is_normal = self.ick(ICmp::UGe, e, 113);
        let is_sub = self.ick(ICmp::UGe, e, 103);
        let r = self.k(0);
        let r = self.sel(is_sub, sub, r);
        let r = self.sel(is_normal, normal, r);
        let over = self.ick(ICmp::UGe, e, 143);
        let r = self.selk(over, 0x7bff, r);
        let inf_nan = self.ick(ICmp::Eq, e, 255);
        let zero = self.k(0);
        let has_m = self.ic(ICmp::Ne, m, zero);
        let nan = self.k(0x7e00);
        let inf = self.k(0x7c00);
        let spec = self.sel(has_m, nan, inf);
        let r = self.sel(inf_nan, spec, r);
        self.ib(IBin::Or, r, sign)
    }

    fn cmp_class(&mut self, a: K::V, mask: K::V) -> K::B {
        let e = self.shr_k(a, 23);
        let e = self.and_k(e, 0xff);
        let m = self.and_k(a, 0x7f_ffff);
        let neg = {
            let s = self.and_k(a, SIGN);
            let z = self.k(0);
            self.ic(ICmp::Ne, s, z)
        };
        let zero = self.k(0);
        let e255 = self.ick(ICmp::Eq, e, 255);
        let e0 = self.ick(ICmp::Eq, e, 0);
        let m0 = self.ic(ICmp::Eq, m, zero);
        let m_nz = self.b.bnot(m0);
        let nan = self.b.band(e255, m_nz);
        let quiet = {
            let q = self.and_k(m, 0x40_0000);
            self.ic(ICmp::Ne, q, zero)
        };
        let inf = self.b.band(e255, m0);
        let z = self.b.band(e0, m0);
        let den = self.b.band(e0, m_nz);
        // bit index: nan 0 (signalling) / 1 (quiet); negative 2 inf, 3 normal, 4 denormal,
        // 5 zero; positive 6 zero, 7 denormal, 8 normal, 9 inf
        let ks: Vec<K::V> = (0..10).map(|n| self.k(n)).collect();
        let idx = self.sel(neg, ks[3], ks[8]);
        let d = self.sel(neg, ks[4], ks[7]);
        let idx = self.sel(den, d, idx);
        let zc = self.sel(neg, ks[5], ks[6]);
        let idx = self.sel(z, zc, idx);
        let ic = self.sel(neg, ks[2], ks[9]);
        let idx = self.sel(inf, ic, idx);
        let nc = self.sel(quiet, ks[1], ks[0]);
        let idx = self.sel(nan, nc, idx);
        let bit = self.ib(IBin::Shr, mask, idx);
        let bit = self.and_k(bit, 1);
        self.ick(ICmp::Ne, bit, 0)
    }

    fn bit(&mut self, bit: Bit, a: Opd<K>, b: Option<Opd<K>>, want_b: bool) -> Opd<K> {
        if want_b {
            let x = self.as_b(a);
            let y = match b {
                Some(b) => self.as_b(b),
                None => x,
            };
            let r = match bit {
                Bit::And => self.b.band(x, y),
                Bit::Or => self.b.bor(x, y),
                Bit::Xor => self.b.bxor(x, y),
                Bit::Andn2 => {
                    let ny = self.b.bnot(y);
                    self.b.band(x, ny)
                }
                Bit::Orn2 => {
                    let ny = self.b.bnot(y);
                    self.b.bor(x, ny)
                }
                Bit::Andn1 => {
                    let nx = self.b.bnot(x);
                    self.b.band(nx, y)
                }
                Bit::Orn1 => {
                    let nx = self.b.bnot(x);
                    self.b.bor(nx, y)
                }
                Bit::Nand => {
                    let t = self.b.band(x, y);
                    self.b.bnot(t)
                }
                Bit::Nor => {
                    let t = self.b.bor(x, y);
                    self.b.bnot(t)
                }
                Bit::Xnor => {
                    let t = self.b.bxor(x, y);
                    self.b.bnot(t)
                }
                Bit::Not => self.b.bnot(x),
                Bit::Mov => x,
            };
            Opd::B(r)
        } else {
            let x = self.as_v(a);
            let y = match b {
                Some(b) => self.as_v(b),
                None => x,
            };
            let r = match bit {
                Bit::And => self.ib(IBin::And, x, y),
                Bit::Or => self.ib(IBin::Or, x, y),
                Bit::Xor => self.ib(IBin::Xor, x, y),
                Bit::Andn2 => {
                    let ny = self.b.iun(IUn::Not, y);
                    self.ib(IBin::And, x, ny)
                }
                Bit::Orn2 => {
                    let ny = self.b.iun(IUn::Not, y);
                    self.ib(IBin::Or, x, ny)
                }
                Bit::Andn1 => {
                    let nx = self.b.iun(IUn::Not, x);
                    self.ib(IBin::And, nx, y)
                }
                Bit::Orn1 => {
                    let nx = self.b.iun(IUn::Not, x);
                    self.ib(IBin::Or, nx, y)
                }
                Bit::Nand => {
                    let t = self.ib(IBin::And, x, y);
                    self.b.iun(IUn::Not, t)
                }
                Bit::Nor => {
                    let t = self.ib(IBin::Or, x, y);
                    self.b.iun(IUn::Not, t)
                }
                Bit::Xnor => {
                    let t = self.ib(IBin::Xor, x, y);
                    self.b.iun(IUn::Not, t)
                }
                Bit::Not => self.b.iun(IUn::Not, x),
                Bit::Mov => x,
            };
            Opd::V(r)
        }
    }

    fn writes_scc(&self) -> bool {
        matches!(self.c.inst.dsts.last(), Some((Reg::Scc, _)))
    }

    /// SCC = "the result is not zero" over all the result dwords.
    fn scc_nz(&mut self, data: &[Opd<K>]) -> Option<K::V> {
        let mut acc: Option<K::B> = None;
        for &d in data {
            let n = self.nz(d);
            acc = Some(match acc {
                Some(a) => self.b.bor(a, n),
                None => n,
            });
        }
        acc.map(|a| self.bv(a))
    }

    fn width(&self) -> usize {
        self.c
            .inst
            .dsts
            .iter()
            .filter(|(r, _)| *r != Reg::Scc)
            .count()
    }

    fn finish(&mut self, res: Res<K>) -> R<Vec<Opd<K>>> {
        let dsts = &self.c.inst.dsts;
        let nd = self.width();
        let mut data = res.data;
        if data.len() > nd || (data.is_empty() && nd > 0) {
            return Err(self.un());
        }
        while data.len() < nd {
            let last = *data.last().ok_or_else(|| self.un())?;
            data.push(last);
        }
        for (j, d) in data.iter_mut().enumerate() {
            let reg = dsts[j].0;
            if let Some(act) = self.c.active {
                match reg {
                    Reg::V(_) => {
                        let new = match *d {
                            Opd::V(v) => v,
                            Opd::B(c) => {
                                let (one, zero) = (self.b.k(1), self.b.k(0));
                                self.b.select(c, one, zero)
                            }
                        };
                        let old = match self.c.imps.iter().find(|(r, _)| *r == reg) {
                            Some((_, Opd::V(v))) => *v,
                            _ => new,
                        };
                        *d = Opd::V(self.b.select(act, new, old));
                    }
                    _ => {
                        let c = match *d {
                            Opd::B(c) => c,
                            Opd::V(v) => {
                                let one = self.b.k(1);
                                let t = self.b.ibin(IBin::And, v, one);
                                let z = self.b.k(0);
                                self.b.icmp(ICmp::Ne, t, z)
                            }
                        };
                        *d = Opd::B(self.b.band(act, c));
                    }
                }
            }
            let want_b = self.c.dst_mask.get(j).copied().unwrap_or(false);
            *d = if want_b {
                Opd::B(self.as_b(*d))
            } else {
                Opd::V(self.as_v(*d))
            };
        }
        if matches!(dsts.last(), Some((Reg::Scc, _))) {
            let s = res.scc.ok_or_else(|| self.un())?;
            data.push(Opd::V(s));
        }
        Ok(data)
    }

    fn run(&mut self) -> R<Res<K>> {
        let i = self.i;
        let m = self.mods;
        let vop3 = matches!(i.enc, Enc::Vop3 | Enc::Vop3p) || i.sdwa.is_some();
        if vop3 && (m.clamp || m.omod != 0) && !float_result(self.name) {
            return Err(self.un());
        }
        match i.enc {
            Enc::Sop1 | Enc::Sop2 | Enc::Sopc | Enc::Sopk => self.scalar(),
            _ => {
                let mut res = self.vector()?;
                if let Some(sd) = i.sdwa {
                    if sd.dst_sel != 6 && matches!(res.data.first(), Some(Opd::V(_))) {
                        let Some(Opd::V(r)) = res.data.first().copied() else {
                            return Err(self.un());
                        };
                        res.data[0] = Opd::V(self.sdwa_dst(r, sd)?);
                    }
                }
                Ok(res)
            }
        }
    }
}

impl<K: Backend> Res<K> {
    fn v(v: K::V) -> Self {
        Self {
            data: vec![Opd::V(v)],
            scc: None,
        }
    }
    fn b(b: K::B) -> Self {
        Self {
            data: vec![Opd::B(b)],
            scc: None,
        }
    }
    fn scc(s: K::V) -> Self {
        Self {
            data: vec![],
            scc: Some(s),
        }
    }
    fn with_scc(mut self, s: K::V) -> Self {
        self.scc = Some(s);
        self
    }
}

/// Opcodes whose result is an f32 (the VOP3 clamp and output modifiers apply).
fn float_result(n: &str) -> bool {
    n.starts_with("v_cvt_f32_")
        || (n.ends_with("_f32")
            && !n.starts_with("v_cmp")
            && !n.starts_with("v_frexp_exp")
            && !(n.starts_with("v_cvt_") && !n.starts_with("v_cvt_f32_")))
        || n.starts_with("v_cube")
        || n.starts_with("v_cmp")
        || n.starts_with("v_fma_mix")
}

macro_rules! f1 {
    ($s:ident, $op:expr) => {{
        let a = $s.fsrc(0)?;
        let r = $s.b.fun($op, a);
        $s.fout(r)
    }};
}

macro_rules! f2 {
    ($s:ident, |$a:ident, $b:ident| $e:expr) => {{
        let $a = $s.fsrc(0)?;
        let $b = $s.fsrc(1)?;
        let r = $e;
        $s.fout(r)
    }};
}

macro_rules! f3 {
    ($s:ident, |$a:ident, $b:ident, $c:ident| $e:expr) => {{
        let $a = $s.fsrc(0)?;
        let $b = $s.fsrc(1)?;
        let $c = $s.fsrc(2)?;
        let r = $e;
        $s.fout(r)
    }};
}

macro_rules! i2 {
    ($s:ident, |$a:ident, $b:ident| $e:expr) => {{
        let $a = $s.v(0)?;
        let $b = $s.v(1)?;
        let r = $e;
        Ok(Res::v(r))
    }};
}

macro_rules! i3 {
    ($s:ident, |$a:ident, $b:ident, $c:ident| $e:expr) => {{
        let $a = $s.v(0)?;
        let $b = $s.v(1)?;
        let $c = $s.v(2)?;
        let r = $e;
        Ok(Res::v(r))
    }};
}

impl<K: Backend> X<'_, K> {
    /// Source `i`, dword `k`, of a mask operation. A scalar that is not a mask gives the
    /// lane's bit in its dword 0 (the probe's lane is 0): every dword of a wave64 mask holds
    /// that one bit.
    fn mask_src(&self, i: usize, k: usize, want_b: bool) -> R<Opd<K>> {
        match self.src(i, k)? {
            Opd::V(_) if want_b && k > 0 => self.src(i, 0),
            o => Ok(o),
        }
    }

    fn bitop(&mut self, bit: Bit, words: usize) -> R<Res<K>> {
        let unary = matches!(bit, Bit::Not | Bit::Mov);
        let mut data = Vec::new();
        for k in 0..words {
            let want_b = self.c.dst_mask.get(k).copied().unwrap_or(false);
            let a = self.mask_src(0, k, want_b)?;
            let b = if unary {
                None
            } else {
                Some(self.mask_src(1, k, want_b)?)
            };
            data.push(self.bit(bit, a, b, want_b));
        }
        let scc = if self.writes_scc() {
            self.scc_nz(&data)
        } else {
            None
        };
        Ok(Res { data, scc })
    }

    fn saveexec(&mut self) -> R<Res<K>> {
        let n = self.name;
        let (op, tail) = n
            .strip_prefix("s_")
            .and_then(|r| {
                r.split_once("_saveexec_")
                    .or_else(|| r.split_once("_wrexec_"))
            })
            .ok_or_else(|| self.un())?;
        let wr = n.contains("_wrexec_");
        let bit = bit_named(op).ok_or_else(|| self.un())?;
        let w = if tail == "b64" { 2 } else { 1 };
        let mut old = Vec::new();
        let mut new = Vec::new();
        for k in 0..w {
            let reg = if k == 0 { Reg::ExecLo } else { Reg::ExecHi };
            let e = self.imp(reg).ok_or_else(|| self.un())?;
            let want_b = self.c.dst_mask.get(w + k).copied().unwrap_or(false);
            let s = self.mask_src(0, k, want_b)?;
            old.push(e);
            new.push(self.bit(bit, s, Some(e), want_b));
        }
        let scc = self.scc_nz(&new);
        if wr {
            // The SGPR gets the new EXEC too. ISA text, not verified on hardware:
            // `s_andn1_wrexec`: `exec = sdst = ~src & exec`; `s_andn2_wrexec`:
            // `exec = sdst = src & ~exec`; SCC = result != 0.
            let mut data = Vec::new();
            for (k, &e) in old.iter().enumerate() {
                let want_b = self.c.dst_mask.get(k).copied().unwrap_or(false);
                let s = self.mask_src(0, k, want_b)?;
                data.push(self.bit(bit, s, Some(e), want_b));
            }
            data.extend(new);
            return Ok(Res { data, scc });
        }
        old.extend(new);
        Ok(Res { data: old, scc })
    }

    /// The hardware-register writes the table accepts: everything that does not touch the f32
    /// rounding and denormal bits of MODE.
    fn setreg(&mut self) -> R<Res<K>> {
        let s = u32::from(self.i.simm16());
        let (id, off, size) = (s & 0x3f, (s >> 6) & 31, ((s >> 11) & 31) + 1);
        let bits = if size >= 32 {
            u32::MAX
        } else {
            ((1u32 << size) - 1) << off
        };
        if id == 1 && bits & 0x33 != 0 {
            return Err(Unsupported {
                chunk: 6,
                op: "s_setreg of the f32 round / denormal mode".to_string(),
            });
        }
        Ok(Res {
            data: vec![],
            scc: None,
        })
    }

    fn scalar(&mut self) -> R<Res<K>> {
        let name = self.name;
        if name.contains("saveexec") || name.contains("wrexec") {
            return self.saveexec();
        }
        if let Some(rest) = name.strip_prefix("s_") {
            let split = rest
                .strip_suffix("_b64")
                .map(|s| (s, 2))
                .or_else(|| rest.strip_suffix("_b32").map(|s| (s, 1)));
            if let Some((stem, words)) = split {
                if let Some(bit) = bit_named(stem) {
                    return self.bitop(bit, words);
                }
            }
        }
        if let Some(r) = name
            .strip_prefix("s_cmp_")
            .or_else(|| name.strip_prefix("s_cmpk_"))
        {
            return self.scmp(r);
        }
        match name {
            "s_add_u32" | "s_addc_u32" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                let cin = if name == "s_addc_u32" {
                    let c = self.imp_scc()?;
                    Some(c)
                } else {
                    None
                };
                let cv = match cin {
                    Some(c) => self.bv(c),
                    None => self.k(0),
                };
                let t = self.ib(IBin::Add, a, b);
                let r = self.ib(IBin::Add, t, cv);
                let mut carry = self.ic(ICmp::ULt, r, a);
                if let Some(c) = cin {
                    let eq = self.ic(ICmp::Eq, r, a);
                    let e = self.b.band(c, eq);
                    carry = self.b.bor(carry, e);
                }
                let s = self.bv(carry);
                Ok(Res::v(r).with_scc(s))
            }
            "s_sub_u32" | "s_subb_u32" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                let cin = if name == "s_subb_u32" {
                    Some(self.imp_scc()?)
                } else {
                    None
                };
                let cv = match cin {
                    Some(c) => self.bv(c),
                    None => self.k(0),
                };
                let t = self.ib(IBin::Sub, a, b);
                let r = self.ib(IBin::Sub, t, cv);
                let mut borrow = self.ic(ICmp::ULt, a, b);
                if let Some(c) = cin {
                    let eq = self.ic(ICmp::Eq, a, b);
                    let e = self.b.band(c, eq);
                    borrow = self.b.bor(borrow, e);
                }
                let s = self.bv(borrow);
                Ok(Res::v(r).with_scc(s))
            }
            "s_add_i32" | "s_addk_i32" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                let r = self.ib(IBin::Add, a, b);
                let x = self.ib(IBin::Xor, a, r);
                let y = self.ib(IBin::Xor, b, r);
                let o = self.ib(IBin::And, x, y);
                let s = self.shr_k(o, 31);
                Ok(Res::v(r).with_scc(s))
            }
            "s_sub_i32" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                let r = self.ib(IBin::Sub, a, b);
                let x = self.ib(IBin::Xor, a, b);
                let y = self.ib(IBin::Xor, a, r);
                let o = self.ib(IBin::And, x, y);
                let s = self.shr_k(o, 31);
                Ok(Res::v(r).with_scc(s))
            }
            "s_min_i32" | "s_min_u32" | "s_max_i32" | "s_max_u32" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                let op = match name {
                    "s_min_i32" => ICmp::SLt,
                    "s_min_u32" => ICmp::ULt,
                    "s_max_i32" => ICmp::SGt,
                    _ => ICmp::UGt,
                };
                let c = self.ic(op, a, b);
                let r = self.sel(c, a, b);
                let s = self.bv(c);
                Ok(Res::v(r).with_scc(s))
            }
            "s_mul_i32" | "s_mulk_i32" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                Ok(Res::v(self.ib(IBin::Mul, a, b)))
            }
            "s_mul_hi_u32" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                Ok(Res::v(self.ib(IBin::UMulHi, a, b)))
            }
            "s_mul_hi_i32" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                Ok(Res::v(self.ib(IBin::SMulHi, a, b)))
            }
            "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                let op = match name {
                    "s_lshl_b32" => IBin::Shl,
                    "s_lshr_b32" => IBin::Shr,
                    _ => IBin::Sar,
                };
                let r = self.shift(op, a, b);
                let s = self.scc_nz(&[Opd::V(r)]);
                Ok(Res {
                    data: vec![Opd::V(r)],
                    scc: s,
                })
            }
            "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => {
                let (lo, hi) = (self.vk(0, 0)?, self.vk(0, 1)?);
                let amt = self.v(1)?;
                let op = match name {
                    "s_lshl_b64" => IBin::Shl,
                    "s_lshr_b64" => IBin::Shr,
                    _ => IBin::Sar,
                };
                let (l, h) = self.shift64(op, lo, hi, amt);
                let data = vec![Opd::V(l), Opd::V(h)];
                let s = self.scc_nz(&data);
                Ok(Res { data, scc: s })
            }
            "s_bfm_b32" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                let n = self.and_k(a, 31);
                let one = self.k(1);
                let m = self.ib(IBin::Shl, one, n);
                let m = self.ibk(IBin::Sub, m, 1);
                Ok(Res::v(self.shift(IBin::Shl, m, b)))
            }
            "s_bfe_u32" | "s_bfe_i32" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                let w = self.shr_k(b, 16);
                let w = self.and_k(w, 0x7f);
                let r = self.bfe(a, b, w, name == "s_bfe_i32");
                let s = self.scc_nz(&[Opd::V(r)]);
                Ok(Res {
                    data: vec![Opd::V(r)],
                    scc: s,
                })
            }
            "s_absdiff_i32" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                let lt = self.ic(ICmp::SLt, a, b);
                let ab = self.ib(IBin::Sub, a, b);
                let ba = self.ib(IBin::Sub, b, a);
                let r = self.sel(lt, ba, ab);
                let s = self.scc_nz(&[Opd::V(r)]);
                Ok(Res {
                    data: vec![Opd::V(r)],
                    scc: s,
                })
            }
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                let n = u32::from(name.as_bytes()[6] - b'0');
                let (a, b) = (self.v(0)?, self.v(1)?);
                let sh = self.shl_k(a, n);
                let r = self.ib(IBin::Add, sh, b);
                let spill = self.shr_k(a, 32 - n);
                let c1 = self.ick(ICmp::Ne, spill, 0);
                let c2 = self.ic(ICmp::ULt, r, sh);
                let c = self.b.bor(c1, c2);
                let s = self.bv(c);
                Ok(Res::v(r).with_scc(s))
            }
            "s_pack_ll_b32_b16" | "s_pack_lh_b32_b16" | "s_pack_hl_b32_b16"
            | "s_pack_hh_b32_b16" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                let lo = if name.starts_with("s_pack_l") {
                    self.and_k(a, 0xffff)
                } else {
                    self.shr_k(a, 16)
                };
                let hi = if name.ends_with("hl_b32_b16") || name.ends_with("ll_b32_b16") {
                    self.shl_k(b, 16)
                } else {
                    self.and_k(b, 0xffff_0000)
                };
                Ok(Res::v(self.ib(IBin::Or, lo, hi)))
            }
            "s_cselect_b32" | "s_cselect_b64" => {
                let c = self.imp_scc()?;
                let words = if name.ends_with("b64") { 2 } else { 1 };
                let mut data = Vec::new();
                for k in 0..words {
                    let (a, b) = (self.src(0, k)?, self.src(1, k)?);
                    if self.c.dst_mask.get(k).copied().unwrap_or(false) {
                        let (a, b) = (self.as_b(a), self.as_b(b));
                        let t = self.b.band(c, a);
                        let nc = self.b.bnot(c);
                        let f = self.b.band(nc, b);
                        data.push(Opd::B(self.b.bor(t, f)));
                    } else {
                        let (a, b) = (self.as_v(a), self.as_v(b));
                        data.push(Opd::V(self.sel(c, a, b)));
                    }
                }
                Ok(Res { data, scc: None })
            }
            "s_cmov_b32" | "s_cmov_b64" | "s_cmovk_i32" => {
                let c = self.imp_scc()?;
                let words = self.width();
                let mut data = Vec::new();
                for k in 0..words {
                    let a = self.src(0, k)?;
                    let a = self.as_v(a);
                    let reg = self.c.inst.dsts[k].0;
                    let old = self.imp(reg).ok_or_else(|| self.un())?;
                    let old = self.as_v(old);
                    data.push(Opd::V(self.sel(c, a, old)));
                }
                Ok(Res { data, scc: None })
            }
            "s_mov_b32" | "s_movk_i32" => Ok(Res::v(self.v(0)?)),
            "s_brev_b32" => {
                let a = self.v(0)?;
                Ok(Res::v(self.b.iun(IUn::BitRev, a)))
            }
            "s_brev_b64" => {
                let (lo, hi) = (self.vk(0, 0)?, self.vk(0, 1)?);
                let l = self.b.iun(IUn::BitRev, hi);
                let h = self.b.iun(IUn::BitRev, lo);
                Ok(Res {
                    data: vec![Opd::V(l), Opd::V(h)],
                    scc: None,
                })
            }
            "s_bcnt0_i32_b32" | "s_bcnt1_i32_b32" | "s_bcnt0_i32_b64" | "s_bcnt1_i32_b64" => {
                let words = if name.ends_with("b64") { 2 } else { 1 };
                let mut total = self.k(0);
                for k in 0..words {
                    let a = self.vk(0, k)?;
                    let a = if name.starts_with("s_bcnt0") {
                        self.b.iun(IUn::Not, a)
                    } else {
                        a
                    };
                    let c = self.b.iun(IUn::BitCount, a);
                    total = self.ib(IBin::Add, total, c);
                }
                let s = self.scc_nz(&[Opd::V(total)]);
                Ok(Res {
                    data: vec![Opd::V(total)],
                    scc: s,
                })
            }
            "s_ff1_i32_b32" | "s_ff0_i32_b32" | "s_ff1_i32_b64" | "s_ff0_i32_b64" => {
                let words = if name.ends_with("b64") { 2 } else { 1 };
                let mut r = self.k(u32::MAX);
                for k in (0..words).rev() {
                    let a = self.vk(0, k)?;
                    let a = if name.starts_with("s_ff0") {
                        self.b.iun(IUn::Not, a)
                    } else {
                        a
                    };
                    let f = self.b.iun(IUn::FindLsb, a);
                    let found = self.ick(ICmp::Ne, f, u32::MAX);
                    let f = self.ibk(IBin::Add, f, 32 * k as u32);
                    r = self.sel(found, f, r);
                }
                Ok(Res::v(r))
            }
            "s_flbit_i32_b32" | "s_flbit_i32_b64" | "s_flbit_i32" => {
                let signed = name == "s_flbit_i32";
                let words = if name.ends_with("b64") { 2 } else { 1 };
                let mut r = self.k(u32::MAX);
                for k in 0..words {
                    let a = self.vk(0, k)?;
                    let m = self
                        .b
                        .iun(if signed { IUn::FindSMsb } else { IUn::FindUMsb }, a);
                    let found = self.ick(ICmp::Ne, m, u32::MAX);
                    let k31 = self.k(31 + 32 * (words as u32 - 1 - k as u32));
                    let clz = self.ib(IBin::Sub, k31, m);
                    r = self.sel(found, clz, r);
                }
                Ok(Res::v(r))
            }
            "s_sext_i32_i8" | "s_sext_i32_i16" => {
                let a = self.v(0)?;
                let n = if name.ends_with("i8") { 24 } else { 16 };
                let t = self.shl_k(a, n);
                Ok(Res::v(self.ibk(IBin::Sar, t, n)))
            }
            "s_abs_i32" => {
                let a = self.v(0)?;
                let z = self.k(0);
                let neg = self.ic(ICmp::SLt, a, z);
                let n = self.ib(IBin::Sub, z, a);
                let r = self.sel(neg, n, a);
                let s = self.scc_nz(&[Opd::V(r)]);
                Ok(Res {
                    data: vec![Opd::V(r)],
                    scc: s,
                })
            }
            "s_bitcmp0_b32" | "s_bitcmp1_b32" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                let t = self.shift(IBin::Shr, a, b);
                let t = self.and_k(t, 1);
                let want = u32::from(name == "s_bitcmp1_b32");
                let c = self.ick(ICmp::Eq, t, want);
                let s = self.bv(c);
                Ok(Res::scc(s))
            }
            "s_bitset0_b32" | "s_bitset1_b32" => {
                let a = self.v(0)?;
                let old = self.old_dst()?;
                let one = self.k(1);
                let bit = self.shift(IBin::Shl, one, a);
                Ok(Res::v(if name == "s_bitset1_b32" {
                    self.ib(IBin::Or, old, bit)
                } else {
                    let nb = self.b.iun(IUn::Not, bit);
                    self.ib(IBin::And, old, nb)
                }))
            }
            "s_setreg_b32" | "s_setreg_imm32_b32" => self.setreg(),
            _ => Err(self.un()),
        }
    }

    /// `s_cmp_*` and `s_cmpk_*`: SCC only.
    fn scmp(&mut self, r: &str) -> R<Res<K>> {
        let (op, ty) = r.split_once('_').ok_or_else(|| self.un())?;
        let c = match ty {
            "i32" | "u32" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                let signed = ty == "i32";
                let o = match (op, signed) {
                    ("eq", _) => ICmp::Eq,
                    ("lg", _) => ICmp::Ne,
                    ("gt", true) => ICmp::SGt,
                    ("ge", true) => ICmp::SGe,
                    ("lt", true) => ICmp::SLt,
                    ("le", true) => ICmp::SLe,
                    ("gt", false) => ICmp::UGt,
                    ("ge", false) => ICmp::UGe,
                    ("lt", false) => ICmp::ULt,
                    ("le", false) => ICmp::ULe,
                    _ => return Err(self.un()),
                };
                self.ic(o, a, b)
            }
            "u64" if op == "eq" || op == "lg" => {
                let lo = {
                    let (a, b) = (self.vk(0, 0)?, self.vk(1, 0)?);
                    self.ic(ICmp::Eq, a, b)
                };
                let hi = {
                    let (a, b) = (self.vk(0, 1)?, self.vk(1, 1)?);
                    self.ic(ICmp::Eq, a, b)
                };
                let e = self.b.band(lo, hi);
                if op == "eq" {
                    e
                } else {
                    self.b.bnot(e)
                }
            }
            _ => return Err(self.un()),
        };
        let s = self.bv(c);
        Ok(Res::scc(s))
    }
}

impl<K: Backend> X<'_, K> {
    /// `v_cmp_*` and `v_cmpx_*`: the lane's result as a mask.
    fn vcmp(&mut self) -> R<Option<Res<K>>> {
        let rest = match self
            .name
            .strip_prefix("v_cmpx_")
            .or_else(|| self.name.strip_prefix("v_cmp_"))
        {
            Some(r) => r,
            None => return Ok(None),
        };
        if rest == "class_f32" {
            let a = self.fsrc(0)?;
            let mask = self.v(1)?;
            return Ok(Some(Res::b(self.cmp_class(a, mask))));
        }
        let (op, ty) = rest.split_once('_').ok_or_else(|| self.un())?;
        let c = match ty {
            "f32" => {
                if op == "f" || op == "tru" {
                    let c = self.b.kb(op == "tru");
                    return Ok(Some(Res::b(c)));
                }
                let o = match op {
                    "lt" => FCmp::OLt,
                    "eq" => FCmp::OEq,
                    "le" => FCmp::OLe,
                    "gt" => FCmp::OGt,
                    "lg" => FCmp::ONe,
                    "ge" => FCmp::OGe,
                    "o" => FCmp::Ord,
                    "u" => FCmp::Uno,
                    "nge" => FCmp::ULt,
                    "nlg" => FCmp::UEq,
                    "ngt" => FCmp::ULe,
                    "nle" => FCmp::UGt,
                    "neq" => FCmp::UNe,
                    "nlt" => FCmp::UGe,
                    _ => return Err(self.un()),
                };
                let (a, b) = (self.fsrc(0)?, self.fsrc(1)?);
                self.fc(o, a, b)
            }
            "i32" | "u32" | "i16" | "u16" => {
                if op == "f" || op == "t" {
                    let c = self.b.kb(op == "t");
                    return Ok(Some(Res::b(c)));
                }
                let s = ty.starts_with('i');
                let o = match (op, s) {
                    ("eq", _) => ICmp::Eq,
                    ("ne", _) => ICmp::Ne,
                    ("lt", true) => ICmp::SLt,
                    ("le", true) => ICmp::SLe,
                    ("gt", true) => ICmp::SGt,
                    ("ge", true) => ICmp::SGe,
                    ("lt", false) => ICmp::ULt,
                    ("le", false) => ICmp::ULe,
                    ("gt", false) => ICmp::UGt,
                    ("ge", false) => ICmp::UGe,
                    _ => return Err(self.un()),
                };
                let (mut a, mut b) = (self.v(0)?, self.v(1)?);
                if ty.ends_with("16") {
                    if self.i.mods.opsel != 0 {
                        return Err(self.un());
                    }
                    for x in [&mut a, &mut b] {
                        let t = self.shl_k(*x, 16);
                        *x = self.ibk(if s { IBin::Sar } else { IBin::Shr }, t, 16);
                    }
                }
                self.ic(o, a, b)
            }
            "i64" | "u64" => {
                if op == "f" || op == "t" {
                    let c = self.b.kb(op == "t");
                    return Ok(Some(Res::b(c)));
                }
                let (alo, ahi) = (self.vk(0, 0)?, self.vk(0, 1)?);
                let (blo, bhi) = (self.vk(1, 0)?, self.vk(1, 1)?);
                let hi_eq = self.ic(ICmp::Eq, ahi, bhi);
                let lo_eq = self.ic(ICmp::Eq, alo, blo);
                let eq = self.b.band(hi_eq, lo_eq);
                let s = ty == "i64";
                let (hlt, hgt) = if s {
                    (self.ic(ICmp::SLt, ahi, bhi), self.ic(ICmp::SGt, ahi, bhi))
                } else {
                    (self.ic(ICmp::ULt, ahi, bhi), self.ic(ICmp::UGt, ahi, bhi))
                };
                let llt = self.ic(ICmp::ULt, alo, blo);
                let lgt = self.ic(ICmp::UGt, alo, blo);
                let t = self.b.band(hi_eq, llt);
                let lt = self.b.bor(hlt, t);
                let t = self.b.band(hi_eq, lgt);
                let gt = self.b.bor(hgt, t);
                match op {
                    "eq" => eq,
                    "ne" => self.b.bnot(eq),
                    "lt" => lt,
                    "gt" => gt,
                    "le" => self.b.bor(lt, eq),
                    "ge" => self.b.bor(gt, eq),
                    _ => return Err(self.un()),
                }
            }
            _ => return Err(self.un()),
        };
        Ok(Some(Res::b(c)))
    }

    /// `v_cube{id,sc,tc,ma}_f32` of the major axis of (x, y, z).
    fn cube(&mut self, what: &str) -> R<Res<K>> {
        let (x, y, z) = (self.fsrc(0)?, self.fsrc(1)?, self.fsrc(2)?);
        let (ax, ay, az) = (self.and_k(x, ABS), self.and_k(y, ABS), self.and_k(z, ABS));
        let zx = self.fc(FCmp::OGe, az, ax);
        let zy = self.fc(FCmp::OGe, az, ay);
        let z_major = self.b.band(zx, zy);
        let y_major = self.fc(FCmp::OGe, ay, ax);
        let zero = self.fk(0.0);
        let (xn, yn, zn) = (
            self.fc(FCmp::OLt, x, zero),
            self.fc(FCmp::OLt, y, zero),
            self.fc(FCmp::OLt, z, zero),
        );
        let r = match what {
            "id" => {
                let (k0, k1, k2, k3, k4, k5) = (
                    self.fk(0.0),
                    self.fk(1.0),
                    self.fk(2.0),
                    self.fk(3.0),
                    self.fk(4.0),
                    self.fk(5.0),
                );
                let xm = self.sel(xn, k1, k0);
                let ym = self.sel(yn, k3, k2);
                let zm = self.sel(zn, k5, k4);
                let r = self.sel(y_major, ym, xm);
                self.sel(z_major, zm, r)
            }
            "ma" => {
                let two = self.fk(2.0);
                let m = self.sel(y_major, y, x);
                let m = self.sel(z_major, z, m);
                self.fb(FBin::Mul, m, two)
            }
            "sc" => {
                let nx = self.fneg(x);
                let nz = self.fneg(z);
                let zs = self.sel(zn, nx, x);
                let xs = self.sel(xn, z, nz);
                let r = self.sel(y_major, x, xs);
                self.sel(z_major, zs, r)
            }
            _ => {
                let ny = self.fneg(y);
                let nz = self.fneg(z);
                let ys = self.sel(yn, nz, z);
                let r = self.sel(y_major, ys, ny);
                self.sel(z_major, ny, r)
            }
        };
        self.fout(r)
    }

    fn vector(&mut self) -> R<Res<K>> {
        if let Some(r) = self.vcmp()? {
            return Ok(r);
        }
        let name = self.name;
        match name {
            // -- moves and bit operations
            "v_mov_b32" | "v_readfirstlane_b32" | "v_readlane_b32" => Ok(Res::v(self.v(0)?)),
            // the console lane index: `mbcnt(-1, x)` counts the lanes below this one
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                if self.c.inst.srcs[0].first() != Some(&Src::Const(u32::MAX)) {
                    // the count of a real mask: a ballot (chunk 7c)
                    return Err(Unsupported {
                        chunk: 7,
                        op: format!("{name} of a mask"),
                    });
                }
                let Some(lane) = self.c.lane else {
                    return Err(Unsupported {
                        chunk: 7,
                        op: format!("{name} (the lane index of a vertex or pixel program)"),
                    });
                };
                let x = self.v(1)?;
                let c32 = self.k(32);
                let r = if name == "v_mbcnt_lo_u32_b32" {
                    let l = self.b.ibin(IBin::UMin, lane, c32);
                    self.b.ibin(IBin::Add, l, x)
                } else {
                    let l = self.b.ibin(IBin::UMax, lane, c32);
                    let l = self.b.ibin(IBin::Sub, l, c32);
                    self.b.ibin(IBin::Add, l, x)
                };
                Ok(Res::v(r))
            }
            "v_not_b32" => {
                let a = self.v(0)?;
                Ok(Res::v(self.b.iun(IUn::Not, a)))
            }
            "v_bfrev_b32" => {
                let a = self.v(0)?;
                Ok(Res::v(self.b.iun(IUn::BitRev, a)))
            }
            "v_ffbl_b32" => {
                let a = self.v(0)?;
                Ok(Res::v(self.b.iun(IUn::FindLsb, a)))
            }
            "v_ffbh_u32" | "v_ffbh_i32" => {
                let a = self.v(0)?;
                let op = if name.ends_with("u32") {
                    IUn::FindUMsb
                } else {
                    IUn::FindSMsb
                };
                let m = self.b.iun(op, a);
                let found = self.ick(ICmp::Ne, m, u32::MAX);
                let k31 = self.k(31);
                let clz = self.ib(IBin::Sub, k31, m);
                let none = self.k(u32::MAX);
                Ok(Res::v(self.sel(found, clz, none)))
            }
            "v_cndmask_b32" => {
                let (a, b) = (self.fsrc(0)?, self.fsrc(1)?);
                let m = self.m(2)?;
                Ok(Res::v(self.sel(m, b, a)))
            }
            "v_and_b32" => i2!(self, |a, b| self.ib(IBin::And, a, b)),
            "v_or_b32" => i2!(self, |a, b| self.ib(IBin::Or, a, b)),
            "v_xor_b32" => i2!(self, |a, b| self.ib(IBin::Xor, a, b)),
            "v_xnor_b32" => i2!(self, |a, b| {
                let t = self.ib(IBin::Xor, a, b);
                self.b.iun(IUn::Not, t)
            }),
            "v_lshlrev_b32" => i2!(self, |a, b| self.shift(IBin::Shl, b, a)),
            "v_lshrrev_b32" => i2!(self, |a, b| self.shift(IBin::Shr, b, a)),
            "v_ashrrev_i32" => i2!(self, |a, b| self.shift(IBin::Sar, b, a)),
            // 16-bit integer operations: the low half is the operand and the result, the high
            // half of the result is zero
            "v_lshrrev_b16" | "v_lshlrev_b16" | "v_ashrrev_i16" | "v_add_nc_u16"
            | "v_sub_nc_u16" => {
                if self.i.mods.opsel != 0 {
                    return Err(self.un());
                }
                let (a, b) = (self.v(0)?, self.v(1)?);
                let r = match name {
                    "v_add_nc_u16" => self.ib(IBin::Add, a, b),
                    "v_sub_nc_u16" => self.ib(IBin::Sub, a, b),
                    _ => {
                        let n = self.and_k(a, 15);
                        match name {
                            "v_lshrrev_b16" => {
                                let x = self.and_k(b, 0xffff);
                                self.ib(IBin::Shr, x, n)
                            }
                            "v_lshlrev_b16" => self.ib(IBin::Shl, b, n),
                            _ => {
                                let x = self.shl_k(b, 16);
                                let x = self.ibk(IBin::Sar, x, 16);
                                self.ib(IBin::Sar, x, n)
                            }
                        }
                    }
                };
                Ok(Res::v(self.and_k(r, 0xffff)))
            }
            "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => {
                let amt = self.v(0)?;
                let (lo, hi) = (self.vk(1, 0)?, self.vk(1, 1)?);
                let op = match name {
                    "v_lshlrev_b64" => IBin::Shl,
                    "v_lshrrev_b64" => IBin::Shr,
                    _ => IBin::Sar,
                };
                let (l, h) = self.shift64(op, lo, hi, amt);
                Ok(Res {
                    data: vec![Opd::V(l), Opd::V(h)],
                    scc: None,
                })
            }
            // -- integer arithmetic
            "v_add_nc_u32" => i2!(self, |a, b| self.ib(IBin::Add, a, b)),
            "v_sub_nc_u32" => i2!(self, |a, b| self.ib(IBin::Sub, a, b)),
            "v_subrev_nc_u32" => i2!(self, |a, b| self.ib(IBin::Sub, b, a)),
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                let (r, c) = match name {
                    "v_add_co_u32" => {
                        let r = self.ib(IBin::Add, a, b);
                        (r, self.ic(ICmp::ULt, r, a))
                    }
                    "v_sub_co_u32" => (self.ib(IBin::Sub, a, b), self.ic(ICmp::ULt, a, b)),
                    _ => (self.ib(IBin::Sub, b, a), self.ic(ICmp::ULt, b, a)),
                };
                Ok(Res {
                    data: vec![Opd::V(r), Opd::B(c)],
                    scc: None,
                })
            }
            "v_add_co_ci_u32" | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => {
                let (mut a, mut b) = (self.v(0)?, self.v(1)?);
                let cin = self.m(2)?;
                let cv = self.bv(cin);
                let (r, c) = if name == "v_add_co_ci_u32" {
                    let t = self.ib(IBin::Add, a, b);
                    let r = self.ib(IBin::Add, t, cv);
                    let c1 = self.ic(ICmp::ULt, r, a);
                    let eq = self.ic(ICmp::Eq, r, a);
                    let c2 = self.b.band(cin, eq);
                    (r, self.b.bor(c1, c2))
                } else {
                    if name == "v_subrev_co_ci_u32" {
                        std::mem::swap(&mut a, &mut b);
                    }
                    let t = self.ib(IBin::Sub, a, b);
                    let r = self.ib(IBin::Sub, t, cv);
                    let c1 = self.ic(ICmp::ULt, a, b);
                    let eq = self.ic(ICmp::Eq, a, b);
                    let c2 = self.b.band(cin, eq);
                    (r, self.b.bor(c1, c2))
                };
                Ok(Res {
                    data: vec![Opd::V(r), Opd::B(c)],
                    scc: None,
                })
            }
            "v_min_i32" => i2!(self, |a, b| self.ib(IBin::SMin, a, b)),
            "v_max_i32" => i2!(self, |a, b| self.ib(IBin::SMax, a, b)),
            "v_min_u32" => i2!(self, |a, b| self.ib(IBin::UMin, a, b)),
            "v_max_u32" => i2!(self, |a, b| self.ib(IBin::UMax, a, b)),
            "v_min3_i32" | "v_max3_i32" | "v_min3_u32" | "v_max3_u32" => {
                let op = match (name.contains("min"), name.ends_with("i32")) {
                    (true, true) => IBin::SMin,
                    (true, false) => IBin::UMin,
                    (false, true) => IBin::SMax,
                    (false, false) => IBin::UMax,
                };
                i3!(self, |a, b, c| {
                    let t = self.ib(op, a, b);
                    self.ib(op, t, c)
                })
            }
            "v_med3_i32" | "v_med3_u32" => {
                let (mn, mx) = if name.ends_with("i32") {
                    (IBin::SMin, IBin::SMax)
                } else {
                    (IBin::UMin, IBin::UMax)
                };
                i3!(self, |a, b, c| {
                    let lo = self.ib(mn, a, b);
                    let hi = self.ib(mx, a, b);
                    let m = self.ib(mn, hi, c);
                    self.ib(mx, lo, m)
                })
            }
            "v_mul_lo_u32" => i2!(self, |a, b| self.ib(IBin::Mul, a, b)),
            "v_mul_hi_u32" => i2!(self, |a, b| self.ib(IBin::UMulHi, a, b)),
            "v_mul_hi_i32" => i2!(self, |a, b| self.ib(IBin::SMulHi, a, b)),
            "v_mul_u32_u24" | "v_mul_hi_u32_u24" | "v_mad_u32_u24" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                let (a, b) = (self.and_k(a, 0xff_ffff), self.and_k(b, 0xff_ffff));
                let r = if name == "v_mul_hi_u32_u24" {
                    self.ib(IBin::UMulHi, a, b)
                } else {
                    self.ib(IBin::Mul, a, b)
                };
                if name == "v_mad_u32_u24" {
                    let c = self.v(2)?;
                    Ok(Res::v(self.ib(IBin::Add, r, c)))
                } else {
                    Ok(Res::v(r))
                }
            }
            "v_mul_i32_i24" | "v_mul_hi_i32_i24" | "v_mad_i32_i24" => {
                let (a, b) = (self.v(0)?, self.v(1)?);
                let a = self.shl_k(a, 8);
                let a = self.ibk(IBin::Sar, a, 8);
                let b = self.shl_k(b, 8);
                let b = self.ibk(IBin::Sar, b, 8);
                let r = if name == "v_mul_hi_i32_i24" {
                    self.ib(IBin::SMulHi, a, b)
                } else {
                    self.ib(IBin::Mul, a, b)
                };
                if name == "v_mad_i32_i24" {
                    let c = self.v(2)?;
                    Ok(Res::v(self.ib(IBin::Add, r, c)))
                } else {
                    Ok(Res::v(r))
                }
            }
            "v_add3_u32" => i3!(self, |a, b, c| {
                let t = self.ib(IBin::Add, a, b);
                self.ib(IBin::Add, t, c)
            }),
            "v_lshl_add_u32" => i3!(self, |a, b, c| {
                let t = self.shift(IBin::Shl, a, b);
                self.ib(IBin::Add, t, c)
            }),
            "v_add_lshl_u32" => i3!(self, |a, b, c| {
                let t = self.ib(IBin::Add, a, b);
                self.shift(IBin::Shl, t, c)
            }),
            "v_lshl_or_b32" => i3!(self, |a, b, c| {
                let t = self.shift(IBin::Shl, a, b);
                self.ib(IBin::Or, t, c)
            }),
            "v_and_or_b32" => i3!(self, |a, b, c| {
                let t = self.ib(IBin::And, a, b);
                self.ib(IBin::Or, t, c)
            }),
            "v_or3_b32" => i3!(self, |a, b, c| {
                let t = self.ib(IBin::Or, a, b);
                self.ib(IBin::Or, t, c)
            }),
            "v_xad_u32" => i3!(self, |a, b, c| {
                let t = self.ib(IBin::Xor, a, b);
                self.ib(IBin::Add, t, c)
            }),
            "v_bcnt_u32_b32" => i2!(self, |a, b| {
                let c = self.b.iun(IUn::BitCount, a);
                self.ib(IBin::Add, c, b)
            }),
            "v_bfm_b32" => i2!(self, |a, b| {
                let n = self.and_k(a, 31);
                let one = self.k(1);
                let m = self.ib(IBin::Shl, one, n);
                let m = self.ibk(IBin::Sub, m, 1);
                self.shift(IBin::Shl, m, b)
            }),
            "v_bfe_u32" | "v_bfe_i32" => i3!(self, |a, b, c| {
                let w = self.and_k(c, 31);
                self.bfe(a, b, w, name == "v_bfe_i32")
            }),
            "v_bfi_b32" => i3!(self, |m, a, b| {
                let t = self.ib(IBin::And, m, a);
                let nm = self.b.iun(IUn::Not, m);
                let u = self.ib(IBin::And, nm, b);
                self.ib(IBin::Or, t, u)
            }),
            "v_alignbit_b32" => i3!(self, |a, b, c| {
                let s = self.and_k(c, 31);
                self.funnel(a, b, s)
            }),
            "v_alignbyte_b32" => i3!(self, |a, b, c| {
                let s = self.and_k(c, 3);
                let s = self.shl_k(s, 3);
                self.funnel(a, b, s)
            }),
            "v_perm_b32" => i3!(self, |a, b, sel| {
                let mut out = self.k(0);
                for byte in 0..4u32 {
                    let s = self.shr_k(sel, 8 * byte);
                    let s = self.and_k(s, 0xff);
                    // bytes 0..=3 of b, 4..=7 of a
                    let sh = self.and_k(s, 3);
                    let sh = self.shl_k(sh, 3);
                    let from_b = self.ib(IBin::Shr, b, sh);
                    let from_a = self.ib(IBin::Shr, a, sh);
                    let is_a = self.ick(ICmp::UGe, s, 4);
                    let v = self.sel(is_a, from_a, from_b);
                    let v = self.and_k(v, 0xff);
                    // 8..=11: 0xff when bit 15 / 31 of b (8, 9) or of a (10, 11) is set
                    let sg = {
                        let hi_pair = self.ick(ICmp::UGe, s, 10);
                        let x = self.sel(hi_pair, a, b);
                        let t = self.and_k(s, 1);
                        let odd = self.ick(ICmp::Ne, t, 0);
                        let b15 = self.shr_k(x, 15);
                        let b31 = self.shr_k(x, 31);
                        let bit = self.sel(odd, b31, b15);
                        let bit = self.and_k(bit, 1);
                        let nz = self.ick(ICmp::Ne, bit, 0);
                        let zero = self.k(0);
                        self.selk(nz, 0xff, zero)
                    };
                    let is_sign = {
                        let ge8 = self.ick(ICmp::UGe, s, 8);
                        let le11 = self.ick(ICmp::ULe, s, 11);
                        self.b.band(ge8, le11)
                    };
                    let v = self.sel(is_sign, sg, v);
                    let z = self.ick(ICmp::Eq, s, 12);
                    let zero = self.k(0);
                    let v = self.sel(z, zero, v);
                    let hi = self.ick(ICmp::UGe, s, 13);
                    let v = self.selk(hi, 0xff, v);
                    let v = self.shl_k(v, 8 * byte);
                    out = self.ib(IBin::Or, out, v);
                }
                out
            }),
            // -- float arithmetic
            "v_add_f32" => f2!(self, |a, b| self.fb(FBin::Add, a, b)),
            "v_sub_f32" => f2!(self, |a, b| self.fb(FBin::Sub, a, b)),
            "v_subrev_f32" => f2!(self, |a, b| self.fb(FBin::Sub, b, a)),
            "v_mul_f32" => f2!(self, |a, b| self.fb(FBin::Mul, a, b)),
            "v_mul_legacy_f32" => f2!(self, |a, b| self.mul_legacy(a, b)),
            "v_min_f32" => f2!(self, |a, b| self.fminmax(a, b, false)),
            "v_max_f32" => f2!(self, |a, b| self.fminmax(a, b, true)),
            "v_fma_f32" => f3!(self, |a, b, c| self.b.fma(a, b, c)),
            "v_mad_f32" => f3!(self, |a, b, c| {
                let t = self.fb(FBin::Mul, a, b);
                self.fb(FBin::Add, t, c)
            }),
            "v_mad_legacy_f32" => f3!(self, |a, b, c| {
                let t = self.mul_legacy(a, b);
                self.fb(FBin::Add, t, c)
            }),
            "v_fmaak_f32" | "v_fmamk_f32" => f3!(self, |a, b, c| self.b.fma(a, b, c)),
            "v_madak_f32" | "v_madmk_f32" => f3!(self, |a, b, c| {
                let t = self.fb(FBin::Mul, a, b);
                self.fb(FBin::Add, t, c)
            }),
            "v_fmac_f32" | "v_mac_f32" | "v_mac_legacy_f32" | "v_fmac_legacy_f32" => {
                let (a, b) = (self.fsrc(0)?, self.fsrc(1)?);
                let d = self.old_dst()?;
                let r = match name {
                    "v_fmac_f32" => self.b.fma(a, b, d),
                    "v_mac_f32" => {
                        let t = self.fb(FBin::Mul, a, b);
                        self.fb(FBin::Add, t, d)
                    }
                    _ => {
                        let t = self.mul_legacy(a, b);
                        self.fb(FBin::Add, t, d)
                    }
                };
                self.fout(r)
            }
            "v_min3_f32" => f3!(self, |a, b, c| {
                let t = self.fminmax(a, b, false);
                self.fminmax(t, c, false)
            }),
            "v_max3_f32" => f3!(self, |a, b, c| {
                let t = self.fminmax(a, b, true);
                self.fminmax(t, c, true)
            }),
            "v_med3_f32" => f3!(self, |a, b, c| self.fmed3(a, b, c)),
            "v_ldexp_f32" => {
                let a = self.fsrc(0)?;
                let e = self.v(1)?;
                let e = self.ibk(IBin::SMin, e, 400);
                let e = self.ibk(IBin::SMax, e, (-400i32) as u32);
                let r = self.b.ldexp(a, e);
                self.fout(r)
            }
            "v_floor_f32" => f1!(self, FUn::Floor),
            "v_ceil_f32" => f1!(self, FUn::Ceil),
            "v_trunc_f32" => f1!(self, FUn::Trunc),
            "v_rndne_f32" => f1!(self, FUn::RoundEven),
            "v_sqrt_f32" => f1!(self, FUn::Sqrt),
            "v_rsq_f32" => f1!(self, FUn::Rsq),
            "v_rcp_f32" | "v_rcp_iflag_f32" => f1!(self, FUn::Rcp),
            "v_exp_f32" => f1!(self, FUn::Exp2),
            "v_log_f32" => f1!(self, FUn::Log2),
            "v_sin_f32" | "v_cos_f32" => {
                let a = self.fsrc(0)?;
                let tau = self.fk(std::f32::consts::TAU);
                let t = self.fb(FBin::Mul, a, tau);
                let op = if name == "v_sin_f32" {
                    FUn::Sin
                } else {
                    FUn::Cos
                };
                let r = self.b.fun(op, t);
                self.fout(r)
            }
            "v_fract_f32" => {
                let a = self.fsrc(0)?;
                let fl = self.b.fun(FUn::Floor, a);
                let r = self.fb(FBin::Sub, a, fl);
                self.fout(r)
            }
            "v_frexp_exp_i32_f32" => {
                let a = self.fsrc(0)?;
                let e = self.shr_k(a, 23);
                let e = self.and_k(e, 0xff);
                let r = self.ibk(IBin::Sub, e, 126);
                let e0 = self.ick(ICmp::Eq, e, 0);
                let e255 = self.ick(ICmp::Eq, e, 255);
                let special = self.b.bor(e0, e255);
                let zero = self.k(0);
                Ok(Res::v(self.sel(special, zero, r)))
            }
            "v_frexp_mant_f32" => {
                let a = self.fsrc(0)?;
                let e = self.shr_k(a, 23);
                let e = self.and_k(e, 0xff);
                let m = self.and_k(a, 0x807f_ffff);
                let m = self.ibk(IBin::Or, m, 0x3f00_0000);
                let e0 = self.ick(ICmp::Eq, e, 0);
                let e255 = self.ick(ICmp::Eq, e, 255);
                let special = self.b.bor(e0, e255);
                let r = self.sel(special, a, m);
                self.fout(r)
            }
            "v_cubeid_f32" => self.cube("id"),
            "v_cubesc_f32" => self.cube("sc"),
            "v_cubetc_f32" => self.cube("tc"),
            "v_cubema_f32" => self.cube("ma"),
            // -- conversions
            "v_cvt_f32_u32" => {
                let a = self.v(0)?;
                let r = self.b.cvt(Cvt::UToF, a);
                self.fout(r)
            }
            "v_cvt_f32_i32" => {
                let a = self.v(0)?;
                let r = self.b.cvt(Cvt::IToF, a);
                self.fout(r)
            }
            "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte1" | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3" => {
                let n = u32::from(name.as_bytes()[name.len() - 1] - b'0');
                let a = self.v(0)?;
                let a = self.shr_k(a, 8 * n);
                let a = self.and_k(a, 0xff);
                let r = self.b.cvt(Cvt::UToF, a);
                self.fout(r)
            }
            "v_cvt_u32_f32" => {
                let a = self.fsrc(0)?;
                Ok(Res::v(self.f2u(a)))
            }
            "v_cvt_i32_f32" => {
                let a = self.fsrc(0)?;
                Ok(Res::v(self.f2i(a)))
            }
            "v_cvt_flr_i32_f32" | "v_cvt_rpi_i32_f32" => {
                let a = self.fsrc(0)?;
                let a = if name == "v_cvt_rpi_i32_f32" {
                    let h = self.fk(0.5);
                    self.fb(FBin::Add, a, h)
                } else {
                    a
                };
                let fl = self.b.fun(FUn::Floor, a);
                Ok(Res::v(self.f2i(fl)))
            }
            "v_cvt_f32_f16" => {
                let a = self.v(0)?;
                let r = self.b.unpack_half(a);
                self.fout(r)
            }
            "v_cvt_f16_f32" => {
                let a = self.fsrc(0)?;
                Ok(Res::v(self.b.pack_half(a)))
            }
            "v_cvt_pkrtz_f16_f32" => {
                let (a, b) = (self.fsrc(0)?, self.fsrc(1)?);
                let lo = self.f16_rtz(a);
                let hi = self.f16_rtz(b);
                let hi = self.shl_k(hi, 16);
                Ok(Res::v(self.ib(IBin::Or, lo, hi)))
            }
            "v_pack_b32_f16" => {
                let m = self.mods;
                if m.abs | m.neg != 0 {
                    return Err(self.un());
                }
                let (a, b) = (self.v(0)?, self.v(1)?);
                let a = if m.opsel & 1 != 0 {
                    self.shr_k(a, 16)
                } else {
                    a
                };
                let b = if m.opsel & 2 != 0 {
                    self.shr_k(b, 16)
                } else {
                    b
                };
                let lo = self.and_k(a, 0xffff);
                let hi = self.shl_k(b, 16);
                Ok(Res::v(self.ib(IBin::Or, lo, hi)))
            }
            "v_cvt_pk_u8_f32" => {
                let (a, sel, c) = (self.fsrc(0)?, self.v(1)?, self.v(2)?);
                let u = self.f2u(a);
                let u = self.ibk(IBin::UMin, u, 255);
                let sh = self.and_k(sel, 3);
                let sh = self.shl_k(sh, 3);
                let ins = self.ib(IBin::Shl, u, sh);
                let ff = self.k(0xff);
                let m = self.ib(IBin::Shl, ff, sh);
                let nm = self.b.iun(IUn::Not, m);
                let keep = self.ib(IBin::And, c, nm);
                Ok(Res::v(self.ib(IBin::Or, keep, ins)))
            }
            "v_fma_mix_f32" => {
                let (a, b, c) = (self.mix_src(0)?, self.mix_src(1)?, self.mix_src(2)?);
                let r = self.b.fma(a, b, c);
                self.fout(r)
            }
            "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                let (a, b, c) = (self.mix_src(0)?, self.mix_src(1)?, self.mix_src(2)?);
                let r = self.b.fma(a, b, c);
                let r = if self.mods.clamp { self.fclamp(r) } else { r };
                let h = self.b.pack_half(r);
                let old = self.old_dst()?;
                if name.ends_with("lo_f16") {
                    let keep = self.and_k(old, 0xffff_0000);
                    Ok(Res::v(self.ib(IBin::Or, keep, h)))
                } else {
                    let keep = self.and_k(old, 0xffff);
                    let h = self.shl_k(h, 16);
                    Ok(Res::v(self.ib(IBin::Or, keep, h)))
                }
            }
            _ => Err(self.un()),
        }
    }
}

/// Whether `exec` supports this instruction (what the corpus statistics ask).
pub fn supported(inst: &Inst) -> Result<(), Unsupported> {
    let ctx: Ctx<Dry> = Ctx {
        inst,
        srcs: inst
            .srcs
            .iter()
            .map(|s| s.iter().map(|_| Opd::V(())).collect())
            .collect(),
        imps: inst.imps.iter().map(|(r, _)| (*r, Opd::V(()))).collect(),
        dst_mask: vec![false; inst.dsts.len()],
        active: None,
        lane: Some(()),
    };
    exec(&mut Dry, &ctx).map(|_| ())
}
