//! Stage inputs and outputs of the hardware VS and PS (`docs/shaders.md`, Inputs and outputs;
//! chunk 8): what the stage data and the container signatures say about the entry registers,
//! the vertex attributes, the pixel inputs and the render targets, and the linking of a
//! VS+PS pair by semantic. The emitter (`spirv/emit/stage.rs`) turns a `StageIo` into the
//! entry point.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;

use super::container::{SigElement, Signature};
use super::gfx10;
use super::ir::{self, EntryRegs, IrError, PS_INPUT_DWORDS};
use super::native::{PsInput, Record, StageData, VsInput, VsOutput};
use super::resource::Translated;
use super::spirv::rect::RectInterface;
use super::spirv::{self, SpirvError};
use super::structure::{self, StructError};
use super::ParsedShader;

/// The component class of an input or output: what the Vulkan format is read as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    Float,
    Uint,
    Sint,
}

/// The class of a DXGI format (`D3D12_GRAPHICS_PIPELINE_STATE_DESC` render target formats):
/// the integer formats, float for the rest (UNORM, SNORM, FLOAT) and for the unknown.
pub fn dxgi_class(format: u32) -> Class {
    match format {
        3 | 7 | 12 | 17 | 25 | 30 | 36 | 42 | 50 | 57 | 62 => Class::Uint,
        4 | 8 | 14 | 18 | 32 | 38 | 43 | 52 | 59 | 64 => Class::Sint,
        _ => Class::Float,
    }
}

/// Whether a render target format is A8_UNORM, which the replay stores as an R8 image: its
/// exports keep only channel 3 (alpha), written to red.
pub fn a8_target(format: u32) -> bool {
    format == 65
}

fn sig_class(e: Option<&SigElement>) -> Class {
    match e.map(|e| e.component_type) {
        Some(1) => Class::Uint,
        Some(2) => Class::Sint,
        _ => Class::Float,
    }
}

/// The signature element a stage-data row names: its register is the row.
fn element(sig: Option<&Signature>, row: u8, mask: u8) -> Option<&SigElement> {
    let sig = sig?;
    let on_row = || sig.elements.iter().filter(|e| e.register == u32::from(row));
    on_row()
        .find(|e| e.mask & mask != 0)
        .or_else(|| on_row().next())
}

/// A semantic: upper-cased name and index.
type Semantic = (String, u32);

fn semantic(e: &SigElement) -> Semantic {
    (e.name.to_ascii_uppercase(), e.index)
}

/// A vertex attribute: the fetch routine's result, from a Vulkan vertex input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VsAttr {
    /// The input's index in the stage data.
    pub location: u32,
    pub first_vgpr: u16,
    /// Channels fetched (bit c: VGPR `first_vgpr + c`).
    pub mask: u8,
    pub class: Class,
    pub semantic: Option<(String, u32)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VsIo {
    pub user: Range<u16>,
    /// The user SGPRs the layout marks as tables or heaps (root types 3, 5, 6).
    pub tables: BTreeSet<u16>,
    pub attrs: Vec<VsAttr>,
    /// `exp param n` → location (an output nothing reads has none and is dropped).
    pub params: BTreeMap<u8, u32>,
    /// The stage data lists clip distance outputs (usage 7).
    pub clip: bool,
}

/// What a pixel-shader input VGPR holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PsVgpr {
    /// A barycentric coordinate (`pair` 0 is i, 1 is j).
    Bary {
        linear: bool,
        centroid: bool,
        sample: bool,
        pair: u8,
    },
    /// Fragment position x, y, z, w (0..3).
    Pos(u8),
    FrontFace,
    Ancillary,
    Coverage,
    /// Anything the translation does not provide (pull model, line stipple, fixed position).
    Other(&'static str),
}

/// A pixel input (`v_interp attr n`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PsAttr {
    /// `None`: no vertex output of that semantic, the input reads its default.
    pub location: Option<u32>,
    pub usage: u8,
    pub mask: u8,
    pub default: u8,
    pub flat: bool,
    pub linear: bool,
    pub fp16: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PsIo {
    pub user: Range<u16>,
    pub tables: BTreeSet<u16>,
    pub vgprs: BTreeMap<u16, PsVgpr>,
    pub attrs: BTreeMap<u8, PsAttr>,
    /// Class of each render target (`Float` past the end).
    pub targets: Vec<Class>,
    /// Targets of format A8_UNORM (stored as R8: the exported alpha lands in red, see `a8_target`).
    pub a8: Vec<bool>,
}

/// A used-resource row of a compute program: a descriptor at a byte offset of the table the
/// user SGPR `sgpr` holds, and what it is (`UsedResource::kind`: 0 SRV, 3 UAV).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsUsed {
    pub sgpr: u16,
    pub offset: u32,
    pub kind: u8,
}

/// A compute program's entry (`docs/shaders.md`, chunk 10c): the user SGPRs from s0, the
/// thread-group ids, the thread-id VGPRs and the workgroup size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsIo {
    pub user: Range<u16>,
    pub tables: BTreeSet<u16>,
    /// Thread-group size (`LocalSize`).
    pub group: [u32; 3],
    /// The SGPR of each enabled thread-group id and its axis (0 x, 1 y, 2 z).
    pub group_ids: Vec<(u16, u8)>,
    /// Thread-id VGPRs v0.. (1 to 3): `LocalInvocationId` x, y, z.
    pub thread_ids: u16,
    /// The descriptor rows the program uses: an image load or store finds its kind here.
    pub used: Vec<CsUsed>,
    /// LDS size in bytes.
    pub lds_bytes: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageIo {
    Vertex(VsIo),
    Pixel(PsIo),
    Compute(CsIo),
}

impl StageIo {
    pub fn user(&self) -> &Range<u16> {
        match self {
            Self::Vertex(v) => &v.user,
            Self::Pixel(p) => &p.user,
            Self::Compute(c) => &c.user,
        }
    }

    pub fn tables(&self) -> &BTreeSet<u16> {
        match self {
            Self::Vertex(v) => &v.tables,
            Self::Pixel(p) => &p.tables,
            Self::Compute(c) => &c.tables,
        }
    }

    /// What the descriptor at byte `offset` of the table in user SGPR `sgpr` is (0 SRV, 3 UAV,
    /// ...): only a compute program knows, from its used-resource rows.
    pub fn descriptor_kind(&self, sgpr: u16, offset: u32) -> Option<u8> {
        let Self::Compute(c) = self else {
            return None;
        };
        c.used
            .iter()
            .find(|u| u.sgpr == sgpr && u.offset == offset)
            .map(|u| u.kind)
    }
}

/// The user SGPRs the layout marks as tables (type 3) or heaps (types 5 and 6): each holds the
/// low 32 bits of an address the root block carries in full.
fn table_sgprs(rec: &Record) -> BTreeSet<u16> {
    rec.layout
        .iter()
        .filter(|l| matches!(l.root_type, 3 | 5 | 6) && l.user_sgprs > 0)
        .map(|l| u16::from(l.first_user_dword))
        .collect()
}

/// The locations a VS+PS pair shares.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Link {
    /// VS `exp param n` → location.
    pub params: BTreeMap<u8, u32>,
    /// PS `attr n` → location, `None` when no VS output has its semantic.
    pub attrs: BTreeMap<u8, Option<u32>>,
}

/// The semantic of each VS parameter export.
fn vs_params(outputs: &[VsOutput], sig: Option<&Signature>) -> Vec<(u8, Option<Semantic>)> {
    outputs
        .iter()
        .filter(|o| o.usage == 6)
        .map(|o| (o.param, element(sig, o.row, 0xf).map(semantic)))
        .collect()
}

/// The semantic of each PS input.
fn ps_semantics(inputs: &[PsInput], sig: Option<&Signature>) -> Vec<(u8, Option<Semantic>)> {
    inputs
        .iter()
        .map(|i| (i.attr, element(sig, i.row, i.mask).map(semantic)))
        .collect()
}

/// Links a VS and a PS by semantic name and index: the location of a matched pair is the PS
/// input's attribute number, which both sides use. A VS output nothing reads gets none; a PS
/// input no VS output feeds reads its default.
pub fn link(vs: &ParsedShader, ps: &ParsedShader) -> Link {
    let (Some(StageData::Vertex(v)), Some(StageData::Pixel(p))) =
        (vs.record().map(|r| &r.data), ps.record().map(|r| &r.data))
    else {
        return Link::default();
    };
    let outs = vs_params(&v.outputs, vs.output_signature.as_ref());
    let mut l = Link::default();
    for (attr, sem) in ps_semantics(&p.inputs, ps.input_signature.as_ref()) {
        let hit = sem.as_ref().and_then(|s| {
            outs.iter()
                .find(|(_, o)| o.as_ref() == Some(s))
                .map(|(n, _)| *n)
        });
        if let Some(n) = hit {
            l.params.insert(n, u32::from(attr));
        }
        l.attrs.insert(attr, hit.map(|_| u32::from(attr)));
    }
    l
}

fn vs_io(p: &ParsedShader, rec: &Record, link: Option<&Link>) -> Option<VsIo> {
    let StageData::Vertex(d) = &rec.data else {
        return None;
    };
    let attrs = d
        .inputs
        .iter()
        .enumerate()
        .map(|(n, i): (usize, &VsInput)| {
            let e = element(p.input_signature.as_ref(), i.row, i.mask);
            VsAttr {
                location: n as u32,
                first_vgpr: u16::from(i.first_vgpr),
                mask: i.mask,
                class: sig_class(e),
                semantic: e.map(semantic),
            }
        })
        .collect();
    let params = match link {
        Some(l) => l.params.clone(),
        None => d
            .outputs
            .iter()
            .filter(|o| o.usage == 6)
            .map(|o| (o.param, u32::from(o.param)))
            .collect(),
    };
    Some(VsIo {
        user: ir::user_range(rec),
        tables: table_sgprs(rec),
        attrs,
        params,
        clip: d.outputs.iter().any(|o| o.usage == 7),
    })
}

fn ps_io(rec: &Record, targets: &[u32], link: Option<&Link>) -> Option<PsIo> {
    let StageData::Pixel(d) = &rec.data else {
        return None;
    };
    let mut vgprs = BTreeMap::new();
    let mut next = 0u16;
    for (bit, n) in PS_INPUT_DWORDS.iter().enumerate() {
        if d.spi_ps_input_addr >> bit & 1 == 0 {
            continue;
        }
        for k in 0..*n {
            let v = match bit {
                0..=2 | 4..=6 if k < 2 => PsVgpr::Bary {
                    linear: bit >= 4,
                    centroid: bit % 4 == 2,
                    sample: bit % 4 == 0,
                    pair: k as u8,
                },
                3 => PsVgpr::Other("the pull model"),
                0..=6 => PsVgpr::Other("barycentric"),
                7 => PsVgpr::Other("line stipple"),
                8..=11 => PsVgpr::Pos((bit - 8) as u8),
                12 => PsVgpr::FrontFace,
                13 => PsVgpr::Ancillary,
                14 => PsVgpr::Coverage,
                _ => PsVgpr::Other("fixed position"),
            };
            vgprs.insert(next + k, v);
        }
        next += n;
    }
    let attrs = d
        .inputs
        .iter()
        .map(|i| {
            let location = match link {
                Some(l) => l.attrs.get(&i.attr).copied().flatten(),
                None => Some(u32::from(i.attr)),
            };
            (
                i.attr,
                PsAttr {
                    location,
                    usage: i.usage,
                    mask: i.mask,
                    default: i.default,
                    flat: i.flags & 1 != 0,
                    linear: i.flags & 2 != 0,
                    fp16: i.flags & 0x18 != 0,
                },
            )
        })
        .collect();
    Some(PsIo {
        user: ir::user_range(rec),
        tables: table_sgprs(rec),
        vgprs,
        attrs,
        targets: targets.iter().map(|&f| dxgi_class(f)).collect(),
        a8: targets.iter().map(|&f| a8_target(f)).collect(),
    })
}

fn cs_io(rec: &Record) -> Option<CsIo> {
    let StageData::Compute(d) = &rec.data else {
        return None;
    };
    let group_ids = (0..3u8)
        .filter(|&a| d.rsrc2 >> (7 + a) & 1 != 0)
        .zip(d.system_sgprs())
        .map(|(a, s)| (s, a))
        .collect();
    Some(CsIo {
        user: 0..d.user_sgprs(),
        tables: table_sgprs(rec),
        group: d.group,
        group_ids,
        thread_ids: d.thread_id_vgprs(),
        used: rec
            .used
            .iter()
            .filter(|u| u.root_type == 3 && matches!(u.kind, 0 | 3))
            .map(|u| CsUsed {
                sgpr: u16::from(u.first_user_dword),
                offset: u.table_offset.wrapping_mul(4),
                kind: u.kind,
            })
            .collect(),
        lds_bytes: rec.lds_bytes,
    })
}

/// The stage's I/O, or `None` for a stage other than vertex, pixel and compute. `link` is the
/// pair's linking; without it a stage numbers its own parameters (VS `param n`, PS `attr n`).
pub fn stage_io(p: &ParsedShader, targets: &[u32], link: Option<&Link>) -> Option<StageIo> {
    let rec = p.record()?;
    match &rec.data {
        StageData::Vertex(_) => vs_io(p, rec, link).map(StageIo::Vertex),
        StageData::Pixel(_) => ps_io(rec, targets, link).map(StageIo::Pixel),
        StageData::Compute(_) => cs_io(rec).map(StageIo::Compute),
        _ => None,
    }
}

/// Index of the 32_ABGR variant of an export patch (`native::PATCH_VARIANT_NAMES`).
const PATCH_32_ABGR: usize = 9;

/// The program's code, with the pixel shader's export patches in their raw 32-bit form: every
/// target is exported as four 32-bit channels and Vulkan converts to the target's format.
pub fn stage_code(p: &ParsedShader, rec: &Record) -> Vec<u8> {
    let mut code = p.native.code(rec).to_vec();
    if let StageData::Pixel(d) = &rec.data {
        for patch in &d.patches {
            let Some(v) = patch.variants.get(PATCH_32_ABGR) else {
                continue;
            };
            let at = 4 * patch.code_dword as usize;
            for (k, w) in v.iter().take(d.patch_dwords as usize).enumerate() {
                if let Some(dst) = code.get_mut(at + 4 * k..at + 4 * k + 4) {
                    dst.copy_from_slice(&w.to_le_bytes());
                }
            }
        }
    }
    code
}

/// Why a stage does not translate.
#[derive(Debug)]
pub enum StageError {
    /// Not a vertex or pixel program.
    NotAStage,
    Ir(IrError),
    Structure(StructError),
    Spirv(SpirvError),
}

impl fmt::Display for StageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAStage => f.write_str("not a vertex or pixel program"),
            Self::Ir(e) => write!(f, "IR error: {e}"),
            Self::Structure(e) => write!(f, "structuring: {e}"),
            Self::Spirv(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for StageError {}

/// The entry states of a compute program the translation does not provide, by name: the
/// thread-group info SGPR (RSRC2 bit 10), the scratch offset (bit 0), wave32.
pub fn refuse_entry(rec: &Record) -> Result<(), SpirvError> {
    let StageData::Compute(d) = &rec.data else {
        return Ok(());
    };
    let what = if d.rsrc2 >> 10 & 1 != 0 {
        "a compute entry with the thread-group info SGPR"
    } else if d.rsrc2 & 1 != 0 {
        "a compute entry with a scratch offset"
    } else if rec.wave32() {
        "a wave32 compute program"
    } else {
        return Ok(());
    };
    Err(SpirvError::Unsupported(spirv::Unsupported {
        chunk: 10,
        op: what.to_string(),
    }))
}

/// Refuses a program that needs more lanes in one subgroup than the host's `subgroup_size`
/// (`docs/shaders.md`, Wave64 on narrower subgroups): one whose wave is wider and whose
/// result depends on lanes outside the invocation's own subgroup.
pub fn refuse_wave(
    func: &ir::Func,
    st: &structure::Structured,
    subgroup_size: u32,
) -> Result<(), SpirvError> {
    let wave = if func.wave32 { 32 } else { 64 };
    if subgroup_size >= wave {
        return Ok(());
    }
    let Some(d) = structure::wave_dependence(func, st) else {
        return Ok(());
    };
    Err(SpirvError::Unsupported(spirv::Unsupported {
        chunk: 7,
        op: format!(
            "a wave{wave} program with {} (at +{:#x}) on subgroups of {subgroup_size}",
            d.what, d.at
        ),
    }))
}

/// Translates one vertex, pixel or compute program to its entry point (`Vertex`, `Fragment`,
/// `GLCompute`). `targets` are the DXGI formats of the render targets (pixel only; float where
/// missing). `subgroup_size` is the host's: see `refuse_wave`.
pub fn translate_stage(
    p: &ParsedShader,
    targets: &[u32],
    link: Option<&Link>,
    subgroup_size: u32,
) -> Result<Translated, StageError> {
    let rec = p.record().ok_or(StageError::NotAStage)?;
    let io = stage_io(p, targets, link).ok_or(StageError::NotAStage)?;
    refuse_entry(rec).map_err(StageError::Spirv)?;
    let code = stage_code(p, rec);
    let regs = EntryRegs::from_stage(rec);
    let func = ir::build(&code, gfx10::ENTRY, rec.wave32(), &regs).map_err(StageError::Ir)?;
    let st = structure::structure(&func).map_err(StageError::Structure)?;
    refuse_wave(&func, &st, subgroup_size).map_err(StageError::Spirv)?;
    spirv::translate_io(&func, &st, &io).map_err(StageError::Spirv)
}

/// Translates a VS+PS pair with the locations they share. The static samplers are one list
/// for the pair (set 1): the VS's keep their bindings, the PS's are deduplicated against them
/// and renumbered after them; both results carry the whole list.
pub fn translate_pair(
    vs: &ParsedShader,
    ps: &ParsedShader,
    targets: &[u32],
    subgroup_size: u32,
) -> Result<(Translated, Translated), StageError> {
    let l = link(vs, ps);
    let mut v = translate_stage(vs, targets, Some(&l), subgroup_size)?;
    let mut p = translate_stage(ps, targets, Some(&l), subgroup_size)?;
    let mut shared = v.samplers.clone();
    let renumber: Vec<u32> = p
        .samplers
        .iter()
        .map(|d| {
            let k = shared.iter().position(|s| s == d).unwrap_or_else(|| {
                shared.push(*d);
                shared.len() - 1
            });
            k as u32
        })
        .collect();
    if renumber.iter().enumerate().any(|(i, &k)| k as usize != i) {
        renumber_static_samplers(&mut p.words, &renumber);
    }
    v.samplers = shared.clone();
    p.samplers = shared;
    Ok((v, p))
}

/// The outputs the geometry shader of a rectangle list passes on (`spirv::rect`): the locations
/// the vertex shader writes (as linked to `ps`, or its own parameter numbers without one), flat
/// where the pixel shader reads them flat, and whether it writes clip distances. `None` for a
/// program that is not a vertex shader.
pub fn rect_interface(
    vs: &ParsedShader,
    ps: Option<&ParsedShader>,
    targets: &[u32],
) -> Option<RectInterface> {
    let link = ps.map(|ps| link(vs, ps));
    let StageIo::Vertex(v) = stage_io(vs, &[], link.as_ref())? else {
        return None;
    };
    let flat: BTreeSet<u32> = match ps.and_then(|ps| stage_io(ps, targets, link.as_ref())) {
        Some(StageIo::Pixel(p)) => p
            .attrs
            .values()
            .filter(|a| a.flat)
            .filter_map(|a| a.location)
            .collect(),
        _ => BTreeSet::new(),
    };
    let locations: BTreeSet<u32> = v.params.values().copied().collect();
    Some(RectInterface {
        locations: locations
            .into_iter()
            .map(|l| (l, flat.contains(&l)))
            .collect(),
        clip: v.clip,
    })
}

/// Rewrites the `Binding` of every variable in descriptor set 1 (the static samplers) from `k`
/// to `map[k]`.
fn renumber_static_samplers(words: &mut [u32], map: &[u32]) {
    const OP_DECORATE: u32 = 71;
    const BINDING: u32 = 33;
    const DESCRIPTOR_SET: u32 = 34;
    let instructions = |words: &[u32]| {
        let mut at = 5;
        let mut out = Vec::new();
        while at < words.len() {
            let n = (words[at] >> 16) as usize;
            if n == 0 {
                break;
            }
            out.push((at, words[at] & 0xffff, n));
            at += n;
        }
        out
    };
    let list = instructions(words);
    let in_set1: Vec<u32> = list
        .iter()
        .filter(|&&(at, op, n)| {
            op == OP_DECORATE && n >= 4 && words[at + 2] == DESCRIPTOR_SET && words[at + 3] == 1
        })
        .map(|&(at, _, _)| words[at + 1])
        .collect();
    for (at, op, n) in list {
        if op == OP_DECORATE
            && n >= 4
            && words[at + 2] == BINDING
            && in_set1.contains(&words[at + 1])
        {
            if let Some(&k) = map.get(words[at + 3] as usize) {
                words[at + 3] = k;
            }
        }
    }
}
