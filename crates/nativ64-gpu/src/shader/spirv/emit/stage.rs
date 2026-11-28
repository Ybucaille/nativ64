//! The real entry points of the vertex and pixel stages (chunk 8, `docs/shaders.md`, Inputs
//! and outputs): the entry registers from the root block and the built-ins, the vertex fetch
//! call as `Input` variables, `v_interp` raised to `Input` variables, exports written to
//! `Output` variables, and the pixel discard as a demote.

use std::collections::{BTreeMap, HashMap};

use spirv::{BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, Op, StorageClass};

use super::resource::{self, Res};
use super::{bad, Gen, Opd, SpirvError, Tr, R};
use crate::shader::gfx10::{Enc, Operand, Slot};
use crate::shader::io::{Class, PsAttr, PsVgpr, StageIo, VsAttr};
use crate::shader::ir::{Reg, Src, ValDef};
use crate::shader::spirv::alu::{Backend, Ctx, Unsupported};

/// How a pixel input is interpolated: `(flat, no perspective, centroid, sample)`.
type Mode = (bool, bool, bool, bool);

/// The variables of the entry point, created on first use.
pub(super) struct Stage<'a> {
    io: &'a StageIo,
    /// Every `Input`, `Output` and push constant variable (the entry point's interface).
    vars: Vec<u32>,
    builtins: HashMap<u32, u32>,
    attrs: HashMap<u32, u32>,
    outputs: HashMap<(u32, Class), u32>,
    position: Option<u32>,
    clip: Option<u32>,
    depth: Option<u32>,
    stencil: Option<u32>,
    mask_out: Option<u32>,
    interps: BTreeMap<u8, (Mode, u32)>,
    /// Memory and resources (chunk 9).
    pub(super) res: Res,
}

impl<'a> Stage<'a> {
    pub(super) fn new(io: &'a StageIo, analysis: resource::Analysis) -> Self {
        Self {
            io,
            vars: Vec::new(),
            builtins: HashMap::new(),
            attrs: HashMap::new(),
            outputs: HashMap::new(),
            position: None,
            clip: None,
            depth: None,
            stencil: None,
            mask_out: None,
            interps: BTreeMap::new(),
            res: Res::new(analysis),
        }
    }

    pub(super) fn interface(&self) -> Vec<u32> {
        self.vars.clone()
    }

    pub(super) fn model(&self) -> ExecutionModel {
        match self.io {
            StageIo::Vertex(_) => ExecutionModel::Vertex,
            StageIo::Pixel(_) => ExecutionModel::Fragment,
            StageIo::Compute(_) => ExecutionModel::GLCompute,
        }
    }

    pub(super) fn exec_modes(&self, m: &mut super::Module, main: u32) {
        if let StageIo::Compute(c) = self.io {
            m.exec_mode(main, ExecutionMode::LocalSize, &c.group);
        }
        if matches!(self.io, StageIo::Pixel(_)) {
            m.exec_mode(main, ExecutionMode::OriginUpperLeft, &[]);
            if self.depth.is_some() {
                m.exec_mode(main, ExecutionMode::DepthReplacing, &[]);
            }
            if self.stencil.is_some() {
                m.exec_mode(main, ExecutionMode::StencilRefReplacingEXT, &[]);
            }
        }
    }
}

fn unsupported<T>(op: &str) -> R<T> {
    Err(SpirvError::Unsupported(Unsupported {
        chunk: 8,
        op: op.to_string(),
    }))
}

/// The attribute and channel a VINTRP / VOP3 interpolation names.
fn attr_of(i: &crate::shader::gfx10::Instr) -> Option<(u8, u8)> {
    if i.enc == Enc::Vintrp {
        return Some((i.intrp.attr, i.intrp.chan));
    }
    let k = i.info.slots.iter().position(|s| matches!(s, Slot::Attr))?;
    match i.operands.get(k)? {
        Operand::Imm(v) => Some((*v as u8 & 0x3f, (*v >> 6) as u8 & 3)),
        _ => None,
    }
}

/// The VGPR holding the barycentric coordinate of an interpolation.
fn bary_of(i: &crate::shader::gfx10::Instr) -> Option<u16> {
    if i.enc == Enc::Vintrp {
        return Some(u16::from(i.intrp.vsrc));
    }
    let k = i
        .info
        .slots
        .iter()
        .position(|s| matches!(s, Slot::Src1(_)))?;
    match i.operands.get(k)? {
        Operand::Reg { code, .. } if *code >= 256 => Some(code - 256),
        _ => None,
    }
}

impl<'a> Tr<'a> {
    pub(super) fn st(&mut self) -> &mut Stage<'a> {
        self.stage.as_mut().expect("a stage entry")
    }

    pub(super) fn io(&self) -> &'a StageIo {
        self.stage.as_ref().expect("a stage entry").io
    }

    pub(super) fn t_i32(&mut self) -> u32 {
        self.g.m.global(Op::TypeInt, None, &[32, 1])
    }

    fn scalar(&mut self, c: Class) -> u32 {
        match c {
            Class::Float => self.g.t_f32(),
            Class::Uint => self.g.t_u32(),
            Class::Sint => self.t_i32(),
        }
    }

    pub(super) fn vec4(&mut self, c: Class) -> u32 {
        let t = self.scalar(c);
        self.g.m.global(Op::TypeVector, None, &[t, 4])
    }

    pub(super) fn global(&mut self, class: StorageClass, ty: u32) -> u32 {
        let p = self.g.t_ptr(class, ty);
        let v = self.g.m.global_var(p, class, None);
        self.st().vars.push(v);
        v
    }

    pub(super) fn bits_of(&mut self, o: Opd<Gen>) -> u32 {
        match o {
            Opd::V(v) => v,
            Opd::B(b) => {
                let (one, zero) = (self.g.k(1), self.g.k(0));
                self.g.select(b, one, zero)
            }
        }
    }

    /// The bits of a `class` value, as the register's `u32`.
    fn as_reg_bits(&mut self, c: Class, v: u32) -> u32 {
        match c {
            Class::Uint => v,
            _ => self.g.as_bits(v),
        }
    }

    /// A register's bits as a `class` value.
    fn typed_from_bits(&mut self, c: Class, bits: u32) -> u32 {
        match c {
            Class::Uint => bits,
            Class::Float => self.g.as_f32(bits),
            Class::Sint => {
                let t = self.t_i32();
                self.g.op(Op::Bitcast, t, &[bits])
            }
        }
    }

    fn builtin_in(&mut self, b: BuiltIn, ty: u32) -> u32 {
        if let Some(&v) = self.st().builtins.get(&(b as u32)) {
            return v;
        }
        let v = self.global(StorageClass::Input, ty);
        self.g.m.decorate(v, Decoration::BuiltIn, &[b as u32]);
        self.st().builtins.insert(b as u32, v);
        v
    }

    fn load_builtin(&mut self, b: BuiltIn, ty: u32) -> u32 {
        let v = self.builtin_in(b, ty);
        self.g.load(ty, v)
    }

    /// The value of an entry register, as a `Src` the program reads.
    /// The console lane index of this invocation: `LocalInvocationIndex` modulo the wave size
    /// in compute (0 in the probe). A vertex or pixel invocation cannot tell it: `None`.
    pub(super) fn lane_index(&mut self, wave32: bool) -> R<Option<u32>> {
        let tu = self.g.t_u32();
        match self.stage.as_ref().map(|s| s.io) {
            None => Ok(Some(self.g.k(0))),
            Some(StageIo::Compute(_)) => {
                let i = self.load_builtin(BuiltIn::LocalInvocationIndex, tu);
                let mask = self.g.k(if wave32 { 31 } else { 63 });
                Ok(Some(self.g.op(Op::BitwiseAnd, tu, &[i, mask])))
            }
            Some(_) => Ok(None),
        }
    }

    /// `SubgroupLocalInvocationId`: the host lane, whose low two bits are the lane of the quad.
    fn subgroup_id(&mut self) -> u32 {
        let tu = self.g.t_u32();
        let var = match self.sg_id {
            Some(v) => v,
            None => {
                let p = self.g.t_ptr(StorageClass::Input, tu);
                let v = self.g.m.global_var(p, StorageClass::Input, None);
                self.g.m.decorate(
                    v,
                    Decoration::BuiltIn,
                    &[BuiltIn::SubgroupLocalInvocationId as u32],
                );
                if matches!(self.stage.as_ref().map(|s| s.io), Some(StageIo::Pixel(_))) {
                    self.g.m.decorate(v, Decoration::Flat, &[]);
                }
                self.extra_iface.push(v);
                self.sg_id = Some(v);
                v
            }
        };
        self.g.load(tu, var)
    }

    /// Lane `k` of every quad reads the value of lane `sel[k]` of its quad.
    pub(super) fn quad_perm(&mut self, a: u32, sel: [u8; 4]) -> u32 {
        if sel.iter().all(|&s| s == sel[0]) {
            return self.g.quad_broadcast(a, u32::from(sel[0]));
        }
        for (dir, swap) in [[1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]
            .iter()
            .enumerate()
        {
            if sel == *swap {
                return self.g.quad_swap(a, dir as u32);
            }
        }
        let tu = self.g.t_u32();
        let tb = self.g.t_bool();
        let id = self.subgroup_id();
        let three = self.g.k(3);
        let q = self.g.op(Op::BitwiseAnd, tu, &[id, three]);
        let mut cache: HashMap<u8, u32> = HashMap::new();
        let mut pick = |t: &mut Self, j: u8| -> u32 {
            *cache
                .entry(j)
                .or_insert_with(|| t.g.quad_broadcast(a, u32::from(j)))
        };
        let mut r = pick(self, sel[3]);
        for k in (0..3).rev() {
            let b = pick(self, sel[k]);
            let kk = self.g.k(k as u32);
            let is = self.g.op(Op::IEqual, tb, &[q, kk]);
            r = self.g.select(is, b, r);
        }
        r
    }

    pub(super) fn begin_stage(&mut self) -> R<()> {
        let io = self.io();
        let tu = self.g.t_u32();
        self.check_barycentrics()?;
        // the root block's address is read once, in the entry block, where it dominates
        // every use
        let root = self.root_pointer();
        self.st().res.root = Some(root);
        let args = self.f.args.clone();
        for (reg, val) in args {
            let v = match reg {
                Reg::ExecLo | Reg::ExecHi => Opd::B(self.g.kb(true)),
                Reg::VccLo | Reg::VccHi | Reg::M0 | Reg::Scc | Reg::T(_) => Opd::V(self.g.k(0)),
                Reg::S(n) if io.user().contains(&n) => {
                    if n >= 32 {
                        return unsupported(&format!("user data s{n}"));
                    }
                    Opd::V(self.user_dword(root, u32::from(n)))
                }
                Reg::S(n) if matches!(io, StageIo::Compute(_)) => {
                    let StageIo::Compute(c) = io else {
                        unreachable!()
                    };
                    let Some(&(_, axis)) = c.group_ids.iter().find(|(s, _)| *s == n) else {
                        return unsupported(&format!("entry {reg}"));
                    };
                    let v3 = self.g.m.global(Op::TypeVector, None, &[tu, 3]);
                    let id = self.load_builtin(BuiltIn::WorkgroupId, v3);
                    Opd::V(self.g.op(Op::CompositeExtract, tu, &[id, u32::from(axis)]))
                }
                Reg::S(n) => {
                    // below the user SGPRs: the VS fetch routine's address (s4, s5); above:
                    // the PS primitive mask and ordered-section id, only read for m0
                    let ok = match io {
                        StageIo::Vertex(_) => n == 4 || n == 5,
                        StageIo::Pixel(p) => n >= p.user.end && n < p.user.end + 2,
                        StageIo::Compute(_) => false,
                    };
                    if !ok {
                        return unsupported(&format!("entry {reg}"));
                    }
                    Opd::V(self.g.k(0))
                }
                Reg::V(n) => Opd::V(self.entry_vgpr(n)?),
                Reg::Null => return bad("a null entry register"),
            };
            self.write(reg, val, v)?;
        }
        Ok(())
    }

    /// A pointer to the root block, from the push constant that holds its address.
    fn root_pointer(&mut self) -> u32 {
        self.g.m.capability(Capability::Int64);
        let t64 = self.g.m.global(Op::TypeInt, None, &[64, 0]);
        let pc = self.g.m.global(Op::TypeStruct, None, &[t64]);
        self.g.m.decorate(pc, Decoration::Block, &[]);
        self.g.m.member_decorate(pc, 0, Decoration::Offset, &[0]);
        let var = self.global(StorageClass::PushConstant, pc);
        let tu = self.g.t_u32();
        let n = self.g.k(32);
        let arr = self.g.m.global(Op::TypeArray, None, &[tu, n]);
        self.g.m.decorate(arr, Decoration::ArrayStride, &[4]);
        let arr64 = self.g.m.global(Op::TypeArray, None, &[t64, n]);
        self.g.m.decorate(arr64, Decoration::ArrayStride, &[8]);
        // user_data[32], table_base[32], view_table
        let root = self.g.m.global(Op::TypeStruct, None, &[arr, arr64, t64]);
        self.g.m.decorate(root, Decoration::Block, &[]);
        self.g.m.member_decorate(root, 0, Decoration::Offset, &[0]);
        self.g
            .m
            .member_decorate(root, 1, Decoration::Offset, &[128]);
        self.g
            .m
            .member_decorate(root, 2, Decoration::Offset, &[384]);
        self.g.m.name(root, "RootBlock");
        let pp = self.g.t_ptr(StorageClass::PushConstant, t64);
        let zero = self.g.k(0);
        let a = self.g.op(Op::AccessChain, pp, &[var, zero]);
        let addr = self.g.load(t64, a);
        let pr = self.g.t_ptr(StorageClass::PhysicalStorageBuffer, root);
        self.g.op(Op::ConvertUToPtr, pr, &[addr])
    }

    /// `user_data[n]` of the root block.
    pub(super) fn user_dword(&mut self, root: u32, n: u32) -> u32 {
        let tu = self.g.t_u32();
        let p = self.g.t_ptr(StorageClass::PhysicalStorageBuffer, tu);
        let (zero, k) = (self.g.k(0), self.g.k(n));
        let a = self.g.op(Op::AccessChain, p, &[root, zero, k]);
        // MemoryAccess Aligned, 4
        self.g.op(Op::Load, tu, &[a, 2, 4])
    }

    fn entry_vgpr(&mut self, n: u16) -> R<u32> {
        let tu = self.g.t_u32();
        match self.io() {
            StageIo::Vertex(_) => {
                self.g.m.capability(Capability::DrawParameters);
                let (id, base) = match n {
                    0 => (BuiltIn::VertexIndex, BuiltIn::BaseVertex),
                    3 => (BuiltIn::InstanceIndex, BuiltIn::BaseInstance),
                    // the primitive id is not a vertex-stage value
                    2 => return Ok(self.g.k(0)),
                    _ => return unsupported(&format!("entry v{n}")),
                };
                let (a, b) = (self.load_builtin(id, tu), self.load_builtin(base, tu));
                Ok(self.g.op(Op::ISub, tu, &[a, b]))
            }
            StageIo::Compute(c) => {
                if n >= c.thread_ids {
                    return unsupported(&format!("entry v{n}"));
                }
                let v3 = self.g.m.global(Op::TypeVector, None, &[tu, 3]);
                let id = self.load_builtin(BuiltIn::LocalInvocationId, v3);
                Ok(self.g.op(Op::CompositeExtract, tu, &[id, u32::from(n)]))
            }
            StageIo::Pixel(p) => match p.vgprs.get(&n).copied() {
                Some(PsVgpr::Bary { .. }) => Ok(self.g.k(0)),
                Some(PsVgpr::Pos(c)) => {
                    let tf = self.g.t_f32();
                    let v4 = self.vec4(Class::Float);
                    let pos = self.load_builtin(BuiltIn::FragCoord, v4);
                    let x = self.g.op(Op::CompositeExtract, tf, &[pos, u32::from(c)]);
                    Ok(self.g.as_bits(x))
                }
                Some(PsVgpr::FrontFace) => {
                    // nonzero when front facing
                    let tb = self.g.t_bool();
                    let ff = self.load_builtin(BuiltIn::FrontFacing, tb);
                    let (all, zero) = (self.g.k(u32::MAX), self.g.k(0));
                    Ok(self.g.select(ff, all, zero))
                }
                Some(PsVgpr::Ancillary) => {
                    // the sample id sits in bits 8-11
                    self.g.m.capability(Capability::SampleRateShading);
                    let id = self.load_builtin(BuiltIn::SampleId, tu);
                    let eight = self.g.k(8);
                    Ok(self.g.op(Op::ShiftLeftLogical, tu, &[id, eight]))
                }
                Some(PsVgpr::Coverage) => {
                    let one = self.g.k(1);
                    let arr = self.g.m.global(Op::TypeArray, None, &[tu, one]);
                    let ptr = self.builtin_in(BuiltIn::SampleMask, arr);
                    let pu = self.g.t_ptr(StorageClass::Input, tu);
                    let zero = self.g.k(0);
                    let a = self.g.op(Op::AccessChain, pu, &[ptr, zero]);
                    Ok(self.g.load(tu, a))
                }
                Some(PsVgpr::Other(what)) => unsupported(&format!("entry v{n} ({what})")),
                None => unsupported(&format!("entry v{n}")),
            },
        }
    }

    /// The VGPR whose entry value `v` is: the entry value itself, or a phi that only merges it.
    fn entry_root(&self, v: crate::shader::ir::Val) -> Option<u16> {
        let mut seen = vec![v];
        let mut at = 0;
        let mut root = None;
        while at < seen.len() {
            let info = &self.f.vals[seen[at].0 as usize];
            at += 1;
            let Reg::V(n) = info.reg else {
                return None;
            };
            match info.def {
                ValDef::Arg => {
                    if root.is_some_and(|r| r != n) {
                        return None;
                    }
                    root = Some(n);
                }
                ValDef::Phi { block, index } => {
                    for (_, s) in &self.f.blocks[block].phis[index].args {
                        match s {
                            Src::Val(u) if !seen.contains(u) => seen.push(*u),
                            Src::Val(_) => {}
                            _ => return None,
                        }
                    }
                }
                ValDef::Inst { .. } => return None,
            }
        }
        root
    }

    /// A barycentric VGPR read other than by `v_interp` is an error (`docs/shaders.md`).
    fn check_barycentrics(&self) -> R<()> {
        let StageIo::Pixel(p) = self.io() else {
            return Ok(());
        };
        let bary = |s: &Src| match s {
            Src::Val(v) => self
                .entry_root(*v)
                .is_some_and(|n| matches!(p.vgprs.get(&n), Some(PsVgpr::Bary { .. }))),
            _ => false,
        };
        let err = || unsupported("barycentric coordinates used outside v_interp");
        for blk in &self.f.blocks {
            for i in &blk.insts {
                if i.instr.name().starts_with("v_interp") {
                    continue;
                }
                // the implicit VGPR reads are the old values of the written registers
                let imps = i.imps.iter().filter(|(r, _)| !matches!(r, Reg::V(_)));
                let srcs = i.srcs.iter().flatten().chain(imps.map(|(_, s)| s));
                if srcs.into_iter().any(bary) {
                    return err();
                }
            }
        }
        Ok(())
    }

    /// The branch of a pixel discard no longer leaves: the lane is demoted and runs on.
    pub(super) fn demote_cond(&mut self, b: usize) -> Option<u32> {
        self.stage.as_ref()?;
        let d = self.s.demotes.iter().find(|d| d.branch == b)?;
        let crate::shader::ir::Term::Branch { taken, .. } = self.f.blocks[b].term else {
            return None;
        };
        Some(self.g.kb(d.exit != taken))
    }

    /// The instructions this stage translates itself: `None` leaves them to the ALU table.
    pub(super) fn stage_exec(
        &mut self,
        c: &Ctx<Gen>,
        b: usize,
        k: usize,
    ) -> R<Option<Vec<Opd<Gen>>>> {
        if self.stage.is_none() {
            return Ok(None);
        }
        if self.s.demotes.iter().any(|d| d.block == b && d.inst == k) {
            self.demote(c)?;
        }
        let i = &c.inst.instr;
        let name = i.name();
        if name == "s_barrier" {
            return self.barrier().map(|()| Some(Vec::new()));
        }
        if name == "s_swappc_b64" {
            return self.fetch(c).map(Some);
        }
        if i.enc == Enc::Exp {
            self.export(c)?;
            return Ok(Some(Vec::new()));
        }
        if name.starts_with("v_interp") {
            return self.interp(c).map(|v| Some(vec![v]));
        }
        self.resource_exec(c, b, k)
    }

    /// `s_andn2 live, live, exec`: the lanes in EXEC are killed. They become helper lanes.
    fn demote(&mut self, c: &Ctx<Gen>) -> R<()> {
        let Some(&e) = c.srcs.get(1).and_then(|s| s.first()) else {
            return bad("a discard without its EXEC operand");
        };
        let tb = self.g.t_bool();
        let kill = match e {
            Opd::B(b) => b,
            Opd::V(v) => {
                let z = self.g.k(0);
                self.g.op(Op::INotEqual, tb, &[v, z])
            }
        };
        self.g.m.capability(Capability::DemoteToHelperInvocation);
        self.g.m.extension("SPV_EXT_demote_to_helper_invocation");
        self.guarded(Some(kill), |t| {
            t.g.op0(Op::DemoteToHelperInvocation, &[]);
            Ok(())
        })
    }

    /// Runs `body` in the invocations where `cond` holds (all of them for `None`).
    pub(super) fn guarded(
        &mut self,
        cond: Option<u32>,
        body: impl FnOnce(&mut Self) -> R<()>,
    ) -> R<()> {
        let Some(c) = cond else {
            return body(self);
        };
        let (then, merge) = (self.g.m.id(), self.g.m.id());
        self.g.op0(Op::SelectionMerge, &[merge, 0]);
        self.g.term(Op::BranchConditional, &[c, then, merge]);
        self.g.label(then);
        body(self)?;
        self.g.branch(merge);
        self.g.label(merge);
        Ok(())
    }

    // -----------------------------------------------------------------------------------
    // Vertex input

    fn attr_var(&mut self, a: &VsAttr) -> u32 {
        if let Some(&v) = self.st().attrs.get(&a.location) {
            return v;
        }
        let ty = self.vec4(a.class);
        let v = self.global(StorageClass::Input, ty);
        self.g.m.decorate(v, Decoration::Location, &[a.location]);
        self.st().attrs.insert(a.location, v);
        v
    }

    /// The fetch call: the attribute VGPRs come from the vertex inputs, the return address
    /// registers are written with 0.
    fn fetch(&mut self, c: &Ctx<Gen>) -> R<Vec<Opd<Gen>>> {
        let StageIo::Vertex(vs) = self.io() else {
            return unsupported("s_swappc_b64");
        };
        let mut out = Vec::new();
        for (reg, _) in &c.inst.dsts {
            let Reg::V(n) = *reg else {
                out.push(Opd::V(self.g.k(0)));
                continue;
            };
            let hit = vs.attrs.iter().find_map(|a| {
                (0..4u16)
                    .find(|ch| a.mask >> ch & 1 != 0 && a.first_vgpr + ch == n)
                    .map(|ch| (a, ch))
            });
            let Some((a, ch)) = hit else {
                return bad("a fetched VGPR no vertex input names");
            };
            let var = self.attr_var(a);
            let v4 = self.vec4(a.class);
            let all = self.g.load(v4, var);
            let t = self.scalar(a.class);
            let x = self.g.op(Op::CompositeExtract, t, &[all, u32::from(ch)]);
            let bits = self.as_reg_bits(a.class, x);
            out.push(Opd::V(bits));
        }
        Ok(out)
    }

    // -----------------------------------------------------------------------------------
    // Pixel input

    /// The value of channel `chan` of a pixel input read in `mode`.
    fn attr_value(&mut self, attr: u8, a: &PsAttr, mode: Mode, chan: u8) -> R<u32> {
        let Some(loc) = a.location else {
            // no vertex output feeds it: the stage data's default (0,0,0,0 / 0,0,0,1 /
            // 1,1,1,0 / 1,1,1,1)
            let one = match a.default & 3 {
                0 => false,
                1 => chan == 3,
                2 => chan != 3,
                _ => true,
            };
            return Ok(self.g.k(if one { 0x3f80_0000 } else { 0 }));
        };
        let var = match self.st().interps.get(&attr) {
            Some(&(m, v)) if m == mode => v,
            Some(_) => return unsupported("an attribute interpolated in two ways"),
            None => {
                let ty = self.vec4(Class::Float);
                let v = self.global(StorageClass::Input, ty);
                self.g.m.decorate(v, Decoration::Location, &[loc]);
                let (flat, linear, centroid, sample) = mode;
                if flat {
                    self.g.m.decorate(v, Decoration::Flat, &[]);
                } else {
                    if linear {
                        self.g.m.decorate(v, Decoration::NoPerspective, &[]);
                    }
                    if centroid {
                        self.g.m.decorate(v, Decoration::Centroid, &[]);
                    }
                    if sample {
                        self.g.m.capability(Capability::SampleRateShading);
                        self.g.m.decorate(v, Decoration::Sample, &[]);
                    }
                }
                self.st().interps.insert(attr, (mode, v));
                v
            }
        };
        let (tf, v4) = (self.g.t_f32(), self.vec4(Class::Float));
        let all = self.g.load(v4, var);
        let x = self.g.op(Op::CompositeExtract, tf, &[all, u32::from(chan)]);
        Ok(self.g.as_bits(x))
    }

    /// `v_interp_p1` + `v_interp_p2` of one attribute channel become the channel of an `Input`
    /// variable (p1 holds the value, p2 passes it on); `v_interp_mov p0` reads a flat one.
    fn interp(&mut self, c: &Ctx<Gen>) -> R<Opd<Gen>> {
        let i = &c.inst.instr;
        let name = i.name();
        let StageIo::Pixel(ps) = self.io() else {
            return unsupported(name);
        };
        if name.contains("f16") {
            return unsupported(name);
        }
        let dst = c.inst.dsts.first().map(|d| d.0);
        if name == "v_interp_p2_f32" {
            let old = c.imps.iter().find(|(r, _)| Some(*r) == dst);
            return match old {
                Some(&(_, o)) => Ok(o),
                None => bad("v_interp_p2 without the old value of its destination"),
            };
        }
        let Some((attr, chan)) = attr_of(i) else {
            return bad("an interpolation without its attribute");
        };
        let Some(a) = ps.attrs.get(&attr) else {
            return unsupported("interpolation of an attribute the stage data does not list");
        };
        if a.usage != 6 {
            return unsupported("a pixel input that is not a generic attribute");
        }
        if a.fp16 {
            return unsupported("a half-precision pixel input");
        }
        let mode = if name == "v_interp_mov_f32" {
            if i.intrp.vsrc != 2 {
                return unsupported("v_interp_mov of p10 / p20");
            }
            (true, false, false, false)
        } else {
            let from_entry = bary_of(i).and_then(|n| {
                let arg = c
                    .inst
                    .srcs
                    .iter()
                    .flatten()
                    .any(|s| matches!(s, Src::Val(v) if self.entry_root(*v) == Some(n)));
                arg.then(|| ps.vgprs.get(&n).copied()).flatten()
            });
            match from_entry {
                Some(PsVgpr::Bary {
                    linear,
                    centroid,
                    sample,
                    ..
                }) => (a.flat, linear || a.linear, centroid, sample),
                _ => return unsupported("interpolation with computed barycentrics"),
            }
        };
        let bits = self.attr_value(attr, a, mode, chan)?;
        Ok(Opd::V(bits))
    }

    // -----------------------------------------------------------------------------------
    // Exports

    /// The pointer to component `ch` of an output variable and its scalar type.
    fn out_chan(&mut self, var: u32, scalar: u32, ch: u32) -> u32 {
        let p = self.g.t_ptr(StorageClass::Output, scalar);
        let k = self.g.k(ch);
        self.g.op(Op::AccessChain, p, &[var, k])
    }

    fn out_var(&mut self, loc: u32, class: Class) -> u32 {
        if let Some(&v) = self.st().outputs.get(&(loc, class)) {
            return v;
        }
        let ty = self.vec4(class);
        let v = self.global(StorageClass::Output, ty);
        self.g.m.decorate(v, Decoration::Location, &[loc]);
        self.st().outputs.insert((loc, class), v);
        v
    }

    fn export(&mut self, c: &Ctx<Gen>) -> R<()> {
        let e = c.inst.instr.exp;
        if e.compr {
            return unsupported("compressed export");
        }
        // the enabled channels, in order, are the instruction's sources
        let mut chans: Vec<(u32, u32)> = Vec::new();
        let mut srcs = c.srcs.iter();
        for ch in 0..4u32 {
            if e.en >> ch & 1 != 0 {
                let Some(&o) = srcs.next().and_then(|s| s.first()) else {
                    return bad("an export source is missing");
                };
                chans.push((ch, self.bits_of(o)));
            }
        }
        let io = self.io();
        match (e.target, io) {
            (9, _) => Ok(()),
            (12, StageIo::Vertex(_)) => {
                let v4 = self.vec4(Class::Float);
                let var = match self.st().position {
                    Some(v) => v,
                    None => {
                        let v = self.global(StorageClass::Output, v4);
                        self.g
                            .m
                            .decorate(v, Decoration::BuiltIn, &[BuiltIn::Position as u32]);
                        self.st().position = Some(v);
                        v
                    }
                };
                self.guarded(c.active, |t| {
                    let tf = t.g.t_f32();
                    for &(ch, bits) in &chans {
                        let p = t.out_chan(var, tf, ch);
                        let x = t.g.as_f32(bits);
                        t.g.store(p, x);
                    }
                    Ok(())
                })
            }
            (13 | 14, StageIo::Vertex(vs)) if vs.clip => {
                let base = 4 * (e.target as u32 - 13);
                let tf = self.g.t_f32();
                let n = self.g.k(8);
                let arr = self.g.m.global(Op::TypeArray, None, &[tf, n]);
                let var = match self.st().clip {
                    Some(v) => v,
                    None => {
                        self.g.m.capability(Capability::ClipDistance);
                        let v = self.global(StorageClass::Output, arr);
                        self.g
                            .m
                            .decorate(v, Decoration::BuiltIn, &[BuiltIn::ClipDistance as u32]);
                        self.st().clip = Some(v);
                        v
                    }
                };
                self.guarded(c.active, |t| {
                    for &(ch, bits) in &chans {
                        let p = t.out_chan(var, tf, base + ch);
                        let x = t.g.as_f32(bits);
                        t.g.store(p, x);
                    }
                    Ok(())
                })
            }
            (13..=15, StageIo::Vertex(_)) => unsupported("exp pos1..3 other than clip distances"),
            (32..=63, StageIo::Vertex(vs)) => {
                let Some(&loc) = vs.params.get(&(e.target - 32)) else {
                    return Ok(()); // nothing reads this parameter
                };
                let var = self.out_var(loc, Class::Float);
                self.guarded(c.active, |t| {
                    let tf = t.g.t_f32();
                    for &(ch, bits) in &chans {
                        let p = t.out_chan(var, tf, ch);
                        let x = t.g.as_f32(bits);
                        t.g.store(p, x);
                    }
                    Ok(())
                })
            }
            (0..=7, StageIo::Pixel(ps)) => {
                let class = ps
                    .targets
                    .get(e.target as usize)
                    .copied()
                    .unwrap_or(Class::Float);
                let a8 = ps.a8.get(e.target as usize).copied().unwrap_or(false);
                let var = self.out_var(u32::from(e.target), class);
                self.guarded(c.active, |t| {
                    let ty = t.scalar(class);
                    for &(ch, bits) in &chans {
                        // A8 is stored as R8: the exported alpha is the stored value, in red
                        // (and stays in alpha for the blend factors that read it).
                        let ch = if a8 && ch == 3 {
                            let p = t.out_chan(var, ty, 0);
                            let x = t.typed_from_bits(class, bits);
                            t.g.store(p, x);
                            3
                        } else if a8 {
                            continue;
                        } else {
                            ch
                        };
                        let p = t.out_chan(var, ty, ch);
                        let x = t.typed_from_bits(class, bits);
                        t.g.store(p, x);
                    }
                    Ok(())
                })
            }
            (8, StageIo::Pixel(_)) => self.export_z(c, &chans),
            _ => unsupported(&format!("exp target {}", e.target)),
        }
    }

    /// `exp mrtz`: channel 0 depth, 1 stencil reference, 2 sample mask.
    fn export_z(&mut self, c: &Ctx<Gen>, chans: &[(u32, u32)]) -> R<()> {
        for &(ch, bits) in chans {
            match ch {
                0 => {
                    let tf = self.g.t_f32();
                    let var = match self.st().depth {
                        Some(v) => v,
                        None => {
                            let v = self.global(StorageClass::Output, tf);
                            self.g
                                .m
                                .decorate(v, Decoration::BuiltIn, &[BuiltIn::FragDepth as u32]);
                            self.st().depth = Some(v);
                            v
                        }
                    };
                    self.guarded(c.active, |t| {
                        let x = t.g.as_f32(bits);
                        t.g.store(var, x);
                        Ok(())
                    })?;
                }
                1 => {
                    let ti = self.t_i32();
                    let var = match self.st().stencil {
                        Some(v) => v,
                        None => {
                            self.g.m.capability(Capability::StencilExportEXT);
                            self.g.m.extension("SPV_EXT_shader_stencil_export");
                            let v = self.global(StorageClass::Output, ti);
                            self.g.m.decorate(
                                v,
                                Decoration::BuiltIn,
                                &[BuiltIn::FragStencilRefEXT as u32],
                            );
                            self.st().stencil = Some(v);
                            v
                        }
                    };
                    self.guarded(c.active, |t| {
                        let x = t.typed_from_bits(Class::Sint, bits);
                        t.g.store(var, x);
                        Ok(())
                    })?;
                }
                2 => {
                    let tu = self.g.t_u32();
                    let one = self.g.k(1);
                    let arr = self.g.m.global(Op::TypeArray, None, &[tu, one]);
                    let var = match self.st().mask_out {
                        Some(v) => v,
                        None => {
                            let v = self.global(StorageClass::Output, arr);
                            self.g.m.decorate(
                                v,
                                Decoration::BuiltIn,
                                &[BuiltIn::SampleMask as u32],
                            );
                            self.st().mask_out = Some(v);
                            v
                        }
                    };
                    self.guarded(c.active, |t| {
                        let p = t.out_chan(var, tu, 0);
                        t.g.store(p, bits);
                        Ok(())
                    })?;
                }
                _ => return unsupported("exp mrtz alpha channel"),
            }
        }
        Ok(())
    }
}
