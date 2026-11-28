//! Graphics and compute pipelines from the title's pipeline objects (chunk 10a, `docs/shaders.md`,
//! Chunk 10 design): the VS and PS translated as a linked pair (`io::translate_pair`; a missing
//! PS is a depth-only pipeline; the pair's static samplers are the immutable samplers of set 1
//! and pick the pipeline layout, chunk 10b), the fixed state read by `pipeline_state`, the formats from the
//! description, and dynamic rendering (no render pass).
//!
//! A pipeline is built at the first draw that uses it and cached by its serial with its outcome:
//! `Built`, or `Failed(reason)` logged once, after which its draws are skipped. A pipeline is
//! refused (with the reason) when it has a stage other than VS and PS, reads
//! an address the backend cannot translate, uses a logic op, dual-source blending, conservative
//! rasterization, multisampling, a patch or adjacency topology, an instance step rate above 1,
//! or a format with no Vulkan mapping. A rectangle list (list topology 17) uses a second variant
//! of the pipeline with a geometry stage made from the vertex outputs (`spirv::rect`).
//! Set `NATIV64_SHADER_VALIDATE` to have every module run through `spirv-val` (a failure is logged; the pipeline is still built).
//!
//! Dynamic state: viewport, scissor, blend constants, stencil reference, primitive topology,
//! vertex binding strides and primitive restart; everything else (cull, front face, depth
//! test, blend) is the pipeline's own. A vertex input is a VS input matched with the input
//! element of the same semantic name and index (the VS's own location), binding = input slot.
//!
//! A compute pipeline (chunk 10c) is its one CS translated by `io::translate_stage` (pipeline
//! layout as the graphics ones: set 0, the static samplers' set 1, the root block address as
//! push constant); a translator refusal is its `Failed` reason.
//!
//! shortcuts: the depth bias clamp and the depth bounds test are ignored; one viewport and one
//! scissor; `FrontCounterClockwise` false is `CLOCKWISE` **[unverified]** (the viewport is
//! flipped, which keeps the D3D screen orientation); an element whose format has no vertex
//! mapping fails the pipeline.

use super::bindless::{Bindless, Layouts};
use super::format::{self, FormatInfo, FLAG_DEPTH_STENCIL, FLAG_RENDER_TARGET};
use super::sampler::SamplerCache;
use super::{log_once, profile, Gpu};
use crate::d3d12::pipeline::{GraphicsState, PipelineData};
use crate::d3d12::pipeline_state::{fixed_state, Blend, FixedState, StencilFace};
use crate::shader::io::{self, StageIo};
use crate::shader::native::RootLayout;
use crate::shader::spirv::rect::rectangle_gs;
use crate::shader::{parse, ParsedShader};
use ash::vk;
use std::collections::HashMap;

/// A built pipeline and what a draw needs to know about it.
#[derive(Clone)]
pub struct Built {
    pub pipeline: vk::Pipeline,
    /// The pipeline layout and the set of static samplers to bind at set 1, if any.
    pub layouts: Layouts,
    /// The layout rows of both stages, to fill the root block.
    pub rows: Vec<RootLayout>,
    /// Per render target slot: its DXGI format (0: no target) and Vulkan format.
    pub color: Vec<(u32, vk::Format)>,
    /// The depth-stencil format, if the pipeline has one.
    pub depth: Option<(u32, FormatInfo)>,
    /// Input slots the vertex input reads, in order.
    pub vertex_slots: Vec<u32>,
    /// The index buffer's strip cut is on.
    pub strip_cut: bool,
    /// The raw topology type the pipeline's description or stream held.
    pub topology_type: u32,
    /// The list topology of the pipeline's class (dynamic topologies must stay in it).
    pub topology: vk::PrimitiveTopology,
}

pub enum Outcome {
    Built(Built),
    Failed(String),
}

pub struct Pipelines<'g> {
    gpu: &'g Gpu,
    /// By serial and whether the pipeline is the rectangle-list variant (it has a geometry stage).
    map: HashMap<(u64, bool, Vec<u32>), Outcome>,
}

impl<'g> Pipelines<'g> {
    pub fn new(gpu: &'g Gpu) -> Self {
        Pipelines {
            gpu,
            map: HashMap::new(),
        }
    }

    /// The outcome for `pd`, building it the first time. `rect` asks for the variant that
    /// completes rectangle lists with a geometry stage (`spirv::rect`). `formats` replaces the
    /// description's render target formats (DXGI, per slot, 0: keep) for a pipeline that is
    /// bound to a target of another format of the same family (sRGB against UNORM); empty keeps
    /// the description's.
    pub fn get(
        &mut self,
        bindless: &mut Bindless<'_>,
        samplers: &mut SamplerCache<'_>,
        pd: &PipelineData,
        rect: bool,
        formats: &[u32],
    ) -> &Outcome {
        let gpu = self.gpu;
        let key = (pd.serial, rect, formats.to_vec());
        self.map.entry(key).or_insert_with(|| {
            let _step = profile::step(profile::Step::Pipeline);
            let t0 = std::time::Instant::now();
            let built = build(gpu, bindless, samplers, pd, rect, formats);
            profile::exec_add(profile::Exec::Pipelines, 1);
            profile::exec_since(profile::Exec::PipelineNs, t0);
            match built {
                Ok(b) => Outcome::Built(b),
                Err(why) => {
                    let kind = if rect { " as a rectangle list" } else { "" };
                    log_once(format!(
                        "pipeline {} is not replayed{kind}: {why}",
                        pd.serial
                    ));
                    Outcome::Failed(why)
                }
            }
        })
    }

    /// How many pipelines were built (tests).
    #[cfg(test)]
    pub fn built(&self) -> usize {
        self.map
            .values()
            .filter(|o| matches!(o, Outcome::Built(_)))
            .count()
    }
}

impl Drop for Pipelines<'_> {
    fn drop(&mut self) {
        for o in self.map.values() {
            if let Outcome::Built(b) = o {
                // SAFETY: built by us; the queue is idle when the backend drops.
                unsafe { self.gpu.device.destroy_pipeline(b.pipeline, None) };
            }
        }
    }
}

fn stage<'a>(pd: &'a PipelineData, name: &str) -> Option<&'a [u8]> {
    pd.stages
        .iter()
        .find(|s| s.name == name)
        .map(|s| &s.bytes[..])
}

/// What the shaders of a pipeline read through the descriptors of the resource heap.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HeapReads {
    /// A stage reads the resource heap directly (a layout row of type 5): any descriptor of the
    /// heap can be read, by an index the shader computes.
    pub whole: bool,
    /// The shader resource and unordered access views read through a root table: root
    /// parameter, slot in the table (the used row's table dword / 8: slots are 32 bytes) and
    /// whether it is an array (a stride), whose length is not known.
    pub views: Vec<(u32, u32, bool)>,
}

/// The `HeapReads` of the stages of `pd`; what a stage that does not parse reads is not listed
/// (the pipeline is not replayed).
pub fn heap_reads(pd: &PipelineData) -> HeapReads {
    let mut out = HeapReads::default();
    for name in ["vertex", "pixel", "compute"] {
        let Some(r) = stage(pd, name)
            .and_then(|b| parse_stage(b, name).ok())
            .and_then(|s| s.record().cloned())
        else {
            continue;
        };
        out.whole |= r.layout.iter().any(|l| l.root_type == 5);
        // shader resource views (kind 0) and unordered access views (kind 3, compute)
        for u in r
            .used
            .iter()
            .filter(|u| matches!(u.kind, 0 | 3) && u.root_type == 3)
        {
            let v = (u32::from(u.root_slot), u.table_offset / 8, u.stride != 0);
            if !out.views.contains(&v) {
                out.views.push(v);
            }
        }
    }
    out
}

fn blend_factor(f: u32) -> Result<vk::BlendFactor, String> {
    use vk::BlendFactor as B;
    Ok(match f {
        1 => B::ZERO,
        2 => B::ONE,
        3 => B::SRC_COLOR,
        4 => B::ONE_MINUS_SRC_COLOR,
        5 => B::SRC_ALPHA,
        6 => B::ONE_MINUS_SRC_ALPHA,
        7 => B::DST_ALPHA,
        8 => B::ONE_MINUS_DST_ALPHA,
        9 => B::DST_COLOR,
        10 => B::ONE_MINUS_DST_COLOR,
        11 => B::SRC_ALPHA_SATURATE,
        14 => B::CONSTANT_COLOR,
        15 => B::ONE_MINUS_CONSTANT_COLOR,
        20 => B::CONSTANT_ALPHA,
        21 => B::ONE_MINUS_CONSTANT_ALPHA,
        16..=19 => return Err("dual-source blending".into()),
        _ => return Err(format!("blend factor {f}")),
    })
}

fn blend_op(o: u32) -> Result<vk::BlendOp, String> {
    Ok(match o {
        1 => vk::BlendOp::ADD,
        2 => vk::BlendOp::SUBTRACT,
        3 => vk::BlendOp::REVERSE_SUBTRACT,
        4 => vk::BlendOp::MIN,
        5 => vk::BlendOp::MAX,
        _ => return Err(format!("blend operation {o}")),
    })
}

/// The blend state of an A8_UNORM target (stored as R8): the alpha factors and operation act on
/// the one stored channel, and only the alpha write bit counts (red in the image).
fn a8_blend(b: &Blend) -> Blend {
    let mut a = *b;
    a.src = b.src_alpha;
    a.dst = b.dst_alpha;
    a.op = b.op_alpha;
    a.write_mask = (b.write_mask >> 3) & 1;
    a
}

fn attachment(b: &Blend) -> Result<vk::PipelineColorBlendAttachmentState, String> {
    if b.logic_enable {
        return Err("a blend logic operation".into());
    }
    let mut a = vk::PipelineColorBlendAttachmentState::default().color_write_mask(
        vk::ColorComponentFlags::from_raw(u32::from(b.write_mask & 0xf)),
    );
    if b.enable {
        a = a
            .blend_enable(true)
            .src_color_blend_factor(blend_factor(b.src)?)
            .dst_color_blend_factor(blend_factor(b.dst)?)
            .color_blend_op(blend_op(b.op)?)
            .src_alpha_blend_factor(blend_factor(b.src_alpha)?)
            .dst_alpha_blend_factor(blend_factor(b.dst_alpha)?)
            .alpha_blend_op(blend_op(b.op_alpha)?);
    }
    Ok(a)
}

fn compare(f: u32) -> vk::CompareOp {
    vk::CompareOp::from_raw((f.clamp(1, 8) - 1) as i32)
}

fn stencil_op(f: &StencilFace, read: u8, write: u8) -> vk::StencilOpState {
    let op = |o: u32| vk::StencilOp::from_raw((o.clamp(1, 8) - 1) as i32);
    vk::StencilOpState::default()
        .fail_op(op(f.fail))
        .depth_fail_op(op(f.depth_fail))
        .pass_op(op(f.pass))
        .compare_op(compare(f.func))
        .compare_mask(u32::from(read))
        .write_mask(u32::from(write))
}

/// The Vulkan topology class of `PrimitiveTopologyType`: the list topology of its class.
fn topology(kind: u32) -> Result<vk::PrimitiveTopology, String> {
    match kind {
        1 => Ok(vk::PrimitiveTopology::POINT_LIST),
        2 => Ok(vk::PrimitiveTopology::LINE_LIST),
        3 | 0 => Ok(vk::PrimitiveTopology::TRIANGLE_LIST),
        // The console's rectangle, line loop, quad and quad strip values reach this field in
        // some streams (19 seen): triangle class, the draw decides what its list topology means.
        17..=20 => Ok(vk::PrimitiveTopology::TRIANGLE_LIST),
        _ => Err(format!("topology type {kind} (patches)")),
    }
}

/// The Vulkan topology of a `D3D_PRIMITIVE_TOPOLOGY`, if it has one without a geometry stage.
pub fn dynamic_topology(t: u32) -> Option<vk::PrimitiveTopology> {
    use vk::PrimitiveTopology as T;
    Some(match t {
        1 => T::POINT_LIST,
        2 => T::LINE_LIST,
        3 => T::LINE_STRIP,
        4 => T::TRIANGLE_LIST,
        5 => T::TRIANGLE_STRIP,
        6 => T::TRIANGLE_FAN,
        _ => return None,
    })
}

/// Whether `t` restarts on the index buffer's cut value.
pub fn is_strip(t: vk::PrimitiveTopology) -> bool {
    matches!(
        t,
        vk::PrimitiveTopology::LINE_STRIP
            | vk::PrimitiveTopology::TRIANGLE_STRIP
            | vk::PrimitiveTopology::TRIANGLE_FAN
    )
}

fn parse_stage(bytes: &[u8], what: &str) -> Result<ParsedShader, String> {
    parse(bytes).map_err(|e| format!("the {what} blob: {e}"))
}

/// `NATIV64_SHADER_VALIDATE`: runs the module through `spirv-val` and logs a failure.
fn validate(words: &[u32], what: &str) {
    if std::env::var_os("NATIV64_SHADER_VALIDATE").is_none() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("nativ64-spv-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let file = dir.join(format!("{what}-{:08x}.spv", words.len()));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    if std::fs::write(&file, bytes).is_err() {
        return;
    }
    match std::process::Command::new("spirv-val")
        .arg("--target-env")
        .arg("vulkan1.3")
        .arg(&file)
        .output()
    {
        Ok(o) if !o.status.success() => log_once(format!(
            "spirv-val rejects the {what} module {}: {}",
            file.display(),
            String::from_utf8_lossy(&o.stderr)
                .lines()
                .next()
                .unwrap_or("")
        )),
        Ok(_) => {}
        Err(e) => log_once(format!(
            "NATIV64_SHADER_VALIDATE: cannot run spirv-val: {e}"
        )),
    }
}

/// The vertex input state: bindings, attributes and the slots used.
struct Input {
    bindings: Vec<vk::VertexInputBindingDescription>,
    attributes: Vec<vk::VertexInputAttributeDescription>,
}

fn vertex_input(g: &GraphicsState, io: &StageIo) -> Result<Input, String> {
    let StageIo::Vertex(v) = io else {
        return Err("the vertex stage has no vertex I/O".into());
    };
    // append offsets (D3D12_APPEND_ALIGNED_ELEMENT) follow the previous element of the slot
    let mut next: HashMap<u32, u32> = HashMap::new();
    let mut offsets = Vec::new();
    for e in &g.input_layout {
        let size = format::vertex(e.format).map_or(0, |f| f.1);
        let at = if e.offset == u32::MAX {
            next.get(&e.slot).copied().unwrap_or(0).next_multiple_of(4)
        } else {
            e.offset
        };
        next.insert(e.slot, at + size);
        offsets.push(at);
    }
    let mut input = Input {
        bindings: Vec::new(),
        attributes: Vec::new(),
    };
    for a in &v.attrs {
        let Some((name, index)) = &a.semantic else {
            log_once(format!(
                "vertex input {} has no semantic: left unfed",
                a.location
            ));
            continue;
        };
        let hit = g
            .input_layout
            .iter()
            .position(|e| e.name.to_ascii_uppercase() == *name && e.index == *index);
        let Some(i) = hit else {
            log_once(format!("no input element for {name}{index}: left unfed"));
            continue;
        };
        let e = &g.input_layout[i];
        let rate = match (e.class, e.step_rate) {
            (0, _) => vk::VertexInputRate::VERTEX,
            (1, 0 | 1) => vk::VertexInputRate::INSTANCE,
            (1, n) => return Err(format!("an instance step rate of {n}")),
            (c, _) => return Err(format!("input classification {c}")),
        };
        let Some((vf, _)) = format::vertex(e.format) else {
            return Err(format!("vertex format {} of {name}{index}", e.format));
        };
        match input.bindings.iter().find(|b| b.binding == e.slot) {
            Some(b) if b.input_rate != rate => {
                return Err(format!(
                    "slot {} mixes per-vertex and per-instance data",
                    e.slot
                ))
            }
            Some(_) => {}
            None => input.bindings.push(vk::VertexInputBindingDescription {
                binding: e.slot,
                stride: 0, // dynamic
                input_rate: rate,
            }),
        }
        input.attributes.push(vk::VertexInputAttributeDescription {
            location: a.location,
            binding: e.slot,
            format: vf,
            offset: offsets[i],
        });
    }
    Ok(input)
}

fn color_formats(g: &GraphicsState) -> Result<Vec<(u32, vk::Format)>, String> {
    g.target_formats
        .iter()
        .map(|&f| {
            if f == 0 {
                Ok((0, vk::Format::UNDEFINED))
            } else {
                format::lookup(f, FLAG_RENDER_TARGET)
                    .map(|i| (f, i.format))
                    .ok_or_else(|| format!("render target format {f}"))
            }
        })
        .collect()
}

fn build(
    gpu: &Gpu,
    bindless: &mut Bindless<'_>,
    samplers: &mut SamplerCache<'_>,
    pd: &PipelineData,
    rect: bool,
    formats: &[u32],
) -> Result<Built, String> {
    let Some(g) = pd.graphics.as_ref() else {
        return build_compute(gpu, bindless, samplers, pd);
    };
    let varied;
    let g = if formats.is_empty() {
        g
    } else {
        let mut v = g.clone();
        for (t, &f) in v.target_formats.iter_mut().zip(formats) {
            if f != 0 {
                *t = f;
            }
        }
        varied = v;
        &varied
    };
    if let Some(s) = pd
        .stages
        .iter()
        .find(|s| !matches!(s.name, "vertex" | "pixel"))
    {
        return Err(format!("a {} stage (chunk 10c)", s.name));
    }
    let vs_bytes = stage(pd, "vertex").ok_or("no vertex stage")?;
    let fixed = fixed_state(g)?;
    check_fixed(&fixed, g)?;
    let vs = parse_stage(vs_bytes, "vertex")?;
    let ps = stage(pd, "pixel")
        .map(|b| parse_stage(b, "pixel"))
        .transpose()?;
    if rect && !gpu.caps.geometry {
        return Err("a rectangle list needs geometryShader, which this device lacks".into());
    }
    let (tv, tp) = match &ps {
        Some(ps) => {
            let (a, b) =
                io::translate_pair(&vs, ps, &g.target_formats).map_err(|e| e.to_string())?;
            (a, Some(b))
        }
        None => (
            io::translate_stage(&vs, &[], None).map_err(|e| e.to_string())?,
            None,
        ),
    };
    for t in std::iter::once(&tv).chain(tp.iter()) {
        if t.memory_addresses {
            return Err("a stage that reads memory through an address it loaded".into());
        }
    }
    // the pair's static samplers: both stages carry the whole list
    let layouts = bindless.layouts(&tv.samplers, samplers)?;
    let vio = io::stage_io(&vs, &[], None).ok_or("the vertex stage has no I/O")?;
    let input = vertex_input(g, &vio)?;
    let color = color_formats(g)?;
    let depth = match g.depth_format {
        0 => None,
        f => Some((
            f,
            format::lookup(f, FLAG_DEPTH_STENCIL).ok_or_else(|| format!("depth format {f}"))?,
        )),
    };
    let topo = topology(g.topology_type)?;
    if rect && topo != vk::PrimitiveTopology::TRIANGLE_LIST {
        return Err(format!(
            "a rectangle list on topology type {}",
            g.topology_type
        ));
    }
    let gs = if rect {
        let iface = io::rect_interface(&vs, ps.as_ref(), &g.target_formats)
            .ok_or("the vertex stage has no I/O")?;
        Some(rectangle_gs(&iface))
    } else {
        None
    };
    let blends: Vec<_> = (0..color.len())
        .map(|i| {
            let b = &fixed.targets[if fixed.independent_blend { i } else { 0 }];
            if g.target_formats.get(i).is_some_and(|&f| io::a8_target(f)) {
                attachment(&a8_blend(b))
            } else {
                attachment(b)
            }
        })
        .collect::<Result<_, _>>()?;
    let mut rows = vs.record().map(|r| r.layout.clone()).unwrap_or_default();
    for r in ps
        .as_ref()
        .and_then(|p| p.record())
        .map_or(&[][..], |r| &r.layout[..])
    {
        if !rows.contains(r) {
            rows.push(*r);
        }
    }
    let d = &gpu.device;
    validate(&tv.words, "vertex");
    if let Some(t) = &tp {
        validate(&t.words, "pixel");
    }
    if let Some(w) = &gs {
        validate(w, "rectangle");
    }
    // SAFETY: every object made here is destroyed before returning (modules) or owned by the
    // returned `Built`.
    unsafe {
        let module = |w: &[u32]| {
            d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(w), None)
                .map_err(|e| format!("vkCreateShaderModule: {e}"))
        };
        let vm = module(&tv.words)?;
        let pm = match tp.as_ref().map(|t| module(&t.words)).transpose() {
            Ok(m) => m,
            Err(e) => {
                d.destroy_shader_module(vm, None);
                return Err(e);
            }
        };
        let gm = match gs.as_ref().map(|w| module(w)).transpose() {
            Ok(m) => m,
            Err(e) => {
                d.destroy_shader_module(vm, None);
                if let Some(pm) = pm {
                    d.destroy_shader_module(pm, None);
                }
                return Err(e);
            }
        };
        let entry = c"main";
        let mut stages = vec![vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vm)
            .name(entry)];
        if let Some(pm) = pm {
            stages.push(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(pm)
                    .name(entry),
            );
        }
        if let Some(gm) = gm {
            stages.insert(
                1,
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::GEOMETRY)
                    .module(gm)
                    .name(entry),
            );
        }
        let vi = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&input.bindings)
            .vertex_attribute_descriptions(&input.attributes);
        let ia = vk::PipelineInputAssemblyStateCreateInfo::default().topology(topo);
        let vp = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let bias = fixed.depth_bias != 0 || fixed.slope_bias != 0.0;
        let rs = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(!fixed.depth_clip)
            .polygon_mode(if fixed.fill == 2 {
                vk::PolygonMode::LINE
            } else {
                vk::PolygonMode::FILL
            })
            .cull_mode(match fixed.cull {
                1 => vk::CullModeFlags::NONE,
                2 => vk::CullModeFlags::FRONT,
                _ => vk::CullModeFlags::BACK,
            })
            .front_face(if fixed.front_ccw {
                vk::FrontFace::COUNTER_CLOCKWISE
            } else {
                vk::FrontFace::CLOCKWISE
            })
            .depth_bias_enable(bias)
            .depth_bias_constant_factor(fixed.depth_bias as f32)
            .depth_bias_slope_factor(fixed.slope_bias)
            .line_width(1.0);
        let mask = [fixed.sample_mask];
        let ms = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_mask(&mask)
            .alpha_to_coverage_enable(fixed.alpha_to_coverage);
        let ds_state = &fixed.depth;
        let has_stencil =
            depth.is_some_and(|(_, i)| i.aspect.contains(vk::ImageAspectFlags::STENCIL));
        let ds = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(depth.is_some() && ds_state.depth_enable)
            .depth_write_enable(depth.is_some() && ds_state.depth_write)
            .depth_compare_op(compare(ds_state.depth_func))
            .stencil_test_enable(has_stencil && ds_state.stencil_enable)
            .front(stencil_op(
                &ds_state.front,
                ds_state.read_mask,
                ds_state.write_mask,
            ))
            .back(stencil_op(
                &ds_state.back,
                ds_state.read_mask,
                ds_state.write_mask,
            ));
        let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);
        let mut dynamic = vec![
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::BLEND_CONSTANTS,
            vk::DynamicState::STENCIL_REFERENCE,
            vk::DynamicState::PRIMITIVE_TOPOLOGY,
            vk::DynamicState::PRIMITIVE_RESTART_ENABLE,
        ];
        // Strides are bound with the vertex buffers: without a binding there is nothing to bind
        // them to (VUID-vkCmdDraw-pStrides-04913).
        if !input.bindings.is_empty() {
            dynamic.push(vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE);
        }
        let dy = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
        let formats: Vec<vk::Format> = color.iter().map(|c| c.1).collect();
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
        if let Some((_, i)) = depth {
            rendering = rendering.depth_attachment_format(i.format);
            if has_stencil {
                rendering = rendering.stencil_attachment_format(i.format);
            }
        }
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vi)
            .input_assembly_state(&ia)
            .viewport_state(&vp)
            .rasterization_state(&rs)
            .multisample_state(&ms)
            .depth_stencil_state(&ds)
            .color_blend_state(&cb)
            .dynamic_state(&dy)
            .layout(layouts.layout)
            .push_next(&mut rendering);
        let made = d.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None);
        d.destroy_shader_module(vm, None);
        for m in [pm, gm].into_iter().flatten() {
            d.destroy_shader_module(m, None);
        }
        let pipeline = match made {
            Ok(p) => p[0],
            Err((_, e)) => return Err(format!("vkCreateGraphicsPipelines: {e}")),
        };
        Ok(Built {
            pipeline,
            layouts,
            rows,
            color,
            depth,
            vertex_slots: input.bindings.iter().map(|b| b.binding).collect(),
            strip_cut: fixed.strip_cut != 0,
            topology: topo,
            topology_type: g.topology_type,
        })
    }
}

/// A translated module and the layout rows of its program.
#[cfg(test)]
type Injected = (Vec<u32>, Vec<RootLayout>);

#[cfg(test)]
thread_local! {
    /// Modules (and layout rows) that stand for the translation of the compute pipelines of
    /// these serials, so that dispatches are tested whatever the translator covers.
    pub(super) static COMPUTE_MODULES: std::cell::RefCell<HashMap<u64, Injected>> =
        Default::default();
}

#[cfg(test)]
fn injected(serial: u64) -> Option<(Vec<u32>, Vec<RootLayout>)> {
    COMPUTE_MODULES.with(|m| m.borrow().get(&serial).cloned())
}

#[cfg(not(test))]
fn injected(_serial: u64) -> Option<(Vec<u32>, Vec<RootLayout>)> {
    None
}

/// A compute pipeline (chunk 10c): the CS translated alone (`io::translate_stage`), the pipeline
/// layout of its static samplers, the push constant being the root block's address.
fn build_compute(
    gpu: &Gpu,
    bindless: &mut Bindless<'_>,
    samplers: &mut SamplerCache<'_>,
    pd: &PipelineData,
) -> Result<Built, String> {
    if let Some(s) = pd.stages.iter().find(|s| s.name != "compute") {
        return Err(format!("a stage {} in a compute pipeline", s.name));
    }
    let (words, static_samplers, rows) = match injected(pd.serial) {
        Some((words, rows)) => (words, Vec::new(), rows),
        None => {
            let bytes = stage(pd, "compute").ok_or("no compute stage")?;
            let cs = parse_stage(bytes, "compute")?;
            let t = io::translate_stage(&cs, &[], None).map_err(|e| e.to_string())?;
            if t.memory_addresses {
                return Err("a stage that reads memory through an address it loaded".into());
            }
            let rows = cs.record().map(|r| r.layout.clone()).unwrap_or_default();
            (t.words, t.samplers, rows)
        }
    };
    let layouts = bindless.layouts(&static_samplers, samplers)?;
    validate(&words, "compute");
    let d = &gpu.device;
    // SAFETY: the module is destroyed before returning; the pipeline is owned by the `Built`.
    unsafe {
        let module = d
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .map_err(|e| format!("vkCreateShaderModule: {e}"))?;
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(c"main"),
            )
            .layout(layouts.layout);
        let made = d.create_compute_pipelines(vk::PipelineCache::null(), &[info], None);
        d.destroy_shader_module(module, None);
        let pipeline = match made {
            Ok(p) => p[0],
            Err((_, e)) => return Err(format!("vkCreateComputePipelines: {e}")),
        };
        Ok(Built {
            pipeline,
            layouts,
            rows,
            color: Vec::new(),
            depth: None,
            vertex_slots: Vec::new(),
            strip_cut: false,
            topology_type: 0,
            topology: vk::PrimitiveTopology::POINT_LIST,
        })
    }
}

/// The state the backend refuses rather than approximates.
fn check_fixed(f: &FixedState, g: &GraphicsState) -> Result<(), String> {
    if f.conservative {
        return Err("conservative rasterization".into());
    }
    if g.sample_count > 1 {
        return Err(format!("{} samples per pixel", g.sample_count));
    }
    Ok(())
}

/// The DXGI format of the other member of an sRGB / UNORM pair of 8-bit formats.
pub fn srgb_twin(dxgi: u32) -> Option<u32> {
    match dxgi {
        28 => Some(29),
        29 => Some(28),
        87 => Some(91),
        91 => Some(87),
        _ => None,
    }
}

#[cfg(test)]
mod a8_tests {
    use super::*;

    #[test]
    fn a8_blend_uses_the_alpha_state_for_the_one_channel() {
        let b = Blend {
            enable: true,
            src: 5,
            dst: 6,
            op: 3,
            src_alpha: 2,
            dst_alpha: 7,
            op_alpha: 1,
            write_mask: 0x8,
            ..Blend::default()
        };
        let a = a8_blend(&b);
        assert_eq!((a.src, a.dst, a.op, a.write_mask), (2, 7, 1, 1));
        assert_eq!(
            a8_blend(&Blend {
                write_mask: 0x7,
                ..b
            })
            .write_mask,
            0
        );
        assert!(io::a8_target(65) && !io::a8_target(28));
    }
}
