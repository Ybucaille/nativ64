//! Draws (chunk 10a, `docs/shaders.md`, Chunk 10 design): the list's graphics state kept while
//! a stream replays, and `Draw` / `DrawIndexed` recorded with dynamic rendering into the
//! images of the bound render targets.
//!
//! A draw is recorded when its pipeline built (`pipeline.rs`), its targets are images of ours
//! in the formats the pipeline was made for, and every address it reads (root block, vertex
//! and index buffers) translates to GPU memory (`gpumem.rs`); otherwise it is counted by
//! reason in `ReplayStats::draw_skips`, logged once, and skipped. Rendering stays open over
//! consecutive draws on the same targets and ends at the next command that is not a draw or
//! a state change (a barrier follows, since images are in `GENERAL` and nothing else orders
//! two rendering instances).
//!
//! Per draw: the root block is written into the submission's arena and its device address
//! pushed (8 bytes, VS and PS); the bindless set is bound; viewport (flipped, `y + height,
//! -height`), scissor, blend constants, stencil reference, topology, primitive restart and
//! vertex strides are set dynamically. A non-zero first vertex or base vertex is logged once:
//! the translated vertex shaders count from `VertexIndex - BaseVertex` (chunk 8).
//!
//! shortcuts: one viewport and one scissor (the first); a target view of several layers draws
//! into its first; the viewport and scissor default to the first target's whole area.

use super::bindless::{Bindless, HostBuf, ViewTable};
use super::format::{self, FLAG_RENDER_TARGET};
use super::gpumem::GpuMemory;
use super::pipeline::{dynamic_topology, is_strip, Outcome, Pipelines};
use super::replay::{global_barrier, mip_size, view_range, Img, VkBackend};
use super::rootblock::{self, RootArgs};
use super::sampler::SamplerCache;
use super::{log_once, Gpu};
use crate::d3d12::command::{Command, IndexView, Rect, VertexView};
use crate::d3d12::descriptor::{depth_stencil_view, render_target_view};
use crate::d3d12::info::get_descriptor_increment_size;
use crate::d3d12::pipeline::PipelineData;
use crate::xobj::{Iface, XObject};
use ash::{vk, Device};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;

/// Why a draw was not recorded.
pub(super) enum Skip {
    /// No pipeline, or one that did not build or is not for graphics.
    Pipeline(String),
    /// An address of the draw is not GPU memory.
    Memory(String),
    /// Missing or mismatched render targets.
    Target(String),
    /// Anything else in the state (topology, vertex buffer, index format, arena).
    State(String),
}

/// Draws skipped, by reason.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawSkips {
    pub pipeline: u64,
    pub memory: u64,
    pub target: u64,
    pub state: u64,
}

impl DrawSkips {
    pub fn total(&self) -> u64 {
        self.pipeline + self.memory + self.target + self.state
    }
}

/// What a list set, as far as draws and dispatches are concerned.
pub(super) struct ListState {
    pub(super) pipeline: usize,
    pub(super) topology: u32,
    pub(super) viewport: Option<[f32; 6]>,
    pub(super) scissor: Option<Rect>,
    pub(super) blend: Option<[f32; 4]>,
    stencil_ref: u32,
    index: Option<IndexView>,
    pub(super) vertex: BTreeMap<u32, VertexView>,
    targets: Vec<u64>,
    depth: Option<u64>,
    pub(super) root: RootArgs,
    /// The arguments of the compute root signature (dispatches).
    pub(super) croot: RootArgs,
}

impl Default for ListState {
    fn default() -> Self {
        ListState {
            pipeline: 0,
            topology: 0,
            viewport: None,
            scissor: None,
            blend: None,
            stencil_ref: 0,
            index: None,
            vertex: BTreeMap::new(),
            targets: Vec::new(),
            depth: None,
            root: RootArgs::default(),
            croot: super::compute::new_args(),
        }
    }
}

impl ListState {
    fn apply(&mut self, c: &Command) {
        match c {
            Command::SetPipeline { pipeline } => self.pipeline = *pipeline,
            Command::ClearState { pipeline } => {
                *self = ListState {
                    pipeline: *pipeline,
                    ..Default::default()
                }
            }
            Command::Topology(t) => self.topology = *t,
            Command::Viewports(v) => self.viewport = v.first().copied(),
            Command::Scissors(s) => self.scissor = s.first().copied(),
            Command::BlendFactor(b) => self.blend = *b,
            Command::StencilRef(r) => self.stencil_ref = *r,
            Command::IndexBuffer(v) => self.index = *v,
            Command::VertexBuffers {
                start,
                count,
                views,
            } => {
                if views.is_empty() {
                    for s in *start..start.saturating_add(*count) {
                        self.vertex.remove(&s);
                    }
                }
                for (i, v) in views.iter().enumerate() {
                    self.vertex.insert(start + i as u32, *v);
                }
            }
            Command::RenderTargets {
                count,
                handles,
                single_range,
                depth,
            } => {
                // SAFETY: the function reads nothing (its object is ignored).
                let step =
                    u64::from(unsafe { get_descriptor_increment_size(std::ptr::null_mut(), 2) });
                self.targets = match (single_range, handles.first()) {
                    (true, Some(&first)) => {
                        (0..u64::from(*count)).map(|i| first + i * step).collect()
                    }
                    (true, None) => Vec::new(),
                    (false, _) => handles.clone(),
                };
                self.depth = *depth;
            }
            _ => {
                self.root.apply(c);
                self.croot.apply(c);
            }
        }
    }
}

/// Key of an image view: resource, mip, layer, depth view.
type ViewKey = (usize, u32, u32, bool);

/// The attachments a rendering instance was begun with.
#[derive(Clone, PartialEq, Eq)]
struct Open {
    colors: Vec<vk::ImageView>,
    depth: Option<vk::ImageView>,
    extent: (u32, u32),
}

/// Everything draws keep besides the images.
pub(super) struct DrawState<'g> {
    gpu: &'g Gpu,
    pub(super) list: ListState,
    pub(super) mem: GpuMemory<'g>,
    /// `Some(None)`: tried and failed.
    pub(super) bindless: Option<Option<Bindless<'g>>>,
    views: Option<ViewTable<'g>>,
    views_ready: bool,
    pub(super) pipelines: Pipelines<'g>,
    image_views: HashMap<ViewKey, vk::ImageView>,
    open: Option<Open>,
    /// The submission's root blocks: the buffer and the bytes used.
    arena: Option<(HostBuf<'g>, usize)>,
    arena_size: usize,
    /// The samplers of heap views and of static sampler sets (dropped after the layouts).
    pub(super) samplers: SamplerCache<'g>,
    /// Index buffers of quads expanded to triangles, by number of quads (for `Draw`).
    quad_indices: HashMap<u32, HostBuf<'g>>,
    /// Index buffers made for one submission (indexed quads), dropped by `end`.
    scratch: Vec<HostBuf<'g>>,
}

impl<'g> DrawState<'g> {
    pub(super) fn new(gpu: &'g Gpu) -> Self {
        DrawState {
            gpu,
            list: ListState::default(),
            mem: GpuMemory::new(gpu),
            bindless: None,
            views: None,
            views_ready: false,
            pipelines: Pipelines::new(gpu),
            image_views: HashMap::new(),
            open: None,
            arena: None,
            arena_size: 0,
            samplers: SamplerCache::new(gpu),
            quad_indices: HashMap::new(),
            scratch: Vec::new(),
        }
    }

    /// Before a submission: drops the imports of memory that changed (and the view records
    /// that held their addresses) and makes room in the arena for `draws` draws. The arena is
    /// kept from one submission to the next and grows by doubling when a submission needs more;
    /// it is made here, before the submission's images take the memory.
    pub(super) fn begin(&mut self, draws: usize) {
        let t = super::profile::start();
        if self.mem.sync() {
            if let Some(v) = &mut self.views {
                super::profile::count(
                    super::profile::Count::Dropped,
                    self.mem.dropped().len() as u64,
                );
                v.invalidate(self.mem.dropped());
            }
        }
        super::profile::stop(super::profile::Phase::Begin, t);
        self.views_ready = false;
        self.arena_size = draws * rootblock::STRIDE;
        if let Some((_, used)) = &mut self.arena {
            *used = 0;
        }
        let have = self.arena_bytes();
        if self.arena_size > have && self.arena_size > 0 {
            let size = self.arena_size.max(have * 2);
            // the GPU is idle between submissions: the old one goes first
            self.arena = None;
            self.make_arena(size);
        }
    }

    fn make_arena(&mut self, size: usize) {
        match HostBuf::new(self.gpu, size.max(rootblock::STRIDE)) {
            Ok(b) => self.arena = Some((b, 0)),
            Err(e) => log_once(format!("no root block arena: {e}")),
        }
    }

    /// The arena's size in bytes (0 when there is none).
    pub(super) fn arena_bytes(&self) -> usize {
        self.arena.as_ref().map_or(0, |(b, _)| b.size)
    }

    /// After a submission: the scratch buffers go; the arena stays.
    pub(super) fn end(&mut self) {
        self.open = None;
        self.scratch.clear();
    }

    /// The state commands of a list.
    pub(super) fn track(&mut self, c: &Command) {
        self.list.apply(c);
    }

    pub(super) fn reset_list(&mut self) {
        self.list = ListState::default();
    }

    /// Drops the views of the image of `res` (it is being replaced or destroyed).
    pub(super) fn forget_image(&mut self, res: usize) {
        let gone: Vec<ViewKey> = self
            .image_views
            .keys()
            .filter(|k| k.0 == res)
            .copied()
            .collect();
        for k in gone {
            if let Some(v) = self.image_views.remove(&k) {
                // SAFETY: made by `image_view`; the queue is idle between submissions.
                unsafe { self.gpu.device.destroy_image_view(v, None) };
            }
        }
    }

    pub(super) fn bindless(&mut self) -> Option<&Bindless<'g>> {
        if self.bindless.is_none() {
            self.bindless = Some(match Bindless::new(self.gpu) {
                Ok(b) => Some(b),
                Err(e) => {
                    log_once(format!("no bindless descriptor set: {e}"));
                    None
                }
            });
        }
        self.bindless.as_ref().and_then(Option::as_ref)
    }

    /// The view table, with the records of the views registered since the last submission.
    pub(super) fn view_table(&mut self) -> Option<u64> {
        if self.views.is_none() {
            match ViewTable::new(self.gpu) {
                Ok(v) => self.views = Some(v),
                Err(e) => {
                    log_once(format!("no view table: {e}"));
                    return None;
                }
            }
        }
        let v = self.views.as_mut()?;
        if !self.views_ready {
            self.views_ready = true;
            let _step = super::profile::step(super::profile::Step::Views);
            let t = super::profile::start();
            v.update(&mut self.mem);
            super::profile::stop(super::profile::Phase::Views, t);
        }
        Some(v.address())
    }

    /// The indices of `quads` quads as triangle lists, `(0 1 2) (0 2 3)` each: one buffer per
    /// count, kept.
    fn quad_buffer(&mut self, quads: u32) -> Result<vk::Buffer, Skip> {
        if !self.quad_indices.contains_key(&quads) {
            let mut bytes = Vec::with_capacity(24 * quads as usize);
            for q in 0..quads {
                for k in [0, 1, 2, 0, 2, 3] {
                    bytes.extend((4 * q + k).to_le_bytes());
                }
            }
            let mut buf = HostBuf::new(self.gpu, bytes.len())
                .map_err(|e| Skip::State(format!("no quad index buffer: {e}")))?;
            buf.write(0, &bytes);
            self.quad_indices.insert(quads, buf);
        }
        Ok(self.quad_indices[&quads].buffer)
    }

    /// Room for a root block in the arena: its offset.
    pub(super) fn arena_slot(&mut self) -> Option<(&mut HostBuf<'g>, usize)> {
        if self.arena.is_none() {
            self.make_arena(self.arena_size);
        }
        let (buf, used) = self.arena.as_mut()?;
        if *used + rootblock::STRIDE > buf.size {
            return None;
        }
        let at = *used;
        *used += rootblock::STRIDE;
        Some((buf, at))
    }
}

impl Drop for DrawState<'_> {
    fn drop(&mut self) {
        // SAFETY: made by `image_view`; the queue is idle when the backend drops.
        for (_, v) in self.image_views.drain() {
            unsafe { self.gpu.device.destroy_image_view(v, None) };
        }
    }
}

/// The pipeline object at `ptr`, if it is one.
///
/// # Safety
/// `ptr` was recorded from a live object of ours.
pub(super) unsafe fn pipeline_data<'a>(ptr: usize) -> Option<&'a PipelineData> {
    let obj = ptr as *mut c_void;
    (XObject::<PipelineData>::prefix(obj).iface() == Iface::PipelineState)
        .then(|| XObject::<PipelineData>::data(obj))
}

fn skip<T>(s: Skip) -> Result<T, Skip> {
    Err(s)
}

/// The console's `D3D_PRIMITIVE_TOPOLOGY` values from here on (17 rectangle list, 18 line loop,
/// 19 quad list, 20 quad strip), and the quad list.
const CONSOLE_TOPOLOGY: u32 = 17;
const QUADS: u32 = 19;
/// The rectangle list: three corners per rectangle, the fourth made by a geometry stage.
const RECTANGLES: u32 = 17;

/// 0 points, 1 lines, 2 triangles.
fn class_of(t: vk::PrimitiveTopology) -> u8 {
    use vk::PrimitiveTopology as T;
    match t {
        T::POINT_LIST => 0,
        T::LINE_LIST | T::LINE_STRIP => 1,
        _ => 2,
    }
}

/// What `clear_rects` writes.
#[derive(Clone, Copy)]
pub(super) enum ClearValue {
    Color([f32; 4]),
    Depth {
        depth: f32,
        stencil: u8,
        do_depth: bool,
        do_stencil: bool,
    },
}

impl<'g> VkBackend<'g> {
    /// Ends the open rendering instance, if any.
    pub(super) fn end_rendering(&mut self, d: &Device, cb: vk::CommandBuffer) {
        if self.draw.open.take().is_some() {
            // SAFETY: `cb` is recording inside a rendering instance.
            unsafe { d.cmd_end_rendering(cb) };
            global_barrier(d, cb);
        }
    }

    fn image_view(
        &mut self,
        res: usize,
        img: Img,
        mip: u32,
        layer: u32,
        depth: bool,
    ) -> Option<vk::ImageView> {
        let key = (res, mip, layer, depth);
        if let Some(&v) = self.draw.image_views.get(&key) {
            return Some(v);
        }
        let ci = vk::ImageViewCreateInfo::default()
            .image(img.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(img.info.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: img.info.aspect,
                base_mip_level: mip,
                level_count: 1,
                base_array_layer: layer,
                layer_count: 1,
            });
        // SAFETY: a valid image of ours; destroyed by `forget_image` or on drop.
        match unsafe { self.gpu.device.create_image_view(&ci, None) } {
            Ok(v) => {
                self.draw.image_views.insert(key, v);
                Some(v)
            }
            Err(e) => {
                log_once(format!("vkCreateImageView: {e}"));
                None
            }
        }
    }

    /// Clears the parts of one mip and layer inside `rects` with a rendering instance per
    /// rectangle (a clear command cannot be scissored). `depth` is `Some((depth, stencil value,
    /// clear depth, clear stencil))` for a depth view. Returns false when no view exists.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn clear_rects(
        &mut self,
        d: &Device,
        cb: vk::CommandBuffer,
        res: usize,
        img: Img,
        (mip, layer): (u32, u32),
        rects: &[[i32; 4]],
        value: ClearValue,
    ) -> bool {
        let depth = matches!(value, ClearValue::Depth { .. });
        let Some(view) = self.image_view(res, img, mip, layer, depth) else {
            return false;
        };
        let (w, h) = (
            (img.width >> mip).max(1) as i32,
            (img.height >> mip).max(1) as i32,
        );
        self.end_rendering(d, cb);
        for r in rects {
            let (x0, y0) = (r[0].clamp(0, w), r[1].clamp(0, h));
            let (x1, y1) = (r[2].clamp(x0, w), r[3].clamp(y0, h));
            if x1 == x0 || y1 == y0 {
                continue;
            }
            let area = vk::Rect2D {
                offset: vk::Offset2D { x: x0, y: y0 },
                extent: vk::Extent2D {
                    width: (x1 - x0) as u32,
                    height: (y1 - y0) as u32,
                },
            };
            let att = |clear: bool, v: vk::ClearValue| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(if clear {
                        vk::AttachmentLoadOp::CLEAR
                    } else {
                        vk::AttachmentLoadOp::LOAD
                    })
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(v)
            };
            let mut info = vk::RenderingInfo::default()
                .render_area(area)
                .layer_count(1);
            let color;
            let (dep, ste);
            match value {
                ClearValue::Color(c) => {
                    color = [att(
                        true,
                        vk::ClearValue {
                            color: vk::ClearColorValue { float32: c },
                        },
                    )];
                    info = info.color_attachments(&color);
                }
                ClearValue::Depth {
                    depth: dv,
                    stencil: sv,
                    do_depth,
                    do_stencil,
                } => {
                    let v = vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: dv,
                            stencil: u32::from(sv),
                        },
                    };
                    if img.info.aspect.contains(vk::ImageAspectFlags::DEPTH) {
                        dep = att(do_depth, v);
                        info = info.depth_attachment(&dep);
                    }
                    if img.info.aspect.contains(vk::ImageAspectFlags::STENCIL) {
                        ste = att(do_stencil, v);
                        info = info.stencil_attachment(&ste);
                    }
                }
            }
            // SAFETY: `cb` is recording outside a rendering instance; the view is ours.
            unsafe {
                d.cmd_begin_rendering(cb, &info);
                d.cmd_end_rendering(cb);
            }
        }
        global_barrier(d, cb);
        true
    }

    /// Whether `image(res)` would reuse an image (so it records nothing).
    fn image_is_ready(&self, res: usize) -> bool {
        // SAFETY: `res` was recorded by a list method from a live resource of ours.
        let Some((desc, _)) = (unsafe { crate::d3d12::resource::info_of(res as *mut _) }) else {
            return false;
        };
        self.image_desc(res) == Some(desc)
    }

    /// Records a `Draw` or `DrawIndexed`.
    pub(super) fn draw_command(&mut self, d: &Device, cb: vk::CommandBuffer, c: &Command) {
        match self.try_draw(d, cb, c, &[]) {
            Ok(()) => self.stats.drawn += 1,
            Err(s) => {
                let (n, why) = match s {
                    Skip::Pipeline(w) => (&mut self.stats.draw_skips.pipeline, w),
                    Skip::Memory(w) => (&mut self.stats.draw_skips.memory, w),
                    Skip::Target(w) => (&mut self.stats.draw_skips.target, w),
                    Skip::State(w) => (&mut self.stats.draw_skips.state, w),
                };
                *n += 1;
                log_once(format!("draw skipped: {why}"));
                if crate::backend::trace::enabled() {
                    // Under NATIV64_TRACE_DRAWS: each (pipeline, reason) once, with the
                    // pipeline's serial and its stages' shader hashes.
                    log_once(format!(
                        "draw skipped: pipeline {}: {why}",
                        self.describe_bound_pipeline()
                    ));
                }
            }
        }
    }

    /// The bound pipeline's serial and stage hashes, for the skip trace.
    fn describe_bound_pipeline(&self) -> String {
        let ptr = self.draw.list.pipeline;
        // SAFETY: the pointer was recorded from a live pipeline object of ours (or is 0).
        match (ptr != 0).then(|| unsafe { pipeline_data(ptr) }).flatten() {
            Some(pd) => {
                let stages: Vec<String> = pd
                    .stages
                    .iter()
                    .map(|s| format!("{}={:016x}", s.name, crate::shader::fnv1a64(&s.bytes)))
                    .collect();
                format!("#{} [{}]", pd.serial, stages.join(" "))
            }
            None => "none".into(),
        }
    }

    /// `over`: the render target formats (DXGI, per slot, 0: keep) to build the pipeline for
    /// instead of its description's, when a target is the sRGB or UNORM twin of the format
    /// the description names.
    fn try_draw(
        &mut self,
        d: &Device,
        cb: vk::CommandBuffer,
        c: &Command,
        over: &[u32],
    ) -> Result<(), Skip> {
        let ptr = self.draw.list.pipeline;
        if ptr == 0 {
            return skip(Skip::Pipeline("no pipeline bound".into()));
        }
        // SAFETY: the pointer was recorded from a live pipeline object of ours.
        let Some(pd) = (unsafe { pipeline_data(ptr) }) else {
            return skip(Skip::Pipeline("the bound object is not a pipeline".into()));
        };
        if self.draw.bindless().is_none() {
            return skip(Skip::State("no bindless descriptor set".into()));
        }
        let draw = &mut self.draw;
        let rects = draw.list.topology == RECTANGLES;
        let Some(Some(bl)) = draw.bindless.as_mut() else {
            return skip(Skip::State("no bindless descriptor set".into()));
        };
        let set = bl.set;
        let built = match draw.pipelines.get(bl, &mut draw.samplers, pd, rects, over) {
            Outcome::Built(b) => b.clone(),
            Outcome::Failed(w) => return skip(Skip::Pipeline(w.clone())),
        };
        let layout = built.layouts.layout;

        // Targets: the images of the bound views, in the formats the pipeline was made for.
        // Making an image records a barrier, which cannot happen inside a rendering instance.
        let list = &self.draw.list;
        let colors: Vec<(usize, u64)> = built
            .color
            .iter()
            .enumerate()
            .filter(|(_, (f, _))| *f != 0)
            .map(|(i, _)| (i, list.targets.get(i).copied().unwrap_or(0)))
            .collect();
        let depth_handle = built.depth.map(|_| list.depth.unwrap_or(0));
        let mut resources = Vec::new();
        for &(_, h) in &colors {
            match render_target_view(h) {
                Some(v) => resources.push(v),
                None => return skip(Skip::Target("a render target slot has no view".into())),
            }
        }
        let depth_view = match depth_handle {
            Some(h) => match depth_stencil_view(h) {
                Some(v) => Some(v),
                None => return skip(Skip::Target("the depth target has no view".into())),
            },
            None => None,
        };
        if resources.is_empty() && depth_view.is_none() {
            return skip(Skip::Target("the pipeline writes no target".into()));
        }
        if !resources
            .iter()
            .chain(depth_view.iter())
            .all(|v| self.image_is_ready(v.resource))
        {
            self.end_rendering(d, cb);
        }
        let mut color_views = Vec::new();
        let mut extent = None;
        for (&(i, _), v) in colors.iter().zip(&resources) {
            let Some(img) = self.image(d, cb, v.resource) else {
                return skip(Skip::Target("a render target has no image".into()));
            };
            self.own(v.resource);
            if img.info.format != built.color[i].1 {
                let twin = super::pipeline::srgb_twin(built.color[i].0).filter(|&t| {
                    format::lookup(t, FLAG_RENDER_TARGET).map(|f| f.format) == Some(img.info.format)
                });
                if let (Some(t), true) = (twin, over.get(i).copied().unwrap_or(0) == 0) {
                    let mut v = over.to_vec();
                    v.resize(built.color.len(), 0);
                    v[i] = t;
                    return self.try_draw(d, cb, c, &v);
                }
                return skip(Skip::Target(format!(
                    "render target {i} is {:?}, the pipeline was made for {:?}",
                    img.info.format, built.color[i].1
                )));
            }
            let desc = (v.described != 0).then_some(v.desc);
            let (mip, layer, _) = view_range(img, desc, false);
            let Some(view) = self.image_view(v.resource, img, mip, layer, false) else {
                return skip(Skip::Target("no image view".into()));
            };
            extent.get_or_insert((mip_size(img.width, mip), mip_size(img.height, mip)));
            color_views.push((i, view));
        }
        let mut depth_att = None;
        if let (Some(v), Some((_, info))) = (depth_view, built.depth) {
            let Some(img) = self.image(d, cb, v.resource) else {
                return skip(Skip::Target("the depth target has no image".into()));
            };
            self.own(v.resource);
            if img.info.format != info.format {
                return skip(Skip::Target("the depth target's format differs".into()));
            }
            let desc = (v.described != 0).then_some(v.desc);
            let (mip, layer, _) = view_range(img, desc, true);
            let Some(view) = self.image_view(v.resource, img, mip, layer, true) else {
                return skip(Skip::Target("no depth view".into()));
            };
            extent.get_or_insert((mip_size(img.width, mip), mip_size(img.height, mip)));
            depth_att = Some(view);
        }
        let Some(extent) = extent else {
            return skip(Skip::Target("no target".into()));
        };

        // Memory: the root block, the vertex and index buffers.
        let Some(view_table) = self.draw.view_table() else {
            return skip(Skip::State("no view table".into()));
        };
        let mem = &mut self.draw.mem;
        let list = &self.draw.list;
        let block = rootblock::fill(&built.rows, &list.root, view_table, &mut |a| {
            mem.translate(a, 1).map(|m| m.address)
        })
        .map_err(|e| match e {
            rootblock::FillError::Memory(w) => Skip::Memory(w),
            rootblock::FillError::Layout(w) => Skip::State(w),
        })?;
        let mut vbufs = Vec::new();
        for &slot in &built.vertex_slots {
            let Some(v) = list.vertex.get(&slot) else {
                return skip(Skip::State(format!(
                    "vertex buffer slot {slot} is not bound"
                )));
            };
            let Some(m) = mem.translate(v.address, u64::from(v.size).max(1)) else {
                return skip(Skip::Memory(format!("vertex buffer {:#x}", v.address)));
            };
            vbufs.push((slot, m, v));
        }
        let index = match c {
            Command::DrawIndexed { .. } => {
                let Some(iv) = list.index else {
                    return skip(Skip::State("no index buffer".into()));
                };
                let ty = match iv.format {
                    57 => vk::IndexType::UINT16,
                    42 => vk::IndexType::UINT32,
                    f => return skip(Skip::State(format!("index format {f}"))),
                };
                let Some(m) = mem.translate(iv.address, u64::from(iv.size).max(1)) else {
                    return skip(Skip::Memory(format!("index buffer {:#x}", iv.address)));
                };
                Some((m, ty))
            }
            _ => None,
        };
        if built.topology_type >= CONSOLE_TOPOLOGY || list.topology >= CONSOLE_TOPOLOGY {
            log_once(format!(
                "pipeline {}: topology type {}, list topology {}",
                pd.serial, built.topology_type, list.topology
            ));
        }
        let quads = list.topology == QUADS;
        let topo = match dynamic_topology(list.topology) {
            Some(t) if class_of(t) == class_of(built.topology) => t,
            None if (quads || rects) && class_of(built.topology) == 2 => {
                vk::PrimitiveTopology::TRIANGLE_LIST
            }
            None if (CONSOLE_TOPOLOGY..=CONSOLE_TOPOLOGY + 3).contains(&list.topology) => {
                let name = match list.topology {
                    17 => "rectangle list",
                    18 => "line loop",
                    _ => "quad strip",
                };
                return skip(Skip::State(format!(
                    "the {name} topology ({}) is not replayed",
                    list.topology
                )));
            }
            _ => {
                return skip(Skip::State(format!(
                    "topology {} for this pipeline",
                    list.topology
                )))
            }
        };
        let (stencil_ref, blend, viewport, scissor) =
            (list.stencil_ref, list.blend, list.viewport, list.scissor);
        let vbufs: Vec<_> = vbufs.into_iter().map(|(s, m, v)| (s, m, *v)).collect();
        // Quads draw as triangle lists through an index buffer: (buffer, indices).
        let quad_index = if quads {
            match (c, index) {
                (Command::Draw { vertices, .. }, _) => match vertices / 4 {
                    0 => return Ok(()),
                    q => Some((self.draw.quad_buffer(q)?, 6 * q)),
                },
                (
                    Command::DrawIndexed {
                        indices,
                        first_index,
                        ..
                    },
                    Some((_, ty)),
                ) => {
                    let Some(iv) = self.draw.list.index else {
                        return skip(Skip::State("no index buffer".into()));
                    };
                    let size = if ty == vk::IndexType::UINT16 { 2 } else { 4 };
                    let (first, n) = (u64::from(*first_index), u64::from(*indices));
                    if (first + n) * size > u64::from(iv.size) {
                        return skip(Skip::State(
                            "an indexed quad draw runs past its indices".into(),
                        ));
                    }
                    let q = *indices / 4;
                    if q == 0 {
                        return Ok(());
                    }
                    // SAFETY: the whole index view translated above, so its pages are
                    // committed and readable; the range is inside it.
                    let raw = unsafe {
                        std::slice::from_raw_parts(
                            (iv.address + first * size) as *const u8,
                            (n * size) as usize,
                        )
                    };
                    let read = |i: usize| -> u32 {
                        if size == 2 {
                            u32::from(u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]))
                        } else {
                            u32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().expect("4 bytes"))
                        }
                    };
                    let mut bytes = Vec::with_capacity(24 * q as usize);
                    for quad in 0..q as usize {
                        let v: [u32; 4] = std::array::from_fn(|k| read(4 * quad + k));
                        for k in [0, 1, 2, 0, 2, 3] {
                            bytes.extend(v[k].to_le_bytes());
                        }
                    }
                    let mut buf = HostBuf::new(self.draw.gpu, bytes.len())
                        .map_err(|e| Skip::State(format!("no quad index buffer: {e}")))?;
                    buf.write(0, &bytes);
                    let handle = buf.buffer;
                    self.draw.scratch.push(buf);
                    Some((handle, 6 * q))
                }
                _ => None,
            }
        } else {
            None
        };
        let Some((arena, at)) = self.draw.arena_slot() else {
            return skip(Skip::State("no room for the root block".into()));
        };
        arena.write(at, &block);
        let root_address = arena.address + at as u64;

        if self.inspecting() {
            self.inspect_draw(c, pd, &resources, extent, rects);
        }
        self.note_draw(c, pd.serial, resources.first().map(|t| t.resource));
        if self.skipped_by_filter() {
            return Ok(());
        }

        // Rendering.
        let want = Open {
            colors: color_views.iter().map(|c| c.1).collect(),
            depth: depth_att,
            extent,
        };
        if self.draw.open.as_ref() != Some(&want) {
            self.end_rendering(d, cb);
            let mut atts = vec![vk::RenderingAttachmentInfo::default(); built.color.len()];
            for &(i, view) in &color_views {
                atts[i] = vk::RenderingAttachmentInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE);
            }
            let depth_info = depth_att.map(|view| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)
            });
            let mut info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D {
                        width: extent.0,
                        height: extent.1,
                    },
                })
                .layer_count(1)
                .color_attachments(&atts);
            if let Some(a) = &depth_info {
                info = info.depth_attachment(a);
                if built
                    .depth
                    .is_some_and(|(_, i)| i.aspect.contains(vk::ImageAspectFlags::STENCIL))
                {
                    info = info.stencil_attachment(a);
                }
            }
            // SAFETY: `cb` is recording outside a rendering instance; the views are ours.
            unsafe { d.cmd_begin_rendering(cb, &info) };
            self.draw.open = Some(want);
        }

        let gfx = vk::PipelineBindPoint::GRAPHICS;
        let vp = viewport.map_or(
            vk::Viewport {
                x: 0.0,
                y: extent.1 as f32,
                width: extent.0 as f32,
                height: -(extent.1 as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            },
            |v| vk::Viewport {
                x: v[0],
                y: v[1] + v[3],
                width: v[2],
                height: -v[3],
                min_depth: v[4],
                max_depth: v[5],
            },
        );
        let sc = scissor.map_or(
            vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: extent.0,
                    height: extent.1,
                },
            },
            |r| vk::Rect2D {
                offset: vk::Offset2D {
                    x: r[0].max(0),
                    y: r[1].max(0),
                },
                extent: vk::Extent2D {
                    width: (r[2] - r[0].max(0)).max(0) as u32,
                    height: (r[3] - r[1].max(0)).max(0) as u32,
                },
            },
        );
        // SAFETY: `cb` is recording inside a rendering instance; every handle is live until
        // the submission completes (imports are only dropped between submissions).
        unsafe {
            d.cmd_bind_pipeline(cb, gfx, built.pipeline);
            let sets: Vec<vk::DescriptorSet> = [Some(set), built.layouts.set1]
                .into_iter()
                .flatten()
                .collect();
            d.cmd_bind_descriptor_sets(cb, gfx, layout, 0, &sets, &[]);
            d.cmd_push_constants(
                cb,
                layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                &root_address.to_le_bytes(),
            );
            d.cmd_set_viewport(cb, 0, &[vp]);
            d.cmd_set_scissor(cb, 0, &[sc]);
            d.cmd_set_blend_constants(cb, &blend.unwrap_or([1.0; 4]));
            d.cmd_set_stencil_reference(cb, vk::StencilFaceFlags::FRONT_AND_BACK, stencil_ref);
            d.cmd_set_primitive_topology(cb, topo);
            d.cmd_set_primitive_restart_enable(cb, built.strip_cut && is_strip(topo));
            for (slot, m, v) in &vbufs {
                d.cmd_bind_vertex_buffers2(
                    cb,
                    *slot,
                    &[m.buffer],
                    &[m.offset],
                    Some(&[u64::from(v.size)]),
                    Some(&[u64::from(v.stride)]),
                );
            }
            match (c, index) {
                (c, _) if quad_index.is_some() => {
                    let (buffer, count) = quad_index.expect("checked");
                    d.cmd_bind_index_buffer(cb, buffer, 0, vk::IndexType::UINT32);
                    let (instances, first_instance, base) = match c {
                        Command::Draw {
                            instances,
                            first_instance,
                            first_vertex,
                            ..
                        } => (instances, first_instance, *first_vertex as i32),
                        Command::DrawIndexed {
                            instances,
                            first_instance,
                            base_vertex,
                            ..
                        } => (instances, first_instance, *base_vertex),
                        _ => unreachable!("draw_command is called for draws"),
                    };
                    if base != 0 {
                        log_once("a quad draw with a non-zero base vertex".into());
                    }
                    d.cmd_draw_indexed(cb, count, *instances, 0, base, *first_instance);
                }
                (
                    Command::Draw {
                        vertices,
                        instances,
                        first_vertex,
                        first_instance,
                    },
                    _,
                ) => {
                    if *first_vertex != 0 {
                        log_once("a draw with a non-zero first vertex".into());
                    }
                    d.cmd_draw(cb, *vertices, *instances, *first_vertex, *first_instance);
                }
                (
                    Command::DrawIndexed {
                        indices,
                        instances,
                        first_index,
                        base_vertex,
                        first_instance,
                    },
                    Some((m, ty)),
                ) => {
                    if *base_vertex != 0 {
                        log_once("an indexed draw with a non-zero base vertex".into());
                    }
                    d.cmd_bind_index_buffer(cb, m.buffer, m.offset, ty);
                    d.cmd_draw_indexed(
                        cb,
                        *indices,
                        *instances,
                        *first_index,
                        *base_vertex,
                        *first_instance,
                    );
                }
                _ => unreachable!("draw_command is called for draws"),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::replay::tests::{dsv, rtv, texture};
    use super::super::tests::gpu_or_skip;
    use super::*;
    use crate::d3d12::command::RootView;
    use crate::d3d12::pipeline::{create_stream, PIPELINE_IIDS};
    use crate::shader::spirv::tests::stage;
    use nativ64::api_ms_win_core_memory as table;

    pub(in crate::backend::vk) const BLUE: [u8; 4] = [0, 0, 255, 255];

    /// Committed, writable title memory holding `bytes` (never freed by the test).
    pub(in crate::backend::vk) fn title_memory(bytes: &[u8]) -> u64 {
        let a = table::alloc(
            0,
            bytes.len().max(4096),
            table::MEM_RESERVE | table::MEM_COMMIT,
            0x04,
            None,
        )
        .unwrap();
        // SAFETY: just committed, at least `bytes.len()` long.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), a as *mut u8, bytes.len()) };
        a as u64
    }

    pub(in crate::backend::vk) fn floats(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// A subobject stream: VS, PS, blend, sample mask, rasterizer (culling none), depth-stencil
    /// (everything off), an optional one-element input layout (`POSITION`, `format`, slot 0),
    /// triangle topology and one render target in format 28.
    pub(in crate::backend::vk) fn stream(
        vs: &[u8],
        ps: &[u8],
        input: Option<u32>,
        depth: bool,
    ) -> Vec<u8> {
        let mut s = Vec::new();
        let mut sub = |kind: u32, len: usize| {
            let at = s.len();
            s.resize(at + len, 0);
            s[at..at + 4].copy_from_slice(&kind.to_le_bytes());
            at
        };
        let mut puts: Vec<(usize, Vec<u8>)> = Vec::new();
        for (kind, bytes) in [(1u32, vs), (2, ps)] {
            let at = sub(kind, 24);
            puts.push((at + 8, (bytes.as_ptr() as u64).to_le_bytes().to_vec()));
            puts.push((at + 16, (bytes.len() as u64).to_le_bytes().to_vec()));
        }
        let at = sub(8, 336);
        puts.push((at + 12 + 36, vec![0xf]));
        let at = sub(9, 8);
        puts.push((at + 4, u32::MAX.to_le_bytes().to_vec()));
        let at = sub(10, 48);
        puts.push((at + 4, 3u32.to_le_bytes().to_vec())); // solid
        puts.push((at + 8, 1u32.to_le_bytes().to_vec())); // cull none
        puts.push((at + 28, 1u32.to_le_bytes().to_vec())); // depth clip
        let at = sub(11, 56);
        if depth {
            puts.push((at + 4, 1u32.to_le_bytes().to_vec())); // test on
            puts.push((at + 8, 1u32.to_le_bytes().to_vec())); // write all
            puts.push((at + 12, 2u32.to_le_bytes().to_vec())); // LESS
        }
        if let Some(format) = input {
            let el: &'static mut [u8; 32] = Box::leak(Box::new([0u8; 32]));
            el[0..8].copy_from_slice(&(c"POSITION".as_ptr() as u64).to_le_bytes());
            el[12..16].copy_from_slice(&format.to_le_bytes());
            let at = sub(12, 24);
            puts.push((at + 8, (el.as_ptr() as u64).to_le_bytes().to_vec()));
            puts.push((at + 16, 1u32.to_le_bytes().to_vec()));
        }
        let at = sub(14, 8);
        puts.push((at + 4, 3u32.to_le_bytes().to_vec()));
        let at = sub(15, 40);
        puts.push((at + 4, 28u32.to_le_bytes().to_vec()));
        puts.push((at + 36, 1u32.to_le_bytes().to_vec()));
        if depth {
            let at = sub(16, 8);
            puts.push((at + 4, 40u32.to_le_bytes().to_vec())); // D32_FLOAT
        }
        for (at, b) in puts {
            s[at..at + b.len()].copy_from_slice(&b);
        }
        s
    }

    /// A pipeline object from a stream (the shader blobs are leaked with it).
    pub(in crate::backend::vk) fn pipeline(vs: Vec<u8>, ps: Vec<u8>, input: Option<u32>) -> usize {
        pipeline_with(vs, ps, input, false)
    }

    /// As [`pipeline`], with a depth format of 40 and the depth test (less) on when `depth`.
    pub(in crate::backend::vk) fn pipeline_with(
        vs: Vec<u8>,
        ps: Vec<u8>,
        input: Option<u32>,
        depth: bool,
    ) -> usize {
        let (vs, ps): (&'static [u8], &'static [u8]) = (Vec::leak(vs), Vec::leak(ps));
        let s = Vec::leak(stream(vs, ps, input, depth));
        let desc = [s.len(), s.as_ptr() as usize];
        let mut out = std::ptr::null_mut();
        // SAFETY: a valid stream and out pointer.
        let hr = unsafe { create_stream(std::ptr::null_mut(), &desc, &PIPELINE_IIDS[0], &mut out) };
        assert_eq!(hr, 0);
        out as usize
    }

    pub(in crate::backend::vk) fn state(pipe: usize, target: u64) -> Vec<Command> {
        vec![
            Command::SetPipeline { pipeline: pipe },
            Command::RenderTargets {
                count: 1,
                handles: vec![target],
                single_range: false,
                depth: None,
            },
            Command::Topology(4),
        ]
    }

    pub(in crate::backend::vk) fn clear(handle: u64, color: [f32; 4]) -> Vec<Command> {
        vec![
            Command::ClearRenderTarget {
                handle,
                color,
                rects: vec![],
            },
            Command::Barriers(vec![]),
        ]
    }

    pub(in crate::backend::vk) fn constants(slot: u32, rgba: [f32; 4]) -> Command {
        Command::RootConstants {
            compute: false,
            slot,
            offset: 0,
            values: rgba.map(f32::to_bits).to_vec(),
        }
    }

    pub(in crate::backend::vk) fn draw(vertices: u32) -> Command {
        Command::Draw {
            vertices,
            instances: 1,
            first_vertex: 0,
            first_instance: 0,
        }
    }

    pub(in crate::backend::vk) fn pixel(
        r: &super::super::replay::Readback,
        x: usize,
        y: usize,
    ) -> [u8; 4] {
        let at = 4 * (y * r.width as usize + x);
        r.bytes[at..at + 4].try_into().unwrap()
    }

    /// The full-screen triangle as clip-space `float4`s.
    pub(in crate::backend::vk) const TRIANGLE: [f32; 12] = [
        -1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0,
    ];

    #[test]
    fn a_draw_pulls_vertices_from_a_root_buffer_and_honours_viewport_and_scissor() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        let mut b = VkBackend::new(&g);
        let target = texture(16, 8, 28, 1);
        let handle = rtv(target);
        let verts = title_memory(&floats(&TRIANGLE));
        let pipe = pipeline(
            stage::draw_vs_pulling_from_a_root_buffer(),
            stage::draw_ps_writing_root_constants(),
            None,
        );
        let mut list = clear(handle, [0.0, 0.0, 1.0, 1.0]);
        list.extend(state(pipe, handle));
        list.extend([
            // the top half of the target, and of that only the left 8 columns
            Command::Viewports(vec![[0.0, 0.0, 16.0, 4.0, 0.0, 1.0]]),
            Command::Scissors(vec![[0, 0, 8, 8]]),
            Command::RootView {
                compute: false,
                slot: 0,
                view: RootView::Srv,
                address: verts,
            },
            constants(1, [1.0, 0.0, 0.0, 1.0]),
            draw(3),
        ]);
        b.execute(&[list]);
        assert_eq!((b.stats.draws, b.stats.drawn), (1, 1), "{:?}", b.stats);
        assert_eq!(b.stats.draw_skips.total(), 0);
        let r = b.read_back(target).unwrap();
        assert_eq!(pixel(&r, 2, 1), [255, 0, 0, 255], "top left is drawn");
        assert_eq!(pixel(&r, 12, 1), BLUE, "top right is scissored out");
        assert_eq!(
            pixel(&r, 2, 6),
            BLUE,
            "the bottom half is outside the viewport"
        );
        assert_eq!(pixel(&r, 7, 3), [255, 0, 0, 255]);
        assert_eq!(pixel(&r, 7, 4), BLUE);
        // a second draw on the same targets reuses the pipeline and the rendering instance
        let mut again = vec![
            Command::SetPipeline { pipeline: pipe },
            Command::Viewports(vec![[0.0, 4.0, 16.0, 4.0, 0.0, 1.0]]),
            Command::Scissors(vec![[0, 0, 16, 8]]),
            constants(1, [0.0, 1.0, 0.0, 1.0]),
            draw(3),
        ];
        let mut full = state(pipe, handle);
        full.append(&mut again);
        full.insert(
            3,
            Command::RootView {
                compute: false,
                slot: 0,
                view: RootView::Srv,
                address: verts,
            },
        );
        b.execute(&[full]);
        assert_eq!(b.stats.drawn, 2);
        assert_eq!(b.draw.pipelines.built(), 1);
        let r = b.read_back(target).unwrap();
        assert_eq!(
            pixel(&r, 12, 6),
            [0, 255, 0, 255],
            "the bottom half is now drawn"
        );
        assert_eq!(
            pixel(&r, 2, 1),
            [255, 0, 0, 255],
            "the top half keeps the first draw"
        );
    }

    #[test]
    fn an_indexed_draw_reads_a_vertex_buffer_with_a_base_vertex() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        let mut b = VkBackend::new(&g);
        let target = texture(16, 8, 28, 1);
        let handle = rtv(target);
        // a decoy vertex first: only base vertex 1 makes the triangle right
        let mut v = vec![0.0, 0.0, 0.0, 1.0];
        v.extend(TRIANGLE);
        let verts = title_memory(&floats(&v));
        let indices = title_memory(&[0, 0, 1, 0, 2, 0]);
        let pipe = pipeline(
            stage::draw_vs_with_a_position_input(),
            stage::draw_ps_writing_root_constants(),
            Some(2),
        );
        let mut list = clear(handle, [0.0, 0.0, 1.0, 1.0]);
        list.extend(state(pipe, handle));
        list.extend([
            Command::VertexBuffers {
                start: 0,
                count: 1,
                views: vec![VertexView {
                    address: verts,
                    size: 64,
                    stride: 16,
                }],
            },
            Command::IndexBuffer(Some(IndexView {
                address: indices,
                size: 6,
                format: 57,
            })),
            constants(1, [0.0, 1.0, 0.0, 1.0]),
            Command::DrawIndexed {
                indices: 3,
                instances: 1,
                first_index: 0,
                base_vertex: 1,
                first_instance: 0,
            },
        ]);
        b.execute(&[list]);
        assert_eq!(b.stats.drawn, 1, "{:?}", b.stats);
        let r = b.read_back(target).unwrap();
        for (x, y) in [(0, 0), (15, 0), (0, 7), (15, 7), (8, 4)] {
            assert_eq!(pixel(&r, x, y), [0, 255, 0, 255], "({x}, {y})");
        }
    }

    #[test]
    fn draws_that_cannot_be_replayed_are_skipped_and_counted_by_reason() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        let mut b = VkBackend::new(&g);
        let target = texture(16, 8, 28, 1);
        let handle = rtv(target);
        let verts = title_memory(&floats(&TRIANGLE));
        let pull = || stage::draw_vs_pulling_from_a_root_buffer();
        let good = pipeline(pull(), stage::draw_ps_writing_root_constants(), None);
        let images = pipeline(pull(), stage::draw_ps_sampling_a_texture(), None);
        let root = |address| Command::RootView {
            compute: false,
            slot: 0,
            view: RootView::Srv,
            address,
        };
        let not_memory = vec![0u8; 64];
        let mut list = clear(handle, [0.0, 0.0, 1.0, 1.0]);
        // no pipeline at all
        list.push(draw(3));
        // a pipeline that reads a texture through a descriptor table nobody set
        list.extend(state(images, handle));
        list.extend([root(verts), draw(3)]);
        // a root descriptor that is not GPU memory
        list.extend(state(good, handle));
        list.extend([root(not_memory.as_ptr() as u64), draw(3)]);
        // no render target bound
        list.extend([
            Command::RenderTargets {
                count: 0,
                handles: vec![],
                single_range: false,
                depth: None,
            },
            root(verts),
            draw(3),
        ]);
        // a topology of another class than the pipeline's
        list.extend(state(good, handle));
        list.extend([Command::Topology(2), root(verts), draw(3)]);
        b.execute(&[list]);
        let s = b.stats;
        assert_eq!((s.draws, s.drawn), (5, 0), "{s:?}");
        assert_eq!(
            (
                s.draw_skips.pipeline,
                s.draw_skips.memory,
                s.draw_skips.target,
                s.draw_skips.state
            ),
            (1, 2, 1, 1)
        );
        let r = b.read_back(target).unwrap();
        assert!(r.bytes.chunks(4).all(|p| p == BLUE), "nothing was drawn");
        drop(not_memory);
    }

    #[test]
    fn the_depth_test_keeps_the_nearer_draw() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        let mut b = VkBackend::new(&g);
        let target = texture(16, 8, 28, 1);
        let depth = texture(16, 8, 40, 2);
        let (handle, dhandle) = (rtv(target), dsv(depth));
        let at = |z: f32| {
            let mut t = TRIANGLE;
            for v in 0..3 {
                t[4 * v + 2] = z;
            }
            title_memory(&floats(&t))
        };
        let (near, far) = (at(0.25), at(0.75));
        let pipe = pipeline_with(
            stage::draw_vs_pulling_from_a_root_buffer(),
            stage::draw_ps_writing_root_constants(),
            None,
            true,
        );
        let root = |address| Command::RootView {
            compute: false,
            slot: 0,
            view: RootView::Srv,
            address,
        };
        let mut list = clear(handle, [0.0, 0.0, 1.0, 1.0]);
        list.push(Command::ClearDepthStencil {
            handle: dhandle,
            flags: 1,
            depth: 1.0,
            stencil: 0,
            rects: vec![],
        });
        list.push(Command::SetPipeline { pipeline: pipe });
        list.extend([
            Command::RenderTargets {
                count: 1,
                handles: vec![handle],
                single_range: false,
                depth: Some(dhandle),
            },
            Command::Topology(4),
            // the near triangle is red, then the far one (drawn second) is green
            root(near),
            constants(1, [1.0, 0.0, 0.0, 1.0]),
            draw(3),
            root(far),
            constants(1, [0.0, 1.0, 0.0, 1.0]),
            draw(3),
        ]);
        b.execute(&[list]);
        assert_eq!(b.stats.drawn, 2, "{:?}", b.stats);
        let r = b.read_back(target).unwrap();
        assert_eq!(
            pixel(&r, 8, 4),
            [255, 0, 0, 255],
            "the far draw failed the depth test"
        );
    }

    const QUAD: [f32; 16] = [
        -1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0,
    ];

    fn quad_list(pipe: usize, handle: u64, verts: u64, topology: u32) -> Vec<Command> {
        let mut list = clear(handle, [0.0, 0.0, 1.0, 1.0]);
        list.extend(state(pipe, handle));
        list.extend([
            Command::Topology(topology),
            Command::RootView {
                compute: false,
                slot: 0,
                view: RootView::Srv,
                address: verts,
            },
            constants(1, [1.0, 0.0, 0.0, 1.0]),
        ]);
        list
    }

    fn quad_pipeline() -> usize {
        pipeline(
            stage::draw_vs_pulling_from_a_root_buffer(),
            stage::draw_ps_writing_root_constants(),
            None,
        )
    }

    #[test]
    fn a_quad_draw_covers_the_target_as_two_triangles() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        let mut b = VkBackend::new(&g);
        let target = texture(16, 8, 28, 1);
        let handle = rtv(target);
        let verts = title_memory(&floats(&QUAD));
        let mut list = quad_list(quad_pipeline(), handle, verts, 19);
        // five vertices: one quad and a leftover; the same count again reuses the index buffer
        list.extend([draw(5), draw(5)]);
        b.execute(&[list]);
        assert_eq!(b.stats.drawn, 2, "{:?}", b.stats);
        assert_eq!(b.draw.quad_indices.len(), 1);
        let r = b.read_back(target).unwrap();
        for (x, y) in [(0, 0), (15, 0), (0, 7), (15, 7), (8, 4), (3, 6), (12, 1)] {
            assert_eq!(pixel(&r, x, y), [255, 0, 0, 255], "({x}, {y})");
        }
    }

    #[test]
    fn an_indexed_quad_draw_rewrites_the_indices_it_reads() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        for wide in [false, true] {
            let mut b = VkBackend::new(&g);
            let target = texture(16, 8, 28, 1);
            let handle = rtv(target);
            let verts = title_memory(&floats(&QUAD));
            // two decoy indices, then the quad's corners; read from the third
            let idx: Vec<u8> = if wide {
                [9u32, 9, 0, 1, 2, 3]
                    .iter()
                    .flat_map(|i| i.to_le_bytes())
                    .collect()
            } else {
                [9u16, 9, 0, 1, 2, 3]
                    .iter()
                    .flat_map(|i| i.to_le_bytes())
                    .collect()
            };
            let indices = title_memory(&idx);
            let mut list = quad_list(quad_pipeline(), handle, verts, 19);
            list.extend([
                Command::IndexBuffer(Some(IndexView {
                    address: indices,
                    size: idx.len() as u32,
                    format: if wide { 42 } else { 57 },
                })),
                Command::DrawIndexed {
                    indices: 4,
                    instances: 1,
                    first_index: 2,
                    base_vertex: 0,
                    first_instance: 0,
                },
            ]);
            b.execute(&[list]);
            assert_eq!(b.stats.drawn, 1, "{:?}", b.stats);
            let r = b.read_back(target).unwrap();
            for (x, y) in [(0, 0), (15, 0), (0, 7), (15, 7), (8, 4)] {
                assert_eq!(pixel(&r, x, y), [255, 0, 0, 255], "wide {wide} ({x}, {y})");
            }
        }
    }

    #[test]
    fn line_loops_and_quad_strips_are_refused_by_name() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        let mut b = VkBackend::new(&g);
        let target = texture(16, 8, 28, 1);
        let handle = rtv(target);
        let verts = title_memory(&floats(&QUAD));
        let pipe = quad_pipeline();
        let mut list = quad_list(pipe, handle, verts, 18);
        list.push(draw(4));
        list.extend([Command::Topology(20), draw(4)]);
        b.execute(&[list]);
        assert_eq!((b.stats.draws, b.stats.drawn), (2, 0), "{:?}", b.stats);
        assert_eq!(b.stats.draw_skips.state, 2);
        let r = b.read_back(target).unwrap();
        assert!(r.bytes.chunks(4).all(|p| p == BLUE), "nothing was drawn");
    }

    /// Three corners of a rectangle with the right angle in the middle of the list.
    const RECT: [f32; 12] = [
        1.0, 1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 1.0,
    ];

    #[test]
    fn a_rectangle_list_is_completed_by_the_geometry_stage() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        if !g.caps.geometry {
            return;
        }
        let mut b = VkBackend::new(&g);
        let target = texture(16, 8, 28, 1);
        let handle = rtv(target);
        let verts = title_memory(&floats(&RECT));
        let pipe = pipeline(
            stage::draw_vs_pulling_with_outputs(),
            stage::draw_ps_reading_the_outputs(),
            None,
        );
        let mut list = quad_list(pipe, handle, verts, 17);
        list.push(draw(3));
        b.execute(&[list]);
        assert_eq!((b.stats.draws, b.stats.drawn), (1, 1), "{:?}", b.stats);
        let r = b.read_back(target).unwrap();
        // the colour is the position over 0..1 and the flat blue is on, everywhere: the
        // corner the geometry stage made (top right, v3) is as covered as the others
        for (x, y) in [(0, 0), (15, 0), (0, 7), (15, 7), (8, 4), (13, 1), (2, 6)] {
            let p = pixel(&r, x, y);
            let (wx, wy) = ((x as f32 + 0.5) / 16.0, 1.0 - (y as f32 + 0.5) / 8.0);
            let near = |got: u8, want: f32| (f32::from(got) - want * 255.0).abs() <= 3.0;
            assert!(
                near(p[0], wx) && near(p[1], wy) && p[2] == 255 && p[3] == 255,
                "({x}, {y}): {p:?} wants x {wx} y {wy}"
            );
        }
    }

    #[test]
    fn a_rectangle_list_covers_the_target_wherever_its_right_angle_is() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        if !g.caps.geometry {
            return;
        }
        let corners = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]];
        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let mut b = VkBackend::new(&g);
            let target = texture(16, 8, 28, 1);
            let handle = rtv(target);
            let flat: Vec<f32> = order
                .iter()
                .flat_map(|&i| [corners[i][0], corners[i][1], 0.0, 1.0])
                .collect();
            let verts = title_memory(&floats(&flat));
            let pipe = pipeline(
                stage::draw_vs_pulling_with_outputs(),
                stage::draw_ps_reading_the_outputs(),
                None,
            );
            let mut list = quad_list(pipe, handle, verts, 17);
            list.push(draw(3));
            b.execute(&[list]);
            assert_eq!(b.stats.drawn, 1, "{order:?}: {:?}", b.stats);
            let r = b.read_back(target).unwrap();
            for (x, y) in [(0, 0), (15, 0), (0, 7), (15, 7), (8, 4)] {
                assert_eq!(pixel(&r, x, y)[2], 255, "{order:?} ({x}, {y})");
            }
        }
    }

    #[test]
    fn a_rectangle_list_is_a_skip_by_name_without_geometry_shaders() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        if g.caps.geometry {
            return;
        }
        let mut b = VkBackend::new(&g);
        let target = texture(16, 8, 28, 1);
        let handle = rtv(target);
        let verts = title_memory(&floats(&RECT));
        let mut list = quad_list(quad_pipeline(), handle, verts, 17);
        list.push(draw(3));
        b.execute(&[list]);
        assert_eq!(b.stats.draw_skips.pipeline, 1, "{:?}", b.stats);
    }
}
