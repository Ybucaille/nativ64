//! Replay of the title's recorded command streams (G3): render targets and textures as
//! device-local images made on first use, clears, copies and barriers executed, draws replayed
//! when they can be (`draw.rs`, chunk 10a), compute dispatches likewise (`compute.rs`, chunk
//! 10c), mesh dispatches and indirect draws counted and skipped. Images are always in `GENERAL`
//! layout, so a title barrier is one conservative global memory barrier, never a layout change.
//!
//! Buffers are the title's own memory, so buffer copies are plain CPU copies at record time,
//! in order with the staging of uploads (an upload reads its source bytes while recording).
//! Every submission waits for the GPU to be idle.
//!
//! Chunk 10b: an image is owned by the host (its content is the title's memory, uploaded by
//! `textures.rs` when it is sampled or copied from) until it has been a render or depth target,
//! a clear or copy destination (then the GPU's: its host memory is stale and never read again).

use super::draw::ClearValue;
use super::format::{self, FormatInfo};
use super::imgheap::{ImageHeap, ImageMem};
use super::textures::TexState;
use super::{log_once, vk_try, Gpu, Result, VkError};
use crate::d3d12::command::{Command, CopyLocation, Rect};
use crate::d3d12::descriptor::{depth_stencil_view, render_target_view};
use crate::d3d12::resource::{info_of, ResourceDesc};
use crate::texture::Layout;
use ash::{vk, Device};
use std::collections::{HashMap, HashSet};
use std::mem::Discriminant;

/// Whether `size` bytes at `at` are committed pages of the title's memory.
fn committed(at: usize, size: usize) -> bool {
    size == 0 || nativ64::api_ms_win_core_memory::is_committed(at, size)
}

/// `D3D12_RESOURCE_DIMENSION_BUFFER` and `..._TEXTURE2D`.
const BUFFER: u32 = 1;
const TEXTURE1D: u32 = 2;
const TEXTURE2D: u32 = 3;
const TEXTURE3D: u32 = 4;

/// A device-local image standing for one title resource.
pub(super) struct TitleImage {
    pub(super) image: vk::Image,
    pub(super) memory: ImageMem,
    pub(super) desc: ResourceDesc,
    pub(super) info: FormatInfo,
    /// The title memory the resource covers (its GPU address).
    pub(super) address: u64,
    /// The console layout of the host memory; None when `texture/` does not handle it.
    pub(super) layout: Option<Layout>,
    /// The GPU wrote the image: the host memory is stale and never read again.
    pub(super) gpu_owned: bool,
    /// Hash of the host memory at the last upload.
    pub(super) uploaded: Option<u64>,
    /// How many times it was uploaded.
    pub(super) uploads: u32,
    /// The submission whose host memory was last looked at (hashed).
    pub(super) seen: u64,
    /// The submission that last used the image (eviction spares the current one's).
    pub(super) touched: u64,
    /// Bytes of device memory (`vkGetImageMemoryRequirements`).
    pub(super) size: u64,
    /// What the write watch told this image's pages last.
    pub(super) watched: super::watch::Cursor,
    /// The sampled views made of the image, by description (storage views too, with the
    /// top bit of the first word set).
    pub(super) views: HashMap<super::textures::TexKey, vk::ImageView>,
    /// The image has `STORAGE` usage (unordered access, chunk 10c).
    pub(super) storage: bool,
}

/// The copyable part of a `TitleImage`, used while recording.
#[derive(Clone, Copy)]
pub(super) struct Img {
    pub(super) image: vk::Image,
    pub(super) info: FormatInfo,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) mips: u32,
    /// Array layers (1 for a 3D image).
    pub(super) layers: u32,
    /// Depth of a 3D image, 1 otherwise.
    pub(super) depth: u32,
}

/// Mip levels of a texture description (0 means the whole chain).
pub(super) fn mip_count(desc: &ResourceDesc) -> u32 {
    match u32::from(desc.mip_levels) {
        0 => {
            let mut biggest = (desc.width as u32).max(desc.height);
            if desc.dimension == TEXTURE3D {
                biggest = biggest.max(u32::from(desc.depth_or_array_size));
            }
            32 - biggest.max(1).leading_zeros()
        }
        n => n,
    }
}

impl TitleImage {
    fn img(&self) -> Img {
        let n = u32::from(self.desc.depth_or_array_size).max(1);
        let volume = self.desc.dimension == TEXTURE3D;
        Img {
            image: self.image,
            info: self.info,
            width: self.desc.width as u32,
            height: self.desc.height,
            mips: mip_count(&self.desc),
            layers: if volume { 1 } else { n },
            depth: if volume { n } else { 1 },
        }
    }
}

/// A host-visible buffer that lives until the submission using it is done.
pub(super) struct Staging {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

fn status_mib(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|t| {
            t.lines().find_map(|l| {
                l.strip_prefix(key)?
                    .trim()
                    .trim_end_matches("kB")
                    .trim()
                    .parse::<u64>()
                    .ok()
            })
        })
        .unwrap_or(0)
        >> 10
}

/// The host memory of the process, for the memory line: anonymous and shared (memfd) resident
/// sizes, the pool's pages held by the title against what its file really holds, and what
/// returning the allocator's free pages to the system gave back (`malloc_trim`).
fn host_memory_note() -> String {
    let before = status_mib("RssAnon:");
    // SAFETY: plain glibc call; nothing else reads the arenas' free memory.
    unsafe { libc::malloc_trim(0) };
    let after = status_mib("RssAnon:");
    let (held, file) = crate::xmem::pool_usage();
    format!(
        "host: anon {after} MiB (trim freed {} MiB), shmem {} MiB, xmem pages {} MiB, xmem file {} MiB",
        before.saturating_sub(after),
        status_mib("RssShmem:"),
        held >> 20,
        file >> 20
    )
}

/// Frees staging buffers whose submission has completed (or failed before using them).
fn free_staging(gpu: &Gpu, staging: Vec<Staging>) {
    for s in staging {
        // SAFETY: see above.
        unsafe {
            gpu.device.destroy_buffer(s.buffer, None);
            gpu.device.free_memory(s.memory, None);
        }
    }
}

/// What a replay has seen so far (for logs and tests).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplayStats {
    pub lists: u64,
    pub commands: u64,
    pub clears: u64,
    pub copies: u64,
    /// `Draw`, `DrawIndexed` and `ExecuteIndirect` commands seen.
    pub draws: u64,
    /// Draws recorded into Vulkan (the rest are in `draw_skips`, or indirect).
    pub drawn: u64,
    pub draw_skips: super::draw::DrawSkips,
    /// `Dispatch`, `DispatchMesh` and `DispatchMeshX` commands seen.
    pub dispatches: u64,
    /// Compute dispatches recorded into Vulkan (the rest are in `dispatch_skips`, or mesh).
    pub dispatched: u64,
    pub dispatch_skips: super::compute::DispatchSkips,
    pub skipped: u64,
    /// Bytes of host texture memory hashed, and textures uploaded because the hash changed.
    pub hashed: u64,
    /// Textures not hashed because the write watch saw none of their pages written.
    pub watch_skips: u64,
    pub uploads: u64,
    /// Images and bytes of device memory alive for resources, and for placed views.
    pub images: u64,
    pub image_bytes: u64,
    pub placed: u64,
    pub placed_bytes: u64,
    /// Released resources drained, and host-owned images freed to make room.
    pub released: u64,
    pub evictions: u64,
}

/// A read-back image: `bytes` are rows of `width * bytes_per_texel`, top row first.
pub struct Readback {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub bytes: Vec<u8>,
}

/// The Vulkan replay of recorded streams, owning the images of the title's resources.
pub struct VkBackend<'g> {
    pub(super) gpu: &'g Gpu,
    pub(super) images: HashMap<usize, Option<TitleImage>>,
    /// Images of placed views (memory with a description, no resource), by address and
    /// description words.
    pub(super) placed: HashMap<super::textures::PlacedKey, Option<TitleImage>>,
    /// The device memory of `images` and `placed`.
    heap: ImageHeap,
    pub(super) tex: TexState,
    pub(super) draw: super::draw::DrawState<'g>,
    pub stats: ReplayStats,
    logged: HashSet<Discriminant<Command>>,
    pub(super) swapchains: HashMap<usize, Option<super::Swapchain<'g>>>,
    pub(super) dump: Option<std::path::PathBuf>,
    pub(super) frames: u64,
    /// `NATIV64_DUMP_DRAWS`: the snapshots of the frame being recorded.
    pub(super) dd: super::inspect::DrawDump,
    /// Tests: the most device memory the images may hold (a failing allocation beyond it).
    #[cfg(test)]
    pub(super) budget: Option<u64>,
    /// In tests: the bytes of images kept alive after a sync, in place of the device's.
    #[cfg(test)]
    pub(super) trim_budget: Option<u64>,
}

fn range(
    aspect: vk::ImageAspectFlags,
    mip: u32,
    mips: u32,
    layer: u32,
    layers: u32,
) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: mip,
        level_count: mips,
        base_array_layer: layer,
        layer_count: layers,
    }
}

/// One memory barrier between everything before and after (all stages and accesses).
pub(super) fn global_barrier(d: &Device, cb: vk::CommandBuffer) {
    let access = vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
    let b = [vk::MemoryBarrier::default()
        .src_access_mask(access)
        .dst_access_mask(access)];
    // SAFETY: `cb` is recording.
    unsafe {
        d.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &b,
            &[],
            &[],
        )
    };
}

pub(super) fn mip_size(v: u32, mip: u32) -> u32 {
    (v >> mip).max(1)
}

/// Whether one of `rects` (or the absence of any) covers the whole `w` x `h` area.
fn covers(rects: &[Rect], w: u32, h: u32) -> bool {
    rects.is_empty()
        || rects
            .iter()
            .any(|r| r[0] <= 0 && r[1] <= 0 && r[2] >= w as i32 && r[3] >= h as i32)
}

/// The mips and layers a render or depth view names, from its description words
/// (`dimension_at` is where `ViewDimension` sits, `tex2d` and `array` the dimension values).
pub(super) fn view_range(img: Img, desc: Option<[u32; 6]>, depth: bool) -> (u32, u32, u32) {
    let Some(w) = desc else {
        return (0, 0, img.layers);
    };
    let (tex2d, array, base) = if depth { (3, 4, 3) } else { (4, 5, 2) };
    let v = w[1];
    if v == tex2d {
        (w[base].min(img.mips - 1), 0, 1)
    } else if v == array {
        let first = w[base + 1].min(img.layers - 1);
        let n = w[base + 2].min(img.layers - first).max(1);
        (w[base].min(img.mips - 1), first, n)
    } else {
        (0, 0, img.layers)
    }
}

/// A host-visible staging buffer holding `data`, kept in `staging` until the submission is done.
pub(super) fn stage_bytes(g: &Gpu, data: &[u8], staging: &mut Vec<Staging>) -> Result<vk::Buffer> {
    let d = &g.device;
    // SAFETY: handles are destroyed in `execute` or right here on failure.
    unsafe {
        let buffer = vk_try(
            "vkCreateBuffer",
            d.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(data.len().max(1) as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            ),
        )?;
        let req = d.get_buffer_memory_requirements(buffer);
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let Some(ty) = g.memory_type(req.memory_type_bits, flags) else {
            d.destroy_buffer(buffer, None);
            return Err(VkError::NoMemoryType);
        };
        let memory = match d.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(ty),
            None,
        ) {
            Ok(m) => m,
            Err(e) => {
                d.destroy_buffer(buffer, None);
                return Err(VkError::Vk("vkAllocateMemory", e));
            }
        };
        let filled = d
            .bind_buffer_memory(buffer, memory, 0)
            .and_then(|_| d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()))
            .map(|p| {
                std::ptr::copy_nonoverlapping(data.as_ptr(), p.cast::<u8>(), data.len());
                d.unmap_memory(memory);
            });
        if let Err(e) = filled {
            d.destroy_buffer(buffer, None);
            d.free_memory(memory, None);
            return Err(VkError::Vk("staging", e));
        }
        staging.push(Staging { buffer, memory });
        Ok(buffer)
    }
}

impl<'g> VkBackend<'g> {
    pub fn new(gpu: &'g Gpu) -> Self {
        VkBackend {
            gpu,
            images: HashMap::new(),
            placed: HashMap::new(),
            tex: TexState::default(),
            draw: super::draw::DrawState::new(gpu),
            stats: ReplayStats::default(),
            logged: HashSet::new(),
            swapchains: HashMap::new(),
            dump: None,
            frames: 0,
            dd: Default::default(),
            #[cfg(test)]
            budget: None,
            heap: ImageHeap::default(),
            #[cfg(test)]
            trim_budget: None,
        }
    }

    /// Replay `lists` in order and wait for the GPU to be idle.
    pub fn execute(&mut self, lists: &[Vec<Command>]) {
        let gpu = self.gpu;
        let t0 = std::time::Instant::now();
        super::profile::exec_take();
        let _call = super::profile::call(false, super::profile::Step::Begin);
        let mut staging = Vec::new();
        self.reap();
        // draws and dispatches: each needs a root block, and the resources they read
        let draws = lists
            .iter()
            .flatten()
            .filter(|c| {
                matches!(
                    c,
                    Command::Draw { .. } | Command::DrawIndexed { .. } | Command::Dispatch { .. }
                )
            })
            .count();
        self.draw.begin(draws);
        let changed = self.draw.mem.changed().to_vec();
        self.drop_placed_over(&changed);
        self.tex.submission += 1;
        super::profile::count(super::profile::Count::Submissions, 1);
        let ts = super::profile::start();
        let t_sync = std::time::Instant::now();
        let mut passes = 0u32;
        let step = super::profile::step(super::profile::Step::Sync);
        if draws > 0 {
            // what the draws can read gets an image; the rest stays out of device memory
            let prep = super::profile::exec_timer(super::profile::Exec::PrepNs);
            self.tex.wanted = super::residency::scan(lists, &mut self.tex.reads);
            drop(prep);
            // The textures are synced in passes of at most `UPLOAD_CAP` staged bytes, each
            // submitted and waited for, so that the staging of a scene change's textures
            // does not pile up (pinned host memory) before the draws.
            self.tex.capped = true;
            loop {
                passes += 1;
                self.tex.staged = 0;
                self.tex.more = false;
                let r = gpu.submit_once(|d, cb| {
                    let t = super::profile::start();
                    self.sync_textures(d, cb);
                    super::profile::stop(super::profile::Phase::SyncRec, t);
                });
                self.tex.ring.end_submission(gpu);
                if let Err(e) = &r {
                    log_once(format!("texture sync failed: {e}"));
                }
                if !self.tex.more || r.is_err() {
                    break;
                }
            }
            self.tex.capped = false;
        }
        super::profile::stop(super::profile::Phase::Sync, ts);
        let sync = t_sync.elapsed();
        drop(step);
        let _step = super::profile::step(super::profile::Step::Record);
        let t_record = std::time::Instant::now();
        let tr = super::profile::start();
        let r = gpu.submit_once(|d, cb| {
            for list in lists {
                self.stats.lists += 1;
                self.draw.reset_list();
                for c in list {
                    self.stats.commands += 1;
                    if self.inspecting() {
                        self.inspect_command(c);
                    }
                    let tc = super::profile::start();
                    self.record(d, cb, c, &mut staging);
                    super::profile::stop(super::profile::Phase::Cmds, tc);
                    self.snap_after(d, cb, c);
                    if matches!(c, Command::Dispatch { .. }) {
                        super::profile::stop(super::profile::Phase::Disp, tc);
                    }
                    if matches!(
                        c,
                        Command::ClearRenderTarget { .. }
                            | Command::ClearDepthStencil { .. }
                            | Command::CopyTextureRegion { .. }
                            | Command::CopyResource { .. }
                    ) {
                        global_barrier(d, cb);
                    }
                }
                self.end_rendering(d, cb);
            }
        });
        if let Err(e) = r {
            log_once(format!("replay submission failed: {e}"));
        }
        self.finish_draw_dump();
        self.draw.end();
        free_staging(gpu, staging);
        self.tex.ring.end_submission(gpu);
        self.refresh_stats();
        super::profile::stop(super::profile::Phase::Record, tr);
        let total = t0.elapsed();
        if total >= super::profile::SLOW_EXECUTE {
            eprintln!(
                "nativ64-gpu: {}",
                super::profile::slow_line(
                    total,
                    sync,
                    passes,
                    t_record.elapsed(),
                    draws,
                    super::profile::exec_take()
                )
            );
        }
    }

    /// Image of `res`, made (and moved to `GENERAL`) at first use; None when the resource is
    /// not ours, not a 2D texture, or in a format without a mapping.
    /// The colour image of `res` when a replay already made one: image, width, height.
    pub(super) fn colour_image(&self, res: usize) -> Option<(vk::Image, u32, u32)> {
        let t = self.images.get(&res)?.as_ref()?;
        (t.info.aspect == vk::ImageAspectFlags::COLOR).then_some((
            t.image,
            t.desc.width as u32,
            t.desc.height,
        ))
    }

    pub(super) fn image(&mut self, d: &Device, cb: vk::CommandBuffer, res: usize) -> Option<Img> {
        // SAFETY: `res` was recorded by a list method from a live resource of ours.
        let (desc, address) = unsafe { info_of(res as *mut _) }?;
        let submission = self.tex.submission;
        if let Some(Some(t)) = self.images.get_mut(&res) {
            if t.desc == desc {
                t.touched = submission;
                return Some(t.img());
            }
        }
        if let Some(Some(old)) = self.images.remove(&res) {
            self.draw.forget_image(res);
            // the descriptors that held its views are written again
            self.tex.rewrite_res(res);
            // SAFETY: the GPU is idle between submissions; this one has not used it yet.
            unsafe { self.destroy(old) };
        }
        let made = self.make_image(d, cb, desc, address);
        let out = made.as_ref().map(TitleImage::img);
        self.images.insert(res, made);
        out
    }

    /// Marks the image of `res` as written by the GPU (a target, a clear or copy destination).
    pub(super) fn own(&mut self, res: usize) {
        if let Some(Some(t)) = self.images.get_mut(&res) {
            t.gpu_owned = true;
        }
    }

    /// The description of the image made for `res`, if there is one.
    pub(super) fn image_desc(&self, res: usize) -> Option<ResourceDesc> {
        self.images.get(&res)?.as_ref().map(|t| t.desc)
    }

    pub(super) fn make_image(
        &mut self,
        d: &Device,
        cb: vk::CommandBuffer,
        desc: ResourceDesc,
        address: u64,
    ) -> Option<TitleImage> {
        let _t = super::profile::exec_timer(super::profile::Exec::ImageNs);
        super::profile::exec_add(super::profile::Exec::Images, 1);
        if desc.dimension == TEXTURE1D {
            log_once("no image for a 1D texture".into());
            return None;
        }
        if desc.dimension != TEXTURE2D && desc.dimension != TEXTURE3D {
            log_once(format!(
                "no image for resource dimension {}",
                desc.dimension
            ));
            return None;
        }
        let Some(info) = format::lookup(desc.format, desc.flags) else {
            log_once(format!("no Vulkan format for DXGI format {}", desc.format));
            return None;
        };
        if info.compressed && !self.gpu.caps.bc {
            log_once("no image in a BC format: the device lacks textureCompressionBC".into());
            return None;
        }
        if desc.sample_count > 1 {
            log_once(format!(
                "no image for a multisampled texture ({} samples)",
                desc.sample_count
            ));
            return None;
        }
        let (Ok(w), h) = (u32::try_from(desc.width), desc.height) else {
            return None;
        };
        if w == 0 || h == 0 {
            return None;
        }
        let mut usage = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED;
        if desc.flags & format::FLAG_DEPTH_STENCIL != 0 {
            usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        } else if desc.flags & format::FLAG_RENDER_TARGET != 0 {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        // Unordered access (D3D12 flag 0x4) when the format, or the UNORM twin of an sRGB
        // one (storage views take it), supports storage images.
        let mut flags = vk::ImageCreateFlags::MUTABLE_FORMAT;
        let mut storage = false;
        if desc.flags & format::FLAG_UNORDERED != 0 && info.aspect == vk::ImageAspectFlags::COLOR {
            let twin_ok = super::pipeline::srgb_twin(desc.format)
                .and_then(|t| format::lookup(t, 0))
                .is_some_and(|t| self.gpu.storage_format(t.format));
            if !info.compressed && self.gpu.storage_format(info.format) {
                storage = true;
            } else if twin_ok {
                storage = true;
                flags |= vk::ImageCreateFlags::EXTENDED_USAGE;
            }
        }
        if storage {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        let n = u32::from(desc.depth_or_array_size).max(1);
        let volume = desc.dimension == TEXTURE3D;
        // views may reinterpret the format (typeless, sRGB); 2D arrays of 6 layers or a
        // multiple may be cubes
        if !volume && n % 6 == 0 && u64::from(h) == desc.width {
            flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }
        let ci = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(if volume {
                vk::ImageType::TYPE_3D
            } else {
                vk::ImageType::TYPE_2D
            })
            .format(info.format)
            .extent(vk::Extent3D {
                width: w,
                height: h,
                depth: if volume { n } else { 1 },
            })
            .mip_levels(mip_count(&desc))
            .array_layers(if volume { 1 } else { n })
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: objects are destroyed on each failure path.
        unsafe {
            let image = match d.create_image(&ci, None) {
                Ok(i) => i,
                Err(e) => {
                    log_once(format!("vkCreateImage for {desc:?}: {e}"));
                    return None;
                }
            };
            let req = d.get_image_memory_requirements(image);
            let mut mem = self.allocate_image_memory(d, &req);
            if mem.is_none() {
                let _t = super::profile::exec_timer(super::profile::Exec::EvictNs);
                // out of device memory: host-owned images not used by this submission are
                // freed (they upload again when next made), oldest first
                for victim in self.eviction_order() {
                    self.evict(victim);
                    mem = self.allocate_image_memory(d, &req);
                    if mem.is_some() {
                        break;
                    }
                }
            }
            let Some(memory) = mem else {
                log_once(format!("no memory for an image of {desc:?}"));
                d.destroy_image(image, None);
                return None;
            };
            if d.bind_image_memory(image, memory.memory, memory.offset)
                .is_err()
            {
                d.destroy_image(image, None);
                self.heap.free(d, memory);
                return None;
            }
            let b = [vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range(
                    info.aspect,
                    0,
                    vk::REMAINING_MIP_LEVELS,
                    0,
                    vk::REMAINING_ARRAY_LAYERS,
                ))];
            d.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &b,
            );
            let layout = match Layout::new(&desc) {
                Ok(l) => Some(l),
                Err(e) => {
                    log_once(format!("no host layout for {desc:?}: {e}"));
                    None
                }
            };
            Some(TitleImage {
                image,
                memory,
                desc,
                info,
                address,
                layout,
                gpu_owned: false,
                uploaded: None,
                uploads: 0,
                seen: 0,
                touched: self.tex.submission,
                size: req.size,
                watched: Default::default(),
                views: HashMap::new(),
                storage,
            })
        }
    }

    /// Device-local memory for an image, None when the device has none (or, in tests, when the
    /// budget is spent).
    fn allocate_image_memory(
        &mut self,
        d: &Device,
        req: &vk::MemoryRequirements,
    ) -> Option<ImageMem> {
        let _t = super::profile::exec_timer(super::profile::Exec::AllocNs);
        #[cfg(test)]
        if let Some(budget) = self.budget {
            let live: u64 = self
                .images
                .values()
                .chain(self.placed.values())
                .flatten()
                .map(|t| t.size)
                .sum();
            if live + req.size > budget {
                return None;
            }
        }
        let Some(ty) = self
            .gpu
            .memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        else {
            log_once(format!(
                "no device-local memory type in {:#x} for an image of {} bytes",
                req.memory_type_bits, req.size
            ));
            return None;
        };
        match self.heap.alloc(d, ty, req) {
            Ok(m) => Some(m),
            Err(e) => {
                log_once(format!(
                    "vkAllocateMemory of {} bytes failed: {e}",
                    req.size
                ));
                None
            }
        }
    }

    /// The counters of the images alive, from the maps.
    pub(super) fn refresh_stats(&mut self) {
        let (n, bytes) = self
            .images
            .values()
            .flatten()
            .fold((0, 0), |(n, b), t| (n + 1, b + t.size));
        self.stats.images = n;
        self.stats.image_bytes = bytes;
        let (n, bytes) = self
            .placed
            .values()
            .flatten()
            .fold((0, 0), |(n, b), t| (n + 1, b + t.size));
        self.stats.placed = n;
        self.stats.placed_bytes = bytes;
    }

    /// A line of memory counters (every 600 frames, and with `NATIV64_PROFILE`).
    pub(super) fn memory_line(&mut self) -> String {
        self.refresh_stats();
        let host = host_memory_note();
        let keys = self.tex.placed_keys();
        let addresses = keys.iter().map(|k| k.0).collect::<HashSet<_>>().len();
        let s = &self.stats;
        format!(
            "nativ64-gpu: frame {}: images {} ({} MiB), placed {} ({} MiB), arena {} KiB, \
             released {}, evictions {}, wanted {}, placed keys {} over {} addresses; {host}",
            self.frames,
            s.images,
            s.image_bytes >> 20,
            s.placed,
            s.placed_bytes >> 20,
            self.draw.arena_bytes() >> 10,
            s.released,
            s.evictions,
            self.tex.wanted.len(),
            keys.len(),
            addresses
        )
    }

    /// # Safety
    /// The GPU no longer uses the image.
    pub(super) unsafe fn destroy(&mut self, t: TitleImage) {
        for v in t.views.values() {
            self.gpu.device.destroy_image_view(*v, None);
        }
        self.gpu.device.destroy_image(t.image, None);
        self.heap.free(&self.gpu.device, t.memory);
    }

    /// Log (once per kind of command) that a command is not replayed, and count it.
    fn skip(&mut self, c: &Command, why: &str) {
        self.stats.skipped += 1;
        if self.logged.insert(std::mem::discriminant(c)) {
            let text = format!("{c:?}");
            let name = text.split([' ', '(', '{']).next().unwrap_or("?");
            log_once(format!("replay skips {name}: {why}"));
        }
    }

    fn staging(&self, data: &[u8], staging: &mut Vec<Staging>) -> Result<vk::Buffer> {
        stage_bytes(self.gpu, data, staging)
    }

    fn record(
        &mut self,
        d: &Device,
        cb: vk::CommandBuffer,
        c: &Command,
        staging: &mut Vec<Staging>,
    ) {
        use Command as C;
        self.draw.track(c);
        // Rendering stays open over draws and state changes; anything else ends it.
        if !matches!(
            c,
            C::Draw { .. }
                | C::DrawIndexed { .. }
                | C::SetPipeline { .. }
                | C::ClearState { .. }
                | C::Topology(_)
                | C::Viewports(_)
                | C::Scissors(_)
                | C::BlendFactor(_)
                | C::StencilRef(_)
                | C::DepthBounds { .. }
                | C::IndexBuffer(_)
                | C::VertexBuffers { .. }
                | C::StreamOutput { .. }
                | C::RenderTargets { .. }
                | C::Predication { .. }
                | C::Heaps(_)
                | C::RootSignature { .. }
                | C::RootTable { .. }
                | C::RootConstants { .. }
                | C::RootView { .. }
        ) {
            self.end_rendering(d, cb);
        }
        match c {
            C::Draw { .. } | C::DrawIndexed { .. } => {
                self.stats.draws += 1;
                self.draw_command(d, cb, c);
            }
            C::Dispatch { .. } => {
                self.stats.dispatches += 1;
                self.dispatch_command(d, cb, c);
            }
            C::DispatchMesh { .. } | C::DispatchMeshX { .. } => self.stats.dispatches += 1,
            C::ExecuteIndirect { .. } => {
                self.stats.draws += 1;
                self.stats.draw_skips.state += 1;
            }
            C::Barriers(_) | C::BarrierGroups(_) | C::Sync { .. } => global_barrier(d, cb),
            C::ClearRenderTarget {
                handle,
                color,
                rects,
            } => {
                let Some(v) = render_target_view(*handle) else {
                    return self.skip(c, "no render target view at the handle");
                };
                let Some(img) = self.image(d, cb, v.resource) else {
                    return self.skip(c, "the resource has no image");
                };
                self.own(v.resource);
                let desc = (v.described != 0).then_some(v.desc);
                let (mip, layer, n) = view_range(img, desc, false);
                if !covers(rects, mip_size(img.width, mip), mip_size(img.height, mip)) {
                    for l in layer..layer + n {
                        if !self.clear_rects(
                            d,
                            cb,
                            v.resource,
                            img,
                            (mip, l),
                            rects,
                            ClearValue::Color(*color),
                        ) {
                            return self.skip(c, "no view for a partial clear");
                        }
                    }
                    self.stats.clears += 1;
                    return;
                }
                // SAFETY: `cb` is recording; the image is in GENERAL.
                unsafe {
                    d.cmd_clear_color_image(
                        cb,
                        img.image,
                        vk::ImageLayout::GENERAL,
                        &vk::ClearColorValue { float32: *color },
                        &[range(vk::ImageAspectFlags::COLOR, mip, 1, layer, n)],
                    )
                };
                self.stats.clears += 1;
            }
            C::ClearDepthStencil {
                handle,
                flags,
                depth,
                stencil,
                rects,
            } => {
                let Some(v) = depth_stencil_view(*handle) else {
                    return self.skip(c, "no depth view at the handle");
                };
                let Some(img) = self.image(d, cb, v.resource) else {
                    return self.skip(c, "the resource has no image");
                };
                self.own(v.resource);
                let mut aspect = vk::ImageAspectFlags::empty();
                if flags & 1 != 0 {
                    aspect |= vk::ImageAspectFlags::DEPTH;
                }
                if flags & 2 != 0 {
                    aspect |= vk::ImageAspectFlags::STENCIL;
                }
                aspect &= img.info.aspect;
                if aspect.is_empty() {
                    return self.skip(c, "the image has none of the cleared aspects");
                }
                let desc = (v.described != 0).then_some(v.desc);
                let (mip, layer, n) = view_range(img, desc, true);
                if !covers(rects, mip_size(img.width, mip), mip_size(img.height, mip)) {
                    for l in layer..layer + n {
                        if !self.clear_rects(
                            d,
                            cb,
                            v.resource,
                            img,
                            (mip, l),
                            rects,
                            ClearValue::Depth {
                                depth: *depth,
                                stencil: *stencil,
                                do_depth: aspect.contains(vk::ImageAspectFlags::DEPTH),
                                do_stencil: aspect.contains(vk::ImageAspectFlags::STENCIL),
                            },
                        ) {
                            return self.skip(c, "no view for a partial clear");
                        }
                    }
                    self.stats.clears += 1;
                    return;
                }
                // SAFETY: as above.
                unsafe {
                    d.cmd_clear_depth_stencil_image(
                        cb,
                        img.image,
                        vk::ImageLayout::GENERAL,
                        &vk::ClearDepthStencilValue {
                            depth: *depth,
                            stencil: u32::from(*stencil),
                        },
                        &[range(aspect, mip, 1, layer, n)],
                    )
                };
                self.stats.clears += 1;
            }
            C::CopyBufferRegion {
                dst,
                dst_offset,
                src,
                src_offset,
                len,
            } => {
                // SAFETY: resources recorded from live objects of ours.
                let (a, b) = unsafe { (info_of(*dst as *mut _), info_of(*src as *mut _)) };
                let (Some((da, pa)), Some((db, pb))) = (a, b) else {
                    return self.skip(c, "not our resources");
                };
                let fits = |d: &ResourceDesc, off: u64| {
                    d.dimension == BUFFER && off.checked_add(*len).is_some_and(|e| e <= d.width)
                };
                if !fits(&da, *dst_offset) || !fits(&db, *src_offset) {
                    return self.skip(c, "outside the buffers");
                }
                // SAFETY: both ranges are inside buffers backed by title memory.
                unsafe {
                    std::ptr::copy(
                        (pb + src_offset) as *const u8,
                        (pa + dst_offset) as *mut u8,
                        *len as usize,
                    )
                };
                self.stats.copies += 1;
            }
            C::CopyResource { dst, src, .. } => self.copy_resource(d, cb, c, *dst, *src),
            C::CopyTextureRegion {
                dst,
                x,
                y,
                z,
                src,
                source_box,
                ..
            } => self.copy_texture(d, cb, staging, c, (*dst, *src), (*x, *y, *z), *source_box),
            // State only: nothing to replay without draws.
            C::SetPipeline { .. }
            | C::ClearState { .. }
            | C::Discard { .. }
            | C::Topology(_)
            | C::Viewports(_)
            | C::Scissors(_)
            | C::BlendFactor(_)
            | C::StencilRef(_)
            | C::DepthBounds { .. }
            | C::IndexBuffer(_)
            | C::VertexBuffers { .. }
            | C::StreamOutput { .. }
            | C::RenderTargets { .. }
            | C::Predication { .. }
            | C::Heaps(_)
            | C::RootSignature { .. }
            | C::RootTable { .. }
            | C::RootConstants { .. }
            | C::RootView { .. } => {}
            C::CopyMemory { dst, src, len, .. } => {
                let (Ok(d), Ok(s), Ok(n)) = (
                    usize::try_from(*dst),
                    usize::try_from(*src),
                    usize::try_from(*len),
                ) else {
                    return self.skip(c, "an address out of range");
                };
                if !committed(d, n) || !committed(s, n) {
                    return self.skip(c, "memory that is not committed");
                }
                // SAFETY: both ranges lie in committed title memory (checked above).
                unsafe { std::ptr::copy(s as *const u8, d as *mut u8, n) };
                self.stats.copies += 1;
            }
            C::FillMemory {
                address,
                size,
                pattern,
                width,
            } => {
                let (Ok(a), Ok(n)) = (usize::try_from(*address), usize::try_from(*size)) else {
                    return self.skip(c, "an address out of range");
                };
                let w = usize::from(*width).clamp(1, 16);
                if !committed(a, n) {
                    return self.skip(c, "memory that is not committed");
                }
                // SAFETY: the range lies in committed title memory (checked above).
                let dst = unsafe { std::slice::from_raw_parts_mut(a as *mut u8, n) };
                for chunk in dst.chunks_mut(w) {
                    let k = chunk.len();
                    chunk.copy_from_slice(&pattern[..k]);
                }
                self.stats.copies += 1;
            }
            C::ResolveSubresource { .. }
            | C::WriteBottom { .. }
            | C::WaitValue { .. }
            | C::ClearUav { .. }
            | C::BeginQuery { .. }
            | C::EndQuery { .. }
            | C::ResolveQueries { .. } => self.skip(c, "not replayed yet"),
        }
    }

    fn copy_resource(
        &mut self,
        d: &Device,
        cb: vk::CommandBuffer,
        c: &Command,
        dst: usize,
        src: usize,
    ) {
        // SAFETY: resources recorded from live objects of ours.
        let (a, b) = unsafe { (info_of(dst as *mut _), info_of(src as *mut _)) };
        let (Some((da, pa)), Some((db, pb))) = (a, b) else {
            return self.skip(c, "not our resources");
        };
        if da.dimension == BUFFER && db.dimension == BUFFER {
            let n = da.width.min(db.width) as usize;
            // SAFETY: both buffers are title memory of at least `n` bytes.
            unsafe { std::ptr::copy(pb as *const u8, pa as *mut u8, n) };
            self.stats.copies += 1;
            return;
        }
        let same = (
            da.dimension,
            da.width,
            da.height,
            da.mip_levels,
            da.depth_or_array_size,
        ) == (
            db.dimension,
            db.width,
            db.height,
            db.mip_levels,
            db.depth_or_array_size,
        ) && format::lookup(da.format, da.flags).map(|f| f.bytes)
            == format::lookup(db.format, db.flags).map(|f| f.bytes);
        if !same {
            return self.skip(c, "different textures");
        }
        let (Some(di), Some(si)) = (self.image(d, cb, dst), self.image(d, cb, src)) else {
            return self.skip(c, "the resource has no image");
        };
        if di.depth > 1 || si.depth > 1 {
            return self.skip(c, "a 3D texture copy");
        }
        self.host_to_image(d, cb, src);
        self.own(dst);
        let regions: Vec<vk::ImageCopy> = (0..si.mips)
            .map(|m| {
                let layers = vk::ImageSubresourceLayers {
                    aspect_mask: si.info.aspect & di.info.aspect,
                    mip_level: m,
                    base_array_layer: 0,
                    layer_count: si.layers,
                };
                vk::ImageCopy {
                    src_subresource: layers,
                    dst_subresource: layers,
                    extent: vk::Extent3D {
                        width: mip_size(si.width, m),
                        height: mip_size(si.height, m),
                        depth: 1,
                    },
                    ..Default::default()
                }
            })
            .collect();
        // SAFETY: `cb` is recording; both images are in GENERAL.
        unsafe {
            d.cmd_copy_image(
                cb,
                si.image,
                vk::ImageLayout::GENERAL,
                di.image,
                vk::ImageLayout::GENERAL,
                &regions,
            )
        };
        self.stats.copies += 1;
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_texture(
        &mut self,
        d: &Device,
        cb: vk::CommandBuffer,
        staging: &mut Vec<Staging>,
        c: &Command,
        (dst, src): (CopyLocation, CopyLocation),
        (x, y, z): (u32, u32, u32),
        source_box: Option<[u32; 6]>,
    ) {
        let CopyLocation::Subresource {
            resource: dres,
            index: dsub,
        } = dst
        else {
            return self.skip(c, "texture to buffer");
        };
        let Some(di) = self.image(d, cb, dres) else {
            return self.skip(c, "the destination has no image");
        };
        if di.depth > 1 {
            return self.skip(c, "a 3D texture copy");
        }
        self.own(dres);
        let (dmip, dlayer) = (dsub % di.mips, dsub / di.mips);
        if dlayer >= di.layers {
            return self.skip(c, "subresource out of range");
        }
        let layers = |mip, layer| vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: mip,
            base_array_layer: layer,
            layer_count: 1,
        };
        let offset = vk::Offset3D {
            x: x as i32,
            y: y as i32,
            z: z as i32,
        };
        match src {
            CopyLocation::Footprint {
                resource,
                offset: boff,
                width,
                height,
                depth,
                row_pitch,
                ..
            } => {
                if di.info.aspect != vk::ImageAspectFlags::COLOR
                    || di.info.compressed
                    || source_box.is_some_and(|b| b[0] != 0 || b[1] != 0 || b[2] != 0)
                {
                    return self.skip(c, "depth upload or offset source box");
                }
                // SAFETY: a resource recorded from a live object of ours.
                let Some((bd, base)) = (unsafe { info_of(resource as *mut _) }) else {
                    return self.skip(c, "the source is not ours");
                };
                let bpp = di.info.bytes;
                let (w, h) = source_box.map_or((width, height), |b| (b[3], b[4]));
                let fits = x + w <= mip_size(di.width, dmip) && y + h <= mip_size(di.height, dmip);
                if bd.dimension != BUFFER
                    || row_pitch == 0
                    || row_pitch % bpp != 0
                    || w == 0
                    || h == 0
                    || !fits
                    || depth != 1
                {
                    return self.skip(c, "unsupported footprint");
                }
                let len =
                    (u64::from(row_pitch) * u64::from(height)).min(bd.width.saturating_sub(boff));
                if len < u64::from(row_pitch) * u64::from(h - 1) + u64::from(w * bpp) {
                    return self.skip(c, "footprint outside the buffer");
                }
                // SAFETY: the range is inside a buffer backed by title memory.
                let bytes =
                    unsafe { std::slice::from_raw_parts((base + boff) as *const u8, len as usize) };
                let buffer = match self.staging(bytes, staging) {
                    Ok(b) => b,
                    Err(e) => return log_once(format!("staging failed: {e}")),
                };
                let region = [vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: row_pitch / bpp,
                    buffer_image_height: height,
                    image_subresource: layers(dmip, dlayer),
                    image_offset: offset,
                    image_extent: vk::Extent3D {
                        width: w,
                        height: h,
                        depth: 1,
                    },
                }];
                // SAFETY: `cb` is recording; the staging buffer lives until the submit ends.
                unsafe {
                    d.cmd_copy_buffer_to_image(
                        cb,
                        buffer,
                        di.image,
                        vk::ImageLayout::GENERAL,
                        &region,
                    )
                };
                self.stats.copies += 1;
            }
            CopyLocation::Subresource {
                resource: sres,
                index: ssub,
            } => {
                let Some(si) = self.image(d, cb, sres) else {
                    return self.skip(c, "the source has no image");
                };
                self.host_to_image(d, cb, sres);
                let (smip, slayer) = (ssub % si.mips, ssub / si.mips);
                let (sw, sh) = (mip_size(si.width, smip), mip_size(si.height, smip));
                let b = source_box.unwrap_or([0, 0, 0, sw, sh, 1]);
                let (w, h) = (b[3].saturating_sub(b[0]), b[4].saturating_sub(b[1]));
                let fits = b[3] <= sw
                    && b[4] <= sh
                    && x + w <= mip_size(di.width, dmip)
                    && y + h <= mip_size(di.height, dmip);
                let blocks = [x, y, b[0], b[1]].iter().any(|v| v % 4 != 0);
                if slayer >= si.layers
                    || si.depth > 1
                    || (di.info.compressed && blocks)
                    || si.info.bytes != di.info.bytes
                    || si.info.aspect != di.info.aspect
                    || w == 0
                    || h == 0
                    || !fits
                {
                    return self.skip(c, "unsupported texture copy");
                }
                let aspect = di.info.aspect;
                let sub = |mip, layer| vk::ImageSubresourceLayers {
                    aspect_mask: aspect,
                    ..layers(mip, layer)
                };
                let region = [vk::ImageCopy {
                    src_subresource: sub(smip, slayer),
                    src_offset: vk::Offset3D {
                        x: b[0] as i32,
                        y: b[1] as i32,
                        z: 0,
                    },
                    dst_subresource: sub(dmip, dlayer),
                    dst_offset: offset,
                    extent: vk::Extent3D {
                        width: w,
                        height: h,
                        depth: 1,
                    },
                }];
                // SAFETY: `cb` is recording; both images are in GENERAL.
                unsafe {
                    d.cmd_copy_image(
                        cb,
                        si.image,
                        vk::ImageLayout::GENERAL,
                        di.image,
                        vk::ImageLayout::GENERAL,
                        &region,
                    )
                };
                self.stats.copies += 1;
            }
            CopyLocation::Other { .. } => self.skip(c, "unknown source location"),
        }
    }

    /// Mip 0, layer 0 of the colour image made for `resource`, read back to the CPU. None when
    /// the resource has no colour image yet.
    pub fn read_back(&mut self, resource: usize) -> Option<Readback> {
        let t = self.images.get(&resource)?.as_ref()?;
        if t.info.aspect != vk::ImageAspectFlags::COLOR {
            return None;
        }
        let g = self.gpu;
        let (w, h, bytes) = (t.desc.width as u32, t.desc.height, t.info.bytes);
        let size = u64::from(w) * u64::from(h) * u64::from(bytes);
        let image = t.image;
        let d = &g.device;
        // SAFETY: the buffer and its memory are destroyed before returning; the copy has
        // completed when `submit_once` returns.
        unsafe {
            let buffer = d
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(vk::BufferUsageFlags::TRANSFER_DST),
                    None,
                )
                .ok()?;
            let req = d.get_buffer_memory_requirements(buffer);
            let flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let memory = g.memory_type(req.memory_type_bits, flags).and_then(|ty| {
                d.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(ty),
                    None,
                )
                .ok()
            });
            let Some(memory) = memory else {
                d.destroy_buffer(buffer, None);
                return None;
            };
            let mut out = None;
            if d.bind_buffer_memory(buffer, memory, 0).is_ok() {
                let region = [vk::BufferImageCopy {
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_extent: vk::Extent3D {
                        width: w,
                        height: h,
                        depth: 1,
                    },
                    ..Default::default()
                }];
                let done = g.submit_once(|d, cb| {
                    d.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::GENERAL, buffer, &region)
                });
                if done.is_ok() {
                    if let Ok(p) = d.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) {
                        let mut v = vec![0u8; size as usize];
                        std::ptr::copy_nonoverlapping(p.cast::<u8>(), v.as_mut_ptr(), v.len());
                        d.unmap_memory(memory);
                        out = Some(Readback {
                            width: w,
                            height: h,
                            format: t.info.format,
                            bytes: v,
                        });
                    }
                }
            }
            d.destroy_buffer(buffer, None);
            d.free_memory(memory, None);
            out
        }
    }
}

impl Drop for VkBackend<'_> {
    fn drop(&mut self) {
        // SAFETY: the GPU is idle after the wait; images are ours.
        unsafe {
            let _ = self.gpu.device.device_wait_idle();
            self.swapchains.clear();
            for (_, t) in std::mem::take(&mut self.images) {
                if let Some(t) = t {
                    self.destroy(t);
                }
            }
            for (_, t) in std::mem::take(&mut self.placed) {
                if let Some(t) = t {
                    self.destroy(t);
                }
            }
            self.heap.destroy(&self.gpu.device);
            self.tex.ring.destroy(self.gpu);
            self.tex.destroy(&self.gpu.device);
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::tests::gpu_or_skip;
    use super::*;
    use crate::d3d12::descriptor::{create_depth_stencil_view, create_render_target_view};
    use crate::d3d12::resource::create_at_address;
    use crate::xobj::IID_IUNKNOWN;
    use std::ffi::c_void;

    /// A texture resource over `mem` (the address is never read for textures).
    pub(in crate::backend::vk) fn texture(w: u64, h: u32, format: u32, flags: u32) -> usize {
        let desc = ResourceDesc {
            dimension: TEXTURE2D,
            alignment: 0,
            width: w,
            height: h,
            depth_or_array_size: 1,
            mip_levels: 1,
            format,
            sample_count: 1,
            sample_quality: 0,
            layout: 0,
            flags,
        };
        resource_at(0x1000, desc)
    }

    /// A resource of description `desc` over the memory at `address`.
    pub(in crate::backend::vk) fn resource_at(address: u64, desc: ResourceDesc) -> usize {
        let mut out: *mut c_void = std::ptr::null_mut();
        // SAFETY: `desc` and `out` are valid for the call.
        let hr = unsafe {
            create_at_address(
                std::ptr::null_mut(),
                address,
                &desc,
                0,
                std::ptr::null(),
                &IID_IUNKNOWN,
                &mut out,
            )
        };
        assert_eq!(hr, 0);
        out as usize
    }

    pub(in crate::backend::vk) fn buffer(mem: &mut [u8]) -> usize {
        let desc = ResourceDesc {
            dimension: BUFFER,
            width: mem.len() as u64,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            ..Default::default()
        };
        let mut out: *mut c_void = std::ptr::null_mut();
        // SAFETY: as above.
        let hr = unsafe {
            create_at_address(
                std::ptr::null_mut(),
                mem.as_mut_ptr() as u64,
                &desc,
                0,
                std::ptr::null(),
                &IID_IUNKNOWN,
                &mut out,
            )
        };
        assert_eq!(hr, 0);
        out as usize
    }

    /// An RTV slot for `res`; the slot's memory is leaked with the test.
    pub(in crate::backend::vk) fn rtv(res: usize) -> u64 {
        let slot = Box::leak(Box::new([0u8; 128])).as_mut_ptr() as u64;
        // SAFETY: a live resource and a 128-byte slot.
        unsafe {
            create_render_target_view(std::ptr::null_mut(), res as *mut _, std::ptr::null(), slot)
        };
        slot
    }

    pub(in crate::backend::vk) fn dsv(res: usize) -> u64 {
        let slot = Box::leak(Box::new([0u8; 128])).as_mut_ptr() as u64;
        // SAFETY: as above.
        unsafe {
            create_depth_stencil_view(std::ptr::null_mut(), res as *mut _, std::ptr::null(), slot)
        };
        slot
    }

    #[test]
    fn clear_then_read_back() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        let mut b = VkBackend::new(&g);
        let res = texture(16, 8, 28, 1);
        let handle = rtv(res);
        let clear = |color| Command::ClearRenderTarget {
            handle,
            color,
            rects: vec![],
        };
        b.execute(&[vec![
            Command::Draw {
                vertices: 3,
                instances: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            clear([1.0, 0.0, 0.2, 1.0]),
            Command::Barriers(vec![]),
        ]]);
        assert_eq!(b.stats.clears, 1);
        assert_eq!(b.stats.draws, 1);
        let r = b.read_back(res).unwrap();
        assert_eq!((r.width, r.height), (16, 8));
        assert!(r.bytes.chunks(4).all(|p| p == [255, 0, 51, 255]));
        // A second clear on the same image replaces the contents.
        b.execute(&[vec![clear([0.0, 1.0, 0.0, 1.0])]]);
        let r = b.read_back(res).unwrap();
        assert!(r.bytes.chunks(4).all(|p| p == [0, 255, 0, 255]));
    }

    #[test]
    fn memory_copies_and_fills_run_on_committed_memory() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        use nativ64::api_ms_win_core_memory as table;
        let Some(g) = gpu_or_skip() else { return };
        let mut b = VkBackend::new(&g);
        let a = table::alloc(
            0,
            0x1000,
            table::MEM_RESERVE | table::MEM_COMMIT,
            0x04,
            None,
        )
        .unwrap();
        // SAFETY: the page was just committed read-write.
        let page = unsafe { std::slice::from_raw_parts_mut(a as *mut u8, 0x1000) };
        page[..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let mut p = [0u8; 16];
        p[..4].copy_from_slice(&[9, 8, 7, 6]);
        b.execute(&[vec![
            Command::CopyMemory {
                dst: a as u64 + 0x100,
                src: a as u64,
                len: 8,
                flags: 0,
            },
            Command::FillMemory {
                address: a as u64 + 0x200,
                size: 10,
                pattern: p,
                width: 4,
            },
            // not committed: skipped, not a crash
            Command::CopyMemory {
                dst: 0x1000,
                src: a as u64,
                len: 8,
                flags: 0,
            },
        ]]);
        assert_eq!(&page[0x100..0x108], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(&page[0x200..0x20a], &[9, 8, 7, 6, 9, 8, 7, 6, 9, 8]);
        assert_eq!(b.stats.copies, 2);
        assert_eq!(b.stats.skipped, 1);
    }

    #[test]
    fn partial_rectangles_clear_only_inside() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        let mut b = VkBackend::new(&g);
        let res = texture(16, 8, 28, 1);
        b.execute(&[vec![
            Command::ClearRenderTarget {
                handle: rtv(res),
                color: [0.0; 4],
                rects: vec![],
            },
            Command::ClearRenderTarget {
                handle: rtv(res),
                color: [1.0; 4],
                rects: vec![[0, 0, 4, 4], [8, 6, 100, 100]],
            },
        ]]);
        assert_eq!((b.stats.clears, b.stats.skipped), (2, 0));
        let px = b.read_back(res).unwrap().bytes;
        let at = |x: usize, y: usize| px[(y * 16 + x) * 4];
        assert_eq!((at(0, 0), at(3, 3), at(4, 0), at(0, 4)), (255, 255, 0, 0));
        assert_eq!((at(8, 6), at(15, 7), at(7, 6), at(8, 5)), (255, 255, 0, 0));
    }

    #[test]
    fn depth_clear_and_unknown_view() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        let mut b = VkBackend::new(&g);
        let res = texture(8, 8, 40, 2);
        b.execute(&[vec![
            Command::ClearDepthStencil {
                handle: dsv(res),
                flags: 1,
                depth: 0.5,
                stencil: 0,
                rects: vec![],
            },
            Command::ClearRenderTarget {
                handle: Box::leak(Box::new([0u8; 128])).as_mut_ptr() as u64,
                color: [0.0; 4],
                rects: vec![],
            },
        ]]);
        assert_eq!((b.stats.clears, b.stats.skipped), (1, 1));
    }

    #[test]
    fn buffer_upload_copy_and_texture_copy() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        let mut b = VkBackend::new(&g);
        // 4x2 texels, row pitch 32 bytes (8 texels) in the buffer.
        let mut mem = vec![0u8; 64];
        for r in 0..2usize {
            for t in 0..4usize {
                mem[r * 32 + t * 4..][..4].copy_from_slice(&[r as u8 + 1, t as u8, 7, 255]);
            }
        }
        let src_buf = buffer(&mut mem);
        let mut mem2 = vec![0u8; 64];
        let dst_buf = buffer(&mut mem2);
        let tex = texture(4, 2, 28, 0);
        let copy = tex_copy(tex, src_buf);
        let tex2 = texture(4, 2, 28, 0);
        b.execute(&[vec![
            Command::CopyBufferRegion {
                dst: dst_buf,
                dst_offset: 0,
                src: src_buf,
                src_offset: 0,
                len: 64,
            },
            copy,
            Command::CopyResource {
                dst: tex2,
                src: tex,
                flags: 0,
            },
        ]]);
        assert_eq!(mem2, mem);
        assert_eq!(b.stats.copies, 3);
        for t in [tex, tex2] {
            let r = b.read_back(t).unwrap();
            assert_eq!(r.bytes.len(), 32);
            assert_eq!(&r.bytes[0..4], &[1, 0, 7, 255]);
            assert_eq!(&r.bytes[16..20], &[2, 0, 7, 255]);
            assert_eq!(&r.bytes[28..32], &[2, 3, 7, 255]);
        }
    }

    #[test]
    fn present_dumps_a_png() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        let mut b = VkBackend::new(&g);
        let dir = std::env::temp_dir().join(format!("nativ64-dump-{}", std::process::id()));
        b.set_dump_dir(dir.clone());
        let res = texture(6, 4, 28, 1);
        b.execute(&[vec![Command::ClearRenderTarget {
            handle: rtv(res),
            color: [0.0, 0.0, 1.0, 1.0],
            rects: vec![],
        }]]);
        b.present(res, None, None);
        b.present(res, Some([0, 0, 2, 2]), None);
        assert_eq!(b.frames(), 2);
        let file = std::fs::File::open(dir.join("frame_00001.png")).unwrap();
        let mut reader = png::Decoder::new(std::io::BufReader::new(file))
            .read_info()
            .unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!((info.width, info.height), (6, 4));
        assert!(buf[..96].chunks(4).all(|p| p == [0, 0, 255, 255]));
        assert!(dir.join("frame_00002.png").exists());
        std::fs::remove_dir_all(dir).unwrap();
        // A resource nothing rendered to presents nothing and dumps nothing.
        let n = b.frames();
        b.present(texture(4, 4, 28, 1), None, None);
        assert_eq!(b.frames(), n + 1);
    }

    fn rss_kb() -> u64 {
        let s = std::fs::read_to_string("/proc/self/statm").unwrap();
        s.split_whitespace().nth(1).unwrap().parse::<u64>().unwrap() * 4
    }

    /// Frames of a title: clear a 4K back buffer, present it. The resident size must not grow
    /// with the frame count (images are reused, per-frame buffers freed).
    #[test]
    fn long_run_keeps_a_stable_footprint() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        let mut b = VkBackend::new(&g);
        let dir = std::env::temp_dir().join(format!("nativ64-rss-{}", std::process::id()));
        b.set_dump_dir(dir.clone());
        let back = [texture(3840, 2160, 28, 1), texture(3840, 2160, 28, 1)];
        let views = back.map(rtv);
        let mut frame = |i: usize| {
            b.execute(&[vec![Command::ClearRenderTarget {
                handle: views[i % 2],
                color: [0.0; 4],
                rects: vec![],
            }]]);
            b.present(back[i % 2], None, None);
        };
        for i in 0..8 {
            frame(i);
        }
        let before = rss_kb();
        for i in 8..136 {
            frame(i);
        }
        let grown = rss_kb().saturating_sub(before);
        let _ = std::fs::remove_dir_all(dir);
        assert!(
            grown < 200_000,
            "resident size grew by {grown} kB over 128 frames"
        );
    }

    fn tex_copy(tex: usize, buf: usize) -> Command {
        Command::CopyTextureRegion {
            dst: CopyLocation::Subresource {
                resource: tex,
                index: 0,
            },
            x: 0,
            y: 0,
            z: 0,
            src: CopyLocation::Footprint {
                resource: buf,
                offset: 0,
                format: 28,
                width: 4,
                height: 2,
                depth: 1,
                row_pitch: 32,
            },
            source_box: None,
            flags: 0,
        }
    }
}
