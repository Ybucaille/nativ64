//! The bindless descriptor set, the view table and the per-submission arena of root blocks
//! (chunk 10a, `docs/shaders.md`, Chunk 10 design).
//!
//! * `Bindless`: one set layout (set 0: binding 0 sampled images, 1 storage images, 2 samplers,
//!   each `UPDATE_AFTER_BIND | PARTIALLY_BOUND` and sized by what the device allows, at most
//!   `1 << 20`), one pool, one set and the pipeline layout (that set and an 8-byte push constant,
//!   the device address of the draw's root block). Chunk 10b writes bindings 0 and 2 (`textures.rs`)
//!   and adds, per list of static samplers a pipeline pair uses, a set 1 of immutable samplers
//!   and the pipeline layout with it (`Bindless::layouts`).
//! * `ViewTable`: one 32-byte record per descriptor-registry index (`descriptor::VIEWS`),
//!   `{u32 kind, u32 dxgi format, u64 address, u32 size, u32 stride, u32 0, u32 0}`, in a
//!   host-visible, device-addressable buffer that grows by doubling. Kind 1 is a buffer (the
//!   address already translated to the device); kind 0 is anything else (10b fills images).
//! * `HostBuf`: a persistently mapped, device-addressable buffer (the table and the arena).
//!
//! shortcuts: the view table is host-visible memory only (no staging into device-local
//! memory); a record whose memory does not translate stays kind 0 and is tried again at the
//! next submission.

use super::gpumem::GpuMemory;
use super::sampler::{SamplerCache, Spec};
use super::{log_once, vk_try, Gpu, Result, VkError};
use crate::d3d12::descriptor::View;
use crate::d3d12::resource::info_of_checked;
use crate::shader::resource::SamplerDesc;
use ash::vk;
use std::collections::HashMap;

/// Descriptors per binding the bindless set is sized for, at most.
const MAX_BINDLESS: u32 = 1 << 20;
/// Bytes of one view record.
pub const RECORD: usize = 32;

/// A host-visible buffer with a device address, mapped for its whole life.
pub struct HostBuf<'g> {
    gpu: &'g Gpu,
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    pub size: usize,
    pub address: vk::DeviceAddress,
}

impl<'g> HostBuf<'g> {
    pub fn new(gpu: &'g Gpu, size: usize) -> Result<HostBuf<'g>> {
        let d = &gpu.device;
        let size = size.max(RECORD);
        // SAFETY: each object is destroyed on the error paths and by `Drop`.
        unsafe {
            let buffer = vk_try(
                "vkCreateBuffer",
                d.create_buffer(
                    &vk::BufferCreateInfo::default().size(size as u64).usage(
                        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::INDEX_BUFFER,
                    ),
                    None,
                ),
            )?;
            let req = d.get_buffer_memory_requirements(buffer);
            let host =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            // Device-local and host-visible when there is such memory (resizable BAR, integrated
            // GPUs), else plain host memory.
            let ty = gpu
                .memory_type(
                    req.memory_type_bits,
                    host | vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .or_else(|| gpu.memory_type(req.memory_type_bits, host));
            let Some(ty) = ty else {
                d.destroy_buffer(buffer, None);
                return Err(VkError::NoMemoryType);
            };
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let memory = match d.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(ty)
                    .push_next(&mut flags),
                None,
            ) {
                Ok(m) => m,
                Err(e) => {
                    d.destroy_buffer(buffer, None);
                    return Err(VkError::Vk("vkAllocateMemory", e));
                }
            };
            let mapped = d
                .bind_buffer_memory(buffer, memory, 0)
                .and_then(|_| d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()));
            let ptr = match mapped {
                Ok(p) => p.cast::<u8>(),
                Err(e) => {
                    d.free_memory(memory, None);
                    d.destroy_buffer(buffer, None);
                    return Err(VkError::Vk("vkMapMemory", e));
                }
            };
            std::ptr::write_bytes(ptr, 0, size);
            let address =
                d.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            Ok(HostBuf {
                gpu,
                buffer,
                memory,
                ptr,
                size,
                address,
            })
        }
    }

    /// Copies `bytes` to `at` (inside the buffer).
    pub fn write(&mut self, at: usize, bytes: &[u8]) {
        assert!(at + bytes.len() <= self.size);
        // SAFETY: inside the mapping, which lives as long as the buffer.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(at), bytes.len()) };
    }

    /// `n` bytes at `at`, read back.
    pub fn read(&self, at: usize, n: usize) -> Vec<u8> {
        assert!(at + n <= self.size);
        // SAFETY: inside the mapping.
        unsafe { std::slice::from_raw_parts(self.ptr.add(at), n) }.to_vec()
    }
}

impl Drop for HostBuf<'_> {
    fn drop(&mut self) {
        // SAFETY: created by `new`; the queue is idle when a backend drops or replaces it.
        unsafe {
            self.gpu.device.destroy_buffer(self.buffer, None);
            self.gpu.device.free_memory(self.memory, None);
        }
    }
}

/// The stages that read the push constant (the root block's address).
pub const PUSH_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(
    vk::ShaderStageFlags::VERTEX.as_raw()
        | vk::ShaderStageFlags::FRAGMENT.as_raw()
        | vk::ShaderStageFlags::COMPUTE.as_raw(),
);

/// Sets of immutable samplers a pool holds room for.
const STATIC_SETS: u32 = 512;
/// Samplers in one such set, at most.
const STATIC_SAMPLERS: u32 = 16;

/// The pipeline layout of one list of static samplers: set 0, set 1 (when the list is not
/// empty) and the push constant.
#[derive(Clone, Copy)]
pub struct Layouts {
    pub layout: vk::PipelineLayout,
    pub set1: Option<vk::DescriptorSet>,
}

struct StaticSet {
    set_layout: vk::DescriptorSetLayout,
    layouts: Layouts,
}

/// The bindless set and the pipeline layouts draws use.
pub struct Bindless<'g> {
    gpu: &'g Gpu,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    /// Pool of the sets of static samplers (made at the first one).
    pool1: vk::DescriptorPool,
    pub set: vk::DescriptorSet,
    /// The layout without static samplers.
    pub layout: vk::PipelineLayout,
    /// Descriptors per binding: registry indices from here on cannot be written.
    pub count: u32,
    statics: HashMap<Vec<SamplerDesc>, StaticSet>,
}

impl<'g> Bindless<'g> {
    pub fn new(gpu: &'g Gpu) -> Result<Bindless<'g>> {
        let d = &gpu.device;
        // SAFETY: a query on our device; the objects are destroyed by `Drop` (or here on error).
        unsafe {
            let mut p12 = vk::PhysicalDeviceVulkan12Properties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut p12);
            gpu.instance
                .get_physical_device_properties2(gpu.phys, &mut p2);
            let n = [
                p12.max_per_stage_descriptor_update_after_bind_sampled_images,
                p12.max_per_stage_descriptor_update_after_bind_storage_images,
                p12.max_per_stage_descriptor_update_after_bind_samplers,
                p12.max_descriptor_set_update_after_bind_sampled_images,
                p12.max_descriptor_set_update_after_bind_storage_images,
                p12.max_descriptor_set_update_after_bind_samplers,
            ]
            .into_iter()
            .min()
            .unwrap_or(0)
            // room for the static samplers of a pipeline's own set, which count against the
            // same per-stage and per-set limits (validation: 1048577 > 1048576)
            .saturating_sub(STATIC_SAMPLERS)
            .min(MAX_BINDLESS);
            if n == 0 {
                return Err(VkError::Invalid("no update-after-bind descriptors"));
            }
            let stages = vk::ShaderStageFlags::ALL;
            let kinds = [
                vk::DescriptorType::SAMPLED_IMAGE,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::DescriptorType::SAMPLER,
            ];
            let bindings: Vec<_> = kinds
                .iter()
                .enumerate()
                .map(|(i, &t)| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i as u32)
                        .descriptor_type(t)
                        .descriptor_count(n)
                        .stage_flags(stages)
                })
                .collect();
            let flags = [vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND; 3];
            let mut bf =
                vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
            let set_layout = vk_try(
                "vkCreateDescriptorSetLayout",
                d.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                        .bindings(&bindings)
                        .push_next(&mut bf),
                    None,
                ),
            )?;
            // `this` owns what exists so far: an error returns it and `Drop` cleans up.
            let mut this = Bindless {
                gpu,
                set_layout,
                pool: vk::DescriptorPool::null(),
                pool1: vk::DescriptorPool::null(),
                set: vk::DescriptorSet::null(),
                layout: vk::PipelineLayout::null(),
                count: n,
                statics: HashMap::new(),
            };
            let sizes: Vec<_> = kinds
                .iter()
                .map(|&t| vk::DescriptorPoolSize::default().ty(t).descriptor_count(n))
                .collect();
            this.pool = vk_try(
                "vkCreateDescriptorPool",
                d.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                        .max_sets(1)
                        .pool_sizes(&sizes),
                    None,
                ),
            )?;
            let layouts = [set_layout];
            this.set = vk_try(
                "vkAllocateDescriptorSets",
                d.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(this.pool)
                        .set_layouts(&layouts),
                ),
            )?[0];
            let push = [vk::PushConstantRange::default()
                .stage_flags(PUSH_STAGES)
                .offset(0)
                .size(8)];
            this.layout = vk_try(
                "vkCreatePipelineLayout",
                d.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&layouts)
                        .push_constant_ranges(&push),
                    None,
                ),
            )?;
            Ok(this)
        }
    }

    /// The pipeline layout for `samplers` (the pair's static samplers, set 1 binding k being
    /// `samplers[k]`) and the set to bind at set 1, made on first use.
    pub fn layouts(
        &mut self,
        samplers: &[SamplerDesc],
        cache: &mut SamplerCache<'_>,
    ) -> std::result::Result<Layouts, String> {
        if samplers.is_empty() {
            return Ok(Layouts {
                layout: self.layout,
                set1: None,
            });
        }
        if let Some(s) = self.statics.get(samplers) {
            return Ok(s.layouts);
        }
        if samplers.len() > STATIC_SAMPLERS as usize {
            return Err(format!("{} static samplers", samplers.len()));
        }
        let handles = samplers
            .iter()
            .map(|d| cache.get(Spec::from_static(d)).ok_or("a static sampler"))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let d = &self.gpu.device;
        // SAFETY: objects made here are destroyed on each error path or owned by `statics`.
        unsafe {
            if self.pool1 == vk::DescriptorPool::null() {
                let size = [vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLER)
                    .descriptor_count(STATIC_SETS * STATIC_SAMPLERS)];
                self.pool1 = d
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .max_sets(STATIC_SETS)
                            .pool_sizes(&size),
                        None,
                    )
                    .map_err(|e| format!("vkCreateDescriptorPool: {e}"))?;
            }
            let each: Vec<[vk::Sampler; 1]> = handles.iter().map(|&h| [h]).collect();
            let bindings: Vec<_> = each
                .iter()
                .enumerate()
                .map(|(k, h)| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(k as u32)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .stage_flags(vk::ShaderStageFlags::ALL)
                        .immutable_samplers(h)
                })
                .collect();
            let set_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| format!("vkCreateDescriptorSetLayout: {e}"))?;
            let both = [self.set_layout, set_layout];
            let set = d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.pool1)
                    .set_layouts(&both[1..]),
            );
            let set = match set {
                Ok(s) => s[0],
                Err(e) => {
                    d.destroy_descriptor_set_layout(set_layout, None);
                    return Err(format!("vkAllocateDescriptorSets (static samplers): {e}"));
                }
            };
            let push = [vk::PushConstantRange::default()
                .stage_flags(PUSH_STAGES)
                .offset(0)
                .size(8)];
            let layout = match d.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&both)
                    .push_constant_ranges(&push),
                None,
            ) {
                Ok(l) => l,
                Err(e) => {
                    d.destroy_descriptor_set_layout(set_layout, None);
                    return Err(format!("vkCreatePipelineLayout: {e}"));
                }
            };
            let layouts = Layouts {
                layout,
                set1: Some(set),
            };
            self.statics.insert(
                samplers.to_vec(),
                StaticSet {
                    set_layout,
                    layouts,
                },
            );
            Ok(layouts)
        }
    }
}

impl Drop for Bindless<'_> {
    fn drop(&mut self) {
        let d = &self.gpu.device;
        // SAFETY: created by `new` on this device, the queue idle; null handles are no-ops.
        unsafe {
            for s in self.statics.values() {
                d.destroy_pipeline_layout(s.layouts.layout, None);
                d.destroy_descriptor_set_layout(s.set_layout, None);
            }
            d.destroy_descriptor_pool(self.pool1, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_pool(self.pool, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// What a registry view comes to in the table.
enum Rec {
    /// A buffer view whose memory translated: the record.
    /// With the title address span it points into.
    Buffer([u8; RECORD], (u64, u64)),
    /// Kind 0: not a buffer (10b handles images and samplers).
    None,
    /// A buffer view whose memory does not translate (yet).
    Retry,
}

/// The record of one registry view: a buffer view of memory that translates is kind 1.
fn record_of(view: &View, mem: &mut GpuMemory<'_>) -> Rec {
    match buffer_view(view) {
        None => Rec::None,
        Some((format, address, size, stride)) => match mem.translate(address, size.max(1)) {
            None => Rec::Retry,
            Some(m) => {
                let mut r = [0u8; RECORD];
                r[0..4].copy_from_slice(&1u32.to_le_bytes());
                r[4..8].copy_from_slice(&format.to_le_bytes());
                r[8..16].copy_from_slice(&m.address.to_le_bytes());
                let len = size.max(1);
                let size = u32::try_from(size).unwrap_or(u32::MAX);
                r[16..20].copy_from_slice(&size.to_le_bytes());
                r[20..24].copy_from_slice(&stride.to_le_bytes());
                Rec::Buffer(r, (address, address.saturating_add(len)))
            }
        },
    }
}

/// `(dxgi format, title address, size in bytes, stride)` of a view of a buffer.
fn buffer_view(view: &View) -> Option<(u32, u64, u64, u32)> {
    Some(match view {
        // a constant buffer is read by vec4 index (`idxen`): the console runtime gives its V# a
        // stride of 16
        View::Constant(address, size) => (0, *address, u64::from(*size), 16),
        View::Shader(res, desc) => {
            // the registry keeps resource pointers the title created views of, released or not
            let (rd, base) = info_of_checked(*res as *mut _)?;
            if rd.dimension != 1 {
                return None;
            }
            match desc {
                // `{format, dimension (1: buffer), mapping, first element (u64), count, stride,
                // flags (1: raw)}`
                Some(w) if w[1] == 1 => {
                    let (first, count, stride, raw) = (
                        u64::from(w[3]) | u64::from(w[4]) << 32,
                        w[5],
                        w[6],
                        w[7] & 1 != 0,
                    );
                    let elem = if raw { 4 } else { u64::from(stride.max(1)) };
                    (
                        w[0],
                        base + first * elem,
                        u64::from(count) * elem,
                        if raw { 0 } else { stride },
                    )
                }
                Some(_) => return None,
                None => (0, base, rd.width, 0),
            }
        }
        View::Unordered(res, _, desc) => {
            let (rd, base) = info_of_checked(*res as *mut _)?;
            if rd.dimension != 1 {
                return None;
            }
            match desc {
                // `{format, dimension, first element (u64), count, stride, counter offset, flags}`
                Some(w) if w[1] == 1 => {
                    let (first, count, stride, raw) = (
                        u64::from(w[2]) | u64::from(w[3]) << 32,
                        w[4],
                        w[5],
                        w[8] & 1 != 0,
                    );
                    let elem = if raw { 4 } else { u64::from(stride.max(1)) };
                    (
                        w[0],
                        base + first * elem,
                        u64::from(count) * elem,
                        if raw { 0 } else { stride },
                    )
                }
                Some(_) => return None,
                None => (0, base, rd.width, 0),
            }
        }
        // A buffer SRV of memory without a resource object (device slot 70): the resource
        // description's width is its size, the view description's words 10 and 11 its address
        // (as `textures::classify` reads them for images).
        View::Placed(false, rd, vd) if rd[0] == BUFFER_DIMENSION => {
            let address = u64::from(vd[10]) | u64::from(vd[11]) << 32;
            if address == 0 {
                return None;
            }
            let width = u64::from(rd[4]) | u64::from(rd[5]) << 32;
            let (first, stride, raw) = (
                u64::from(vd[3]) | u64::from(vd[4]) << 32,
                vd[6],
                vd[7] & 1 != 0,
            );
            let elem = if raw { 4 } else { u64::from(stride.max(1)) };
            let skip = first.saturating_mul(elem).min(width);
            (
                vd[0],
                address + skip,
                width - skip,
                if raw { 0 } else { stride },
            )
        }
        _ => return None,
    })
}

/// `D3D12_RESOURCE_DIMENSION_BUFFER`.
const BUFFER_DIMENSION: u32 = 1;

/// The view records in a buffer, indexed like the descriptor registry.
pub struct ViewTable<'g> {
    gpu: &'g Gpu,
    buf: HostBuf<'g>,
    /// Records the buffer holds room for.
    capacity: usize,
    /// Its cursor in the registry's change log.
    changes: crate::d3d12::descriptor::Consumer,
    /// Indices of buffer views whose memory did not translate: tried again.
    retry: Vec<usize>,
    /// Per index, the title address span its record points into (empty when it has none).
    spans: Vec<(u64, u64)>,
}

impl<'g> ViewTable<'g> {
    pub fn new(gpu: &'g Gpu) -> Result<ViewTable<'g>> {
        let capacity = 1024;
        Ok(ViewTable {
            gpu,
            buf: HostBuf::new(gpu, capacity * RECORD)?,
            capacity,
            changes: crate::d3d12::descriptor::Consumer::new(),
            retry: Vec::new(),
            spans: vec![(0, 0); capacity],
        })
    }

    /// The device address of record 0.
    pub fn address(&self) -> vk::DeviceAddress {
        self.buf.address
    }

    /// The records pointing into a dropped import (title ranges `(start, end)`) are written
    /// again at the next `update`: their addresses changed.
    pub fn invalidate(&mut self, dropped: &[(usize, usize)]) {
        if dropped.is_empty() {
            return;
        }
        let mut sorted: Vec<(u64, u64)> =
            dropped.iter().map(|&(a, b)| (a as u64, b as u64)).collect();
        sorted.sort_unstable();
        for index in 0..self.spans.len() {
            let (a, b) = self.spans[index];
            if a >= b {
                continue;
            }
            // the ranges starting before `b`; one of them reaching past `a` overlaps
            let n = sorted.partition_point(|&(s, _)| s < b);
            if sorted[..n].iter().any(|&(_, e)| e > a) {
                self.retry.push(index);
                super::profile::count(super::profile::Count::Rewritten, 1);
            }
        }
    }

    fn grow(&mut self, need: usize) -> Result<()> {
        if need <= self.capacity {
            return Ok(());
        }
        let mut capacity = self.capacity;
        while capacity < need {
            capacity *= 2;
        }
        let mut new = HostBuf::new(self.gpu, capacity * RECORD)?;
        let old = self.buf.read(0, self.capacity * RECORD);
        new.write(0, &old);
        self.buf = new;
        self.capacity = capacity;
        self.spans.resize(capacity, (0, 0));
        Ok(())
    }

    /// Stores the record of `index`; a record that went (None, memory lost) is zeroed: kind 0.
    fn put(&mut self, index: usize, rec: Rec) {
        self.spans[index] = (0, 0);
        match rec {
            Rec::Buffer(r, span) => {
                self.buf.write(index * RECORD, &r);
                self.spans[index] = span;
            }
            Rec::Retry => {
                self.retry.push(index);
                self.buf.write(index * RECORD, &[0u8; RECORD]);
            }
            Rec::None => self.buf.write(index * RECORD, &[0u8; RECORD]),
        }
    }

    /// Writes the records of the registry indices that changed since the last call (a new or
    /// reused index gets its view's record, a freed one a zeroed record), and retries the ones
    /// whose memory did not translate. The queue is idle.
    pub fn update(&mut self, mem: &mut GpuMemory<'_>) {
        let changed = self.changes.changes();
        let need = changed
            .last()
            .map(|c| c.0 as usize)
            .into_iter()
            .chain(self.retry.iter().copied())
            .max()
            .map_or(0, |i| i + 1);
        if self.grow(need).is_err() {
            log_once("view table: cannot grow the buffer".into());
            // looked at again (as they are then) by the next call
            self.retry.extend(changed.iter().map(|c| c.0 as usize));
            return;
        }
        // views tried before and not translated, then the new ones
        let mut again = std::mem::take(&mut self.retry);
        again.sort_unstable();
        again.dedup();
        let views: Vec<(usize, Option<View>)> = again
            .into_iter()
            .map(|index| (index, crate::d3d12::descriptor::view_at(index)))
            .chain(changed.into_iter().map(|(index, v)| (index as usize, v)))
            .collect();
        // their memory imported first, in as few imports as it allows
        let accesses: Vec<(u64, u64)> = views
            .iter()
            .filter_map(|(_, v)| buffer_view(v.as_ref()?))
            .map(|(_, address, size, _)| (address, size.max(1)))
            .collect();
        mem.prepare(&accesses);
        for (index, v) in views {
            let rec = v.as_ref().map_or(Rec::None, |v| record_of(v, mem));
            self.put(index, rec);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::draw::tests::title_memory;
    use super::super::replay::tests::{buffer, texture};
    use super::super::tests::gpu_or_skip;
    use super::*;
    use crate::d3d12::descriptor::{create_constant_buffer_view, create_shader_resource_view};

    /// The registry index a view slot holds (its second word).
    fn index_of(slot: &[u8; 64]) -> usize {
        u32::from_le_bytes(slot[4..8].try_into().unwrap()) as usize
    }

    fn u32_at(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
    }

    fn u64_at(b: &[u8], at: usize) -> u64 {
        u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
    }

    #[test]
    fn a_placed_buffer_view_is_a_buffer_record() {
        // From the title's trace: a 128-byte buffer, stride 32, at 0x10b7de20.
        let rd = [1, 0, 0, 0, 128, 0, 1, 0x10001, 0, 1, 0, 1, 0, 0];
        let vd = [
            39,
            1,
            0x1688,
            0,
            0,
            0,
            32,
            0,
            1,
            21974,
            0x10b7_de20,
            0,
            22,
            0,
            0,
            0,
        ];
        assert_eq!(
            buffer_view(&View::Placed(false, rd, vd)),
            Some((39, 0x10b7_de20, 128, 32))
        );
        // first element 1 of 32 bytes
        let mut skip = vd;
        skip[3] = 1;
        assert_eq!(
            buffer_view(&View::Placed(false, rd, skip)),
            Some((39, 0x10b7_de40, 96, 32))
        );
        // a texture, an unordered view and an address of 0 are not buffer records
        let mut tex = rd;
        tex[0] = 3;
        assert_eq!(buffer_view(&View::Placed(false, tex, vd)), None);
        assert_eq!(buffer_view(&View::Placed(true, rd, vd)), None);
        let mut none = vd;
        (none[10], none[11]) = (0, 0);
        assert_eq!(buffer_view(&View::Placed(false, rd, none)), None);
    }

    #[test]
    fn a_constant_buffer_view_is_indexed_by_vec4() {
        assert_eq!(
            buffer_view(&View::Constant(0x1000, 1536)),
            Some((0, 0x1000, 1536, 16))
        );
    }

    #[test]
    fn buffer_views_get_records_with_translated_addresses() {
        let Some(g) = gpu_or_skip() else { return };
        let mut mem = GpuMemory::new(&g);
        let mut table = ViewTable::new(&g).unwrap();
        let base = title_memory(&[0u8; 4096]);
        // SAFETY: committed title memory of 4096 bytes.
        let res = buffer(unsafe { std::slice::from_raw_parts_mut(base as *mut u8, 4096) });
        let (srv, cbv, lost, tex) = (
            Box::leak(Box::new([0u8; 64])),
            Box::leak(Box::new([0u8; 64])),
            Box::leak(Box::new([0u8; 64])),
            Box::leak(Box::new([0u8; 64])),
        );
        let outside = [0u8; 64];
        // SAFETY: valid descriptions and 64-byte slots.
        unsafe {
            // structured: first element 2, 8 elements of 16 bytes
            let desc = [0u32, 1, 0x1688, 2, 0, 8, 16, 0, 0, 0];
            create_shader_resource_view(
                std::ptr::null_mut(),
                res as *mut _,
                desc.as_ptr().cast(),
                srv.as_mut_ptr() as u64,
            );
            let words = |a: u64, size: u32| [a as u32, (a >> 32) as u32, size, 0];
            create_constant_buffer_view(
                std::ptr::null_mut(),
                words(base + 256, 64).as_ptr().cast(),
                cbv.as_mut_ptr() as u64,
            );
            create_constant_buffer_view(
                std::ptr::null_mut(),
                words(outside.as_ptr() as u64, 64).as_ptr().cast(),
                lost.as_mut_ptr() as u64,
            );
            let t = texture(4, 4, 28, 0);
            create_shader_resource_view(
                std::ptr::null_mut(),
                t as *mut _,
                std::ptr::null(),
                tex.as_mut_ptr() as u64,
            );
        }
        table.update(&mut mem);
        let first = mem.translate(base, 1).unwrap().address;
        let r = table.buf.read(index_of(srv) * RECORD, RECORD);
        assert_eq!(u32_at(&r, 0), 1, "a buffer view is kind 1");
        assert_eq!(u64_at(&r, 8), first + 32);
        assert_eq!((u32_at(&r, 16), u32_at(&r, 20)), (128, 16));
        let r = table.buf.read(index_of(cbv) * RECORD, RECORD);
        assert_eq!(
            (u32_at(&r, 0), u64_at(&r, 8), u32_at(&r, 16)),
            (1, first + 256, 64)
        );
        // read by vec4 index (`idxen`): the record's stride is 16
        assert_eq!(u32_at(&r, 20), 16);
        // memory that is not GPU memory and a texture stay kind 0; only the first is retried
        for slot in [&*lost, &*tex] {
            let r = table.buf.read(index_of(slot) * RECORD, RECORD);
            assert_eq!(u32_at(&r, 0), 0);
        }
        // (the registry is global: other tests' views may be in the list too)
        assert!(table.retry.contains(&index_of(lost)));
        assert!(!table.retry.contains(&index_of(tex)));
        // a dropped import rewrites the records into it (the addresses may have changed)
        mem.forget(&[(base as usize, 1)]);
        assert!(!mem.dropped().is_empty());
        table.invalidate(mem.dropped());
        assert!(table.retry.contains(&index_of(srv)));
        assert!(table.retry.contains(&index_of(cbv)));
        assert!(!table.retry.contains(&index_of(tex)));
        table.update(&mut mem);
        assert!(table.retry.contains(&index_of(lost)));
        let r = table.buf.read(index_of(srv) * RECORD, RECORD);
        let again = mem.translate(base, 1).unwrap().address;
        assert_eq!(u64_at(&r, 8), again + 32);
    }

    #[test]
    fn only_records_into_a_dropped_import_are_rewritten() {
        let Some(g) = gpu_or_skip() else { return };
        let mut mem = GpuMemory::new(&g);
        let mut table = ViewTable::new(&g).unwrap();
        let (a, b) = (title_memory(&[0u8; 4096]), title_memory(&[0u8; 4096]));
        let (sa, sb) = (
            Box::leak(Box::new([0u8; 64])),
            Box::leak(Box::new([0u8; 64])),
        );
        let words = |a: u64, size: u32| [a as u32, (a >> 32) as u32, size, 0];
        // SAFETY: valid descriptions and 64-byte slots.
        unsafe {
            create_constant_buffer_view(
                std::ptr::null_mut(),
                words(a, 64).as_ptr().cast(),
                sa.as_mut_ptr() as u64,
            );
            create_constant_buffer_view(
                std::ptr::null_mut(),
                words(b, 64).as_ptr().cast(),
                sb.as_mut_ptr() as u64,
            );
        }
        table.update(&mut mem);
        mem.forget(&[(a as usize, 1)]);
        table.invalidate(mem.dropped());
        assert!(table.retry.contains(&index_of(sa)));
        assert!(!table.retry.contains(&index_of(sb)));
        // a record whose memory is not translated any more is zeroed, not left stale
        table.update(&mut mem);
        let r = table.buf.read(index_of(sa) * RECORD, RECORD);
        assert_eq!(u64_at(&r, 8), mem.translate(a, 1).unwrap().address);
    }

    #[test]
    fn layouts_of_static_samplers_are_cached_by_list() {
        let Some(g) = gpu_or_skip() else { return };
        let mut bl = Bindless::new(&g).unwrap();
        let mut cache = SamplerCache::new(&g);
        let a = SamplerDesc::from_words([0x400, 0x00c0_0000, 0x18f0_0000, 0]);
        let b = SamplerDesc::from_words([0x401, 0x00c0_0000, 0x0, 0]);
        let none = bl.layouts(&[], &mut cache).unwrap();
        assert_eq!((none.layout, none.set1), (bl.layout, None));
        let one = bl.layouts(&[a], &mut cache).unwrap();
        assert!(one.set1.is_some());
        assert_ne!(one.layout, bl.layout);
        let again = bl.layouts(&[a], &mut cache).unwrap();
        assert_eq!((again.layout, again.set1), (one.layout, one.set1));
        let two = bl.layouts(&[a, b], &mut cache).unwrap();
        assert_ne!(two.layout, one.layout);
        assert!(bl.layouts(&[a; 17], &mut cache).is_err());
    }
}
