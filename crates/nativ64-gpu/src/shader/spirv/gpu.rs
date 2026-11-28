//! Runs a probe module on a Vulkan device (tests only, off by default: set
//! `NATIV64_SPIRV_RUN=1`, `NATIV64_SPIRV_DEVICE=<name substring>` picks the device, lavapipe
//! in CI). One dispatch of one invocation per input vector, so the module's results can be
//! compared with the CPU evaluator on the same bytes.

use crate::backend::vk::subgroup::Subgroups;
use ash::vk;

pub fn enabled() -> bool {
    std::env::var_os("NATIV64_SPIRV_RUN").is_some()
}

pub struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    mem_type: u32,
    /// The smallest and largest subgroup the device may run a shader with.
    pub subgroup_sizes: (u32, u32),
}

/// A module's pipeline with its two buffers.
pub struct Run<'a> {
    gpu: &'a Gpu,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    set: vk::DescriptorSet,
    bufs: [(vk::Buffer, vk::DeviceMemory, *mut u32); 2],
    n_in: usize,
    n_out: usize,
}

impl Gpu {
    /// The device `NATIV64_SPIRV_DEVICE` names, the first one when unset.
    pub fn new() -> Result<Self, String> {
        Self::named(&std::env::var("NATIV64_SPIRV_DEVICE").unwrap_or_default())
    }

    /// The first device whose name contains `want`.
    pub fn named(want: &str) -> Result<Self, String> {
        let e = |x: vk::Result| format!("{x:?}");
        let entry = unsafe { ash::Entry::load() }.map_err(|e| e.to_string())?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
        }
        .map_err(e)?;
        let devs = unsafe { instance.enumerate_physical_devices() }.map_err(e)?;
        let mut pick = None;
        for d in devs {
            let p = unsafe { instance.get_physical_device_properties(d) };
            let name = p
                .device_name_as_c_str()
                .map(|c| c.to_string_lossy().to_string())
                .unwrap_or_default();
            if pick.is_none() && name.contains(want) {
                pick = Some((d, name));
            }
        }
        let (pd, name) = pick.ok_or("no matching Vulkan device")?;
        eprintln!("spirv probe runs on {name}");
        let q = unsafe { instance.get_physical_device_queue_family_properties(pd) }
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or("no compute queue")? as u32;
        let prio = [1.0f32];
        let qi = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(q)
            .queue_priorities(&prio)];
        // everything the device supports: the stage tests need 64-bit integers, bindless
        // arrays and formatless storage images
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f12);
        unsafe { instance.get_physical_device_features2(pd, &mut f2) };
        let sg = Subgroups::query(&instance, pd);
        let mut p13 = vk::PhysicalDeviceVulkan13Properties::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut p13);
        unsafe { instance.get_physical_device_properties2(pd, &mut p2) };
        let subgroup_sizes = (p13.min_subgroup_size, p13.max_subgroup_size);
        let mut mr = vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default()
            .shader_maximal_reconvergence(true);
        let mut exts = Vec::new();
        let mut info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qi)
            .push_next(&mut f2);
        if sg.max_reconvergence {
            exts.push(ash::khr::shader_maximal_reconvergence::NAME.as_ptr());
            info = info.push_next(&mut mr);
        }
        info = info.enabled_extension_names(&exts);
        let device = unsafe { instance.create_device(pd, &info, None) }.map_err(e)?;
        let queue = unsafe { device.get_device_queue(q, 0) };
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(q),
                None,
            )
        }
        .map_err(e)?;
        let mp = unsafe { instance.get_physical_device_memory_properties(pd) };
        let want_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mem_type = (0..mp.memory_type_count)
            .find(|&i| {
                mp.memory_types[i as usize]
                    .property_flags
                    .contains(want_flags)
            })
            .ok_or("no host-visible memory")?;
        Ok(Self {
            _entry: entry,
            instance,
            device,
            queue,
            pool,
            mem_type,
            subgroup_sizes,
        })
    }

    pub fn load(&self, words: &[u32], n_in: usize, n_out: usize) -> Result<Run<'_>, String> {
        let e = |x: vk::Result| format!("{x:?}");
        let d = &self.device;
        unsafe {
            let shader = d
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
                .map_err(e)?;
            let binds: Vec<_> = (0..2)
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds),
                    None,
                )
                .map_err(e)?;
            let layouts = [set_layout];
            let layout = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                    None,
                )
                .map_err(e)?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(c"main");
            let pipeline = d
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(layout)],
                    None,
                )
                .map_err(|(_, x)| format!("{x:?}"))?[0];
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2)];
            let desc_pool = d
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(e)?;
            let mut bufs = [(
                vk::Buffer::null(),
                vk::DeviceMemory::null(),
                std::ptr::null_mut(),
            ); 2];
            for (k, n) in [n_in.max(1), n_out.max(1)].into_iter().enumerate() {
                let size = 4 * n as u64;
                let b = d
                    .create_buffer(
                        &vk::BufferCreateInfo::default()
                            .size(size)
                            .usage(vk::BufferUsageFlags::STORAGE_BUFFER),
                        None,
                    )
                    .map_err(e)?;
                let req = d.get_buffer_memory_requirements(b);
                let m = d
                    .allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(req.size)
                            .memory_type_index(self.mem_type),
                        None,
                    )
                    .map_err(e)?;
                d.bind_buffer_memory(b, m, 0).map_err(e)?;
                let p = d
                    .map_memory(m, 0, size, vk::MemoryMapFlags::empty())
                    .map_err(e)? as *mut u32;
                bufs[k] = (b, m, p);
            }
            let set = d
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(desc_pool)
                        .set_layouts(&layouts),
                )
                .map_err(e)?[0];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = bufs
                .iter()
                .map(|b| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(b.0)
                        .range(vk::WHOLE_SIZE)]
                })
                .collect();
            let writes: Vec<_> = (0..2)
                .map(|k| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(k as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&infos[k])
                })
                .collect();
            d.update_descriptor_sets(&writes, &[]);
            Ok(Run {
                gpu: self,
                layout,
                pipeline,
                set,
                bufs,
                n_in,
                n_out,
            })
        }
    }
}

/// An image of a stage run: bound at registry index `index` of the sampled-image array
/// (binding 0) or the storage-image array (binding 1), RGBA32F, laid out as `texels` (four
/// floats per texel, x fastest).
pub struct StageImage {
    pub index: u32,
    pub storage: bool,
    pub three_d: bool,
    pub extent: [u32; 3],
    pub texels: Vec<f32>,
}

type HostBuffer = (vk::Buffer, vk::DeviceMemory, *mut u8);

impl Gpu {
    fn host_buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<HostBuffer, String> {
        let e = |x: vk::Result| format!("{x:?}");
        let d = &self.device;
        unsafe {
            let b = d
                .create_buffer(
                    &vk::BufferCreateInfo::default().size(size).usage(usage),
                    None,
                )
                .map_err(e)?;
            let req = d.get_buffer_memory_requirements(b);
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let m = d
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(self.mem_type)
                        .push_next(&mut flags),
                    None,
                )
                .map_err(e)?;
            d.bind_buffer_memory(b, m, 0).map_err(e)?;
            let p = d
                .map_memory(m, 0, size, vk::MemoryMapFlags::empty())
                .map_err(e)? as *mut u8;
            std::ptr::write_bytes(p, 0, size as usize);
            Ok((b, m, p))
        }
    }

    /// Runs a translated compute stage (`io::translate_stage`) once: `groups` workgroups, the
    /// root block holding `user` dwords, a zeroed output buffer of `out_words` dwords whose
    /// address is user dwords 0 and 1, and tables (`(user SGPR, dwords)`: the SGPR holds the
    /// low half of the table's address, the root block its full address). Returns the output
    /// buffer and the content of each image afterwards.
    pub fn run_stage(
        &self,
        words: &[u32],
        groups: [u32; 3],
        user: &[(usize, u32)],
        tables: &[(usize, Vec<u32>)],
        images: &[StageImage],
        out_words: usize,
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>), String> {
        let e = |x: vk::Result| format!("{x:?}");
        let d = &self.device;
        const OUT_AT: u64 = 4096;
        let texel_bytes = |im: &StageImage| 4 * im.texels.len() as u64;
        let stage_size: u64 = images.iter().map(texel_bytes).sum::<u64>().max(16);
        unsafe {
            let size = OUT_AT + 4 * out_words.max(1) as u64;
            let (buf, _mem, p) = self.host_buffer(
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )?;
            let base =
                d.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf));
            let put32 = |at: u64, v: u32| *(p.add(at as usize) as *mut u32) = v;
            let put64 = |at: u64, v: u64| *(p.add(at as usize) as *mut u64) = v;
            // root block: user_data[32], table_base[32], view_table
            let out_addr = base + OUT_AT;
            put32(0, out_addr as u32);
            put32(4, (out_addr >> 32) as u32);
            for (n, (sgpr, dw)) in tables.iter().enumerate() {
                let at = 512 + 256 * n as u64;
                for (k, v) in dw.iter().enumerate() {
                    put32(at + 4 * k as u64, *v);
                }
                put32(4 * *sgpr as u64, (base + at) as u32);
                put64(128 + 8 * *sgpr as u64, base + at);
            }
            for &(k, v) in user {
                put32(4 * k as u64, v);
            }
            let shader = d
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
                .map_err(e)?;
            let binds: Vec<_> = [
                (0, vk::DescriptorType::SAMPLED_IMAGE),
                (1, vk::DescriptorType::STORAGE_IMAGE),
            ]
            .into_iter()
            .map(|(b, ty)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(ty)
                    .descriptor_count(8)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
            let flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND; 2];
            let mut bf =
                vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
            let set_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .bindings(&binds)
                        .push_next(&mut bf),
                    None,
                )
                .map_err(e)?;
            let layouts = [set_layout];
            let range = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(8)];
            let layout = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&layouts)
                        .push_constant_ranges(&range),
                    None,
                )
                .map_err(e)?;
            let pipeline = d
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .module(shader)
                                .name(c"main"),
                        )
                        .layout(layout)],
                    None,
                )
                .map_err(|(_, x)| format!("{x:?}"))?[0];
            let sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(8),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(8),
            ];
            let pool = d
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(e)?;
            let set = d
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
                .map_err(e)?[0];
            // the images, with a host-visible staging buffer for the upload and the readback
            let (stage, _smem, sp) = self.host_buffer(
                stage_size,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            )?;
            let colour = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            let mut made = Vec::new();
            let mut at = 0u64;
            for im in images {
                let img = d
                    .create_image(
                        &vk::ImageCreateInfo::default()
                            .image_type(if im.three_d {
                                vk::ImageType::TYPE_3D
                            } else {
                                vk::ImageType::TYPE_2D
                            })
                            .format(vk::Format::R32G32B32A32_SFLOAT)
                            .extent(vk::Extent3D {
                                width: im.extent[0],
                                height: im.extent[1],
                                depth: im.extent[2],
                            })
                            .mip_levels(1)
                            .array_layers(1)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .tiling(vk::ImageTiling::OPTIMAL)
                            .usage(
                                vk::ImageUsageFlags::STORAGE
                                    | vk::ImageUsageFlags::SAMPLED
                                    | vk::ImageUsageFlags::TRANSFER_SRC
                                    | vk::ImageUsageFlags::TRANSFER_DST,
                            )
                            .initial_layout(vk::ImageLayout::UNDEFINED),
                        None,
                    )
                    .map_err(e)?;
                let req = d.get_image_memory_requirements(img);
                let mt = (0..32u32)
                    .find(|&i| req.memory_type_bits >> i & 1 != 0)
                    .ok_or("no memory type for an image")?;
                let mem = d
                    .allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(req.size)
                            .memory_type_index(mt),
                        None,
                    )
                    .map_err(e)?;
                d.bind_image_memory(img, mem, 0).map_err(e)?;
                let view = d
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(img)
                            .view_type(if im.three_d {
                                vk::ImageViewType::TYPE_3D
                            } else {
                                vk::ImageViewType::TYPE_2D
                            })
                            .format(vk::Format::R32G32B32A32_SFLOAT)
                            .subresource_range(colour),
                        None,
                    )
                    .map_err(e)?;
                std::ptr::copy_nonoverlapping(
                    im.texels.as_ptr() as *const u8,
                    sp.add(at as usize),
                    texel_bytes(im) as usize,
                );
                made.push((img, view, at));
                at += texel_bytes(im);
            }
            let infos: Vec<[vk::DescriptorImageInfo; 1]> = made
                .iter()
                .map(|m| {
                    [vk::DescriptorImageInfo::default()
                        .image_view(m.1)
                        .image_layout(vk::ImageLayout::GENERAL)]
                })
                .collect();
            let writes: Vec<_> = images
                .iter()
                .zip(&infos)
                .map(|(im, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(u32::from(im.storage))
                        .dst_array_element(im.index)
                        .descriptor_type(if im.storage {
                            vk::DescriptorType::STORAGE_IMAGE
                        } else {
                            vk::DescriptorType::SAMPLED_IMAGE
                        })
                        .image_info(info)
                })
                .collect();
            if !writes.is_empty() {
                d.update_descriptor_sets(&writes, &[]);
            }
            let cb = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.pool)
                        .command_buffer_count(1),
                )
                .map_err(e)?[0];
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())
                .map_err(e)?;
            let to_general: Vec<_> = made
                .iter()
                .map(|m| {
                    vk::ImageMemoryBarrier::default()
                        .image(m.0)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .subresource_range(colour)
                })
                .collect();
            d.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_general,
            );
            let region = |im: &StageImage, at: u64| {
                vk::BufferImageCopy::default()
                    .buffer_offset(at)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: im.extent[0],
                        height: im.extent[1],
                        depth: im.extent[2],
                    })
            };
            for (im, m) in images.iter().zip(&made) {
                d.cmd_copy_buffer_to_image(
                    cb,
                    stage,
                    m.0,
                    vk::ImageLayout::GENERAL,
                    &[region(im, m.2)],
                );
            }
            let mem_barrier = |src, dst| {
                [vk::MemoryBarrier::default()
                    .src_access_mask(src)
                    .dst_access_mask(dst)]
            };
            d.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &mem_barrier(
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                ),
                &[],
                &[],
            );
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            d.cmd_push_constants(
                cb,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &base.to_le_bytes(),
            );
            d.cmd_dispatch(cb, groups[0], groups[1], groups[2]);
            d.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &mem_barrier(
                    vk::AccessFlags::SHADER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                ),
                &[],
                &[],
            );
            for (im, m) in images.iter().zip(&made) {
                d.cmd_copy_image_to_buffer(
                    cb,
                    m.0,
                    vk::ImageLayout::GENERAL,
                    stage,
                    &[region(im, m.2)],
                );
            }
            d.end_command_buffer(cb).map_err(e)?;
            let fence = d
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(e)?;
            let cbs = [cb];
            d.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&cbs)],
                fence,
            )
            .map_err(e)?;
            d.wait_for_fences(&[fence], true, u64::MAX).map_err(e)?;
            let out: Vec<u32> = (0..out_words)
                .map(|k| *(p.add(OUT_AT as usize + 4 * k) as *const u32))
                .collect();
            let back = images
                .iter()
                .zip(&made)
                .map(|(im, m)| {
                    (0..im.texels.len())
                        .map(|k| *(sp.add(m.2 as usize + 4 * k) as *const f32))
                        .collect()
                })
                .collect();
            Ok((out, back))
        }
    }
}

impl Run<'_> {
    /// One dispatch with `ins` in the first buffer; the second buffer comes back.
    pub fn go(&self, ins: &[u32]) -> Result<Vec<u32>, String> {
        let e = |x: vk::Result| format!("{x:?}");
        let g = self.gpu;
        let d = &g.device;
        unsafe {
            for (k, v) in ins.iter().enumerate().take(self.n_in) {
                *self.bufs[0].2.add(k) = *v;
            }
            for k in 0..self.n_out {
                *self.bufs[1].2.add(k) = 0xdead_beef;
            }
            let cb = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(g.pool)
                        .command_buffer_count(1),
                )
                .map_err(e)?[0];
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())
                .map_err(e)?;
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            d.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.set],
                &[],
            );
            d.cmd_dispatch(cb, 1, 1, 1);
            d.end_command_buffer(cb).map_err(e)?;
            let fence = d
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(e)?;
            let cbs = [cb];
            d.queue_submit(
                g.queue,
                &[vk::SubmitInfo::default().command_buffers(&cbs)],
                fence,
            )
            .map_err(e)?;
            d.wait_for_fences(&[fence], true, u64::MAX).map_err(e)?;
            d.destroy_fence(fence, None);
            d.free_command_buffers(g.pool, &cbs);
            Ok((0..self.n_out).map(|k| *self.bufs[1].2.add(k)).collect())
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
