//! Swapchain on a host window (`RawSurface`): FIFO, two frames in flight, recreated on resize,
//! `OUT_OF_DATE` and `SUBOPTIMAL`. G2.2 only clears and presents; replaying real frames comes
//! with G3. A `Swapchain` borrows its `Gpu` and must be dropped before the window is destroyed.

use super::{vk_try, Gpu, Result, VkError};
use ash::{khr, vk};
use nativ64::windowing::RawSurface;

const FRAMES: usize = 2;

struct Frame {
    pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    acquire: vk::Semaphore,
}

pub struct Swapchain<'g> {
    gpu: &'g Gpu,
    surf_fn: khr::surface::Instance,
    sc_fn: khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    /// One per swapchain image: signalled by the submit, waited by the present.
    render_done: Vec<vk::Semaphore>,
    frames: Vec<Frame>,
    cur: usize,
    /// The size the window wants (used when the surface does not dictate one).
    want: (u32, u32),
    stale: bool,
}

/// Prefer UNORM (what a title's clear values mean), then sRGB; 8 bits per channel only.
fn pick_format(offered: &[vk::SurfaceFormatKHR]) -> Option<vk::SurfaceFormatKHR> {
    const ORDER: [vk::Format; 4] = [
        vk::Format::B8G8R8A8_UNORM,
        vk::Format::R8G8B8A8_UNORM,
        vk::Format::B8G8R8A8_SRGB,
        vk::Format::R8G8B8A8_SRGB,
    ];
    ORDER
        .iter()
        .find_map(|f| offered.iter().find(|o| o.format == *f).copied())
}

/// The extent to use: the surface's own when it has one, else the requested size clamped to
/// what the surface allows.
fn pick_extent(caps: &vk::SurfaceCapabilitiesKHR, want: (u32, u32)) -> vk::Extent2D {
    if caps.current_extent.width != u32::MAX {
        return caps.current_extent;
    }
    vk::Extent2D {
        width: want
            .0
            .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
        height: want
            .1
            .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
    }
}

impl Gpu {
    /// A swapchain on the window behind `surface`, `size` being the client size to use while
    /// the compositor leaves the choice to us. Fails when the instance or device lacks the
    /// window-system extensions or the queue cannot present to this surface.
    pub fn create_swapchain(&self, surface: RawSurface, size: (u32, u32)) -> Result<Swapchain<'_>> {
        let RawSurface::Wayland { display, surface } = surface;
        if !self.wsi_instance || !self.wsi_device {
            return Err(VkError::NoDevice(
                "no VK_KHR_surface/wayland_surface/swapchain on this Vulkan setup".into(),
            ));
        }
        let wl_fn = khr::wayland_surface::Instance::new(&self.entry, &self.instance);
        let surf_fn = khr::surface::Instance::new(&self.entry, &self.instance);
        let sc_fn = khr::swapchain::Device::new(&self.instance, &self.device);
        let info = vk::WaylandSurfaceCreateInfoKHR::default()
            .display(display.cast())
            .surface(surface.cast());
        // SAFETY: the pointers come from a live `RawSurface` (contract of `surface()`).
        let surface = vk_try("vkCreateWaylandSurfaceKHR", unsafe {
            wl_fn.create_wayland_surface(&info, None)
        })?;
        let mut sc = Swapchain {
            gpu: self,
            surf_fn,
            sc_fn,
            surface,
            swapchain: vk::SwapchainKHR::null(),
            format: vk::SurfaceFormatKHR::default(),
            extent: vk::Extent2D::default(),
            images: Vec::new(),
            render_done: Vec::new(),
            frames: Vec::new(),
            cur: 0,
            want: size,
            stale: true,
        };
        // From here `Drop` cleans up whatever exists.
        // SAFETY: valid handles.
        let ok = unsafe {
            sc.surf_fn
                .get_physical_device_surface_support(self.phys, self.queue_family, surface)
        };
        if !vk_try("vkGetPhysicalDeviceSurfaceSupportKHR", ok)? {
            return Err(VkError::NoDevice(
                "the queue family cannot present to this window".into(),
            ));
        }
        sc.create_frames()?;
        sc.rebuild()?;
        Ok(sc)
    }
}

impl Swapchain<'_> {
    fn create_frames(&mut self) -> Result<()> {
        let d = &self.gpu.device;
        for _ in 0..FRAMES {
            // SAFETY: plain creations on a valid device; `Drop` destroys what was pushed.
            unsafe {
                let pool = vk_try(
                    "vkCreateCommandPool",
                    d.create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                            .queue_family_index(self.gpu.queue_family),
                        None,
                    ),
                )?;
                let mut f = Frame {
                    pool,
                    cb: vk::CommandBuffer::null(),
                    fence: vk::Fence::null(),
                    acquire: vk::Semaphore::null(),
                };
                let r = (|| {
                    f.cb = vk_try(
                        "vkAllocateCommandBuffers",
                        d.allocate_command_buffers(
                            &vk::CommandBufferAllocateInfo::default()
                                .command_pool(pool)
                                .command_buffer_count(1),
                        ),
                    )?[0];
                    f.fence = vk_try(
                        "vkCreateFence",
                        d.create_fence(
                            &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                            None,
                        ),
                    )?;
                    f.acquire = vk_try(
                        "vkCreateSemaphore",
                        d.create_semaphore(&vk::SemaphoreCreateInfo::default(), None),
                    )?;
                    Ok(())
                })();
                self.frames.push(f);
                r?;
            }
        }
        Ok(())
    }

    /// (Re)create the swapchain for the current surface state. `false` when the surface has
    /// no area at the moment (nothing to render to).
    fn rebuild(&mut self) -> Result<bool> {
        let g = self.gpu;
        // SAFETY: queries and creations on valid handles; the old swapchain is retired through
        // `old_swapchain` and destroyed once the device is idle.
        unsafe {
            let caps = vk_try(
                "vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
                self.surf_fn
                    .get_physical_device_surface_capabilities(g.phys, self.surface),
            )?;
            let extent = pick_extent(&caps, self.want);
            if extent.width == 0 || extent.height == 0 {
                return Ok(false);
            }
            let formats = vk_try(
                "vkGetPhysicalDeviceSurfaceFormatsKHR",
                self.surf_fn
                    .get_physical_device_surface_formats(g.phys, self.surface),
            )?;
            let format = pick_format(&formats).ok_or_else(|| {
                VkError::NoDevice("the surface offers no 8-bit RGBA/BGRA format".into())
            })?;
            let mut count = caps.min_image_count + 1;
            if caps.max_image_count != 0 {
                count = count.min(caps.max_image_count);
            }
            let alpha = if caps
                .supported_composite_alpha
                .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
            {
                vk::CompositeAlphaFlagsKHR::OPAQUE
            } else {
                // The lowest bit that is set.
                vk::CompositeAlphaFlagsKHR::from_raw(
                    1 << caps.supported_composite_alpha.as_raw().trailing_zeros(),
                )
            };
            let usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT;
            if !caps.supported_usage_flags.contains(usage) {
                return Err(VkError::NoDevice(
                    "the surface cannot be a transfer/colour target".into(),
                ));
            }
            let old = self.swapchain;
            let info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(usage)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(alpha)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true)
                .old_swapchain(old);
            let _ = g.device.device_wait_idle();
            let new = vk_try(
                "vkCreateSwapchainKHR",
                self.sc_fn.create_swapchain(&info, None),
            );
            if old != vk::SwapchainKHR::null() {
                self.sc_fn.destroy_swapchain(old, None);
            }
            self.swapchain = vk::SwapchainKHR::null();
            self.swapchain = new?;
            self.format = format;
            self.extent = extent;
            self.images = vk_try(
                "vkGetSwapchainImagesKHR",
                self.sc_fn.get_swapchain_images(self.swapchain),
            )?;
            for s in self.render_done.drain(..) {
                g.device.destroy_semaphore(s, None);
            }
            for _ in 0..self.images.len() {
                let s = vk_try(
                    "vkCreateSemaphore",
                    g.device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None),
                )?;
                self.render_done.push(s);
            }
        }
        self.stale = false;
        Ok(true)
    }

    /// Tell the swapchain the window's client size (used when the surface leaves it to us).
    pub fn resize(&mut self, w: u32, h: u32) {
        if self.want != (w, h) {
            self.want = (w, h);
            self.stale = true;
        }
    }

    /// Current image size.
    pub fn extent(&self) -> (u32, u32) {
        (self.extent.width, self.extent.height)
    }

    /// Format of the swapchain images.
    pub fn format(&self) -> vk::Format {
        self.format.format
    }

    /// Acquire an image, clear it to `rgba` (each 0..1), present it. `Ok(false)` when nothing
    /// was presented (zero-sized window, or no image became available within 100 ms because
    /// the compositor is not consuming frames).
    pub fn present_clear(&mut self, rgba: [f32; 4]) -> Result<bool> {
        self.present_with(|d, cb, image, range, _| {
            // SAFETY: `cb` is recording and `image` is in TRANSFER_DST_OPTIMAL.
            unsafe {
                d.cmd_clear_color_image(
                    cb,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue { float32: rgba },
                    &[range],
                )
            }
        })
    }

    /// Acquire an image, let `record` write it (it arrives in `TRANSFER_DST_OPTIMAL`, with its
    /// subresource range, its extent and the command buffer recording), present it. `Ok(false)` as for
    /// `present_clear`.
    pub fn present_with<F>(&mut self, record: F) -> Result<bool>
    where
        F: FnOnce(
            &ash::Device,
            vk::CommandBuffer,
            vk::Image,
            vk::ImageSubresourceRange,
            vk::Extent2D,
        ),
    {
        let g = self.gpu;
        let d = &g.device;
        let fi = self.cur;
        // SAFETY: handles belong to this swapchain/device; the queue is used under its mutex;
        // the frame's fence guards reuse of its command buffer and acquire semaphore.
        unsafe {
            vk_try(
                "vkWaitForFences",
                super::device::wait_fence(d, self.frames[fi].fence, "a frame"),
            )?;
            let mut tries = 0;
            let index = loop {
                if self.stale && !self.rebuild()? {
                    return Ok(false);
                }
                let r = self.sc_fn.acquire_next_image(
                    self.swapchain,
                    100_000_000,
                    self.frames[fi].acquire,
                    vk::Fence::null(),
                );
                match r {
                    Ok((i, suboptimal)) => {
                        // Use this image, then rebuild for the next frame.
                        self.stale |= suboptimal;
                        break i;
                    }
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) if tries < 3 => {
                        tries += 1;
                        self.stale = true;
                    }
                    Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => return Ok(false),
                    Err(e) => return Err(VkError::Vk("vkAcquireNextImageKHR", e)),
                }
            };
            let image = self.images[index as usize];
            let f = &self.frames[fi];
            vk_try("vkResetFences", d.reset_fences(&[f.fence]))?;
            vk_try(
                "vkResetCommandBuffer",
                d.reset_command_buffer(f.cb, vk::CommandBufferResetFlags::empty()),
            )?;
            vk_try(
                "vkBeginCommandBuffer",
                d.begin_command_buffer(
                    f.cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                ),
            )?;
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            let barrier = |old, new, src, dst| {
                vk::ImageMemoryBarrier::default()
                    .old_layout(old)
                    .new_layout(new)
                    .src_access_mask(src)
                    .dst_access_mask(dst)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(range)
            };
            d.cmd_pipeline_barrier(
                f.cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                )],
            );
            record(d, f.cb, image, range, self.extent);
            d.cmd_pipeline_barrier(
                f.cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::empty(),
                )],
            );
            vk_try("vkEndCommandBuffer", d.end_command_buffer(f.cb))?;
            let waits = [f.acquire];
            let stages = [vk::PipelineStageFlags::TRANSFER];
            let cbs = [f.cb];
            let signals = [self.render_done[index as usize]];
            let sub = [vk::SubmitInfo::default()
                .wait_semaphores(&waits)
                .wait_dst_stage_mask(&stages)
                .command_buffers(&cbs)
                .signal_semaphores(&signals)];
            let q = g.queue.lock().unwrap();
            vk_try("vkQueueSubmit", d.queue_submit(*q, &sub, f.fence))?;
            let chains = [self.swapchain];
            let indices = [index];
            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(&signals)
                .swapchains(&chains)
                .image_indices(&indices);
            match self.sc_fn.queue_present(*q, &present) {
                Ok(suboptimal) => self.stale |= suboptimal,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.stale = true,
                Err(e) => return Err(VkError::Vk("vkQueuePresentKHR", e)),
            }
        }
        self.cur = (fi + 1) % FRAMES;
        Ok(true)
    }
}

impl Drop for Swapchain<'_> {
    fn drop(&mut self) {
        let d = &self.gpu.device;
        // SAFETY: the device is idle first; every handle was created by this swapchain.
        unsafe {
            let _ = d.device_wait_idle();
            for s in self.render_done.drain(..) {
                d.destroy_semaphore(s, None);
            }
            for f in self.frames.drain(..) {
                if f.acquire != vk::Semaphore::null() {
                    d.destroy_semaphore(f.acquire, None);
                }
                if f.fence != vk::Fence::null() {
                    d.destroy_fence(f.fence, None);
                }
                d.destroy_command_pool(f.pool, None);
            }
            if self.swapchain != vk::SwapchainKHR::null() {
                self.sc_fn.destroy_swapchain(self.swapchain, None);
            }
            self.surf_fn.destroy_surface(self.surface, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(f: vk::Format) -> vk::SurfaceFormatKHR {
        vk::SurfaceFormatKHR {
            format: f,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        }
    }

    #[test]
    fn format_preference() {
        let offered = [
            fmt(vk::Format::R8G8B8A8_SRGB),
            fmt(vk::Format::R8G8B8A8_UNORM),
            fmt(vk::Format::R16G16B16A16_SFLOAT),
        ];
        assert_eq!(
            pick_format(&offered).unwrap().format,
            vk::Format::R8G8B8A8_UNORM
        );
        assert_eq!(
            pick_format(&offered[..1]).unwrap().format,
            vk::Format::R8G8B8A8_SRGB
        );
        assert!(pick_format(&offered[2..]).is_none());
    }

    #[test]
    fn extent_choice() {
        let mut caps = vk::SurfaceCapabilitiesKHR {
            current_extent: vk::Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            },
            min_image_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            max_image_extent: vk::Extent2D {
                width: 4096,
                height: 4096,
            },
            ..Default::default()
        };
        let e = pick_extent(&caps, (640, 99999));
        assert_eq!((e.width, e.height), (640, 4096));
        caps.current_extent = vk::Extent2D {
            width: 800,
            height: 600,
        };
        let e = pick_extent(&caps, (640, 480));
        assert_eq!((e.width, e.height), (800, 600));
    }
}
