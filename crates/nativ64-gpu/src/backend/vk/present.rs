//! `PresentX` on Vulkan: the first resource of plane 0 is blitted to the title's window and,
//! with a dump directory, written as a PNG (`frame_00001.png`, ...).

use super::log_once;
use super::replay::{Readback, VkBackend};
use crate::d3d12::command::Rect;
use ash::vk;
use nativ64::windowing::RawSurface;
use std::path::Path;

/// The window a frame goes to: its handle, host surface and client size.
#[derive(Clone, Copy)]
pub struct Window {
    pub hwnd: usize,
    pub surface: RawSurface,
    pub size: (u32, u32),
}

/// RGBA8 bytes of a read-back image, or None for formats the dump does not convert.
pub fn rgba8(rb: &Readback) -> Option<Vec<u8>> {
    match rb.format {
        vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => Some(rb.bytes.clone()),
        // Single-channel planes (video luma/chroma) show as grey.
        vk::Format::R8_UNORM => Some(rb.bytes.iter().flat_map(|&v| [v, v, v, 255]).collect()),
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => {
            let mut v = rb.bytes.clone();
            v.as_chunks_mut::<4>()
                .0
                .iter_mut()
                .for_each(|p| p.swap(0, 2));
            Some(v)
        }
        _ => None,
    }
}

/// Write `rgba` (`w` x `h`, 4 bytes per pixel) as a PNG file.
pub fn write_png(path: &Path, w: u32, h: u32, rgba: &[u8]) -> std::io::Result<()> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_compression(png::Compression::Fast);
    let mut writer = enc.write_header().map_err(std::io::Error::other)?;
    writer.write_image_data(rgba).map_err(std::io::Error::other)
}

/// Which presented frames are dumped: 1, 2, 4, 8, ... and every 60th, so a long run does not
/// fill the disk with 4K PNGs. `NATIV64_DUMP_FROM=<n>` skips the frames before `n` (a late frame
/// is then reached without paying for the dumps of the early ones). `NATIV64_DUMP_EVERY=<n>`
/// also keeps every `n`th frame (dense dumps around a moment of interest).
pub fn dump_wanted(frame: u64) -> bool {
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let every = *EVERY.get_or_init(|| {
        std::env::var("NATIV64_DUMP_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    });
    static FROM: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let from = *FROM.get_or_init(|| {
        std::env::var("NATIV64_DUMP_FROM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    });
    frame >= from
        && (frame.is_power_of_two()
            || frame.is_multiple_of(60)
            || (every > 0 && frame.is_multiple_of(every)))
}

impl<'g> VkBackend<'g> {
    /// Dump presented images (see `dump_wanted`) as PNGs in `dir` (created when missing).
    pub fn set_dump_dir(&mut self, dir: std::path::PathBuf) {
        if let Err(e) = std::fs::create_dir_all(&dir) {
            log_once(format!("cannot create {}: {e}", dir.display()));
        }
        self.dump = Some(dir);
    }

    /// Frames presented so far.
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// Present `resource` (after the earlier submissions, which have completed): dump it, and
    /// blit `rect` of it (all when None) to `window`.
    pub fn present(&mut self, resource: usize, rect: Option<Rect>, window: Option<Window>) {
        let _call = super::profile::call(true, super::profile::Step::Present);
        self.frames += 1;
        if self.frames.is_multiple_of(60) {
            super::profile::report(self.frames, 60);
        }
        if self.frames.is_multiple_of(600)
            || (super::profile::enabled() && self.frames.is_multiple_of(60))
        {
            let line = self.memory_line();
            eprintln!("{line}");
        }
        let Some((image, w, h)) = self.colour_image(resource) else {
            log_once("PresentX of a resource nothing rendered to: nothing shown".into());
            return;
        };
        if let Some(dir) = self.dump.clone().filter(|_| dump_wanted(self.frames)) {
            let t = super::profile::start();
            self.dump_frame(resource, &dir);
            super::profile::stop(super::profile::Phase::Dump, t);
        }
        if self.inspecting_at_present() {
            self.dump_targets();
        }
        let Some(win) = window else { return };
        if !self.swapchains.contains_key(&win.hwnd) {
            let sc = match self.gpu.create_swapchain(win.surface, win.size) {
                Ok(sc) => Some(sc),
                Err(e) => {
                    log_once(format!("no swapchain for the window: {e}"));
                    None
                }
            };
            self.swapchains.insert(win.hwnd, sc);
        }
        let Some(Some(sc)) = self.swapchains.get_mut(&win.hwnd) else {
            return;
        };
        sc.resize(win.size.0, win.size.1);
        // The source region: the rectangle clamped to the image, or all of it.
        let r = rect
            .map(|r| {
                [
                    r[0].clamp(0, w as i32),
                    r[1].clamp(0, h as i32),
                    r[2].clamp(0, w as i32),
                    r[3].clamp(0, h as i32),
                ]
            })
            .filter(|r| r[2] > r[0] && r[3] > r[1])
            .unwrap_or([0, 0, w as i32, h as i32]);
        let result = sc.present_with(|d, cb, dst, _, extent| {
            let layer = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };
            let off = |x: i32, y: i32| vk::Offset3D { x, y, z: 0 };
            let region = [vk::ImageBlit {
                src_subresource: layer,
                src_offsets: [off(r[0], r[1]), off(r[2], r[3])],
                dst_subresource: layer,
                dst_offsets: [off(0, 0), off(extent.width as i32, extent.height as i32)],
            }];
            // SAFETY: `cb` is recording; the source is in GENERAL, the target in
            // TRANSFER_DST_OPTIMAL; the source image outlives the submission.
            unsafe {
                d.cmd_blit_image(
                    cb,
                    image,
                    vk::ImageLayout::GENERAL,
                    dst,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &region,
                    vk::Filter::LINEAR,
                )
            };
        });
        if let Err(e) = result {
            log_once(format!("present failed: {e}"));
        }
    }

    fn dump_frame(&mut self, resource: usize, dir: &Path) {
        let Some(rb) = self.read_back(resource) else {
            return;
        };
        let Some(rgba) = rgba8(&rb) else {
            log_once(format!("the dump does not convert {:?}", rb.format));
            return;
        };
        let path = dir.join(format!("frame_{:05}.png", self.frames));
        if let Err(e) = write_png(&path, rb.width, rb.height, &rgba) {
            log_once(format!("cannot write {}: {e}", path.display()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dump_subset() {
        let picked: Vec<u64> = (1..=130).filter(|&f| dump_wanted(f)).collect();
        assert_eq!(picked, [1, 2, 4, 8, 16, 32, 60, 64, 120, 128]);
    }

    #[test]
    fn single_channel_plane_is_grey() {
        let rb = Readback {
            width: 2,
            height: 1,
            format: vk::Format::R8_UNORM,
            bytes: vec![7, 9],
        };
        assert_eq!(rgba8(&rb).unwrap(), [7, 7, 7, 255, 9, 9, 9, 255]);
    }

    #[test]
    fn bgra_is_swapped_to_rgba() {
        let rb = Readback {
            width: 1,
            height: 1,
            format: vk::Format::B8G8R8A8_UNORM,
            bytes: vec![1, 2, 3, 4],
        };
        assert_eq!(rgba8(&rb).unwrap(), [3, 2, 1, 4]);
        let rb = Readback {
            format: vk::Format::R16_UNORM,
            ..rb
        };
        assert!(rgba8(&rb).is_none());
    }
}
