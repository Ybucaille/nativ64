//! Vulkan backend (G2.1): instance, device choice, one queue, zero-copy import of host memory
//! and a device-local image with read-back. Nothing here knows about D3D12 or windows.
//!
//! The loader (`libvulkan.so.1`) is opened at run time; a machine without it gets
//! `VkError::Load`, never a link error. Driver failures are `VkError`s, not panics.

mod bindless;
mod compute;
mod device;
mod draw;
mod format;
mod gpumem;
mod image;
mod imgheap;
mod inspect;
mod instance;
mod memory;
mod pipeline;
mod present;
mod profile;
mod replay;
mod residency;
mod rootblock;
mod sampler;
mod staging;
pub(crate) mod subgroup;
mod swapchain;
mod textures;
mod watch;

use ash::vk;
use std::collections::HashSet;
use std::fmt;
use std::sync::{Mutex, OnceLock};

pub use compute::DispatchSkips;
pub use device::Gpu;
pub use draw::DrawSkips;
pub use memory::{AlignedBuf, HostBuffer};
pub use present::Window;
pub use replay::{Readback, ReplayStats, VkBackend};
pub use swapchain::Swapchain;

#[derive(Debug)]
pub enum VkError {
    /// The Vulkan loader could not be opened.
    Load(String),
    /// A Vulkan call failed (name of the call, result).
    Vk(&'static str, vk::Result),
    /// No physical device is usable (or the override matched none); the text says why.
    NoDevice(String),
    /// Host pointer or length not aligned to `minImportedHostPointerAlignment`.
    Misaligned { value: usize, align: usize },
    /// No memory type satisfies the request.
    NoMemoryType,
    /// A caller mistake (zero size, ...).
    Invalid(&'static str),
}

impl fmt::Display for VkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VkError::Load(e) => write!(f, "cannot load the Vulkan loader: {e}"),
            VkError::Vk(call, r) => write!(f, "{call} failed: {r}"),
            VkError::NoDevice(e) => write!(f, "no usable Vulkan device: {e}"),
            VkError::Misaligned { value, align } => {
                write!(f, "{value:#x} is not a multiple of {align:#x}")
            }
            VkError::NoMemoryType => write!(f, "no suitable memory type"),
            VkError::Invalid(e) => write!(f, "invalid argument: {e}"),
        }
    }
}

impl std::error::Error for VkError {}

pub type Result<T> = std::result::Result<T, VkError>;

/// Attach the name of the failing call to a raw Vulkan result.
fn vk_try<T>(call: &'static str, r: ash::prelude::VkResult<T>) -> Result<T> {
    r.map_err(|e| VkError::Vk(call, e))
}

/// Print `msg` to stderr unless the same message was already printed.
fn log_once(msg: String) {
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(Default::default);
    if seen.lock().unwrap().insert(msg.clone()) {
        eprintln!("nativ64-gpu: {msg}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Held by every test that drains the process-wide list of changed memory
    /// (`GpuMemory::sync`, which `execute` calls): run beside one that expects a change, it
    /// could take it. Also held by the test that measures the process's shared memory.
    pub fn gpu_changes_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A `Gpu`, or None after printing the skip line (no loader or no suitable device).
    pub fn gpu_or_skip() -> Option<Gpu> {
        match Gpu::open() {
            Ok(g) => Some(g),
            Err(e) => {
                println!("skipped: no Vulkan device ({e})");
                None
            }
        }
    }

    #[test]
    fn errors_display() {
        let e = VkError::Misaligned {
            value: 3,
            align: 4096,
        };
        assert!(e.to_string().contains("0x1000"));
    }
}
