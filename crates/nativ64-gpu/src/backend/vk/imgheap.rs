//! Device memory of the title images, suballocated from shared blocks.
//!
//! One `vkAllocateMemory` per image costs 1-2 ms on the NVIDIA driver once some ten thousand
//! allocations are alive: a scene load that made 20k images spent 23 s allocating, under the
//! backend lock, and the title's watchdog fired. Images up to `DEDICATED` bytes share blocks of
//! `BLOCK` bytes (first fit, free ranges merged); larger ones keep an allocation of their own. A
//! block that becomes empty goes back to the driver, so the device memory held follows the
//! images alive (plus the free space inside the blocks still in use).

use ash::vk;
use ash::Device;
use std::collections::BTreeMap;

/// Size of a shared block.
const BLOCK: u64 = 64 << 20;
/// Images larger than this get their own allocation.
const DEDICATED: u64 = BLOCK / 4;

/// The memory of one image: bind it at `offset` of `memory`.
pub(super) struct ImageMem {
    pub(super) memory: vk::DeviceMemory,
    pub(super) offset: u64,
    size: u64,
    /// The block it lies in (index in `ImageHeap::blocks`); None for a dedicated allocation.
    block: Option<usize>,
}

struct Block {
    memory: vk::DeviceMemory,
    ty: u32,
    /// Free ranges: offset -> length, never adjacent (merged on free).
    free: BTreeMap<u64, u64>,
    used: u64,
}

#[derive(Default)]
pub(super) struct ImageHeap {
    blocks: Vec<Option<Block>>,
}

/// Takes `size` bytes aligned to `align` (a power of two) from the first free range that holds
/// them: their offset.
fn take(free: &mut BTreeMap<u64, u64>, size: u64, align: u64) -> Option<u64> {
    let align = align.max(1);
    let (start, len, at) = free.iter().find_map(|(&start, &len)| {
        let at = start.next_multiple_of(align);
        (at + size <= start + len).then_some((start, len, at))
    })?;
    free.remove(&start);
    if at > start {
        free.insert(start, at - start);
    }
    if at + size < start + len {
        free.insert(at + size, start + len - at - size);
    }
    Some(at)
}

/// Gives `offset..offset + size` back, merged with the free ranges it touches.
fn give(free: &mut BTreeMap<u64, u64>, offset: u64, size: u64) {
    let (mut start, mut end) = (offset, offset + size);
    if let Some((&s, &l)) = free.range(..offset).next_back() {
        if s + l == offset {
            free.remove(&s);
            start = s;
        }
    }
    if let Some(l) = free.remove(&end) {
        end += l;
    }
    free.insert(start, end - start);
}

impl ImageHeap {
    /// Memory of type `ty` for an image with requirements `req`: from a block with room, else a
    /// new block (or a dedicated allocation for a large image); the driver's error when it has
    /// no memory left.
    pub(super) fn alloc(
        &mut self,
        d: &Device,
        ty: u32,
        req: &vk::MemoryRequirements,
    ) -> Result<ImageMem, vk::Result> {
        if req.size > DEDICATED {
            let memory = allocate(d, ty, req.size)?;
            return Ok(ImageMem {
                memory,
                offset: 0,
                size: req.size,
                block: None,
            });
        }
        for (i, b) in self.blocks.iter_mut().enumerate() {
            let Some(b) = b.as_mut().filter(|b| b.ty == ty) else {
                continue;
            };
            if let Some(offset) = take(&mut b.free, req.size, req.alignment) {
                b.used += req.size;
                return Ok(ImageMem {
                    memory: b.memory,
                    offset,
                    size: req.size,
                    block: Some(i),
                });
            }
        }
        let memory = allocate(d, ty, BLOCK)?;
        let mut free = BTreeMap::new();
        free.insert(req.size, BLOCK - req.size);
        let block = Block {
            memory,
            ty,
            free,
            used: req.size,
        };
        let i = match self.blocks.iter().position(Option::is_none) {
            Some(i) => {
                self.blocks[i] = Some(block);
                i
            }
            None => {
                self.blocks.push(Some(block));
                self.blocks.len() - 1
            }
        };
        Ok(ImageMem {
            memory,
            offset: 0,
            size: req.size,
            block: Some(i),
        })
    }

    /// Gives `m` back; a block left empty is freed.
    ///
    /// # Safety
    /// The image bound to `m` is destroyed and the GPU no longer uses it.
    pub(super) unsafe fn free(&mut self, d: &Device, m: ImageMem) {
        let Some(i) = m.block else {
            // SAFETY: a dedicated allocation of ours, unused (caller).
            unsafe { d.free_memory(m.memory, None) };
            return;
        };
        let Some(b) = self.blocks.get_mut(i).and_then(Option::as_mut) else {
            return;
        };
        give(&mut b.free, m.offset, m.size);
        b.used -= m.size;
        if b.used == 0 {
            // SAFETY: nothing is bound in the block any more.
            unsafe { d.free_memory(b.memory, None) };
            self.blocks[i] = None;
        }
    }

    /// Frees every block (the images are destroyed).
    ///
    /// # Safety
    /// As `free`, for every image.
    pub(super) unsafe fn destroy(&mut self, d: &Device) {
        for b in self.blocks.drain(..).flatten() {
            // SAFETY: caller.
            unsafe { d.free_memory(b.memory, None) };
        }
    }
}

fn allocate(d: &Device, ty: u32, size: u64) -> Result<vk::DeviceMemory, vk::Result> {
    // SAFETY: a plain allocation; the heap frees it.
    unsafe {
        d.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(size)
                .memory_type_index(ty),
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn whole() -> BTreeMap<u64, u64> {
        BTreeMap::from([(0, 1000)])
    }

    #[test]
    fn take_aligns_and_keeps_the_gap_free() {
        let mut f = whole();
        assert_eq!(take(&mut f, 100, 1), Some(0));
        assert_eq!(take(&mut f, 100, 256), Some(256));
        assert_eq!(f, BTreeMap::from([(100, 156), (356, 644)]));
        assert_eq!(take(&mut f, 150, 4), Some(100));
        assert_eq!(take(&mut f, 700, 1), None);
    }

    #[test]
    fn give_merges_with_both_neighbours() {
        let mut f = whole();
        let a = take(&mut f, 100, 1).unwrap();
        let b = take(&mut f, 100, 1).unwrap();
        let c = take(&mut f, 100, 1).unwrap();
        give(&mut f, a, 100);
        give(&mut f, c, 100);
        assert_eq!(f, BTreeMap::from([(0, 100), (200, 800)]));
        give(&mut f, b, 100);
        assert_eq!(f, whole());
    }

    #[test]
    fn a_freed_range_is_reused_first() {
        let mut f = whole();
        let a = take(&mut f, 300, 1).unwrap();
        take(&mut f, 300, 1).unwrap();
        give(&mut f, a, 300);
        assert_eq!(take(&mut f, 200, 1), Some(0));
    }
}
