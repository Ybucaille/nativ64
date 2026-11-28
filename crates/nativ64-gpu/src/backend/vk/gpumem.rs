//! GPU memory (chunk 10a, `docs/shaders.md`, Chunk 10 design): title addresses are host
//! addresses, so the backend imports the host pages a draw reads and hands the shader (and the
//! vertex and index bindings) device addresses translated **on the CPU**.
//!
//! `GpuMemory` keeps the imports by host range. A miss imports the run of committed pages the
//! memory table reports around the address (`nativ64::api_ms_win_core_memory::committed_run`),
//! or the memory of a descriptor heap that owns its own (`descriptor::heap_ranges`); anything
//! else (an address outside both, a page that is not committed) does not translate, and the
//! caller skips the draw. The ranges the table or a heap records as changed
//! (`take_gpu_changes`) drop the imports overlapping them at the start of each submission:
//! the queue is idle between submissions, so a dropped import is never in use.
//!
//! Commits are not recorded as changes: a run that grew after its import (more pages committed
//! next to it) is imported again, bigger, when a translate runs past the old import; the old
//! import is kept (`retired`) until the next `sync`, since commands already recorded in the
//! submission in progress name its buffer.
//!
//! shortcuts: a run is imported in windows of `WINDOW` (an access longer than the window gets
//! its own import, so a driver limit on the allocation size surfaces as an import error,
//! logged, and the draw is skipped); the run is shrunk inwards to the import alignment, so a
//! run smaller than it cannot be imported;
//! a pointer stored in memory is not translated (the stage is skipped instead, see
//! `Translated::memory_addresses`).

use super::{log_once, profile, Gpu, HostBuffer};
use ash::vk;
use nativ64::api_ms_win_core_memory as table;
use std::collections::BTreeMap;

/// Device addresses must stay below this (a built V# keeps 48 bits of base).
const ADDRESS_LIMIT: u64 = 1 << 48;

/// The most a miss imports around the address (more when the access itself is longer): the
/// driver pins the host pages of an import, and a title can commit a multi-gigabyte run (2.2 GB
/// in the first title) of which a draw reads a few megabytes.
const WINDOW: usize = 8 << 20;
/// Window over pool-backed (shared) memory: the driver pins, and so populates, all of an import,
/// and the title commits big pools it touches sparsely, so those imports stay small.
const POOL_WINDOW: usize = 256 << 10;

/// Where a title address lives on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mapped {
    pub buffer: vk::Buffer,
    /// Offset of the address in `buffer`.
    pub offset: u64,
    /// The device address of the same byte.
    pub address: vk::DeviceAddress,
}

/// The most `prepare` merges into one import: windows that only touch (no overlap) stop
/// merging there. Overlapping windows (an access running past its window) always merge.
const MERGE_CAP: usize = WINDOW;

/// A window a miss imports, with the aligned run it lies in (windows of different runs never
/// merge).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Need {
    start: usize,
    end: usize,
    run: (usize, usize),
}

/// The run of `runs` holding `at`, else the one `find` gives (added to `runs`).
fn run_cached(
    runs: &mut Vec<(usize, usize)>,
    at: usize,
    find: impl FnOnce(usize) -> Option<(usize, usize)>,
) -> Option<(usize, usize)> {
    if let Some(&r) = runs.iter().find(|&&(a, b)| a <= at && at < b) {
        return Some(r);
    }
    let r = find(at)?;
    runs.push(r);
    Some(r)
}

/// `va..va+len` (at least 1 byte) as host addresses.
fn bounds(va: u64, len: u64) -> Option<(usize, usize)> {
    let at = usize::try_from(va).ok()?;
    let end = at.checked_add(usize::try_from(len.max(1)).ok()?)?;
    Some((at, end))
}

/// The imports covering `needs`: per run, windows that overlap merge, and windows that touch
/// merge while the result stays within `cap` bytes.
fn coalesce(mut needs: Vec<Need>, cap: usize) -> Vec<(usize, usize)> {
    needs.sort_unstable_by_key(|n| (n.run, n.start, n.end));
    let mut out: Vec<Need> = Vec::new();
    for n in needs {
        if let Some(last) = out.last_mut() {
            let merged_end = last.end.max(n.end);
            if last.run == n.run
                && (n.start < last.end || (n.start == last.end && merged_end - last.start <= cap))
            {
                last.end = merged_end;
                continue;
            }
        }
        out.push(n);
    }
    out.into_iter().map(|n| (n.start, n.end)).collect()
}

struct Import<'g> {
    end: usize,
    buffer: HostBuffer<'g>,
    /// The `epoch` of the last submission that translated an address in it.
    used: u64,
}

/// Imports unused for this many submissions are dropped: the driver pins (and populates) the
/// pages of an import, so an import nobody reads any more only holds host memory.
const IDLE_SUBMISSIONS: u64 = 300;
/// Most bytes kept imported across submissions: past it, the least recently used imports go.
/// An import maps memory the title already has (no copy, no VRAM), but a cap below the working
/// set of the view records made every submission drop dozens of windows and rewrite thousands
/// of view records to import them again (tens of ms per frame). `NATIV64_IMPORT_CAP_MB`
/// overrides it for measurement.
const IMPORT_CAP: usize = 2048 << 20;

fn import_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("NATIV64_IMPORT_CAP_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map_or(IMPORT_CAP, |mb| mb << 20)
    })
}

pub struct GpuMemory<'g> {
    gpu: &'g Gpu,
    imports: BTreeMap<usize, Import<'g>>,
    /// Imports replaced by a bigger one during the submission in progress.
    retired: Vec<Import<'g>>,
    /// Submissions that synced.
    epoch: u64,
    /// The ranges the last `sync` saw changed (the placed images over them go too).
    changed: Vec<(usize, usize)>,
    /// The `(start, end)` title ranges of the imports the last `sync` dropped.
    dropped: Vec<(usize, usize)>,
}

impl<'g> GpuMemory<'g> {
    pub fn new(gpu: &'g Gpu) -> Self {
        table::watch_gpu_changes();
        crate::xmem::return_big_blocks_to_the_host();
        GpuMemory {
            gpu,
            imports: BTreeMap::new(),
            retired: Vec::new(),
            epoch: 0,
            changed: Vec::new(),
            dropped: Vec::new(),
        }
    }

    /// The title ranges `(start, end)` of the imports the last `sync` dropped: what was
    /// translated into them has another device address now.
    pub fn dropped(&self) -> &[(usize, usize)] {
        &self.dropped
    }

    /// The ranges the last `sync` saw changed.
    pub fn changed(&self) -> &[(usize, usize)] {
        &self.changed
    }

    /// Bytes imported.
    #[cfg(test)]
    pub fn imported_bytes(&self) -> usize {
        self.imports.values().map(|i| i.buffer.len).sum()
    }

    /// Drops the imports not used for `IDLE_SUBMISSIONS`, then the least recently used while
    /// more than `IMPORT_CAP` bytes are imported. True when any went.
    fn retire(&mut self, idle: u64, cap: usize) -> bool {
        let epoch = self.epoch;
        let before = self.imports.len();
        let dropped = &mut self.dropped;
        self.imports.retain(|&s, i| {
            let keep = epoch - i.used <= idle;
            if !keep {
                dropped.push((s, i.end));
            }
            keep
        });
        let mut bytes: usize = self.imports.values().map(|i| i.buffer.len).sum();
        if bytes > cap {
            let mut by_age: Vec<(u64, usize)> =
                self.imports.iter().map(|(&s, i)| (i.used, s)).collect();
            by_age.sort_unstable();
            for (_, s) in by_age {
                if bytes <= cap {
                    break;
                }
                if let Some(i) = self.imports.remove(&s) {
                    bytes -= i.buffer.len;
                    self.dropped.push((s, i.end));
                }
            }
        }
        self.imports.len() != before
    }

    /// How many ranges are imported.
    #[cfg(test)]
    pub fn imports(&self) -> usize {
        self.imports.len()
    }

    /// How many replaced imports are waiting for the submission to finish.
    #[cfg(test)]
    pub fn retired(&self) -> usize {
        self.retired.len()
    }

    /// Drops the imports overlapping a range the memory table or a heap recorded as changed.
    /// Called with the GPU idle. True when anything changed (device addresses may differ now).
    pub fn sync(&mut self) -> bool {
        self.retired.clear();
        self.dropped.clear();
        self.epoch += 1;
        self.changed = table::take_gpu_changes();
        let mut any = !self.changed.is_empty();
        if any {
            let changes = std::mem::take(&mut self.changed);
            self.forget(&changes);
            self.changed = changes;
        }
        any |= self.retire(IDLE_SUBMISSIONS, import_cap());
        any
    }

    pub fn forget(&mut self, changes: &[(usize, usize)]) {
        let dropped = &mut self.dropped;
        self.imports.retain(|&start, i| {
            let hit = changes
                .iter()
                .any(|&(a, n)| a < i.end && start < a.saturating_add(n));
            if hit {
                dropped.push((start, i.end));
            }
            !hit
        });
    }

    /// The device side of `len` bytes (at least 1) at title address `va`, importing the memory
    /// on first use; None when the address is not memory we can import, or the range runs out
    /// of what is imported.
    pub fn translate(&mut self, va: u64, len: u64) -> Option<Mapped> {
        let (at, end) = bounds(va, len)?;
        if let Some(m) = self.hit(at, end) {
            return Some(m);
        }
        let need = self.plan(at, end)?;
        self.import(need.start, need.end)?;
        self.hit(at, end)
    }

    /// Imports, ahead of the `translate` calls that will follow, what the accesses `(va, len)`
    /// not yet imported need, the windows of a run that touch or overlap merged into one
    /// import of at most `MERGE_CAP` bytes: many small accesses over a pool (one
    /// `POOL_WINDOW` window each) then cost one driver call instead of one each, and only the
    /// windows they read are pinned, as before. What fails to import here is tried again, a
    /// window at a time, by `translate`.
    pub fn prepare(&mut self, accesses: &[(u64, u64)]) {
        // The runs found so far: finding one scans the page states of its whole allocation
        // (a pool is hundreds of thousands of pages), so a view table of many accesses into
        // one pool must not do it once per access.
        let mut runs = Vec::new();
        let needs: Vec<Need> = accesses
            .iter()
            .filter_map(|&(va, len)| {
                let (at, end) = bounds(va, len)?;
                if self.hit_quiet(at, end) {
                    return None;
                }
                let run = run_cached(&mut runs, at, |at| self.run_around(at))?;
                self.plan_in(at, end, run)
            })
            .collect();
        for (start, end) in coalesce(needs, MERGE_CAP) {
            self.import(start, end);
        }
    }

    /// The import holding `at..end`, marked used.
    fn hit(&mut self, at: usize, end: usize) -> Option<Mapped> {
        let (&start, i) = self.imports.range_mut(..=at).next_back()?;
        if at < i.end && end <= i.end {
            i.used = self.epoch;
            return Some(Self::mapped(start, i, at));
        }
        None
    }

    /// Whether an import holds `at..end` (not marked used).
    fn hit_quiet(&self, at: usize, end: usize) -> bool {
        self.imports
            .range(..=at)
            .next_back()
            .is_some_and(|(_, i)| at < i.end && end <= i.end)
    }

    /// The window a miss on `at..end` imports; None (logged) when the access does not fit an
    /// import of its run.
    fn plan(&self, at: usize, end: usize) -> Option<Need> {
        self.plan_in(at, end, self.run_around(at)?)
    }

    /// [`Self::plan`] in the run `(start, run_end)` around `at`.
    fn plan_in(&self, at: usize, end: usize, (start, run_end): (usize, usize)) -> Option<Need> {
        let align = self.gpu.import_alignment().max(1);
        let (start, run_end) = (start.next_multiple_of(align), run_end / align * align);
        if !(start <= at && end <= run_end) {
            log_once(format!(
                "GPU memory: {at:#x} (+{:#x}) does not fit an import of {start:#x}..{run_end:#x} \
                 (alignment {align:#x})",
                end - at
            ));
            return None;
        }
        // Only a window of the run is imported (see `WINDOW`), aligned to the window and
        // extended to the end of the access when that crosses the window's end.
        let window = if crate::xmem::is_pool_backed(at) {
            POOL_WINDOW
        } else {
            WINDOW
        }
        .next_multiple_of(align);
        let run = (start, run_end);
        let start = start + (at - start) / window * window;
        let end = run_end.min((start + window).max(end.next_multiple_of(align)));
        Some(Need { start, end, run })
    }

    /// Imports `start..end`, replacing the imports overlapping it (too small, or the run grew).
    fn import(&mut self, start: usize, end: usize) -> Option<()> {
        let old: Vec<usize> = self
            .imports
            .range(..end)
            .filter(|(_, i)| i.end > start)
            .map(|(&s, _)| s)
            .collect();
        let ti = profile::start();
        let t0 = std::time::Instant::now();
        let step = profile::step(profile::Step::Import);
        let imported = self.gpu.import_host(start as *mut u8, end - start);
        drop(step);
        profile::stop(profile::Phase::Import, ti);
        profile::count(profile::Count::Imports, 1);
        profile::exec_since(profile::Exec::ImportNs, t0);
        profile::exec_add(profile::Exec::Imports, 1);
        let buffer = match imported {
            Ok(b) => b,
            Err(e) => {
                log_once(format!(
                    "GPU memory: cannot import {start:#x}..{end:#x}: {e}"
                ));
                return None;
            }
        };
        if (buffer.address + buffer.len as u64) >= ADDRESS_LIMIT {
            log_once(format!(
                "GPU memory: the device address {:#x} of {start:#x}.. is above 2^48",
                buffer.address
            ));
            return None;
        }
        for s in old {
            if let Some(i) = self.imports.remove(&s) {
                self.retired.push(i);
            }
        }
        self.imports.insert(
            start,
            Import {
                end,
                buffer,
                used: self.epoch,
            },
        );
        Some(())
    }

    /// The host range around `at` an import may cover: the committed run of the memory table,
    /// else the memory of a descriptor heap.
    fn run_around(&self, at: usize) -> Option<(usize, usize)> {
        table::committed_run(at).or_else(|| {
            crate::d3d12::descriptor::heap_ranges()
                .into_iter()
                .find(|&(a, n)| a <= at && at < a + n)
                .map(|(a, n)| (a, a + n))
        })
    }

    fn mapped(start: usize, i: &Import<'_>, at: usize) -> Mapped {
        let offset = (at - start) as u64;
        Mapped {
            buffer: i.buffer.buffer,
            offset,
            address: i.buffer.address + offset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::gpu_or_skip;
    use super::*;

    const RW: u32 = 0x04;
    const COMMIT: u32 = table::MEM_COMMIT;
    const RESERVE: u32 = table::MEM_RESERVE;

    #[test]
    fn imports_on_demand_reuses_and_forgets_changed_ranges() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        let mut m = GpuMemory::new(&g);
        let a = table::alloc(0, 0x40000, RESERVE | COMMIT, RW, None).unwrap();
        let first = m.translate(a as u64 + 0x100, 16).unwrap();
        assert_eq!(m.imports(), 1);
        // the same run again, further in: the same buffer
        let second = m.translate(a as u64 + 0x2100, 4).unwrap();
        assert_eq!(m.imports(), 1);
        assert_eq!(first.buffer, second.buffer);
        assert_eq!(second.address - first.address, 0x2000);
        assert_eq!(first.offset, 0x100);
        // a range running past the import does not translate
        assert!(m.translate(a as u64 + 0x3fff0, 0x20).is_none());
        // a decommitted page is recorded as changed: the import is dropped, and the page no
        // longer translates, while its neighbours import again
        table::free(a + 0x1000, 0x1000, table::MEM_DECOMMIT).unwrap();
        m.sync();
        assert_eq!(m.imports(), 0);
        assert!(m.translate(a as u64 + 0x1000, 4).is_none());
        assert!(m.translate(a as u64 + 0x2000, 4).is_some());
        assert_eq!(m.imports(), 1);
        table::release(a).unwrap();
        m.sync();
        assert_eq!(m.imports(), 0);
    }

    #[test]
    fn a_run_that_grew_is_imported_again_and_the_old_import_outlives_the_submission() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        let mut m = GpuMemory::new(&g);
        let a = table::alloc(0, 0x100000, RESERVE, RW, None).unwrap();
        table::alloc(a, 0x40000, COMMIT, RW, None).unwrap();
        let first = m.translate(a as u64 + 0x100, 16).unwrap();
        // not committed yet
        assert!(m.translate(a as u64 + 0x50000, 4).is_none());
        assert_eq!(m.imports(), 1);
        // more pages are committed next to the run (no change is recorded)
        table::alloc(a + 0x40000, 0x40000, COMMIT, RW, None).unwrap();
        m.sync();
        assert_eq!(m.imports(), 1);
        let grown = m.translate(a as u64 + 0x50000, 4).unwrap();
        assert_eq!(m.imports(), 1);
        assert_eq!(m.retired(), 1);
        assert_ne!(first.buffer, grown.buffer);
        // the whole run is one buffer now, the earlier addresses included
        let again = m.translate(a as u64 + 0x100, 16).unwrap();
        assert_eq!(again.buffer, grown.buffer);
        assert_eq!(grown.address - again.address, 0x50000 - 0x100);
        // the next submission lets the old import go
        m.sync();
        assert_eq!(m.retired(), 0);
        table::release(a).unwrap();
        m.sync();
    }

    #[test]
    fn a_huge_run_is_imported_in_windows() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        let mut m = GpuMemory::new(&g);
        let run = 4 * WINDOW;
        let a = table::alloc(0, run, RESERVE | COMMIT, RW, None).unwrap();
        let first = m.translate(a as u64 + 0x100, 16).unwrap();
        assert_eq!(m.imports(), 1);
        // an address in another window imports that window only
        let far = m.translate((a + 3 * WINDOW + 8) as u64, 4).unwrap();
        assert_eq!(m.imports(), 2);
        assert_ne!(first.buffer, far.buffer);
        assert_eq!(far.offset, 8);
        // an access across a window's end gets an import covering it
        let across = m.translate((a + WINDOW - 8) as u64, 16).unwrap();
        assert_eq!(across.offset, WINDOW as u64 - 8);
        table::release(a).unwrap();
        m.sync();
    }

    #[test]
    fn idle_and_surplus_imports_are_retired() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        let Some(g) = gpu_or_skip() else { return };
        let mut m = GpuMemory::new(&g);
        let a = table::alloc(0, 4 * WINDOW, RESERVE | COMMIT, RW, None).unwrap();
        for w in 0..4 {
            m.translate((a + w * WINDOW + 8) as u64, 4).unwrap();
        }
        assert_eq!(m.imports(), 4);
        // over the cap: the least recently used (first translated) go first
        m.sync();
        assert!(m.retire(u64::MAX, 2 * WINDOW));
        assert_eq!(m.imports(), 2);
        assert!(m.imported_bytes() <= 2 * WINDOW);
        assert!(m.translate((a + 3 * WINDOW + 8) as u64, 4).is_some());
        assert_eq!(m.imports(), 2, "the newest import was kept");
        // an import read in every submission outlives the idle one
        let busy = (a + 3 * WINDOW + 8) as u64;
        for _ in 0..=IDLE_SUBMISSIONS {
            m.sync();
            m.translate(busy, 4).unwrap();
        }
        assert_eq!(m.imports(), 1);
        table::release(a).unwrap();
        m.sync();
    }

    #[test]
    fn touching_windows_merge_up_to_the_cap_and_overlapping_ones_always() {
        let w = POOL_WINDOW;
        let run = (0, 64 * w);
        let other = (64 * w, 128 * w);
        let need = |start, end, run| Need { start, end, run };
        let needs = vec![
            need(2 * w, 3 * w, run),
            need(0, w, run),
            need(w, 2 * w, run),
            // the same window twice
            need(w, 2 * w, run),
            // a gap, then a window and an access running past it into the next
            need(5 * w, 6 * w, run),
            need(6 * w, 7 * w + 8, run),
            need(7 * w, 8 * w, run),
            // touching, but in another run
            need(64 * w, 65 * w, other),
        ];
        assert_eq!(
            coalesce(needs.clone(), 3 * w),
            vec![(0, 3 * w), (5 * w, 8 * w), (64 * w, 65 * w)]
        );
        // a smaller cap stops touching windows, never overlapping ones
        assert_eq!(
            coalesce(needs, 2 * w),
            vec![
                (0, 2 * w),
                (2 * w, 3 * w),
                (5 * w, 6 * w),
                (6 * w, 8 * w),
                (64 * w, 65 * w)
            ]
        );
        assert!(coalesce(Vec::new(), w).is_empty());
    }

    #[test]
    fn a_run_is_looked_up_once() {
        let mut runs = Vec::new();
        let finds = std::cell::Cell::new(0);
        let find = |at: usize| {
            finds.set(finds.get() + 1);
            let run = if at < 0x9000 {
                (0x1000, 0x9000)
            } else {
                (0x9000, 0xa000)
            };
            (at >= 0x1000).then_some(run)
        };
        assert_eq!(run_cached(&mut runs, 0x2000, find), Some((0x1000, 0x9000)));
        assert_eq!(run_cached(&mut runs, 0x8fff, find), Some((0x1000, 0x9000)));
        assert_eq!(run_cached(&mut runs, 0x1000, find), Some((0x1000, 0x9000)));
        assert_eq!(finds.get(), 1);
        // outside the runs found: looked up, and a miss is not kept
        assert_eq!(run_cached(&mut runs, 0x9000, find), Some((0x9000, 0xa000)));
        assert_eq!(run_cached(&mut runs, 0x10, find), None);
        assert_eq!(finds.get(), 3);
        assert_eq!(runs.len(), 2);
    }

    #[test]
    fn prepared_pool_windows_are_one_import() {
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        use crate::xmem::*;
        let Some(g) = gpu_or_skip() else { return };
        let mut m = GpuMemory::new(&g);
        let n = 64; // 4 MiB of pool pages: 16 windows
        let mut count = n;
        let mut pages = vec![0usize; n];
        // SAFETY: the console calls on a fresh reservation and pages of the pool.
        let base = unsafe {
            let base = XMemVirtualAlloc(std::ptr::null_mut(), n << 16, 0x2000, 3, RW);
            assert_eq!(
                XMemAllocatePhysicalPages(0, &mut count, pages.as_mut_ptr()),
                1
            );
            assert_eq!(XMemMapPhysicalPages(base, n, pages.as_ptr()), base);
            base as usize
        } as u64;
        let w = POOL_WINDOW as u64;
        // windows 0, 1, 2 (touching), then 5: two imports
        let accesses = [
            (base + 8, 16),
            (base + w + 8, 16),
            (base + 2 * w, 4),
            (base + 5 * w, 4),
        ];
        m.prepare(&accesses);
        assert_eq!(m.imports(), 2);
        let first = m.translate(base + 8, 16).unwrap();
        let third = m.translate(base + 2 * w, 4).unwrap();
        assert_eq!(first.buffer, third.buffer);
        assert_eq!(third.address - first.address, 2 * w - 8);
        assert!(m.translate(base + 5 * w, 4).is_some());
        // what is imported already is not imported again
        m.prepare(&accesses);
        assert_eq!(m.imports(), 2);
        assert_eq!(m.retired(), 0);
    }

    #[test]
    fn memory_outside_the_table_does_not_translate() {
        let Some(g) = gpu_or_skip() else { return };
        let mut m = GpuMemory::new(&g);
        let v = vec![0u8; 8192];
        assert!(m.translate(v.as_ptr() as u64, 4).is_none());
        assert!(m.translate(0, 4).is_none());
        assert_eq!(m.imports(), 0);
    }

    #[test]
    fn descriptor_heap_memory_translates() {
        use crate::d3d12::descriptor::{create_descriptor_heap, heap_ranges};
        let Some(g) = gpu_or_skip() else { return };
        let mut m = GpuMemory::new(&g);
        // kind 0 (views), 4 descriptors; the zero words after the description are no placement
        let desc = [0u32, 4, 0, 0, 0, 0, 0, 0];
        let mut out = std::ptr::null_mut();
        // SAFETY: a description with room after it, and a valid out pointer.
        let hr = unsafe {
            create_descriptor_heap(
                std::ptr::null_mut(),
                desc.as_ptr().cast(),
                &crate::d3d12::descriptor::DESCRIPTOR_HEAP_IIDS[0],
                &mut out,
            )
        };
        assert_eq!(hr, 0);
        let (a, n) = *heap_ranges().last().unwrap();
        assert!(m.translate(a as u64 + 8, 8).is_some());
        assert!(m.translate((a + n) as u64, 4).is_none());
    }

    fn rss_shmem_kb() -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .unwrap()
            .lines()
            .find_map(|l| l.strip_prefix("RssShmem:"))
            .map(|v| v.trim().trim_end_matches("kB").trim().parse().unwrap())
            .unwrap()
    }

    #[test]
    fn importing_pool_memory_does_not_populate_it() {
        // The measure is process-wide: the tests that import memory must not run beside it.
        let _changes = crate::backend::vk::tests::gpu_changes_lock();
        use crate::xmem::*;
        let Some(g) = gpu_or_skip() else { return };
        let mut m = GpuMemory::new(&g);
        let n = 128; // 8 MiB of pool pages
        let mut count = n;
        let mut pages = vec![0usize; n];
        let base = unsafe {
            let base = XMemVirtualAlloc(std::ptr::null_mut(), n << 16, 0x2000, 3, RW);
            assert_eq!(
                XMemAllocatePhysicalPages(0, &mut count, pages.as_mut_ptr()),
                1
            );
            assert_eq!(XMemMapPhysicalPages(base, n, pages.as_ptr()), base);
            base as usize
        };
        assert!(is_pool_backed(base + 0x1000));
        assert!(!is_pool_backed(base + (n << 16)));
        let before = rss_shmem_kb();
        m.translate(base as u64 + 0x100, 16).unwrap();
        let after = rss_shmem_kb();
        eprintln!("RssShmem import: before {before} kB after {after} kB");
        assert!(
            after - before < 512,
            "import populated {} kB",
            after - before
        );
    }
}
