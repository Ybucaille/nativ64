//! A process-wide write watch over host memory (chunk 10b, part 4): which pages were written
//! since the last look, for the textures that are hashed only when their pages changed.
//!
//! The kernel does the tracking: a `userfaultfd` in asynchronous write-protect mode
//! (`UFFD_FEATURE_WP_ASYNC`) over a committed run, and `PAGEMAP_SCAN` on `/proc/self/pagemap`
//! reporting (and re-protecting) the pages written since the previous scan. Writes never
//! fault into the process, and writes made by the kernel (`read(2)` into a watched page) are
//! tracked too. Everything works unprivileged (`UFFD_USER_MODE_ONLY`).
//!
//! Several textures can share a page, and a scan clears the written state of its pages, so the
//! pages found written are kept in a short log: each looker holds a `Cursor` and is told about
//! every write since its own previous look. When the log is trimmed, or a run is registered
//! again, the older cursors answer "written".
//!
//! Write protection is per mapping: pages of the physical pool that the title maps at a second
//! address (`xmem::alias_pages`, a second map) are written through that other address without
//! the watch seeing it. Enumerating and watching every alias (and the ones made later) would
//! need a registry per run and a re-registration at each remap; instead a range with a page
//! that is aliased anywhere is never answered "unchanged" (`xmem::is_aliased`, a lookup per
//! 64 KiB page in a table bounded by the pool): the caller hashes it, as it did before the watch.
//!
//! Where the kernel refuses (no `userfaultfd`, no `PAGEMAP_SCAN`, a run that was unmapped and
//! mapped again) the answer is None: the caller treats it as written and keeps its old path.

use super::log_once;
use super::profile::{self, Count, Phase};
use std::collections::BTreeMap;
use std::sync::Mutex;

const PAGE: usize = 4096;
const UFFD_API: u64 = 0xAA;
const UFFD_USER_MODE_ONLY: libc::c_int = 1;
const FEATURE_WP_UNPOPULATED: u64 = 1 << 13;
const FEATURE_WP_ASYNC: u64 = 1 << 15;
const REGISTER_MODE_WP: u64 = 1 << 1;
/// `_IOWR(0xAA, 0x3F, struct uffdio_api)`, `_IOWR(0xAA, 0x00, struct uffdio_register)`,
/// `_IOR(0xAA, 0x01, struct uffdio_range)`.
const UFFDIO_API: libc::c_ulong = 0xc018_aa3f;
const UFFDIO_REGISTER: libc::c_ulong = 0xc020_aa00;
const UFFDIO_UNREGISTER: libc::c_ulong = 0x8010_aa01;
/// `_IOWR('f', 16, struct pm_scan_arg)`.
const PAGEMAP_SCAN: libc::c_ulong = 0xc060_6610;
const PAGE_IS_WRITTEN: u64 = 1 << 1;
const PM_SCAN_WP_MATCHING: u64 = 1;
const PM_SCAN_CHECK_WPASYNC: u64 = 2;
/// Regions per scan call; a full vector means another call.
const VEC: usize = 64;
/// Entries kept in the log before it is dropped.
const LOG_MAX: usize = 4096;

#[repr(C)]
#[derive(Default)]
struct UffdioApi {
    api: u64,
    features: u64,
    ioctls: u64,
}

#[repr(C)]
#[derive(Default)]
struct UffdioRange {
    start: u64,
    len: u64,
}

#[repr(C)]
#[derive(Default)]
struct UffdioRegister {
    range: UffdioRange,
    mode: u64,
    ioctls: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct PageRegion {
    start: u64,
    end: u64,
    categories: u64,
}

#[repr(C)]
#[derive(Default)]
struct PmScanArg {
    size: u64,
    flags: u64,
    start: u64,
    end: u64,
    walk_end: u64,
    vec: u64,
    vec_len: u64,
    max_pages: u64,
    category_inverted: u64,
    category_mask: u64,
    category_anyof_mask: u64,
    return_mask: u64,
}

/// What a looker has seen of the log: the default has seen nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Cursor(u64);

struct Watch {
    uffd: libc::c_int,
    pagemap: libc::c_int,
    /// The registered runs, start to end.
    runs: BTreeMap<usize, usize>,
    /// Incremented by every look.
    epoch: u64,
    /// The cursors below it lost part of the log.
    floor: u64,
    /// Pages found written: epoch of the scan, start, end.
    log: Vec<(u64, usize, usize)>,
}

static WATCH: Mutex<Option<Option<Watch>>> = Mutex::new(None);

fn open() -> Result<Watch, String> {
    // SAFETY: plain system calls with valid arguments.
    let uffd = unsafe {
        libc::syscall(
            libc::SYS_userfaultfd,
            libc::O_CLOEXEC | libc::O_NONBLOCK | UFFD_USER_MODE_ONLY,
        )
    } as libc::c_int;
    if uffd < 0 {
        return Err(format!("userfaultfd: {}", std::io::Error::last_os_error()));
    }
    let mut api = UffdioApi {
        api: UFFD_API,
        features: FEATURE_WP_ASYNC | FEATURE_WP_UNPOPULATED,
        ioctls: 0,
    };
    // SAFETY: `api` is the argument of UFFDIO_API.
    if unsafe { libc::ioctl(uffd, UFFDIO_API, &mut api) } < 0 {
        let err = std::io::Error::last_os_error();
        // SAFETY: closing the descriptor opened above.
        unsafe { libc::close(uffd) };
        return Err(format!("UFFDIO_API (WP_ASYNC): {err}"));
    }
    // SAFETY: a NUL-terminated path.
    let pagemap = unsafe {
        libc::open(
            c"/proc/self/pagemap".as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC,
        )
    };
    if pagemap < 0 {
        let err = std::io::Error::last_os_error();
        // SAFETY: closing the descriptor opened above.
        unsafe { libc::close(uffd) };
        return Err(format!("/proc/self/pagemap: {err}"));
    }
    Ok(Watch {
        uffd,
        pagemap,
        runs: BTreeMap::new(),
        epoch: 0,
        floor: 0,
        log: Vec::new(),
    })
}

fn with<T>(f: impl FnOnce(&mut Watch) -> T) -> Option<T> {
    let mut g = WATCH.lock().unwrap_or_else(|e| e.into_inner());
    let w = g.get_or_insert_with(|| match open() {
        Ok(w) => Some(w),
        Err(e) => {
            log_once(format!(
                "no write watch, textures are hashed every time ({e})"
            ));
            None
        }
    });
    w.as_mut().map(f)
}

impl Watch {
    fn register(&mut self, start: usize, end: usize) -> Result<(), String> {
        let mut reg = UffdioRegister {
            range: UffdioRange {
                start: start as u64,
                len: (end - start) as u64,
            },
            mode: REGISTER_MODE_WP,
            ioctls: 0,
        };
        // SAFETY: `reg` is the argument of UFFDIO_REGISTER.
        if unsafe { libc::ioctl(self.uffd, UFFDIO_REGISTER, &mut reg) } < 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        self.runs.insert(start, end);
        Ok(())
    }

    fn unregister(&mut self, start: usize, end: usize) {
        let mut range = UffdioRange {
            start: start as u64,
            len: (end - start) as u64,
        };
        // SAFETY: `range` is the argument of UFFDIO_UNREGISTER; a failure (the memory is
        // gone) leaves nothing to undo.
        unsafe { libc::ioctl(self.uffd, UFFDIO_UNREGISTER, &mut range) };
    }

    /// Forgets every run that overlaps `start..end`.
    fn forget(&mut self, start: usize, end: usize) {
        let gone: Vec<_> = self
            .runs
            .iter()
            .filter(|(s, e)| **s < end && start < **e)
            .map(|(s, e)| (*s, *e))
            .collect();
        for (s, e) in gone {
            self.runs.remove(&s);
            self.unregister(s, e);
        }
        // whoever looked before cannot know what happened in between
        self.floor = self.epoch + 1;
        self.log.clear();
    }

    fn covered(&self, start: usize, end: usize) -> bool {
        self.runs
            .range(..=start)
            .next_back()
            .is_some_and(|(_, e)| end <= *e)
    }

    /// Scans `start..end`, re-protecting what was written; the written regions.
    fn scan(&self, start: usize, end: usize) -> Result<Vec<(usize, usize)>, ()> {
        let mut out = Vec::new();
        let mut regions = [PageRegion::default(); VEC];
        let mut at = start as u64;
        while (at as usize) < end {
            let mut arg = PmScanArg {
                size: size_of::<PmScanArg>() as u64,
                flags: PM_SCAN_WP_MATCHING | PM_SCAN_CHECK_WPASYNC,
                start: at,
                end: end as u64,
                vec: regions.as_mut_ptr() as u64,
                vec_len: VEC as u64,
                category_mask: PAGE_IS_WRITTEN,
                return_mask: PAGE_IS_WRITTEN,
                ..Default::default()
            };
            // SAFETY: `arg` and the region vector it points to outlive the call.
            let n = unsafe { libc::ioctl(self.pagemap, PAGEMAP_SCAN, &mut arg) };
            profile::count(Count::ScanCalls, 1);
            if n < 0 {
                return Err(());
            }
            out.extend(
                regions[..n as usize]
                    .iter()
                    .map(|r| (r.start as usize, r.end as usize)),
            );
            let walked = arg.walk_end & 0x0000_ffff_ffff_ffff;
            if walked <= at {
                return Err(());
            }
            at = walked;
        }
        Ok(out)
    }
}

/// Watches the committed run `start..end` (any page-aligned memory the process maps and may
/// write): false when the kernel cannot.
pub(super) fn watch(run: (usize, usize)) -> bool {
    let t = profile::start();
    let (start, end) = (run.0 & !(PAGE - 1), run.1.next_multiple_of(PAGE));
    let ok = with(|w| {
        if w.covered(start, end) {
            return true;
        }
        w.forget(start, end);
        match w.register(start, end) {
            Ok(()) => true,
            Err(e) => {
                log_once(format!("cannot watch {start:#x}..{end:#x}: {e}"));
                false
            }
        }
    })
    .unwrap_or(false);
    profile::stop(Phase::Watch, t);
    ok
}

/// Whether `range` lies in a run that is already watched: the cheap check before `watch`,
/// which needs the whole committed run.
pub(super) fn is_watched(range: (usize, usize)) -> bool {
    let (start, end) = (range.0 & !(PAGE - 1), range.1.next_multiple_of(PAGE));
    with(|w| w.covered(start, end)).unwrap_or(false)
}

/// Whether a page of `range` was written since this `cursor` last looked (a write by the
/// process or by the kernel on its behalf), then moves the cursor. The first look of a cursor
/// says written. None when it cannot tell (not watched, or the memory was mapped again, in
/// which case it is watched again): take it as written.
pub(super) fn written(range: (usize, usize), cursor: &mut Cursor) -> Option<bool> {
    let t = profile::start();
    let (start, end) = (range.0 & !(PAGE - 1), range.1.next_multiple_of(PAGE));
    let answer = with(|w| {
        if !w.covered(start, end) || {
            let _t = profile::exec_timer(profile::Exec::AliasNs);
            crate::xmem::is_aliased(start, end)
        } {
            return None;
        }
        w.epoch += 1;
        let epoch = w.epoch;
        match w.scan(start, end) {
            Ok(found) => w.log.extend(found.into_iter().map(|(s, e)| (epoch, s, e))),
            Err(()) => {
                // the run was unmapped or mapped again: register what it is now
                let run = w
                    .runs
                    .range(..=start)
                    .next_back()
                    .map(|(s, e)| (*s, *e))
                    .unwrap_or((start, end));
                w.forget(run.0, run.1);
                if w.register(run.0, run.1).is_err() {
                    log_once(format!("cannot watch {:#x}..{:#x} again", run.0, run.1));
                }
                *cursor = Cursor(w.epoch);
                profile::count(Count::ScanNone, 1);
                return None;
            }
        }
        let seen = cursor.0;
        *cursor = Cursor(epoch);
        let hit = seen < w.floor
            || seen == 0
            || w.log
                .iter()
                .any(|&(at, s, e)| at > seen && s < end && start < e);
        profile::count_max(Count::LogMax, w.log.len() as u64);
        if w.log.len() > LOG_MAX {
            profile::count(Count::LogDrops, 1);
            w.log.clear();
            w.floor = epoch + 1;
        }
        Some(hit)
    })
    .flatten();
    profile::stop(Phase::Watch, t);
    answer
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The watch is process-wide and a test's unmap drops the log of the others: one at a time.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn map(pages: usize) -> usize {
        // SAFETY: a fresh anonymous mapping.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                pages * PAGE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(p, libc::MAP_FAILED);
        p as usize
    }

    /// Unmaps a test's region and forgets its run, as the next test may get the same address.
    fn unmap(a: usize, pages: usize) {
        // SAFETY: the mapping of the calling test.
        unsafe { libc::munmap(a as *mut _, pages * PAGE) };
        let _ = with(|w| w.forget(a, a + pages * PAGE));
    }

    /// True when the kernel has no write watch (the tests then have nothing to check).
    fn unavailable(run: (usize, usize)) -> bool {
        if watch(run) {
            return false;
        }
        println!("skipped: no write watch");
        true
    }

    #[test]
    fn a_page_is_written_once_per_write() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let a = map(8);
        let run = (a, a + 8 * PAGE);
        if unavailable(run) {
            return;
        }
        // populated before the first look, and written since
        unsafe { *(a as *mut u8) = 1 };
        let mut c = Cursor::default();
        assert_eq!(written(run, &mut c), Some(true));
        assert_eq!(written(run, &mut c), Some(false));
        unsafe { *((a + 3 * PAGE + 5) as *mut u8) = 7 };
        assert_eq!(written((a, a + 2 * PAGE), &mut c), Some(false));
        // the page of the write, alone, and in the whole run
        let mut other = c;
        assert_eq!(
            written((a + 3 * PAGE, a + 4 * PAGE), &mut other),
            Some(true)
        );
        assert_eq!(
            written((a + 3 * PAGE, a + 4 * PAGE), &mut other),
            Some(false)
        );
        // the scan of `other` cleared the page: `c` still hears of it from the log
        assert_eq!(written(run, &mut c), Some(true));
        assert_eq!(written(run, &mut c), Some(false));
        unmap(a, 8);
    }

    #[test]
    fn a_read_by_the_kernel_into_a_page_is_a_write() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let a = map(2);
        let run = (a, a + 2 * PAGE);
        if unavailable(run) {
            return;
        }
        let mut c = Cursor::default();
        assert_eq!(written(run, &mut c), Some(true));
        assert_eq!(written(run, &mut c), Some(false));
        let mut fds = [0; 2];
        // SAFETY: a pipe, a write and a read of 4 bytes into the watched page.
        unsafe {
            assert_eq!(libc::pipe(fds.as_mut_ptr()), 0);
            assert_eq!(libc::write(fds[1], b"abcd".as_ptr().cast(), 4), 4);
            assert_eq!(libc::read(fds[0], (a + PAGE + 8) as *mut _, 4), 4);
            libc::close(fds[0]);
            libc::close(fds[1]);
            assert_eq!(*((a + PAGE + 8) as *const u8), b'a');
        }
        assert_eq!(written((a + PAGE, a + 2 * PAGE), &mut c), Some(true));
        assert_eq!(written(run, &mut c), Some(false));
        unmap(a, 2);
    }

    #[test]
    fn pages_aliased_elsewhere_are_never_unchanged() {
        use crate::xmem::*;
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        const PP: usize = 64 << 10;
        const RW: u32 = 0x04;
        // SAFETY: the pool and reservations of this test only.
        unsafe {
            let mut count = 1usize;
            let mut pages = [0usize; 1];
            assert_eq!(
                XMemAllocatePhysicalPages(0, &mut count, pages.as_mut_ptr()),
                1
            );
            let pool = XMemVirtualAlloc(std::ptr::null_mut(), PP, 0x2000, 3, RW) as usize;
            let dst = XMemVirtualAlloc(std::ptr::null_mut(), PP, 0x2000, 3, RW) as usize;
            assert_eq!(
                XMemMapPhysicalPages(pool as *mut _, 1, pages.as_ptr()) as usize,
                pool
            );
            let run = (pool, pool + PP);
            if unavailable(run) {
                return;
            }
            let mut c = Cursor::default();
            assert_eq!(written(run, &mut c), Some(true));
            assert_eq!(written(run, &mut c), Some(false));
            assert!(alias_pages(dst, &[pool]));
            // a write through the alias is invisible to the watch of `pool`
            (dst as *mut u8).write(5);
            assert_ne!(written(run, &mut c), Some(false));
            assert_ne!(written(run, &mut c), Some(false));
            assert!(unmap_pages(dst, 1));
            let _ = written(run, &mut c);
            assert_eq!(written(run, &mut c), Some(false));
            let _ = with(|w| w.forget(pool, pool + PP));
            XMemFreePhysicalPages(1, pages.as_ptr());
        }
    }

    #[test]
    fn memory_mapped_again_is_watched_again() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let a = map(2);
        let run = (a, a + 2 * PAGE);
        if unavailable(run) {
            return;
        }
        let mut c = Cursor::default();
        assert_eq!(written(run, &mut c), Some(true));
        // SAFETY: replaces the mapping of this test at the same address.
        unsafe {
            libc::munmap(a as *mut _, 2 * PAGE);
            let p = libc::mmap(
                a as *mut _,
                2 * PAGE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED,
                -1,
                0,
            );
            assert_eq!(p as usize, a);
        }
        assert_eq!(written(run, &mut c), None);
        let _ = written(run, &mut c);
        assert_eq!(written(run, &mut c), Some(false));
        unsafe { *(a as *mut u8) = 1 };
        assert_eq!(written(run, &mut c), Some(true));
        unmap(a, 2);
    }
}
