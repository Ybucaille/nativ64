//! `xmem.dll`: the console's memory manager. Graphics memory is ordinary host memory here (GPU
//! addresses are CPU addresses), so XMem flags only steer where an allocation lands.
//!
//! Physical pages (`XMemAllocatePhysicalPages`, as the title calls it: `(attributes, SIZE_T
//! *count, ULONG_PTR *pages) -> BOOL`, `count` in and out, 64 KiB pages) are slots of one
//! `memfd` pool: the page "frame number" handed out is the slot index in the file, which later
//! mapping calls can `mmap` over any address. The file is sparse and grows on demand; freed
//! slots are reused. At most `MAX_PHYSICAL_PAGES` pages are outstanding. The attributes are
//! recorded by no one: all pages are the same ordinary host memory.

use nativ64::api_ms_win_core_memory::Placement;
use std::ffi::c_void;
use std::sync::atomic::Ordering;
use std::sync::{Mutex, OnceLock};

const MEM_COMMIT: u32 = 0x1000;
const MEM_RESERVE: u32 = 0x2000;
const MEM_REPLACE_PLACEHOLDER: u32 = 0x4000;
/// 2 MiB pages (alignment); with `PAGES_64K_BIT`, 64 KiB pages.
const PAGES_2M: u32 = 0x2000_0000;
const PAGES_64K_BIT: u32 = 0x40_0000;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// `XMemAlloc` attributes: alignment as log2 (5 bits at 24), bit 33 asks for a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
const MIB: usize = 1 << 20;
/// Host protection constants (the console adds GPU bits above 0x1ff, dropped here).
const HOST_PROTECTION: u32 = 0x1ff;
/// Our block header: the raw pointer and the size, just below the pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for 32-bit memory; else from
/// 4 GiB (kept for 32-bit requests) up to 46 down to 40 bits from the field at bit 12, or 40 bits
/// without one: the console keeps title memory there, and ray-tracing structures must be.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field as usize,
        _ => return (0x1000_0000, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

fn virtual_alloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let (lo, hi) = address_range(flags);
    let align = if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
        2 * MIB
    } else {
        64 << 10
    };
    // The allocation table lives in `nativ64` so `VirtualFree`/`VirtualQuery` see these blocks.
    let protect = match protect & HOST_PROTECTION {
        0 => 0x01,
        p => p,
    };
    let result = if kind & MEM_REPLACE_PLACEHOLDER != 0 {
        // Replaces a placeholder `VirtualFree(.., MEM_PRESERVE_PLACEHOLDER)` made (the memory is
        // mapped later by physical page, so the range stays reserved).
        nativ64::api_ms_win_core_memory::replace_placeholder(base as usize, size, protect)
    } else {
        nativ64::api_ms_win_core_memory::alloc(
            base as usize,
            size,
            kind & (MEM_COMMIT | MEM_RESERVE),
            protect,
            Some(Placement { lo, hi, align }),
        )
    };
    match result {
        Ok(a) => a as *mut c_void,
        Err(e) => {
            nativ64::api_ms_win_core_errorhandling::set_last_error(e);
            std::ptr::null_mut()
        }
    }
}

/// `XMEM_WORKING_SET_STATISTICS`, the layout the title reads its leading fields from.
#[repr(C)]
#[derive(Default)]
pub struct WorkingSet {
    title_limit: usize,
    title_used: usize,
    tools_limit: usize,
    tools_used: usize,
    stack_used: u32,
    physical_pages: u32,
    gpu_optimal_limit: usize,
    gpu_optimal_used: usize,
    tools_gpu_optimal_limit: usize,
    tools_gpu_optimal_used: usize,
    kernel_pool_used: u32,
    kernel_pool_limit: u32,
}

/// The title's memory on a retail console (13.5 GiB, 10 GiB of it GPU-optimal).
const TITLE_MEMORY: usize = 13 << 30 | 1 << 29;
const GPU_OPTIMAL_MEMORY: usize = 10 << 30;
const PAGE_SIZE: usize = 4096;
/// Bytes handed back to the system by `XMemTransferMemory`.
static TRANSFERRED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Size of a physical page of the console's allocator.
const PHYSICAL_PAGE: usize = 64 << 10;
/// Most pages outstanding at once (16 GiB of address space in the pool file, sparse).
const MAX_PHYSICAL_PAGES: usize = 1 << 18;

#[derive(Default)]
struct PhysicalPool {
    /// The memfd, created on first use.
    fd: Option<libc::c_int>,
    /// Pages the file spans (slots ever handed out).
    spanned: usize,
    free: Vec<usize>,
    in_use: usize,
    /// Slots handed out and not freed.
    live: std::collections::HashSet<usize>,
    /// Reverse table: host address of each 64 KiB page that views a pool slot (filled by the map
    /// calls and `alias_pages`, cleared on remap, unmap and free).
    views: std::collections::HashMap<usize, usize>,
    /// Views each slot has now (entries only above zero: at most one per entry of `views`).
    mapped: std::collections::HashMap<usize, u32>,
}

fn pool() -> &'static Mutex<PhysicalPool> {
    static P: OnceLock<Mutex<PhysicalPool>> = OnceLock::new();
    P.get_or_init(Default::default)
}

/// The page number the title sees for a pool slot: a count of 4 KiB frames, so a multiple of 16
/// (titles keep it shifted right by 4), and never 0 (0 means "unmap" to the scatter call).
fn page_number(slot: usize) -> usize {
    (slot + 1) * (PHYSICAL_PAGE >> 12)
}

/// The pool slot of a page number, if that page is allocated.
fn live_slot(p: &PhysicalPool, number: usize) -> Option<usize> {
    let frames = PHYSICAL_PAGE >> 12;
    (number.is_multiple_of(frames) && number >= frames)
        .then(|| number / frames - 1)
        .filter(|s| p.live.contains(s))
}

/// Maps `pages` (numbers) consecutively from `address` over what is there; no pages unmaps `n`
/// pages. `address` must lie in one VirtualAlloc/XMem reservation.
fn map_physical(address: *mut c_void, n: usize, pages: *const usize) -> bool {
    let base = address as usize;
    let Some(len) = n.checked_mul(PHYSICAL_PAGE) else {
        return false;
    };
    if n == 0 || !base.is_multiple_of(PHYSICAL_PAGE) {
        return false;
    }
    let api = nativ64::api_ms_win_core_memory::free;
    if pages.is_null() {
        let mut p = pool().lock().unwrap_or_else(|e| e.into_inner());
        forget_views(&mut p, base, n);
        drop(p);
        return api(base, len, 0x4000).is_ok();
    }
    // SAFETY: the title passes `n` page numbers.
    let numbers = unsafe { std::slice::from_raw_parts(pages, n) };
    let mut p = pool().lock().unwrap_or_else(|e| e.into_inner());
    let mut slots = Vec::with_capacity(n);
    for &number in numbers {
        match live_slot(&p, number) {
            Some(s) => slots.push(s),
            None => return false,
        }
    }
    let Some(fd) = p.fd else { return false };
    // Inside one reservation, and reported committed from now on.
    let protect = view_protect(base);
    if nativ64::api_ms_win_core_memory::mark_mapped(base, len, protect).is_err() {
        return false;
    }
    forget_views(&mut p, base, n);
    for (i, &s) in slots.iter().enumerate() {
        add_view(&mut p, base + i * PHYSICAL_PAGE, s);
    }
    let mut i = 0;
    while i < n {
        let run = 1 + slots[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&(s, k)| *s == slots[i] + k)
            .count();
        // SAFETY: replaces pages of a reservation the table owns with a view of the pool file.
        let m = unsafe {
            libc::mmap(
                (base + i * PHYSICAL_PAGE) as *mut c_void,
                run * PHYSICAL_PAGE,
                host_prot(protect),
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd,
                (slots[i] * PHYSICAL_PAGE) as libc::off_t,
            )
        };
        if m == libc::MAP_FAILED {
            return false;
        }
        i += run;
    }
    true
}

/// Protection a new view at `base` takes: the reservation's, or read/write.
fn view_protect(base: usize) -> u32 {
    // not `query`: it also measures the run of equal pages, a walk over the rest of the pool
    // for every page mapped, with the pool lock held
    match nativ64::api_ms_win_core_memory::allocation_protect(base) {
        Some(p) if p != 0x01 && p != 0 => p,
        _ => 0x04,
    }
}

/// Drops the reverse-table entries of `n` pages from `base`.
fn forget_views(p: &mut PhysicalPool, base: usize, n: usize) {
    for i in 0..n {
        if let Some(s) = p.views.remove(&(base + i * PHYSICAL_PAGE)) {
            if let Some(c) = p.mapped.get_mut(&s) {
                *c -= 1;
                if *c == 0 {
                    p.mapped.remove(&s);
                }
            }
        }
    }
}

/// Records that the page at `address` views `slot`.
fn add_view(p: &mut PhysicalPool, address: usize, slot: usize) {
    // a view replaced without `forget_views` first: drop the old slot's count
    if let Some(old) = p.views.insert(address, slot) {
        if let Some(c) = p.mapped.get_mut(&old) {
            *c -= 1;
            if *c == 0 {
                p.mapped.remove(&old);
            }
        }
    }
    *p.mapped.entry(slot).or_insert(0) += 1;
}

/// Whether a physical page viewed in `start..end` is also viewed at another address (an
/// alias made by `alias_pages` or a second map): writes through the other address do not
/// show in a per-mapping write watch of this one.
pub fn is_aliased(start: usize, end: usize) -> bool {
    let p = pool().lock().unwrap_or_else(|e| e.into_inner());
    if p.mapped.is_empty() {
        return false;
    }
    let first = start & !(PHYSICAL_PAGE - 1);
    (first..end).step_by(PHYSICAL_PAGE).any(|a| {
        p.views
            .get(&a)
            .is_some_and(|s| p.mapped.get(s).copied().unwrap_or(0) > 1)
    })
}

/// Makes the `pages` pages at `dst` a second view of the physical pages that `src` views, so
/// writes through either show in both. All or nothing: false when `dst` is misaligned, or a
/// source page is not one of our physical pages (or `dst` lies outside a reservation).
pub fn alias_pages(dst: usize, src: &[usize]) -> bool {
    let n = src.len();
    let Some(len) = n.checked_mul(PHYSICAL_PAGE) else {
        return false;
    };
    if n == 0 || !dst.is_multiple_of(PHYSICAL_PAGE) {
        return false;
    }
    let mut p = pool().lock().unwrap_or_else(|e| e.into_inner());
    let Some(fd) = p.fd else { return false };
    let mut slots = Vec::with_capacity(n);
    for &a in src {
        match p.views.get(&a) {
            Some(&s) if p.live.contains(&s) => slots.push(s),
            _ => return false,
        }
    }
    let protect = view_protect(dst);
    if nativ64::api_ms_win_core_memory::mark_mapped(dst, len, protect).is_err() {
        return false;
    }
    forget_views(&mut p, dst, n);
    for (i, &s) in slots.iter().enumerate() {
        // SAFETY: replaces pages of a reservation the table owns with a view of the pool file.
        let m = unsafe {
            libc::mmap(
                (dst + i * PHYSICAL_PAGE) as *mut c_void,
                PHYSICAL_PAGE,
                host_prot(protect),
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd,
                (s * PHYSICAL_PAGE) as libc::off_t,
            )
        };
        if m == libc::MAP_FAILED {
            return false;
        }
        add_view(&mut p, dst + i * PHYSICAL_PAGE, s);
    }
    true
}

/// Gives `pages` pages from `dst` fresh zero memory (an unmapped tile reads zero).
pub fn unmap_pages(dst: usize, pages: usize) -> bool {
    let Some(len) = pages.checked_mul(PHYSICAL_PAGE) else {
        return false;
    };
    if pages == 0 || !dst.is_multiple_of(PHYSICAL_PAGE) {
        return false;
    }
    let protect = view_protect(dst);
    if nativ64::api_ms_win_core_memory::mark_mapped(dst, len, protect).is_err() {
        return false;
    }
    let mut p = pool().lock().unwrap_or_else(|e| e.into_inner());
    forget_views(&mut p, dst, pages);
    // SAFETY: replaces pages of a reservation the table owns with fresh anonymous memory.
    let m = unsafe {
        libc::mmap(
            dst as *mut c_void,
            len,
            host_prot(protect),
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED,
            -1,
            0,
        )
    };
    m != libc::MAP_FAILED
}

/// Windows protection to host protection bits (the console's extra bits dropped).
fn host_prot(protect: u32) -> libc::c_int {
    match protect & 0xff {
        0x02 => libc::PROT_READ,
        0x04 | 0x08 => libc::PROT_READ | libc::PROT_WRITE,
        0x10 => libc::PROT_EXEC,
        0x20 => libc::PROT_READ | libc::PROT_EXEC,
        0x40 | 0x80 => libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
        _ => libc::PROT_NONE,
    }
}

/// Whether `address` lies in a 64 KiB page that views the pool file (shared memory: a host
/// access, or a driver pinning it for a host-pointer import, populates every page it touches).
pub fn is_pool_backed(address: usize) -> bool {
    let p = pool().lock().unwrap_or_else(|e| e.into_inner());
    p.views.contains_key(&(address & !(PHYSICAL_PAGE - 1)))
}

/// The pool: bytes of the pages the title holds, and bytes the memfd really holds (`st_blocks`,
/// holes excluded). The second stays above the first only while freed pages are not punched.
pub fn pool_usage() -> (u64, u64) {
    let p = pool().lock().unwrap_or_else(|e| e.into_inner());
    let held = (p.in_use * PHYSICAL_PAGE) as u64;
    let Some(fd) = p.fd else { return (held, 0) };
    // SAFETY: fstat on the pool's own descriptor into a zeroed struct.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    let ok = unsafe { libc::fstat(fd, &mut st) } == 0;
    (held, if ok { st.st_blocks as u64 * 512 } else { 0 })
}

/// Takes `n` page frame numbers from the pool (all or none); `None` when there is no room or the
/// file cannot grow.
fn take_physical_pages(n: usize) -> Option<Vec<usize>> {
    let mut p = pool().lock().unwrap_or_else(|e| e.into_inner());
    if n == 0 || n > MAX_PHYSICAL_PAGES - p.in_use {
        return None;
    }
    if p.fd.is_none() {
        // SAFETY: plain syscall with a NUL-terminated name.
        let fd = unsafe { libc::memfd_create(c"nativ64-xmem".as_ptr(), libc::MFD_CLOEXEC) };
        if fd < 0 {
            return None;
        }
        p.fd = Some(fd);
    }
    let fresh = n.saturating_sub(p.free.len());
    if fresh > 0 {
        let end = (p.spanned + fresh) * PHYSICAL_PAGE;
        // SAFETY: grows the pool file; the new tail reads as zeros.
        if unsafe { libc::ftruncate(p.fd.unwrap(), end as libc::off_t) } != 0 {
            return None;
        }
    }
    let mut pages = Vec::with_capacity(n);
    for _ in 0..n {
        pages.push(match p.free.pop() {
            Some(f) => f,
            None => {
                p.spanned += 1;
                p.spanned - 1
            }
        });
    }
    p.in_use += n;
    p.live.extend(pages.iter().copied());
    Some(pages)
}

/// Attribute bits `XMemAlloc` callers pass: bits 14-15 are the allocation class (2 is the
/// fixed-size class of GPU memory), bits 16-23 the allocator id the title's hooks are keyed on.
const CLASS_MASK: u64 = 0xc000;
const CLASS_GPU: u64 = 0x8000;
/// Attributes of the memory the D3D device takes for committed resources, as the title's hooks
/// see them (allocator id 0x80, GPU class).
pub const GPU_ALLOC_ATTRIBUTES: u64 = 0x0080_0000 | CLASS_GPU;

/// The title's `__XMemAllocationHooks__` export: `{ alloc(size, attributes) -> ptr,
/// free(ptr, attributes), .. }`. Zero until the base registers it.
static HOOKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The title's heap (`malloc`, and the default `XMemAlloc`) is glibc's, whose mmap threshold
/// grows to 32 MiB as big blocks are freed: later big blocks then come from the arenas and a
/// free leaves them in the process. Fixed thresholds (which also stop the growth) make every
/// block of 1 MiB or more its own mapping, `munmap`ped by `free`, and the arenas' free tops
/// trimmed from 8 MiB.
pub fn return_big_blocks_to_the_host() {
    const M_TRIM_THRESHOLD: libc::c_int = -1;
    const M_MMAP_THRESHOLD: libc::c_int = -3;
    // SAFETY: plain glibc tunables.
    unsafe {
        libc::mallopt(M_MMAP_THRESHOLD, 1 << 20);
        libc::mallopt(M_TRIM_THRESHOLD, 8 << 20);
    }
}

/// Registers the address of the title's `__XMemAllocationHooks__` table; `XMemAlloc` and
/// `XMemFree` go through its functions from then on.
pub fn set_allocation_hooks(table: usize) {
    HOOKS.store(table, Ordering::Release);
    return_big_blocks_to_the_host();
}

/// `XMemAlloc`: through the title's allocation hook when it has one, else the default allocator.
pub fn alloc(size: usize, attributes: u64) -> *mut c_void {
    alloc_with(
        HOOKS.load(Ordering::Acquire) as *const usize,
        size,
        attributes,
    )
}

fn alloc_with(table: *const usize, size: usize, attributes: u64) -> *mut c_void {
    // SAFETY: the base registered the address of the export table of the mapped title.
    let hook = (!table.is_null())
        .then(|| unsafe { table.read() })
        .filter(|&f| f != 0);
    match hook {
        Some(f) => {
            // SAFETY: the table's first entry is the title's `void *(size, attributes)`.
            let f: unsafe extern "win64" fn(usize, u64) -> *mut c_void =
                unsafe { std::mem::transmute(f) };
            unsafe { f(size, attributes) }
        }
        None => default_alloc(size, attributes),
    }
}

/// `XMemFree`: through the title's free hook when it has one, else the default allocator.
pub fn free(address: *mut c_void, attributes: u64) {
    free_with(
        HOOKS.load(Ordering::Acquire) as *const usize,
        address,
        attributes,
    )
}

fn free_with(table: *const usize, address: *mut c_void, attributes: u64) {
    // SAFETY: as in `alloc`.
    let hook = (!table.is_null())
        .then(|| unsafe { table.add(1).read() })
        .filter(|&f| f != 0);
    match hook {
        Some(f) => {
            // SAFETY: the table's second entry is the title's `void (ptr, attributes)`.
            let f: unsafe extern "win64" fn(*mut c_void, u64) = unsafe { std::mem::transmute(f) };
            unsafe { f(address, attributes) }
        }
        None => default_free(address, attributes),
    }
}

/// Graphics-class memory: 64 KiB (or 2 MiB) aligned, where GPU addresses can reach it.
fn gpu_alloc(size: usize, attributes: u64) -> *mut c_void {
    let align_log2 = attributes >> ALIGNMENT_SHIFT & 0x1f;
    let kind = MEM_RESERVE | MEM_COMMIT | if align_log2 >= 21 { PAGES_2M } else { 0 };
    virtual_alloc(std::ptr::null_mut(), size, kind, 3, 0x04)
}

fn default_alloc(size: usize, attributes: u64) -> *mut c_void {
    if attributes & CLASS_MASK == CLASS_GPU {
        return gpu_alloc(size, attributes);
    }
    if attributes & ALLOC_32BIT != 0 {
        return virtual_alloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            3 | ADDRESS_32BIT,
            0x04,
        );
    }
    let align = (1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f)).max(16);
    let mut raw = std::ptr::null_mut();
    let Some(total) = size.checked_add(HEADER) else {
        return std::ptr::null_mut();
    };
    // SAFETY: the alignment is a power of two and a multiple of a pointer's size.
    if unsafe { libc::posix_memalign(&mut raw, align, total + align) } != 0 {
        return std::ptr::null_mut();
    }
    // SAFETY: `raw` holds `total + align` bytes; the user pointer, `align` bytes in, keeps
    // the header below it inside the block.
    unsafe {
        let user = (raw as *mut u8).add(align);
        (user.sub(HEADER) as *mut [usize; 2]).write([raw as usize, size]);
        user as *mut c_void
    }
}

fn default_free(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    if attributes & (CLASS_MASK) == CLASS_GPU || attributes & ALLOC_32BIT != 0 {
        let _ = nativ64::api_ms_win_core_memory::release(address as usize);
        return;
    }
    // SAFETY: the address came from `default_alloc`, which wrote the header.
    unsafe {
        let [raw, _] = (address as *const u8)
            .sub(HEADER)
            .cast::<[usize; 2]>()
            .read();
        libc::free(raw as *mut c_void);
    }
}

nativ64::win32! {
    module "xmem";

    // `*count` pages are allocated into `pages` (all or none); on failure `*count` is 0.
    Native fn XMemAllocatePhysicalPages(_attributes: u64, count: *mut usize, pages: *mut usize) -> i32 {
        if count.is_null() || pages.is_null() || *count == 0 {
            nativ64::api_ms_win_core_errorhandling::set_last_error(87); // ERROR_INVALID_PARAMETER
            return 0;
        }
        match take_physical_pages(*count) {
            Some(v) => {
                for (i, slot) in v.iter().enumerate() {
                    pages.add(i).write(page_number(*slot));
                }
                1
            }
            None => {
                *count = 0;
                nativ64::api_ms_win_core_errorhandling::set_last_error(8); // ERROR_NOT_ENOUGH_MEMORY
                0
            }
        }
    }

    // Frees pages by number: the file range is punched out (the memory goes back to the host)
    // and the views still mapped become fresh zero memory of their own, which no longer
    // reaches the file; the GPU backend is told, to drop what it imported over them.
    Native fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> i32 {
        if pages.is_null() {
            return 0;
        }
        let mut p = pool().lock().unwrap_or_else(|e| e.into_inner());
        for &number in std::slice::from_raw_parts(pages, n) {
            let Some(slot) = live_slot(&p, number) else { continue };
            p.live.remove(&slot);
            p.free.push(slot);
            p.in_use -= 1;
            p.mapped.remove(&slot);
            let views: Vec<usize> = p.views.iter().filter(|(_, s)| **s == slot).map(|(a, _)| *a).collect();
            for a in views {
                p.views.remove(&a);
                let protect = view_protect(a);
                libc::mmap(
                    a as *mut c_void,
                    PHYSICAL_PAGE,
                    host_prot(protect),
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED,
                    -1,
                    0,
                );
                nativ64::api_ms_win_core_memory::note_change(a, PHYSICAL_PAGE);
            }
            if let Some(fd) = p.fd {
                // Reused slots start zeroed.
                libc::fallocate(
                    fd,
                    libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                    (slot * PHYSICAL_PAGE) as libc::off_t,
                    PHYSICAL_PAGE as libc::off_t,
                );
            }
        }
        1
    }

    // Maps `n` consecutive pages at `address`, in place of what is there; null `pages` unmaps.
    // Returns `address`, or null (unknown page, address outside a reservation).
    Native fn XMemMapPhysicalPages(address: *mut c_void, n: usize, pages: *const usize) -> *mut c_void {
        if map_physical(address, n, pages) {
            address
        } else {
            nativ64::api_ms_win_core_errorhandling::set_last_error(87); // ERROR_INVALID_PARAMETER
            std::ptr::null_mut()
        }
    }

    // One page at each address; null `pages` or a page number 0 unmaps that address.
    Native fn XMemMapPhysicalPagesScatter(addresses: *const *mut c_void, n: usize, pages: *const usize) -> i32 {
        if addresses.is_null() {
            return 0;
        }
        for i in 0..n {
            let address = addresses.add(i).read();
            let page = if pages.is_null() || pages.add(i).read() == 0 {
                std::ptr::null()
            } else {
                pages.add(i)
            };
            if !map_physical(address, 1, page) {
                return 0;
            }
        }
        1
    }

    // shortcut: every field is filled whatever `which` asks for; use is not split by bandwidth
    // class (GPU-optimal use reads 0).
    Native fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> i32 {
        if out.is_null() {
            return 0;
        }
        let resident = std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|s| s.split_whitespace().nth(1)?.parse::<usize>().ok())
            .unwrap_or(0)
            * PAGE_SIZE;
        out.write(WorkingSet {
            title_limit: TITLE_MEMORY - TRANSFERRED.load(Ordering::Relaxed),
            title_used: resident,
            gpu_optimal_limit: GPU_OPTIMAL_MEMORY,
            ..Default::default()
        });
        1
    }

    // Hands title memory (standard, in 2 MiB pages) to the system; kind 0 is the only transfer a
    // retail console has (the others involve tools memory).
    Native fn XMemTransferMemory(kind: u32, size: usize) -> i32 {
        if size == 0 {
            return 1;
        }
        let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
            return 0;
        };
        let standard = TITLE_MEMORY - GPU_OPTIMAL_MEMORY;
        (kind == 0
            && TRANSFERRED
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                    t.checked_add(size).filter(|&t| t <= standard)
                })
                .is_ok()) as i32
    }

    Native fn XMemVirtualAlloc(base: *mut c_void, size: usize, kind: u32, flags: u64, protect: u32) -> *mut c_void {
        virtual_alloc(base, size, kind, flags, protect)
    }

    Native fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
        default_alloc(size, attributes)
    }

    Native fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
        default_free(address, attributes)
    }

    // The title's `__XMemAllocationHooks__` export is consulted (see `set_allocation_hooks`).
    Native fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
        alloc(size, attributes)
    }

    Native fn XMemFree(address: *mut c_void, attributes: u64) {
        free(address, attributes)
    }

    // A retail console has no tools memory.
    Native fn XMemIsToolsMemAvailable() -> i32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RW: u32 = 0x04;

    #[test]
    fn graphics_memory_lands_at_or_above_4gib() {
        let p = unsafe {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * MIB,
                MEM_RESERVE | MEM_COMMIT,
                3,
                RW,
            )
        };
        assert!(!p.is_null());
        assert!(p as usize >= 1 << 32 && (p as usize) < 1 << 40);
        unsafe { (p as *mut u8).add(3 * MIB - 1).write(7) };
    }

    #[test]
    fn two_mib_pages_are_aligned() {
        let p = unsafe {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                MIB,
                MEM_RESERVE | MEM_COMMIT | PAGES_2M,
                3,
                RW,
            )
        };
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
    }

    #[test]
    fn reserve_then_commit() {
        let p = unsafe { XMemVirtualAlloc(std::ptr::null_mut(), 1 << 20, MEM_RESERVE, 3, RW) };
        assert!(!p.is_null());
        let q = unsafe { XMemVirtualAlloc(p, 4096, MEM_COMMIT, 3, RW) };
        assert_eq!(q, p);
        unsafe { (p as *mut u8).write(1) };
        assert!(unsafe { XMemVirtualAlloc(std::ptr::null_mut(), 0, MEM_RESERVE, 3, RW) }.is_null());
    }

    #[test]
    fn alloc_32bit_below_4gib_and_free() {
        let p = unsafe { XMemAllocDefault(64 << 10, ALLOC_32BIT) };
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        unsafe { (p as *mut u8).write(1) };
        unsafe { XMemFreeDefault(p, ALLOC_32BIT) };
    }

    #[test]
    fn physical_pages_are_distinct_slots() {
        let mut n = 5usize;
        let mut pages = [0usize; 5];
        assert_eq!(
            unsafe { XMemAllocatePhysicalPages(0x380, &mut n, pages.as_mut_ptr()) },
            1
        );
        assert_eq!(n, 5);
        let mut sorted = pages.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 5);
        let mut m = 2usize;
        let mut more = [0usize; 2];
        assert_eq!(
            unsafe { XMemAllocatePhysicalPages(0, &mut m, more.as_mut_ptr()) },
            1
        );
        assert!(more.iter().all(|p| !pages.contains(p)));
    }

    #[test]
    fn physical_pages_errors() {
        let mut pages = [0usize; 1];
        let mut n = 1usize;
        assert_eq!(
            unsafe { XMemAllocatePhysicalPages(0, std::ptr::null_mut(), pages.as_mut_ptr()) },
            0
        );
        assert_eq!(
            unsafe { XMemAllocatePhysicalPages(0, &mut n, std::ptr::null_mut()) },
            0
        );
        let mut zero = 0usize;
        assert_eq!(
            unsafe { XMemAllocatePhysicalPages(0, &mut zero, pages.as_mut_ptr()) },
            0
        );
        // more than the pool allows: refused, count reset, nothing taken
        let mut huge = MAX_PHYSICAL_PAGES + 1;
        assert_eq!(
            unsafe { XMemAllocatePhysicalPages(0, &mut huge, pages.as_mut_ptr()) },
            0
        );
        assert_eq!(huge, 0);
    }

    #[test]
    fn placeholders_split_then_map_physical_pages() {
        unsafe {
            let base =
                XMemVirtualAlloc(std::ptr::null_mut(), 4 * PHYSICAL_PAGE, MEM_RESERVE, 3, RW);
            assert!(!base.is_null());
            let b = base as usize;
            // split into placeholders, then replace the first one as the title does
            let kind = MEM_RESERVE | MEM_REPLACE_PLACEHOLDER | PAGES_2M | 0x40_0000;
            assert_eq!(
                nativ64::api_ms_win_core_memory::free(b, PHYSICAL_PAGE, 0x8002),
                Ok(())
            );
            assert_eq!(XMemVirtualAlloc(base, PHYSICAL_PAGE, kind, 3, RW), base);
            // a range that is not exactly one placeholder cannot be replaced
            assert!(XMemVirtualAlloc(base, 2 * PHYSICAL_PAGE, kind, 3, RW).is_null());
            let mut n = 2usize;
            let mut pages = [0usize; 2];
            assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
            assert!(pages.iter().all(|p| p % 16 == 0 && *p != 0));
            assert_eq!(XMemMapPhysicalPages(base, 1, pages.as_ptr()), base);
            let next = (b + PHYSICAL_PAGE) as *mut c_void;
            assert_eq!(XMemMapPhysicalPages(next, 1, pages.as_ptr().add(1)), next);
            // a run spanning two allocations is refused
            assert!(XMemMapPhysicalPages(base, 2, pages.as_ptr()).is_null());
            (b as *mut u8).write(5);
            ((b + PHYSICAL_PAGE) as *mut u8).write(6);
            // the same pages seen at a second address alias
            let second = XMemVirtualAlloc(std::ptr::null_mut(), PHYSICAL_PAGE, MEM_RESERVE, 3, RW);
            assert_eq!(
                XMemMapPhysicalPages(second, 1, pages.as_ptr().add(1)),
                second
            );
            assert_eq!((second as *const u8).read(), 6);
            // errors: unknown page, address outside any reservation, unmapped address
            assert!(XMemMapPhysicalPages(base, 1, [0x1230usize].as_ptr()).is_null());
            assert!(
                XMemMapPhysicalPages(0x7000_0000_0000 as *mut c_void, 1, pages.as_ptr()).is_null()
            );
            assert!(XMemMapPhysicalPages(null_mut_misaligned(), 1, pages.as_ptr()).is_null());
            // scatter, with a 0 page unmapping
            let at = [base, (b + PHYSICAL_PAGE) as *mut c_void];
            assert_eq!(
                XMemMapPhysicalPagesScatter(at.as_ptr(), 2, [pages[1], 0].as_ptr()),
                1
            );
            assert_eq!((b as *const u8).read(), 6);
            assert_eq!(
                XMemMapPhysicalPagesScatter(std::ptr::null(), 1, pages.as_ptr()),
                0
            );
            assert_eq!(XMemMapPhysicalPages(base, 1, std::ptr::null()), base);
            // free: twice is harmless, null fails
            assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
            assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
            assert_eq!(XMemFreePhysicalPages(1, std::ptr::null()), 0);
            assert!(XMemMapPhysicalPages(base, 1, pages.as_ptr()).is_null());
        }
    }

    /// A reserved region of `n` pages, and `m` physical pages mapped at its start.
    unsafe fn region_with_pages(n: usize, m: usize) -> (usize, Vec<usize>) {
        let base = XMemVirtualAlloc(std::ptr::null_mut(), n * PHYSICAL_PAGE, MEM_RESERVE, 3, RW);
        assert!(!base.is_null());
        let mut count = m;
        let mut pages = vec![0usize; m];
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut count, pages.as_mut_ptr()),
            1
        );
        assert_eq!(XMemMapPhysicalPages(base, m, pages.as_ptr()), base);
        (base as usize, pages)
    }

    /// True when the pool file holds data in the slot's range (`SEEK_DATA`), whatever other
    /// tests do with other slots.
    fn slot_has_data(number: usize) -> bool {
        let p = pool().lock().unwrap_or_else(|e| e.into_inner());
        let slot = live_slot(&p, number).expect("live slot");
        let at = (slot * PHYSICAL_PAGE) as libc::off_t;
        // SAFETY: plain lseek on the pool's own descriptor.
        let data = unsafe { libc::lseek(p.fd.unwrap(), at, libc::SEEK_DATA) };
        data >= 0 && data < at + PHYSICAL_PAGE as libc::off_t
    }

    #[test]
    fn freed_physical_pages_leave_the_file_and_their_views_become_private_zeros() {
        unsafe {
            let (base, pages) = region_with_pages(4, 4);
            std::ptr::write_bytes(base as *mut u8, 0xab, 4 * PHYSICAL_PAGE);
            assert!(pages.iter().all(|&n| slot_has_data(n)));
            assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
            assert!(!is_pool_backed(base) && !is_pool_backed(base + PHYSICAL_PAGE));
            assert!(is_pool_backed(base + 2 * PHYSICAL_PAGE));
            assert_eq!((base as *const u8).read(), 0, "a freed view reads zero");
            assert_eq!(((base + 2 * PHYSICAL_PAGE) as *const u8).read(), 0xab);
            assert!(slot_has_data(pages[2]) && slot_has_data(pages[3]));
            // a write through a freed view stays private: it never reaches the file, and a
            // slot taken again starts empty
            std::ptr::write_bytes(base as *mut u8, 0xcd, 2 * PHYSICAL_PAGE);
            let mut again = 2usize;
            let mut more = [0usize; 2];
            assert_eq!(
                XMemAllocatePhysicalPages(0, &mut again, more.as_mut_ptr()),
                1
            );
            assert!(
                more.iter().all(|&n| !slot_has_data(n)),
                "reused slots start empty"
            );
            XMemFreePhysicalPages(2, more.as_ptr());
            XMemFreePhysicalPages(2, pages[2..].as_ptr());
        }
    }

    #[test]
    fn alias_pages_share_memory_and_unmap_reads_zero() {
        unsafe {
            let (pool, _pages) = region_with_pages(2, 2);
            let dst = XMemVirtualAlloc(std::ptr::null_mut(), 6 * PHYSICAL_PAGE, MEM_RESERVE, 3, RW)
                as usize;
            // write through the pool, read at the destination and back
            (pool as *mut u8).write(11);
            ((pool + PHYSICAL_PAGE) as *mut u8).write(22);
            assert!(alias_pages(dst, &[pool, pool + PHYSICAL_PAGE]));
            assert_eq!((dst as *const u8).read(), 11);
            assert_eq!(((dst + PHYSICAL_PAGE) as *const u8).read(), 22);
            (dst as *mut u8).add(1).write(33);
            assert_eq!((pool as *const u8).add(1).read(), 33);
            // one page repeated
            let rep = dst + 2 * PHYSICAL_PAGE;
            assert!(alias_pages(rep, &[pool; 3]));
            for i in 0..3 {
                assert_eq!(((rep + i * PHYSICAL_PAGE) as *const u8).read(), 11);
            }
            ((rep + PHYSICAL_PAGE) as *mut u8).write(44);
            assert_eq!((pool as *const u8).read(), 44);
            assert_eq!((dst as *const u8).read(), 44);
            // unmapped reads zero, and no longer writes through
            assert!(unmap_pages(dst, 1));
            assert_eq!((dst as *const u8).read(), 0);
            (dst as *mut u8).write(9);
            assert_eq!((pool as *const u8).read(), 44);
            // not physical memory, misaligned, empty: nothing changes
            let plain =
                XMemVirtualAlloc(std::ptr::null_mut(), PHYSICAL_PAGE, MEM_RESERVE, 3, RW) as usize;
            assert!(!alias_pages(dst, &[plain]));
            assert!(!alias_pages(dst, &[pool, plain]));
            assert_eq!((dst as *const u8).read(), 9);
            assert!(!alias_pages(dst + 1, &[pool]));
            assert!(!alias_pages(dst, &[]));
        }
    }

    #[test]
    fn remapping_the_pool_updates_the_reverse_table() {
        unsafe {
            let (pool, pages) = region_with_pages(2, 2);
            let dst =
                XMemVirtualAlloc(std::ptr::null_mut(), PHYSICAL_PAGE, MEM_RESERVE, 3, RW) as usize;
            (pool as *mut u8).write(1);
            // the pool address now views the second physical page
            assert_eq!(
                XMemMapPhysicalPages(pool as *mut c_void, 1, pages.as_ptr().add(1)),
                pool as *mut c_void
            );
            (pool as *mut u8).write(2);
            assert!(alias_pages(dst, &[pool]));
            assert_eq!((dst as *const u8).read(), 2);
            // unmapped, then freed: no longer a source
            assert_eq!(
                XMemMapPhysicalPages(pool as *mut c_void, 1, std::ptr::null()),
                pool as *mut c_void
            );
            assert!(!alias_pages(dst, &[pool]));
            let (pool2, pages2) = region_with_pages(1, 1);
            assert!(alias_pages(dst, &[pool2]));
            assert_eq!(XMemFreePhysicalPages(1, pages2.as_ptr()), 1);
            assert!(!alias_pages(dst, &[pool2]));
        }
    }

    fn null_mut_misaligned() -> *mut c_void {
        0x1234 as *mut c_void
    }

    #[test]
    fn working_set_and_transfer() {
        unsafe {
            assert_eq!(XMemGetWorkingSetStatistics(0, std::ptr::null_mut()), 0);
            let mut w = WorkingSet::default();
            assert_eq!(XMemGetWorkingSetStatistics(0, &mut w), 1);
            assert!(w.title_limit > w.gpu_optimal_limit && w.title_used > 0);
            assert_eq!(XMemTransferMemory(0, 0), 1);
            assert_eq!(XMemTransferMemory(1, 4096), 0);
            assert_eq!(XMemTransferMemory(0, usize::MAX), 0);
            assert_eq!(XMemTransferMemory(0, 4096), 1);
            assert_eq!(XMemTransferMemory(0, 16 << 30), 0);
        }
    }

    unsafe extern "win64" fn test_alloc(size: usize, attributes: u64) -> *mut c_void {
        // marks the call: the size and the attributes end up in the returned "address"
        (size ^ attributes as usize) as *mut c_void
    }

    unsafe extern "win64" fn test_free(address: *mut c_void, _attributes: u64) {
        FREED.store(address as usize, Ordering::SeqCst);
    }

    static FREED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    #[test]
    fn title_hooks_see_alloc_and_free() {
        let table = [
            test_alloc as *const () as usize,
            test_free as *const () as usize,
        ];
        let attrs = GPU_ALLOC_ATTRIBUTES | 16 << ALIGNMENT_SHIFT;
        let p = alloc_with(table.as_ptr(), 0x1000, attrs);
        assert_eq!(p as usize, 0x1000 ^ attrs as usize);
        free_with(table.as_ptr(), 0x1234 as *mut c_void, attrs);
        assert_eq!(FREED.load(Ordering::SeqCst), 0x1234);
    }

    #[test]
    fn gpu_class_memory_is_graphics_memory_without_hooks() {
        let attrs = GPU_ALLOC_ATTRIBUTES | 16 << ALIGNMENT_SHIFT;
        let p = alloc_with(std::ptr::null(), 3 * MIB, attrs);
        assert!(!p.is_null());
        assert!(p as usize >= 1 << 32 && (p as usize) < 1 << 40);
        assert_eq!(p as usize % (64 << 10), 0);
        unsafe { (p as *mut u8).add(3 * MIB - 1).write(7) };
        free_with(std::ptr::null(), p, attrs);
    }

    #[test]
    fn alloc_honours_alignment() {
        let attrs = 12u64 << ALIGNMENT_SHIFT;
        let p = unsafe { XMemAlloc(100, attrs) };
        assert!(!p.is_null());
        assert_eq!(p as usize % 4096, 0);
        unsafe { XMemFree(p, attrs) };
        assert_eq!(unsafe { XMemIsToolsMemAvailable() }, 0);
    }
}
