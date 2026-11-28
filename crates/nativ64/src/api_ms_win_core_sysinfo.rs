use crate::api_ms_win_core_errorhandling::set_last_error;
use crate::api_ms_win_core_profile::monotonic_ns;
use crate::topology::{
    self, RELATION_ALL, RELATION_GROUP, RELATION_NUMA_NODE_EX, RELATION_PROCESSOR_DIE,
    RELATION_PROCESSOR_MODULE,
};

use crate::filetime::UNIX_EPOCH_AS_FILETIME;

/// The current UTC time as a FILETIME value (100 ns intervals since 1601).
fn filetime_now() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid out pointer.
    unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut ts) };
    UNIX_EPOCH_AS_FILETIME + ts.tv_sec as u64 * 10_000_000 + ts.tv_nsec as u64 / 100
}

/// Writes a FILETIME (two little-endian DWORDs, 4-byte aligned) through `out`.
unsafe fn store_filetime(out: *mut u32, t: u64) {
    out.write_unaligned(t as u32);
    out.add(1).write_unaligned((t >> 32) as u32);
}

/// Writes the current time as a SYSTEMTIME (eight little-endian WORDs), in UTC or local time.
unsafe fn store_systemtime(out: *mut u16, local: bool) {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    libc::clock_gettime(libc::CLOCK_REALTIME, &mut ts);
    let mut tm: libc::tm = std::mem::zeroed();
    if local {
        libc::localtime_r(&ts.tv_sec, &mut tm);
    } else {
        libc::gmtime_r(&ts.tv_sec, &mut tm);
    }
    let f = [
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_wday,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec.min(59),
        (ts.tv_nsec / 1_000_000) as i32,
    ];
    for (i, v) in f.iter().enumerate() {
        out.add(i).write_unaligned(*v as u16);
    }
}

const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
const ERROR_INVALID_PARAMETER: u32 = 87;

/// The two-call size protocol of the `GetLogicalProcessorInformation*` functions: `emit` is
/// run once to size the records, and once more to copy them when `*length` is big enough.
/// Otherwise `*length` receives the size needed and the call fails with
/// `ERROR_INSUFFICIENT_BUFFER`.
unsafe fn copy_records(
    buffer: *mut u8,
    length: *mut u32,
    emit: impl Fn(&mut dyn FnMut(&[u8])),
) -> i32 {
    if length.is_null() {
        set_last_error(ERROR_INVALID_PARAMETER);
        return 0;
    }
    let mut need = 0usize;
    emit(&mut |r| need += r.len());
    if (length.read_unaligned() as usize) < need || buffer.is_null() {
        length.write_unaligned(need as u32);
        set_last_error(ERROR_INSUFFICIENT_BUFFER);
        return 0;
    }
    let mut at = 0usize;
    emit(&mut |r| {
        // SAFETY: the caller's buffer holds `*length` >= `need` bytes and `at + r.len()` <= `need`.
        unsafe { std::ptr::copy_nonoverlapping(r.as_ptr(), buffer.add(at), r.len()) };
        at += r.len();
    });
    length.write_unaligned(need as u32);
    1
}

/// `SYSTEM_INFO` (48 bytes): the processors this process may use, 4 KiB pages, 64 KiB
/// allocation granularity, the user half of the address space. The process is never WOW64,
/// so `GetSystemInfo` answers the same.
///
/// The processor count and mask follow this process's affinity, as Windows' do for the
/// processors of the caller's group (the group is the 64 processors, at most, it may run on).
/// A title sizes its worker pools from them, so counting CPUs it cannot be scheduled on (cgroup
/// or `taskset` limits) would oversubscribe it; the topology records use the same mask.
unsafe fn system_info(out: *mut u8) {
    if out.is_null() {
        return;
    }
    let mask = topology::processor_mask();
    let (level, revision) = processor_level_and_revision();
    std::ptr::write_bytes(out, 0, 48);
    out.cast::<u16>().write_unaligned(9); // PROCESSOR_ARCHITECTURE_AMD64
    out.add(4).cast::<u32>().write_unaligned(4096);
    out.add(8).cast::<u64>().write_unaligned(0x1_0000);
    out.add(16).cast::<u64>().write_unaligned(0x7FFF_FFFE_FFFF);
    out.add(24).cast::<u64>().write_unaligned(mask);
    out.add(32).cast::<u32>().write_unaligned(mask.count_ones());
    out.add(36).cast::<u32>().write_unaligned(8664); // PROCESSOR_AMD_X8664
    out.add(40).cast::<u32>().write_unaligned(0x1_0000);
    out.add(44).cast::<u16>().write_unaligned(level);
    out.add(46).cast::<u16>().write_unaligned(revision);
}

/// `wProcessorLevel` (the family) and `wProcessorRevision` (model << 8 | stepping), from CPUID
/// leaf 1 with the extended family and model folded in.
fn processor_level_and_revision() -> (u16, u16) {
    let eax = std::arch::x86_64::__cpuid(1).eax;
    let family = (eax >> 8 & 0xf) as u16;
    let mut model = (eax >> 4 & 0xf) as u16;
    if family == 0x6 || family == 0xf {
        model |= ((eax >> 16 & 0xf) as u16) << 4;
    }
    let level = if family == 0xf {
        family + (eax >> 20 & 0xff) as u16
    } else {
        family
    };
    (level, model << 8 | (eax & 0xf) as u16)
}

/// `MemAvailable` from `/proc/meminfo`, in bytes: free memory plus what the kernel can reclaim
/// (page cache). `sysinfo.freeram` leaves the cache out and reads far too low.
fn mem_available() -> Option<u64> {
    let mut buf = [0u8; 512];
    // SAFETY: the path is a NUL-terminated literal; the descriptor is closed below.
    let fd = unsafe { libc::open(c"/proc/meminfo".as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
    if fd < 0 {
        return None;
    }
    // SAFETY: `buf` is valid for `buf.len()` bytes. MemAvailable is the third line, well
    // within the first 512 bytes.
    let n = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), buf.len()) };
    // SAFETY: `fd` is open and owned here.
    unsafe { libc::close(fd) };
    let text = std::str::from_utf8(buf.get(..usize::try_from(n).ok()?)?).ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb = line["MemAvailable:".len()..]
        .trim()
        .strip_suffix("kB")?
        .trim();
    kb.parse::<u64>().ok().map(|k| k * 1024)
}

win32! {
    module "api-ms-win-core-sysinfo" => "kernel32";
    Native fn GetLogicalProcessorInformation(buffer: *mut u8, length: *mut u32) -> i32 {
        copy_records(buffer, length, |f| topology::topology().for_each_legacy(|e| f(e)))
    }
    // Cores (with SMT), caches, packages, one NUMA node and one group, from the host topology.
    Native fn GetLogicalProcessorInformationEx(
        relationship: u32,
        buffer: *mut u8,
        length: *mut u32
    ) -> i32 {
        if relationship > RELATION_GROUP
            && relationship != RELATION_PROCESSOR_DIE
            && relationship != RELATION_NUMA_NODE_EX
            && relationship != RELATION_PROCESSOR_MODULE
            && relationship != RELATION_ALL
        {
            set_last_error(ERROR_INVALID_PARAMETER);
            return 0;
        }
        copy_records(buffer, length, |f| topology::topology().for_each_ex(relationship, |r| f(r)))
    }
    Native fn GetNativeSystemInfo(out: *mut u8) {
        system_info(out);
    }
    Native fn GetSystemInfo(out: *mut u8) {
        system_info(out);
    }
    // Windows' default clock tick (15.625 ms) with periodic adjustment disabled.
    Native fn GetSystemTimeAdjustment(
        adjustment: *mut u32,
        increment: *mut u32,
        disabled: *mut i32
    ) -> i32 {
        if adjustment.is_null() || increment.is_null() || disabled.is_null() {
            set_last_error(ERROR_INVALID_PARAMETER);
            return 0;
        }
        adjustment.write_unaligned(156_250);
        increment.write_unaligned(156_250);
        disabled.write_unaligned(1);
        1
    }
    Native fn GetSystemTimeAsFileTime(out: *mut u32) {
        store_filetime(out, filetime_now());
    }
    // The clock already has nanosecond resolution, so the precise variant is the same.
    Native fn GetSystemTimePreciseAsFileTime(out: *mut u32) {
        store_filetime(out, filetime_now());
    }
    Native fn GetSystemTime(out: *mut u16) {
        store_systemtime(out, false);
    }
    Native fn GetLocalTime(out: *mut u16) {
        store_systemtime(out, true);
    }
    // MEMORYSTATUSEX (64 bytes): the host's RAM (available = MemAvailable) and swap; virtual
    // space is the user half (128 TiB).
    Native fn GlobalMemoryStatusEx(out: *mut u8) -> i32 {
        if out.is_null() || (out as *const u32).read_unaligned() != 64 {
            set_last_error(ERROR_INVALID_PARAMETER);
            return 0;
        }
        let mut si: libc::sysinfo = std::mem::zeroed();
        libc::sysinfo(&mut si);
        let unit = si.mem_unit.max(1) as u64;
        let total = si.totalram as u64 * unit;
        let avail = mem_available().unwrap_or(si.freeram as u64 * unit).min(total);
        let (swap, swap_free) = (si.totalswap as u64 * unit, si.freeswap as u64 * unit);
        let virt = 0x7FFF_FFFE_FFFFu64;
        let load = (100 - (avail * 100).checked_div(total).unwrap_or(100)) as u32;
        let f = [total, avail, total + swap, avail + swap_free, virt, virt, 0];
        out.add(4).cast::<u32>().write_unaligned(load);
        for (i, v) in f.iter().enumerate() {
            out.add(8 + 8 * i).cast::<u64>().write_unaligned(*v);
        }
        1
    }
    Native fn GetTickCount() -> u32 {
        (monotonic_ns() / 1_000_000) as u32
    }
    Native fn GetTickCount64() -> u64 {
        monotonic_ns() / 1_000_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_memory_status() {
        let mut b = [0u8; 64];
        unsafe {
            assert_eq!(GlobalMemoryStatusEx(b.as_mut_ptr()), 0);
            assert_eq!(GlobalMemoryStatusEx(std::ptr::null_mut()), 0);
            b[0] = 64;
            assert_eq!(GlobalMemoryStatusEx(b.as_mut_ptr()), 1);
        }
        let total = u64::from_le_bytes(b[8..16].try_into().unwrap());
        let avail = u64::from_le_bytes(b[16..24].try_into().unwrap());
        assert!(total > 0 && avail <= total);
        assert!(b[4] <= 100);
    }

    #[test]
    fn systemtime_is_current_and_well_formed() {
        let mut utc = [0u16; 8];
        let mut local = [0u16; 8];
        unsafe {
            GetSystemTime(utc.as_mut_ptr());
            GetLocalTime(local.as_mut_ptr());
        }
        let year = 1970
            + std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                / 31_556_952;
        for t in [utc, local] {
            assert!((t[0] as u64).abs_diff(year) <= 1, "{t:?}");
            assert!((1..=12).contains(&t[1]) && t[2] <= 6 && (1..=31).contains(&t[3]));
            assert!(t[4] < 24 && t[5] < 60 && t[6] < 60 && t[7] < 1000);
        }
    }

    #[test]
    fn filetime_is_current_utc() {
        for f in [
            GetSystemTimeAsFileTime as unsafe extern "win64" fn(*mut u32),
            GetSystemTimePreciseAsFileTime,
        ] {
            let mut ft = [0u32; 2];
            unsafe { f(ft.as_mut_ptr()) };
            let v = ft[0] as u64 | (ft[1] as u64) << 32;
            let unix = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
                / 100;
            let expect = UNIX_EPOCH_AS_FILETIME + unix;
            assert!(expect.abs_diff(v) < 10_000_000, "{v} vs {expect}");
        }
    }

    #[test]
    fn filetime_advances() {
        let mut a = [0u32; 2];
        let mut b = [0u32; 2];
        unsafe {
            GetSystemTimePreciseAsFileTime(a.as_mut_ptr());
            std::thread::sleep(std::time::Duration::from_millis(20));
            GetSystemTimePreciseAsFileTime(b.as_mut_ptr());
        }
        let (a, b) = (
            a[0] as u64 | (a[1] as u64) << 32,
            b[0] as u64 | (b[1] as u64) << 32,
        );
        assert!(b - a >= 150_000);
    }

    #[test]
    fn ticks() {
        unsafe {
            let a = GetTickCount64();
            std::thread::sleep(std::time::Duration::from_millis(30));
            let b = GetTickCount64();
            assert!(b - a >= 25);
            assert!(GetTickCount().abs_diff(b as u32) < 1000);
        }
    }

    use crate::api_ms_win_core_errorhandling::GetLastError;

    #[test]
    fn logical_processors() {
        unsafe {
            let mut len = 0u32;
            assert_eq!(
                GetLogicalProcessorInformation(std::ptr::null_mut(), &mut len),
                0
            );
            assert_eq!(GetLastError(), 122);
            assert!(len >= 64 && len.is_multiple_of(32));
            let mut buf = vec![0u8; len as usize];
            let mut short = len - 32;
            assert_eq!(
                GetLogicalProcessorInformation(buf.as_mut_ptr(), &mut short),
                0
            );
            assert_eq!((GetLastError(), short), (122, len));
            assert_eq!(
                GetLogicalProcessorInformation(buf.as_mut_ptr(), &mut len),
                1
            );
            let mut cores = 0u64;
            for e in buf.chunks(32) {
                let mask = u64::from_le_bytes(e[..8].try_into().unwrap());
                if e[8] == 0 {
                    assert_eq!(cores & mask, 0, "a processor in two cores");
                    cores |= mask;
                }
            }
            assert_eq!(cores, topology::processor_mask());
            let last = &buf[len as usize - 32..];
            assert_eq!(u32::from_le_bytes(last[8..12].try_into().unwrap()), 1);
            assert_eq!(
                GetLogicalProcessorInformation(buf.as_mut_ptr(), std::ptr::null_mut()),
                0
            );
            assert_eq!(GetLastError(), 87);
        }
    }

    #[test]
    fn logical_processors_ex() {
        unsafe {
            let mut len = 0u32;
            assert_eq!(
                GetLogicalProcessorInformationEx(RELATION_ALL, std::ptr::null_mut(), &mut len),
                0
            );
            assert_eq!(GetLastError(), 122);
            let mut buf = vec![0u8; len as usize];
            assert_eq!(
                GetLogicalProcessorInformationEx(RELATION_ALL, buf.as_mut_ptr(), &mut len),
                1
            );
            let (mut at, mut kinds) = (0usize, 0u32);
            while at < len as usize {
                let kind = u32::from_le_bytes(buf[at..at + 4].try_into().unwrap());
                let size = u32::from_le_bytes(buf[at + 4..at + 8].try_into().unwrap());
                assert!(size >= 48 && kind <= 7 && kind != 6, "{kind} {size}");
                kinds |= 1 << kind;
                at += size as usize;
            }
            assert_eq!(at, len as usize);
            // Cores, NUMA node, packages and the group are always there; caches when sysfs has them.
            assert_eq!(kinds & 0b11011, 0b11011);
            // One kind only: the package records.
            let mut plen = len;
            assert_eq!(
                GetLogicalProcessorInformationEx(3, buf.as_mut_ptr(), &mut plen),
                1
            );
            assert!(plen > 0 && plen.is_multiple_of(48) && plen < len);
            assert_eq!(&buf[..4], &3u32.to_le_bytes());
            for ok in [5, 7] {
                let mut l = 0u32;
                GetLogicalProcessorInformationEx(ok, std::ptr::null_mut(), &mut l);
                assert_eq!(GetLastError(), 122, "relationship {ok}");
                let mut one = vec![0u8; l as usize];
                assert_eq!(
                    GetLogicalProcessorInformationEx(ok, one.as_mut_ptr(), &mut l),
                    1
                );
                assert_eq!(&one[..4], &(ok as u32).to_le_bytes());
            }
            for bad in [8, 0xfffe] {
                assert_eq!(
                    GetLogicalProcessorInformationEx(bad, buf.as_mut_ptr(), &mut len),
                    0
                );
                assert_eq!(GetLastError(), 87);
            }
            assert_eq!(
                GetLogicalProcessorInformationEx(0, buf.as_mut_ptr(), std::ptr::null_mut()),
                0
            );
        }
    }

    #[test]
    fn native_system_info() {
        let mut b = [0xAAu8; 48];
        unsafe {
            GetNativeSystemInfo(b.as_mut_ptr());
            GetNativeSystemInfo(std::ptr::null_mut());
        }
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        assert_eq!(u16::from_le_bytes([b[0], b[1]]), 9);
        assert_eq!(u32_at(4), 4096);
        assert_eq!((u64_at(8), u64_at(16)), (0x1_0000, 0x7FFF_FFFE_FFFF));
        assert_eq!(u64_at(24), topology::processor_mask());
        assert_eq!(u32_at(32), u64_at(24).count_ones());
        assert_eq!((u32_at(36), u32_at(40)), (8664, 0x1_0000));
        assert!(u16::from_le_bytes([b[44], b[45]]) >= 6);
        let mut c = [0u8; 48];
        unsafe { GetSystemInfo(c.as_mut_ptr()) };
        assert_eq!(b, c);
    }

    #[test]
    fn system_time_adjustment() {
        let (mut adj, mut inc, mut off) = (0u32, 0u32, 0i32);
        unsafe {
            assert_eq!(GetSystemTimeAdjustment(&mut adj, &mut inc, &mut off), 1);
            assert_eq!((adj, inc, off), (156_250, 156_250, 1));
            assert_eq!(
                GetSystemTimeAdjustment(&mut adj, std::ptr::null_mut(), &mut off),
                0
            );
            assert_eq!(GetLastError(), 87);
        }
    }

    #[test]
    fn available_memory_counts_the_page_cache() {
        let avail = mem_available().unwrap();
        let mut si: libc::sysinfo = unsafe { std::mem::zeroed() };
        unsafe { libc::sysinfo(&mut si) };
        let unit = si.mem_unit.max(1) as u64;
        assert!(avail <= si.totalram as u64 * unit);
        // MemAvailable includes the free pages (give or take what moved between the reads).
        assert!(avail + (64 << 20) >= si.freeram as u64 * unit);
    }
}
