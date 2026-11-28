use crate::api_ms_win_core_errorhandling::set_last_error;
use crate::handles::{self, ThreadStart, ERROR_INVALID_PARAMETER};
use crate::registry::{Export, Status};
use crate::slots::{SlotTable, Values};
use std::cell::RefCell;
use std::ffi::c_void;

pub const TLS_OUT_OF_INDEXES: u32 = 0xFFFF_FFFF;
const ERROR_NOT_ENOUGH_MEMORY: u32 = 8;
const ERROR_INVALID_HANDLE: u32 = 6;
static PRIORITY_BOOST_DISABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// 64 slots plus 1024 expansion slots, as Windows.
const MAX_TLS: usize = 1088;

const STARTUPINFO_SIZE: usize = 104;

static TLS: SlotTable = SlotTable::new(MAX_TLS);

thread_local! {
    static TLS_VALUES: RefCell<Values> = const { RefCell::new(Values::new(&TLS)) };
}

/// `SetThreadAffinityMask` is also imported from `api-ms-win-core-processtopology-obsolete`;
/// `Get/SetThreadGroupAffinity` come from `api-ms-win-core-processtopology`.
pub static ALIASES: &[Export] = &[
    Export {
        module: "api-ms-win-core-processtopology-obsolete",
        name: "SetThreadAffinityMask",
        addr: SetThreadAffinityMask as *const (),
        status: Status::Native,
    },
    Export {
        module: "api-ms-win-core-processtopology",
        name: "GetThreadGroupAffinity",
        addr: GetThreadGroupAffinity as *const (),
        status: Status::Native,
    },
    Export {
        module: "kernel32",
        name: "GetThreadGroupAffinity",
        addr: GetThreadGroupAffinity as *const (),
        status: Status::Native,
    },
    Export {
        module: "api-ms-win-core-processtopology",
        name: "SetThreadGroupAffinity",
        addr: SetThreadGroupAffinity as *const (),
        status: Status::Native,
    },
    Export {
        module: "kernel32",
        name: "SetThreadGroupAffinity",
        addr: SetThreadGroupAffinity as *const (),
        status: Status::Native,
    },
];

/// The `PF_*` features (winnt.h numbering) this machine has, as Windows would answer them:
/// CPUID bits, AVX states only when the OS saves them (XCR0), FSGSBASE only when the kernel
/// enabled it. Computed once.
fn processor_features() -> u64 {
    static FEATURES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *FEATURES.get_or_init(|| {
        use std::arch::x86_64::{__cpuid, __cpuid_count};
        let leaf1 = __cpuid(1);
        let leaf7 = __cpuid_count(7, 0);
        let ext1 = __cpuid(0x8000_0001);
        let osxsave = leaf1.ecx & 1 << 27 != 0;
        let xcr0 = if osxsave {
            // SAFETY: OSXSAVE is set, so XGETBV is enabled; XCR0 is index 0.
            unsafe { std::arch::x86_64::_xgetbv(0) }
        } else {
            0
        };
        let ymm = xcr0 & 0x6 == 0x6;
        let zmm = xcr0 & 0xe6 == 0xe6;
        // SAFETY: getauxval has no preconditions.
        let hwcap2 = unsafe { libc::getauxval(libc::AT_HWCAP2) };
        let bit = |b: bool, n: u32| (b as u64) << n;
        // Every x86-64 CPU: CMPXCHG8B, MMX, SSE, RDTSC, PAE, SSE2, SSE DAZ, NX. Not
        // PF_FASTFAIL_AVAILABLE (23): our `int 0x29` is a catchable access violation, not
        // an uncatchable end with 0xC0000409, so the CRT must use its own fallback.
        let baseline = [2, 3, 6, 8, 9, 10, 11, 12]
            .iter()
            .fold(0, |m, n| m | 1 << n);
        baseline
            | bit(leaf1.ecx & 1 != 0, 13) // SSE3
            | bit(leaf1.ecx & 1 << 13 != 0, 14) // CMPXCHG16B
            | bit(osxsave, 17) // XSAVE
            | bit(hwcap2 & 2 != 0, 22) // RDWRFSGSBASE (HWCAP2_FSGSBASE)
            | bit(leaf1.ecx & 1 << 30 != 0, 28) // RDRAND
            | bit(ext1.edx & 1 << 27 != 0, 32) // RDTSCP
            | bit(leaf7.ecx & 1 << 22 != 0, 33) // RDPID
            | bit(ext1.ecx & 1 << 29 != 0, 35) // MONITORX
            | bit(leaf1.ecx & 1 << 9 != 0, 36) // SSSE3
            | bit(leaf1.ecx & 1 << 19 != 0, 37) // SSE4.1
            | bit(leaf1.ecx & 1 << 20 != 0, 38) // SSE4.2
            | bit(leaf1.ecx & 1 << 28 != 0 && ymm, 39) // AVX
            | bit(leaf7.ebx & 1 << 5 != 0 && ymm, 40) // AVX2
            | bit(leaf7.ebx & 1 << 16 != 0 && zmm, 41) // AVX512F
            | bit(leaf7.ebx & 1 << 9 != 0, 42) // ERMS
    })
}

win32! {
    module "api-ms-win-core-processthreads" => "kernel32";
    // Unknown numbers, SLAT and virtualization firmware answer false.
    Native fn IsProcessorFeaturePresent(feature: u32) -> i32 {
        (feature < 64 && processor_features() >> feature & 1 != 0) as i32
    }
    Native fn GetCurrentProcessId() -> u32 {
        libc::getpid() as u32
    }
    Native fn GetCurrentThreadId() -> u32 {
        libc::syscall(libc::SYS_gettid) as u32
    }
    // TEB offsets 0x10 (stack limit) and 0x08 (stack base): the bounds recorded for the calling thread's stack.
    Native fn GetCurrentThreadStackLimits(low: *mut usize, high: *mut usize) {
        let teb: usize;
        core::arch::asm!("mov {}, gs:[0x30]", out(reg) teb, options(nostack, readonly, preserves_flags));
        let teb = teb as *const usize;
        if !low.is_null() {
            *low = *teb.byte_add(0x10);
        }
        if !high.is_null() {
            *high = *teb.byte_add(0x08);
        }
    }
    // Security attributes and inheritance are not modelled.
    Native fn CreateThread(_attrs: *const c_void, stack: usize, start: Option<ThreadStart>,
                           param: *mut c_void, flags: u32, id: *mut u32) -> usize {
        match handles::create_thread(stack, start, param, flags) {
            Some((h, tid)) => {
                if !id.is_null() {
                    *id = tid;
                }
                h
            }
            None => 0,
        }
    }
    // The pseudo-handle -1, the one `DuplicateHandle` accepts as the current process.
    Native fn GetCurrentProcess() -> usize { usize::MAX }
    // Only the current process (the pseudo-handle) is known; no other process handle exists.
    Native fn GetProcessId(h: usize) -> u32 {
        if h == usize::MAX {
            libc::getpid() as u32
        } else {
            set_last_error(ERROR_INVALID_HANDLE);
            0
        }
    }
    // Ends the process at once, without the CRT's exit handlers, as the real call does.
    Native fn TerminateProcess(h: usize, code: u32) -> i32 {
        if h == usize::MAX {
            libc::_exit(code as i32)
        }
        set_last_error(ERROR_INVALID_HANDLE);
        0
    }
    // Only the current process (the pseudo-handle) can be queried; the boost is a stored flag.
    Native fn SetProcessPriorityBoost(h: usize, disable: i32) -> i32 {
        if h != usize::MAX {
            set_last_error(ERROR_INVALID_HANDLE);
            return 0;
        }
        PRIORITY_BOOST_DISABLED.store(disable != 0, std::sync::atomic::Ordering::Relaxed);
        1
    }
    Native fn GetProcessPriorityBoost(h: usize, disabled: *mut i32) -> i32 {
        if h != usize::MAX {
            set_last_error(ERROR_INVALID_HANDLE);
            return 0;
        }
        if disabled.is_null() {
            set_last_error(ERROR_INVALID_PARAMETER);
            return 0;
        }
        *disabled = PRIORITY_BOOST_DISABLED.load(std::sync::atomic::Ordering::Relaxed) as i32;
        1
    }
    Native fn GetCurrentThread() -> usize { handles::CURRENT_THREAD }
    Native fn GetExitCodeThread(h: usize, code: *mut u32) -> i32 {
        if code.is_null() {
            set_last_error(ERROR_INVALID_PARAMETER);
            return 0;
        }
        match handles::get_exit_code(h) {
            Some(c) => {
                *code = c;
                1
            }
            None => 0,
        }
    }
    Native fn OpenThread(_access: u32, _inherit: i32, id: u32) -> usize { handles::open_thread(id) }
    Native fn ResumeThread(h: usize) -> u32 { handles::resume_thread(h) }
    // Only threads that have not entered their start routine can be suspended.
    Native fn SuspendThread(h: usize) -> u32 { handles::suspend_thread(h) }
    Native fn SetThreadPriority(h: usize, p: i32) -> i32 { handles::set_priority(h, p) }
    Native fn GetThreadPriority(h: usize) -> i32 { handles::get_priority(h) }
    Native fn SetThreadAffinityMask(h: usize, mask: usize) -> usize { handles::set_affinity(h, mask) }
    // `out` is a GROUP_AFFINITY (16 bytes).
    Native fn GetThreadGroupAffinity(h: usize, out: *mut u8) -> i32 {
        handles::get_group_affinity(h, out)
    }
    Native fn SetThreadGroupAffinity(h: usize, new: *const u8, old: *mut u8) -> i32 {
        handles::set_group_affinity(h, new, old)
    }
    // The ideal processor is a scheduler hint Linux has no counterpart for: accepted and
    // dropped. `ideal` and `previous` are PROCESSOR_NUMBERs (group u16, number u8, reserved
    // u8); the previous one reads as group 0, processor 0. Shortcut: the handle is not checked.
    Native fn SetThreadIdealProcessorEx(_h: usize, ideal: *const u8, previous: *mut u8) -> i32 {
        if ideal.is_null() {
            set_last_error(ERROR_INVALID_PARAMETER);
            return 0;
        }
        if !previous.is_null() {
            previous.cast::<u32>().write_unaligned(0);
        }
        1
    }
    Native fn SwitchToThread() -> i32 { handles::switch_to_thread() }
    // TLS: per-thread slots, 1088 at most; a freed slot reads as NULL in every thread.
    Native fn TlsAlloc() -> u32 {
        match TLS.alloc(0) {
            Some(i) => i,
            None => {
                set_last_error(ERROR_NOT_ENOUGH_MEMORY);
                TLS_OUT_OF_INDEXES
            }
        }
    }
    Native fn TlsFree(index: u32) -> i32 {
        if TLS.free(index, &TLS_VALUES) {
            1
        } else {
            set_last_error(ERROR_INVALID_PARAMETER);
            0
        }
    }
    Native fn TlsGetValue(index: u32) -> *mut c_void {
        match TLS.get(index, &TLS_VALUES) {
            Some(v) => {
                set_last_error(0);
                v as *mut c_void
            }
            None => {
                set_last_error(ERROR_INVALID_PARAMETER);
                std::ptr::null_mut()
            }
        }
    }
    // STARTUPINFOW/A are 104 bytes; a process started without startup options has every
    // field zero but `cb` (no std handles, no window placement, NULL desktop and title).
    Native fn GetStartupInfoW(info: *mut u8) {
        if !info.is_null() {
            std::ptr::write_bytes(info, 0, STARTUPINFO_SIZE);
            *(info as *mut u32) = STARTUPINFO_SIZE as u32;
        }
    }
    Native fn GetStartupInfoA(info: *mut u8) {
        GetStartupInfoW(info)
    }
    Native fn TlsSetValue(index: u32, value: *mut c_void) -> i32 {
        if TLS.set(index, value as usize, &TLS_VALUES) {
            1
        } else {
            set_last_error(ERROR_INVALID_PARAMETER);
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::api_ms_win_core_errorhandling::GetLastError;

    #[test]
    fn processor_features_match_the_host() {
        unsafe {
            for f in [2, 3, 6, 8, 9, 10, 11, 12] {
                assert_eq!(IsProcessorFeaturePresent(f), 1, "baseline feature {f}");
            }
            for f in [0, 1, 7, 15, 20, 21, 23, 64, 0xffff_ffff] {
                assert_eq!(IsProcessorFeaturePresent(f), 0, "feature {f}");
            }
            let leaf1 = std::arch::x86_64::__cpuid(1);
            assert_eq!(IsProcessorFeaturePresent(13), (leaf1.ecx & 1) as i32);
            assert_eq!(IsProcessorFeaturePresent(38), (leaf1.ecx >> 20 & 1) as i32);
            // AVX2 implies AVX: both need the OS to save the YMM state.
            if IsProcessorFeaturePresent(40) == 1 {
                assert_eq!(IsProcessorFeaturePresent(39), 1);
            }
        }
        assert!(crate::lookup("kernel32.dll", "IsProcessorFeaturePresent").is_some());
    }

    #[test]
    fn startup_info() {
        unsafe {
            let mut b = [0xAAu8; 112];
            GetStartupInfoW(b.as_mut_ptr());
            assert_eq!(u32::from_le_bytes(b[..4].try_into().unwrap()), 104);
            assert!(b[4..104].iter().all(|&x| x == 0));
            assert_eq!(b[104], 0xAA);
            GetStartupInfoA(std::ptr::null_mut());
        }
    }

    #[test]
    fn ids() {
        unsafe {
            assert_eq!(GetCurrentProcessId(), std::process::id());
            assert_eq!(GetCurrentProcess(), usize::MAX);
            let mut d = 7i32;
            assert_eq!(SetProcessPriorityBoost(usize::MAX, 1), 1);
            assert_eq!(GetProcessPriorityBoost(usize::MAX, &mut d), 1);
            assert_eq!(d, 1);
            assert_eq!(SetProcessPriorityBoost(usize::MAX, 0), 1);
            assert_eq!(GetProcessPriorityBoost(usize::MAX, &mut d), 1);
            assert_eq!(d, 0);
            assert_eq!(SetProcessPriorityBoost(0x1234, 1), 0);
            assert_eq!(GetProcessPriorityBoost(usize::MAX, std::ptr::null_mut()), 0);
            assert!(crate::lookup("kernel32", "GetCurrentProcess").is_some());
            let main = GetCurrentThreadId();
            assert_ne!(main, 0);
            let other = std::thread::spawn(|| GetCurrentThreadId()).join().unwrap();
            assert_ne!(main, other);
        }
    }

    #[test]
    fn tls_roundtrip_and_errors() {
        use crate::api_ms_win_core_errorhandling::GetLastError;
        unsafe {
            let i = TlsAlloc();
            assert_ne!(i, TLS_OUT_OF_INDEXES);
            set_last_error(5);
            assert!(TlsGetValue(i).is_null());
            assert_eq!(GetLastError(), 0);
            assert_eq!(TlsSetValue(i, 0x99 as *mut _), 1);
            assert_eq!(TlsGetValue(i) as usize, 0x99);
            std::thread::spawn(move || assert!(TlsGetValue(i).is_null()))
                .join()
                .unwrap();
            assert_eq!(TlsFree(i), 1);
            for bad in [i, MAX_TLS as u32, TLS_OUT_OF_INDEXES] {
                set_last_error(0);
                assert!(TlsGetValue(bad).is_null());
                assert_eq!(GetLastError(), ERROR_INVALID_PARAMETER);
                set_last_error(0);
                assert_eq!(TlsSetValue(bad, std::ptr::dangling_mut()), 0);
                assert_eq!(GetLastError(), ERROR_INVALID_PARAMETER);
                set_last_error(0);
                assert_eq!(TlsFree(bad), 0);
                assert_eq!(GetLastError(), ERROR_INVALID_PARAMETER);
            }
            // Reallocated index starts NULL.
            let a = TlsAlloc();
            TlsSetValue(a, 3 as *mut _);
            TlsFree(a);
            let b = TlsAlloc();
            assert_eq!(a, b);
            assert!(TlsGetValue(b).is_null());
            TlsFree(b);
        }
    }

    #[test]
    fn ideal_processor_is_accepted() {
        unsafe {
            let ideal = [0u8, 0, 3, 0];
            let mut previous = [0xffu8; 4];
            assert_eq!(
                SetThreadIdealProcessorEx(
                    GetCurrentThread(),
                    ideal.as_ptr(),
                    previous.as_mut_ptr()
                ),
                1
            );
            assert_eq!(previous, [0; 4]);
            assert_eq!(
                SetThreadIdealProcessorEx(GetCurrentThread(), ideal.as_ptr(), std::ptr::null_mut()),
                1
            );
            set_last_error(0);
            assert_eq!(
                SetThreadIdealProcessorEx(
                    GetCurrentThread(),
                    std::ptr::null(),
                    std::ptr::null_mut()
                ),
                0
            );
            assert_eq!(GetLastError(), ERROR_INVALID_PARAMETER);
        }
    }

    #[test]
    fn affinity_alias_resolves() {
        let a = crate::lookup(
            "api-ms-win-core-processtopology-obsolete-l1-1-0.dll",
            "SetThreadAffinityMask",
        )
        .unwrap();
        let b = crate::lookup("kernel32", "SetThreadAffinityMask").unwrap();
        assert_eq!(a.addr, b.addr);
    }
}
