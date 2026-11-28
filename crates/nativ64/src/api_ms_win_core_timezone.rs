//! `api-ms-win-core-timezone`: `FILETIME` <-> `SYSTEMTIME` conversions (UTC). Only the two
//! conversions the title imports are here; the time zone queries (`GetTimeZoneInformation`
//! and friends) are not imported and stay absent.
//!
//! The calendar is Windows': years 1601 through 30827, a date that does not exist (February 30)
//! or a `FILETIME` of 2^63 or more fails with `ERROR_INVALID_PARAMETER` and leaves the output
//! untouched. See `filetime.rs`.

use crate::api_ms_win_core_errorhandling::set_last_error;
use crate::filetime;

const ERROR_INVALID_PARAMETER: u32 = 87;

unsafe fn filetime_to_systemtime(ft: *const u64, st: *mut u16) -> i32 {
    if ft.is_null() || st.is_null() {
        set_last_error(ERROR_INVALID_PARAMETER);
        return 0;
    }
    // SAFETY: `ft` is a non-NULL FILETIME; it may be 4-byte aligned only.
    let Some(sys) = filetime::to_systemtime(unsafe { ft.read_unaligned() }) else {
        set_last_error(ERROR_INVALID_PARAMETER);
        return 0;
    };
    for (i, v) in sys.iter().enumerate() {
        // SAFETY: `st` is a non-NULL SYSTEMTIME of eight WORDs.
        unsafe { st.add(i).write_unaligned(*v) };
    }
    1
}

unsafe fn systemtime_to_filetime(st: *const u16, ft: *mut u64) -> i32 {
    if st.is_null() || ft.is_null() {
        set_last_error(ERROR_INVALID_PARAMETER);
        return 0;
    }
    let mut sys = [0u16; 8];
    for (i, v) in sys.iter_mut().enumerate() {
        // SAFETY: `st` is a non-NULL SYSTEMTIME of eight WORDs.
        *v = unsafe { st.add(i).read_unaligned() };
    }
    let Some(value) = filetime::from_systemtime(&sys) else {
        set_last_error(ERROR_INVALID_PARAMETER);
        return 0;
    };
    // SAFETY: `ft` is a non-NULL FILETIME.
    unsafe { ft.write_unaligned(value) };
    1
}

win32! {
    module "api-ms-win-core-timezone" => "kernel32";
    Native fn FileTimeToSystemTime(ft: *const u64, st: *mut u16) -> i32 {
        filetime_to_systemtime(ft, st)
    }
    Native fn SystemTimeToFileTime(st: *const u16, ft: *mut u64) -> i32 {
        systemtime_to_filetime(st, ft)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_ms_win_core_errorhandling::GetLastError;
    use crate::lookup;

    #[test]
    fn registered_under_both_names() {
        for name in ["FileTimeToSystemTime", "SystemTimeToFileTime"] {
            assert!(
                lookup("api-ms-win-core-timezone-l1-1-0.dll", name).is_some(),
                "{name}"
            );
            assert!(lookup("kernel32.dll", name).is_some(), "{name}");
        }
    }

    #[test]
    fn epoch_round_trips() {
        let epoch = filetime::UNIX_EPOCH_AS_FILETIME;
        let mut st = [0u16; 8];
        let mut back = 0u64;
        unsafe {
            assert_eq!(FileTimeToSystemTime(&epoch, st.as_mut_ptr()), 1);
            assert_eq!(st, [1970, 1, 4, 1, 0, 0, 0, 0]);
            assert_eq!(SystemTimeToFileTime(st.as_ptr(), &mut back), 1);
            assert_eq!(back, epoch);
        }
    }

    #[test]
    fn failures_set_the_error_and_keep_the_output() {
        let mut st = [0x1234u16; 8];
        let mut ft = 0xdead_beefu64;
        unsafe {
            // 2^63 ticks and more.
            let huge = 1u64 << 63;
            assert_eq!(FileTimeToSystemTime(&huge, st.as_mut_ptr()), 0);
            assert_eq!(GetLastError(), ERROR_INVALID_PARAMETER);
            assert_eq!(st, [0x1234; 8]);
            // February 30.
            set_last_error(0);
            let bad = [2021u16, 2, 0, 30, 0, 0, 0, 0];
            assert_eq!(SystemTimeToFileTime(bad.as_ptr(), &mut ft), 0);
            assert_eq!(GetLastError(), ERROR_INVALID_PARAMETER);
            assert_eq!(ft, 0xdead_beef);
            // NULL pointers.
            set_last_error(0);
            assert_eq!(
                FileTimeToSystemTime(std::ptr::null(), std::ptr::null_mut()),
                0
            );
            assert_eq!(GetLastError(), ERROR_INVALID_PARAMETER);
            set_last_error(0);
            assert_eq!(
                SystemTimeToFileTime(std::ptr::null(), std::ptr::null_mut()),
                0
            );
            assert_eq!(GetLastError(), ERROR_INVALID_PARAMETER);
        }
    }

    #[test]
    fn unaligned_buffers_work() {
        let mut raw = [0u8; 40];
        let ft = filetime::UNIX_EPOCH_AS_FILETIME;
        let mut st = [0u16; 8];
        unsafe {
            raw.as_mut_ptr().add(1).cast::<u64>().write_unaligned(ft);
            assert_eq!(
                FileTimeToSystemTime(raw.as_ptr().add(1).cast(), st.as_mut_ptr()),
                1
            );
        }
        assert_eq!(st[0], 1970);
    }
}
