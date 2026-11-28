//! `api-ms-win-crt-runtime`: the UCRT per-thread `errno` and the invalid parameter handler.
//!
//! `errno` is a per-thread `int` private to this crate (not glibc's `errno`). `_errno` returns
//! its address, so a caller can read and write it directly.
//!
//! The invalid parameter handler is looked up thread-local first, then global. The exported
//! `_invalid_parameter*` functions terminate the process when no handler is installed (the UCRT
//! raises a fast-fail); if the handler returns, `_invalid_parameter` returns and
//! `_invalid_parameter_noinfo_noreturn` terminates anyway.
//!
//! Deviation: the crate-internal [`invalid_parameter`], used by the `_s` functions of the other
//! modules, returns when no handler is installed instead of terminating, so that error paths
//! stay testable and a game that ignores the returned error code keeps running.
//!
//! `__setusermatherr` lives in `api_ms_win_crt_math` (its handler is read from there).
//!
//! Also here: the CRT startup (`argv`, `_initterm`, onexit tables), the exit family, CRT threads,
//! `strerror` and a few leftovers. The command line comes from the base (`set_command_line`).
//!
//! Gaps: no CRT signal table (`_seh_filter_*` always continue the search), `_set_new_handler`
//! only stores the handler (`_callnewh` does not call it yet), no `quick_exit`/`_onexit` family.
//!
//! Functions are registered under their API set and under `ucrtbase`, which forwards to them.

use crate::handles::ThreadStart;
use std::cell::{Cell, UnsafeCell};
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

const EINVAL: i32 = 22;

/// `_invalid_parameter_handler`: `(expression, function, file, line, reserved)`.
pub type InvalidParameterHandler =
    unsafe extern "win64" fn(*const u16, *const u16, *const u16, u32, usize);

static GLOBAL_HANDLER: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static ERRNO: Cell<i32> = const { Cell::new(0) };
    static THREAD_HANDLER: Cell<usize> = const { Cell::new(0) };
}

/// Sets the calling thread's CRT `errno`.
pub(crate) fn set_errno(v: i32) {
    ERRNO.with(|e| e.set(v));
}

/// Reads the calling thread's CRT `errno`.
#[cfg(test)]
pub(crate) fn errno() -> i32 {
    ERRNO.with(|e| e.get())
}

/// The handler in effect (thread-local first), or 0.
fn current_handler() -> usize {
    let t = THREAD_HANDLER.with(|h| h.get());
    if t != 0 {
        t
    } else {
        GLOBAL_HANDLER.load(Ordering::Relaxed)
    }
}

/// Calls the handler if there is one; returns whether it was called.
unsafe fn call_handler(
    expr: *const u16,
    func: *const u16,
    file: *const u16,
    line: u32,
    res: usize,
) -> bool {
    let h = current_handler();
    if h == 0 {
        return false;
    }
    let f: InvalidParameterHandler = std::mem::transmute(h);
    f(expr, func, file, line, res);
    true
}

/// Invokes the invalid parameter handler without information (for the `_s` functions of other
/// modules). Sets nothing; returns when the handler returns or when none is installed.
#[track_caller]
pub(crate) fn invalid_parameter() {
    if crate::log::on(crate::log::CRT) {
        eprintln!(
            "nativ64: invalid parameter at {} (handler {})",
            std::panic::Location::caller(),
            if current_handler() != 0 {
                "installed"
            } else {
                "none"
            }
        );
    }
    unsafe {
        call_handler(std::ptr::null(), std::ptr::null(), std::ptr::null(), 0, 0);
    }
}

/// Sets `errno` to `err`, invokes the handler, and returns `err` (the `_VALIDATE_RETURN_ERRCODE`
/// pattern).
#[track_caller]
pub(crate) fn invalid_parameter_err(err: i32) -> i32 {
    set_errno(err);
    invalid_parameter();
    err
}

fn fatal_invalid_parameter() -> ! {
    eprintln!("nativ64: invalid parameter passed to a C runtime function (no handler installed): terminating");
    std::process::abort()
}

// ---------------------------------------------------------------------------------------------
// Command line and argv.

/// The command line given by the base (`set_command_line`), else `/proc/self/cmdline`.
static COMMAND_LINE: Mutex<Option<Vec<String>>> = Mutex::new(None);

/// Sets the process command line (`argv`, program name first) the CRT startup functions expose.
/// The base calls it once before running the game; it has no effect after the first use of
/// `__p___argv`, `_configure_*_argv` or `_get_narrow_winmain_command_line` (the arrays are built
/// once and live for the process).
pub fn set_command_line(args: &[String]) {
    *COMMAND_LINE.lock().unwrap_or_else(|e| e.into_inner()) = Some(args.to_vec());
}

/// C ABI of [`set_command_line`]: `argv` holds `argc` NUL-terminated strings (UTF-8; invalid
/// bytes are replaced). A negative `argc` or a NULL `argv` with `argc > 0` is ignored.
///
/// # Safety
/// `argv` must be NULL or point to `argc` valid NUL-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn nativ64_set_command_line(argc: i32, argv: *const *const c_char) {
    if argc < 0 || (argc > 0 && argv.is_null()) {
        return;
    }
    let args: Vec<String> = (0..argc as usize)
        .map(|i| {
            let p = *argv.add(i);
            if p.is_null() {
                String::new()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        })
        .collect();
    set_command_line(&args);
}

fn default_command_line() -> Vec<String> {
    let raw = std::fs::read("/proc/self/cmdline").unwrap_or_default();
    let mut v: Vec<String> = raw
        .split(|&b| b == 0)
        .map(|s| String::from_utf8_lossy(s).into_owned())
        .collect();
    if v.last().is_some_and(|s| s.is_empty()) {
        v.pop();
    }
    v
}

/// Quotes one argument the way `CommandLineToArgvW` / the CRT parse it back.
fn quote_arg(a: &str) -> String {
    if !a.is_empty() && !a.contains([' ', '\t', '"']) {
        return a.to_string();
    }
    let mut out = String::from("\"");
    let mut backslashes = 0;
    for c in a.chars() {
        match c {
            '\\' => backslashes += 1,
            '"' => {
                out.push_str(&"\\".repeat(backslashes * 2 + 1));
                out.push('"');
                backslashes = 0;
            }
            _ => {
                out.push_str(&"\\".repeat(backslashes));
                out.push(c);
                backslashes = 0;
            }
        }
    }
    out.push_str(&"\\".repeat(backslashes * 2));
    out.push('"');
    out
}

/// The part of `cmdline` after the program name and the blanks that follow it. The program name
/// ends at the first blank outside quotes (every `"` toggles quoting, there is no escape).
fn after_program_name(cmdline: &str) -> &str {
    let mut in_quotes = false;
    let mut end = cmdline.len();
    for (i, c) in cmdline.char_indices() {
        if c == '"' {
            in_quotes = !in_quotes;
        } else if !in_quotes && (c == ' ' || c == '\t') {
            end = i;
            break;
        }
    }
    cmdline[end..].trim_start_matches([' ', '\t'])
}

/// The arrays handed to the program; built once, never freed or moved.
struct Argv {
    argc: i32,
    narrow: *mut *mut c_char,
    wide: *mut *mut u16,
    /// Command line after the program name (points into `cmdline`).
    winmain: *const c_char,
    /// The whole command line, program name included, narrow and wide (`GetCommandLine*`).
    full_a: *const c_char,
    full_w: *const u16,
    #[allow(dead_code)]
    #[allow(clippy::type_complexity)]
    keep: (
        Vec<CString>,
        Vec<Vec<u16>>,
        Vec<*mut c_char>,
        Vec<*mut u16>,
        CString,
        Vec<u16>,
    ),
}

// Built once and only read (the program may write through the pointers it is given, as in a CRT).
unsafe impl Send for Argv {}
unsafe impl Sync for Argv {}

fn build_argv() -> Argv {
    let args = COMMAND_LINE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_else(default_command_line);
    let narrow: Vec<CString> = args
        .iter()
        .map(|a| CString::new(a.replace('\0', "")).unwrap_or_default())
        .collect();
    let wide: Vec<Vec<u16>> = args
        .iter()
        .map(|a| a.replace('\0', "").encode_utf16().chain([0]).collect())
        .collect();
    let mut np: Vec<*mut c_char> = narrow.iter().map(|s| s.as_ptr() as *mut c_char).collect();
    np.push(std::ptr::null_mut());
    let mut wp: Vec<*mut u16> = wide.iter().map(|s| s.as_ptr() as *mut u16).collect();
    wp.push(std::ptr::null_mut());
    let full = args
        .iter()
        .map(|a| quote_arg(a))
        .collect::<Vec<_>>()
        .join(" ");
    let full = CString::new(full.replace('\0', "")).unwrap_or_default();
    let skip =
        after_program_name(full.to_str().unwrap_or("")).as_ptr() as usize - full.as_ptr() as usize;
    let winmain = unsafe { full.as_ptr().add(skip) };
    let full_w: Vec<u16> = full.to_string_lossy().encode_utf16().chain([0]).collect();
    Argv {
        argc: args.len() as i32,
        narrow: np.as_mut_ptr(),
        wide: wp.as_mut_ptr(),
        winmain,
        full_a: full.as_ptr(),
        full_w: full_w.as_ptr(),
        keep: (narrow, wide, np, wp, full, full_w),
    }
}

/// The whole command line, narrow and wide, valid for the process.
pub(crate) fn full_command_line() -> (*const c_char, *const u16) {
    let a = argv();
    (a.full_a, a.full_w)
}

fn argv() -> &'static Argv {
    static ARGV: OnceLock<Argv> = OnceLock::new();
    ARGV.get_or_init(build_argv)
}

static APP_TYPE: AtomicI32 = AtomicI32::new(0);

/// Calls each non-null function of `[start, end)`; with `stop_on_error`, stops at the first
/// non-zero result and returns it.
unsafe fn init_term(
    start: *const Option<unsafe extern "win64" fn() -> i32>,
    end: *const Option<unsafe extern "win64" fn() -> i32>,
    stop_on_error: bool,
) -> i32 {
    if start.is_null() || end.is_null() {
        return 0;
    }
    let mut p = start;
    while p < end {
        if let Some(f) = *p {
            let r = f();
            if stop_on_error && r != 0 {
                return r;
            }
        }
        p = p.add(1);
    }
    0
}

// ---------------------------------------------------------------------------------------------
// onexit tables.

/// `_onexit_table_t`: the caller owns the struct, we own the array `first..end` (libc `malloc`).
#[repr(C)]
pub struct OnexitTable {
    first: *mut usize,
    last: *mut usize,
    end: *mut usize,
}

/// What an initialized, empty table points at (never dereferenced nor freed).
static EMPTY_SLOT: usize = 0;

fn empty_marker() -> *mut usize {
    std::ptr::addr_of!(EMPTY_SLOT) as *mut usize
}

unsafe fn onexit_init(t: *mut OnexitTable) -> i32 {
    if t.is_null() {
        return -1;
    }
    let t = &mut *t;
    // Same rule as the UCRT: an uninitialized table has `first == end`.
    if t.first == t.end {
        let m = empty_marker();
        (t.first, t.last, t.end) = (m, m, m);
    }
    0
}

unsafe fn onexit_register(t: *mut OnexitTable, f: usize) -> i32 {
    if t.is_null() {
        return -1;
    }
    let t = &mut *t;
    if t.last == t.end {
        let owned = !t.first.is_null() && t.first != empty_marker();
        let count = if owned {
            t.last.offset_from(t.first) as usize
        } else {
            0
        };
        let cap = (count * 2).max(4);
        let p = libc::realloc(
            if owned {
                t.first as *mut c_void
            } else {
                std::ptr::null_mut()
            },
            cap * std::mem::size_of::<usize>(),
        ) as *mut usize;
        if p.is_null() {
            return -1;
        }
        (t.first, t.last, t.end) = (p, p.add(count), p.add(cap));
    }
    *t.last = f;
    t.last = t.last.add(1);
    0
}

/// Runs the table in reverse registration order and empties it. The table is reset before the
/// first call, so a function may register new ones (they run on the next execution).
unsafe fn onexit_execute(t: *mut OnexitTable) -> i32 {
    if t.is_null() {
        return -1;
    }
    let t = &mut *t;
    if t.first.is_null() || t.first == empty_marker() {
        return 0;
    }
    let (first, last) = (t.first, t.last);
    let m = empty_marker();
    (t.first, t.last, t.end) = (m, m, m);
    let mut p = last;
    while p > first {
        p = p.sub(1);
        if *p != 0 {
            let f: unsafe extern "win64" fn() = std::mem::transmute(*p);
            f();
        }
    }
    libc::free(first as *mut c_void);
    0
}

/// Functions of `_crt_atexit` / `_crt_at_quick_exit`, in registration order.
static ATEXIT: Mutex<Vec<usize>> = Mutex::new(Vec::new());
static AT_QUICK_EXIT: Mutex<Vec<usize>> = Mutex::new(Vec::new());
static THREAD_LOCAL_EXE_CALLBACK: AtomicUsize = AtomicUsize::new(0);

fn register_fn(list: &Mutex<Vec<usize>>, f: usize) -> i32 {
    if f == 0 {
        return -1;
    }
    list.lock().unwrap_or_else(|e| e.into_inner()).push(f);
    0
}

/// Runs the thread-local destructor callback of the exe, then the `_crt_atexit` functions in
/// reverse order. Each function runs once (the list is drained, so `exit` called from a handler
/// does not run it again).
fn run_exit_handlers() {
    let cb = THREAD_LOCAL_EXE_CALLBACK.swap(0, Ordering::SeqCst);
    if cb != 0 {
        let f: ThreadLocalExeCallback = unsafe { std::mem::transmute(cb) };
        unsafe { f(std::ptr::null_mut(), 0, std::ptr::null_mut()) };
    }
    loop {
        let f = ATEXIT.lock().unwrap_or_else(|e| e.into_inner()).pop();
        let Some(f) = f else { break };
        let f: unsafe extern "win64" fn() = unsafe { std::mem::transmute(f) };
        unsafe { f() };
    }
}

/// `_register_thread_local_exe_atexit_callback` callback: `(handle, reason, reserved)`, called
/// with `DLL_PROCESS_DETACH` (0).
pub type ThreadLocalExeCallback = unsafe extern "win64" fn(*mut c_void, u32, *mut c_void);

// ---------------------------------------------------------------------------------------------
// Messages and misc.

static NEW_HANDLER: AtomicUsize = AtomicUsize::new(0);

unsafe fn wstr(p: *const u16) -> String {
    if p.is_null() {
        return String::new();
    }
    let mut n = 0;
    while *p.add(n) != 0 {
        n += 1;
    }
    String::from_utf16_lossy(std::slice::from_raw_parts(p, n))
}

fn assert_message(msg: &str, file: &str, line: u32) -> String {
    format!("Assertion failed: {msg}, file {file}, line {line}")
}

/// UCRT `_sys_errlist` for errno values 0 to 42 (Windows texts, not glibc's).
const SYS_ERRLIST: [&str; 43] = [
    "No error",
    "Operation not permitted",
    "No such file or directory",
    "No such process",
    "Interrupted function call",
    "Input/output error",
    "No such device or address",
    "Arg list too long",
    "Exec format error",
    "Bad file descriptor",
    "No child processes",
    "Resource temporarily unavailable",
    "Not enough space",
    "Permission denied",
    "Bad address",
    "Unknown error",
    "Resource device",
    "File exists",
    "Improper link",
    "No such device",
    "Not a directory",
    "Is a directory",
    "Invalid argument",
    "Too many open files in system",
    "Too many open files",
    "Inappropriate I/O control operation",
    "Unknown error",
    "File too large",
    "No space left on device",
    "Invalid seek",
    "Read-only file system",
    "Too many links",
    "Broken pipe",
    "Domain error",
    "Result too large",
    "Unknown error",
    "Resource deadlock avoided",
    "Unknown error",
    "Filename too long",
    "No locks available",
    "Function not implemented",
    "Directory not empty",
    "Illegal byte sequence",
];

fn error_text(n: i32) -> &'static str {
    usize::try_from(n)
        .ok()
        .and_then(|i| SYS_ERRLIST.get(i))
        .copied()
        .unwrap_or("Unknown error")
}

thread_local! {
    static STRERROR_BUF: UnsafeCell<[u8; 64]> = const { UnsafeCell::new([0; 64]) };
}

fn exit_process(code: i32) -> ! {
    unsafe { libc::exit(code) }
}

win32! {
    module "api-ms-win-crt-runtime" => "ucrtbase";
    Native fn _errno() -> *mut i32 { ERRNO.with(|e| e.as_ptr()) }
    Native fn _get_errno(out: *mut i32) -> i32 {
        if out.is_null() {
            return invalid_parameter_err(EINVAL);
        }
        *out = ERRNO.with(|e| e.get());
        0
    }
    Native fn _set_errno(v: i32) -> i32 {
        set_errno(v);
        0
    }
    Native fn _set_invalid_parameter_handler(h: usize) -> usize {
        GLOBAL_HANDLER.swap(h, Ordering::Relaxed)
    }
    Native fn _get_invalid_parameter_handler() -> usize {
        GLOBAL_HANDLER.load(Ordering::Relaxed)
    }
    Native fn _set_thread_local_invalid_parameter_handler(h: usize) -> usize {
        THREAD_HANDLER.with(|c| c.replace(h))
    }
    Native fn _get_thread_local_invalid_parameter_handler() -> usize {
        THREAD_HANDLER.with(|c| c.get())
    }
    Native fn _invalid_parameter(
        expr: *const u16, func: *const u16, file: *const u16, line: u32, res: usize
    ) {
        if !call_handler(expr, func, file, line, res) {
            fatal_invalid_parameter();
        }
    }
    Native fn _invalid_parameter_noinfo() {
        if !call_handler(std::ptr::null(), std::ptr::null(), std::ptr::null(), 0, 0) {
            fatal_invalid_parameter();
        }
    }
    Native fn _invalid_parameter_noinfo_noreturn() {
        call_handler(std::ptr::null(), std::ptr::null(), std::ptr::null(), 0, 0);
        fatal_invalid_parameter();
    }
    // --- argv and startup ---
    Native fn __p___argc() -> *mut i32 { std::ptr::addr_of!(argv().argc) as *mut i32 }
    Native fn __p___argv() -> *mut *mut *mut c_char { std::ptr::addr_of!(argv().narrow) as *mut _ }
    Native fn __p___wargv() -> *mut *mut *mut u16 { std::ptr::addr_of!(argv().wide) as *mut _ }
    // The arrays are always built from the base's command line (no re-parsing), whatever the mode.
    Native fn _configure_narrow_argv(_mode: i32) -> i32 { argv(); 0 }
    Native fn _configure_wide_argv(_mode: i32) -> i32 { argv(); 0 }
    Native fn _get_narrow_winmain_command_line() -> *const c_char { argv().winmain }
    Native fn _initialize_narrow_environment() -> i32 { 0 }
    Native fn _set_app_type(app_type: i32) { APP_TYPE.store(app_type, Ordering::Relaxed) }
    Native fn _initterm(
        start: *const Option<unsafe extern "win64" fn() -> i32>,
        end: *const Option<unsafe extern "win64" fn() -> i32>
    ) {
        init_term(start, end, false);
    }
    Native fn _initterm_e(
        start: *const Option<unsafe extern "win64" fn() -> i32>,
        end: *const Option<unsafe extern "win64" fn() -> i32>
    ) -> i32 {
        init_term(start, end, true)
    }

    // --- onexit tables ---
    Native fn _initialize_onexit_table(t: *mut OnexitTable) -> i32 { onexit_init(t) }
    Native fn _register_onexit_function(t: *mut OnexitTable, f: usize) -> i32 {
        onexit_register(t, f)
    }
    Native fn _execute_onexit_table(t: *mut OnexitTable) -> i32 { onexit_execute(t) }
    Native fn _crt_atexit(f: usize) -> i32 { register_fn(&ATEXIT, f) }
    Native fn _crt_at_quick_exit(f: usize) -> i32 { register_fn(&AT_QUICK_EXIT, f) }
    Native fn _register_thread_local_exe_atexit_callback(f: usize) {
        THREAD_LOCAL_EXE_CALLBACK.store(f, Ordering::SeqCst)
    }

    // --- exit family ---
    Native fn exit(code: i32) -> ! {
        run_exit_handlers();
        exit_process(code)
    }
    Native fn _exit(code: i32) -> ! { libc::_exit(code) }
    Native fn _cexit() { run_exit_handlers() }
    Native fn _c_exit() {}
    Native fn abort() -> ! { std::process::abort() }
    Native fn terminate() -> ! { std::process::abort() }
    Native fn _invoke_watson(_expr: *const u16, _func: *const u16, _file: *const u16, _line: u32, _res: usize) -> ! {
        std::process::abort()
    }

    // --- threads ---
    Native fn _beginthreadex(
        _security: *mut c_void, stack: u32, start: Option<ThreadStart>,
        arg: *mut c_void, flags: u32, thread_id: *mut u32
    ) -> usize {
        let Some(start) = start else {
            invalid_parameter_err(EINVAL);
            return 0;
        };
        match crate::handles::create_thread(stack as usize, Some(start), arg, flags) {
            Some((h, tid)) => {
                if !thread_id.is_null() {
                    *thread_id = tid;
                }
                h
            }
            None => {
                set_errno(11); // EAGAIN
                0
            }
        }
    }
    Native fn _endthreadex(code: u32) -> ! { crate::handles::end_current_thread(code) }

    // --- misc ---
    // No CRT signal table yet: no exception code maps to a handler, so always keep searching.
    Native fn _seh_filter_exe(_code: u32, _ptrs: *mut c_void) -> i32 { 0 }
    Native fn _seh_filter_dll(_code: u32, _ptrs: *mut c_void) -> i32 { 0 }
    Native fn _set_new_handler(h: usize) -> usize { NEW_HANDLER.swap(h, Ordering::SeqCst) }
    Native fn _wassert(msg: *const u16, file: *const u16, line: u32) {
        eprintln!("{}", assert_message(&wstr(msg), &wstr(file), line));
        std::process::abort()
    }
    Native fn strerror(n: i32) -> *mut c_char {
        STRERROR_BUF.with(|b| {
            let buf = &mut *b.get();
            let t = error_text(n).as_bytes();
            buf[..t.len()].copy_from_slice(t);
            buf[t.len()] = 0;
            buf.as_mut_ptr() as *mut c_char
        })
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Serializes tests that install a global handler and counts its calls (per thread).
    use std::cell::Cell;
    use std::sync::{Mutex, MutexGuard};

    static LOCK: Mutex<()> = Mutex::new(());
    thread_local! { static CALLS: Cell<u32> = const { Cell::new(0) }; }

    unsafe extern "win64" fn counting(
        _: *const u16,
        _: *const u16,
        _: *const u16,
        _: u32,
        _: usize,
    ) {
        CALLS.with(|c| c.set(c.get() + 1));
    }

    /// Holds the global handler (a returning, counting one) for the guard's lifetime.
    pub struct HandlerGuard(#[allow(dead_code)] MutexGuard<'static, ()>);

    pub fn install_counting_handler() -> HandlerGuard {
        let g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        CALLS.with(|c| c.set(0));
        unsafe { super::_set_invalid_parameter_handler(counting as *const () as usize) };
        HandlerGuard(g)
    }

    pub fn calls() -> u32 {
        CALLS.with(|c| c.get())
    }

    impl Drop for HandlerGuard {
        fn drop(&mut self) {
            unsafe { super::_set_invalid_parameter_handler(0) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    fn get(name: &str) -> *const () {
        crate::lookup("api-ms-win-crt-runtime-l1-1-0.dll", name)
            .unwrap()
            .addr
    }

    #[test]
    fn every_name_registered() {
        let names = "_errno _get_errno _set_errno _set_invalid_parameter_handler _get_invalid_parameter_handler _set_thread_local_invalid_parameter_handler _get_thread_local_invalid_parameter_handler _invalid_parameter _invalid_parameter_noinfo _invalid_parameter_noinfo_noreturn";
        assert_eq!(names.split(' ').count(), 10);
        for n in names.split(' ') {
            assert!(crate::lookup("api-ms-win-crt-runtime", n).is_some(), "{n}");
            assert!(crate::lookup("ucrtbase", n).is_some(), "{n}");
        }
        let f: unsafe extern "win64" fn() -> *mut i32 =
            unsafe { std::mem::transmute(get("_errno")) };
        assert_eq!(unsafe { f() }, unsafe { _errno() });
    }

    #[test]
    fn errno_roundtrip_and_pointer() {
        unsafe {
            assert_eq!(_set_errno(34), 0);
            assert_eq!(*_errno(), 34);
            *_errno() = 7;
            let mut v = 0;
            assert_eq!(_get_errno(&mut v), 0);
            assert_eq!(v, 7);
        }
    }

    #[test]
    fn errno_is_per_thread() {
        unsafe { _set_errno(11) };
        std::thread::spawn(|| unsafe {
            assert_eq!(*_errno(), 0);
            _set_errno(22);
            assert_eq!(errno(), 22);
        })
        .join()
        .unwrap();
        assert_eq!(errno(), 11);
        // Distinct from glibc's errno.
        unsafe { *libc::__errno_location() = 99 };
        assert_eq!(errno(), 11);
    }

    #[test]
    fn get_errno_null_is_invalid_parameter() {
        let _g = install_counting_handler();
        unsafe {
            _set_errno(0);
            assert_eq!(_get_errno(std::ptr::null_mut()), EINVAL);
        }
        assert_eq!(errno(), EINVAL);
        assert_eq!(calls(), 1);
    }

    #[test]
    fn handler_precedence() {
        unsafe extern "win64" fn thread_h(
            _: *const u16,
            _: *const u16,
            _: *const u16,
            _: u32,
            _: usize,
        ) {
            THREAD_CALLS.with(|c| c.set(c.get() + 1));
        }
        thread_local! { static THREAD_CALLS: Cell<u32> = const { Cell::new(0) }; }
        let _g = install_counting_handler();
        unsafe {
            assert_eq!(_get_thread_local_invalid_parameter_handler(), 0);
            assert_eq!(
                _set_thread_local_invalid_parameter_handler(thread_h as *const () as usize),
                0
            );
            assert_eq!(
                _get_thread_local_invalid_parameter_handler(),
                thread_h as *const () as usize
            );
            _invalid_parameter_noinfo();
            assert_eq!(THREAD_CALLS.with(|c| c.get()), 1);
            assert_eq!(calls(), 0);
            // Another thread sees only the global handler.
            let other = std::thread::spawn(|| {
                _invalid_parameter(std::ptr::null(), std::ptr::null(), std::ptr::null(), 5, 0);
                calls()
            })
            .join()
            .unwrap();
            assert_eq!(other, 1);
            assert_eq!(calls(), 0);
            assert_eq!(
                _set_thread_local_invalid_parameter_handler(0),
                thread_h as *const () as usize
            );
            _invalid_parameter_noinfo();
            assert_eq!(calls(), 1);
            assert_ne!(_get_invalid_parameter_handler(), 0);
        }
    }

    #[test]
    fn handler_set_returns_previous() {
        let _g = install_counting_handler();
        unsafe {
            let cur = _get_invalid_parameter_handler();
            assert_ne!(cur, 0);
            assert_eq!(_set_invalid_parameter_handler(0), cur);
            assert_eq!(_get_invalid_parameter_handler(), 0);
            assert_eq!(_set_invalid_parameter_handler(cur), 0);
        }
    }

    #[test]
    fn internal_helper_without_handler_returns() {
        let _g = install_counting_handler();
        unsafe { _set_invalid_parameter_handler(0) };
        invalid_parameter();
        assert_eq!(calls(), 0);
    }
}

#[cfg(test)]
mod startup_tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    /// Serializes the tests that touch the command line or the global exit lists.
    static GLOBAL: Mutex<()> = Mutex::new(());

    fn global() -> std::sync::MutexGuard<'static, ()> {
        GLOBAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    unsafe fn cstr(p: *const c_char) -> String {
        CStr::from_ptr(p).to_str().unwrap().to_string()
    }

    #[test]
    fn every_new_name_registered() {
        let names = "__p___argc __p___argv __p___wargv _configure_narrow_argv _configure_wide_argv _get_narrow_winmain_command_line _initialize_narrow_environment _set_app_type _initterm _initterm_e _initialize_onexit_table _register_onexit_function _execute_onexit_table _crt_atexit _crt_at_quick_exit _register_thread_local_exe_atexit_callback exit _exit _cexit _c_exit abort terminate _invoke_watson _beginthreadex _endthreadex _seh_filter_exe _seh_filter_dll _set_new_handler _wassert strerror";
        assert_eq!(names.split(' ').count(), 30);
        for n in names.split(' ') {
            assert!(
                crate::lookup("api-ms-win-crt-runtime-l1-1-0.dll", n).is_some(),
                "{n}"
            );
            assert!(crate::lookup("ucrtbase", n).is_some(), "{n}");
        }
    }

    #[test]
    fn argv_built_from_the_base_command_line() {
        let _g = global();
        set_command_line(&s(&["C:\\g\\game.exe", "-w", "a b", "é"]));
        let a = build_argv();
        unsafe {
            assert_eq!(a.argc, 4);
            assert_eq!(cstr(*a.narrow.add(1)), "-w");
            assert_eq!(cstr(*a.narrow.add(3)), "é");
            assert!((*a.narrow.add(4)).is_null());
            let w = *a.wide.add(2);
            assert_eq!(wstr(w), "a b");
            assert_eq!(wstr(*a.wide.add(3)), "é");
            assert!((*a.wide.add(4)).is_null());
            assert_eq!(cstr(a.winmain), "-w \"a b\" é");
        }
        // Back to the default for whoever builds next.
        *COMMAND_LINE.lock().unwrap() = None;
    }

    #[test]
    fn argv_exports_are_stable() {
        let _g = global();
        set_command_line(&s(&["prog", "x"]));
        unsafe {
            let argc = __p___argc();
            let argv1 = *__p___argv();
            // The arrays do not move, nor does the first one change when called again.
            assert_eq!(__p___argv(), __p___argv());
            assert_eq!(*__p___argv(), argv1);
            assert_eq!(_configure_narrow_argv(1), 0);
            assert_eq!(_configure_wide_argv(1), 0);
            assert_eq!(_initialize_narrow_environment(), 0);
            assert!(*argc >= 1);
            assert!(!argv1.is_null());
            assert!(!(*argv1).is_null());
            assert!(!(*(*__p___wargv())).is_null());
            assert!((*argv1.add(*argc as usize)).is_null());
            assert_eq!(wstr(*(*__p___wargv())), cstr(*argv1));
            _set_app_type(2);
            assert_eq!(APP_TYPE.load(Ordering::Relaxed), 2);
        }
        *COMMAND_LINE.lock().unwrap() = None;
    }

    #[test]
    fn c_export_sets_the_command_line() {
        let _g = global();
        let args = [c"one", c"two words"];
        let ptrs: Vec<*const c_char> = args.iter().map(|a| a.as_ptr()).collect();
        unsafe {
            nativ64_set_command_line(2, ptrs.as_ptr());
            assert_eq!(build_argv().argc, 2);
            assert_eq!(cstr(build_argv().winmain), "\"two words\"");
            // Ignored: negative count, NULL array.
            nativ64_set_command_line(-1, ptrs.as_ptr());
            nativ64_set_command_line(2, std::ptr::null());
            assert_eq!(build_argv().argc, 2);
            nativ64_set_command_line(0, std::ptr::null());
            assert_eq!(build_argv().argc, 0);
            assert_eq!(cstr(build_argv().winmain), "");
        }
        *COMMAND_LINE.lock().unwrap() = None;
    }

    #[test]
    fn default_is_proc_self_cmdline() {
        let v = default_command_line();
        assert!(!v.is_empty());
        let me = std::env::args().next().unwrap();
        assert_eq!(v[0], me);
    }

    #[test]
    fn winmain_command_line_quoting_rules() {
        assert_eq!(after_program_name("\"misc\" cmd"), "cmd");
        assert_eq!(after_program_name("misc   cmd arg"), "cmd arg");
        assert_eq!(after_program_name("misc"), "");
        assert_eq!(after_program_name(""), "");
        // The oracle's odd program name: quotes toggle, blanks inside quotes stay in the name.
        assert_eq!(
            after_program_name("\"\"m\"\"\"isc\" \t \"misc\" cmd"),
            "\"misc\" cmd"
        );
        assert_eq!(after_program_name("\"a b\"\targ"), "arg");
        assert_eq!(after_program_name("\"unterminated name"), "");
    }

    #[test]
    fn quoting_arguments() {
        assert_eq!(quote_arg("plain"), "plain");
        assert_eq!(quote_arg(""), "\"\"");
        assert_eq!(quote_arg("a b"), "\"a b\"");
        assert_eq!(quote_arg("say \"hi\""), "\"say \\\"hi\\\"\"");
        assert_eq!(quote_arg("dir \\"), "\"dir \\\\\"");
    }

    static ORDER: Mutex<Vec<u32>> = Mutex::new(Vec::new());
    static COUNT: AtomicU32 = AtomicU32::new(0);

    unsafe extern "win64" fn init_a() -> i32 {
        ORDER.lock().unwrap().push(1);
        0
    }
    unsafe extern "win64" fn init_fail() -> i32 {
        ORDER.lock().unwrap().push(2);
        7
    }
    unsafe extern "win64" fn init_c() -> i32 {
        ORDER.lock().unwrap().push(3);
        0
    }

    #[test]
    fn initterm_and_initterm_e() {
        let _g = global();
        let table: [Option<unsafe extern "win64" fn() -> i32>; 4] =
            [Some(init_a), None, Some(init_fail), Some(init_c)];
        let range = table.as_ptr_range();
        unsafe {
            ORDER.lock().unwrap().clear();
            _initterm(range.start, range.end);
            // `_initterm` ignores results and runs everything, skipping NULL.
            assert_eq!(*ORDER.lock().unwrap(), [1, 2, 3]);

            ORDER.lock().unwrap().clear();
            assert_eq!(_initterm_e(range.start, range.end), 7);
            assert_eq!(*ORDER.lock().unwrap(), [1, 2]);

            ORDER.lock().unwrap().clear();
            assert_eq!(_initterm_e(range.start, range.start), 0);
            assert_eq!(_initterm_e(std::ptr::null(), std::ptr::null()), 0);
            _initterm(std::ptr::null(), std::ptr::null());
            assert!(ORDER.lock().unwrap().is_empty());
        }
    }

    fn new_table() -> OnexitTable {
        OnexitTable {
            first: std::ptr::null_mut(),
            last: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
        }
    }

    #[test]
    fn onexit_initialize() {
        unsafe {
            assert_eq!(_initialize_onexit_table(std::ptr::null_mut()), -1);
            let mut t = new_table();
            assert_eq!(_initialize_onexit_table(&mut t), 0);
            assert!(t.first == t.last && t.first == t.end && !t.first.is_null());
            let mut t2 = new_table();
            assert_eq!(_initialize_onexit_table(&mut t2), 0);
            assert!(t2.first == t.first && t2.last == t.last && t2.end == t.end);

            // Uninitialized garbage with first == end is reset; a live table is untouched.
            let (a, b, c) = (
                0x123 as *mut usize,
                0x456 as *mut usize,
                0x789 as *mut usize,
            );
            t = OnexitTable {
                first: a,
                last: a,
                end: a,
            };
            _initialize_onexit_table(&mut t);
            assert!(t.first == t2.first && t.last == t2.first && t.end == t2.first);
            t = OnexitTable {
                first: a,
                last: b,
                end: a,
            };
            _initialize_onexit_table(&mut t);
            assert!(t.first == t.last && t.first == t.end && t.first != a);
            t = OnexitTable {
                first: std::ptr::null_mut(),
                last: b,
                end: std::ptr::null_mut(),
            };
            _initialize_onexit_table(&mut t);
            assert!(t.first == t.last && t.first == t.end && !t.first.is_null());
            t = OnexitTable {
                first: a,
                last: b,
                end: c,
            };
            _initialize_onexit_table(&mut t);
            assert!(t.first == a && t.last == b && t.end == c);
        }
    }

    unsafe extern "win64" fn count_fn() {
        COUNT.fetch_add(1, Ordering::SeqCst);
    }
    unsafe extern "win64" fn order_1() {
        ORDER.lock().unwrap().push(1);
    }
    unsafe extern "win64" fn order_2() {
        ORDER.lock().unwrap().push(2);
    }

    #[test]
    fn onexit_register_execute() {
        let _g = global();
        unsafe {
            assert_eq!(_register_onexit_function(std::ptr::null_mut(), 0), -1);
            assert_eq!(
                _register_onexit_function(std::ptr::null_mut(), count_fn as *const () as usize),
                -1
            );
            assert_eq!(_execute_onexit_table(std::ptr::null_mut()), -1);

            let mut t = new_table();
            _initialize_onexit_table(&mut t);
            // Empty table.
            assert_eq!(_execute_onexit_table(&mut t), 0);

            // NULL is accepted (and counted) but never called; grows past the first block.
            COUNT.store(0, Ordering::SeqCst);
            let last = t.last;
            assert_eq!(_register_onexit_function(&mut t, 0), 0);
            assert!(t.last != last);
            for _ in 0..20 {
                assert_eq!(
                    _register_onexit_function(&mut t, count_fn as *const () as usize),
                    0
                );
            }
            assert!(t.first != t.end);
            assert_eq!(t.last.offset_from(t.first), 21);
            assert_eq!(_execute_onexit_table(&mut t), 0);
            assert_eq!(COUNT.load(Ordering::SeqCst), 20);
            assert!(t.first == t.end && t.first == t.last);
            // Already empty.
            assert_eq!(_execute_onexit_table(&mut t), 0);
            assert_eq!(COUNT.load(Ordering::SeqCst), 20);

            // Reverse order, and the table is reusable.
            ORDER.lock().unwrap().clear();
            _register_onexit_function(&mut t, order_1 as *const () as usize);
            _register_onexit_function(&mut t, order_2 as *const () as usize);
            _execute_onexit_table(&mut t);
            assert_eq!(*ORDER.lock().unwrap(), [2, 1]);
        }
    }

    #[test]
    fn crt_atexit_runs_in_reverse_and_once() {
        let _g = global();
        unsafe {
            assert_eq!(_crt_atexit(0), -1);
            assert_eq!(_crt_at_quick_exit(0), -1);
            ORDER.lock().unwrap().clear();
            assert_eq!(_crt_atexit(order_1 as *const () as usize), 0);
            assert_eq!(_crt_atexit(order_2 as *const () as usize), 0);
            assert_eq!(_crt_at_quick_exit(count_fn as *const () as usize), 0);
            COUNT.store(0, Ordering::SeqCst);
            _cexit();
            assert_eq!(*ORDER.lock().unwrap(), [2, 1]);
            _cexit();
            assert_eq!(ORDER.lock().unwrap().len(), 2);
            // Quick-exit functions are not run by `_cexit`; `_c_exit` runs nothing.
            assert_eq!(COUNT.load(Ordering::SeqCst), 0);
            AT_QUICK_EXIT.lock().unwrap().clear();
            _crt_atexit(order_1 as *const () as usize);
            _c_exit();
            assert_eq!(ORDER.lock().unwrap().len(), 2);
            ATEXIT.lock().unwrap().clear();
        }
    }

    static CB_ARGS: Mutex<Vec<u32>> = Mutex::new(Vec::new());
    unsafe extern "win64" fn tls_cb(_h: *mut c_void, reason: u32, _r: *mut c_void) {
        CB_ARGS.lock().unwrap().push(reason);
        ORDER.lock().unwrap().push(9);
    }

    #[test]
    fn thread_local_exe_callback_called_once_with_process_detach() {
        let _g = global();
        unsafe {
            ORDER.lock().unwrap().clear();
            CB_ARGS.lock().unwrap().clear();
            _register_thread_local_exe_atexit_callback(tls_cb as *const () as usize);
            _crt_atexit(order_1 as *const () as usize);
            _cexit();
            assert_eq!(*CB_ARGS.lock().unwrap(), [0]);
            assert_eq!(*ORDER.lock().unwrap(), [9, 1]);
            _cexit();
            assert_eq!(CB_ARGS.lock().unwrap().len(), 1);
        }
    }

    #[test]
    fn misc_stubs() {
        unsafe {
            assert_eq!(_seh_filter_exe(0xC000_0005, std::ptr::null_mut()), 0);
            assert_eq!(_seh_filter_dll(0xC000_0005, std::ptr::null_mut()), 0);
        }
        let _g = global();
        unsafe {
            let before = _set_new_handler(0x1000);
            assert_eq!(_set_new_handler(0x2000), 0x1000);
            assert_eq!(_set_new_handler(before), 0x2000);
        }
        assert_eq!(
            assert_message("x > 0", "a.c", 12),
            "Assertion failed: x > 0, file a.c, line 12"
        );
    }

    #[test]
    fn strerror_messages() {
        unsafe {
            assert_eq!(cstr(strerror(0)), "No error");
            assert_eq!(cstr(strerror(2)), "No such file or directory");
            assert_eq!(cstr(strerror(22)), "Invalid argument");
            assert_eq!(cstr(strerror(34)), "Result too large");
            assert_eq!(cstr(strerror(42)), "Illegal byte sequence");
            // Gaps of the table and out-of-range values.
            assert_eq!(cstr(strerror(15)), "Unknown error");
            assert_eq!(cstr(strerror(43)), "Unknown error");
            assert_eq!(cstr(strerror(-1)), "Unknown error");
            assert_eq!(cstr(strerror(i32::MAX)), "Unknown error");
            // Per-thread buffer: another thread's call does not change this one's text.
            let p = strerror(2);
            std::thread::spawn(|| {
                strerror(22);
            })
            .join()
            .unwrap();
            assert_eq!(cstr(p), "No such file or directory");
        }
        assert_eq!(SYS_ERRLIST.len(), 43);
    }

    // --- threads ---

    unsafe extern "win64" fn thread_ret(_p: *mut c_void) -> u32 {
        41
    }
    unsafe extern "win64" fn thread_echo(p: *mut c_void) -> u32 {
        p as usize as u32
    }
    unsafe extern "win64" fn thread_end(p: *mut c_void) -> u32 {
        _endthreadex(p as usize as u32);
    }

    fn wait_thread(h: usize) -> u32 {
        unsafe { crate::api_ms_win_core_synch::WaitForSingleObject(h, 10_000) }
    }

    fn exit_code(h: usize) -> u32 {
        let mut c = 0;
        assert_eq!(
            unsafe { crate::api_ms_win_core_processthreads::GetExitCodeThread(h, &mut c) },
            1
        );
        c
    }

    #[test]
    fn beginthreadex_runs_and_returns_code() {
        unsafe {
            let mut id = 0u32;
            let h = _beginthreadex(
                std::ptr::null_mut(),
                0,
                Some(thread_echo),
                0x1234 as *mut c_void,
                0,
                &mut id,
            );
            assert_ne!(h, 0);
            assert_ne!(id, 0);
            assert_eq!(wait_thread(h), 0);
            assert_eq!(exit_code(h), 0x1234);
            // The thread id pointer is optional.
            let h2 = _beginthreadex(
                std::ptr::null_mut(),
                0,
                Some(thread_ret),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            );
            assert_ne!(h2, 0);
            assert_eq!(wait_thread(h2), 0);
        }
    }

    #[test]
    fn beginthreadex_null_routine_is_einval() {
        let _g = test_support::install_counting_handler();
        unsafe {
            _set_errno(0);
            let h = _beginthreadex(
                std::ptr::null_mut(),
                0,
                None,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            );
            assert_eq!(h, 0);
        }
        assert_eq!(errno(), EINVAL);
        assert_eq!(test_support::calls(), 1);
    }

    #[test]
    fn beginthreadex_suspended() {
        unsafe {
            let h = _beginthreadex(
                std::ptr::null_mut(),
                0,
                Some(thread_ret),
                std::ptr::null_mut(),
                4,
                std::ptr::null_mut(),
            );
            assert_ne!(h, 0);
            assert_eq!(exit_code(h), 259);
            assert_eq!(crate::api_ms_win_core_processthreads::ResumeThread(h), 1);
            assert_eq!(wait_thread(h), 0);
            assert_eq!(exit_code(h), 41);
        }
    }

    static NEST_MUTEX: AtomicUsize = AtomicUsize::new(0);
    static NEST_FLS_SEEN: AtomicUsize = AtomicUsize::new(0);
    static NEST_TLS_DROPPED: AtomicUsize = AtomicUsize::new(0);
    static NEST_FLS_IDX: AtomicUsize = AtomicUsize::new(0);

    struct DropMark;
    impl Drop for DropMark {
        fn drop(&mut self) {
            NEST_TLS_DROPPED.store(1, Ordering::SeqCst);
        }
    }
    thread_local! { static MARK: DropMark = const { DropMark }; }

    unsafe extern "win64" fn nest_fls_cb(p: *mut c_void) {
        NEST_FLS_SEEN.store(p as usize, Ordering::SeqCst);
    }
    unsafe extern "win64" fn nest_inner(depth: u32) -> u32 {
        if depth > 0 {
            return nest_inner(depth - 1) + 1;
        }
        _endthreadex(0x5150)
    }
    unsafe extern "win64" fn nest_start(_p: *mut c_void) -> u32 {
        use crate::api_ms_win_core_fibers::FlsSetValue;
        use crate::api_ms_win_core_synch::WaitForSingleObject;
        MARK.with(|_| {});
        assert_eq!(
            FlsSetValue(NEST_FLS_IDX.load(Ordering::SeqCst) as u32, 9 as *mut c_void),
            1
        );
        assert_eq!(
            WaitForSingleObject(NEST_MUTEX.load(Ordering::SeqCst), 5000),
            0
        );
        nest_inner(5)
    }

    #[test]
    fn endthreadex_from_nested_frames() {
        use crate::api_ms_win_core_fibers::{FlsAlloc, FlsFree};
        use crate::api_ms_win_core_synch::{CreateMutexW, WaitForSingleObject};
        unsafe {
            let m = CreateMutexW(std::ptr::null(), 0, std::ptr::null());
            NEST_MUTEX.store(m, Ordering::SeqCst);
            let i = FlsAlloc(nest_fls_cb as *const () as usize);
            NEST_FLS_IDX.store(i as usize, Ordering::SeqCst);
            let h = _beginthreadex(
                std::ptr::null_mut(),
                0,
                Some(nest_start),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            );
            assert_ne!(h, 0);
            assert_eq!(wait_thread(h), 0);
            assert_eq!(exit_code(h), 0x5150);
            // Owned by the ended thread: abandoned (WAIT_ABANDONED_0).
            assert_eq!(WaitForSingleObject(m, 5000), 0x80);
            // The pthread returned normally, so its cleanups run (shortly after signaling).
            for _ in 0..500 {
                if NEST_TLS_DROPPED.load(Ordering::SeqCst) == 1
                    && NEST_FLS_SEEN.load(Ordering::SeqCst) == 9
                {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            assert_eq!(NEST_TLS_DROPPED.load(Ordering::SeqCst), 1);
            assert_eq!(NEST_FLS_SEEN.load(Ordering::SeqCst), 9);
            FlsFree(i);
        }
    }

    #[test]
    fn endthreadex_ends_only_the_calling_thread() {
        unsafe {
            let h = _beginthreadex(
                std::ptr::null_mut(),
                0,
                Some(thread_end),
                77 as *mut c_void,
                0,
                std::ptr::null_mut(),
            );
            assert_ne!(h, 0);
            assert_eq!(wait_thread(h), 0);
            assert_eq!(exit_code(h), 77);
        }
        // The test thread is still here, with its errno.
        set_errno(5);
        assert_eq!(errno(), 5);
    }

    // --- terminating paths, in a child process (this test binary run again) ---

    const CHILD_ENV: &str = "NATIV64_CRT_CHILD";

    unsafe extern "win64" fn say_handler() {
        println!("HANDLER-RAN");
    }
    unsafe extern "win64" fn say_other() {
        println!("OTHER-RAN");
    }

    /// Entry of the child: does nothing in a normal run.
    #[test]
    fn child_entry() {
        let Ok(mode) = std::env::var(CHILD_ENV) else {
            return;
        };
        unsafe {
            match mode.as_str() {
                "exit" => {
                    _crt_atexit(say_other as *const () as usize);
                    _crt_atexit(say_handler as *const () as usize);
                    exit(3)
                }
                "_exit" => {
                    _crt_atexit(say_handler as *const () as usize);
                    _exit(4)
                }
                "abort" => abort(),
                "terminate" => terminate(),
                "watson" => {
                    _invoke_watson(std::ptr::null(), std::ptr::null(), std::ptr::null(), 0, 0)
                }
                "wassert" => {
                    let m: Vec<u16> = "x > 0\0".encode_utf16().collect();
                    let f: Vec<u16> = "é.c\0".encode_utf16().collect();
                    _wassert(m.as_ptr(), f.as_ptr(), 12)
                }
                _ => {}
            }
        }
    }

    fn run_child(mode: &str) -> std::process::Output {
        std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "api_ms_win_crt_runtime::startup_tests::child_entry",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD_ENV, mode)
            .output()
            .unwrap()
    }

    #[test]
    fn exit_runs_handlers_then_terminates() {
        use std::os::unix::process::ExitStatusExt;
        let o = run_child("exit");
        assert_eq!(o.status.code(), Some(3), "{:?}", o.status);
        let out = String::from_utf8_lossy(&o.stdout);
        let (a, b) = (out.find("HANDLER-RAN"), out.find("OTHER-RAN"));
        assert!(a.is_some() && b.is_some() && a < b, "{out}");

        let o = run_child("_exit");
        assert_eq!(o.status.code(), Some(4));
        assert!(!String::from_utf8_lossy(&o.stdout).contains("HANDLER-RAN"));

        for mode in ["abort", "terminate", "watson"] {
            let o = run_child(mode);
            assert_eq!(
                o.status.signal(),
                Some(libc::SIGABRT),
                "{mode}: {:?}",
                o.status
            );
        }
    }

    #[test]
    fn wassert_prints_utf8_then_aborts() {
        use std::os::unix::process::ExitStatusExt;
        let o = run_child("wassert");
        assert_eq!(o.status.signal(), Some(libc::SIGABRT));
        assert!(
            String::from_utf8_lossy(&o.stderr)
                .contains("Assertion failed: x > 0, file é.c, line 12"),
            "{}",
            String::from_utf8_lossy(&o.stderr)
        );
    }
}
