//! Process-wide facts the base registers: the main module (the image the base mapped).
//!
//! nativ64 does not load PE images, so the base tells it where the main image is and what its
//! path is. `GetModuleHandle(NULL)`, `GetModuleFileName(NULL, ...)` and the like read it here.

use std::sync::Mutex;

/// The main module as registered by the base.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MainModule {
    pub base: usize,
    pub path: String,
}

static MAIN_MODULE: Mutex<Option<MainModule>> = Mutex::new(None);

/// Registers the main image: its base address (the `HMODULE`) and its path (UTF-8). Call it
/// once after mapping, before running the entry point. A later call replaces the earlier one.
pub fn set_main_module(base: usize, path: &str) {
    *MAIN_MODULE.lock().unwrap_or_else(|e| e.into_inner()) = Some(MainModule {
        base,
        path: path.to_string(),
    });
}

/// The registered main module, if any.
pub fn main_module() -> Option<MainModule> {
    MAIN_MODULE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// How an export is asked for: by name or by ordinal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportRef<'a> {
    Name(&'a str),
    Ordinal(u16),
}

/// Looks an export up in a registered module; returns its address.
pub type ExportLookup = std::sync::Arc<dyn Fn(ExportRef<'_>) -> Option<usize> + Send + Sync>;

/// A guest DLL the base mapped and registered with [`register_module`].
#[derive(Clone)]
pub struct RegisteredModule {
    /// File name, lowercase (e.g. `msvcp140.dll`).
    pub name: String,
    pub path: String,
    pub base: usize,
    lookup: ExportLookup,
}

impl RegisteredModule {
    /// The address of an export of this module.
    pub fn export(&self, r: ExportRef<'_>) -> Option<usize> {
        (self.lookup)(r)
    }
}

static MODULES: Mutex<Vec<RegisteredModule>> = Mutex::new(Vec::new());

/// Registers a mapped guest DLL so that `GetModuleHandle*` returns `base` for `name` and
/// `GetProcAddress(base, ...)` resolves through `lookup`. A module registered again under the
/// same name replaces the earlier one.
pub fn register_module(name: &str, path: &str, base: usize, lookup: ExportLookup) {
    let name = name.to_ascii_lowercase();
    let mut v = MODULES.lock().unwrap_or_else(|e| e.into_inner());
    v.retain(|m| m.name != name);
    v.push(RegisteredModule {
        name,
        path: path.to_string(),
        base,
        lookup,
    });
}

/// Forgets every registered module (tests, or a base that unloads everything).
pub fn clear_modules() {
    MODULES.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

/// A registered module by file name (case-insensitive; `.dll` optional).
pub fn registered_module(name: &str) -> Option<RegisteredModule> {
    let want = name.to_ascii_lowercase();
    let strip = |s: &str| s.strip_suffix(".dll").unwrap_or(s).to_string();
    let want = strip(&want);
    MODULES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find(|m| strip(&m.name) == want)
        .cloned()
}

/// A registered module by base address.
pub fn registered_module_at(base: usize) -> Option<RegisteredModule> {
    MODULES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find(|m| m.base == base)
        .cloned()
}

static IMAGE_RANGES: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Records that the image mapped at `base` spans `size` bytes (the main image and every guest
/// DLL), so that an address can be mapped back to its image ([`image_base_of`]). A range
/// registered again at the same base replaces the earlier one.
pub fn register_image_range(base: usize, size: usize) {
    let mut v = IMAGE_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    v.retain(|r| r.0 != base);
    v.push((base, size));
    drop(v);
    crate::seh::register_image_table(base, size);
}

/// Base of the registered image containing `addr`, if any.
pub fn image_base_of(addr: usize) -> Option<usize> {
    IMAGE_RANGES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find(|&&(b, s)| addr >= b && addr - b < s)
        .map(|r| r.0)
}

/// Like [`image_base_of`] but never waits: `None` when the table is locked, so that logging
/// from an exception path cannot deadlock on a lock the faulting code holds.
pub(crate) fn image_base_of_try(addr: usize) -> Option<usize> {
    let ranges = match IMAGE_RANGES.try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => return None,
    };
    ranges
        .iter()
        .find(|&&(b, s)| addr >= b && addr - b < s)
        .map(|r| r.0)
}

/// Hooks the base installs so that threads made by `CreateThread` / `_beginthreadex` get what
/// guest code needs on any thread (a TEB, static TLS, thread attach notifications).
#[derive(Clone, Copy)]
pub struct ThreadHooks {
    /// Runs on the new thread before the start routine; returns an opaque guard.
    pub attach: fn() -> *mut std::ffi::c_void,
    /// Runs on the same thread after the start routine returned or the thread ended itself
    /// (`_endthreadex`), with the guard `attach` returned.
    pub detach: fn(*mut std::ffi::c_void),
}

static THREAD_HOOKS: Mutex<Option<ThreadHooks>> = Mutex::new(None);

/// Installs the thread hooks (once, by the base, before guest threads are created). None set
/// means threads run the start routine as is. A later call replaces the earlier one.
pub fn set_thread_hooks(attach: fn() -> *mut std::ffi::c_void, detach: fn(*mut std::ffi::c_void)) {
    *THREAD_HOOKS.lock().unwrap_or_else(|e| e.into_inner()) = Some(ThreadHooks { attach, detach });
}

/// Removes the thread hooks (tests).
pub fn clear_thread_hooks() {
    *THREAD_HOOKS.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// The installed thread hooks, if any.
pub fn thread_hooks() -> Option<ThreadHooks> {
    *THREAD_HOOKS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Smallest stack a guest thread gets.
pub const MIN_THREAD_STACK: usize = 1 << 20;
/// `CreateThread` flag: the stack size is the reserve, not the initial commit.
pub const STACK_SIZE_PARAM_IS_A_RESERVATION: u32 = 0x10000;

static DEFAULT_STACK_RESERVE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Registers the main image's `SizeOfStackReserve`, the default stack of new threads.
pub fn set_default_stack_reserve(size: usize) {
    DEFAULT_STACK_RESERVE.store(size, std::sync::atomic::Ordering::Relaxed);
}

/// The stack reserve a new thread gets when `CreateThread` is given `stack` and `flags`: the
/// requested size when it is a reservation, otherwise (initial commit) the larger of the request
/// and the exe's default; never below [`MIN_THREAD_STACK`] and capped at 1 GiB.
pub fn thread_stack_size(stack: usize, flags: u32) -> usize {
    let default = DEFAULT_STACK_RESERVE.load(std::sync::atomic::Ordering::Relaxed);
    let size = if stack != 0 && flags & STACK_SIZE_PARAM_IS_A_RESERVATION != 0 {
        stack
    } else {
        stack.max(default)
    };
    size.clamp(MIN_THREAD_STACK, 1 << 30)
}

static NO_THREAD_CALLS: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Records that the module at `base` called `DisableThreadLibraryCalls`.
pub fn disable_thread_calls(base: usize) {
    let mut v = NO_THREAD_CALLS.lock().unwrap_or_else(|e| e.into_inner());
    if !v.contains(&base) {
        v.push(base);
    }
}

/// Whether the module at `base` asked not to get `DLL_THREAD_ATTACH/DETACH` notifications.
pub fn thread_calls_disabled(base: usize) -> bool {
    NO_THREAD_CALLS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .contains(&base)
}

#[cfg(test)]
mod thread_tests {
    use super::*;

    #[test]
    fn stack_size_rules() {
        set_default_stack_reserve(4 << 20);
        assert_eq!(thread_stack_size(0, 0), 4 << 20);
        assert_eq!(thread_stack_size(64 << 10, 0), 4 << 20);
        assert_eq!(thread_stack_size(8 << 20, 0), 8 << 20);
        assert_eq!(
            thread_stack_size(2 << 20, STACK_SIZE_PARAM_IS_A_RESERVATION),
            2 << 20
        );
        assert_eq!(
            thread_stack_size(4096, STACK_SIZE_PARAM_IS_A_RESERVATION),
            MIN_THREAD_STACK
        );
        assert_eq!(thread_stack_size(usize::MAX, 0), 1 << 30);
        set_default_stack_reserve(0);
        assert_eq!(thread_stack_size(0, 0), MIN_THREAD_STACK);
    }

    #[test]
    fn disable_thread_calls_is_recorded() {
        assert!(!thread_calls_disabled(0xdead_0000));
        disable_thread_calls(0xdead_0000);
        disable_thread_calls(0xdead_0000);
        assert!(thread_calls_disabled(0xdead_0000));
    }
}
