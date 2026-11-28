//! `xgameruntime`: the console game runtime DLL a title loads at startup.
//!
//! The title's runtime thunk library does `LoadLibraryExW("xgameruntime.dll")`, resolves the
//! six bootstrap exports below with `GetProcAddress` and calls `InitializeApiImplEx2` with the
//! runtime version and flags. Individual runtime APIs (`XSystem*`, `XUser*`, ...) are then
//! fetched one by one through `QueryApiImpl`.
//!
//! The bootstrap calls are all that is modelled so far: initialization succeeds and keeps a
//! count, `UninitializeApiImpl` undoes one. `QueryApiImpl(api GUID, interface GUID, out)` refuses
//! every API with `E_GAMERUNTIME_API_NOT_FOUND` and a NULL out pointer until the individual
//! APIs are implemented; each distinct (api, interface) pair is logged once. `XErrorReport` is accepted and
//! ignored. No Wine oracle exists for this module (Wine has no game runtime).

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use crate::com::{
    add_ref, query_interface, release, ComObject, Guid, IUnknownVtbl, E_INVALIDARG, E_POINTER,
};

/// A vtable as an array of untyped slots: slots 0-2 are `IUnknown`.
#[repr(C)]
pub struct Table<const N: usize>(pub [*const (); N]);

// SAFETY: the slots are function pointers, never written.
unsafe impl<const N: usize> Sync for Table<N> {}

/// An object over a [`Table`].
pub type Obj<const N: usize> = ComObject<Table<N>, ()>;

/// A new object (one reference) answering `QueryInterface` for `iids`.
pub fn new_object<const N: usize>(table: &'static Table<N>, iids: &'static [Guid]) -> *mut c_void {
    Obj::<N>::create(table, iids, ()) as *mut c_void
}

/// `Table<$n>` with the `IUnknown` slots filled in (for an object whose payload is `$t`, `()` by
/// default), then the given methods (slot 3 on).
macro_rules! table {
    ($n:expr; $($slot:expr),* $(,)?) => {
        $crate::xgameruntime::table!($n, (); $($slot),*)
    };
    ($n:expr, $t:ty; $($slot:expr),* $(,)?) => {
        $crate::xgameruntime::Table::<$n>([
            $crate::com::query_interface::<$crate::xgameruntime::Table<$n>, $t> as *const (),
            $crate::com::add_ref::<$crate::xgameruntime::Table<$n>, $t> as *const (),
            $crate::com::release::<$crate::xgameruntime::Table<$n>, $t> as *const (),
            $($slot as *const ()),*
        ])
    };
}
pub(crate) use table;

/// Names of the areas a [`slot_trap`] can belong to.
pub const AREAS: [&str; 2] = ["user", "task queue"];
pub const AREA_USER: usize = 0;
pub const AREA_QUEUE: usize = 1;

static TRAPPED: Mutex<Option<HashSet<(usize, usize)>>> = Mutex::new(None);

/// A method nobody implemented yet: logs the area and slot the first time, answers `E_NOTIMPL`.
pub extern "win64" fn slot_trap<const AREA: usize, const SLOT: usize>() -> i32 {
    let first = TRAPPED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashSet::new)
        .insert((AREA, SLOT));
    if first {
        eprintln!(
            "nativ64: unimplemented xgameruntime {} method, slot {SLOT} (offset {:#x})",
            AREAS[AREA],
            SLOT * 8
        );
    }
    E_NOTIMPL
}

/// Calls slot `slot` (index) of `obj` as the title does (tests).
///
/// # Safety
/// `F` is the slot's real signature.
#[cfg(test)]
pub unsafe fn method<F: Copy>(obj: *mut c_void, slot: usize) -> F {
    let table = *(obj as *const *const *const ());
    std::mem::transmute_copy(&*table.add(slot))
}

pub const S_OK: i32 = 0;
pub const E_NOTIMPL: i32 = 0x8000_4001u32 as i32;

/// "API not found": what a provider the host does not offer answers.
pub const E_API_NOT_FOUND: i32 = 0x8924_010Du32 as i32;

/// The system API (`XSystem*`) as the title asks for it.
pub const API_SYSTEM: Guid = Guid::new(
    0xe349bd1a,
    0xfc20,
    0x4e40,
    [0xb9, 0x9c, 0x41, 0x78, 0xcc, 0x6b, 0x40, 0x9f],
);
pub const IID_SYSTEM: Guid = Guid::new(
    0x1861cf2e,
    0xe18b,
    0x4834,
    [0xa9, 0xf5, 0xb4, 0xa4, 0xe6, 0xef, 0xb4, 0xcf],
);
/// `E_NOT_SUFFICIENT_BUFFER`.
pub const E_BUFFER_TOO_SMALL: i32 = 0x8007_007Au32 as i32;
/// The sandbox the title is told it runs in.
pub const SANDBOX_ID: &[u8] = b"RETAIL\0";
/// What `XSystemGetConsoleId` answers on a PC and in RETAIL: the console id format (38
/// characters, the size titles reserve) filled with zeros.
pub const CONSOLE_ID: &[u8] = b"00000000.00000000.00000000.00000000.00\0";

extern "win64" fn trap<const SLOT: usize>() -> ! {
    eprintln!(
        "nativ64: unimplemented xgameruntime {} slot {SLOT}",
        guid_string(&API_SYSTEM)
    );
    // SAFETY: abort never returns.
    unsafe { libc::abort() }
}

type SystemObj = ComObject<SystemVtbl, ()>;

/// Slots 0-2 are `IUnknown`, 3 is the console id, 4 the sandbox id; the others are traps.
#[repr(C)]
struct SystemVtbl {
    unknown: IUnknownVtbl<SystemVtbl, ()>,
    console_id: unsafe extern "win64" fn(*mut SystemObj, usize, *mut u8, *mut usize) -> i32,
    sandbox_id: unsafe extern "win64" fn(*mut SystemObj, usize, *mut u8, *mut usize) -> i32,
    slot5: extern "win64" fn() -> !,
    slot6: extern "win64" fn() -> !,
    slot7: extern "win64" fn() -> !,
    slot8: extern "win64" fn() -> !,
    slot9: extern "win64" fn() -> !,
    slot10: extern "win64" fn() -> !,
    slot11: extern "win64" fn() -> !,
    slot12: extern "win64" fn() -> !,
}

/// Copies the NUL-terminated `text` into the caller's buffer, reporting the size it needs.
///
/// # Safety
/// `buffer` is NULL or valid for `size` bytes; `used` is NULL or valid.
unsafe fn write_cstr(text: &[u8], size: usize, buffer: *mut u8, used: *mut usize) -> i32 {
    if buffer.is_null() && size != 0 {
        return E_POINTER;
    }
    if !used.is_null() {
        *used = text.len();
    }
    if size < text.len() {
        return E_BUFFER_TOO_SMALL;
    }
    std::ptr::copy_nonoverlapping(text.as_ptr(), buffer, text.len());
    S_OK
}

/// `GetConsoleId(size, buffer, used)`: the console id with its terminator.
///
/// # Safety
/// `buffer` is NULL or valid for `size` bytes; `used` is NULL or valid.
unsafe extern "win64" fn console_id(
    _this: *mut SystemObj,
    size: usize,
    buffer: *mut u8,
    used: *mut usize,
) -> i32 {
    write_cstr(CONSOLE_ID, size, buffer, used)
}

/// `GetXboxLiveSandboxId(size, buffer, used)`: the id with its terminator.
///
/// # Safety
/// `buffer` is NULL or valid for `size` bytes; `used` is NULL or valid.
unsafe extern "win64" fn sandbox_id(
    _this: *mut SystemObj,
    size: usize,
    buffer: *mut u8,
    used: *mut usize,
) -> i32 {
    write_cstr(SANDBOX_ID, size, buffer, used)
}

static SYSTEM_VTBL: SystemVtbl = SystemVtbl {
    unknown: IUnknownVtbl {
        query_interface: query_interface::<SystemVtbl, ()>,
        add_ref: add_ref::<SystemVtbl, ()>,
        release: release::<SystemVtbl, ()>,
    },
    console_id,
    sandbox_id,
    slot5: trap::<5>,
    slot6: trap::<6>,
    slot7: trap::<7>,
    slot8: trap::<8>,
    slot9: trap::<9>,
    slot10: trap::<10>,
    slot11: trap::<11>,
    slot12: trap::<12>,
};
static SYSTEM_IIDS: [Guid; 1] = [IID_SYSTEM];

/// The feature probe's provider id, which is also its interface id.
pub const API_FEATURE: Guid = Guid::new(
    0x8836_fe87,
    0xedb9,
    0x4fe3,
    [0x8d, 0xad, 0x05, 0xf0, 0xd2, 0xcd, 0x5b, 0x40],
);
static FEATURE_IIDS: [Guid; 1] = [API_FEATURE];

/// `IsFeatureAvailable(feature)`: the console runtime has every feature of its list except two
/// (the numbering is the public header's; 1 and 14 and anything past 21 are absent).
extern "win64" fn is_feature_available(_this: *mut c_void, feature: u32) -> bool {
    matches!(feature, 0 | 2..=13 | 15..=21)
}

static FEATURE_VTBL: Table<4> = table!(4; is_feature_available);

/// The telemetry event provider (`XGameEventWrite`), whose id is also its interface id.
pub const API_EVENT: Guid = Guid::new(
    0xbbfb_dcc7,
    0xbfe7,
    0x409b,
    [0xa5, 0xca, 0xed, 0xf0, 0x54, 0x96, 0x0b, 0x4d],
);
static EVENT_IIDS: [Guid; 1] = [API_EVENT];

/// `XGameEventWrite(user, service config id, session id, name, dimensions, measurements)`:
/// events go nowhere, and the title is told they were written.
extern "win64" fn event_write(_this: *mut c_void) -> i32 {
    S_OK
}

static EVENT_VTBL: Table<4> = table!(4; event_write);

/// The answer to a request for an API: the system, feature, event, user and task queue APIs exist.
fn query_api(api: &Guid, iid: &Guid) -> Option<*mut c_void> {
    if *api == API_SYSTEM && *iid == IID_SYSTEM {
        Some(SystemObj::create(&SYSTEM_VTBL, &SYSTEM_IIDS, ()) as *mut c_void)
    } else if *api == API_FEATURE && *iid == API_FEATURE {
        Some(new_object(&FEATURE_VTBL, &FEATURE_IIDS))
    } else if *api == API_EVENT && *iid == API_EVENT {
        Some(new_object(&EVENT_VTBL, &EVENT_IIDS))
    } else {
        crate::xuser::query(api, iid).or_else(|| crate::xasync::query(api, iid))
    }
}

static SEEN: Mutex<Option<HashSet<(Guid, Guid)>>> = Mutex::new(None);

fn guid_string(g: &Guid) -> String {
    format!(
        "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        g.data1,
        g.data2,
        g.data3,
        g.data4[0],
        g.data4[1],
        g.data4[2],
        g.data4[3],
        g.data4[4],
        g.data4[5],
        g.data4[6],
        g.data4[7]
    )
}

/// Records a (api, interface) request; true the first time the pair is seen.
fn first_request(api: Guid, iid: Guid) -> bool {
    SEEN.lock()
        .unwrap()
        .get_or_insert_with(HashSet::new)
        .insert((api, iid))
}

static INIT_COUNT: AtomicU32 = AtomicU32::new(0);

/// Number of active `InitializeApiImpl*` calls (not yet balanced by `UninitializeApiImpl`).
pub fn init_count() -> u32 {
    INIT_COUNT.load(Ordering::SeqCst)
}

fn initialize() -> i32 {
    INIT_COUNT.fetch_add(1, Ordering::SeqCst);
    S_OK
}

win32! {
    module "xgameruntime";
    // (api GUID, interface GUID, out object): no API is available yet.
    Native fn QueryApiImpl(api: *const Guid, iid: *const Guid, out: *mut *mut c_void) -> i32 {
        if !out.is_null() {
            *out = std::ptr::null_mut();
        }
        if api.is_null() || iid.is_null() {
            return E_API_NOT_FOUND;
        }
        if let Some(obj) = query_api(&*api, &*iid) {
            if !out.is_null() {
                *out = obj;
                return S_OK;
            }
            release(obj as *mut SystemObj);
            return E_INVALIDARG;
        }
        if first_request(*api, *iid) {
            eprintln!(
                "nativ64: xgameruntime api {} interface {} refused",
                guid_string(&*api),
                guid_string(&*iid)
            );
        }
        E_API_NOT_FOUND
    }
    // Pre-versioned entry: (major, minor).
    Native fn InitializeApiImpl(_major: u32, _minor: u32) -> i32 {
        initialize()
    }
    Native fn InitializeApiImplEx(_version: u64, _api: u64, _flags: u32) -> i32 {
        initialize()
    }
    // `out` receives the runtime's internal context handle when not NULL.
    Native fn InitializeApiImplEx2(_version: u64, _api: u64, _flags: u32,
                                   out: *mut *const c_void) -> i32 {
        if !out.is_null() {
            *out = std::ptr::null();
        }
        initialize()
    }
    Native fn UninitializeApiImpl() -> i32 {
        let _ = INIT_COUNT.try_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1));
        S_OK
    }
    // (hresult, message): diagnostics only.
    Native fn XErrorReport(_hr: i32, _msg: *const c_void) -> i32 {
        S_OK
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lookup;

    #[test]
    fn bootstrap_exports_are_resolvable_by_file_name() {
        for n in [
            "QueryApiImpl",
            "InitializeApiImpl",
            "InitializeApiImplEx",
            "InitializeApiImplEx2",
            "UninitializeApiImpl",
            "XErrorReport",
        ] {
            assert!(lookup("xgameruntime.dll", n).is_some(), "{n}");
        }
    }

    #[test]
    fn guid_text_and_first_request() {
        let g = Guid::new(
            0x073b7dcb,
            0x1fcf,
            0x4030,
            [0x94, 0xbe, 0xe3, 0xc9, 0xeb, 0x62, 0x34, 0x28],
        );
        assert_eq!(guid_string(&g), "073b7dcb-1fcf-4030-94be-e3c9eb623428");
        let h = Guid::new(9, 9, 9, [9; 8]);
        assert!(first_request(g, h));
        assert!(!first_request(g, h));
    }

    #[test]
    fn feature_probe_lists_the_console_features() {
        unsafe {
            let mut obj = std::ptr::null_mut();
            assert_eq!(QueryApiImpl(&API_FEATURE, &API_FEATURE, &mut obj), S_OK);
            let probe: extern "win64" fn(*mut c_void, u32) -> bool = method(obj, 3);
            for f in 0..=22 {
                assert_eq!(probe(obj, f), f != 1 && f != 14 && f != 22, "feature {f}");
            }
            assert!(!probe(obj, u32::MAX));
        }
    }

    #[test]
    fn event_writes_succeed_and_go_nowhere() {
        unsafe {
            let mut obj = std::ptr::null_mut();
            assert_eq!(QueryApiImpl(&API_EVENT, &API_EVENT, &mut obj), S_OK);
            let write: extern "win64" fn(*mut c_void) -> i32 = method(obj, 3);
            assert_eq!(write(obj), S_OK);
        }
    }

    #[test]
    fn system_api_gives_the_sandbox_id() {
        unsafe {
            let mut obj = std::ptr::null_mut();
            assert_eq!(QueryApiImpl(&API_SYSTEM, &IID_SYSTEM, &mut obj), S_OK);
            let o = obj as *mut SystemObj;
            let mut buf = [0xffu8; 16];
            let mut used = 0usize;
            assert_eq!(sandbox_id(o, 16, buf.as_mut_ptr(), &mut used), S_OK);
            assert_eq!(used, 7);
            assert_eq!(&buf[..7], b"RETAIL\0");
            // Too small a buffer: error, the needed size is reported, nothing written.
            let mut small = [0xffu8; 4];
            assert_eq!(
                sandbox_id(o, 4, small.as_mut_ptr(), &mut used),
                E_BUFFER_TOO_SMALL
            );
            assert_eq!(used, 7);
            assert_eq!(small, [0xff; 4]);
            assert_eq!(sandbox_id(o, 4, std::ptr::null_mut(), &mut used), E_POINTER);
            let mut id = [0xffu8; 64];
            assert_eq!(console_id(o, 64, id.as_mut_ptr(), &mut used), S_OK);
            assert_eq!(used, 39);
            assert_eq!(&id[..39], CONSOLE_ID);
            assert_eq!(
                console_id(o, 8, id.as_mut_ptr(), &mut used),
                E_BUFFER_TOO_SMALL
            );
            assert_eq!(used, 39);
            assert_eq!(release(o), 0);
            // The interface GUID must match too.
            assert_eq!(
                QueryApiImpl(&API_SYSTEM, &API_SYSTEM, &mut obj),
                E_API_NOT_FOUND
            );
            assert!(obj.is_null());
        }
    }

    #[test]
    fn user_and_task_queue_apis_are_served() {
        use crate::{xasync, xuser};
        unsafe {
            for (api, iid) in [
                (xuser::PROVIDER, xuser::IID),
                (xuser::PROVIDER, xuser::GAMERTAG_IID),
                (xasync::PROVIDER, xasync::PROVIDER),
            ] {
                let mut obj = std::ptr::null_mut();
                assert_eq!(QueryApiImpl(&api, &iid, &mut obj), S_OK);
                assert!(!obj.is_null());
                assert_eq!(release(obj as *mut SystemObj), 0);
            }
            // A slot nobody implemented logs and answers E_NOTIMPL (task queue slot 6).
            let mut q = std::ptr::null_mut();
            QueryApiImpl(&xasync::PROVIDER, &xasync::PROVIDER, &mut q);
            let run: extern "win64" fn(*mut c_void) -> i32 = method(q, 6);
            assert_eq!(run(q), E_NOTIMPL);
            let mut obj = std::ptr::null_mut();
            assert_eq!(
                QueryApiImpl(&xuser::PROVIDER, &xasync::PROVIDER, &mut obj),
                E_API_NOT_FOUND
            );
            assert!(obj.is_null());
        }
    }

    #[test]
    fn init_and_uninit_balance() {
        unsafe {
            let base = init_count();
            let mut out = std::ptr::dangling::<c_void>();
            assert_eq!(InitializeApiImplEx2(1, 2, 6, &mut out), S_OK);
            assert!(out.is_null());
            assert_eq!(InitializeApiImplEx(1, 2, 6), S_OK);
            assert_eq!(init_count(), base + 2);
            assert_eq!(UninitializeApiImpl(), S_OK);
            assert_eq!(UninitializeApiImpl(), S_OK);
            assert_eq!(init_count(), base);
            let g = Guid::new(1, 2, 3, [4; 8]);
            let mut obj = std::ptr::dangling_mut::<c_void>();
            assert_eq!(QueryApiImpl(&g, &g, &mut obj), E_API_NOT_FOUND);
            assert!(obj.is_null());
            assert_eq!(
                QueryApiImpl(std::ptr::null(), &g, &mut obj),
                E_API_NOT_FOUND
            );
        }
    }
}
