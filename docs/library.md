# nativ64: the host API library

`crates/nativ64` reimplements, natively on Linux x86_64, the Windows host APIs the title's
binaries import (kernel32, `api-ms-win-core-*`, `api-ms-win-crt-*`, ...). It does not depend on
the compatibility layer ("base") that loads the game: VIctory today, possibly others later.

## Layout

- One Rust module per Windows module, named like `tools/import_map.py` normalizes it
  (lowercase, no `.dll`, API-set version suffix dropped, `-` becomes `_`):
  `api-ms-win-core-synch` -> `src/api_ms_win_core_synch.rs`.
- `src/handles.rs` (not a Windows module): the process-wide handle table and the kernel objects
  in it (events, mutexes, semaphores), the wait engine and the named-object namespace; the
  exports that use it are declared in `api_ms_win_core_synch.rs` and `api_ms_win_core_handle.rs`.
- `src/registry.rs`: `Export { module, name, addr, status }`, `Status::{Native, Stub}`,
  `lookup(module, name)`, `normalize_module`, and the C function `nativ64_lookup`.
- `src/lib.rs`: `TABLES` lists each module's `EXPORTS`; `exports()` iterates them all.
- `src/bin/nativ64-status.rs`: prints the status file for the import map
  (see "Library view" in `import-map.md`).

## Adding a function

Declare it in the module file with the `win32!` macro; the declaration is also the
registration. The macro registers each function under the API-set name and under its
`kernel32` forwarder.

```rust
win32! {
    module "api-ms-win-core-synch" => "kernel32";
    Native fn Sleep(milliseconds: u32) { /* ... */ }
}
```

Use `Stub` for a function that exists but does nothing useful yet. For a new module, create
the file and add its `EXPORTS` to `TABLES`. Write a unit test that calls through the win64
function. Then check the new tab of the map.

## COM objects (`src/com.rs`)

Host objects the title calls through vtables (GameInput, the game runtime, D3D12) share one
toolkit. `Guid` is the Windows layout; `ComObject<V, T>` is `#[repr(C)]`: vtable pointer first,
then an atomic reference count, the IIDs it answers to and the payload `T`
(`ComObject::create(&VTBL, &IIDS, data)` returns a raw pointer with count 1). `IUnknownVtbl<V, T>`
provides `QueryInterface` (IUnknown and the listed IIDs; `E_NOINTERFACE`, `E_POINTER` for NULL
`out` or IID, `out` always cleared), `AddRef`, and `Release` (frees the object at zero).
To declare an interface: a `#[repr(C)]` vtable struct whose first field is `IUnknownVtbl<V, T>`
(initialised with `IUnknownVtbl::new()`), then one `unsafe extern "win64" fn(*mut ComObject<V, T>, ..)`
per method in published order; fill a `static`. Tests call through the vtable
(`gameinput.rs` is the model). Interfaces that inherit another's methods repeat them first in
their own vtable struct.

The game runtime (`xgameruntime.rs`) serves its APIs from `QueryApiImpl(api, iid)`: the system
API, the user provider (`xuser.rs`: one offline local user "Player", signed in, every privilege)
and the task queue / async provider (`xasync.rs`: queues with manual, thread-pool, serialized and
immediate ports, async blocks completed through the block's queue; the minimum the user path
needs). Their objects are `Table<N>`s of untyped slots built with `table!`; a slot with no method
is `slot_trap::<AREA, SLOT>` which logs the area and slot once and answers `E_NOTIMPL`. Every
other API is refused and logged.

Diagnostics on stderr are off unless `NATIV64_LOG` lists channels (comma separated, read once):
`debug` echoes the strings passed to `OutputDebugString*`, `seh` prints each exception as its
dispatch starts (software ones too), `crt` prints each invalid parameter the C runtime rejects
with the nativ64 function that raised it, `all` turns on every channel.

`GameInput.dll` is imported by ordinal: `src/gameinput.rs` has an `ORDINALS` table like
`ws2_32`, wired in `lookup_ordinal`. It exposes an `IGameInput` with one virtual gamepad (always
connected; device callbacks enumerate it, readings answer `GetGamepadState`). Buttons come from
`NATIV64_INPUT_SCRIPT` (`<mask hex>@<seconds>[+<duration>]`, comma separated, seconds from the
first reading query, default hold 0.3 s: `4@25,4@35` presses A twice) and from
`gameinput::set_keyboard_buttons`, which the Wayland window backend calls from its keyboard
events (OR-ed with the script). Keys are raw evdev codes, i.e. physical positions of a QWERTY
layout (the keymap is ignored); key repeat is ignored and focus loss releases everything.
Enter / Space = A; Escape / Backspace = B; X = X; Y = Y; arrows = dpad; Tab = View; M = Menu;
Q / E = left / right shoulder (0x400 / 0x800). There are no triggers or sticks in the reading,
so those have no keys. Reading callbacks are stored and never fire.

## Windowing (`src/windowing.rs`)

`ext-ms-win-rtcore-ntuser-{window,window-ansi,message,message-ansi}` (also exported under
`user32`) share `windowing.rs`: class registry (atoms from `0xC000`, case-insensitive names,
`ERROR_CLASS_ALREADY_EXISTS` / `ERROR_INVALID_PARAMETER`), window table (HWNDs from `0x10000`,
distinct from kernel handles) and one message queue per thread. `CreateWindowExA` calls the
class procedure synchronously: `WM_NCCREATE` (FALSE aborts, NULL), `WM_CREATE` (-1 aborts), then
`WM_SIZE`, `WM_MOVE`, and `ShowWindow` for `WS_VISIBLE`. `ShowWindow` sends `WM_SHOWWINDOW`,
`WM_SIZE`, `WM_ACTIVATE`. `PeekMessageA` never blocks (hwnd / range filters, `PM_REMOVE`).
`DefWindowProcA`: `WM_NCCREATE` -> TRUE, `WM_CLOSE` -> destroy (`WM_DESTROY`, `WM_NCDESTROY`).
Skipped: `WM_GETMINMAXINFO`, `WM_NCCALCSIZE`, `WM_WINDOWPOSCHANGING/CHANGED`, non-client area,
focus chains, paint/timer, sent messages, `W` variants, `WM_CHAR` from `TranslateMessage`.
The host seam is the `WindowBackend` trait (`on_create`, `on_show`, `on_destroy`,
`poll_events`); only `HeadlessBackend` exists, the Wayland/X11 Vulkan surface comes later.

## File paths (rule for the file APIs)

`\` maps to `/`; a drive prefix (`G:`) is dropped. Reads search a list of directories: the package
root (the exe's directory) first, then the extra directories set with
`nativ64::paths::set_extra_package_dirs` (`nativ64-run --data DIR`, repeatable), then those of the
environment variable `NATIV64_PACKAGE_DIRS` (colon-separated, empty entries ignored). The first
directory where the path exists wins (each component is matched ignoring case when the exact name
is missing); when none has it, the package-root mapping is returned. A path starting with `/` as
written is a host path. The package directories are never written: writes go to
`$XDG_DATA_HOME/nativ64/<exe stem>` (default `~/.local/share`). `nativ64::paths` implements the
read side (`map_read_path`, `package_dirs`), plus `package_root` and `write_dir`.

Guest DLLs the base mapped are registered with `nativ64::process::register_module` (name, base,
export lookup): `GetModuleHandle*` returns their real base and `GetProcAddress` resolves their
exports.

## ABI rules

- Every export is `unsafe extern "win64"`: the Windows x64 convention, whatever the host's.
- Use Windows sizes: `DWORD` = `u32`, `BOOL` = `i32`, `LONG` = `i32` (never `i64`; Linux
  `long` is 64-bit), `LONGLONG` = `i64`, `HANDLE` = pointer-sized.
- Strings are UTF-16 (`u16`) for `W` functions; `A` functions take the ANSI code page.
- Failures return the documented value and set the error with `set_last_error` (read back by
  `GetLastError`, thread-local), e.g. `ERROR_NOACCESS` (998) for a bad pointer.
- Never unwind across the boundary: no panics in exports.

## Base adapters

A base plugs in through a thin adapter. It must:

1. For each import `module!function` (or ordinal) of the loaded binaries, call
   `nativ64_lookup(module, name)` (C) or `nativ64::lookup` (Rust). Module names may be given
   as in the import table; the library normalizes them.
2. Patch the returned address into the import thunk. NULL means "not implemented here":
   the base falls back to its own implementation or traps.
3. Call the functions with the win64 convention (on Linux, the base's thunk or JIT must
   handle the switch from System V).
4. Own what the library does not: loading PE images, TLS setup, and anything outside the
   Windows host APIs (Direct3D, GPU).

### Main module

A base that loads the main image tells the library where it is, once, after mapping and before
running the entry point: Rust `nativ64::process::set_main_module(base, path)` (`path` is UTF-8,
the file's path as the base knows it). `GetModuleHandle(NULL)`, `GetModuleFileName(NULL, ...)`
and `GetModuleHandle(<exe name>)` read it. Without it they fail with `ERROR_MOD_NOT_FOUND`.
The library gives host modules (kernel32, `api-ms-win-*`, ...) synthetic handles for
`LoadLibraryEx`/`GetProcAddress`; other DLLs are not found.
