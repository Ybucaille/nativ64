//! nativ64: Windows host APIs (kernel32, api-ms-win-core-*, ...) reimplemented natively
//! on Linux, independent of the compatibility layer that loads the game.

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
compile_error!("nativ64 supports Linux x86_64 only");

#[macro_use]
mod registry;

pub mod acphal;
pub mod api_ms_win_core_com;
pub mod api_ms_win_core_debug;
pub mod api_ms_win_core_errorhandling;
pub mod api_ms_win_core_fibers;
pub mod api_ms_win_core_file;
pub mod api_ms_win_core_handle;
pub mod api_ms_win_core_interlocked;
pub mod api_ms_win_core_libraryloader;
pub mod api_ms_win_core_localization;
pub mod api_ms_win_core_memory;
pub mod api_ms_win_core_processenvironment;
pub mod api_ms_win_core_processthreads;
pub mod api_ms_win_core_profile;
pub mod api_ms_win_core_psm_appnotify;
pub mod api_ms_win_core_registry;
pub mod api_ms_win_core_rtlsupport;
pub mod api_ms_win_core_string;
pub mod api_ms_win_core_synch;
pub mod api_ms_win_core_sysinfo;
pub mod api_ms_win_core_util;
pub mod api_ms_win_crt_convert;
pub mod api_ms_win_crt_environment;
pub mod api_ms_win_crt_filesystem;
pub mod api_ms_win_crt_heap;
pub mod api_ms_win_crt_locale;
pub mod api_ms_win_crt_math;
pub mod api_ms_win_crt_runtime;
pub mod api_ms_win_crt_stdio;
pub mod api_ms_win_crt_string;
pub mod api_ms_win_crt_time;
pub mod api_ms_win_crt_utility;
pub mod api_ms_win_eventing_provider;
pub mod ext_ms_win_rtcore_ntuser_message;
pub mod ext_ms_win_rtcore_ntuser_message_ansi;
pub mod ext_ms_win_rtcore_ntuser_window;
pub mod ext_ms_win_rtcore_ntuser_window_ansi;

pub mod bcrypt;
pub mod chad;
pub mod com;
pub mod gameinput;
pub mod handles;
pub mod mmdevapi;
pub mod msvcp140;
pub mod paths;
pub mod printf;
pub mod process;
mod scanf;
pub mod seh;
mod slots;
mod topology;
pub mod vcruntime140;
pub mod windowing;
pub mod ws2_32;
pub mod xapu;
pub mod xasync;
pub mod xgameruntime;
pub mod xuser;

pub use api_ms_win_crt_runtime::{nativ64_set_command_line, set_command_line};
pub use registry::{lookup, lookup_ordinal, nativ64_lookup, normalize_module, Export, Status};

/// One table per Windows module file. Add a new module file here.
pub static TABLES: &[&[Export]] = &[
    acphal::EXPORTS,
    api_ms_win_core_com::EXPORTS,
    api_ms_win_core_debug::EXPORTS,
    api_ms_win_core_errorhandling::EXPORTS,
    api_ms_win_core_errorhandling::ASM_EXPORTS,
    api_ms_win_core_fibers::EXPORTS,
    api_ms_win_core_file::EXPORTS,
    api_ms_win_core_handle::EXPORTS,
    api_ms_win_core_interlocked::EXPORTS,
    api_ms_win_core_libraryloader::EXPORTS,
    api_ms_win_core_localization::EXPORTS,
    api_ms_win_core_memory::EXPORTS,
    api_ms_win_core_registry::EXPORTS,
    api_ms_win_core_rtlsupport::EXPORTS,
    api_ms_win_core_rtlsupport::ASM_EXPORTS,
    api_ms_win_core_string::EXPORTS,
    api_ms_win_core_processenvironment::EXPORTS,
    api_ms_win_core_processthreads::EXPORTS,
    api_ms_win_core_processthreads::ALIASES,
    api_ms_win_core_profile::EXPORTS,
    api_ms_win_core_psm_appnotify::EXPORTS,
    api_ms_win_core_synch::EXPORTS,
    api_ms_win_core_sysinfo::EXPORTS,
    api_ms_win_core_util::EXPORTS,
    api_ms_win_crt_convert::EXPORTS,
    api_ms_win_crt_environment::EXPORTS,
    api_ms_win_crt_filesystem::EXPORTS,
    api_ms_win_crt_heap::EXPORTS,
    api_ms_win_crt_locale::EXPORTS,
    api_ms_win_crt_math::EXPORTS,
    api_ms_win_crt_runtime::EXPORTS,
    api_ms_win_crt_stdio::EXPORTS,
    api_ms_win_crt_string::EXPORTS,
    api_ms_win_crt_time::EXPORTS,
    api_ms_win_crt_utility::EXPORTS,
    api_ms_win_eventing_provider::EXPORTS,
    ext_ms_win_rtcore_ntuser_message::EXPORTS,
    ext_ms_win_rtcore_ntuser_message_ansi::EXPORTS,
    ext_ms_win_rtcore_ntuser_window::EXPORTS,
    ext_ms_win_rtcore_ntuser_window_ansi::EXPORTS,
    bcrypt::EXPORTS,
    chad::EXPORTS,
    gameinput::EXPORTS,
    msvcp140::EXPORTS,
    vcruntime140::EXPORTS,
    ws2_32::EXPORTS,
    xapu::EXPORTS,
    xgameruntime::EXPORTS,
];

/// Every registered export (each function appears once per module name it is known under).
pub fn exports() -> impl Iterator<Item = &'static Export> {
    TABLES.iter().flat_map(|t| t.iter())
}
