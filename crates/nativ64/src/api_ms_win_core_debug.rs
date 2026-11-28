//! `api-ms-win-core-debug`: no debugger is ever attached, so debug strings normally go
//! nowhere. `NATIV64_LOG=debug` mirrors them to stderr, which is how a title's own
//! fatal-error text is read without attaching a debugger.

use std::ffi::{c_char, c_void, CStr};
use std::io::Write;

/// Writes `nativ64: debug string: <bytes>` to stderr when the `debug` channel is on. The bytes
/// go out as the title wrote them, in one locked write.
fn echo(bytes: &[u8]) {
    if !crate::log::on(crate::log::DEBUG) {
        return;
    }
    let mut err = std::io::stderr().lock();
    let _ = err.write_all(b"nativ64: debug string: ");
    let _ = err.write_all(bytes);
    let _ = err.write_all(b"\n");
}

/// Decodes UTF-16 to UTF-8; a lone surrogate becomes U+FFFD.
fn utf16_string(units: &[u16]) -> String {
    char::decode_utf16(units.iter().copied())
        .map(|c| c.unwrap_or(char::REPLACEMENT_CHARACTER))
        .collect()
}

win32! {
    module "api-ms-win-core-debug" => "kernel32";
    // Without an attached debugger Windows drops the string (and raises nothing); the log
    // channel only mirrors it to this host's stderr.
    Native fn OutputDebugStringA(text: *const u8) {
        if !text.is_null() && crate::log::on(crate::log::DEBUG) {
            // SAFETY: the title passes a NUL-terminated ANSI string.
            echo(CStr::from_ptr(text.cast::<c_char>()).to_bytes());
        }
    }
    Native fn OutputDebugStringW(text: *const u16) {
        if !text.is_null() && crate::log::on(crate::log::DEBUG) {
            let mut len = 0;
            // SAFETY: the title passes a NUL-terminated UTF-16 string.
            while *text.add(len) != 0 {
                len += 1;
            }
            echo(utf16_string(std::slice::from_raw_parts(text, len)).as_bytes());
        }
    }
    Native fn IsDebuggerPresent() -> i32 {
        0
    }
    Native fn CheckRemoteDebuggerPresent(_process: *mut c_void, present: *mut i32) -> i32 {
        if present.is_null() {
            return 0;
        }
        *present = 0;
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lookup;

    #[test]
    fn utf16_decoding_replaces_lone_surrogates() {
        assert_eq!(utf16_string(&[0x68, 0x69, 0xD800, 0x21]), "hi\u{FFFD}!");
        assert_eq!(utf16_string(&[]), "");
    }

    #[test]
    fn no_debugger_is_attached() {
        unsafe {
            OutputDebugStringA(c"hello".as_ptr().cast());
            OutputDebugStringW(std::ptr::null());
            assert_eq!(IsDebuggerPresent(), 0);
            let mut present = 7;
            assert_eq!(
                CheckRemoteDebuggerPresent(std::ptr::null_mut(), &mut present),
                1
            );
            assert_eq!(present, 0);
            assert_eq!(
                CheckRemoteDebuggerPresent(std::ptr::null_mut(), std::ptr::null_mut()),
                0
            );
        }
        assert!(lookup("api-ms-win-core-debug-l1-1-0.dll", "OutputDebugStringA").is_some());
        assert!(lookup("kernel32.dll", "IsDebuggerPresent").is_some());
    }
}
