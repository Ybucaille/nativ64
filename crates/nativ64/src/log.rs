//! `NATIV64_LOG`: optional diagnostics on stderr, as a comma-separated list of channels read
//! once from the environment.
//!
//! - `debug`: the strings the title passes to `OutputDebugString*`;
//! - `seh`: every exception as its dispatch starts (software ones included, which
//!   `NATIV64_TRACE_FAULTS` does not see);
//! - `crt`: every invalid parameter the C runtime rejects;
//! - `all`: every channel.

use std::fmt;
use std::sync::OnceLock;

pub(crate) const DEBUG: u8 = 1;
pub(crate) const SEH: u8 = 2;
pub(crate) const CRT: u8 = 4;

/// The channels named in `spec`; unknown names are ignored.
fn parse(spec: &str) -> u8 {
    spec.split(',').fold(0, |mask, name| {
        mask | match name.trim() {
            "debug" => DEBUG,
            "seh" => SEH,
            "crt" => CRT,
            "all" => DEBUG | SEH | CRT,
            _ => 0,
        }
    })
}

/// Whether `channel` is on.
pub(crate) fn on(channel: u8) -> bool {
    static CHANNELS: OnceLock<u8> = OnceLock::new();
    let mask = *CHANNELS.get_or_init(|| {
        std::env::var_os("NATIV64_LOG")
            .and_then(|v| v.to_str().map(parse))
            .unwrap_or(0)
    });
    mask & channel != 0
}

/// A guest address, printed relative to the image that holds it when there is one. It never
/// waits for the image table (exception logging may run while it is held): then the raw
/// address is printed.
pub(crate) struct At(pub usize);

impl fmt::Display for At {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match crate::process::image_base_of_try(self.0) {
            Some(base) => write!(f, "image+{:#x}", self.0 - base),
            None => write!(f, "{:#x}", self.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channels() {
        assert_eq!(parse(""), 0);
        assert_eq!(parse("debug"), DEBUG);
        assert_eq!(parse("seh, crt"), SEH | CRT);
        assert_eq!(parse("all"), DEBUG | SEH | CRT);
        assert_eq!(parse("nope,,1"), 0);
        assert_eq!(At(0x1234).to_string(), "0x1234");
    }
}
