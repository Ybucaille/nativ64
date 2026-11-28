//! Optional timing of the replay (`NATIV64_PROFILE=1`): the time spent in each phase of the
//! submissions and a few counters, printed to stderr every 60 presented frames, then reset.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;
use std::time::Instant;

#[derive(Clone, Copy)]
pub(super) enum Phase {
    /// The texture sync passes of the submissions (everything below is part of it).
    Sync,
    /// Write-watch registration and scans.
    Watch,
    /// Hashing of texture memory.
    Hash,
    /// Detiling and staging of the uploads.
    Upload,
    /// Recording and executing the lists (the wait for the GPU included).
    Record,
    /// Frame dumps (PNG).
    Dump,
    /// Making the images of the textures.
    Image,
    /// Writing the descriptors.
    Descr,
    /// The recording of the sync passes (the rest of `Sync` is the submission and its wait).
    SyncRec,
    /// `host_upload` and `placed_upload` as a whole.
    HostUp,
    PlacedUp,
    /// `GpuMemory::sync` and the view invalidation before a submission.
    Begin,
    /// The view table update (records written, memory imported for them).
    Views,
    /// `import_host` calls.
    Import,
    /// The commands of the lists, recorded (the submission and its wait excluded).
    Cmds,
    /// The compute dispatches among them.
    Disp,
}

#[derive(Clone, Copy)]
pub(super) enum Count {
    /// Textures looked at by the sync, and those the watch let skip.
    Looked,
    Skipped,
    /// Scans that could not tell (taken as written), and scan system calls.
    ScanNone,
    ScanCalls,
    /// Times the write log was dropped, and its largest length.
    LogDrops,
    LogMax,
    /// Bytes hashed, textures uploaded.
    Hashed,
    Uploads,
    /// Submissions.
    Submissions,
    /// Entries of the source lists visited, and descriptors written.
    Visited,
    Written,
    /// Import ranges dropped, view records rewritten, imports made.
    Dropped,
    Rewritten,
    Imports,
}

const PHASES: [&str; 16] = [
    "sync", "watch", "hash", "upload", "record", "dump", "image", "descr", "syncrec", "hostup",
    "placedup", "begin", "views", "import", "cmds", "disp",
];
const COUNTS: [&str; 14] = [
    "looked",
    "skipped",
    "scan_none",
    "scan_calls",
    "log_drops",
    "log_max",
    "hashed_mb",
    "uploads",
    "submissions",
    "visited",
    "descr_written",
    "dropped",
    "rewritten",
    "imports",
];

static NANOS: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
static COUNT: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NATIV64_PROFILE").is_some_and(|v| !v.is_empty()))
}

/// The start of a timed phase: None when profiling is off.
pub(super) fn start() -> Option<Instant> {
    enabled().then(Instant::now)
}

/// Adds the time since `since` to `phase`.
pub(super) fn stop(phase: Phase, since: Option<Instant>) {
    if let Some(t) = since {
        NANOS[phase as usize].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    }
}

pub(super) fn count(c: Count, n: u64) {
    if enabled() {
        COUNT[c as usize].fetch_add(n, Relaxed);
    }
}

pub(super) fn count_max(c: Count, n: u64) {
    if enabled() {
        COUNT[c as usize].fetch_max(n, Relaxed);
    }
}

/// Prints and resets the totals of the last `frames` frames.
pub(super) fn report(frame: u64, frames: u64) {
    if !enabled() {
        return;
    }
    let mut line = format!("nativ64-profile: frame {frame} (last {frames}):");
    for (name, n) in PHASES.iter().zip(&NANOS) {
        line += &format!(" {name}={:.2}s", n.swap(0, Relaxed) as f64 / 1e9);
    }
    for (i, (name, n)) in COUNTS.iter().zip(&COUNT).enumerate() {
        let v = n.swap(0, Relaxed);
        line += &format!(
            " {name}={}",
            if i == Count::Hashed as usize {
                v >> 20
            } else {
                v
            }
        );
    }
    eprintln!("{line}");
}

/// What one `execute` spent, measured whatever `NATIV64_PROFILE` says (a few clock reads per
/// import, upload and fence wait), so that a slow submission can say where its time went.
#[derive(Clone, Copy)]
pub(super) enum Exec {
    /// `import_host` calls and their time.
    Imports,
    ImportNs,
    /// Texture uploads, their staged bytes, and their time (hash, detile and staging).
    Uploads,
    UploadBytes,
    UploadNs,
    /// Time spent waiting on submission fences.
    WaitNs,
}

static EXEC: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

pub(super) fn exec_add(e: Exec, n: u64) {
    EXEC[e as usize].fetch_add(n, Relaxed);
}

/// Adds the time since `since` to `e`.
pub(super) fn exec_since(e: Exec, since: Instant) {
    exec_add(e, since.elapsed().as_nanos() as u64);
}

/// The totals since the last call, reset.
pub(super) fn exec_take() -> [u64; 6] {
    std::array::from_fn(|i| EXEC[i].swap(0, Relaxed))
}

/// An execute this long or longer logs where its time went.
pub(super) const SLOW_EXECUTE: std::time::Duration = std::time::Duration::from_secs(1);

/// The line a slow execute logs: `sync` and `record` are wall times of the two halves (fence
/// waits included), `took` the `exec_take` totals.
pub(super) fn slow_line(
    total: std::time::Duration,
    sync: std::time::Duration,
    passes: u32,
    record: std::time::Duration,
    draws: usize,
    took: [u64; 6],
) -> String {
    let ms = |ns: u64| ns as f64 / 1e6;
    format!(
        "slow execute {:.0} ms ({draws} draws): sync {:.0} ms in {passes} passes, record {:.0} ms; \
         imports {} in {:.0} ms, uploads {} ({:.1} MB) in {:.0} ms, fence wait {:.0} ms",
        total.as_secs_f64() * 1e3,
        sync.as_secs_f64() * 1e3,
        record.as_secs_f64() * 1e3,
        took[Exec::Imports as usize],
        ms(took[Exec::ImportNs as usize]),
        took[Exec::Uploads as usize],
        took[Exec::UploadBytes as usize] as f64 / 1e6,
        ms(took[Exec::UploadNs as usize]),
        ms(took[Exec::WaitNs as usize]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn a_slow_execute_line_gives_the_split() {
        let mut took = [0u64; 6];
        took[Exec::Imports as usize] = 120;
        took[Exec::ImportNs as usize] = 900_000_000;
        took[Exec::Uploads as usize] = 3;
        took[Exec::UploadBytes as usize] = 4_500_000;
        took[Exec::UploadNs as usize] = 50_000_000;
        took[Exec::WaitNs as usize] = 7_000_000;
        let line = slow_line(
            Duration::from_millis(1500),
            Duration::from_millis(400),
            2,
            Duration::from_millis(1100),
            42,
            took,
        );
        assert_eq!(
            line,
            "slow execute 1500 ms (42 draws): sync 400 ms in 2 passes, record 1100 ms; \
             imports 120 in 900 ms, uploads 3 (4.5 MB) in 50 ms, fence wait 7 ms"
        );
    }
}
