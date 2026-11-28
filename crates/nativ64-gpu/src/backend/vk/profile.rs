//! Optional timing of the replay (`NATIV64_PROFILE=1`): the time spent in each phase of the
//! submissions and a few counters, printed to stderr every 60 presented frames, then reset.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};
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
    /// Pipelines built (shader translation and Vulkan creation), and their time.
    Pipelines,
    PipelineNs,
    /// Images made (`make_image`) and their time, of which device memory allocation and the
    /// evictions that made room (in `make_image` and `trim`).
    Images,
    ImageNs,
    AllocNs,
    EvictNs,
    /// The write-watch steps of the uploads (commit checks, registration, scans), of which
    /// the alias check (`xmem::is_aliased`, pool lock wait included).
    WatchNs,
    AliasNs,
    /// The overlap check of new placed images against the GPU-written ones.
    OverlapNs,
    /// The sync's preparation: residency scan of the lists and classification of the
    /// changed registry entries.
    PrepNs,
    /// Descriptor writes of the sync (views made included).
    DescrNs,
}

const EXECS: usize = 17;
static EXEC: [AtomicU64; EXECS] = [const { AtomicU64::new(0) }; EXECS];

pub(super) fn exec_add(e: Exec, n: u64) {
    EXEC[e as usize].fetch_add(n, Relaxed);
}

/// Adds the time since `since` to `e`.
pub(super) fn exec_since(e: Exec, since: Instant) {
    exec_add(e, since.elapsed().as_nanos() as u64);
}

/// Adds the time until it drops to `e` (early returns included).
pub(super) struct ExecTimer(Exec, Instant);

pub(super) fn exec_timer(e: Exec) -> ExecTimer {
    ExecTimer(e, Instant::now())
}

impl Drop for ExecTimer {
    fn drop(&mut self) {
        exec_since(self.0, self.1);
    }
}

/// The totals since the last call, reset.
pub(super) fn exec_take() -> [u64; EXECS] {
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
    took: [u64; EXECS],
) -> String {
    let ms = |ns: u64| ns as f64 / 1e6;
    format!(
        "slow execute {:.0} ms ({draws} draws): sync {:.0} ms in {passes} passes, record {:.0} ms; \
         imports {} in {:.0} ms, uploads {} ({:.1} MB) in {:.0} ms, pipelines {} in {:.0} ms, \
         fence wait {:.0} ms; sync split: prep {:.0} ms, images {} in {:.0} ms (alloc {:.0} ms, \
         evict {:.0} ms), overlap {:.0} ms, watch {:.0} ms (alias {:.0} ms), descr {:.0} ms",
        total.as_secs_f64() * 1e3,
        sync.as_secs_f64() * 1e3,
        record.as_secs_f64() * 1e3,
        took[Exec::Imports as usize],
        ms(took[Exec::ImportNs as usize]),
        took[Exec::Uploads as usize],
        took[Exec::UploadBytes as usize] as f64 / 1e6,
        ms(took[Exec::UploadNs as usize]),
        took[Exec::Pipelines as usize],
        ms(took[Exec::PipelineNs as usize]),
        ms(took[Exec::WaitNs as usize]),
        ms(took[Exec::PrepNs as usize]),
        took[Exec::Images as usize],
        ms(took[Exec::ImageNs as usize]),
        ms(took[Exec::AllocNs as usize]),
        ms(took[Exec::EvictNs as usize]),
        ms(took[Exec::OverlapNs as usize]),
        ms(took[Exec::WatchNs as usize]),
        ms(took[Exec::AliasNs as usize]),
        ms(took[Exec::DescrNs as usize]),
    )
}

/// Where the call in the backend is, for the in-progress report: set at each boundary by a
/// `Step` guard, which puts the outer step back when it ends.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(super) enum Step {
    Idle,
    /// Before the texture sync: reaping, memory sync, view invalidation.
    Begin,
    /// The texture sync passes (uploads excluded).
    Sync,
    /// Hashing, detiling and staging one texture.
    Upload,
    /// Recording the lists (the steps below excluded).
    Record,
    /// The view table update.
    Views,
    /// One `import_host` call.
    Import,
    /// Building a pipeline (shader translation and Vulkan creation).
    Pipeline,
    /// Submitting to the queue.
    Submit,
    /// Waiting on a fence.
    Wait,
    /// A present (the steps above excluded).
    Present,
}

const STEPS: [&str; 11] = [
    "idle", "begin", "sync", "upload", "record", "views", "import", "pipeline", "submit", "wait",
    "present",
];

impl Step {
    fn name(self) -> &'static str {
        STEPS[self as usize]
    }
    fn from(v: u8) -> Self {
        [
            Step::Idle,
            Step::Begin,
            Step::Sync,
            Step::Upload,
            Step::Record,
            Step::Views,
            Step::Import,
            Step::Pipeline,
            Step::Submit,
            Step::Wait,
            Step::Present,
        ]
        .get(v as usize)
        .copied()
        .unwrap_or(Step::Idle)
    }
}

static STEP: AtomicU8 = AtomicU8::new(Step::Idle as u8);
/// When the current call started (ms after `epoch()`, plus one; 0: no call), and which.
static CALL_SINCE: AtomicU64 = AtomicU64::new(0);
static CALL_KIND: AtomicU8 = AtomicU8::new(0);
const CALLS: [&str; 2] = ["execute", "present"];
/// How often the in-progress report looks, and logs while a call lasts.
const REPORT_EVERY: std::time::Duration = std::time::Duration::from_secs(2);

fn epoch() -> Instant {
    static E: OnceLock<Instant> = OnceLock::new();
    *E.get_or_init(Instant::now)
}

/// Sets the current step until the guard drops.
pub(super) fn step(s: Step) -> StepGuard {
    StepGuard(STEP.swap(s as u8, Relaxed))
}

pub(super) struct StepGuard(u8);

impl Drop for StepGuard {
    fn drop(&mut self) {
        STEP.store(self.0, Relaxed);
    }
}

/// Marks an `execute` (`present` false) or a present as in progress until the guard drops,
/// at `first`; a host thread, started at the first call, logs every `REPORT_EVERY` while a call
/// lasts longer than that.
pub(super) fn call(present: bool, first: Step) -> CallGuard {
    static WATCHER: OnceLock<()> = OnceLock::new();
    WATCHER.get_or_init(|| {
        let _ = std::thread::Builder::new()
            .name("nativ64-gpu-watch".into())
            .spawn(watch);
    });
    CALL_KIND.store(present as u8, Relaxed);
    CALL_SINCE.store(epoch().elapsed().as_millis() as u64 + 1, Relaxed);
    CallGuard { _step: step(first) }
}

pub(super) struct CallGuard {
    _step: StepGuard,
}

impl Drop for CallGuard {
    fn drop(&mut self) {
        CALL_SINCE.store(0, Relaxed);
    }
}

fn watch() {
    loop {
        std::thread::sleep(REPORT_EVERY);
        let since = CALL_SINCE.load(Relaxed);
        if since == 0 {
            continue;
        }
        let ms = (epoch().elapsed().as_millis() as u64 + 1).saturating_sub(since);
        if ms < REPORT_EVERY.as_millis() as u64 {
            continue;
        }
        let took: [u64; EXECS] = std::array::from_fn(|i| EXEC[i].load(Relaxed));
        eprintln!(
            "nativ64-gpu: {}",
            progress_line(
                CALLS[CALL_KIND.load(Relaxed) as usize & 1],
                ms,
                Step::from(STEP.load(Relaxed)),
                took
            )
        );
    }
}

/// The line of a call in progress for `ms` milliseconds, at `step`, with the execute totals
/// so far.
fn progress_line(call: &str, ms: u64, step: Step, took: [u64; EXECS]) -> String {
    let ms_of = |ns: u64| ns as f64 / 1e6;
    format!(
        "{call} in progress {:.0} s: phase={}, imports {} in {:.0} ms, uploads {} in {:.0} ms, \
         pipelines {} in {:.0} ms, fence wait {:.0} ms",
        ms as f64 / 1e3,
        step.name(),
        took[Exec::Imports as usize],
        ms_of(took[Exec::ImportNs as usize]),
        took[Exec::Uploads as usize],
        ms_of(took[Exec::UploadNs as usize]),
        took[Exec::Pipelines as usize],
        ms_of(took[Exec::PipelineNs as usize]),
        ms_of(took[Exec::WaitNs as usize]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn a_slow_execute_line_gives_the_split() {
        let mut took = [0u64; EXECS];
        took[Exec::Imports as usize] = 120;
        took[Exec::ImportNs as usize] = 900_000_000;
        took[Exec::Uploads as usize] = 3;
        took[Exec::UploadBytes as usize] = 4_500_000;
        took[Exec::UploadNs as usize] = 50_000_000;
        took[Exec::WaitNs as usize] = 7_000_000;
        took[Exec::Pipelines as usize] = 5;
        took[Exec::PipelineNs as usize] = 300_000_000;
        took[Exec::Images as usize] = 9;
        took[Exec::ImageNs as usize] = 80_000_000;
        took[Exec::AllocNs as usize] = 60_000_000;
        took[Exec::WatchNs as usize] = 20_000_000;
        took[Exec::AliasNs as usize] = 10_000_000;
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
             imports 120 in 900 ms, uploads 3 (4.5 MB) in 50 ms, pipelines 5 in 300 ms, \
             fence wait 7 ms; sync split: prep 0 ms, images 9 in 80 ms (alloc 60 ms, evict 0 ms), \
             overlap 0 ms, watch 20 ms (alias 10 ms), descr 0 ms"
        );
    }

    #[test]
    fn a_progress_line_names_the_phase() {
        let mut took = [0u64; EXECS];
        took[Exec::Pipelines as usize] = 40;
        took[Exec::PipelineNs as usize] = 6_000_000_000;
        assert_eq!(
            progress_line("execute", 6400, Step::Pipeline, took),
            "execute in progress 6 s: phase=pipeline, imports 0 in 0 ms, uploads 0 in 0 ms, \
             pipelines 40 in 6000 ms, fence wait 0 ms"
        );
    }

    #[test]
    fn step_names_match() {
        for (i, n) in STEPS.iter().enumerate() {
            assert_eq!(Step::from(i as u8).name(), *n);
        }
    }
}
