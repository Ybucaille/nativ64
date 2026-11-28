//! Crash and timeout reporting for guest code.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};

static IMAGE_BASE: AtomicU64 = AtomicU64::new(0);
static IMAGE_SIZE: AtomicUsize = AtomicUsize::new(0);
const MAX_DLLS: usize = 32;
const NAME_LEN: usize = 32;
static DLL_COUNT: AtomicUsize = AtomicUsize::new(0);
static DLL_BASE: [AtomicU64; MAX_DLLS] = [const { AtomicU64::new(0) }; MAX_DLLS];
static DLL_END: [AtomicU64; MAX_DLLS] = [const { AtomicU64::new(0) }; MAX_DLLS];
static DLL_NAME: [[AtomicU8; NAME_LEN]; MAX_DLLS] =
    [const { [const { AtomicU8::new(0) }; NAME_LEN] }; MAX_DLLS];
const ALT_STACK: usize = 64 * 1024;
/// `NATIV64_NO_SEH=1`: faults are reported, not turned into guest exceptions.
static NO_SEH: AtomicBool = AtomicBool::new(false);

struct Buf {
    b: [u8; 1024],
    n: usize,
}

impl std::fmt::Write for Buf {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let k = s.len().min(self.b.len() - self.n);
        self.b[self.n..self.n + k].copy_from_slice(&s.as_bytes()[..k]);
        self.n += k;
        Ok(())
    }
}

fn say(args: std::fmt::Arguments) {
    use std::fmt::Write;
    let mut b = Buf { b: [0; 1024], n: 0 };
    let _ = b.write_fmt(args);
    unsafe { libc::write(2, b.b.as_ptr().cast(), b.n) };
}

/// Prints `image+0xOFF` / `dll+0xOFF` for a guest `rip`.
fn say_location(rip: u64) {
    let base = IMAGE_BASE.load(Ordering::Relaxed);
    let size = IMAGE_SIZE.load(Ordering::Relaxed) as u64;
    if base != 0 && rip >= base && rip < base + size {
        say(format_args!(" (image+{:#x})", rip - base));
    }
    for i in 0..DLL_COUNT.load(Ordering::Acquire).min(MAX_DLLS) {
        let b = DLL_BASE[i].load(Ordering::Relaxed);
        if rip >= b && rip < DLL_END[i].load(Ordering::Relaxed) {
            let mut n = [0u8; NAME_LEN];
            for (d, s) in n.iter_mut().zip(&DLL_NAME[i]) {
                *d = s.load(Ordering::Relaxed);
            }
            let len = n.iter().position(|&c| c == 0).unwrap_or(NAME_LEN);
            say(format_args!(
                " ({}+{:#x})",
                String::from_utf8_lossy(&n[..len]),
                rip - b
            ));
        }
    }
}

/// For a fatal fault in host code (our binary, a system library): its file and the offset there,
/// for addr2line. Not for the sampler: `dladdr` takes the loader's lock, which the sampled thread
/// may hold.
fn say_host_location(rip: u64) {
    // SAFETY: `dladdr` only reads the loader's tables; `di` is plain data it fills.
    let mut di: libc::Dl_info = unsafe { std::mem::zeroed() };
    if unsafe { libc::dladdr(rip as *const libc::c_void, &mut di) } != 0
        && !di.dli_fbase.is_null()
        && !di.dli_fname.is_null()
    {
        // SAFETY: a NUL-terminated path the loader keeps for the process lifetime.
        let file = unsafe { std::ffi::CStr::from_ptr(di.dli_fname) }.to_string_lossy();
        let file = file.rsplit('/').next().unwrap_or("?");
        say(format_args!(" ({file}+{:#x})", rip - di.dli_fbase as u64));
    }
}

/// The report of an exception nobody handled (`nativ64::seh`'s fatal hook): same shape as a
/// fault in host code, then ends the process.
fn report_unhandled(rec: *const nativ64::seh::ExceptionRecord) {
    let r = unsafe { &*rec };
    say(format_args!(
        "nativ64: fatal exception {:#x} at address {:#x}, rip {:#x}",
        r.code, r.info[1], r.address
    ));
    say_location(r.address as u64);
    say(format_args!("\n"));
    unsafe { libc::_exit(139) }
}

const STATUS_ACCESS_VIOLATION: u32 = 0xc000_0005;
const STATUS_BREAKPOINT: u32 = 0x8000_0003;
const STATUS_ILLEGAL_INSTRUCTION: u32 = 0xc000_001d;
const STATUS_INTEGER_DIVIDE_BY_ZERO: u32 = 0xc000_0094;
const STATUS_INTEGER_OVERFLOW: u32 = 0xc000_0095;
const SI_KERNEL: i32 = 0x80;
const TRAP_BRKPT: i32 = 1;
const FPE_INTOVF: i32 = 2;

/// The Windows exception a signal stands for: `(code, info[0..2], nparams, back)` where `back`
/// is how many bytes `rip` moves back (an `int3` reports the instruction itself). `None` for a
/// signal that has no counterpart (single-step, ...). `err` is the page-fault error code,
/// `addr` the faulting address, `si_code` the signal's code.
fn classify(sig: i32, si_code: i32, err: u64, addr: usize) -> Option<(u32, [usize; 2], u32, u64)> {
    match sig {
        libc::SIGSEGV | libc::SIGBUS => {
            // bit 1: write, bit 4: instruction fetch.
            let kind = if err & 0x10 != 0 { 8 } else { (err >> 1) & 1 };
            Some((STATUS_ACCESS_VIOLATION, [kind as usize, addr], 2, 0))
        }
        libc::SIGILL => Some((STATUS_ILLEGAL_INSTRUCTION, [0; 2], 0, 0)),
        libc::SIGFPE if si_code == FPE_INTOVF => Some((STATUS_INTEGER_OVERFLOW, [0; 2], 0, 0)),
        libc::SIGFPE => Some((STATUS_INTEGER_DIVIDE_BY_ZERO, [0; 2], 0, 0)),
        libc::SIGTRAP if si_code == SI_KERNEL || si_code == TRAP_BRKPT => {
            Some((STATUS_BREAKPOINT, [0; 2], 0, 1))
        }
        _ => None,
    }
}

/// Whether a fault should become a guest exception: `NATIV64_NO_SEH` unset and the (exception)
/// address inside a guest image.
fn dispatchable(rip: u64) -> bool {
    !NO_SEH.load(Ordering::Relaxed) && in_guest_code(rip)
}

/// Rewrites the signal context so that returning from the handler runs
/// [`nativ64::seh::dispatch_fault`] on the guest stack, with the faulting `CONTEXT` and a record
/// copied there. False (nothing changed) when the stack below the fault is not mapped.
unsafe fn divert_to_dispatch(
    uc: *mut libc::ucontext_t,
    code: u32,
    info: [usize; 2],
    nparams: u32,
    back: u64,
) -> bool {
    use nativ64::seh::*;
    let g = &mut (*uc).uc_mcontext.gregs;
    let rsp = g[libc::REG_RSP as usize] as u64;
    let rip = g[libc::REG_RIP as usize] as u64 - back;
    // Below the red zone: context, record, home space and a fake return address.
    let top = (rsp.wrapping_sub(128)) & !15;
    let ctx = top.wrapping_sub(CONTEXT_SIZE as u64);
    let rec = ctx.wrapping_sub(std::mem::size_of::<ExceptionRecord>() as u64) & !15;
    let new_rsp = rec.wrapping_sub(0x28);
    if new_rsp > rsp || !stack_mapped(new_rsp, rsp - new_rsp) {
        return false;
    }
    let c = ctx as *mut u8;
    std::ptr::write_bytes(c, 0, CONTEXT_SIZE);
    let put = |off: usize, v: u64| (c.add(off) as *mut u64).write(v);
    put(CONTEXT_FLAGS_OFF, CONTEXT_CAPTURE_FLAGS as u64);
    let fp = (*uc).uc_mcontext.fpregs;
    if !fp.is_null() {
        std::ptr::copy_nonoverlapping(fp as *const u8, c.add(0x100), 512);
        (c.add(CONTEXT_MXCSR_OFF) as *mut u32).write((*fp).mxcsr);
    }
    (c.add(CONTEXT_SEGCS_OFF) as *mut u16).write(g[libc::REG_CSGSFS as usize] as u16);
    (c.add(CONTEXT_EFLAGS_OFF) as *mut u32).write(g[libc::REG_EFL as usize] as u32);
    for (i, r) in GREG.iter().enumerate() {
        put(CONTEXT_RAX_OFF + 8 * i, g[*r as usize] as u64);
    }
    put(CONTEXT_RIP_OFF, rip);
    let r = rec as *mut ExceptionRecord;
    r.write(std::mem::zeroed());
    (*r).code = code;
    (*r).address = rip as usize;
    (*r).nparams = nparams;
    (*r).info[0] = info[0];
    (*r).info[1] = info[1];
    (new_rsp as *mut u64).write(0);
    g[libc::REG_RCX as usize] = rec as i64;
    g[libc::REG_RDX as usize] = ctx as i64;
    g[libc::REG_RSP as usize] = new_rsp as i64;
    g[libc::REG_RIP as usize] = dispatch_fault as *const () as usize as i64;
    true
}

/// Whether `[addr, addr + len)` (at most a few pages) is mapped, without faulting.
fn stack_mapped(addr: u64, len: u64) -> bool {
    let start = addr & !0xfff;
    let end = (addr + len + 0xfff) & !0xfff;
    let mut v = [0u8; 8];
    let pages = ((end - start) / 4096) as usize;
    pages <= v.len()
        && unsafe {
            libc::mincore(
                start as *mut libc::c_void,
                (end - start) as usize,
                v.as_mut_ptr(),
            )
        } == 0
}

extern "C" fn on_fault(sig: i32, info: *mut libc::siginfo_t, uc: *mut libc::c_void) {
    if sig == libc::SIGILL && unsafe { emulate_cpu_gap(uc as *mut libc::ucontext_t) } {
        return;
    }
    let uc = uc as *mut libc::ucontext_t;
    let (addr, rip, code, err) = unsafe {
        (
            (*info).si_addr() as usize,
            (*uc).uc_mcontext.gregs[libc::REG_RIP as usize] as u64,
            (*info).si_code,
            (*uc).uc_mcontext.gregs[libc::REG_ERR as usize] as u64,
        )
    };
    if let Some((status, args, n, back)) = classify(sig, code, err, addr) {
        if dispatchable(rip - back) && unsafe { divert_to_dispatch(uc, status, args, n, back) } {
            return;
        }
    }
    let name = match sig {
        libc::SIGSEGV => "SIGSEGV",
        libc::SIGILL => "SIGILL",
        libc::SIGBUS => "SIGBUS",
        libc::SIGFPE => "SIGFPE",
        libc::SIGTRAP => "SIGTRAP",
        _ => "signal",
    };
    say(format_args!(
        "nativ64: fatal {name} ({sig}) at address {addr:#x}, rip {rip:#x}"
    ));
    say_location(rip);
    say_host_location(rip);
    say(format_args!("\n"));
    unsafe { libc::_exit(139) }
}

extern "C" fn on_alarm(_: i32) {
    say(format_args!("nativ64: timeout\n"));
    unsafe { libc::_exit(124) }
}

unsafe fn handle(sig: i32, f: usize, flags: i32) {
    let mut sa: libc::sigaction = std::mem::zeroed();
    sa.sa_sigaction = f;
    sa.sa_flags = flags;
    libc::sigemptyset(&mut sa.sa_mask);
    libc::sigaction(sig, &sa, std::ptr::null_mut());
}

/// Installs the SIGSEGV/SIGILL/SIGBUS/SIGFPE/SIGTRAP reporter for a process whose guest image is at
/// `[base, base+size)`. Each guest thread must also call [`setup_thread`].
pub fn install(base: u64, size: usize) {
    IMAGE_BASE.store(base, Ordering::Relaxed);
    IMAGE_SIZE.store(size, Ordering::Relaxed);
    NO_SEH.store(
        std::env::var_os("NATIV64_NO_SEH").is_some_and(|v| v != "0" && !v.is_empty()),
        Ordering::Relaxed,
    );
    nativ64::seh::set_fatal_hook(report_unhandled);
    if std::env::var_os("NATIV64_TRACE_FAULTS").is_some_and(|v| v != "0" && !v.is_empty()) {
        nativ64::seh::set_fault_trace();
    }
    start_sampler();
    start_peek();
    start_trace_trigger();
    for sig in [
        libc::SIGSEGV,
        libc::SIGILL,
        libc::SIGBUS,
        libc::SIGFPE,
        libc::SIGTRAP,
    ] {
        unsafe {
            handle(
                sig,
                on_fault as extern "C" fn(i32, *mut libc::siginfo_t, *mut libc::c_void) as usize,
                libc::SA_SIGINFO | libc::SA_ONSTACK,
            )
        };
    }
}

/// Registers a package DLL so that a fault inside it prints `name+0xOFF` (at most 32 DLLs; names
/// are cut to 31 bytes). Call before the guest runs.
pub fn register_dll(name: &str, base: u64, size: usize) {
    let i = DLL_COUNT.load(Ordering::Relaxed);
    if i >= MAX_DLLS {
        return;
    }
    for (k, c) in name.bytes().take(NAME_LEN - 1).enumerate() {
        DLL_NAME[i][k].store(c, Ordering::Relaxed);
    }
    DLL_BASE[i].store(base, Ordering::Relaxed);
    DLL_END[i].store(base + size as u64, Ordering::Relaxed);
    DLL_COUNT.store(i + 1, Ordering::Release);
}

const MAX_THREADS: usize = 256;
static THREAD_TIDS: [AtomicU64; MAX_THREADS] = [const { AtomicU64::new(0) }; MAX_THREADS];
static THREAD_COUNT: AtomicUsize = AtomicUsize::new(0);

/// `SIGUSR2` handler of `NATIV64_SAMPLE_THREADS`: one line with the thread's tid, its rip
/// (`image+0xOFF` / `dll+0xOFF` when in guest code, else a host address) and up to 8 return
/// addresses into guest code found in the first 4 pages of its stack.
extern "C" fn on_sample(_: i32, _: *mut libc::siginfo_t, uc: *mut libc::c_void) {
    let uc = uc as *mut libc::ucontext_t;
    let (rip, rsp) = unsafe {
        (
            (*uc).uc_mcontext.gregs[libc::REG_RIP as usize] as u64,
            (*uc).uc_mcontext.gregs[libc::REG_RSP as usize] as u64,
        )
    };
    let tid = unsafe { libc::syscall(libc::SYS_gettid) };
    say(format_args!(
        "nativ64: sample tid {tid} rip {rip:#x}{}",
        if in_guest_code(rip) { "" } else { " (host)" }
    ));
    say_location(rip);
    let mut addr = rsp & !7;
    let mut found = 0;
    let mut buf = [0u64; 512];
    // Four pages at most, each read with `process_vm_readv` so an unreadable page (a guard) just
    // ends the scan.
    'pages: for _ in 0..4 {
        let len = 4096 - (addr & 0xfff) as usize;
        let local = libc::iovec {
            iov_base: buf.as_mut_ptr().cast(),
            iov_len: len,
        };
        let remote = libc::iovec {
            iov_base: addr as *mut libc::c_void,
            iov_len: len,
        };
        let n = unsafe { libc::process_vm_readv(libc::getpid(), &local, 1, &remote, 1, 0) };
        if n != len as isize {
            break;
        }
        for v in &buf[..len / 8] {
            if in_guest_code(*v) {
                say(format_args!(" <- {v:#x}"));
                say_location(*v);
                found += 1;
                if found == 8 {
                    break 'pages;
                }
            }
        }
        addr += len as u64;
    }
    say(format_args!("\n"));
}

/// `NATIV64_SAMPLE_THREADS=<secs>`: a host thread signals every registered guest thread each
/// `secs` seconds and each prints where it is (see [`on_sample`]). Off when unset.
/// `NATIV64_SAMPLE_THREADS_AT=<secs>[,<secs>...]` instead samples once at each of these times
/// (seconds after start), to look at the threads without disturbing them all run long.
fn start_sampler() {
    let at: Vec<u64> = std::env::var("NATIV64_SAMPLE_THREADS_AT")
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_default();
    let every = std::env::var("NATIV64_SAMPLE_THREADS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&s| s > 0);
    if at.is_empty() && every.is_none() {
        return;
    }
    unsafe {
        handle(
            libc::SIGUSR2,
            on_sample as extern "C" fn(i32, *mut libc::siginfo_t, *mut libc::c_void) as usize,
            libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_RESTART,
        )
    };
    let t0 = std::time::Instant::now();
    let mut next = at.into_iter();
    let _ = std::thread::Builder::new()
        .name("nativ64-sample".into())
        .spawn(move || loop {
            let wait = match every {
                Some(secs) => std::time::Duration::from_secs(secs),
                None => match next.next() {
                    Some(t) => std::time::Duration::from_secs(t).saturating_sub(t0.elapsed()),
                    None => return,
                },
            };
            std::thread::sleep(wait);
            say(format_args!("nativ64: samples\n"));
            let pid = unsafe { libc::getpid() };
            for slot in &THREAD_TIDS[..THREAD_COUNT.load(Ordering::Acquire).min(MAX_THREADS)] {
                let tid = slot.load(Ordering::Relaxed);
                unsafe { libc::syscall(libc::SYS_tgkill, pid, tid as i64, libc::SIGUSR2) };
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        });
}

/// Reads `len` bytes of this process at `addr`, `None` when unmapped.
fn read_guest(addr: u64, buf: &mut [u8]) -> Option<()> {
    let local = libc::iovec {
        iov_base: buf.as_mut_ptr().cast(),
        iov_len: buf.len(),
    };
    let remote = libc::iovec {
        iov_base: addr as *mut libc::c_void,
        iov_len: buf.len(),
    };
    let n = unsafe { libc::process_vm_readv(libc::getpid(), &local, 1, &remote, 1, 0) };
    (n == buf.len() as isize).then_some(())
}

/// Parses one `NATIV64_PEEK` item: `ADDR` or `ADDR>OFF` (hex, read the qword at ADDR, add OFF,
/// then read there), giving `(addr, off)`.
fn parse_peek(item: &str) -> Option<(u64, Option<u64>)> {
    let hex = |t: &str| u64::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok();
    match item.split_once('>') {
        Some((a, o)) => Some((hex(a)?, Some(hex(o)?))),
        None => Some((hex(item)?, None)),
    }
}

/// `NATIV64_PEEK=addr[,addr>off...]` (hex guest addresses): a host thread reads 8 bytes at each
/// address every 100 ms and prints a line, with the elapsed seconds, whenever one changes
/// (`addr>off` follows the qword stored at `addr` first). Off when unset.
fn start_peek() {
    let Ok(spec) = std::env::var("NATIV64_PEEK") else {
        return;
    };
    let items: Vec<(String, (u64, Option<u64>))> = spec
        .split(',')
        .filter_map(|i| Some((i.to_string(), parse_peek(i)?)))
        .collect();
    let _ = std::thread::Builder::new()
        .name("nativ64-peek".into())
        .spawn(move || {
            let t0 = std::time::Instant::now();
            let mut last: Vec<Option<u64>> = vec![None; items.len()];
            loop {
                for (k, (name, (addr, off))) in items.iter().enumerate() {
                    let mut b = [0u8; 8];
                    let mut at = Some(*addr);
                    if let Some(off) = off {
                        at = read_guest(*addr, &mut b).map(|()| u64::from_le_bytes(b) + off);
                    }
                    let v = at.and_then(|a| read_guest(a, &mut b).map(|()| u64::from_le_bytes(b)));
                    if v != last[k] {
                        last[k] = v;
                        say(format_args!(
                            "nativ64: peek {name} = {v:x?} at {:.1}s\n",
                            t0.elapsed().as_secs_f32()
                        ));
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        });
}

/// `NATIV64_TRACE_TRIGGER=addr` (hex guest address, with `--trace-imports`): the trace lines are
/// kept in a ring of the last 6000 until the byte at `addr` goes from non-zero to zero (polled
/// every 2 ms); the ring is then printed and tracing goes on live. Catches what led to a flag flip.
fn start_trace_trigger() {
    let Some(addr) = std::env::var("NATIV64_TRACE_TRIGGER")
        .ok()
        .and_then(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok())
    else {
        return;
    };
    crate::stubs::trace_ring_start();
    let _ = std::thread::Builder::new()
        .name("nativ64-trigger".into())
        .spawn(move || {
            let mut was_set = false;
            loop {
                let mut b = [0u8; 1];
                match read_guest(addr, &mut b) {
                    Some(()) if b[0] != 0 => was_set = true,
                    Some(()) if was_set => {
                        say(format_args!("nativ64: trace trigger fired\n"));
                        crate::stubs::trace_ring_flush();
                        return;
                    }
                    _ => {}
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        });
}

/// Gives the calling thread an alternate signal stack (leaked: it lives as long as the thread).
pub fn setup_thread() {
    let i = THREAD_COUNT.fetch_add(1, Ordering::AcqRel);
    if i < MAX_THREADS {
        THREAD_TIDS[i].store(
            unsafe { libc::syscall(libc::SYS_gettid) } as u64,
            Ordering::Relaxed,
        );
    }
    let mem = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            ALT_STACK,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if mem == libc::MAP_FAILED {
        return;
    }
    let ss = libc::stack_t {
        ss_sp: mem,
        ss_flags: 0,
        ss_size: ALT_STACK,
    };
    unsafe { libc::sigaltstack(&ss, std::ptr::null_mut()) };
}

/// Exits with status 124 after `secs` seconds (0 disables).
pub fn set_timeout(secs: u32) {
    if secs == 0 {
        return;
    }
    unsafe {
        handle(libc::SIGALRM, on_alarm as extern "C" fn(i32) as usize, 0);
        libc::alarm(secs);
    }
}

/// Registers and flags of a faulted thread, in x86 encoding order (`rax`, `rcx`, `rdx`, `rbx`, `rsp`,
/// `rbp`, `rsi`, `rdi`, `r8`..`r15`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Regs {
    pub gpr: [u64; 16],
    pub rip: u64,
    pub eflags: u64,
}

/// Index of the ucontext greg holding each x86-encoded register.
const GREG: [i32; 16] = [
    libc::REG_RAX,
    libc::REG_RCX,
    libc::REG_RDX,
    libc::REG_RBX,
    libc::REG_RSP,
    libc::REG_RBP,
    libc::REG_RSI,
    libc::REG_RDI,
    libc::REG_R8,
    libc::REG_R9,
    libc::REG_R10,
    libc::REG_R11,
    libc::REG_R12,
    libc::REG_R13,
    libc::REG_R14,
    libc::REG_R15,
];

/// Flags a `cmp` writes: CF, PF, AF, ZF, SF, OF.
const CMP_FLAGS: u64 = 0x8d5;

fn cmp_flags(a: u64, b: u64) -> u64 {
    let r = a.wrapping_sub(b);
    let mut f = 0;
    if a < b {
        f |= 1;
    }
    if (r as u8).count_ones().is_multiple_of(2) {
        f |= 1 << 2;
    }
    if (a ^ b ^ r) & 0x10 != 0 {
        f |= 1 << 4;
    }
    if r == 0 {
        f |= 1 << 6;
    }
    if r >> 63 != 0 {
        f |= 1 << 7;
    }
    if ((a ^ b) & (a ^ r)) >> 63 != 0 {
        f |= 1 << 11;
    }
    f
}

/// Decodes `REX.W cmp r/m64, r64` (`39 /r`, register form) into `(rm, reg)`: the instruction
/// computes `rm - reg`.
fn decode_cmp(b: &[u8]) -> Option<(usize, usize)> {
    let [rex @ (0x48 | 0x49 | 0x4c | 0x4d), 0x39, modrm] = *b else {
        return None;
    };
    if modrm >> 6 != 3 {
        return None;
    }
    let reg = ((modrm >> 3) & 7) as usize | ((rex as usize & 4) << 1);
    let rm = (modrm & 7) as usize | ((rex as usize & 1) << 3);
    Some((rm, reg))
}

/// Shape A tail after `clzero`: `add rax,0x40; cmp rax,b; jb -12`. Returns `b`.
fn shape_a(after: &[u8]) -> Option<usize> {
    let [0x48, 0x83, 0xc0, 0x40, cmp @ .., 0x72, 0xf4] = after else {
        return None;
    };
    match decode_cmp(cmp)? {
        (0, b) if b != 0 => Some(b),
        _ => None,
    }
}

/// Shape B around `clzero`: returns `(a, b)`.
fn shape_b(before: &[u8], after: &[u8]) -> Option<(usize, usize)> {
    let [rex @ (0x48 | 0x4c), 0x89, load, add_rex @ (0x48 | 0x49), 0x83, add, 0x40] = *before
    else {
        return None;
    };
    let [cmp @ .., 0x72, 0xf1] = after else {
        return None;
    };
    let (ca, cb) = decode_cmp(cmp)?;
    if load & 0xc7 != 0xc0 || add & 0xf8 != 0xc0 {
        return None;
    }
    let a = ((load >> 3) & 7) as usize | ((rex as usize & 4) << 1);
    let add_reg = (add & 7) as usize | ((add_rex as usize & 1) << 3);
    (a != 0 && a == add_reg && a == ca && cb != 0 && cb != a).then_some((a, cb))
}

/// Emulates the AMD-only `clzero` (`0f 01 fc`) the console CPU has and this host may lack.
///
/// `code` points at the faulting instruction; `back` and `fwd` are the numbers of bytes readable
/// before it and from it on. Zeroes the 64-byte line holding `rax`. The two loops compilers build
/// around it (`a`, `b` any registers but `rax`) are run whole, leaving registers and flags as their
/// last pass would, since each exception costs about a microsecond:
///
/// ```text
/// top: clzero              top: mov rax, a
///      add rax, 0x40            add a, 0x40
///      cmp rax, b               clzero
///      jb top                   cmp a, b
///                               jb top
/// ```
///
/// Anything else is one line and the next instruction. Returns `false` when `code` is not `clzero`.
///
/// # Safety
/// `code` must be readable as described, and the lines `regs` names are the guest's to zero.
pub(crate) unsafe fn emulate_clzero(
    regs: &mut Regs,
    code: *const u8,
    back: usize,
    fwd: usize,
) -> bool {
    if fwd < 3 || unsafe { std::slice::from_raw_parts(code, 3) } != [0x0f, 0x01, 0xfc] {
        return false;
    }
    let zero = |a: u64| unsafe { ((a & !63) as *mut u8).write_bytes(0, 64) };
    // Shape A: clzero; add rax,0x40; cmp rax,b; jb top.
    if fwd >= 12 {
        let after = unsafe { std::slice::from_raw_parts(code.add(3), 9) };
        if let Some(b) = shape_a(after) {
            let limit = regs.gpr[b];
            loop {
                zero(regs.gpr[0]);
                regs.gpr[0] = regs.gpr[0].wrapping_add(0x40);
                if regs.gpr[0] >= limit {
                    break;
                }
            }
            regs.eflags = regs.eflags & !CMP_FLAGS | cmp_flags(regs.gpr[0], limit);
            regs.rip += 12;
            return true;
        }
    }
    // Shape B: mov rax,a; add a,0x40; clzero; cmp a,b; jb top.
    if back >= 7 && fwd >= 8 {
        let (before, after) = unsafe {
            (
                std::slice::from_raw_parts(code.sub(7), 7),
                std::slice::from_raw_parts(code.add(3), 5),
            )
        };
        if let Some((a, cb)) = shape_b(before, after) {
            {
                let limit = regs.gpr[cb];
                loop {
                    zero(regs.gpr[0]);
                    if regs.gpr[a] >= limit {
                        break;
                    }
                    regs.gpr[0] = regs.gpr[a];
                    regs.gpr[a] = regs.gpr[a].wrapping_add(0x40);
                }
                regs.eflags = regs.eflags & !CMP_FLAGS | cmp_flags(regs.gpr[a], limit);
                regs.rip += 8;
                return true;
            }
        }
    }
    zero(regs.gpr[0]);
    regs.rip += 3;
    true
}

/// Emulates the AMD-only SSE4a `extrq` / `insertq` (register forms: `66 0f 78 /0 ib ib`,
/// `66 0f 79`, `f2 0f 78 ib ib`, `f2 0f 79`, optional REX) on the saved XMM registers, as the
/// vendor manual defines them; returns the instruction's length, `None` for anything else.
///
/// `extrq` replaces the low half of its operand by `len` bits starting at bit `idx`, shifted
/// down; `insertq` replaces `len` bits of the destination's low half, from bit `idx`, by the
/// source's low bits. A length field of 0 means 64. The upper half is architecturally undefined
/// and left as it is. The register forms read `len` and `idx` from the source: its low half
/// (`extrq`) or its high half (`insertq`), 6 bits each, at bits 0 and 8.
///
/// # Safety
/// `code` points at the faulting instruction, readable for as many bytes as it is long.
pub(crate) unsafe fn emulate_sse4a(code: *const u8, xmm: &mut [[u64; 2]; 16]) -> Option<usize> {
    let b = |i: usize| unsafe { code.add(i).read() };
    let prefix = b(0);
    if !matches!(prefix, 0x66 | 0xf2) {
        return None;
    }
    let (rex, at) = if b(1) & 0xf0 == 0x40 {
        (b(1), 2)
    } else {
        (0, 1)
    };
    if b(at) != 0x0f || !matches!(b(at + 1), 0x78 | 0x79) {
        return None;
    }
    let modrm = b(at + 2);
    if modrm >> 6 != 3 {
        return None;
    }
    let reg = usize::from(modrm >> 3 & 7 | (rex & 4) << 1);
    let rm = usize::from(modrm & 7 | (rex & 1) << 3);
    let immediate = b(at + 1) == 0x78;
    let extrq = prefix == 0x66;
    let (fields, len) = if immediate {
        // The immediate extrq has the opcode extension /0 and works on `rm`.
        if extrq && reg & 7 != 0 {
            return None;
        }
        ((u64::from(b(at + 3)), u64::from(b(at + 4))), at + 5)
    } else {
        let f = if extrq { xmm[rm][0] } else { xmm[rm][1] };
        ((f & 0xff, f >> 8 & 0xff), at + 3)
    };
    let (n, shift) = ((fields.0 & 63) as u32, (fields.1 & 63) as u32);
    let bits = if n == 0 { 64 } else { n };
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    if extrq {
        // `extrq xmm1, xmm2`: xmm1 is `reg`, xmm2 (the fields) `rm`; immediate form: `rm`.
        let target = if immediate { rm } else { reg };
        xmm[target][0] = (xmm[target][0] >> shift) & mask;
    } else {
        // `insertq xmm1, xmm2`: xmm1 is `reg`, xmm2 `rm`.
        let value = (xmm[rm][0] & mask) << shift;
        xmm[reg][0] = xmm[reg][0] & !(mask << shift) | value;
    }
    Some(len)
}

/// SIGILL for an instruction the console CPU has and the host lacks: emulates it and resumes.
/// Async-signal-safe: reads and writes only registers and guest memory.
unsafe fn emulate_cpu_gap(uc: *mut libc::ucontext_t) -> bool {
    let g = &mut (*uc).uc_mcontext.gregs;
    let rip = g[libc::REG_RIP as usize] as u64;
    if !in_guest_code(rip) {
        return false;
    }
    let mut regs = Regs {
        rip,
        eflags: g[libc::REG_EFL as usize] as u64,
        ..Regs::default()
    };
    for (r, i) in regs.gpr.iter_mut().zip(GREG) {
        *r = g[i as usize] as u64;
    }
    // `monitorx` / `mwaitx` (`0f 01 fa` / `0f 01 fb`): AMD wait hints. Arming nothing and
    // returning from the wait at once is valid (a wait may wake spuriously); the title re-checks.
    let code = rip as *const u8;
    if rip as usize & 0xfff <= 0xffd
        && *code == 0x0f
        && *code.add(1) == 0x01
        && matches!(*code.add(2), 0xfa | 0xfb)
    {
        if *code.add(2) == 0xfb {
            libc::sched_yield();
        }
        g[libc::REG_RIP as usize] = (rip + 3) as i64;
        return true;
    }
    // SSE4a `extrq` / `insertq` (the console CPU is AMD; Intel hosts lack them).
    let fp = (*uc).uc_mcontext.fpregs;
    if !fp.is_null() && matches!(*code, 0x66 | 0xf2) {
        // The saved XMM registers follow the 160-byte x87 and control area of the FXSAVE image.
        let xmm = &mut *((fp as *mut u8).add(160) as *mut [[u64; 2]; 16]);
        if let Some(len) = emulate_sse4a(code, xmm) {
            g[libc::REG_RIP as usize] = (rip + len as u64) as i64;
            return true;
        }
    }
    // Stay inside the page of the faulting instruction: the neighbouring one may be unmapped.
    let off = rip as usize & 0xfff;
    if !emulate_clzero(&mut regs, rip as *const u8, off, 0x1000 - off) {
        return false;
    }
    for (r, i) in regs.gpr.iter().zip(GREG) {
        g[i as usize] = *r as i64;
    }
    g[libc::REG_RIP as usize] = regs.rip as i64;
    g[libc::REG_EFL as usize] = regs.eflags as i64;
    true
}

fn in_guest_code(rip: u64) -> bool {
    let base = IMAGE_BASE.load(Ordering::Relaxed);
    if base != 0 && rip >= base && rip < base + IMAGE_SIZE.load(Ordering::Relaxed) as u64 {
        return true;
    }
    (0..DLL_COUNT.load(Ordering::Acquire).min(MAX_DLLS)).any(|i| {
        rip >= DLL_BASE[i].load(Ordering::Relaxed) && rip < DLL_END[i].load(Ordering::Relaxed)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs the emulator on `code` placed after `back` padding bytes, with `rip` standing for
    /// `code[back..]`.
    fn run(bytes: &[u8], back: usize, regs: &mut Regs) -> bool {
        let p = bytes.as_ptr();
        regs.rip = 0x1000;
        unsafe { emulate_clzero(regs, p.add(back), back, bytes.len() - back) }
    }

    fn line_at(mem: &mut [u8; 1024], off: usize) -> u64 {
        mem.as_mut_ptr() as u64 + off as u64
    }

    fn setup() -> Box<[u8; 1024]> {
        let mut m = Box::new([0xffu8; 1024]);
        // Align the working area.
        let _ = &mut m;
        m
    }

    fn base(m: &[u8; 1024]) -> u64 {
        (m.as_ptr() as u64 + 63) & !63
    }

    fn zeroed(m: &[u8; 1024], from: u64, to: u64) -> bool {
        let o = m.as_ptr() as u64;
        (0..1024u64).all(|i| (m[i as usize] == 0) == (o + i >= from && o + i < to))
    }

    const A_LOOP: [u8; 12] = [
        0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x4c, 0x39, 0xd0, 0x72, 0xf4,
    ];

    #[test]
    fn shape_a_every_register() {
        for b in (1..16usize).filter(|&b| b != 4) {
            let mut code = A_LOOP;
            // cmp rax, b: 39 /r with rm = rax, reg = b.
            code[7] = 0x48 | if b >= 8 { 4 } else { 0 };
            code[9] = 0xc0 | ((b as u8 & 7) << 3);
            let mut m = setup();
            let s = base(&m);
            let mut regs = Regs::default();
            regs.gpr[0] = s;
            regs.gpr[b] = s + 192;
            assert!(run(&code, 0, &mut regs));
            assert_eq!(regs.gpr[0], s + 192, "b={b}");
            assert_eq!(regs.rip, 0x1000 + 12);
            assert_eq!(regs.eflags & CMP_FLAGS, cmp_flags(s + 192, s + 192));
            assert!(zeroed(&m, s, s + 192));
            let _ = line_at(&mut m, 0);
        }
    }

    #[test]
    fn shape_a_unaligned_and_odd_bound() {
        let m = setup();
        let s = base(&m);
        let mut regs = Regs::default();
        regs.gpr[0] = s + 5;
        regs.gpr[10] = s + 100;
        assert!(run(&A_LOOP, 0, &mut regs));
        // Passes at s+5 and s+69; the next rax, s+133, is past the bound.
        assert_eq!(regs.gpr[0], s + 133);
        assert!(zeroed(&m, s, s + 128));
        assert_eq!(regs.eflags & CMP_FLAGS, cmp_flags(s + 133, s + 100));
    }

    #[test]
    fn shape_b() {
        // mov rax,r10; add r10,0x40; clzero; cmp r10,r11; jb top
        let code = [
            0x4c, 0x89, 0xd0, 0x49, 0x83, 0xc2, 0x40, 0x0f, 0x01, 0xfc, 0x4d, 0x39, 0xda, 0x72,
            0xf1,
        ];
        let m = setup();
        let s = base(&m);
        let mut regs = Regs::default();
        regs.gpr[0] = s;
        regs.gpr[10] = s + 64;
        regs.gpr[11] = s + 256;
        assert!(run(&code, 7, &mut regs));
        assert_eq!(regs.gpr[10], s + 256);
        assert_eq!(regs.gpr[0], s + 192);
        assert_eq!(regs.rip, 0x1000 + 8);
        assert!(zeroed(&m, s, s + 256));
        assert_eq!(regs.eflags & CMP_FLAGS, cmp_flags(s + 256, s + 256));
    }

    #[test]
    fn sse4a_insertq_and_extrq() {
        let mut x = [[0u64; 2]; 16];
        x[0][0] = 0xaaaa_aaaa_aaaa_aaaa;
        x[1][0] = 0x1234_5678;
        // insertq xmm0, xmm1, len 16, idx 16
        assert_eq!(
            unsafe { emulate_sse4a([0xf2, 0x0f, 0x78, 0xc1, 0x10, 0x10].as_ptr(), &mut x) },
            Some(6)
        );
        assert_eq!(x[0][0], 0xaaaa_aaaa_5678_aaaa);
        // extrq xmm0, len 8, idx 20 (immediate form, /0)
        assert_eq!(
            unsafe { emulate_sse4a([0x66, 0x0f, 0x78, 0xc0, 0x08, 0x14].as_ptr(), &mut x) },
            Some(6)
        );
        assert_eq!(x[0][0], 0x67);
        // insertq xmm2, xmm3 with len 8, idx 4 in xmm3's high half
        x[2][0] = u64::MAX;
        x[3] = [0, 0x0408];
        assert_eq!(
            unsafe { emulate_sse4a([0xf2, 0x0f, 0x79, 0xd3].as_ptr(), &mut x) },
            Some(4)
        );
        assert_eq!(x[2][0], !0xff0u64);
        // extrq xmm4, xmm5 with len 4, idx 8 in xmm5's low half
        x[4][0] = 0xfedc;
        x[5] = [0x0804, 0];
        assert_eq!(
            unsafe { emulate_sse4a([0x66, 0x0f, 0x79, 0xe5].as_ptr(), &mut x) },
            Some(4)
        );
        assert_eq!(x[4][0], 0xe);
        // REX.R and REX.B reach xmm8-xmm15; length 0 means 64 bits
        x[9][0] = 0x1111;
        x[8][0] = 7;
        assert_eq!(
            unsafe { emulate_sse4a([0xf2, 0x45, 0x0f, 0x78, 0xc1, 0x00, 0x00].as_ptr(), &mut x) },
            Some(7)
        );
        assert_eq!(x[8][0], 0x1111);
        // Not SSE4a: a memory form, an extrq with another extension, other opcodes.
        assert_eq!(
            unsafe { emulate_sse4a([0xf2, 0x0f, 0x78, 0x01, 0, 0].as_ptr(), &mut x) },
            None
        );
        assert_eq!(
            unsafe { emulate_sse4a([0x66, 0x0f, 0x78, 0xc8, 0, 0].as_ptr(), &mut x) },
            None
        );
        assert_eq!(
            unsafe { emulate_sse4a([0x66, 0x0f, 0x7e, 0xc0, 0, 0].as_ptr(), &mut x) },
            None
        );
    }

    #[test]
    fn single_and_foreign_bytes() {
        let m = setup();
        let s = base(&m);
        let mut regs = Regs::default();
        regs.gpr[0] = s + 70;
        assert!(run(&[0x0f, 0x01, 0xfc, 0x90, 0x90], 0, &mut regs));
        assert_eq!(regs.rip, 0x1000 + 3);
        assert!(zeroed(&m, s + 64, s + 128));
        let mut regs = Regs::default();
        assert!(!run(&[0x0f, 0x01, 0xfd, 0x90], 0, &mut regs));
        assert!(!run(&[0x0f, 0x01], 0, &mut regs));
    }

    #[test]
    fn cpu_with_clzero_runs_real_loop() {
        // cpuid 0x80000008 EBX bit 0: CLZERO. Present: the CPU never traps, nothing to test.
        let has = std::arch::x86_64::__cpuid(0x8000_0008).ebx & 1 != 0;
        if has {
            return;
        }
        let _g = GUEST.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let page = libc::mmap(
                std::ptr::null_mut(),
                8192,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            ) as *mut u8;
            assert_ne!(page as isize, -1);
            // mov rax, rdi; mov r10, rsi; top: clzero; add rax,0x40; cmp rax,r10; jb top; ret
            let code: [u8; 22] = [
                0x48, 0x89, 0xf8, 0x49, 0x89, 0xf2, 0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x4c,
                0x39, 0xd0, 0x72, 0xf4, 0x48, 0x89, 0xc0, 0xc3,
            ];
            std::ptr::copy_nonoverlapping(code.as_ptr(), page, code.len());
            let data = page.add(4096);
            std::ptr::write_bytes(data, 0xff, 4096);
            install(page as u64, 4096);
            setup_thread();
            let f: extern "C" fn(*mut u8, *mut u8) = std::mem::transmute(page);
            f(data.add(64), data.add(64 + 256));
            for i in 0..4096 {
                let inside = (64..64 + 256).contains(&i);
                assert_eq!(*data.add(i) == 0, inside, "byte {i}");
            }
        }
    }

    // ---- faults turned into guest exceptions ----

    use nativ64::api_ms_win_core_errorhandling::AddVectoredExceptionHandler;
    use nativ64::seh::{ExceptionPointers, CONTEXT_RIP_OFF};
    use std::sync::atomic::AtomicU32;
    use std::sync::OnceLock;

    const REGION: usize = 0x10000;
    /// Tests that call `install` (it sets the process-wide image range) run one at a time.
    static GUEST: std::sync::Mutex<()> = std::sync::Mutex::new(());
    static CODE: OnceLock<usize> = OnceLock::new();
    static SEEN_DIVIDE: AtomicU32 = AtomicU32::new(0);
    static SEEN_BREAK: AtomicU32 = AtomicU32::new(0);

    /// Vectored handler: skips the `int3` and the `div ecx` of the test functions (matched by
    /// address, so other tests' exceptions pass through).
    unsafe extern "win64" fn skipper(p: *mut ExceptionPointers) -> i32 {
        let rec = &*(*p).record;
        let base = *CODE.get().unwrap();
        if rec.address < base || rec.address >= base + REGION {
            return 0;
        }
        let rip = ((*p).context.add(CONTEXT_RIP_OFF)) as *mut u64;
        match rec.code {
            STATUS_BREAKPOINT => {
                assert_eq!(rec.address, base + 0x1104, "int3 address");
                SEEN_BREAK.store(1, Ordering::SeqCst);
                *rip += 1;
                -1
            }
            STATUS_INTEGER_DIVIDE_BY_ZERO => {
                SEEN_DIVIDE.store(1, Ordering::SeqCst);
                *rip += 2;
                -1
            }
            _ => 0,
        }
    }

    fn put(base: usize, off: usize, bytes: &[u8]) {
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), (base + off) as *mut u8, bytes.len())
        }
    }

    /// Maps the generated guest code once, installs the handlers like the loader does and
    /// returns its base (with the guard that serializes the tests using `install`).
    ///
    /// Functions (RVA 0x1000 + 0x100 i, unwind info at 0x2000 + 0x100 i, stub at 0x3000):
    /// 0 `__try { *NULL } __except (1) { return 7 }`, 1 `int3; return 42`,
    /// 2 `1 / 0` (returns 5 once skipped).
    fn code() -> (std::sync::MutexGuard<'static, ()>, usize) {
        let guard = GUEST.lock().unwrap_or_else(|e| e.into_inner());
        let base = *CODE.get_or_init(|| {
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    REGION,
                    libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert_ne!(p, libc::MAP_FAILED);
            let b = p as usize;
            // jmp [rip+0] to __C_specific_handler
            put(b, 0x3000, &[0xff, 0x25, 0, 0, 0, 0]);
            let h = nativ64::vcruntime140::__C_specific_handler as *const () as u64;
            put(b, 0x3006, &h.to_le_bytes());
            let info = |flags: u8, scopes: &[[u32; 4]]| {
                let mut v = vec![1 | flags << 3, 4, 1, 0, 4, 0x42, 0, 0];
                if flags != 0 {
                    v.extend_from_slice(&0x3000u32.to_le_bytes());
                    v.extend_from_slice(&(scopes.len() as u32).to_le_bytes());
                    for x in scopes.iter().flatten() {
                        v.extend_from_slice(&x.to_le_bytes());
                    }
                }
                v
            };
            let epilogue = [0x48, 0x83, 0xc4, 0x28, 0xc3];
            let funcs: [(Vec<u8>, Vec<u8>); 3] = [
                (
                    // sub rsp,0x28; xor eax,eax; mov eax,[rax]; add rsp,0x28; ret; mov eax,7; ...
                    [
                        &[0x48, 0x83, 0xec, 0x28, 0x31, 0xc0, 0x8b, 0x00][..],
                        &epilogue,
                        &[0xb8, 7, 0, 0, 0],
                        &epilogue,
                    ]
                    .concat(),
                    info(1, &[[0x1006, 0x1008, 1, 0x100d]]),
                ),
                (
                    [
                        &[0x48, 0x83, 0xec, 0x28, 0xcc, 0xb8, 42, 0, 0, 0][..],
                        &epilogue,
                    ]
                    .concat(),
                    info(0, &[]),
                ),
                (
                    // xor ecx,ecx; mov eax,1; xor edx,edx; div ecx; mov eax,5
                    [
                        &[
                            0x48, 0x83, 0xec, 0x28, 0x31, 0xc9, 0xb8, 1, 0, 0, 0, 0x31, 0xd2, 0xf7,
                            0xf1, 0xb8, 5, 0, 0, 0,
                        ][..],
                        &epilogue,
                    ]
                    .concat(),
                    info(0, &[]),
                ),
            ];
            for (i, (c, u)) in funcs.iter().enumerate() {
                let (begin, ui) = (0x1000 + 0x100 * i as u32, 0x2000 + 0x100 * i as u32);
                put(b, begin as usize, c);
                put(b, ui as usize, u);
                let rf = [begin, begin + c.len() as u32, ui];
                for (k, v) in rf.iter().enumerate() {
                    put(b, 0x800 + 12 * i + 4 * k, &v.to_le_bytes());
                }
            }
            assert!(nativ64::seh::add_function_table(b + 0x800, 3, b));
            unsafe { AddVectoredExceptionHandler(0, skipper as *const () as usize) };
            b
        });
        // The image range is process-wide: take it over for this test (see `GUEST`).
        install(base as u64, REGION);
        setup_thread();
        (guard, base)
    }

    fn call(base: usize, rva: usize) -> u32 {
        let f: extern "win64" fn() -> u32 = unsafe { std::mem::transmute(base + rva) };
        f()
    }

    #[test]
    fn null_read_in_try_resumes_at_the_except_handler() {
        let (_g, b) = code();
        assert_eq!(call(b, 0x1000), 7);
    }

    #[test]
    fn int3_under_a_vectored_handler_is_skipped() {
        let (_g, b) = code();
        assert_eq!(call(b, 0x1100), 42);
        assert_eq!(SEEN_BREAK.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn integer_divide_by_zero_raises_c0000094() {
        let (_g, b) = code();
        assert_eq!(call(b, 0x1200), 5);
        assert_eq!(SEEN_DIVIDE.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn signals_map_to_windows_codes() {
        // page-fault error code: bit 1 write, bit 4 instruction fetch
        let av = |err| classify(libc::SIGSEGV, 1, err, 0x1234);
        assert_eq!(av(0), Some((STATUS_ACCESS_VIOLATION, [0, 0x1234], 2, 0)));
        assert_eq!(av(2).unwrap().1[0], 1);
        assert_eq!(av(0x14).unwrap().1[0], 8);
        assert_eq!(
            classify(libc::SIGBUS, 2, 0, 0).unwrap().0,
            STATUS_ACCESS_VIOLATION
        );
        assert_eq!(
            classify(libc::SIGILL, 1, 0, 0).unwrap().0,
            STATUS_ILLEGAL_INSTRUCTION
        );
        assert_eq!(
            classify(libc::SIGFPE, 1, 0, 0).unwrap().0,
            STATUS_INTEGER_DIVIDE_BY_ZERO
        );
        assert_eq!(
            classify(libc::SIGFPE, FPE_INTOVF, 0, 0).unwrap().0,
            STATUS_INTEGER_OVERFLOW
        );
        assert_eq!(
            classify(libc::SIGTRAP, SI_KERNEL, 0, 0),
            Some((STATUS_BREAKPOINT, [0; 2], 0, 1))
        );
        assert_eq!(
            classify(libc::SIGTRAP, 2, 0, 0),
            None,
            "single-step is not an exception"
        );
    }

    #[test]
    fn only_guest_code_is_dispatched() {
        let (_g, b) = code();
        let b = b as u64;
        assert!(dispatchable(b + 0x1000));
        assert!(!dispatchable(
            signals_map_to_windows_codes as *const () as u64
        ));
        assert!(!dispatchable(0));
    }
}
