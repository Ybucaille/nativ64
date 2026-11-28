//! Structured exception handling, level 1: function tables, stack unwinding and context
//! capture/restore (design in `docs/seh.md`). Exports live in `api_ms_win_core_rtlsupport`.
//!
//! Everything here follows the public Microsoft x64 unwind format (`.pdata` runtime functions,
//! `UNWIND_INFO` with its unwind codes, epilog emulation). Dispatch, `RaiseException` and
//! `RtlUnwindEx` come later.

use std::sync::Mutex;

/// `RUNTIME_FUNCTION`: image-relative function range and its `UNWIND_INFO`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeFunction {
    pub begin: u32,
    pub end: u32,
    pub unwind: u32,
}

pub const UNW_FLAG_EHANDLER: u32 = 1;
pub const UNW_FLAG_UHANDLER: u32 = 2;
const UNW_FLAG_CHAININFO: u8 = 4;

pub const CONTEXT_SIZE: usize = 0x4d0;
pub const CONTEXT_FLAGS_OFF: usize = 0x30;
pub const CONTEXT_MXCSR_OFF: usize = 0x34;
pub const CONTEXT_SEGCS_OFF: usize = 0x38;
pub const CONTEXT_EFLAGS_OFF: usize = 0x44;
/// Offset of `Rax`; the 16 general registers follow in hardware order (rax, rcx, rdx, rbx,
/// rsp, rbp, rsi, rdi, r8..r15).
pub const CONTEXT_RAX_OFF: usize = 0x78;
pub const CONTEXT_RSP_OFF: usize = 0x98;
pub const CONTEXT_RIP_OFF: usize = 0xf8;
pub const CONTEXT_XMM0_OFF: usize = 0x1a0;
/// `CONTEXT_AMD64 | CONTROL | INTEGER | FLOATING_POINT`.
pub const CONTEXT_CAPTURE_FLAGS: u32 = 0x10_000b;

struct Table {
    base: usize,
    entries: usize,
    count: usize,
    dynamic: bool,
}

static TABLES: Mutex<Vec<Table>> = Mutex::new(Vec::new());

fn tables() -> std::sync::MutexGuard<'static, Vec<Table>> {
    TABLES.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether `[addr, addr + len)` is mapped (checked with `mincore`, so nothing faults).
fn mapped(addr: usize, len: usize) -> bool {
    let start = addr & !0xfff;
    let end = (addr + len + 0xfff) & !0xfff;
    let mut v = vec![0u8; (end - start) / 4096];
    unsafe { libc::mincore(start as *mut libc::c_void, end - start, v.as_mut_ptr()) == 0 }
}

/// Reads the exception directory (data directory 3) of the PE image mapped at `base` and keeps
/// its function table. Silently ignores a range that is not a mapped PE image.
pub fn register_image_table(base: usize, size: usize) {
    tables().retain(|t| t.dynamic || t.base != base);
    if size < 0x200 || !mapped(base, size.min(0x1000)) {
        return;
    }
    unsafe {
        let r16 = |o: usize| (base as *const u8).add(o).cast::<u16>().read_unaligned();
        let r32 = |o: usize| (base as *const u8).add(o).cast::<u32>().read_unaligned();
        if r16(0) != 0x5a4d {
            return;
        }
        let pe = r32(0x3c) as usize;
        if pe + 0x108 > size.min(0x1000) || r32(pe) != 0x4550 {
            return;
        }
        let opt = pe + 24;
        if r16(opt) != 0x20b || r32(opt + 0x6c) < 4 {
            return;
        }
        let (rva, len) = (
            r32(opt + 0x70 + 3 * 8) as usize,
            r32(opt + 0x74 + 3 * 8) as usize,
        );
        if rva == 0 || len < 12 || rva + len > size {
            return;
        }
        tables().push(Table {
            base,
            entries: base + rva,
            count: len / 12,
            dynamic: false,
        });
    }
}

/// `RtlAddFunctionTable`: the entries' RVAs are relative to `base`.
pub fn add_function_table(entries: usize, count: u32, base: usize) -> bool {
    if entries == 0 || count == 0 || !entries.is_multiple_of(4) {
        return false;
    }
    let mut t = tables();
    if t.iter().any(|x| x.dynamic && x.entries == entries) {
        return false;
    }
    t.push(Table {
        base,
        entries,
        count: count as usize,
        dynamic: true,
    });
    true
}

/// `RtlDeleteFunctionTable`: false when `entries` was never added.
pub fn delete_function_table(entries: usize) -> bool {
    let mut t = tables();
    let n = t.len();
    t.retain(|x| !(x.dynamic && x.entries == entries));
    t.len() != n
}

/// `RtlLookupFunctionEntry`: the base of the table (the image base for a miss inside a
/// registered image, else 0) and the runtime function covering `pc` (0 for a miss).
///
/// # Safety
/// Registered tables must still be readable.
pub unsafe fn lookup_function_entry(pc: usize) -> (usize, usize) {
    for t in tables().iter() {
        if pc < t.base || pc - t.base > u32::MAX as usize {
            continue;
        }
        let rva = (pc - t.base) as u32;
        let e = std::slice::from_raw_parts(t.entries as *const RuntimeFunction, t.count);
        if rva < e[0].begin || rva >= e[t.count - 1].end {
            continue;
        }
        let i = e.partition_point(|f| f.begin <= rva);
        if i > 0 && rva < e[i - 1].end {
            return (t.base, t.entries + (i - 1) * 12);
        }
    }
    (crate::process::image_base_of(pc).unwrap_or(0), 0)
}

#[derive(Debug, PartialEq, Eq)]
pub enum UnwindError {
    BadUnwindVersion,
    BadOpcode,
}

/// Result of one unwind step.
pub struct Unwound {
    pub handler: usize,
    pub handler_data: usize,
    pub establisher: usize,
}

#[derive(Clone, Copy)]
struct Ctx(*mut u8);

impl Ctx {
    unsafe fn get(self, off: usize) -> u64 {
        self.0.add(off).cast::<u64>().read_unaligned()
    }
    unsafe fn set(self, off: usize, v: u64) {
        self.0.add(off).cast::<u64>().write_unaligned(v)
    }
    unsafe fn gp(self, r: usize) -> u64 {
        self.get(CONTEXT_RAX_OFF + 8 * r)
    }
    unsafe fn set_gp(self, r: usize, v: u64) {
        self.set(CONTEXT_RAX_OFF + 8 * r, v)
    }
    unsafe fn rsp(self) -> u64 {
        self.get(CONTEXT_RSP_OFF)
    }
    unsafe fn set_rsp(self, v: u64) {
        self.set(CONTEXT_RSP_OFF, v)
    }
}

unsafe fn rd64(addr: u64) -> u64 {
    (addr as *const u64).read_unaligned()
}
unsafe fn rd32(addr: usize) -> u32 {
    (addr as *const u32).read_unaligned()
}
unsafe fn rd16(addr: usize) -> u16 {
    (addr as *const u16).read_unaligned()
}
unsafe fn rd8(addr: usize) -> u8 {
    *(addr as *const u8)
}

/// Loads general register `r` from the stack slot at `addr` and records the slot.
unsafe fn load_gp(c: Ctx, ptrs: *mut usize, r: usize, addr: u64) {
    c.set_gp(r, rd64(addr));
    if !ptrs.is_null() {
        *ptrs.add(16 + r) = addr as usize;
    }
}

/// If the code at `pc` is an epilog (`add rsp`/`lea rsp`, pops, then `ret`/`jmp`), runs the rest
/// of it on `c` and returns true; otherwise changes nothing.
unsafe fn unwind_epilog(c: Ctx, ptrs: *mut usize, pc: usize, begin: usize, end: usize) -> bool {
    enum Adj {
        None,
        Add(i64),
        Lea(usize, i64),
    }
    let mut p = pc;
    let mut adj = Adj::None;
    let (b0, b1) = (rd8(p), rd8(p + 1));
    if b0 == 0x48 && b1 == 0x83 && rd8(p + 2) == 0xc4 {
        adj = Adj::Add(rd8(p + 3) as i8 as i64);
        p += 4;
    } else if b0 == 0x48 && b1 == 0x81 && rd8(p + 2) == 0xc4 {
        adj = Adj::Add(rd32(p + 3) as i32 as i64);
        p += 7;
    } else if (b0 == 0x48 || b0 == 0x49) && b1 == 0x8d {
        let m = rd8(p + 2);
        let rm = (m & 7) as usize + if b0 == 0x49 { 8 } else { 0 };
        if m & 0x38 == 0x20 && m & 7 != 4 {
            if m >> 6 == 1 {
                adj = Adj::Lea(rm, rd8(p + 3) as i8 as i64);
                p += 4;
            } else if m >> 6 == 2 {
                adj = Adj::Lea(rm, rd32(p + 3) as i32 as i64);
                p += 7;
            }
        }
    }
    let mut pops = Vec::new();
    loop {
        let b = rd8(p);
        if (0x58..=0x5f).contains(&b) {
            pops.push((b - 0x58) as usize);
            p += 1;
        } else if b == 0x41 && (0x58..=0x5f).contains(&rd8(p + 1)) {
            pops.push((rd8(p + 1) - 0x58) as usize + 8);
            p += 2;
        } else {
            break;
        }
    }
    let jmp_out = |target: i64| target < begin as i64 || target >= end as i64;
    let terminated = match rd8(p) {
        0xc3 => true,
        0xf3 => rd8(p + 1) == 0xc3,
        0xe9 => jmp_out(p as i64 + 5 + rd32(p + 1) as i32 as i64),
        0xeb => jmp_out(p as i64 + 2 + rd8(p + 1) as i8 as i64),
        0x48 => {
            // rex.w jmp [rip+disp32] or jmp [reg]
            let m = rd8(p + 2);
            rd8(p + 1) == 0xff
                && m & 0x38 == 0x20
                && (m == 0x25 || (m >> 6 == 0 && m & 7 != 4 && m & 7 != 5))
        }
        _ => false,
    };
    if !terminated {
        return false;
    }
    match adj {
        Adj::None => {}
        Adj::Add(n) => c.set_rsp(c.rsp().wrapping_add(n as u64)),
        Adj::Lea(r, n) => c.set_rsp(c.gp(r).wrapping_add(n as u64)),
    }
    for r in pops {
        let sp = c.rsp();
        load_gp(c, ptrs, r, sp);
        c.set_rsp(sp + 8);
    }
    let sp = c.rsp();
    c.set(CONTEXT_RIP_OFF, rd64(sp));
    c.set_rsp(sp + 8);
    true
}

/// `RtlVirtualUnwind`: moves `ctx` from the frame of `pc` (a function of `entry`, or a leaf when
/// `entry` is null) to its caller, filling the non-volatile context pointers (32 words) when
/// `ptrs` is not null. On error the context and pointers are left unchanged.
///
/// # Safety
/// `ctx` must be a writable CONTEXT, `base`/`entry` consistent, and the stack and code readable.
pub unsafe fn virtual_unwind(
    handler_type: u32,
    base: usize,
    pc: usize,
    entry: *const RuntimeFunction,
    ctx: *mut u8,
    ptrs: *mut usize,
) -> Result<Unwound, UnwindError> {
    let c = Ctx(ctx);
    let mut out = Unwound {
        handler: 0,
        handler_data: 0,
        establisher: c.rsp() as usize,
    };
    if entry.is_null() {
        c.set(CONTEXT_RIP_OFF, rd64(c.rsp()));
        c.set_rsp(c.rsp() + 8);
        return Ok(out);
    }
    let f = *entry;
    let begin = base + f.begin as usize;
    let offset = pc.wrapping_sub(begin) as u64;
    let first = base + f.unwind as usize;
    let vf = rd8(first);
    if vf & 7 != 1 && vf & 7 != 2 {
        return Err(UnwindError::BadUnwindVersion);
    }
    let prolog = rd8(first + 1) as u64;
    if offset >= prolog && unwind_epilog(c, ptrs, pc, begin, base + f.end as usize) {
        return Ok(out);
    }

    let mut saved = [0u8; CONTEXT_SIZE];
    std::ptr::copy_nonoverlapping(ctx, saved.as_mut_ptr(), CONTEXT_SIZE);
    let mut saved_ptrs = [0usize; 32];
    if !ptrs.is_null() {
        std::ptr::copy_nonoverlapping(ptrs, saved_ptrs.as_mut_ptr(), 32);
    }
    let r = unwind_codes(c, ptrs, base, first, offset, &mut out, handler_type, prolog);
    if r.is_err() {
        std::ptr::copy_nonoverlapping(saved.as_ptr(), ctx, CONTEXT_SIZE);
        if !ptrs.is_null() {
            std::ptr::copy_nonoverlapping(saved_ptrs.as_ptr(), ptrs, 32);
        }
    }
    r.map(|_| out)
}

#[allow(clippy::too_many_arguments)]
unsafe fn unwind_codes(
    c: Ctx,
    ptrs: *mut usize,
    base: usize,
    first: usize,
    offset: u64,
    out: &mut Unwound,
    handler_type: u32,
    prolog: u64,
) -> Result<(), UnwindError> {
    let mut info = first;
    let mut off = offset;
    let mut is_first = true;
    let mut establisher = c.rsp();
    let mut machframe = false;
    loop {
        let v = rd8(info) & 7;
        if v != 1 && v != 2 {
            return Err(UnwindError::BadUnwindVersion);
        }
        let flags = rd8(info) >> 3;
        let count = rd8(info + 2) as usize;
        let fr = rd8(info + 3);
        let (frame_reg, frame_off) = ((fr & 15) as usize, (fr >> 4) as u64 * 16);
        let codes = info + 4;
        if is_first && frame_reg != 0 {
            // The frame register only holds the frame once SET_FPREG has run.
            let mut i = 0;
            while i < count {
                let (o, ob) = (rd8(codes + i * 2), rd8(codes + i * 2 + 1));
                if ob & 15 == 3 && o as u64 <= off {
                    establisher = c.gp(frame_reg).wrapping_sub(frame_off);
                }
                i += op_slots(ob)?;
            }
        }
        let mut i = 0;
        while i < count {
            let (o, ob) = (rd8(codes + i * 2), rd8(codes + i * 2 + 1));
            let (op, inf) = (ob & 15, (ob >> 4) as usize);
            let n = op_slots(ob)?;
            if i + n > count {
                return Err(UnwindError::BadOpcode);
            }
            let slot = |k: usize| rd16(codes + (i + k) * 2) as u64;
            let slot32 = || slot(1) | (slot(2) << 16);
            if o as u64 <= off {
                match op {
                    0 => {
                        let sp = c.rsp();
                        load_gp(c, ptrs, inf, sp);
                        c.set_rsp(sp + 8);
                    }
                    1 => {
                        let size = if inf == 0 { slot(1) * 8 } else { slot32() };
                        c.set_rsp(c.rsp().wrapping_add(size));
                    }
                    2 => c.set_rsp(c.rsp() + inf as u64 * 8 + 8),
                    3 => c.set_rsp(c.gp(frame_reg).wrapping_sub(frame_off)),
                    4 | 5 => {
                        let o = if op == 4 { slot(1) * 8 } else { slot32() };
                        load_gp(c, ptrs, inf, establisher.wrapping_add(o));
                    }
                    6 | 7 => {}
                    8 | 9 => {
                        let o = if op == 8 { slot(1) * 16 } else { slot32() };
                        let addr = establisher.wrapping_add(o);
                        let dst = CONTEXT_XMM0_OFF + 16 * inf;
                        std::ptr::copy_nonoverlapping(addr as *const u8, c.0.add(dst), 16);
                        if !ptrs.is_null() {
                            *ptrs.add(inf) = addr as usize;
                        }
                    }
                    10 => {
                        let mut sp = c.rsp();
                        if inf == 1 {
                            sp += 8;
                        }
                        c.set(CONTEXT_RIP_OFF, rd64(sp));
                        c.set(CONTEXT_SEGCS_OFF, rd64(sp + 8) & 0xffff);
                        c.set(CONTEXT_EFLAGS_OFF, rd64(sp + 16) & 0xffff_ffff);
                        c.set_rsp(rd64(sp + 24));
                        machframe = true;
                    }
                    _ => return Err(UnwindError::BadOpcode),
                }
            }
            i += n;
        }
        if flags & UNW_FLAG_CHAININFO != 0 {
            // The parent's codes have all run by definition.
            let rf = codes + ((count + 1) & !1) * 2;
            info = base + rd32(rf + 8) as usize;
            off = u64::MAX;
            is_first = false;
            continue;
        }
        if flags as u32 & handler_type & 3 != 0 && (offset >= prolog || !is_first) {
            let h = codes + ((count + 1) & !1) * 2;
            out.handler = base + rd32(h) as usize;
            out.handler_data = h + 4;
        }
        break;
    }
    out.establisher = establisher as usize;
    if !machframe {
        let sp = c.rsp();
        c.set(CONTEXT_RIP_OFF, rd64(sp));
        c.set_rsp(sp + 8);
    }
    Ok(())
}

/// Number of 16-bit slots an unwind code takes (itself included).
fn op_slots(ob: u8) -> Result<usize, UnwindError> {
    Ok(match (ob & 15, ob >> 4) {
        (0, _) | (2, _) | (3, _) | (6, _) | (7, _) | (10, _) => 1,
        (1, 0) | (4, _) | (8, _) => 2,
        (1, _) | (5, _) | (9, _) => 3,
        _ => return Err(UnwindError::BadOpcode),
    })
}

// RtlCaptureContext(ctx) and RtlRestoreContext(ctx, record), win64 convention. Both are
// frameless on purpose: the capture sees its caller's registers, with `Rip` the return address
// and `Rsp` the value after return.
std::arch::global_asm!(
    ".text",
    ".globl nativ64_rtl_capture_context",
    ".type nativ64_rtl_capture_context,@function",
    "nativ64_rtl_capture_context:",
    "mov [rcx+0x78], rax",
    "mov [rcx+0x80], rcx",
    "mov [rcx+0x88], rdx",
    "mov [rcx+0x90], rbx",
    "mov [rcx+0xa0], rbp",
    "mov [rcx+0xa8], rsi",
    "mov [rcx+0xb0], rdi",
    "mov [rcx+0xb8], r8",
    "mov [rcx+0xc0], r9",
    "mov [rcx+0xc8], r10",
    "mov [rcx+0xd0], r11",
    "mov [rcx+0xd8], r12",
    "mov [rcx+0xe0], r13",
    "mov [rcx+0xe8], r14",
    "mov [rcx+0xf0], r15",
    "lea rax, [rsp+8]",
    "mov [rcx+0x98], rax",
    "mov rax, [rsp]",
    "mov [rcx+0xf8], rax",
    "pushfq",
    "pop rax",
    "mov [rcx+0x44], eax",
    "mov ax, cs",
    "mov [rcx+0x38], ax",
    "stmxcsr [rcx+0x34]",
    "fxsave [rcx+0x100]",
    "mov dword ptr [rcx+0x30], 0x10000b",
    "ret",
    ".size nativ64_rtl_capture_context, .-nativ64_rtl_capture_context",
    ".globl nativ64_rtl_restore_context",
    ".type nativ64_rtl_restore_context,@function",
    "nativ64_rtl_restore_context:",
    "fxrstor [rcx+0x100]",
    "ldmxcsr [rcx+0x34]",
    "mov rax, [rcx+0x78]",
    "mov rdx, [rcx+0x88]",
    "mov rbx, [rcx+0x90]",
    "mov rbp, [rcx+0xa0]",
    "mov rsi, [rcx+0xa8]",
    "mov rdi, [rcx+0xb0]",
    "mov r8, [rcx+0xb8]",
    "mov r9, [rcx+0xc0]",
    "mov r12, [rcx+0xd8]",
    "mov r13, [rcx+0xe0]",
    "mov r14, [rcx+0xe8]",
    "mov r15, [rcx+0xf0]",
    // Push the target `rip` and flags below the target stack, then `popfq; ret`.
    "mov r11, [rcx+0x98]",
    "mov r10, [rcx+0xf8]",
    "mov [r11-8], r10",
    "mov r10d, [rcx+0x44]",
    "and r10d, 0xcd5",
    "or r10d, 0x202",
    "mov [r11-16], r10",
    "lea rsp, [r11-16]",
    "mov r10, [rcx+0xc8]",
    "mov r11, [rcx+0xd0]",
    "mov rcx, [rcx+0x80]",
    "popfq",
    "ret",
    ".size nativ64_rtl_restore_context, .-nativ64_rtl_restore_context",
);

unsafe extern "win64" {
    pub fn nativ64_rtl_capture_context(ctx: *mut u8);
    pub fn nativ64_rtl_restore_context(ctx: *const u8, record: *const u8) -> !;
}

// ---------------------------------------------------------------------------------------------
// Level 2: exception dispatch, `RaiseException`, `RtlUnwindEx` / `RtlUnwind`, our own
// `__C_specific_handler`.
// ---------------------------------------------------------------------------------------------

pub const STATUS_NONCONTINUABLE_EXCEPTION: u32 = 0xc000_0025;
pub const STATUS_INVALID_DISPOSITION: u32 = 0xc000_0026;
pub const STATUS_UNWIND: u32 = 0xc000_0027;
pub const STATUS_INVALID_UNWIND_TARGET: u32 = 0xc000_0029;

pub const EXCEPTION_NONCONTINUABLE: u32 = 1;
pub const EXCEPTION_UNWINDING: u32 = 2;
pub const EXCEPTION_EXIT_UNWIND: u32 = 4;
pub const EXCEPTION_NESTED_CALL: u32 = 0x10;
pub const EXCEPTION_TARGET_UNWIND: u32 = 0x20;
pub const EXCEPTION_COLLIDED_UNWIND: u32 = 0x40;

/// A walk or a handler chain longer than this is taken for a corrupt stack.
const MAX_FRAMES: usize = 100_000;

/// `EXCEPTION_RECORD`.
#[repr(C)]
pub struct ExceptionRecord {
    pub code: u32,
    pub flags: u32,
    pub record: usize,
    pub address: usize,
    pub nparams: u32,
    pub pad: u32,
    pub info: [usize; 15],
}

impl ExceptionRecord {
    fn new(code: u32, flags: u32, address: usize) -> ExceptionRecord {
        ExceptionRecord {
            code,
            flags,
            record: 0,
            address,
            nparams: 0,
            pad: 0,
            info: [0; 15],
        }
    }
}

/// `EXCEPTION_POINTERS`.
#[repr(C)]
pub struct ExceptionPointers {
    pub record: *mut ExceptionRecord,
    pub context: *mut u8,
}

/// `DISPATCHER_CONTEXT` (0x50 bytes).
#[repr(C)]
pub struct DispatcherContext {
    pub control_pc: usize,
    pub image_base: usize,
    pub function_entry: usize,
    pub establisher_frame: usize,
    pub target_ip: usize,
    pub context_record: usize,
    pub language_handler: usize,
    pub handler_data: usize,
    pub history_table: usize,
    pub scope_index: u32,
    pub fill0: u32,
}

type FrameHandler =
    unsafe extern "win64" fn(*mut ExceptionRecord, usize, *mut u8, *mut DispatcherContext) -> i32;

/// A CONTEXT on the Rust stack.
#[repr(C, align(16))]
#[derive(Clone)]
struct CtxBuf([u8; CONTEXT_SIZE]);

impl CtxBuf {
    unsafe fn from(p: *const u8) -> CtxBuf {
        let mut b = CtxBuf([0; CONTEXT_SIZE]);
        std::ptr::copy_nonoverlapping(p, b.0.as_mut_ptr(), CONTEXT_SIZE);
        b
    }
    fn ptr(&mut self) -> *mut u8 {
        self.0.as_mut_ptr()
    }
}

thread_local! {
    /// Contexts of the exceptions being dispatched on this thread (innermost last). A second
    /// pass that reaches a frame no unwind info describes (our own dispatcher) continues from
    /// the innermost one, as the Windows dispatcher frames would have been unwound.
    static ACTIVE: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn active_top() -> Option<usize> {
    ACTIVE.with(|a| a.borrow().last().copied())
}

/// Tests that install a top-level filter hold this, so that they do not step on each other.
#[cfg(test)]
pub static FILTER_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
thread_local! {
    static STOPPED: std::cell::RefCell<Vec<(u32, usize)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The `(code, address)` of every fatal exception reported on this thread (test builds only:
/// they record instead of ending the process).
#[cfg(test)]
pub fn take_stopped() -> Vec<(u32, usize)> {
    STOPPED.with(|s| std::mem::take(&mut *s.borrow_mut()))
}

/// A reporter for unhandled exceptions (the loader's, which names `image+0xOFF` and DLLs). It
/// gets the record (`code`, `address`, `info`) and should end the process; if it returns, the
/// default report runs.
pub type FatalHook = fn(*const ExceptionRecord);

static FATAL_HOOK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Installs the unhandled-exception reporter.
pub fn set_fatal_hook(hook: FatalHook) {
    FATAL_HOOK.store(hook as usize, std::sync::atomic::Ordering::Release);
}

/// `NATIV64_TRACE_FAULTS`: when set (by the loader), the epoch that trace lines count from.
static FAULT_TRACE: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

/// How many times each `(code, rip, outcome, place)` was seen, for rate-limiting.
static FAULT_SEEN: Mutex<Vec<(FaultKey, u64)>> = Mutex::new(Vec::new());

/// Distinct keys kept; past it, new keys always print (the table stays bounded).
const FAULT_KEYS_MAX: usize = 4096;

type FaultKey = (u32, usize, u8, usize);

/// Outcome of a traced fault: who took it and where it went.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FaultOutcome {
    /// A vectored handler (address) returned `EXCEPTION_CONTINUE_EXECUTION`.
    Vectored(usize),
    /// A frame handler (address, frame pc) returned `ExceptionContinueExecution`.
    Continued(usize, usize),
    /// A frame handler (address, frame pc) took it by unwinding to `target`.
    Unwound {
        handler: usize,
        frame: usize,
        target: usize,
    },
    /// Nobody took it.
    Unhandled,
}

/// One fault being dispatched on this thread: record, time, and the last frame handler called.
#[derive(Clone, Copy)]
struct Traced {
    rec: usize,
    code: u32,
    rip: usize,
    addr: usize,
    secs: f32,
    handler: usize,
    frame: usize,
}

thread_local! {
    /// Faults being dispatched on this thread (innermost last), with `NATIV64_TRACE_FAULTS`.
    static TRACED: std::cell::RefCell<Vec<Traced>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Turns on `NATIV64_TRACE_FAULTS`: from now on, every hardware fault dispatched to the title's
/// handlers prints `nativ64: fault [TID] Ts code C rip LOC addr A -> OUTCOME` (identical lines
/// rate-limited). Time counts from this call.
pub fn set_fault_trace() {
    let _ = FAULT_TRACE.set(std::time::Instant::now());
}

fn fault_trace_on() -> bool {
    FAULT_TRACE.get().is_some()
}

/// `image+0xOFF` (or the bare address outside every image).
fn loc(a: usize) -> String {
    match crate::process::image_base_of(a) {
        Some(b) => format!("image+{:#x}", a - b),
        None => format!("{a:#x}"),
    }
}

/// Whether the `n`-th identical line (from 1) prints: the first 3, then each power of two.
pub fn fault_line_due(n: u64) -> bool {
    n <= 3 || n.is_power_of_two()
}

/// The trace line of one fault (`n` = how many identical ones so far).
#[allow(clippy::too_many_arguments)]
pub fn fault_line(
    tid: u32,
    secs: f32,
    code: u32,
    rip: &str,
    addr: usize,
    outcome: &str,
    n: u64,
) -> String {
    let mut l = format!(
        "nativ64: fault [{tid}] {secs:.3}s code {code:#x} rip {rip} addr {addr:#x} -> {outcome}"
    );
    if n > 1 {
        l.push_str(&format!(" (x{n})"));
    }
    l
}

fn outcome_text(o: FaultOutcome) -> String {
    match o {
        FaultOutcome::Vectored(h) => format!("continue (vectored {})", loc(h)),
        FaultOutcome::Continued(h, f) => {
            format!("continue (handler {} frame {})", loc(h), loc(f))
        }
        FaultOutcome::Unwound {
            handler,
            frame,
            target,
        } => format!(
            "caught: unwind to {} (handler {} frame {})",
            loc(target),
            loc(handler),
            loc(frame)
        ),
        FaultOutcome::Unhandled => "unhandled".to_string(),
    }
}

/// Prints the line of `t` ending in `o`, unless rate-limited.
fn trace_fault(t: &Traced, o: FaultOutcome) {
    let (kind, place) = match o {
        FaultOutcome::Vectored(h) => (0, h),
        FaultOutcome::Continued(h, f) => (1, h ^ f.rotate_left(32)),
        FaultOutcome::Unwound { target, .. } => (2, target),
        FaultOutcome::Unhandled => (3, 0),
    };
    let key = (t.code, t.rip, kind, place);
    let n = {
        let mut seen = FAULT_SEEN.lock().unwrap_or_else(|e| e.into_inner());
        match seen.iter_mut().find(|e| e.0 == key) {
            Some(e) => {
                e.1 += 1;
                e.1
            }
            None => {
                if seen.len() < FAULT_KEYS_MAX {
                    seen.push((key, 1));
                }
                1
            }
        }
    };
    if fault_line_due(n) {
        let tid = unsafe { libc::syscall(libc::SYS_gettid) as u32 };
        eprintln!(
            "{}",
            fault_line(
                tid,
                t.secs,
                t.code,
                &loc(t.rip),
                t.addr,
                &outcome_text(o),
                n
            )
        );
    }
}

/// Records the frame handler about to see the traced fault `rec`.
fn trace_handler(rec: *mut ExceptionRecord, handler: usize, frame: usize) {
    TRACED.with(|t| {
        if let Some(e) = t.borrow_mut().last_mut() {
            if e.rec == rec as usize {
                e.handler = handler;
                e.frame = frame;
            }
        }
    });
}

/// An unwind to `target` (stack pointer `sp`) while a traced fault is being dispatched: the
/// innermost one was caught; the dispatches below `sp` are abandoned.
fn trace_unwind(target: usize, sp: usize) {
    let top = TRACED.with(|t| {
        let mut t = t.borrow_mut();
        let top = t.last().copied();
        t.retain(|e| e.rec >= sp);
        top
    });
    if let Some(e) = top {
        trace_fault(
            &e,
            FaultOutcome::Unwound {
                handler: e.handler,
                frame: e.frame,
                target,
            },
        );
    }
}

/// Entry for a hardware fault turned into an exception: the loader's signal handler copies the
/// `CONTEXT` and `EXCEPTION_RECORD` onto the faulting thread's stack, then resumes the thread
/// here (as if called, so `rsp` is 8 mod 16). Dispatches like `RaiseException`; resumes the
/// (possibly changed) context when a handler continued, otherwise reports and ends the process.
///
/// # Safety
/// `rec` and `ctx` must be a valid record and a 16-aligned `CONTEXT` that stay readable and
/// writable for the whole dispatch (they live on the guest stack below the fault).
pub unsafe extern "win64" fn dispatch_fault(rec: *mut ExceptionRecord, ctx: *mut u8) -> ! {
    let traced = FAULT_TRACE.get().map(|t0| Traced {
        rec: rec as usize,
        code: (*rec).code,
        rip: (*rec).address,
        addr: (*rec).info[1],
        secs: t0.elapsed().as_secs_f32(),
        handler: 0,
        frame: 0,
    });
    if let Some(t) = traced {
        TRACED.with(|v| v.borrow_mut().push(t));
    }
    let handled = search(rec, ctx);
    if traced.is_some() {
        if let Some(e) = TRACED.with(|v| v.borrow_mut().pop()) {
            let o = if !handled {
                FaultOutcome::Unhandled
            } else if e.frame == 0 {
                FaultOutcome::Vectored(e.handler)
            } else {
                FaultOutcome::Continued(e.handler, e.frame)
            };
            trace_fault(&e, o);
        }
    }
    if !handled {
        unhandled(rec, ctx);
    }
    nativ64_rtl_restore_context(ctx, std::ptr::null())
}

/// Reports a fatal exception (`fatal exception 0x… at image+0x…`) and ends the process.
fn stop(code: u32, address: usize) {
    #[cfg(test)]
    {
        STOPPED.with(|s| s.borrow_mut().push((code, address)));
    }
    #[cfg(not(test))]
    {
        let hook = FATAL_HOOK.load(std::sync::atomic::Ordering::Acquire);
        if hook != 0 {
            let rec = ExceptionRecord::new(code, 0, address);
            let hook: FatalHook = unsafe { std::mem::transmute(hook) };
            hook(&rec);
        }
        match crate::process::image_base_of(address) {
            Some(b) => eprintln!(
                "nativ64: fatal exception {code:#x} at image {b:#x}+{:#x}",
                address - b
            ),
            None => eprintln!("nativ64: fatal exception {code:#x} at {address:#x}"),
        }
        std::process::exit(1);
    }
}

/// One step of a walk.
struct Frame {
    pc: usize,
    base: usize,
    entry: usize,
    u: Unwound,
}

/// Unwinds `c` to the caller of its current frame. `None` outside every image, on malformed
/// unwind info, an unreadable stack or a step that does not climb the stack.
unsafe fn step(c: *mut u8, handler_type: u32) -> Option<Frame> {
    let pc = Ctx(c).get(CONTEXT_RIP_OFF) as usize;
    let rsp = Ctx(c).rsp();
    let (base, entry) = lookup_function_entry(pc);
    if base == 0 || (entry == 0 && !mapped(rsp as usize, 8)) {
        return None;
    }
    let u = virtual_unwind(
        handler_type,
        base,
        pc,
        entry as *const RuntimeFunction,
        c,
        std::ptr::null_mut(),
    )
    .ok()?;
    if Ctx(c).rsp() <= rsp {
        return None;
    }
    Some(Frame { pc, base, entry, u })
}

fn dispatcher(f: &Frame, target_ip: usize, context: *mut u8) -> DispatcherContext {
    DispatcherContext {
        control_pc: f.pc,
        image_base: f.base,
        function_entry: f.entry,
        establisher_frame: f.u.establisher,
        target_ip,
        context_record: context as usize,
        language_handler: f.u.handler,
        handler_data: f.u.handler_data,
        history_table: 0,
        scope_index: 0,
        fill0: 0,
    }
}

/// First pass: vectored handlers, then the frames' exception handlers. True when one continued
/// execution; false when nobody took the exception (a handler that takes it by unwinding never
/// returns).
unsafe fn search(rec: *mut ExceptionRecord, ctx: *mut u8) -> bool {
    if crate::log::on(crate::log::SEH) {
        log_exception(rec, ctx);
    }
    ACTIVE.with(|a| a.borrow_mut().push(ctx as usize));
    let r = search_frames(rec, ctx);
    ACTIVE.with(|a| {
        a.borrow_mut().pop();
    });
    r
}

/// `NATIV64_LOG=seh`: one line per exception as its dispatch starts: its code, where it was
/// raised, the faulting instruction and the record's two information words. Software
/// exceptions show too, which `NATIV64_TRACE_FAULTS` does not see.
fn log_exception(rec: *const ExceptionRecord, ctx: *const u8) {
    use crate::log::At;
    // SAFETY: the dispatcher only passes records it built.
    let Some(r) = (unsafe { rec.as_ref() }) else {
        return;
    };
    let rip = if ctx.is_null() {
        0
    } else {
        // SAFETY: a non-NULL context is a valid CONTEXT (see `dispatch_fault`).
        unsafe { Ctx(ctx as *mut u8).get(CONTEXT_RIP_OFF) as usize }
    };
    eprintln!(
        "nativ64: exception {:#x} at {} rip {} info {:#x} {:#x}",
        r.code,
        At(r.address),
        At(rip),
        r.info[0],
        r.info[1]
    );
}

unsafe fn search_frames(rec: *mut ExceptionRecord, ctx: *mut u8) -> bool {
    let mut ptrs = ExceptionPointers {
        record: rec,
        context: ctx,
    };
    for h in crate::api_ms_win_core_errorhandling::vectored_handlers() {
        let f: unsafe extern "win64" fn(*mut ExceptionPointers) -> i32 = std::mem::transmute(h);
        if fault_trace_on() {
            trace_handler(rec, h, 0);
        }
        if f(&mut ptrs) == -1 {
            return resume(rec, ctx);
        }
    }
    let mut c = CtxBuf::from(ctx);
    for i in 0..MAX_FRAMES {
        let mut frame_ctx = c.clone();
        let Some(f) = step(c.ptr(), UNW_FLAG_EHANDLER) else {
            break;
        };
        if f.u.handler != 0 {
            // The first frame gets the exception context itself, so a handler can change it.
            let fc = if i == 0 { ctx } else { frame_ctx.ptr() };
            let mut d = dispatcher(&f, 0, fc);
            if fault_trace_on() {
                trace_handler(rec, f.u.handler, f.pc);
            }
            let h: FrameHandler = std::mem::transmute(f.u.handler);
            match h(rec, f.u.establisher, fc, &mut d) {
                0 => return resume(rec, ctx),
                1 => {}
                // The handler raised and survived an exception of its own: keep searching,
                // marked (the nested record chain is not followed).
                2 => (*rec).flags |= EXCEPTION_NESTED_CALL,
                _ => return escalate(rec, ctx, STATUS_INVALID_DISPOSITION),
            }
        }
    }
    false
}

/// A handler continues execution: refused for a non-continuable exception.
unsafe fn resume(rec: *mut ExceptionRecord, ctx: *mut u8) -> bool {
    if (*rec).flags & EXCEPTION_NONCONTINUABLE == 0 {
        return true;
    }
    escalate(rec, ctx, STATUS_NONCONTINUABLE_EXCEPTION)
}

/// Raises `code` (non-continuable) about `old`, from the same context; always true (execution
/// goes on only if the new exception was handled, or ends).
unsafe fn escalate(old: *mut ExceptionRecord, ctx: *mut u8, code: u32) -> bool {
    if (*old).code == code {
        // The escalation itself was continued: give up.
        stop(code, (*old).address);
        return true;
    }
    let mut new = ExceptionRecord::new(code, EXCEPTION_NONCONTINUABLE, (*old).address);
    new.record = old as usize;
    if !search(&mut new, ctx) {
        unhandled(&mut new, ctx);
    }
    true
}

/// Nobody handled `rec`: the top-level filter, then report and stop.
unsafe fn unhandled(rec: *mut ExceptionRecord, ctx: *mut u8) {
    let ptrs = ExceptionPointers {
        record: rec,
        context: ctx,
    };
    if filter_verdict(&ptrs) != -1 {
        #[cfg(not(test))]
        eprintln!("{}", backtrace_line(ctx, 16));
        stop((*rec).code, (*rec).address);
    }
}

/// The return addresses above the faulting frame of `ctx`, up to `max`, as
/// `nativ64: backtrace image+0x… image+0x…` (addresses outside every image print bare). Stops
/// at the first frame no unwind info describes.
unsafe fn backtrace_line(ctx: *mut u8, max: usize) -> String {
    let mut line = String::from("nativ64: backtrace");
    if ctx.is_null() {
        return line;
    }
    let mut buf = CtxBuf::from(ctx);
    for _ in 0..max {
        if step(buf.ptr(), 0).is_none() {
            break;
        }
        let ret = Ctx(buf.ptr()).get(CONTEXT_RIP_OFF) as usize;
        match crate::process::image_base_of(ret) {
            Some(b) => line.push_str(&format!(" image+{:#x}", ret - b)),
            None => line.push_str(&format!(" {ret:#x}")),
        }
    }
    line
}

/// What the title's top-level filter says: `-1` continue, `1` execute handler (terminate), `0`
/// when there is none or it declines.
unsafe fn filter_verdict(info: *const ExceptionPointers) -> i32 {
    let f = crate::api_ms_win_core_errorhandling::top_level_filter();
    if f == 0 {
        return 0;
    }
    let f: unsafe extern "win64" fn(*const ExceptionPointers) -> i32 = std::mem::transmute(f);
    match f(info) {
        -1 => -1,
        1 => 1,
        _ => 0,
    }
}

/// `UnhandledExceptionFilter`: returns the top-level filter's verdict (`-1` continue, `1`
/// execute handler); without a filter, or when it declines, reports the exception, stops, and
/// returns `1`.
///
/// # Safety
/// `info` must be a valid `EXCEPTION_POINTERS`.
pub unsafe fn unhandled_exception_filter(info: *const ExceptionPointers) -> i32 {
    let v = filter_verdict(info);
    if v == 0 {
        let rec = (*info).record;
        stop((*rec).code, (*rec).address);
        return 1;
    }
    v
}

/// The body of `RaiseException`, called with the caller's captured context.
unsafe extern "win64" fn raise_exception_impl(
    code: u32,
    flags: u32,
    nargs: u32,
    args: *const usize,
    ctx: *mut u8,
    _caller_stack: *const usize,
) {
    let mut rec = ExceptionRecord::new(code, flags, Ctx(ctx).get(CONTEXT_RIP_OFF) as usize);
    if !args.is_null() {
        let n = nargs.min(15) as usize;
        rec.nparams = n as u32;
        std::ptr::copy_nonoverlapping(args, rec.info.as_mut_ptr(), n);
    }
    if search(&mut rec, ctx) {
        return;
    }
    // The thread-name request is for a debugger: it is served only when no handler took it.
    if code == crate::api_ms_win_core_errorhandling::THREAD_NAME_CODE
        && crate::api_ms_win_core_errorhandling::name_thread(nargs, args)
    {
        return;
    }
    unhandled(&mut rec, ctx);
}

/// `RtlUnwindEx`'s second pass. Walks from `start` (the caller's context), calling the unwind
/// handlers, until the frame whose establisher is `target_frame` (`0`: every frame), then
/// resumes there at `target_ip` with `rax = retval`. `bridged`: `start` is already the
/// exception context (the caller is our own dispatcher). Returns only on error.
///
/// # Safety
/// `start` is a CONTEXT; `rec_in` null or an `EXCEPTION_RECORD`.
pub unsafe fn unwind_ex(
    target_frame: usize,
    target_ip: usize,
    rec_in: *mut ExceptionRecord,
    retval: usize,
    start: *const u8,
    mut bridged: bool,
) {
    let mut local = ExceptionRecord::new(STATUS_UNWIND, 0, 0);
    let rec: *mut ExceptionRecord = if rec_in.is_null() {
        local.address = Ctx(start as *mut u8).get(CONTEXT_RIP_OFF) as usize;
        &mut local
    } else {
        rec_in
    };
    (*rec).flags |= EXCEPTION_UNWINDING;
    if target_frame == 0 {
        (*rec).flags |= EXCEPTION_EXIT_UNWIND;
    }
    let mut c = CtxBuf::from(start);
    for _ in 0..MAX_FRAMES {
        let mut frame = c.clone();
        let Some(f) = step(c.ptr(), UNW_FLAG_UHANDLER) else {
            if !bridged {
                if let Some(p) = active_top() {
                    c = CtxBuf::from(p as *const u8);
                    bridged = true;
                    continue;
                }
            }
            break;
        };
        if target_frame != 0 && f.u.establisher > target_frame {
            break;
        }
        let is_target = target_frame != 0 && f.u.establisher == target_frame;
        if f.u.handler != 0 {
            let mut flags = (*rec).flags & !EXCEPTION_TARGET_UNWIND;
            if is_target {
                flags |= EXCEPTION_TARGET_UNWIND;
            }
            (*rec).flags = flags;
            let mut d = dispatcher(&f, target_ip, frame.ptr());
            let h: FrameHandler = std::mem::transmute(f.u.handler);
            match h(rec, f.u.establisher, frame.ptr(), &mut d) {
                1 => {}
                3 => {
                    // Collided unwind: the handler unwound past this frame itself and left
                    // the context to continue from.
                    (*rec).flags |= EXCEPTION_COLLIDED_UNWIND;
                    c = frame.clone();
                    if step(c.ptr(), 0).is_none() {
                        break;
                    }
                    continue;
                }
                _ => {
                    stop(STATUS_INVALID_DISPOSITION, f.pc);
                    return;
                }
            }
        }
        if is_target {
            let fc = Ctx(frame.ptr());
            fc.set_gp(0, retval as u64);
            if target_ip != 0 {
                fc.set(CONTEXT_RIP_OFF, target_ip as u64);
            }
            // The dispatches this unwind abandons are those whose context is below the target.
            let sp = fc.rsp() as usize;
            ACTIVE.with(|a| a.borrow_mut().retain(|&p| p >= sp));
            if fault_trace_on() {
                trace_unwind(fc.get(CONTEXT_RIP_OFF) as usize, sp);
            }
            nativ64_rtl_restore_context(frame.ptr(), std::ptr::null());
        }
    }
    if target_frame == 0 && target_ip != 0 {
        // Exit unwind: every frame is gone, resume at the target with the last context.
        let fc = Ctx(c.ptr());
        fc.set_gp(0, retval as u64);
        fc.set(CONTEXT_RIP_OFF, target_ip as u64);
        ACTIVE.with(|a| a.borrow_mut().clear());
        if fault_trace_on() {
            trace_unwind(target_ip, usize::MAX);
        }
        nativ64_rtl_restore_context(c.ptr(), std::ptr::null());
    }
    stop(STATUS_INVALID_UNWIND_TARGET, (*rec).address);
}

/// The bodies of `RtlUnwindEx` / `RtlUnwind`, called with the caller's captured context.
unsafe extern "win64" fn rtl_unwind_ex_impl(
    target_frame: usize,
    target_ip: usize,
    record: *mut ExceptionRecord,
    retval: usize,
    ctx: *mut u8,
    _caller_stack: *const usize,
) {
    unwind_ex(target_frame, target_ip, record, retval, ctx, false);
}

/// `__C_specific_handler` (the frame handler of the MSVC `__try` family; format in
/// `docs/seh.md`).
///
/// # Safety
/// The arguments are those of a frame handler: record, establisher frame, context and
/// dispatcher context with this function's language data.
pub unsafe fn c_specific_handler(
    rec: *mut ExceptionRecord,
    frame: usize,
    ctx: *mut u8,
    disp: *mut DispatcherContext,
) -> i32 {
    let d = &mut *disp;
    let base = d.image_base;
    let data = d.handler_data as *const u8;
    let count = data.cast::<u32>().read_unaligned() as usize;
    let scope = |i: usize| {
        let p = data.add(4 + 16 * i).cast::<u32>();
        (
            p.read_unaligned(),
            p.add(1).read_unaligned(),
            p.add(2).read_unaligned(),
            p.add(3).read_unaligned(),
        )
    };
    let rva = d.control_pc.wrapping_sub(base) as u32;
    let flags = (*rec).flags;
    if flags & EXCEPTION_UNWINDING != 0 {
        let target_rva = d.target_ip.wrapping_sub(base) as u32;
        for i in d.scope_index as usize..count {
            let (b, e, h, t) = scope(i);
            if rva < b || rva >= e || t != 0 {
                continue;
            }
            // The frame that stays: a __finally around the landing pad does not run.
            if flags & EXCEPTION_TARGET_UNWIND != 0 && target_rva >= b && target_rva < e {
                continue;
            }
            d.scope_index = i as u32 + 1;
            let fin: unsafe extern "win64" fn(u8, usize) = std::mem::transmute(base + h as usize);
            fin(1, frame);
        }
        return 1;
    }
    for i in d.scope_index as usize..count {
        let (b, e, h, t) = scope(i);
        if rva < b || rva >= e || t == 0 {
            continue;
        }
        if h != 1 {
            let mut ptrs = ExceptionPointers {
                record: rec,
                context: ctx,
            };
            let filter: unsafe extern "win64" fn(*mut ExceptionPointers, usize) -> i32 =
                std::mem::transmute(base + h as usize);
            let r = filter(&mut ptrs, frame);
            if r < 0 {
                return 0;
            }
            if r == 0 {
                continue;
            }
        }
        d.scope_index = i as u32;
        let start = active_top().unwrap_or(d.context_record) as *const u8;
        unwind_ex(frame, base + t as usize, rec, 0, start, true);
        return 1;
    }
    1
}

// The entries that capture the caller's context before running Rust: `RaiseException`,
// `RtlUnwindEx`, `RtlUnwind`. Frame: 0x20 of outgoing home space, the 5th and 6th arguments
// (the captured context and the address of the caller's stack arguments), the context at
// rsp+0x30. They resume through `RtlRestoreContext` (RaiseException continues after its call,
// the unwinds only get there on error).
macro_rules! context_entry {
    ($name:literal, $imp:path) => {
        std::arch::global_asm!(
            ".text",
            concat!(".globl ", $name),
            concat!(".type ", $name, ",@function"),
            concat!($name, ":"),
            "sub rsp, 0x508",
            "lea r10, [rsp+0x30]",
            "mov [r10+0x78], rax",
            "mov [r10+0x80], rcx",
            "mov [r10+0x88], rdx",
            "mov [r10+0x90], rbx",
            "mov [r10+0xa0], rbp",
            "mov [r10+0xa8], rsi",
            "mov [r10+0xb0], rdi",
            "mov [r10+0xb8], r8",
            "mov [r10+0xc0], r9",
            "mov [r10+0xd8], r12",
            "mov [r10+0xe0], r13",
            "mov [r10+0xe8], r14",
            "mov [r10+0xf0], r15",
            "lea rax, [rsp+0x510]",
            "mov [r10+0x98], rax",
            "mov rax, [rsp+0x508]",
            "mov [r10+0xf8], rax",
            "pushfq",
            "pop rax",
            "mov [r10+0x44], eax",
            "mov ax, cs",
            "mov [r10+0x38], ax",
            "stmxcsr [r10+0x34]",
            "fxsave [r10+0x100]",
            "mov dword ptr [r10+0x30], 0x10000b",
            "mov [rsp+0x20], r10",
            "lea rax, [rsp+0x530]",
            "mov [rsp+0x28], rax",
            "call {imp}",
            "lea rcx, [rsp+0x30]",
            "xor edx, edx",
            "call nativ64_rtl_restore_context",
            concat!(".size ", $name, ", .-", $name),
            imp = sym $imp,
        );
    };
}

context_entry!("nativ64_raise_exception", raise_exception_impl);
context_entry!("nativ64_rtl_unwind_ex", rtl_unwind_ex_impl);
context_entry!("nativ64_rtl_unwind", rtl_unwind_ex_impl);

unsafe extern "win64" {
    /// `RaiseException(code, flags, nargs, args)`.
    pub fn nativ64_raise_exception(code: u32, flags: u32, nargs: u32, args: *const usize);
    /// `RtlUnwindEx(target_frame, target_ip, record, retval, context, history)`.
    pub fn nativ64_rtl_unwind_ex(
        target_frame: usize,
        target_ip: usize,
        record: *mut ExceptionRecord,
        retval: usize,
        context: *mut u8,
        history: usize,
    );
    /// `RtlUnwind(target_frame, target_ip, record, retval)`.
    pub fn nativ64_rtl_unwind(
        target_frame: usize,
        target_ip: usize,
        record: *mut ExceptionRecord,
        retval: usize,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fault_lines_are_rate_limited_and_shaped() {
        let due: Vec<u64> = (1..=20).filter(|&n| fault_line_due(n)).collect();
        assert_eq!(due, [1, 2, 3, 4, 8, 16]);
        assert_eq!(
            fault_line(7, 1.5, 0xc000_0005, "image+0x10", 0x20, "unhandled", 1),
            "nativ64: fault [7] 1.500s code 0xc0000005 rip image+0x10 addr 0x20 -> unhandled"
        );
        assert!(fault_line(7, 1.5, 5, "x", 0, "y", 8).ends_with(" (x8)"));
    }
    use crate::api_ms_win_core_rtlsupport::{
        RtlAddFunctionTable, RtlDeleteFunctionTable, RtlLookupFunctionEntry, RtlVirtualUnwind,
    };

    const RAX: usize = 0;
    const RBX: usize = 3;
    const RSP: usize = 4;
    const RBP: usize = 5;
    const RSI: usize = 6;
    const RDI: usize = 7;

    #[repr(C, align(16))]
    struct Context([u8; CONTEXT_SIZE]);

    impl Context {
        fn new() -> Box<Context> {
            Box::new(Context([0; CONTEXT_SIZE]))
        }
        fn get(&self, off: usize) -> u64 {
            u64::from_le_bytes(self.0[off..off + 8].try_into().unwrap())
        }
        fn set(&mut self, off: usize, v: u64) {
            self.0[off..off + 8].copy_from_slice(&v.to_le_bytes());
        }
        fn gp(&self, r: usize) -> u64 {
            self.get(CONTEXT_RAX_OFF + 8 * r)
        }
        fn set_gp(&mut self, r: usize, v: u64) {
            self.set(CONTEXT_RAX_OFF + 8 * r, v)
        }
        fn rip(&self) -> u64 {
            self.get(CONTEXT_RIP_OFF)
        }
        fn xmm(&self, i: usize) -> [u8; 16] {
            let o = CONTEXT_XMM0_OFF + 16 * i;
            self.0[o..o + 16].try_into().unwrap()
        }
    }

    /// 16-aligned qword stack addressed by index.
    struct Stack(Vec<u64>);

    impl Stack {
        fn new(n: usize) -> Stack {
            Stack(vec![0; n + 2])
        }
        fn p(&self, i: usize) -> u64 {
            ((self.0.as_ptr() as u64 + 15) & !15) + 8 * i as u64
        }
        fn set(&self, i: usize, v: u64) {
            unsafe { *(self.p(i) as *mut u64) = v }
        }
    }

    /// Executable scratch memory: code at 0, unwind info at 0x400, runtime functions at 0x800.
    struct Mem(usize);

    impl Mem {
        fn new() -> Mem {
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    0x10000,
                    libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert_ne!(p, libc::MAP_FAILED);
            Mem(p as usize)
        }
        fn put(&self, off: usize, bytes: &[u8]) {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    (self.0 + off) as *mut u8,
                    bytes.len(),
                )
            }
        }
        fn put32(&self, off: usize, v: u32) {
            self.put(off, &v.to_le_bytes());
        }
        /// Runtime function `i` of the table at 0x800.
        fn rf(&self, i: usize, begin: u32, end: u32, info: u32) {
            self.put32(0x800 + 12 * i, begin);
            self.put32(0x804 + 12 * i, end);
            self.put32(0x808 + 12 * i, info);
        }
        fn register(&self, count: u32) {
            assert_eq!(
                unsafe { RtlAddFunctionTable(self.0 + 0x800, count, self.0) },
                1
            );
        }
        /// One function at 0 with its unwind info at 0x400, registered.
        fn single(code: &[u8], info: &[u8]) -> Mem {
            let m = Mem::new();
            m.put(0, code);
            m.put(0x400, info);
            m.rf(0, 0, code.len() as u32, 0x400);
            m.register(1);
            m
        }
    }

    impl Drop for Mem {
        fn drop(&mut self) {
            unsafe {
                RtlDeleteFunctionTable(self.0 + 0x800);
                libc::munmap(self.0 as *mut libc::c_void, 0x10000);
            }
        }
    }

    struct Step {
        handler: usize,
        data: usize,
        frame: usize,
        ptrs: [usize; 32],
    }

    unsafe fn unwind(pc: usize, ctx: &mut Context, ty: u32) -> Step {
        let mut base = 0usize;
        let e = RtlLookupFunctionEntry(pc, &mut base, 0);
        let (mut data, mut frame) = (0usize, 0usize);
        let mut ptrs = [0usize; 32];
        let handler = RtlVirtualUnwind(
            ty,
            base,
            pc,
            e as *const RuntimeFunction,
            ctx.0.as_mut_ptr(),
            &mut data,
            &mut frame,
            ptrs.as_mut_ptr(),
        );
        Step {
            handler,
            data,
            frame,
            ptrs,
        }
    }

    /// push rbp; push rbx; sub rsp,0x28; body (6..10); add rsp,0x28; pop rbx; pop rbp; ret.
    fn standard(info_head: u8, tail: &[u8]) -> Mem {
        let mut code = vec![0x55, 0x53, 0x48, 0x83, 0xec, 0x28, 0x90, 0x90, 0x90, 0x90];
        code.extend_from_slice(&[0x48, 0x83, 0xc4, 0x28, 0x5b, 0x5d, 0xc3]);
        let mut info = vec![info_head, 6, 3, 0, 6, 0x42, 2, 0x30, 1, 0x50, 0, 0];
        info.extend_from_slice(tail);
        Mem::single(&code, &info)
    }

    /// Stack of the standard frame: 5 locals, saved rbx, saved rbp, return address.
    fn standard_stack() -> Stack {
        let s = Stack::new(24);
        s.set(5, 0xb3);
        s.set(6, 0xb5);
        s.set(7, 0xdead);
        s
    }

    #[test]
    fn exports_registered() {
        for n in [
            "RtlLookupFunctionEntry",
            "RtlVirtualUnwind",
            "RtlCaptureContext",
            "RtlRestoreContext",
            "RtlAddFunctionTable",
            "RtlDeleteFunctionTable",
        ] {
            assert!(
                crate::lookup("api-ms-win-core-rtlsupport-l1-1-0.dll", n).is_some(),
                "{n}"
            );
            assert!(crate::lookup("kernel32.dll", n).is_some(), "{n}");
        }
    }

    #[test]
    fn lookup_hits_and_misses() {
        let m = Mem::new();
        m.rf(0, 0x100, 0x110, 0x400);
        m.rf(1, 0x120, 0x130, 0x400);
        m.register(2);
        let at = |rva: usize| unsafe {
            let mut base = 7usize;
            let e = RtlLookupFunctionEntry(m.0 + rva, &mut base, 0);
            (e, base)
        };
        assert_eq!(at(0x100), (m.0 + 0x800, m.0));
        assert_eq!(at(0x10f), (m.0 + 0x800, m.0));
        assert_eq!(at(0x120), (m.0 + 0x80c, m.0));
        assert_eq!(at(0x12f), (m.0 + 0x80c, m.0));
        assert_eq!(at(0x110).0, 0);
        assert_eq!(at(0x118), (0, 0));
        assert_eq!(at(0x130).0, 0);
        assert_eq!(at(0xff).0, 0);
        assert_eq!(
            unsafe { RtlLookupFunctionEntry(0x10, std::ptr::null_mut(), 0) },
            0
        );
        // duplicate and invalid tables
        assert_eq!(unsafe { RtlAddFunctionTable(m.0 + 0x800, 2, m.0) }, 0);
        assert_eq!(unsafe { RtlAddFunctionTable(0, 2, m.0) }, 0);
        assert_eq!(unsafe { RtlAddFunctionTable(m.0 + 0x900, 0, m.0) }, 0);
        // deleting removes it, once
        assert_eq!(unsafe { RtlDeleteFunctionTable(m.0 + 0x800) }, 1);
        assert_eq!(unsafe { RtlDeleteFunctionTable(m.0 + 0x800) }, 0);
        assert_eq!(at(0x100).0, 0);
    }

    #[test]
    fn lookup_in_a_pe_image() {
        let m = Mem::new();
        // minimal PE32+ header whose exception directory holds two runtime functions
        m.put(0, &[0x4d, 0x5a]);
        m.put32(0x3c, 0x80);
        m.put32(0x80, 0x4550);
        m.put(0x98, &[0x0b, 0x02]);
        m.put32(0x98 + 0x6c, 16);
        m.put32(0x98 + 0x70 + 24, 0x1000);
        m.put32(0x98 + 0x74 + 24, 24);
        for (i, v) in [0x2000u32, 0x2010, 0x3000, 0x2020, 0x2030, 0x3000]
            .into_iter()
            .enumerate()
        {
            m.put32(0x1000 + 4 * i, v);
        }
        crate::process::register_image_range(m.0, 0x10000);
        unsafe {
            let mut base = 0usize;
            assert_eq!(
                RtlLookupFunctionEntry(m.0 + 0x2005, &mut base, 0),
                m.0 + 0x1000
            );
            assert_eq!(base, m.0);
            assert_eq!(
                RtlLookupFunctionEntry(m.0 + 0x202f, &mut base, 0),
                m.0 + 0x100c
            );
            // a miss inside the image still reports its base
            base = 0;
            assert_eq!(RtlLookupFunctionEntry(m.0 + 0x2015, &mut base, 0), 0);
            assert_eq!(base, m.0);
        }
        // registering the range again without a directory drops the table
        m.put32(0x98 + 0x74 + 24, 0);
        crate::process::register_image_range(m.0, 0x10000);
        unsafe {
            assert_eq!(RtlLookupFunctionEntry(m.0 + 0x2005, &mut 0, 0), 0);
        }
    }

    #[test]
    fn standard_frame_body() {
        let m = standard(1, &[]);
        let s = standard_stack();
        let mut c = Context::new();
        c.set_gp(RSP, s.p(0));
        let st = unsafe { unwind(m.0 + 6, &mut c, 1) };
        assert_eq!(c.rip(), 0xdead);
        assert_eq!(c.gp(RSP), s.p(8));
        assert_eq!((c.gp(RBX), c.gp(RBP)), (0xb3, 0xb5));
        assert_eq!(st.ptrs[16 + RBX], s.p(5) as usize);
        assert_eq!(st.ptrs[16 + RBP], s.p(6) as usize);
        assert_eq!(st.ptrs[16 + RSI], 0);
        assert_eq!(st.frame, s.p(0) as usize);
        assert_eq!(st.handler, 0);
    }

    #[test]
    fn standard_frame_epilog_steps() {
        let m = standard(1, &[]);
        // (rip offset, rsp index, rbx already popped)
        for (off, sp, rbx_popped) in [(10, 0, false), (14, 5, false), (15, 6, true), (16, 7, true)]
        {
            let s = standard_stack();
            let mut c = Context::new();
            c.set_gp(RSP, s.p(sp));
            if rbx_popped {
                c.set_gp(RBX, 0xb3);
            }
            if sp == 7 {
                c.set_gp(RBP, 0xb5);
            }
            let st = unsafe { unwind(m.0 + off, &mut c, 1) };
            assert_eq!(c.rip(), 0xdead, "off {off}");
            assert_eq!(c.gp(RSP), s.p(8), "off {off}");
            assert_eq!(c.gp(RBX), 0xb3, "off {off}");
            assert_eq!(c.gp(RBP), 0xb5, "off {off}");
            assert_eq!(st.handler, 0);
            if sp <= 5 {
                assert_eq!(st.ptrs[16 + RBX], s.p(5) as usize);
            }
        }
    }

    #[test]
    fn standard_frame_partial_prolog() {
        let m = standard(1, &[]);
        // rip offset -> index of rsp, which registers are already loaded
        for (off, sp, rbx, rbp) in [
            (0, 7, false, false),
            (1, 6, false, true),
            (2, 5, true, true),
            (3, 5, true, true),
            (5, 5, true, true),
        ] {
            let s = standard_stack();
            let mut c = Context::new();
            c.set_gp(RSP, s.p(sp));
            unsafe { unwind(m.0 + off, &mut c, 1) };
            assert_eq!(c.rip(), 0xdead, "off {off}");
            assert_eq!(c.gp(RSP), s.p(8), "off {off}");
            assert_eq!(c.gp(RBX), if rbx { 0xb3 } else { 0 }, "off {off}");
            assert_eq!(c.gp(RBP), if rbp { 0xb5 } else { 0 }, "off {off}");
        }
    }

    #[test]
    fn handler_only_for_its_type_and_past_the_prolog() {
        // handler at +0x500, then one data word
        for (flags, e, u) in [(1u8, true, false), (2, false, true), (3, true, true)] {
            let m = standard(1 | (flags << 3), &[0, 5, 0, 0, 0xfe, 0xca, 0, 0]);
            let s = standard_stack();
            let run = |pc: usize, sp: usize, ty: u32| {
                let mut c = Context::new();
                c.set_gp(RSP, s.p(sp));
                unsafe { unwind(m.0 + pc, &mut c, ty) }
            };
            let st = run(6, 0, 1);
            assert_eq!(st.handler, if e { m.0 + 0x500 } else { 0 });
            if e {
                assert_eq!(st.data, m.0 + 0x400 + 4 + 8 + 4);
                assert_eq!(unsafe { *(st.data as *const u32) }, 0xcafe);
            }
            assert_eq!(run(6, 0, 2).handler, if u { m.0 + 0x500 } else { 0 });
            assert_eq!(run(6, 0, 0).handler, 0);
            // inside the prolog and in an epilog: no handler
            assert_eq!(run(2, 5, 1).handler, 0);
            assert_eq!(run(14, 5, 1).handler, 0);
        }
    }

    #[test]
    fn frame_pointer() {
        // push rbp; sub rsp,0x20; lea rbp,[rsp+0x10]; body; lea rsp,[rbp+0x10]; pop rbp; ret
        let mut code = vec![0x55, 0x48, 0x83, 0xec, 0x20, 0x48, 0x8d, 0x6c, 0x24, 0x10];
        code.extend_from_slice(&[0x90; 4]);
        code.extend_from_slice(&[0x48, 0x8d, 0x65, 0x10, 0x5d, 0xc3]);
        let info = [1, 10, 3, 0x15, 10, 0x03, 5, 0x32, 1, 0x50, 0, 0];
        let m = Mem::single(&code, &info);
        let s = Stack::new(32);
        s.set(20, 0xb5);
        s.set(21, 0xdead);
        let frame = s.p(16);
        let check = |c: &Context, st: &Step| {
            assert_eq!(c.rip(), 0xdead);
            assert_eq!(c.gp(RSP), s.p(22));
            assert_eq!(c.gp(RBP), 0xb5);
            assert_eq!(st.frame, frame as usize);
            assert_eq!(st.ptrs[16 + RBP], s.p(20) as usize);
        };
        // body: rsp moved by alloca, rbp holds the frame
        let mut c = Context::new();
        c.set_gp(RSP, s.p(8));
        c.set_gp(RBP, frame + 0x10);
        let st = unsafe { unwind(m.0 + 10, &mut c, 1) };
        check(&c, &st);
        // before the lea: the frame register is not set up yet
        let mut c = Context::new();
        c.set_gp(RSP, frame);
        c.set_gp(RBP, 0x1111);
        let st = unsafe { unwind(m.0 + 5, &mut c, 1) };
        check(&c, &st);
        // epilog starting with lea rsp,[rbp+0x10]
        let mut c = Context::new();
        c.set_gp(RSP, s.p(8));
        c.set_gp(RBP, frame + 0x10);
        unsafe { unwind(m.0 + 14, &mut c, 1) };
        assert_eq!((c.rip(), c.gp(RSP), c.gp(RBP)), (0xdead, s.p(22), 0xb5));
    }

    #[test]
    fn saves_in_the_frame() {
        // sub rsp,0x68 (4); rsi at +0x10 (9); rdi far at +0x40 (14); xmm6 at +0x30 (19);
        // xmm7 far at +0x50 (24)
        let info = [
            1, 24, 11, 0, //
            24, 0x79, 0x50, 0, 0, 0, //
            19, 0x68, 3, 0, //
            14, 0x75, 0x40, 0, 0, 0, //
            9, 0x64, 2, 0, //
            4, 0xc2, 0, 0,
        ];
        let m = Mem::single(&[0x90; 40], &info);
        let s = Stack::new(20);
        s.set(2, 0x5151);
        s.set(8, 0xd1d1);
        for i in 0..2 {
            s.set(6 + i, 0x6000 + i as u64);
            s.set(10 + i, 0x7000 + i as u64);
        }
        s.set(13, 0xdead);
        let mut c = Context::new();
        c.set_gp(RSP, s.p(0));
        let st = unsafe { unwind(m.0 + 30, &mut c, 1) };
        assert_eq!(c.rip(), 0xdead);
        assert_eq!(c.gp(RSP), s.p(14));
        assert_eq!((c.gp(RSI), c.gp(RDI)), (0x5151, 0xd1d1));
        assert_eq!(c.xmm(6)[..8], 0x6000u64.to_le_bytes());
        assert_eq!(c.xmm(6)[8..], 0x6001u64.to_le_bytes());
        assert_eq!(c.xmm(7)[..8], 0x7000u64.to_le_bytes());
        assert_eq!(st.ptrs[6], s.p(6) as usize);
        assert_eq!(st.ptrs[7], s.p(10) as usize);
        assert_eq!(st.ptrs[16 + RSI], s.p(2) as usize);
        assert_eq!(st.ptrs[16 + RDI], s.p(8) as usize);
        assert_eq!(st.frame, s.p(0) as usize);

        // rip right after the rsi save (offset 9): the later saves have not happened
        let mut c = Context::new();
        c.set_gp(RSP, s.p(0));
        unsafe { unwind(m.0 + 9, &mut c, 1) };
        assert_eq!((c.gp(RSI), c.gp(RDI)), (0x5151, 0));
        assert_eq!(c.xmm(6), [0; 16]);
        assert_eq!(c.gp(RSP), s.p(14));
    }

    #[test]
    fn large_allocations() {
        // ALLOC_LARGE, form 0: size in the next slot, in units of 8
        let m = Mem::single(&[0x90; 16], &[1, 4, 2, 0, 4, 0x01, 0x00, 0x02]);
        let s = Stack::new(600);
        s.set(512, 0xdead);
        let mut c = Context::new();
        c.set_gp(RSP, s.p(0));
        unsafe { unwind(m.0 + 8, &mut c, 1) };
        assert_eq!((c.rip(), c.gp(RSP)), (0xdead, s.p(513)));
        // form 1: 32-bit size in the next two slots
        let m = Mem::single(
            &[0x90; 16],
            &[1, 4, 3, 0, 4, 0x11, 0x00, 0x00, 0x02, 0x00, 0, 0],
        );
        let s = Stack::new(0x4010);
        s.set(0x4000, 0xbeef);
        let mut c = Context::new();
        c.set_gp(RSP, s.p(0));
        unsafe { unwind(m.0 + 8, &mut c, 1) };
        assert_eq!((c.rip(), c.gp(RSP)), (0xbeef, s.p(0x4001)));
    }

    #[test]
    fn machine_frame() {
        for (info, f) in [(0x0au8, 0usize), (0x1a, 1)] {
            let m = Mem::single(&[0x90; 8], &[1, 0, 1, 0, 0, info, 0, 0]);
            let s = Stack::new(20);
            s.set(0, 0xe44); // error code, skipped for info 1
            s.set(f, 0xabc);
            s.set(f + 1, 0x33);
            s.set(f + 2, 0x246);
            s.set(f + 3, s.p(15));
            s.set(f + 4, 0x2b);
            let mut c = Context::new();
            c.set_gp(RSP, s.p(0));
            unsafe { unwind(m.0 + 2, &mut c, 1) };
            assert_eq!(c.rip(), 0xabc);
            assert_eq!(c.gp(RSP), s.p(15));
            assert_eq!(c.get(CONTEXT_SEGCS_OFF) & 0xffff, 0x33);
            assert_eq!(c.get(CONTEXT_EFLAGS_OFF) & 0xffff_ffff, 0x246);
        }
    }

    #[test]
    fn chained_info() {
        let m = Mem::new();
        m.put(0, &[0x90; 0x20]);
        // primary: push rbp, exception handler at 0x500 with one data word
        m.put(
            0x400,
            &[0x09, 1, 1, 0, 1, 0x50, 0, 0, 0, 5, 0, 0, 0x77, 0, 0, 0],
        );
        // part at 0x100: push rbx, then the primary's runtime function
        m.put(0x440, &[0x21, 0, 1, 0, 0, 0x30, 0, 0]);
        m.put32(0x448, 0);
        m.put32(0x44c, 0x20);
        m.put32(0x450, 0x400);
        m.rf(0, 0, 0x20, 0x400);
        m.rf(1, 0x100, 0x120, 0x440);
        m.register(2);
        let s = Stack::new(8);
        s.set(0, 0xb3);
        s.set(1, 0xb5);
        s.set(2, 0xdead);
        let mut c = Context::new();
        c.set_gp(RSP, s.p(0));
        let st = unsafe { unwind(m.0 + 0x104, &mut c, 1) };
        assert_eq!(
            (c.rip(), c.gp(RSP), c.gp(RBX), c.gp(RBP)),
            (0xdead, s.p(3), 0xb3, 0xb5)
        );
        assert_eq!(st.handler, m.0 + 0x500);
        assert_eq!(unsafe { *(st.data as *const u32) }, 0x77);
        let mut c = Context::new();
        c.set_gp(RSP, s.p(0));
        assert_eq!(unsafe { unwind(m.0 + 0x104, &mut c, 2) }.handler, 0);
        // the gap between the two parts has no entry
        let e = unsafe { RtlLookupFunctionEntry(m.0 + 0x80, std::ptr::null_mut(), 0) };
        assert_eq!(e, 0);
    }

    #[test]
    fn leaf_callee_then_framed_caller() {
        let m = standard(1, &[]);
        let s = standard_stack();
        let leaf = 0x7fff_0000_1000usize;
        // stack: leaf's return address (into the caller's body), then the caller's frame
        s.set(8, (m.0 + 6) as u64);
        for (i, v) in [0u64, 0, 0, 0, 0, 0xc3, 0xc5, 0xfeed]
            .into_iter()
            .enumerate()
        {
            s.set(9 + i, v);
        }
        let mut c = Context::new();
        c.set_gp(RSP, s.p(8));
        let st = unsafe { unwind(leaf, &mut c, 1) };
        assert_eq!(
            (c.rip(), c.gp(RSP), st.handler),
            ((m.0 + 6) as u64, s.p(9), 0)
        );
        let pc = c.rip() as usize;
        unsafe { unwind(pc, &mut c, 1) };
        assert_eq!(
            (c.rip(), c.gp(RSP), c.gp(RBX), c.gp(RBP)),
            (0xfeed, s.p(17), 0xc3, 0xc5)
        );
    }

    #[test]
    fn malformed_unwind_info_leaves_the_context() {
        // version 3: an unsupported layout
        let m = Mem::single(&[0x90; 8], &[3, 0, 0, 0]);
        let s = Stack::new(4);
        let mut c = Context::new();
        c.set_gp(RSP, s.p(0));
        let before = c.0;
        assert_eq!(unsafe { unwind(m.0 + 2, &mut c, 1) }.handler, 0);
        assert!(c.0 == before);
        // a valid push, then an undefined opcode: the push is rolled back
        let m = Mem::single(&[0x90; 8], &[1, 0, 2, 0, 0, 0x30, 0, 0x0c]);
        let mut c = Context::new();
        c.set_gp(RSP, s.p(0));
        let before = c.0;
        let mut ptrs = [0usize; 32];
        let r = unsafe {
            virtual_unwind(
                1,
                m.0,
                m.0 + 2,
                (m.0 + 0x800) as *const RuntimeFunction,
                c.0.as_mut_ptr(),
                ptrs.as_mut_ptr(),
            )
        };
        assert_eq!(r.err(), Some(UnwindError::BadOpcode));
        assert!(c.0 == before);
        assert_eq!(ptrs, [0; 32]);
    }

    #[test]
    fn capture_context_fields() {
        let m = Mem::new();
        let ctx = Context::new();
        let out = Box::new(0u64);
        // sub rsp,0x28; mov r10,0x55; mov rcx,ctx; mov rax,capture; call rax; then record rsp
        let mut g = vec![0x48, 0x83, 0xec, 0x28, 0x49, 0xc7, 0xc2, 0x55, 0, 0, 0];
        g.extend_from_slice(&[0x48, 0xb9]);
        g.extend_from_slice(&(ctx.0.as_ptr() as u64).to_le_bytes());
        g.extend_from_slice(&[0x48, 0xb8]);
        g.extend_from_slice(&(nativ64_rtl_capture_context as *const () as u64).to_le_bytes());
        g.extend_from_slice(&[0xff, 0xd0]);
        let after_call = g.len();
        g.extend_from_slice(&[0x48, 0xb8]);
        g.extend_from_slice(&(&*out as *const u64 as u64).to_le_bytes());
        g.extend_from_slice(&[0x48, 0x89, 0x20, 0x48, 0x83, 0xc4, 0x28, 0xc3]);
        m.put(0, &g);
        let f: extern "win64" fn() = unsafe { std::mem::transmute(m.0) };
        f();
        assert_eq!(ctx.get(CONTEXT_FLAGS_OFF) as u32, CONTEXT_CAPTURE_FLAGS);
        assert_eq!(ctx.gp(10), 0x55);
        assert_eq!(ctx.rip(), (m.0 + after_call) as u64);
        assert_eq!(ctx.gp(RSP), unsafe { std::ptr::read_volatile(&*out) });
        assert_eq!(ctx.gp(1), ctx.0.as_ptr() as u64);
        assert_eq!(ctx.get(CONTEXT_MXCSR_OFF) & 0xffff, 0x1f80);
        assert_ne!(ctx.get(CONTEXT_SEGCS_OFF) & 0xffff, 0);
    }

    /// Captures a context, then restores a modified copy that runs `gen_at` (which records some
    /// registers and restores the capture again). Returns how often the code after the capture
    /// ran, the recorded words, and the stack top given to the modified context.
    #[inline(never)]
    unsafe fn round_trip(gen_at: usize, patch_at: usize) -> (u32, [u64; 8], u64) {
        let mut a = Context::new();
        let mut b = Context::new();
        let mut res = Box::new([0u64; 8]);
        let scratch = Stack::new(16);
        let flag: *mut u32 = Box::leak(Box::new(0u32));
        nativ64_rtl_capture_context(a.0.as_mut_ptr());
        if std::ptr::read_volatile(flag) == 0 {
            std::ptr::write_volatile(flag, 1);
            *((patch_at) as *mut u64) = a.0.as_ptr() as u64;
            b.0 = a.0;
            b.set_gp(RAX, 0xaaaa_bbbb_cccc_dddd);
            b.set_gp(RBX, res.as_mut_ptr() as u64);
            b.set_gp(12, 0x1212);
            let pat: [u8; 16] = std::array::from_fn(|i| 0xa0 + i as u8);
            let o = CONTEXT_XMM0_OFF + 48;
            b.0[o..o + 16].copy_from_slice(&pat);
            b.set_gp(RSP, scratch.p(8));
            b.set(CONTEXT_RIP_OFF, gen_at as u64);
            b.set(CONTEXT_EFLAGS_OFF, 0x45);
            nativ64_rtl_restore_context(b.0.as_ptr(), std::ptr::null());
        }
        (std::ptr::read_volatile(flag), *res, scratch.p(8))
    }

    #[test]
    fn capture_then_restore_resumes_after_the_capture() {
        let m = Mem::new();
        // Runs on the restored context: record rax, r12, xmm3, rsp and the flags into [rbx],
        // then restore the original capture (rcx patched in by `round_trip`).
        let mut g = vec![0x48, 0x89, 0x03, 0x4c, 0x89, 0x63, 0x08];
        g.extend_from_slice(&[0xf3, 0x0f, 0x7f, 0x5b, 0x10, 0x48, 0x89, 0x63, 0x20]);
        g.extend_from_slice(&[0x9c, 0x58, 0x48, 0x89, 0x43, 0x28]);
        g.extend_from_slice(&[0x48, 0xb9]);
        let patch = g.len();
        g.extend_from_slice(&[0; 8]);
        g.extend_from_slice(&[0x31, 0xd2, 0x48, 0xb8]);
        g.extend_from_slice(&(nativ64_rtl_restore_context as *const () as u64).to_le_bytes());
        g.extend_from_slice(&[0xff, 0xe0]);
        m.put(0, &g);
        let (n, r, top) = unsafe { round_trip(m.0, m.0 + patch) };
        assert_eq!(n, 1, "no resume after the capture");
        assert_eq!(r[0], 0xaaaa_bbbb_cccc_dddd);
        assert_eq!(r[1], 0x1212);
        let x: [u8; 16] = std::array::from_fn(|i| 0xa0 + i as u8);
        assert_eq!(r[2], u64::from_le_bytes(x[..8].try_into().unwrap()));
        assert_eq!(r[3], u64::from_le_bytes(x[8..].try_into().unwrap()));
        assert_eq!(r[4], top);
        assert_eq!(r[5] & 0xcd5, 0x45);
    }

    /// Dispatch, `RaiseException`, `RtlUnwindEx` and `__C_specific_handler` on generated code.
    mod dispatch {
        use super::*;
        use crate::api_ms_win_core_errorhandling::{
            AddVectoredExceptionHandler, RemoveVectoredExceptionHandler,
            SetUnhandledExceptionFilter, UnhandledExceptionFilter,
        };
        use std::cell::{Cell, RefCell};

        type Log = Vec<(&'static str, u32, u32)>;

        thread_local! {
            static LOG: RefCell<Log> = const { RefCell::new(Vec::new()) };
            static MODE: Cell<i32> = const { Cell::new(0) };
            static FILTER: Cell<i32> = const { Cell::new(0) };
        }

        fn log(tag: &'static str, a: u32, b: u32) {
            LOG.with(|l| l.borrow_mut().push((tag, a, b)));
        }
        fn take_log() -> Log {
            LOG.with(|l| std::mem::take(&mut *l.borrow_mut()))
        }

        /// Offset of the return address of the call, and of the landing pad.
        const RET: u32 = 29;
        const LABEL: u32 = 40;
        const CALL_BEGIN: u32 = 4;
        const CALL_END: u32 = 34;

        /// `sub rsp,0x28; call target(ecx = code, 0, 0, 0); mov eax,tail; add rsp,0x28; ret;
        /// [landing pad: mov eax,label; ] add rsp,0x28; ret`.
        fn body(target: usize, ecx: u32, tail: u32, label: Option<u32>) -> Vec<u8> {
            let mut v = vec![0x48, 0x83, 0xec, 0x28, 0xb9];
            v.extend_from_slice(&ecx.to_le_bytes());
            v.extend_from_slice(&[0x31, 0xd2, 0x45, 0x31, 0xc0, 0x45, 0x31, 0xc9, 0x48, 0xb8]);
            v.extend_from_slice(&(target as u64).to_le_bytes());
            v.extend_from_slice(&[0xff, 0xd0, 0xb8]);
            v.extend_from_slice(&tail.to_le_bytes());
            v.extend_from_slice(&[0x48, 0x83, 0xc4, 0x28, 0xc3, 0x90]);
            assert_eq!(v.len(), LABEL as usize);
            if let Some(x) = label {
                v.push(0xb8);
                v.extend_from_slice(&x.to_le_bytes());
            }
            v.extend_from_slice(&[0x48, 0x83, 0xc4, 0x28, 0xc3]);
            v
        }

        /// Unwind info of `body` (ALLOC_SMALL 0x28) with a language handler and scope table.
        fn info(flags: u8, handler: u32, scopes: &[[u32; 4]]) -> Vec<u8> {
            let mut v = vec![1 | flags << 3, 4, 1, 0, 4, 0x42, 0, 0];
            if flags != 0 {
                v.extend_from_slice(&handler.to_le_bytes());
                v.extend_from_slice(&(scopes.len() as u32).to_le_bytes());
                for s in scopes {
                    for x in s {
                        v.extend_from_slice(&x.to_le_bytes());
                    }
                }
            }
            v
        }

        /// Function `i` of the image: code at 0x1000 + 0x100 i, unwind info at 0x2000 + 0x100 i.
        fn func(m: &Mem, i: usize, code: Vec<u8>, info: Vec<u8>) -> u32 {
            let (begin, ui) = (0x1000 + 0x100 * i as u32, 0x2000 + 0x100 * i as u32);
            m.put(begin as usize, &code);
            m.put(ui as usize, &info);
            m.rf(i, begin, begin + code.len() as u32, ui);
            begin
        }

        /// A jump to a Rust function, reachable by RVA; returns the RVA.
        fn stub(m: &Mem, k: usize, addr: usize) -> u32 {
            let off = 0x3000 + 0x10 * k;
            m.put(off, &[0xff, 0x25, 0, 0, 0, 0]);
            m.put(off + 6, &(addr as u64).to_le_bytes());
            off as u32
        }

        unsafe fn call0(addr: usize) -> u32 {
            let f: unsafe extern "win64" fn() -> u32 = std::mem::transmute(addr);
            f()
        }

        // ---- handlers ----

        unsafe extern "win64" fn vec_resume(p: *mut ExceptionPointers) -> i32 {
            let r = &*(*p).record;
            if r.code != 0xe001_0001 {
                return 0;
            }
            log("vec", r.nparams, (r.address & 0xfff) as u32);
            -1
        }

        unsafe extern "win64" fn vec_args(p: *mut ExceptionPointers) -> i32 {
            let r = &*(*p).record;
            if r.code != 0xe001_0002 {
                return 0;
            }
            log("args", r.nparams, r.info[2] as u32);
            -1
        }

        unsafe extern "win64" fn vec_codes(p: *mut ExceptionPointers) -> i32 {
            let r = &*(*p).record;
            match r.code {
                0xe003_0001 => {
                    log("vec", r.code, r.flags);
                    -1
                }
                0xe003_0002 | STATUS_NONCONTINUABLE_EXCEPTION | STATUS_INVALID_DISPOSITION => {
                    let orig = if r.record != 0 {
                        (*(r.record as *const ExceptionRecord)).code
                    } else {
                        0
                    };
                    log("vec", r.code, orig);
                    0
                }
                _ => 0,
            }
        }

        unsafe extern "win64" fn h_inner(
            rec: *mut ExceptionRecord,
            _frame: usize,
            _ctx: *mut u8,
            _d: *mut DispatcherContext,
        ) -> i32 {
            let code = (*rec).code;
            if code != 0xe002_0001 && code != STATUS_UNWIND {
                return 1;
            }
            log("inner", (*rec).flags, code);
            if MODE.get() == 4 && (*rec).flags & EXCEPTION_UNWINDING != 0 {
                return 3;
            }
            1
        }

        unsafe extern "win64" fn h_outer(
            rec: *mut ExceptionRecord,
            frame: usize,
            ctx: *mut u8,
            d: *mut DispatcherContext,
        ) -> i32 {
            let code = (*rec).code;
            if code != 0xe002_0001 && code != STATUS_UNWIND {
                return 1;
            }
            let flags = (*rec).flags;
            log("outer", flags, code);
            if flags & EXCEPTION_UNWINDING == 0 {
                let begin = (*((*d).function_entry as *const RuntimeFunction)).begin as usize;
                let target = (*d).image_base + begin + LABEL as usize;
                match MODE.get() {
                    1 => nativ64_rtl_unwind(frame, target, std::ptr::null_mut(), 99),
                    2 => nativ64_rtl_unwind_ex(frame + 0x1_0000, target, rec, 99, ctx, 0),
                    3 => nativ64_rtl_unwind_ex(0, 0, rec, 99, ctx, 0),
                    _ => nativ64_rtl_unwind_ex(frame, target, rec, 99, ctx, 0),
                }
            }
            1
        }

        unsafe extern "win64" fn h_bad(
            rec: *mut ExceptionRecord,
            _frame: usize,
            _ctx: *mut u8,
            _d: *mut DispatcherContext,
        ) -> i32 {
            if (*rec).code == 0xe003_0002 || (*rec).code == STATUS_INVALID_DISPOSITION {
                9
            } else {
                1
            }
        }

        unsafe extern "win64" fn filt(p: *mut ExceptionPointers, frame: usize) -> i32 {
            assert_ne!(frame, 0);
            log("filter", (*(*p).record).code, 0);
            FILTER.get()
        }

        unsafe extern "win64" fn fin(abnormal: u8, frame: usize) {
            assert_ne!(frame, 0);
            log("fin", abnormal as u32, 0);
        }

        unsafe extern "win64" fn uef(p: *const ExceptionPointers) -> i32 {
            match (*(*p).record).code {
                0xe004_0001 => -1,
                0xe004_0002 => 1,
                _ => 0,
            }
        }

        // ---- tests ----

        #[test]
        fn raise_continued_by_a_vectored_handler() {
            let m = Mem::new();
            let raise = nativ64_raise_exception as *const () as usize;
            let f = func(&m, 0, body(raise, 0xe001_0001, 42, None), info(0, 0, &[]));
            m.register(1);
            unsafe {
                let h = AddVectoredExceptionHandler(0, vec_resume as *const () as usize);
                assert_eq!(
                    call0(m.0 + f as usize),
                    42,
                    "execution resumes after the call"
                );
                assert_eq!(RemoveVectoredExceptionHandler(h), 1);
            }
            assert_eq!(take_log(), [("vec", 0, RET)]);
            assert!(take_stopped().is_empty());
        }

        #[test]
        fn raise_passes_its_arguments() {
            let args: [usize; 20] = std::array::from_fn(|i| 10 + i);
            unsafe {
                let h = AddVectoredExceptionHandler(0, vec_args as *const () as usize);
                nativ64_raise_exception(0xe001_0002, 0, 20, args.as_ptr());
                nativ64_raise_exception(0xe001_0002, 0, 3, args.as_ptr());
                // arguments announced but not given
                nativ64_raise_exception(0xe001_0002, 0, 5, std::ptr::null());
                RemoveVectoredExceptionHandler(h);
            }
            assert_eq!(
                take_log(),
                [("args", 15, 12), ("args", 3, 12), ("args", 0, 0)]
            );
        }

        /// A (outer) calls B (inner) which raises; B's handler searches on, A's unwinds.
        fn chain(mode: i32) -> (u32, Log) {
            MODE.set(mode);
            let m = Mem::new();
            let b_addr = m.0 + 0x1100;
            let ho = stub(&m, 0, h_outer as *const () as usize);
            let hi = stub(&m, 1, h_inner as *const () as usize);
            func(&m, 0, body(b_addr, 0, 5, None), info(3, ho, &[]));
            let raise = nativ64_raise_exception as *const () as usize;
            func(&m, 1, body(raise, 0xe002_0001, 6, None), info(3, hi, &[]));
            m.register(2);
            let r = unsafe { call0(m.0 + 0x1000) };
            (r, take_log())
        }

        #[test]
        fn unwind_calls_unwind_handlers_and_lands_on_the_target() {
            let c = 0xe002_0001;
            let (r, l) = chain(0);
            assert_eq!(r, 99, "the landing pad sees the return value");
            assert_eq!(
                l,
                [
                    ("inner", 0, c),
                    ("outer", 0, c),
                    ("inner", EXCEPTION_UNWINDING, c),
                    ("outer", EXCEPTION_UNWINDING | EXCEPTION_TARGET_UNWIND, c),
                ]
            );
            assert!(take_stopped().is_empty());
        }

        #[test]
        fn rtl_unwind_without_a_record_makes_one() {
            let (r, l) = chain(1);
            assert_eq!(r, 99);
            let u = STATUS_UNWIND;
            assert_eq!(l[2], ("inner", EXCEPTION_UNWINDING, u));
            assert_eq!(
                l[3],
                ("outer", EXCEPTION_UNWINDING | EXCEPTION_TARGET_UNWIND, u)
            );
        }

        #[test]
        fn unwind_to_a_frame_that_is_not_on_the_stack_is_fatal() {
            let (r, l) = chain(2);
            // the failed unwind returned to the handler, the search went on and found nobody;
            // the raise returned, then A's own tail ran
            assert_eq!(r, 5);
            assert_eq!(l.len(), 4, "{l:?}");
            let codes: Vec<u32> = take_stopped().iter().map(|s| s.0).collect();
            assert_eq!(codes, [STATUS_INVALID_UNWIND_TARGET, 0xe002_0001]);
        }

        #[test]
        fn exit_unwind_has_its_flag() {
            let (_, l) = chain(3);
            let f = EXCEPTION_UNWINDING | EXCEPTION_EXIT_UNWIND;
            assert!(l.contains(&("inner", f, 0xe002_0001)), "{l:?}");
            assert!(l.contains(&("outer", f, 0xe002_0001)), "{l:?}");
            assert_eq!(take_stopped()[0].0, STATUS_INVALID_UNWIND_TARGET);
        }

        #[test]
        fn a_collided_unwind_goes_on_from_the_handlers_context() {
            let (r, l) = chain(4);
            assert_eq!(r, 99);
            let c = 0xe002_0001;
            assert_eq!(l[2], ("inner", EXCEPTION_UNWINDING, c));
            let flags = EXCEPTION_UNWINDING | EXCEPTION_COLLIDED_UNWIND | EXCEPTION_TARGET_UNWIND;
            assert_eq!(l[3], ("outer", flags, c));
        }

        /// H (scopes given) calls K (a `__finally` around its call) which raises.
        fn c_specific(scopes: &dyn Fn(u32, u32) -> Vec<[u32; 4]>, filter: i32) -> (u32, Log) {
            FILTER.set(filter);
            let m = Mem::new();
            let hs = stub(
                &m,
                0,
                crate::vcruntime140::__C_specific_handler as *const () as usize,
            );
            let fl = stub(&m, 1, filt as *const () as usize);
            let fi = stub(&m, 2, fin as *const () as usize);
            let hscopes = scopes(fl, 0x1000 + LABEL);
            func(
                &m,
                0,
                body(m.0 + 0x1100, 0, 1, Some(7)),
                info(3, hs, &hscopes),
            );
            let kscopes = [[0x1100 + CALL_BEGIN, 0x1100 + CALL_END, fi, 0]];
            let raise = nativ64_raise_exception as *const () as usize;
            func(
                &m,
                1,
                body(raise, 0xe005_0001, 5, None),
                info(3, hs, &kscopes),
            );
            m.register(2);
            let r = unsafe { call0(m.0 + 0x1000) };
            (r, take_log())
        }

        fn except(handler: u32, target: u32) -> [u32; 4] {
            [0x1000 + CALL_BEGIN, 0x1000 + CALL_END, handler, target]
        }

        const C: u32 = 0xe005_0001;

        #[test]
        fn except_filter_executes_and_runs_the_finally() {
            let (r, l) = c_specific(&|fl, label| vec![except(fl, label)], 1);
            assert_eq!(r, 7, "lands in the __except body");
            assert_eq!(l, [("filter", C, 0), ("fin", 1, 0)]);
            assert!(take_stopped().is_empty());
        }

        #[test]
        fn except_filter_searches_on_then_the_exception_is_fatal() {
            let (r, l) = c_specific(&|fl, label| vec![except(fl, label)], 0);
            assert_eq!(r, 1, "RaiseException returned in a test build");
            assert_eq!(l, [("filter", C, 0)], "no unwind, no finally");
            let s = take_stopped();
            assert_eq!(s.len(), 1);
            assert_eq!(s[0].0, C);
        }

        #[test]
        fn except_filter_continues_execution() {
            let (r, l) = c_specific(&|fl, label| vec![except(fl, label)], -1);
            assert_eq!(r, 1);
            assert_eq!(l, [("filter", C, 0)]);
            assert!(take_stopped().is_empty());
        }

        #[test]
        fn constant_execute_handler() {
            let (r, l) = c_specific(&|_, label| vec![except(1, label)], 0);
            assert_eq!(r, 7);
            assert_eq!(l, [("fin", 1, 0)]);
        }

        #[test]
        fn nested_scopes_inner_filter_declines() {
            let (r, l) = c_specific(&|fl, label| vec![except(fl, label), except(1, label)], 0);
            assert_eq!(r, 7);
            assert_eq!(l, [("filter", C, 0), ("fin", 1, 0)]);
        }

        #[test]
        fn a_finally_around_the_landing_pad_does_not_run() {
            let (r, l) = c_specific(
                &|fin_stub, label| {
                    // a __try/__finally enclosing the whole __try/__except, landing pad
                    // included (`fin_stub` is the filter stub: calling it would be logged)
                    vec![
                        [0x1000 + CALL_BEGIN, 0x1000 + LABEL + 5, fin_stub, 0],
                        except(1, label),
                    ]
                },
                0,
            );
            assert_eq!(r, 7);
            // only K's own finally ran
            assert_eq!(l, [("fin", 1, 0)]);
        }

        #[test]
        fn noncontinuable_exception_cannot_be_continued() {
            unsafe {
                let h = AddVectoredExceptionHandler(0, vec_codes as *const () as usize);
                let no_args = std::ptr::null();
                nativ64_raise_exception(0xe003_0001, EXCEPTION_NONCONTINUABLE, 0, no_args);
                RemoveVectoredExceptionHandler(h);
            }
            assert_eq!(
                take_log(),
                [
                    ("vec", 0xe003_0001, EXCEPTION_NONCONTINUABLE),
                    ("vec", STATUS_NONCONTINUABLE_EXCEPTION, 0xe003_0001),
                ]
            );
            assert_eq!(take_stopped()[0].0, STATUS_NONCONTINUABLE_EXCEPTION);
        }

        #[test]
        fn a_bad_disposition_raises_invalid_disposition() {
            let m = Mem::new();
            let hb = stub(&m, 0, h_bad as *const () as usize);
            let raise = nativ64_raise_exception as *const () as usize;
            let f = func(&m, 0, body(raise, 0xe003_0002, 3, None), info(1, hb, &[]));
            m.register(1);
            unsafe {
                let h = AddVectoredExceptionHandler(0, vec_codes as *const () as usize);
                assert_eq!(call0(m.0 + f as usize), 3);
                RemoveVectoredExceptionHandler(h);
            }
            assert_eq!(
                take_log(),
                [
                    ("vec", 0xe003_0002, 0),
                    ("vec", STATUS_INVALID_DISPOSITION, 0xe003_0002)
                ]
            );
            assert_eq!(take_stopped()[0].0, STATUS_INVALID_DISPOSITION);
        }

        #[test]
        fn backtrace_without_a_context_is_empty() {
            unsafe {
                assert_eq!(
                    backtrace_line(std::ptr::null_mut(), 16),
                    "nativ64: backtrace"
                );
                // a context in no image unwinds nowhere
                let mut ctx = Context([0; CONTEXT_SIZE]);
                assert_eq!(backtrace_line(ctx.0.as_mut_ptr(), 16), "nativ64: backtrace");
            }
        }

        #[test]
        fn unhandled_exception_goes_to_the_top_level_filter() {
            let _lock = FILTER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            unsafe {
                let prev = SetUnhandledExceptionFilter(uef as *const () as usize);
                let no_args = std::ptr::null();
                // CONTINUE_EXECUTION: the raise returns, nothing is reported
                nativ64_raise_exception(0xe004_0001, 0, 0, no_args);
                assert!(take_stopped().is_empty());
                // EXECUTE_HANDLER or a refusal: reported and stopped
                nativ64_raise_exception(0xe004_0002, 0, 0, no_args);
                assert_eq!(take_stopped()[0].0, 0xe004_0002);
                nativ64_raise_exception(0xe004_0003, 0, 0, no_args);
                assert_eq!(take_stopped()[0].0, 0xe004_0003);

                // the export itself
                let mut rec = ExceptionRecord::new(0xe004_0001, 0, 0x1234);
                let p = ExceptionPointers {
                    record: &mut rec,
                    context: std::ptr::null_mut(),
                };
                assert_eq!(UnhandledExceptionFilter(&p), -1);
                (*p.record).code = 0xe004_0002;
                assert_eq!(UnhandledExceptionFilter(&p), 1);
                assert!(
                    take_stopped().is_empty(),
                    "the filter's verdict is not reported"
                );
                (*p.record).code = 0xe004_0003;
                assert_eq!(UnhandledExceptionFilter(&p), 1);
                assert_eq!(take_stopped(), [(0xe004_0003, 0x1234)]);
                SetUnhandledExceptionFilter(0);
                assert_eq!(
                    UnhandledExceptionFilter(&p),
                    1,
                    "no filter: report and stop"
                );
                assert_eq!(take_stopped().len(), 1);
                SetUnhandledExceptionFilter(prev);
            }
        }
    }
}
