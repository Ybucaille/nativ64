//! File APIs of `kernel32` (`api-ms-win-core-file`), on the path rule of [`crate::paths`].
//! Done so far: `GetFileAttributesW/A` (directory, read-only or normal), `GetFileAttributesExW`,
//! `CreateFileW`, `ReadFile`, `WriteFile`, `SetFilePointerEx`, `GetFileSizeEx`, `SetEndOfFile`,
//! `FlushFileBuffers`, `FindFirstFileW/ExW`, `FindNextFileW`, `FindClose`, `CreateDirectoryW`,
//! `DeleteFileW`, `RemoveDirectoryW`, `MoveFileExW`, `GetDiskFreeSpaceExW`.
//!
//! Reads look under the write directory first (what the title saved), then in the package;
//! anything that may modify a file lives only under the write directory (`paths`). Overlapped
//! I/O is done synchronously at the offset in the `OVERLAPPED`. Sharing and flags are ignored.

use crate::api_ms_win_core_errorhandling::set_last_error;
use crate::paths::map_read_path;
use std::ffi::c_void;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::fs::MetadataExt;

const INVALID_FILE_ATTRIBUTES: u32 = 0xFFFF_FFFF;
const FILE_ATTRIBUTE_READONLY: u32 = 0x1;
const FILE_ATTRIBUTE_DIRECTORY: u32 = 0x10;
const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
pub(crate) const ERROR_FILE_NOT_FOUND: u32 = 2;
pub(crate) const ERROR_PATH_NOT_FOUND: u32 = 3;
const ERROR_ACCESS_DENIED: u32 = 5;
const ERROR_INVALID_HANDLE: u32 = 6;
const ERROR_FILE_EXISTS: u32 = 80;
const ERROR_ALREADY_EXISTS: u32 = 183;
const ERROR_NO_MORE_FILES: u32 = 18;
const ERROR_NOT_SUPPORTED: u32 = 50;
const ERROR_DIR_NOT_EMPTY: u32 = 145;
const ERROR_DIRECTORY: u32 = 267;
const MOVEFILE_REPLACE_EXISTING: u32 = 1;
const FIND_DATA_SIZE: usize = 592;
const _: () = assert!(FIND_DATA_SIZE == 44 + 520 + 28);
const INVALID_HANDLE_VALUE: usize = usize::MAX;
const WRITE_ACCESS: u32 = 0x4000_0000 | 0x1000_0000 | 0x2 | 0x4;
const CREATE_NEW: u32 = 1;
const CREATE_ALWAYS: u32 = 2;
const OPEN_EXISTING: u32 = 3;
const OPEN_ALWAYS: u32 = 4;
const TRUNCATE_EXISTING: u32 = 5;

fn last_error_of(e: i32) -> u32 {
    match e {
        libc::ENOTDIR => ERROR_PATH_NOT_FOUND,
        libc::EACCES | libc::EPERM | libc::EROFS | libc::EISDIR => ERROR_ACCESS_DENIED,
        libc::EEXIST => ERROR_FILE_EXISTS,
        libc::EBADF => ERROR_INVALID_HANDLE,
        _ => ERROR_FILE_NOT_FOUND,
    }
}

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

fn open_path(path: &str, access: u32, disposition: u32) -> Result<(OwnedFd, bool), u32> {
    let write = access & WRITE_ACCESS != 0
        || matches!(
            disposition,
            CREATE_NEW | CREATE_ALWAYS | TRUNCATE_EXISTING | OPEN_ALWAYS
        );
    let (host, flags) = if write {
        let Some(h) = crate::paths::map_write_path(path) else {
            return Err(ERROR_FILE_NOT_FOUND);
        };
        if let Some(parent) = h.parent() {
            if disposition != OPEN_EXISTING && disposition != TRUNCATE_EXISTING {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        // One descriptor serves reads and writes.
        let mut f = libc::O_RDWR;
        f |= match disposition {
            CREATE_NEW => libc::O_CREAT | libc::O_EXCL,
            CREATE_ALWAYS => libc::O_CREAT | libc::O_TRUNC,
            OPEN_ALWAYS => libc::O_CREAT,
            TRUNCATE_EXISTING => libc::O_TRUNC,
            _ => 0,
        };
        (h, f)
    } else {
        let saved = crate::paths::map_write_path(path).filter(|p| p.exists());
        (saved.unwrap_or_else(|| map_read_path(path)), libc::O_RDONLY)
    };
    let existed = host.exists();
    let c = std::ffi::CString::new(host.as_os_str().as_encoded_bytes())
        .map_err(|_| ERROR_FILE_NOT_FOUND)?;
    // SAFETY: a valid NUL-terminated path.
    let fd = unsafe { libc::open(c.as_ptr(), flags | libc::O_CLOEXEC, 0o644) };
    if fd < 0 {
        return Err(last_error_of(errno()));
    }
    // SAFETY: a descriptor we just opened.
    Ok((unsafe { OwnedFd::from_raw_fd(fd) }, existed))
}

/// The offset of an `OVERLAPPED` (`Offset`, `OffsetHigh` at bytes 16 and 20).
unsafe fn overlapped_offset(o: *const c_void) -> u64 {
    let p = o as *const u32;
    (*p.add(5) as u64) << 32 | *p.add(4) as u64
}

unsafe fn transfer(
    h: usize,
    buf: *mut u8,
    n: u32,
    done: *mut u32,
    ov: *const c_void,
    write: bool,
) -> i32 {
    let r = crate::handles::with_file(h, |f| {
        let fd = f.fd.as_raw_fd();
        let at = if ov.is_null() {
            *f.pos.lock().unwrap_or_else(|e| e.into_inner())
        } else {
            overlapped_offset(ov)
        };
        let got = loop {
            let r = if write {
                libc::pwrite(fd, buf as *const c_void, n as usize, at as libc::off_t)
            } else {
                libc::pread(fd, buf as *mut c_void, n as usize, at as libc::off_t)
            };
            if r < 0 && errno() == libc::EINTR {
                continue;
            }
            break r;
        };
        if got >= 0 && ov.is_null() {
            *f.pos.lock().unwrap_or_else(|e| e.into_inner()) = at + got as u64;
        }
        (got, errno())
    });
    match r {
        Some((got, _)) if got >= 0 => {
            if !done.is_null() {
                *done = got as u32;
            }
            1
        }
        Some((_, e)) => {
            set_last_error(last_error_of(e));
            0
        }
        None => 0,
    }
}

/// Moves the FILETIME at `ft_in` by the local UTC offset (`direction` 1: UTC to local, -1: back)
/// into `ft_out`, as `FileTimeToLocalFileTime` and its inverse do.
unsafe fn convert_local(ft_in: *const u64, ft_out: *mut u64, direction: i64) -> i32 {
    if ft_in.is_null() || ft_out.is_null() {
        set_last_error(crate::handles::ERROR_INVALID_PARAMETER);
        return 0;
    }
    // SAFETY: `ft_in` is a non-NULL FILETIME, possibly only 4-byte aligned.
    let ft = unsafe { ft_in.read_unaligned() };
    match crate::filetime::shift(ft, direction * crate::filetime::local_offset_now()) {
        Some(moved) => {
            // SAFETY: `ft_out` is a non-NULL FILETIME.
            unsafe { ft_out.write_unaligned(moved) };
            1
        }
        None => {
            set_last_error(crate::handles::ERROR_INVALID_PARAMETER);
            0
        }
    }
}

/// FILETIME (100 ns since 1601) of a Unix time.
fn filetime(sec: i64, nsec: i64) -> u64 {
    const EPOCH_DIFF: i64 = 11_644_473_600;
    ((sec + EPOCH_DIFF).max(0) as u64) * 10_000_000 + nsec as u64 / 100
}

fn attrs_of(m: &std::fs::Metadata) -> u32 {
    let mut a = if m.is_dir() {
        FILE_ATTRIBUTE_DIRECTORY
    } else {
        FILE_ATTRIBUTE_NORMAL
    };
    if m.mode() & 0o222 == 0 {
        a = (a & !FILE_ATTRIBUTE_NORMAL) | FILE_ATTRIBUTE_READONLY;
    }
    a
}

/// Metadata of a title path: what the title saved first, then the package.
pub(crate) fn stat_path(path: &str) -> std::io::Result<std::fs::Metadata> {
    if !path.starts_with('/') {
        if let Some(m) = crate::paths::map_write_path(path).and_then(|p| std::fs::metadata(p).ok())
        {
            return Ok(m);
        }
    }
    std::fs::metadata(map_read_path(path))
}

fn io_error(e: &std::io::Error) -> u32 {
    match e.raw_os_error() {
        Some(libc::ENOTDIR) => ERROR_PATH_NOT_FOUND,
        Some(libc::EACCES) => ERROR_ACCESS_DENIED,
        _ => ERROR_FILE_NOT_FOUND,
    }
}

fn exists_in_view(path: &str) -> bool {
    stat_path(path).is_ok()
}

/// `(parent, last component)` of a title path (separators unified, drive dropped); the parent
/// is empty for a top-level name and `/` for a host path at the root.
fn split_path(path: &str) -> (String, String) {
    let u = path.replace('\\', "/");
    let b = u.as_bytes();
    let rest = if b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' {
        &u[2..]
    } else {
        &u[..]
    };
    match rest.rfind('/') {
        Some(0) => ("/".to_string(), rest[1..].to_string()),
        Some(i) => (rest[..i].to_string(), rest[i + 1..].to_string()),
        None => (String::new(), rest.to_string()),
    }
}

/// The error for a missing target: its parent directory missing is a path error.
fn missing_error(path: &str) -> u32 {
    let (parent, _) = split_path(path);
    if parent.is_empty() || parent == "/" || exists_in_view(&parent) {
        ERROR_FILE_NOT_FOUND
    } else {
        ERROR_PATH_NOT_FOUND
    }
}

/// Windows wildcard match (`*`, `?`), ignoring case; `*.*` matches every name.
fn wild_match(pattern: &str, name: &str) -> bool {
    let pat: Vec<char> = if pattern == "*.*" {
        vec!['*']
    } else {
        pattern.chars().flat_map(char::to_lowercase).collect()
    };
    let nm: Vec<char> = name.chars().flat_map(char::to_lowercase).collect();
    let (mut p, mut n, mut star, mut mark) = (0, 0, usize::MAX, 0);
    while n < nm.len() {
        if p < pat.len() && (pat[p] == '?' || pat[p] == nm[n]) && pat[p] != '*' {
            p += 1;
            n += 1;
        } else if p < pat.len() && pat[p] == '*' {
            star = p;
            mark = n;
            p += 1;
        } else if star != usize::MAX {
            p = star + 1;
            mark += 1;
            n = mark;
        } else {
            return false;
        }
    }
    while p < pat.len() && pat[p] == '*' {
        p += 1;
    }
    p == pat.len()
}

/// A `WIN32_FIND_DATAW` record.
fn find_record(name: &str, m: &std::fs::Metadata) -> Option<Vec<u8>> {
    let units: Vec<u16> = name.encode_utf16().collect();
    if units.len() > 259 {
        return None;
    }
    let mut r = vec![0u8; FIND_DATA_SIZE];
    r[0..4].copy_from_slice(&attrs_of(m).to_le_bytes());
    r[4..12].copy_from_slice(&filetime(m.ctime(), m.ctime_nsec()).to_le_bytes());
    r[12..20].copy_from_slice(&filetime(m.atime(), m.atime_nsec()).to_le_bytes());
    r[20..28].copy_from_slice(&filetime(m.mtime(), m.mtime_nsec()).to_le_bytes());
    let size = if m.is_dir() { 0 } else { m.len() };
    r[28..32].copy_from_slice(&((size >> 32) as u32).to_le_bytes());
    r[32..36].copy_from_slice(&(size as u32).to_le_bytes());
    for (i, u) in units.iter().enumerate() {
        r[44 + 2 * i..46 + 2 * i].copy_from_slice(&u.to_le_bytes());
    }
    Some(r)
}

/// The records matching `path` (its last component is the pattern), the directory's entries
/// of the write directory first, then of each package directory; `.` and `..` included.
fn find_entries(path: &str, dirs_only: bool) -> Result<Vec<Vec<u8>>, u32> {
    let (parent, pattern) = split_path(path);
    if pattern.is_empty() {
        return Err(ERROR_PATH_NOT_FOUND);
    }
    let mut hosts = Vec::new();
    if !path.starts_with('/') {
        let w = if parent.is_empty() {
            Some(crate::paths::write_dir())
        } else {
            crate::paths::map_write_path(&parent)
        };
        hosts.extend(w);
    }
    for d in crate::paths::package_dirs() {
        hosts.push(crate::paths::map_read_path_in(&[d], &parent));
    }
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    let mut dir_found = false;
    for h in hosts.iter().filter(|h| h.is_dir()) {
        dir_found = true;
        let Ok(dm) = std::fs::metadata(h) else {
            continue;
        };
        let mut names: Vec<(String, std::fs::Metadata)> =
            vec![(".".to_string(), dm.clone()), ("..".to_string(), dm)];
        if let Ok(rd) = std::fs::read_dir(h) {
            for e in rd.flatten() {
                if let Ok(m) = std::fs::metadata(e.path()) {
                    names.push((e.file_name().to_string_lossy().into_owned(), m));
                }
            }
        }
        for (n, m) in names {
            if (dirs_only && !m.is_dir()) || !wild_match(&pattern, &n) {
                continue;
            }
            if !seen.insert(n.to_lowercase()) {
                continue;
            }
            out.extend(find_record(&n, &m));
        }
    }
    if !dir_found {
        return Err(ERROR_PATH_NOT_FOUND);
    }
    if out.is_empty() {
        return Err(ERROR_FILE_NOT_FOUND);
    }
    Ok(out)
}

unsafe fn find_first(path: *const u16, data: *mut u8, level: u32, op: u32) -> usize {
    let Some(p) = crate::handles::name_w(path).filter(|_| !data.is_null()) else {
        set_last_error(crate::handles::ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    };
    if level > 1 || op > 2 {
        set_last_error(crate::handles::ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    if op == 2 {
        set_last_error(ERROR_NOT_SUPPORTED);
        return INVALID_HANDLE_VALUE;
    }
    match find_entries(&p, op == 1) {
        Ok(mut v) => {
            let first = v.remove(0);
            std::ptr::copy_nonoverlapping(first.as_ptr(), data, FIND_DATA_SIZE);
            crate::handles::create_find_handle(v)
        }
        Err(e) => {
            set_last_error(e);
            INVALID_HANDLE_VALUE
        }
    }
}

/// Maps a write target: refuses what is only in the package (never written).
fn write_target(path: &str) -> Result<std::path::PathBuf, u32> {
    crate::paths::map_write_path(path).ok_or(ERROR_PATH_NOT_FOUND)
}

/// Creates the write-side parent of `path`; the parent must exist in the view.
fn prepare_parent(path: &str, host: &std::path::Path) -> Result<(), u32> {
    let (parent, _) = split_path(path);
    if !(parent.is_empty() || parent == "/" || stat_path(&parent).is_ok_and(|m| m.is_dir())) {
        return Err(ERROR_PATH_NOT_FOUND);
    }
    if let Some(p) = host.parent() {
        std::fs::create_dir_all(p).map_err(|e| io_error(&e))?;
    }
    Ok(())
}

fn create_directory(path: &str) -> Result<(), u32> {
    let host = write_target(path)?;
    if exists_in_view(path) {
        return Err(ERROR_ALREADY_EXISTS);
    }
    prepare_parent(path, &host)?;
    std::fs::create_dir(&host).map_err(|e| match e.raw_os_error() {
        Some(libc::EEXIST) => ERROR_ALREADY_EXISTS,
        _ => io_error(&e),
    })
}

pub(crate) fn delete_file(path: &str) -> Result<(), u32> {
    let host = write_target(path)?;
    match std::fs::symlink_metadata(&host) {
        Ok(m) if m.is_dir() => Err(ERROR_ACCESS_DENIED),
        Ok(_) => std::fs::remove_file(&host).map_err(|e| io_error(&e)),
        // Only in the package, which is never written.
        Err(_) if exists_in_view(path) => Err(ERROR_ACCESS_DENIED),
        Err(_) => Err(missing_error(path)),
    }
}

fn remove_directory(path: &str) -> Result<(), u32> {
    let host = write_target(path)?;
    match std::fs::symlink_metadata(&host) {
        Ok(m) if !m.is_dir() => Err(ERROR_DIRECTORY),
        Ok(_) => std::fs::remove_dir(&host).map_err(|e| match e.raw_os_error() {
            Some(libc::ENOTEMPTY) | Some(libc::EEXIST) => ERROR_DIR_NOT_EMPTY,
            _ => io_error(&e),
        }),
        Err(_) => match stat_path(path) {
            Ok(m) if m.is_dir() => Err(ERROR_ACCESS_DENIED),
            Ok(_) => Err(ERROR_DIRECTORY),
            Err(_) => Err(missing_error(path)),
        },
    }
}

fn move_file(from: &str, to: &str, flags: u32) -> Result<(), u32> {
    let src = write_target(from)?;
    let dst = write_target(to)?;
    if std::fs::symlink_metadata(&src).is_err() {
        // A package file cannot be moved away.
        return Err(if exists_in_view(from) {
            ERROR_ACCESS_DENIED
        } else {
            missing_error(from)
        });
    }
    if src == dst {
        return Ok(());
    }
    if exists_in_view(to) {
        if flags & MOVEFILE_REPLACE_EXISTING == 0 {
            return Err(ERROR_ALREADY_EXISTS);
        }
        if stat_path(to).is_ok_and(|m| m.is_dir()) {
            return Err(ERROR_ACCESS_DENIED);
        }
    }
    prepare_parent(to, &dst)?;
    std::fs::rename(&src, &dst).map_err(|e| io_error(&e))
}

fn disk_space(dir: Option<&str>) -> Result<(u64, u64, u64), u32> {
    let host = match dir {
        None => Some(crate::paths::package_root())
            .filter(|p| p.is_dir())
            .unwrap_or_else(|| ".".into()),
        Some(d) => {
            let (parent, last) = split_path(d);
            if last.is_empty() && parent.is_empty() {
                // A drive root: the write directory (or the nearest existing ancestor).
                let mut w = crate::paths::write_dir();
                while !w.exists() && w.pop() {}
                w
            } else {
                let w = crate::paths::map_write_path(d).filter(|p| p.is_dir());
                match w.or_else(|| Some(map_read_path(d)).filter(|p| p.is_dir())) {
                    Some(p) => p,
                    None => return Err(ERROR_PATH_NOT_FOUND),
                }
            }
        }
    };
    let c = std::ffi::CString::new(host.as_os_str().as_encoded_bytes())
        .map_err(|_| ERROR_PATH_NOT_FOUND)?;
    // SAFETY: a valid path and a zeroed out-structure.
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return Err(last_error_of(errno()));
    }
    let fr = st.f_frsize as u64;
    Ok((
        st.f_bavail as u64 * fr,
        st.f_blocks as u64 * fr,
        st.f_bfree as u64 * fr,
    ))
}

fn attributes(path: &str) -> u32 {
    match stat_path(path) {
        Ok(m) => attrs_of(&m),
        Err(e) => {
            set_last_error(io_error(&e));
            INVALID_FILE_ATTRIBUTES
        }
    }
}

/// BOOL result of an operation, setting the last error on failure.
fn status(r: Result<(), u32>) -> i32 {
    match r {
        Ok(()) => 1,
        Err(e) => {
            set_last_error(e);
            0
        }
    }
}

crate::win32! {
    module "api-ms-win-core-file" => "kernel32";

    // The UTC FILETIME at `ft_in` as local time, with the bias in effect now (Windows ignores
    // the date, daylight saving included). Fails when the result leaves the FILETIME range.
    Native fn FileTimeToLocalFileTime(ft_in: *const u64, ft_out: *mut u64) -> i32 {
        convert_local(ft_in, ft_out, 1)
    }
    // The inverse of `FileTimeToLocalFileTime`.
    Native fn LocalFileTimeToFileTime(ft_in: *const u64, ft_out: *mut u64) -> i32 {
        convert_local(ft_in, ft_out, -1)
    }

    Native fn GetFileAttributesW(path: *const u16) -> u32 {
        match crate::handles::name_w(path) {
            Some(p) => attributes(&p),
            None => {
                set_last_error(crate::handles::ERROR_INVALID_PARAMETER);
                INVALID_FILE_ATTRIBUTES
            }
        }
    }
    Native fn CreateFileW(path: *const u16, access: u32, share: u32, security: *const c_void, disposition: u32, flags: u32, template: usize) -> usize {
        let _ = (share, security, flags, template);
        let Some(p) = crate::handles::name_w(path) else {
            set_last_error(crate::handles::ERROR_INVALID_PARAMETER);
            return INVALID_HANDLE_VALUE;
        };
        match open_path(&p, access, disposition) {
            Ok((fd, existed)) => {
                set_last_error(if existed && matches!(disposition, CREATE_ALWAYS | OPEN_ALWAYS) {
                    ERROR_ALREADY_EXISTS
                } else {
                    0
                });
                crate::handles::create_file_handle(fd)
            }
            Err(e) => {
                set_last_error(e);
                INVALID_HANDLE_VALUE
            }
        }
    }
    Native fn ReadFile(h: usize, buf: *mut u8, n: u32, read: *mut u32, overlapped: *const c_void) -> i32 {
        transfer(h, buf, n, read, overlapped, false)
    }
    Native fn WriteFile(h: usize, buf: *const u8, n: u32, written: *mut u32, overlapped: *const c_void) -> i32 {
        transfer(h, buf as *mut u8, n, written, overlapped, true)
    }
    // method: 0 begin, 1 current, 2 end.
    Native fn SetFilePointerEx(h: usize, distance: i64, new_pos: *mut i64, method: u32) -> i32 {
        let r = crate::handles::with_file(h, |f| {
            let mut pos = f.pos.lock().unwrap_or_else(|e| e.into_inner());
            let base = match method {
                0 => 0,
                1 => *pos as i64,
                2 => {
                    let mut st: libc::stat = std::mem::zeroed();
                    libc::fstat(f.fd.as_raw_fd(), &mut st);
                    st.st_size
                }
                _ => return None,
            };
            let target = base.checked_add(distance).filter(|t| *t >= 0)?;
            *pos = target as u64;
            Some(target)
        });
        match r {
            Some(Some(t)) => {
                if !new_pos.is_null() {
                    *new_pos = t;
                }
                1
            }
            Some(None) => {
                set_last_error(crate::handles::ERROR_INVALID_PARAMETER);
                0
            }
            None => 0,
        }
    }
    Native fn GetFileSizeEx(h: usize, size: *mut i64) -> i32 {
        let r = crate::handles::with_file(h, |f| {
            let mut st: libc::stat = std::mem::zeroed();
            (libc::fstat(f.fd.as_raw_fd(), &mut st) == 0).then_some(st.st_size)
        });
        match r {
            Some(Some(s)) if !size.is_null() => {
                *size = s;
                1
            }
            Some(_) => {
                set_last_error(crate::handles::ERROR_INVALID_PARAMETER);
                0
            }
            None => 0,
        }
    }
    // Creation, access and write times as FILETIMEs (100 ns since 1601); creation uses the
    // status-change time. Any of the three pointers may be NULL.
    Native fn GetFileTime(h: usize, created: *mut u64, accessed: *mut u64, written: *mut u64) -> i32 {
        let ft = filetime;
        let r = crate::handles::with_file(h, |f| {
            let mut st: libc::stat = std::mem::zeroed();
            (libc::fstat(f.fd.as_raw_fd(), &mut st) == 0).then_some(st)
        });
        match r {
            Some(Some(st)) => {
                for (p, v) in [
                    (created, ft(st.st_ctime, st.st_ctime_nsec)),
                    (accessed, ft(st.st_atime, st.st_atime_nsec)),
                    (written, ft(st.st_mtime, st.st_mtime_nsec)),
                ] {
                    if !p.is_null() {
                        p.write_unaligned(v);
                    }
                }
                1
            }
            Some(None) => {
                set_last_error(ERROR_INVALID_HANDLE);
                0
            }
            None => 0,
        }
    }
    Native fn FlushFileBuffers(h: usize) -> i32 {
        crate::handles::with_file(h, |f| {
            libc::fsync(f.fd.as_raw_fd());
            1
        })
        .unwrap_or(0)
    }
    Native fn GetFileAttributesExW(path: *const u16, level: u32, data: *mut u8) -> i32 {
        let Some(p) = crate::handles::name_w(path).filter(|_| !data.is_null() && level == 0) else {
            set_last_error(crate::handles::ERROR_INVALID_PARAMETER);
            return 0;
        };
        match stat_path(&p) {
            Ok(m) => {
                // WIN32_FILE_ATTRIBUTE_DATA: attributes, three FILETIMEs, size high and low.
                let size = if m.is_dir() { 0 } else { m.len() };
                let mut r = [0u8; 36];
                r[0..4].copy_from_slice(&attrs_of(&m).to_le_bytes());
                r[4..12].copy_from_slice(&filetime(m.ctime(), m.ctime_nsec()).to_le_bytes());
                r[12..20].copy_from_slice(&filetime(m.atime(), m.atime_nsec()).to_le_bytes());
                r[20..28].copy_from_slice(&filetime(m.mtime(), m.mtime_nsec()).to_le_bytes());
                r[28..32].copy_from_slice(&((size >> 32) as u32).to_le_bytes());
                r[32..36].copy_from_slice(&(size as u32).to_le_bytes());
                std::ptr::copy_nonoverlapping(r.as_ptr(), data, 36);
                1
            }
            Err(e) => {
                set_last_error(io_error(&e));
                0
            }
        }
    }
    Native fn FindFirstFileW(path: *const u16, data: *mut u8) -> usize {
        find_first(path, data, 0, 0)
    }
    // level: 0 standard, 1 basic (same record); op: 0 name match, 1 directories only.
    Native fn FindFirstFileExW(path: *const u16, level: u32, data: *mut u8, op: u32, filter: *const c_void, flags: u32) -> usize {
        let _ = (filter, flags);
        find_first(path, data, level, op)
    }
    Native fn FindNextFileW(h: usize, data: *mut u8) -> i32 {
        match crate::handles::find_next(h) {
            Some(Some(r)) if !data.is_null() => {
                std::ptr::copy_nonoverlapping(r.as_ptr(), data, FIND_DATA_SIZE);
                1
            }
            Some(Some(_)) => {
                set_last_error(crate::handles::ERROR_INVALID_PARAMETER);
                0
            }
            Some(None) => {
                set_last_error(ERROR_NO_MORE_FILES);
                0
            }
            None => 0,
        }
    }
    Native fn FindClose(h: usize) -> i32 {
        crate::handles::close_find(h)
    }
    Native fn CreateDirectoryW(path: *const u16, security: *const c_void) -> i32 {
        let _ = security;
        let r = match crate::handles::name_w(path) {
            Some(p) => create_directory(&p),
            None => Err(crate::handles::ERROR_INVALID_PARAMETER),
        };
        status(r)
    }
    Native fn DeleteFileW(path: *const u16) -> i32 {
        let r = match crate::handles::name_w(path) {
            Some(p) => delete_file(&p),
            None => Err(crate::handles::ERROR_INVALID_PARAMETER),
        };
        status(r)
    }
    Native fn RemoveDirectoryW(path: *const u16) -> i32 {
        let r = match crate::handles::name_w(path) {
            Some(p) => remove_directory(&p),
            None => Err(crate::handles::ERROR_INVALID_PARAMETER),
        };
        status(r)
    }
    Native fn MoveFileExW(from: *const u16, to: *const u16, flags: u32) -> i32 {
        let r = match (crate::handles::name_w(from), crate::handles::name_w(to)) {
            (Some(f), Some(t)) => move_file(&f, &t, flags),
            _ => Err(crate::handles::ERROR_INVALID_PARAMETER),
        };
        status(r)
    }
    // Truncates or extends the file at its current position.
    Native fn SetEndOfFile(h: usize) -> i32 {
        let r = crate::handles::with_file(h, |f| {
            let at = *f.pos.lock().unwrap_or_else(|e| e.into_inner());
            libc::ftruncate(f.fd.as_raw_fd(), at as libc::off_t) == 0
        });
        match r {
            Some(true) => 1,
            Some(false) => {
                set_last_error(ERROR_ACCESS_DENIED);
                0
            }
            None => 0,
        }
    }
    Native fn GetDiskFreeSpaceExW(dir: *const u16, avail: *mut u64, total: *mut u64, free: *mut u64) -> i32 {
        let d = crate::handles::name_w(dir);
        match disk_space(d.as_deref()) {
            Ok((a, t, f)) => {
                for (p, v) in [(avail, a), (total, t), (free, f)] {
                    if !p.is_null() {
                        p.write_unaligned(v);
                    }
                }
                1
            }
            Err(e) => {
                set_last_error(e);
                0
            }
        }
    }
    Native fn GetFileAttributesA(path: *const u8) -> u32 {
        match crate::handles::name_a(path) {
            Some(p) => attributes(&p),
            None => {
                set_last_error(crate::handles::ERROR_INVALID_PARAMETER);
                INVALID_FILE_ATTRIBUTES
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_file_times_shift_by_the_current_offset() {
        let ft = 132_000_000_000_000_000u64;
        let off = crate::filetime::local_offset_now() * 10_000_000;
        let (mut local, mut back) = (0u64, 0u64);
        unsafe {
            assert_eq!(FileTimeToLocalFileTime(&ft, &mut local), 1);
            assert_eq!(local as i64, ft as i64 + off);
            assert_eq!(LocalFileTimeToFileTime(&local, &mut back), 1);
            assert_eq!(back, ft);
            // Out of range and NULL pointers fail with ERROR_INVALID_PARAMETER.
            let top = 1u64 << 63;
            set_last_error(0);
            assert_eq!(FileTimeToLocalFileTime(&top, &mut local), 0);
            assert_eq!(crate::api_ms_win_core_errorhandling::GetLastError(), 87);
            set_last_error(0);
            assert_eq!(LocalFileTimeToFileTime(std::ptr::null(), &mut local), 0);
            assert_eq!(crate::api_ms_win_core_errorhandling::GetLastError(), 87);
        }
        assert!(crate::lookup("kernel32.dll", "FileTimeToLocalFileTime").is_some());
    }

    #[test]
    fn attributes_of_dir_file_and_missing() {
        unsafe {
            let dir = std::env::temp_dir();
            let w: Vec<u16> = dir.to_str().unwrap().encode_utf16().chain([0]).collect();
            assert_eq!(GetFileAttributesW(w.as_ptr()), FILE_ATTRIBUTE_DIRECTORY);
            let f = dir.join(format!("nativ64-attr-{}", std::process::id()));
            std::fs::write(&f, b"x").unwrap();
            let c = std::ffi::CString::new(f.to_str().unwrap()).unwrap();
            assert_eq!(
                GetFileAttributesA(c.as_ptr() as *const u8),
                FILE_ATTRIBUTE_NORMAL
            );
            std::fs::remove_file(&f).unwrap();
            assert_eq!(
                GetFileAttributesA(c.as_ptr() as *const u8),
                INVALID_FILE_ATTRIBUTES
            );
            assert_eq!(
                crate::api_ms_win_core_errorhandling::GetLastError(),
                ERROR_FILE_NOT_FOUND
            );
            assert!(crate::lookup("kernel32", "GetFileAttributesW").is_some());
        }
    }

    #[test]
    fn open_read_seek_and_missing() {
        unsafe {
            let f = std::env::temp_dir().join(format!("nativ64-cf-{}", std::process::id()));
            std::fs::write(&f, b"0123456789").unwrap();
            let w: Vec<u16> = f.to_str().unwrap().encode_utf16().chain([0]).collect();
            let h = CreateFileW(
                w.as_ptr(),
                0x8000_0000,
                1,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                0,
            );
            assert_ne!(h, INVALID_HANDLE_VALUE);
            let mut size = 0i64;
            assert_eq!(GetFileSizeEx(h, &mut size), 1);
            assert_eq!(size, 10);
            let mut pos = 0i64;
            assert_eq!(SetFilePointerEx(h, -4, &mut pos, 2), 1);
            assert_eq!(pos, 6);
            let mut buf = [0u8; 8];
            let mut n = 0u32;
            assert_eq!(
                ReadFile(h, buf.as_mut_ptr(), 8, &mut n, std::ptr::null()),
                1
            );
            assert_eq!((n, &buf[..4]), (4, &b"6789"[..]));
            assert_eq!(
                ReadFile(h, buf.as_mut_ptr(), 8, &mut n, std::ptr::null()),
                1
            );
            assert_eq!(n, 0);
            assert_eq!(SetFilePointerEx(h, -1, &mut pos, 0), 0);
            assert_eq!(crate::handles::close_handle(h), 1);
            assert_eq!(
                ReadFile(h, buf.as_mut_ptr(), 1, &mut n, std::ptr::null()),
                0
            );
            std::fs::remove_file(&f).unwrap();
            let h = CreateFileW(
                w.as_ptr(),
                0x8000_0000,
                1,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                0,
            );
            assert_eq!(h, INVALID_HANDLE_VALUE);
            assert_eq!(
                crate::api_ms_win_core_errorhandling::GetLastError(),
                ERROR_FILE_NOT_FOUND
            );
        }
    }

    fn w(s: &str) -> Vec<u16> {
        s.encode_utf16().chain([0]).collect()
    }

    unsafe fn last_error() -> u32 {
        crate::api_ms_win_core_errorhandling::GetLastError()
    }

    /// Points the write directory at a fresh temp directory (once per process).
    fn sandbox() -> std::path::PathBuf {
        static ROOT: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
        ROOT.get_or_init(|| {
            let r = std::env::temp_dir().join(format!("nativ64-wr-{}", std::process::id()));
            std::fs::create_dir_all(&r).unwrap();
            std::env::set_var("XDG_DATA_HOME", &r);
            r
        })
        .clone()
    }

    fn name_of(rec: &[u8]) -> String {
        let u: Vec<u16> = rec[44..564]
            .chunks(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .take_while(|&c| c != 0)
            .collect();
        String::from_utf16_lossy(&u)
    }

    #[test]
    fn find_wildcards_dots_and_errors() {
        unsafe {
            let dir = std::env::temp_dir().join(format!("nativ64-find-{}", std::process::id()));
            std::fs::create_dir_all(dir.join("Sub")).unwrap();
            std::fs::write(dir.join("a.TXT"), b"12345").unwrap();
            std::fs::write(dir.join("b.txt"), b"").unwrap();
            std::fs::write(dir.join("noext"), b"").unwrap();
            let search = |pat: &str| -> Result<Vec<String>, u32> {
                let p = w(&format!("{}/{pat}", dir.display()));
                let mut d = [0u8; FIND_DATA_SIZE];
                let h = FindFirstFileW(p.as_ptr(), d.as_mut_ptr());
                if h == INVALID_HANDLE_VALUE {
                    return Err(last_error());
                }
                let mut v = vec![name_of(&d)];
                while FindNextFileW(h, d.as_mut_ptr()) == 1 {
                    v.push(name_of(&d));
                }
                assert_eq!(last_error(), ERROR_NO_MORE_FILES);
                assert_eq!(FindClose(h), 1);
                assert_eq!(FindClose(h), 0);
                assert_eq!(last_error(), ERROR_INVALID_HANDLE);
                v.sort();
                Ok(v)
            };
            assert_eq!(search("*.txt").unwrap(), ["a.TXT", "b.txt"]);
            assert_eq!(search("?.t*").unwrap(), ["a.TXT", "b.txt"]);
            assert_eq!(
                search("*.*").unwrap(),
                [".", "..", "Sub", "a.TXT", "b.txt", "noext"]
            );
            assert_eq!(search("sub").unwrap(), ["Sub"]);
            assert_eq!(search("*.zzz"), Err(ERROR_FILE_NOT_FOUND));
            assert_eq!(search("Sub/../nodir/*"), Err(ERROR_PATH_NOT_FOUND));
            // Record contents: attributes, size, FILETIME in the 21st century.
            let p = w(&format!("{}/a.txt", dir.display()));
            let mut d = [0u8; FIND_DATA_SIZE];
            let h = FindFirstFileW(p.as_ptr(), d.as_mut_ptr());
            assert_ne!(h, INVALID_HANDLE_VALUE);
            assert_eq!(
                u32::from_le_bytes(d[0..4].try_into().unwrap()),
                FILE_ATTRIBUTE_NORMAL
            );
            assert_eq!(u32::from_le_bytes(d[32..36].try_into().unwrap()), 5);
            assert!(u64::from_le_bytes(d[20..28].try_into().unwrap()) > 132_000_000_000_000_000);
            assert_eq!(FindNextFileW(h, d.as_mut_ptr()), 0);
            FindClose(h);
            // Ex: directories only, bad level, NULL path.
            let p = w(&format!("{}/*", dir.display()));
            let h = FindFirstFileExW(p.as_ptr(), 1, d.as_mut_ptr(), 1, std::ptr::null(), 0);
            assert_ne!(h, INVALID_HANDLE_VALUE);
            let mut n = vec![name_of(&d)];
            while FindNextFileW(h, d.as_mut_ptr()) == 1 {
                n.push(name_of(&d));
            }
            n.sort();
            assert_eq!(n, [".", "..", "Sub"]);
            FindClose(h);
            assert_eq!(
                FindFirstFileExW(p.as_ptr(), 2, d.as_mut_ptr(), 0, std::ptr::null(), 0),
                INVALID_HANDLE_VALUE
            );
            assert_eq!(last_error(), crate::handles::ERROR_INVALID_PARAMETER);
            assert_eq!(
                FindFirstFileW(std::ptr::null(), d.as_mut_ptr()),
                INVALID_HANDLE_VALUE
            );
            assert_eq!(FindNextFileW(0x7777_0000, d.as_mut_ptr()), 0);
            assert_eq!(last_error(), ERROR_INVALID_HANDLE);
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }

    #[test]
    fn directory_file_and_move_operations() {
        unsafe {
            let _g = crate::paths::MAIN_MODULE_LOCK
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let root = sandbox();
            let wd = crate::paths::write_dir();
            let (d, a, b) = (w("nd1"), w("nd1\\a.bin"), w("nd1\\b.bin"));
            assert_eq!(CreateDirectoryW(d.as_ptr(), std::ptr::null()), 1);
            assert!(wd.join("nd1").is_dir());
            assert_eq!(CreateDirectoryW(d.as_ptr(), std::ptr::null()), 0);
            assert_eq!(last_error(), ERROR_ALREADY_EXISTS);
            let deep = w("missing\\deep");
            assert_eq!(CreateDirectoryW(deep.as_ptr(), std::ptr::null()), 0);
            assert_eq!(last_error(), ERROR_PATH_NOT_FOUND);
            assert_eq!(CreateDirectoryW(std::ptr::null(), std::ptr::null()), 0);
            assert_eq!(last_error(), crate::handles::ERROR_INVALID_PARAMETER);

            // A file through CreateFileW, then attributes, SetEndOfFile, GetFileAttributesExW.
            let h = CreateFileW(
                a.as_ptr(),
                0xC000_0000,
                0,
                std::ptr::null(),
                CREATE_ALWAYS,
                0,
                0,
            );
            assert_ne!(h, INVALID_HANDLE_VALUE);
            let mut n = 0u32;
            assert_eq!(
                WriteFile(h, b"0123456789".as_ptr(), 10, &mut n, std::ptr::null()),
                1
            );
            let mut pos = 0i64;
            assert_eq!(SetFilePointerEx(h, 4, &mut pos, 0), 1);
            assert_eq!(SetEndOfFile(h), 1);
            assert_eq!(std::fs::metadata(wd.join("nd1/a.bin")).unwrap().len(), 4);
            assert_eq!(SetEndOfFile(0x7777_0000), 0);
            assert_eq!(last_error(), ERROR_INVALID_HANDLE);
            crate::handles::close_handle(h);
            let mut ex = [0u8; 36];
            assert_eq!(GetFileAttributesExW(a.as_ptr(), 0, ex.as_mut_ptr()), 1);
            assert_eq!(
                u32::from_le_bytes(ex[0..4].try_into().unwrap()),
                FILE_ATTRIBUTE_NORMAL
            );
            assert_eq!(u32::from_le_bytes(ex[32..36].try_into().unwrap()), 4);
            assert_eq!(GetFileAttributesExW(a.as_ptr(), 1, ex.as_mut_ptr()), 0);
            assert_eq!(last_error(), crate::handles::ERROR_INVALID_PARAMETER);
            let nf = w("nd1\\none");
            assert_eq!(GetFileAttributesExW(nf.as_ptr(), 0, ex.as_mut_ptr()), 0);
            assert_eq!(last_error(), ERROR_FILE_NOT_FOUND);
            let np = w("nodir\\none");
            assert_eq!(GetFileAttributesExW(np.as_ptr(), 0, ex.as_mut_ptr()), 0);
            assert_eq!(GetFileAttributesW(a.as_ptr()), FILE_ATTRIBUTE_NORMAL);

            // Moves.
            assert_eq!(MoveFileExW(a.as_ptr(), b.as_ptr(), 0), 1);
            assert!(!wd.join("nd1/a.bin").exists() && wd.join("nd1/b.bin").exists());
            assert_eq!(MoveFileExW(a.as_ptr(), b.as_ptr(), 0), 0);
            assert_eq!(last_error(), ERROR_FILE_NOT_FOUND);
            std::fs::write(wd.join("nd1/a.bin"), b"zz").unwrap();
            assert_eq!(MoveFileExW(a.as_ptr(), b.as_ptr(), 0), 0);
            assert_eq!(last_error(), ERROR_ALREADY_EXISTS);
            assert_eq!(
                MoveFileExW(a.as_ptr(), b.as_ptr(), MOVEFILE_REPLACE_EXISTING),
                1
            );
            assert_eq!(std::fs::read(wd.join("nd1/b.bin")).unwrap(), b"zz");
            assert_eq!(
                MoveFileExW(b.as_ptr(), d.as_ptr(), MOVEFILE_REPLACE_EXISTING),
                0
            );
            assert_eq!(last_error(), ERROR_ACCESS_DENIED);
            let bad = w("nodir\\x");
            assert_eq!(MoveFileExW(b.as_ptr(), bad.as_ptr(), 0), 0);
            assert_eq!(last_error(), ERROR_PATH_NOT_FOUND);
            assert_eq!(MoveFileExW(b.as_ptr(), std::ptr::null(), 0), 0);

            // Deletion.
            assert_eq!(DeleteFileW(d.as_ptr()), 0);
            assert_eq!(last_error(), ERROR_ACCESS_DENIED);
            assert_eq!(RemoveDirectoryW(d.as_ptr()), 0);
            assert_eq!(last_error(), ERROR_DIR_NOT_EMPTY);
            assert_eq!(RemoveDirectoryW(b.as_ptr()), 0);
            assert_eq!(last_error(), ERROR_DIRECTORY);
            assert_eq!(DeleteFileW(b.as_ptr()), 1);
            assert_eq!(DeleteFileW(b.as_ptr()), 0);
            assert_eq!(last_error(), ERROR_FILE_NOT_FOUND);
            assert_eq!(DeleteFileW(np.as_ptr()), 0);
            assert_eq!(last_error(), ERROR_PATH_NOT_FOUND);
            assert_eq!(RemoveDirectoryW(d.as_ptr()), 1);
            assert_eq!(RemoveDirectoryW(d.as_ptr()), 0);
            assert_eq!(last_error(), ERROR_FILE_NOT_FOUND);

            // A package file (read-only side) is never modified.
            let pkg_file = root.join("pkg-only.bin");
            std::fs::write(&pkg_file, b"x").unwrap();
            let pkg = w(pkg_file.to_str().unwrap());
            assert_eq!(DeleteFileW(pkg.as_ptr()), 0);
            assert_eq!(last_error(), ERROR_ACCESS_DENIED);
            assert!(pkg_file.exists());
            assert_eq!(MoveFileExW(pkg.as_ptr(), b.as_ptr(), 0), 0);
            assert_eq!(last_error(), ERROR_ACCESS_DENIED);
            assert!(pkg_file.exists());
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn disk_free_space() {
        unsafe {
            let (mut a, mut t, mut f) = (0u64, 0u64, 0u64);
            let tmp = w(&std::env::temp_dir().to_string_lossy());
            assert_eq!(GetDiskFreeSpaceExW(tmp.as_ptr(), &mut a, &mut t, &mut f), 1);
            assert!(t > 0 && a <= t && f <= t);
            assert_eq!(
                GetDiskFreeSpaceExW(
                    std::ptr::null(),
                    &mut a,
                    std::ptr::null_mut(),
                    std::ptr::null_mut()
                ),
                1
            );
            let bad = w("/nonexistent-nativ64-dir");
            assert_eq!(GetDiskFreeSpaceExW(bad.as_ptr(), &mut a, &mut t, &mut f), 0);
            assert_eq!(last_error(), ERROR_PATH_NOT_FOUND);
            assert!(crate::lookup("kernel32", "FindFirstFileW").is_some());
        }
    }
}
