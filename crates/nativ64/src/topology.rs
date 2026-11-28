//! The host's processor topology (cores, SMT siblings, caches, packages), read once from
//! `/sys/devices/system/cpu` and serialized in the two layouts Windows uses:
//! `SYSTEM_LOGICAL_PROCESSOR_INFORMATION` (fixed 32-byte entries) and
//! `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` (variable-size records).
//!
//! Only the logical processors this process may run on are reported, capped at 64 (one
//! processor group). One NUMA node and one group cover all of them.

use std::path::Path;
use std::sync::OnceLock;

pub(crate) const RELATION_PROCESSOR_CORE: u32 = 0;
pub(crate) const RELATION_NUMA_NODE: u32 = 1;
pub(crate) const RELATION_CACHE: u32 = 2;
pub(crate) const RELATION_PROCESSOR_PACKAGE: u32 = 3;
pub(crate) const RELATION_GROUP: u32 = 4;
pub(crate) const RELATION_PROCESSOR_DIE: u32 = 5;
pub(crate) const RELATION_NUMA_NODE_EX: u32 = 6;
pub(crate) const RELATION_PROCESSOR_MODULE: u32 = 7;
pub(crate) const RELATION_ALL: u32 = 0xffff;
/// `PROCESSOR_RELATIONSHIP.Flags`: the core runs more than one logical processor.
const LTP_PC_SMT: u8 = 1;
/// `PROCESSOR_CACHE_TYPE`.
const CACHE_UNIFIED: u32 = 0;
const CACHE_INSTRUCTION: u32 = 1;
const CACHE_DATA: u32 = 2;
/// `CACHE_DESCRIPTOR.Associativity` of a fully associative cache.
const CACHE_FULLY_ASSOCIATIVE: u8 = 0xff;
/// Size of one `SYSTEM_LOGICAL_PROCESSOR_INFORMATION`.
pub(crate) const LPI_SIZE: usize = 32;
/// The largest `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` record written (the group one).
const MAX_EX: usize = 80;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cache {
    pub level: u8,
    pub kind: u32,
    pub associativity: u8,
    pub line: u16,
    pub size: u32,
    pub mask: u64,
}

#[derive(Debug, Default, PartialEq)]
pub(crate) struct Topology {
    /// Every reported logical processor.
    pub all: u64,
    /// One mask per physical core (more than one bit with SMT).
    pub cores: Vec<u64>,
    /// `EfficiencyClass` of each core in `cores` (higher is faster; 0 on a uniform CPU).
    pub efficiency: Vec<u8>,
    pub caches: Vec<Cache>,
    pub packages: Vec<u64>,
}

/// The logical processors this process may run on, as a mask (at most 64: one processor group).
pub(crate) fn processor_mask() -> u64 {
    // SAFETY: a zeroed cpu_set_t is valid, and the call fills it.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // SAFETY: `set` is a valid out pointer of the size given.
    let ok = unsafe { libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut set) } == 0;
    let mut mask = 0u64;
    if ok {
        for i in 0..64 {
            // SAFETY: `i` is below CPU_SETSIZE.
            if unsafe { libc::CPU_ISSET(i, &set) } {
                mask |= 1 << i;
            }
        }
    }
    if mask == 0 {
        1
    } else {
        mask
    }
}

/// The host topology, read on first use.
pub(crate) fn topology() -> &'static Topology {
    static TOPOLOGY: OnceLock<Topology> = OnceLock::new();
    TOPOLOGY.get_or_init(|| read(Path::new("/sys/devices/system/cpu"), processor_mask()))
}

fn read_trimmed(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_owned())
}

/// A sysfs CPU list (`0-3,8,10-11`) as a mask of the CPUs below 64.
fn parse_list(s: &str) -> u64 {
    let mut mask = 0u64;
    for part in s.split(',').filter(|p| !p.is_empty()) {
        let (lo, hi) = match part.split_once('-') {
            Some((a, b)) => (a.parse::<u32>(), b.parse::<u32>()),
            None => (part.parse(), part.parse()),
        };
        if let (Ok(lo), Ok(hi)) = (lo, hi) {
            for cpu in lo..=hi.min(63) {
                mask |= 1 << cpu;
            }
        }
    }
    mask
}

/// A sysfs cache size (`48K`, `2048K`, `32M`) in bytes.
fn parse_size(s: &str) -> Option<u32> {
    let (digits, unit) = match s.as_bytes().last()? {
        b'K' => (&s[..s.len() - 1], 1 << 10),
        b'M' => (&s[..s.len() - 1], 1 << 20),
        b'G' => (&s[..s.len() - 1], 1 << 30),
        _ => (s, 1),
    };
    digits.parse::<u32>().ok()?.checked_mul(unit)
}

fn read_list(path: &Path) -> Option<u64> {
    read_trimmed(path).map(|s| parse_list(&s))
}

fn read_cache(dir: &Path, all: u64) -> Option<Cache> {
    let level = read_trimmed(&dir.join("level"))?.parse().ok()?;
    let kind = match read_trimmed(&dir.join("type"))?.as_str() {
        "Data" => CACHE_DATA,
        "Instruction" => CACHE_INSTRUCTION,
        "Unified" => CACHE_UNIFIED,
        _ => return None,
    };
    let size = parse_size(&read_trimmed(&dir.join("size"))?)?;
    let line = read_trimmed(&dir.join("coherency_line_size"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    // sysfs reports 0 ways for a fully associative cache.
    let associativity = match read_trimmed(&dir.join("ways_of_associativity"))
        .and_then(|s| s.parse::<u32>().ok())
    {
        Some(0) | None => CACHE_FULLY_ASSOCIATIVE,
        Some(w) => w.min(0xfe) as u8,
    };
    let mask = read_list(&dir.join("shared_cpu_list"))? & all;
    Some(Cache {
        level,
        kind,
        associativity,
        line,
        size,
        mask,
    })
}

/// The EfficiencyClass (0 or 1, higher is faster) of each CPU in `all`. On a hybrid Intel CPU
/// the P cores (`devices/cpu_core/cpus`) are 1 and the E cores (`devices/cpu_atom/cpus`) 0;
/// otherwise the CPUs with the highest `cpu_capacity` are 1, unless all are equal (then 0).
fn efficiency_classes(root: &Path, devices: Option<&Path>, all: u64) -> u64 {
    let listed = |name: &str| {
        devices
            .and_then(|d| read_list(&d.join(name).join("cpus")))
            .unwrap_or(0)
    };
    let (perf, atom) = (listed("cpu_core"), listed("cpu_atom"));
    if perf | atom != 0 {
        return perf & all;
    }
    let caps: Vec<(u32, u32)> = (0..64)
        .filter(|c| all >> c & 1 != 0)
        .filter_map(|c| {
            read_trimmed(&root.join(format!("cpu{c}/cpu_capacity")))
                .and_then(|s| s.parse().ok())
                .map(|cap| (c, cap))
        })
        .collect();
    let top = caps.iter().map(|c| c.1).max().unwrap_or(0);
    if caps.iter().all(|c| c.1 == top) {
        return 0;
    }
    caps.iter()
        .filter(|c| c.1 == top)
        .fold(0, |m, c| m | 1 << c.0)
}

/// Reads the topology of the CPUs in `all` from a sysfs CPU directory
/// (`/sys/devices/system/cpu`; the hybrid lists live beside it, in `/sys/devices`).
pub(crate) fn read(root: &Path, all: u64) -> Topology {
    let devices = root.parent().and_then(Path::parent);
    read_with(root, devices, all)
}

/// [`read`] with the `devices` directory given. A CPU whose topology is missing counts as its
/// own core; a missing package means one package for everything.
fn read_with(root: &Path, devices: Option<&Path>, all: u64) -> Topology {
    let fast = efficiency_classes(root, devices, all);
    let mut t = Topology {
        all,
        ..Default::default()
    };
    for cpu in (0..64).filter(|c| all >> c & 1 != 0) {
        let dir = root.join(format!("cpu{cpu}"));
        let topo = dir.join("topology");
        let core = read_list(&topo.join("core_cpus_list"))
            .or_else(|| read_list(&topo.join("thread_siblings_list")))
            .map(|m| m & all | 1 << cpu)
            .unwrap_or(1 << cpu);
        if !t.cores.contains(&core) {
            t.cores.push(core);
            // A core's class is that of its first logical processor.
            t.efficiency.push((fast >> cpu & 1) as u8);
        }
        let package = read_list(&topo.join("package_cpus_list"))
            .or_else(|| read_list(&topo.join("core_siblings_list")))
            .map(|m| m & all | 1 << cpu);
        if let Some(p) = package.filter(|p| !t.packages.contains(p)) {
            t.packages.push(p);
        }
        for index in 0.. {
            let cache_dir = dir.join(format!("cache/index{index}"));
            if !cache_dir.exists() {
                break;
            }
            if let Some(c) = read_cache(&cache_dir, all).filter(|c| !t.caches.contains(c)) {
                t.caches.push(c);
            }
        }
    }
    // A package whose CPUs were already claimed by another (a malformed tree) is dropped.
    let mut seen = 0u64;
    t.packages.retain(|&p| {
        let fresh = p & seen == 0;
        seen |= p;
        fresh
    });
    if t.packages.is_empty() || seen != all {
        t.packages = vec![all];
    }
    t.caches
        .sort_by_key(|c| (c.level, c.mask.trailing_zeros(), c.kind));
    t
}

/// `GROUP_AFFINITY` (16 bytes) for group 0 at `out`.
fn group_affinity(out: &mut [u8], mask: u64) {
    out[..8].copy_from_slice(&mask.to_le_bytes());
}

impl Topology {
    /// Calls `f` with each `SYSTEM_LOGICAL_PROCESSOR_INFORMATION` entry: cores, caches,
    /// packages, then the NUMA node.
    pub(crate) fn for_each_legacy(&self, mut f: impl FnMut(&[u8; LPI_SIZE])) {
        let entry = |mask: u64, relation: u32| {
            let mut e = [0u8; LPI_SIZE];
            e[..8].copy_from_slice(&mask.to_le_bytes());
            e[8..12].copy_from_slice(&relation.to_le_bytes());
            e
        };
        for &core in &self.cores {
            let mut e = entry(core, RELATION_PROCESSOR_CORE);
            e[16] = if core.count_ones() > 1 { LTP_PC_SMT } else { 0 };
            f(&e);
        }
        for c in &self.caches {
            let mut e = entry(c.mask, RELATION_CACHE);
            e[16] = c.level;
            e[17] = c.associativity;
            e[18..20].copy_from_slice(&c.line.to_le_bytes());
            e[20..24].copy_from_slice(&c.size.to_le_bytes());
            e[24..28].copy_from_slice(&c.kind.to_le_bytes());
            f(&e);
        }
        for &p in &self.packages {
            f(&entry(p, RELATION_PROCESSOR_PACKAGE));
        }
        f(&entry(self.all, RELATION_NUMA_NODE));
    }

    /// Calls `f` with each `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` record of `relation`
    /// (`RELATION_ALL` for every kind). The caller checks that `relation` is known.
    pub(crate) fn for_each_ex(&self, relation: u32, mut f: impl FnMut(&[u8])) {
        let wants = |r: u32| relation == RELATION_ALL || relation == r;
        let mut record = |relation: u32, size: usize, fill: &dyn Fn(&mut [u8])| {
            let mut r = [0u8; MAX_EX];
            r[..4].copy_from_slice(&relation.to_le_bytes());
            r[4..8].copy_from_slice(&(size as u32).to_le_bytes());
            fill(&mut r[8..size]);
            f(&r[..size]);
        };
        // PROCESSOR_RELATIONSHIP: Flags, EfficiencyClass, Reserved[20], GroupCount, GroupMask.
        let processor = |mask: u64, flags: u8, class: u8| {
            move |b: &mut [u8]| {
                b[0] = flags;
                b[1] = class;
                b[22..24].copy_from_slice(&1u16.to_le_bytes());
                group_affinity(&mut b[24..40], mask);
            }
        };
        if wants(RELATION_PROCESSOR_CORE) {
            for (&core, &class) in self.cores.iter().zip(&self.efficiency) {
                let flags = if core.count_ones() > 1 { LTP_PC_SMT } else { 0 };
                record(RELATION_PROCESSOR_CORE, 48, &processor(core, flags, class));
            }
        }
        if wants(RELATION_CACHE) {
            // CACHE_RELATIONSHIP: Level, Associativity, LineSize, CacheSize, Type,
            // Reserved[18], GroupCount, GroupMask.
            for c in &self.caches {
                record(RELATION_CACHE, 56, &|b: &mut [u8]| {
                    b[0] = c.level;
                    b[1] = c.associativity;
                    b[2..4].copy_from_slice(&c.line.to_le_bytes());
                    b[4..8].copy_from_slice(&c.size.to_le_bytes());
                    b[8..12].copy_from_slice(&c.kind.to_le_bytes());
                    b[30..32].copy_from_slice(&1u16.to_le_bytes());
                    group_affinity(&mut b[32..48], c.mask);
                });
            }
        }
        if wants(RELATION_PROCESSOR_PACKAGE) {
            for &p in &self.packages {
                record(RELATION_PROCESSOR_PACKAGE, 48, &processor(p, 0, 0));
            }
        }
        if wants(RELATION_NUMA_NODE) || relation == RELATION_NUMA_NODE_EX {
            // NUMA_NODE_RELATIONSHIP: NodeNumber, Reserved[18], GroupCount, GroupMask.
            record(RELATION_NUMA_NODE, 48, &|b: &mut [u8]| {
                b[22..24].copy_from_slice(&1u16.to_le_bytes());
                group_affinity(&mut b[24..40], self.all);
            });
        }
        if wants(RELATION_GROUP) {
            // GROUP_RELATIONSHIP: MaximumGroupCount, ActiveGroupCount, Reserved[20], then one
            // PROCESSOR_GROUP_INFO: MaximumProcessorCount, ActiveProcessorCount,
            // Reserved[38], ActiveProcessorMask.
            record(RELATION_GROUP, 80, &|b: &mut [u8]| {
                b[0..2].copy_from_slice(&1u16.to_le_bytes());
                b[2..4].copy_from_slice(&1u16.to_le_bytes());
                let n = self.all.count_ones() as u8;
                b[24] = n;
                b[25] = n;
                b[64..72].copy_from_slice(&self.all.to_le_bytes());
            });
        }
        // Linux has no die or module level that sysfs gives us portably, so the mapping is
        // fixed: one die per package (the same mask) and one module per core (the core's mask).
        if wants(RELATION_PROCESSOR_DIE) {
            for &p in &self.packages {
                record(RELATION_PROCESSOR_DIE, 48, &processor(p, 0, 0));
            }
        }
        if wants(RELATION_PROCESSOR_MODULE) {
            for (&core, &class) in self.cores.iter().zip(&self.efficiency) {
                record(RELATION_PROCESSOR_MODULE, 48, &processor(core, 0, class));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn lists_and_sizes() {
        assert_eq!(parse_list("0-3,8,10-11"), 0b1101_0000_1111);
        assert_eq!(parse_list("63-70"), 1 << 63);
        assert_eq!(parse_list(""), 0);
        assert_eq!(parse_list("x"), 0);
        assert_eq!(parse_size("48K"), Some(48 << 10));
        assert_eq!(parse_size("32M"), Some(32 << 20));
        assert_eq!(parse_size("512"), Some(512));
        assert_eq!(parse_size("8G"), None);
        assert_eq!(parse_size(""), None);
    }

    fn write(path: PathBuf, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    /// A fake sysfs tree: 2 cores x 2 SMT threads in one package, private L1d/L2, shared L3.
    fn fake_tree(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("nativ64-topo-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for cpu in 0..4u32 {
            let dir = root.join(format!("cpu{cpu}"));
            let siblings = if cpu < 2 { "0-1" } else { "2-3" };
            write(dir.join("topology/core_cpus_list"), siblings);
            write(dir.join("topology/package_cpus_list"), "0-3\n");
            let caches = [
                ("1", "Data", "48K", "12", siblings),
                ("2", "Unified", "1280K", "0", siblings),
                ("3", "Unified", "12M", "12", "0-3"),
            ];
            for (i, (level, kind, size, ways, shared)) in caches.iter().enumerate() {
                let c = dir.join(format!("cache/index{i}"));
                write(c.join("level"), level);
                write(c.join("type"), kind);
                write(c.join("size"), size);
                write(c.join("coherency_line_size"), "64");
                write(c.join("ways_of_associativity"), ways);
                write(c.join("shared_cpu_list"), shared);
            }
        }
        root
    }

    #[test]
    fn reads_smt_cores_caches_and_package() {
        let root = fake_tree("smt");
        let t = read(&root, 0b1111);
        assert_eq!(t.cores, vec![0b0011, 0b1100]);
        assert_eq!(t.packages, vec![0b1111]);
        assert_eq!(t.caches.len(), 5);
        let l3 = t.caches.last().unwrap();
        assert_eq!(
            (l3.level, l3.kind, l3.size, l3.mask),
            (3, CACHE_UNIFIED, 12 << 20, 0b1111)
        );
        let l2 = t.caches.iter().find(|c| c.level == 2).unwrap();
        assert_eq!(l2.associativity, CACHE_FULLY_ASSOCIATIVE);
        // An affinity mask restricts every record.
        let t = read(&root, 0b0101);
        assert_eq!(t.cores, vec![0b0001, 0b0100]);
        assert_eq!(t.packages, vec![0b0101]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn efficiency_from_hybrid_lists_then_capacity_then_zero() {
        let root = fake_tree("eff");
        let devices = root.join("devices");
        // Uniform: every core is class 0.
        assert_eq!(read_with(&root, None, 0b1111).efficiency, vec![0, 0]);
        // cpu_capacity: the higher one is class 1.
        for cpu in 0..4 {
            let cap = if cpu < 2 { "1024" } else { "512" };
            write(root.join(format!("cpu{cpu}/cpu_capacity")), cap);
        }
        assert_eq!(read_with(&root, None, 0b1111).efficiency, vec![1, 0]);
        // Hybrid lists win over capacity: here cpus 2-3 are the P cores.
        write(devices.join("cpu_core/cpus"), "2-3\n");
        write(devices.join("cpu_atom/cpus"), "0-1\n");
        let t = read_with(&root, Some(&devices), 0b1111);
        assert_eq!(t.efficiency, vec![0, 1]);
        let mut classes = Vec::new();
        t.for_each_ex(RELATION_PROCESSOR_CORE, |r| classes.push(r[9]));
        assert_eq!(classes, [0, 1]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_tree_falls_back_to_one_core_per_cpu() {
        let t = read(Path::new("/nonexistent-sysfs"), 0b111);
        assert_eq!(t.cores, vec![1, 2, 4]);
        assert_eq!(t.packages, vec![0b111]);
        assert!(t.caches.is_empty());
    }

    #[test]
    fn ex_records_have_the_right_kinds_and_sizes() {
        let root = fake_tree("ex");
        let t = read(&root, 0b1111);
        std::fs::remove_dir_all(root).unwrap();
        let mut seen = Vec::new();
        t.for_each_ex(RELATION_ALL, |r| {
            let kind = u32::from_le_bytes(r[..4].try_into().unwrap());
            let size = u32::from_le_bytes(r[4..8].try_into().unwrap()) as usize;
            assert_eq!(size, r.len());
            seen.push((kind, size, r[8]));
        });
        let kinds: Vec<_> = seen.iter().map(|s| s.0).collect();
        assert_eq!(kinds, [0, 0, 2, 2, 2, 2, 2, 3, 1, 4, 5, 7, 7]);
        // Both cores are SMT; the package is not flagged.
        assert_eq!(seen[0].2, LTP_PC_SMT);
        assert_eq!(seen[7], (RELATION_PROCESSOR_PACKAGE, 48, 0));
        let mut packages = 0;
        t.for_each_ex(RELATION_PROCESSOR_PACKAGE, |r| {
            assert_eq!(&r[..4], &3u32.to_le_bytes());
            assert_eq!(u64::from_le_bytes(r[32..40].try_into().unwrap()), 0b1111);
            packages += 1;
        });
        assert_eq!(packages, 1);
        t.for_each_ex(RELATION_GROUP, |r| {
            assert_eq!((r[32], r[33]), (4, 4));
            assert_eq!(u64::from_le_bytes(r[72..80].try_into().unwrap()), 0b1111);
        });
    }
}
