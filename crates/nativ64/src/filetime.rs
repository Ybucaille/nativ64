//! `FILETIME` / `SYSTEMTIME` conversions shared by the time and file modules.
//!
//! A `FILETIME` is a `u64` of 100 ns intervals since 1601-01-01 UTC. A `SYSTEMTIME` is eight
//! `u16`: year, month, day of week, day, hour, minute, second, millisecond. The calendar is the
//! proleptic Gregorian one in integer arithmetic (no host C library, no allocation), over the
//! range Windows accepts: years 1601 through 30827, where a `FILETIME` reaches 2^63.

/// 100 ns intervals between 1601-01-01 and 1970-01-01.
pub(crate) const UNIX_EPOCH_AS_FILETIME: u64 = 116_444_736_000_000_000;

/// 100 ns intervals in one second, one millisecond and one day.
const TICKS_PER_SECOND: u64 = 10_000_000;
const TICKS_PER_MS: u64 = 10_000;
const TICKS_PER_DAY: u64 = 86_400 * TICKS_PER_SECOND;
/// Days from 1601-01-01 to 1970-01-01.
const DAYS_1601_TO_1970: i64 = 134_774;
/// A `FILETIME` must stay below this (Windows treats it as a signed 64-bit count).
const FILETIME_LIMIT: u64 = 1 << 63;

const FIRST_YEAR: u16 = 1601;
const LAST_YEAR: u16 = 30827;

/// A `SYSTEMTIME` as its eight fields.
pub(crate) type SystemTime = [u16; 8];

/// Days since 1970-01-01 of a civil date (month 1-12, day 1-31).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = year - i64::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The civil date (year, month, day) of a day count since 1970-01-01.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(month <= 2), month, day)
}

fn is_leap(year: u16) -> bool {
    year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400))
}

fn days_in_month(year: u16, month: u16) -> u16 {
    match month {
        2 if is_leap(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// The UTC `SYSTEMTIME` of a `FILETIME`, or `None` from 2^63 on (Windows rejects those).
pub(crate) fn to_systemtime(ft: u64) -> Option<SystemTime> {
    if ft >= FILETIME_LIMIT {
        return None;
    }
    let days = (ft / TICKS_PER_DAY) as i64 - DAYS_1601_TO_1970;
    let in_day = ft % TICKS_PER_DAY;
    let (year, month, day) = civil_from_days(days);
    let secs = in_day / TICKS_PER_SECOND;
    // 1601-01-01 was a Monday; Sunday is 0.
    let weekday = ((ft / TICKS_PER_DAY) + 1) % 7;
    Some([
        year as u16,
        month as u16,
        weekday as u16,
        day as u16,
        (secs / 3600) as u16,
        (secs / 60 % 60) as u16,
        (secs % 60) as u16,
        (in_day % TICKS_PER_SECOND / TICKS_PER_MS) as u16,
    ])
}

/// The `FILETIME` of a UTC `SYSTEMTIME` (its day of week is ignored, as on Windows), or `None`
/// for a field out of range, a date that does not exist (February 30) (the year range ends before 2^63 ticks does).
pub(crate) fn from_systemtime(st: &SystemTime) -> Option<u64> {
    let [year, month, _, day, hour, minute, second, millis] = *st;
    if !(FIRST_YEAR..=LAST_YEAR).contains(&year)
        || !(1..=12).contains(&month)
        || day == 0
        || day > days_in_month(year, month)
        || hour > 23
        || minute > 59
        || second > 59
        || millis > 999
    {
        return None;
    }
    let days =
        days_from_civil(i64::from(year), i64::from(month), i64::from(day)) + DAYS_1601_TO_1970;
    let secs = u64::from(hour) * 3600 + u64::from(minute) * 60 + u64::from(second);
    let ft =
        (days as u64) * TICKS_PER_DAY + secs * TICKS_PER_SECOND + u64::from(millis) * TICKS_PER_MS;
    (ft < FILETIME_LIMIT).then_some(ft)
}

/// The offset of local time from UTC now, in seconds east of UTC (0 when the host cannot say).
/// Windows applies the current bias, daylight saving included, to any `FILETIME` it converts
/// between UTC and local time, whatever the date.
pub(crate) fn local_offset_now() -> i64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid out pointer.
    unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut now) };
    // SAFETY: a zeroed `tm` is a valid value, and `localtime_r` fills it.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: both pointers are valid for the call.
    if unsafe { libc::localtime_r(&now.tv_sec, &mut tm) }.is_null() {
        return 0;
    }
    tm.tm_gmtoff
}

/// `ft` moved by `offset_secs`, or `None` when it would leave `0..2^63`.
pub(crate) fn shift(ft: u64, offset_secs: i64) -> Option<u64> {
    let ticks = offset_secs.checked_mul(TICKS_PER_SECOND as i64)?;
    let moved = ft.checked_add_signed(ticks)?;
    (ft < FILETIME_LIMIT && moved < FILETIME_LIMIT).then_some(moved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_instants() {
        assert_eq!(to_systemtime(0), Some([1601, 1, 1, 1, 0, 0, 0, 0])); // a Monday
        assert_eq!(
            to_systemtime(UNIX_EPOCH_AS_FILETIME),
            Some([1970, 1, 4, 1, 0, 0, 0, 0]) // a Thursday
        );
        // 2020-02-29 12:34:56.789 UTC, a Saturday and a leap day.
        let ft = UNIX_EPOCH_AS_FILETIME + 1_582_979_696 * TICKS_PER_SECOND + 7_890_000;
        let st = [2020, 2, 6, 29, 12, 34, 56, 789];
        assert_eq!(to_systemtime(ft), Some(st));
        assert_eq!(from_systemtime(&st), Some(ft));
        // The last representable instant is in 30828, past what SYSTEMTIME input accepts.
        let last = to_systemtime(FILETIME_LIMIT - 1).unwrap();
        assert_eq!(last[0], 30828);
        assert_eq!(to_systemtime(FILETIME_LIMIT), None);
    }

    #[test]
    fn round_trips_every_day_of_a_few_centuries() {
        let mut ft = 0u64;
        while ft < UNIX_EPOCH_AS_FILETIME * 2 {
            let st = to_systemtime(ft).unwrap();
            assert_eq!(from_systemtime(&st), Some(ft - ft % TICKS_PER_MS));
            // Every field is in range, and the weekday advances with the day.
            assert!(
                st[1] >= 1 && st[1] <= 12 && st[3] >= 1 && st[3] <= days_in_month(st[0], st[1])
            );
            ft += TICKS_PER_DAY * 37 + 123_456_789;
        }
    }

    #[test]
    fn leap_years_and_month_ends() {
        let ok = |y, m, d| from_systemtime(&[y, m, 0, d, 23, 59, 59, 999]).is_some();
        assert!(ok(2000, 2, 29));
        assert!(!ok(1900, 2, 29));
        assert!(ok(2024, 2, 29));
        assert!(!ok(2023, 2, 29));
        assert!(ok(2023, 12, 31) && !ok(2023, 4, 31));
    }

    #[test]
    fn invalid_systemtime_is_rejected() {
        for st in [
            [2021u16, 2, 0, 30, 0, 0, 0, 0],
            [2021, 13, 0, 1, 0, 0, 0, 0],
            [2021, 0, 0, 1, 0, 0, 0, 0],
            [2021, 1, 0, 1, 24, 0, 0, 0],
            [2021, 1, 0, 1, 0, 60, 0, 0],
            [2021, 1, 0, 1, 0, 0, 60, 0],
            [2021, 1, 0, 1, 0, 0, 0, 1000],
            [2021, 1, 0, 0, 0, 0, 0, 0],
            [1600, 12, 0, 31, 0, 0, 0, 0],
            [30829, 1, 0, 1, 0, 0, 0, 0],
        ] {
            assert_eq!(from_systemtime(&st), None, "{st:?}");
        }
        // The day of week is not checked.
        assert!(from_systemtime(&[2021, 1, 6, 1, 0, 0, 0, 0]).is_some());
    }

    #[test]
    fn shifting_stays_in_range() {
        assert_eq!(
            shift(UNIX_EPOCH_AS_FILETIME, 3600),
            Some(UNIX_EPOCH_AS_FILETIME + 36_000_000_000)
        );
        assert_eq!(
            shift(UNIX_EPOCH_AS_FILETIME, -3600),
            Some(UNIX_EPOCH_AS_FILETIME - 36_000_000_000)
        );
        assert_eq!(shift(0, -1), None);
        assert_eq!(shift(FILETIME_LIMIT - 1, 1), None);
        assert_eq!(shift(FILETIME_LIMIT, 0), None);
        assert_eq!(shift(5, i64::MAX), None);
    }

    #[test]
    fn local_offset_is_a_sane_zone() {
        assert!(local_offset_now().abs() <= 15 * 3600);
    }
}
