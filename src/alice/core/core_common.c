// (C) Copyright 2026 Matyas Constans
// Licensed under the MIT License (https://opensource.org/license/mit/)

// NOTE(cmat): Julian Day Number to Gregorian calendar conversion algorithm.
// TODO(cmat): Double check implementation.
function Local_Time local_time_from_unix_time(U64 unix_seconds, U64 unix_microseconds) {
  // NOTE(cmat): Derived math, it works.
  U64 until_today_seconds       = (unix_seconds % (3600LL * 24));
  U64 until_today_microseconds  = unix_microseconds;
  U64 until_1970_days           = unix_seconds / (3600LL * 24);
  U64 t                         = until_1970_days;
  U64 a                         = (4 * t + 102032) / 146097 + 15;
  U64 b                         = t + 2442113 + a - (a / 4);
  U64 c                         = (20 * b - 2442) / 7305;
  U64 d                         = b - 365 * c - (c / 4);
  U64 e                         = d * 1000 / 30601;
  U64 f                         = d - e * 30 - e * 601 / 1000;

  // NOTE(cmat): Handle January and February. Counted as month 13 & 14 of the previous year.
  if (e <= 13) {
    c -= 4716;
    e -= 1;
  } else {
    c -= 4715;
    e -= 13;
  }

  U32 current_year  = (U32)c;
  U08 current_month = (U08)e;
  U08 current_day   = (U08)f;

  Local_Time time = {
    .year         = current_year,
    .month        = current_month,
    .day          = current_day,
    .hours        = (U08)(until_today_seconds / 3600),
    .minutes      = (U08)((until_today_seconds / 60) % 60),
    .seconds      = (U08)(until_today_seconds % 60),
    .microseconds = (U16)(until_today_microseconds),
  };

  return time;
}
