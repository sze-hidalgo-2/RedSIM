#include <math.h>

#define RS_METEO_PI 3.14159265358979323846

// ------------------------------------------------------------
// #-- CSV Loaders

// NOTE(cmat): Peeks (without consuming) whether the field at the scan's current position is
// - empty - i.e. the next non-space/tab byte is the ';' delimiter, a line ending, or EOF.
// - Deliberately does NOT use scan_skip_whitespace/scan_char, since those treat '\r'/'\n' as
// - ordinary whitespace: calling scan_f64 on an empty *last* field of a line would silently
// - skip across the line ending and start parsing the *next* row's data as this field's
// - value, desynchronizing every row after it (and eventually crashing str08_slice once the
// - scan runs off the end of the file). Checking this first, before ever calling scan_f64,
// - avoids that path entirely.
function B32 rs_meteo_field_is_empty(Scan *scan) {
  U64 at = scan->at;
  while (at < scan->stream.len && (scan->stream.txt[at] == ' ' || scan->stream.txt[at] == '\t')) {
    at += 1;
  }

  U08 c = (at < scan->stream.len) ? scan->stream.txt[at] : 0;
  B32 result = (c == ';' || c == '\r' || c == '\n' || c == 0);
  return result;
}

// NOTE(cmat): Attempts to parse one F64 field (e.g. a single hourly reading in a meteo CSV
// - row). Station data occasionally has missing readings, which show up as an empty CSV cell.
// - On a missing or malformed field this logs a warning with the offending line/column,
// - reports the field as invalid via `out_valid`, and returns 0.0 as a placeholder - the
// - caller fills the placeholder in properly afterwards via linear/circular interpolation
// - against the table's other readings, so a handful of bad/missing readings degrade
// - gracefully instead of aborting the whole table load.
function F64 rs_meteo_scan_f64_optional(Scan *scan, B32 *out_valid, Str08 file_path, Str08 field_name) {
  *out_valid = 1;

  if (rs_meteo_field_is_empty(scan)) {
    *out_valid = 0;
    log_warning("%S: %llu:%llu: missing value for %S, will interpolate", file_path, scan->line_at, scan->char_at, field_name);
    return 0.0;
  }

  Scan_Error *error_first_before = scan->error_first;
  Scan_Error *error_last_before  = scan->error_last;
  U64         line_at            = scan->line_at;
  U64         char_at            = scan->char_at;

  F64 result = scan_f64(scan);

  if (scan->error_last != error_last_before) {
    // NOTE(cmat): scan_f64's failure branch pushes exactly one error - roll it back off
    // - the list so it doesn't get treated as a real parse failure by the caller.
    scan->error_first = error_first_before;
    scan->error_last  = error_last_before;
    if (scan->error_last) {
      scan->error_last->next = 0;
    }

    *out_valid = 0;
    log_warning("%S: %llu:%llu: invalid value for %S, will interpolate", file_path, line_at, char_at, field_name);
    result = 0.0;
  }

  return result;
}

// NOTE(cmat): Fills gaps (runs of `valid[i] == 0`) in a table's `global_radiation_w_m2`
// - series by linearly interpolating between the nearest valid reading before and after the
// - gap - treating the whole table as one flat chronological sequence of `len * 24` hourly
// - values, so a gap that straddles midnight (last hour of one day, first hours of the next)
// - is interpolated correctly instead of per-day in isolation. A gap with no valid reading on
// - one side (the very start/end of the record) holds the nearest available value instead.
function void rs_meteo_radiation_interpolate_gaps(RS_Meteo_Radiation_Table *table, B32 *valid, Str08 file_path) {
  U64 count  = table->len * 24;
  U64 filled = 0;
  U64 i      = 0;

  while (i < count) {
    if (valid[i]) { i += 1; continue; }

    U64 gap_start = i;
    U64 gap_end   = i;
    while (gap_end < count && !valid[gap_end]) { gap_end += 1; }

    B32 has_left  = gap_start > 0;
    B32 has_right = gap_end < count;
    F64 left_val  = has_left  ? table->dat[(gap_start - 1) / 24].global_radiation_w_m2[(gap_start - 1) % 24] : 0.0;
    F64 right_val = has_right ? table->dat[gap_end / 24].global_radiation_w_m2[gap_end % 24]                 : 0.0;

    for (U64 k = gap_start; k < gap_end; k += 1) {
      F64 v;
      if      (has_left && has_right) { F64 t = (F64)(k - gap_start + 1) / (F64)(gap_end - gap_start + 1); v = left_val + (right_val - left_val) * t; }
      else if (has_left)              { v = left_val; }
      else if (has_right)             { v = right_val; }
      else                            { v = 0.0; }

      table->dat[k / 24].global_radiation_w_m2[k % 24] = v;
    }

    filled += (gap_end - gap_start);
    i       = gap_end;
  }

  if (filled > 0) {
    log_warning("%S: linearly interpolated %llu missing global_radiation_w_m2 reading(s)", file_path, filled);
  }
}

// NOTE(cmat): Same gap-filling scheme as rs_meteo_radiation_interpolate_gaps, applied to
// - `temperature_c`.
function void rs_meteo_temperature_interpolate_gaps(RS_Meteo_Temperature_Table *table, B32 *valid, Str08 file_path) {
  U64 count  = table->len * 24;
  U64 filled = 0;
  U64 i      = 0;

  while (i < count) {
    if (valid[i]) { i += 1; continue; }

    U64 gap_start = i;
    U64 gap_end   = i;
    while (gap_end < count && !valid[gap_end]) { gap_end += 1; }

    B32 has_left  = gap_start > 0;
    B32 has_right = gap_end < count;
    F64 left_val  = has_left  ? table->dat[(gap_start - 1) / 24].temperature_c[(gap_start - 1) % 24] : 0.0;
    F64 right_val = has_right ? table->dat[gap_end / 24].temperature_c[gap_end % 24]                 : 0.0;

    for (U64 k = gap_start; k < gap_end; k += 1) {
      F64 v;
      if      (has_left && has_right) { F64 t = (F64)(k - gap_start + 1) / (F64)(gap_end - gap_start + 1); v = left_val + (right_val - left_val) * t; }
      else if (has_left)              { v = left_val; }
      else if (has_right)             { v = right_val; }
      else                            { v = 0.0; }

      table->dat[k / 24].temperature_c[k % 24] = v;
    }

    filled += (gap_end - gap_start);
    i       = gap_end;
  }

  if (filled > 0) {
    log_warning("%S: linearly interpolated %llu missing temperature_c reading(s)", file_path, filled);
  }
}

// NOTE(cmat): Same gap-filling scheme as rs_meteo_radiation_interpolate_gaps, applied to
// - `speed_m_s` (a plain scalar, so straight linear interpolation is fine).
function void rs_meteo_wind_speed_interpolate_gaps(RS_Meteo_Wind_Table *table, B32 *valid, Str08 file_path) {
  U64 count  = table->len * 24;
  U64 filled = 0;
  U64 i      = 0;

  while (i < count) {
    if (valid[i]) { i += 1; continue; }

    U64 gap_start = i;
    U64 gap_end   = i;
    while (gap_end < count && !valid[gap_end]) { gap_end += 1; }

    B32 has_left  = gap_start > 0;
    B32 has_right = gap_end < count;
    F64 left_val  = has_left  ? table->dat[(gap_start - 1) / 24].speed_m_s[(gap_start - 1) % 24] : 0.0;
    F64 right_val = has_right ? table->dat[gap_end / 24].speed_m_s[gap_end % 24]                 : 0.0;

    for (U64 k = gap_start; k < gap_end; k += 1) {
      F64 v;
      if      (has_left && has_right) { F64 t = (F64)(k - gap_start + 1) / (F64)(gap_end - gap_start + 1); v = left_val + (right_val - left_val) * t; }
      else if (has_left)              { v = left_val; }
      else if (has_right)             { v = right_val; }
      else                            { v = 0.0; }

      table->dat[k / 24].speed_m_s[k % 24] = v;
    }

    filled += (gap_end - gap_start);
    i       = gap_end;
  }

  if (filled > 0) {
    log_warning("%S: linearly interpolated %llu missing speed_m_s reading(s)", file_path, filled);
  }
}

// NOTE(cmat): Same gap-filling scheme as the others, applied to `dir_from_deg` - but a
// - compass bearing wraps at 0/360, so a straight linear blend would sweep a gap like
// - 350 deg -> 10 deg the "long way" through 180 deg. Instead this interpolates the unit
// - vector (cos, sin) of the bearing and converts back with atan2, which always takes the
// - short way around the circle.
function void rs_meteo_wind_dir_interpolate_gaps(RS_Meteo_Wind_Table *table, B32 *valid, Str08 file_path) {
  U64 count  = table->len * 24;
  U64 filled = 0;
  U64 i      = 0;

  while (i < count) {
    if (valid[i]) { i += 1; continue; }

    U64 gap_start = i;
    U64 gap_end   = i;
    while (gap_end < count && !valid[gap_end]) { gap_end += 1; }

    B32 has_left  = gap_start > 0;
    B32 has_right = gap_end < count;
    F64 left_deg  = has_left  ? table->dat[(gap_start - 1) / 24].dir_from_deg[(gap_start - 1) % 24] : 0.0;
    F64 right_deg = has_right ? table->dat[gap_end / 24].dir_from_deg[gap_end % 24]                 : 0.0;
    F64 left_rad  = left_deg  * (RS_METEO_PI / 180.0);
    F64 right_rad = right_deg * (RS_METEO_PI / 180.0);

    for (U64 k = gap_start; k < gap_end; k += 1) {
      F64 deg;
      if (has_left && has_right) {
        F64 t = (F64)(k - gap_start + 1) / (F64)(gap_end - gap_start + 1);
        F64 x = cos(left_rad) * (1.0 - t) + cos(right_rad) * t;
        F64 y = sin(left_rad) * (1.0 - t) + sin(right_rad) * t;
        deg   = atan2(y, x) * (180.0 / RS_METEO_PI);
        if (deg < 0.0) { deg += 360.0; }
      } else if (has_left)  { deg = left_deg; }
      else if (has_right)   { deg = right_deg; }
      else                  { deg = 0.0; }

      table->dat[k / 24].dir_from_deg[k % 24] = deg;
    }

    filled += (gap_end - gap_start);
    i       = gap_end;
  }

  if (filled > 0) {
    log_warning("%S: circularly interpolated %llu missing dir_from_deg reading(s)", file_path, filled);
  }
}

function RS_Meteo_Radiation_Table rs_meteo_radiation_table_load(Arena *arena, Str08 file_path) {
  profiler_begin_function();
  log_zone_start("Loading global radiation CSV: \"%S\"", file_path);

  RS_Meteo_Radiation_Table result = {0};

  // NOTE(cmat): Same lane-0-loads-then-broadcasts pattern as csv_emission_lines_load.
  if (lane_index() == 0) {
    SYS_File file_in = { };
    SYS_File_Scope(&file_in, file_path, SYS_File_Access_Flag_Read) {

      U64 file_bytes        = sys_file_size(&file_in);
      SYS_File_Map file_map = { };
      SYS_File_Map_Scope(&file_map, &file_in, range1_u64(0, file_bytes)) {
        Str08 file_str = file_map.map_range;
        log_info("CSV file size: %$llu", file_bytes);

        Scratch scratch = { };
        Scratch_Scope(&scratch, arena) {
          // NOTE(cmat): Skip the header row (INDICATIVO;<year>;MES;DIA;RGLO01;...;RGLO24).
          Scan header_scan = { };
          scan_init(&header_scan, scratch.arena, file_str);
          scan_skip_line(&header_scan);
          Str08 data = str08_slice(file_str, header_scan.at, file_str.len - header_scan.at);

          // NOTE(cmat): First pass just counts data rows so we can allocate the array once.
          U64 rows_len = 0;
          {
            Scan count_scan = { };
            scan_init(&count_scan, scratch.arena, data);
            while (!scan_end(&count_scan)) {
              scan_skip_line(&count_scan);
              rows_len += 1;
            }
          }

          result.len = rows_len;
          result.dat = arena_push_count(arena, RS_Meteo_Radiation_Row, rows_len);

          // NOTE(cmat): One flag per hourly reading (row-major, 24 per day) - tracks which
          // - readings were actually present in the CSV vs. filled in below by interpolation.
          B32 *valid = arena_push_count(scratch.arena, B32, rows_len * 24);

          // NOTE(cmat): Second pass parses each row: station;year;month;day;RGLO01..RGLO24.
          Scan scan = { };
          scan_init(&scan, scratch.arena, data);
          for Iter_Index(it, rows_len) {
            scan_u64(&scan);                        scan_require(&scan, str08_lit(";")); // NOTE(cmat): station indicativo, unused.
            U64 year  = scan_u64(&scan);             scan_require(&scan, str08_lit(";"));
            U64 month = scan_u64(&scan);             scan_require(&scan, str08_lit(";"));
            U64 day   = scan_u64(&scan);             scan_require(&scan, str08_lit(";"));

            RS_Meteo_Radiation_Row *row = &result.dat[it];
            row->year  = (U32)year;
            row->month = (U32)month;
            row->day   = (U32)day;

            for Iter_Index(h, 24) {
              B32 ok;
              row->global_radiation_w_m2[h] = rs_meteo_scan_f64_optional(&scan, &ok, file_path, str08_lit("global_radiation_w_m2")); // NOTE(cmat): W/m^2 already - no unit conversion needed.
              valid[it * 24 + h] = ok;
              if (h != 23) { scan_require(&scan, str08_lit(";")); }
            }

            if (scan_error(&scan)) { break; }
            scan_skip_line(&scan); // NOTE(cmat): consume any trailing '\r' before the next row.
          }

          // NOTE(cmat): Log CSV parse errors.
          for (Scan_Error *it = scan_error(&scan); it; it = it->next) {
            log_fatal("rs_meteo_radiation_table_load error: %u:%u: %S", it->line_at, it->char_at, it->message);
          }

          rs_meteo_radiation_interpolate_gaps(&result, valid, file_path);
        }
      }
    }
  }

  // NOTE(cmat): Broadcast result (array lives in `arena`, shared across lanes).
  lane_broadcast_type(&result, 0);

  log_info("Finished loading global radiation CSV: %llu days", result.len);
  log_zone_end();
  profiler_end_function();
  return result;
}

function RS_Meteo_Temperature_Table rs_meteo_temperature_table_load(Arena *arena, Str08 file_path) {
  profiler_begin_function();
  log_zone_start("Loading temperature CSV: \"%S\"", file_path);

  RS_Meteo_Temperature_Table result = {0};

  if (lane_index() == 0) {
    SYS_File file_in = { };
    SYS_File_Scope(&file_in, file_path, SYS_File_Access_Flag_Read) {

      U64 file_bytes        = sys_file_size(&file_in);
      SYS_File_Map file_map = { };
      SYS_File_Map_Scope(&file_map, &file_in, range1_u64(0, file_bytes)) {
        Str08 file_str = file_map.map_range;
        log_info("CSV file size: %$llu", file_bytes);

        Scratch scratch = { };
        Scratch_Scope(&scratch, arena) {
          // NOTE(cmat): Skip the header row (STATION-CODE;YEAR;MONTH;DAY;T00;...;T23).
          Scan header_scan = { };
          scan_init(&header_scan, scratch.arena, file_str);
          scan_skip_line(&header_scan);
          Str08 data = str08_slice(file_str, header_scan.at, file_str.len - header_scan.at);

          U64 rows_len = 0;
          {
            Scan count_scan = { };
            scan_init(&count_scan, scratch.arena, data);
            while (!scan_end(&count_scan)) {
              scan_skip_line(&count_scan);
              rows_len += 1;
            }
          }

          result.len = rows_len;
          result.dat = arena_push_count(arena, RS_Meteo_Temperature_Row, rows_len);

          // NOTE(cmat): One flag per hourly reading - tracks which readings were actually
          // - present in the CSV vs. filled in below by interpolation.
          B32 *valid = arena_push_count(scratch.arena, B32, rows_len * 24);

          // NOTE(cmat): Second pass parses each row: station;year;month;day;T00..T23 (tenths of degC, can be negative).
          Scan scan = { };
          scan_init(&scan, scratch.arena, data);
          for Iter_Index(it, rows_len) {
            scan_u64(&scan);                        scan_require(&scan, str08_lit(";")); // NOTE(cmat): station code, unused.
            U64 year  = scan_u64(&scan);             scan_require(&scan, str08_lit(";"));
            U64 month = scan_u64(&scan);             scan_require(&scan, str08_lit(";"));
            U64 day   = scan_u64(&scan);             scan_require(&scan, str08_lit(";"));

            RS_Meteo_Temperature_Row *row = &result.dat[it];
            row->year  = (U32)year;
            row->month = (U32)month;
            row->day   = (U32)day;

            for Iter_Index(h, 24) {
              B32 ok;
              F64 tenths_c = rs_meteo_scan_f64_optional(&scan, &ok, file_path, str08_lit("temperature_c")); // NOTE(cmat): scan_f64 handles the leading '-' for sub-zero readings.
              row->temperature_c[h] = tenths_c / 10.0; // NOTE(cmat): tenths of degC -> degC (0.0 is just a placeholder for a missing reading here).
              valid[it * 24 + h] = ok;
              if (h != 23) { scan_require(&scan, str08_lit(";")); }
            }

            if (scan_error(&scan)) { break; }
            scan_skip_line(&scan);
          }

          for (Scan_Error *it = scan_error(&scan); it; it = it->next) {
            log_fatal("rs_meteo_temperature_table_load error: %u:%u: %S", it->line_at, it->char_at, it->message);
          }

          rs_meteo_temperature_interpolate_gaps(&result, valid, file_path);
        }
      }
    }
  }

  lane_broadcast_type(&result, 0);

  log_info("Finished loading temperature CSV: %llu days", result.len);
  log_zone_end();
  profiler_end_function();
  return result;
}

function RS_Meteo_Wind_Table rs_meteo_wind_table_load(Arena *arena, Str08 file_path) {
  profiler_begin_function();
  log_zone_start("Loading wind CSV: \"%S\"", file_path);

  RS_Meteo_Wind_Table result = {0};

  if (lane_index() == 0) {
    SYS_File file_in = { };
    SYS_File_Scope(&file_in, file_path, SYS_File_Access_Flag_Read) {

      U64 file_bytes        = sys_file_size(&file_in);
      SYS_File_Map file_map = { };
      SYS_File_Map_Scope(&file_map, &file_in, range1_u64(0, file_bytes)) {
        Str08 file_str = file_map.map_range;
        log_info("CSV file size: %$llu", file_bytes);

        Scratch scratch = { };
        Scratch_Scope(&scratch, arena) {
          // NOTE(cmat): Skip the header row (STATION-CODE;YEAR;MONTH;DAY;DIR_00;speed_00;...;DIR_23;speed_23).
          Scan header_scan = { };
          scan_init(&header_scan, scratch.arena, file_str);
          scan_skip_line(&header_scan);
          Str08 data = str08_slice(file_str, header_scan.at, file_str.len - header_scan.at);

          U64 rows_len = 0;
          {
            Scan count_scan = { };
            scan_init(&count_scan, scratch.arena, data);
            while (!scan_end(&count_scan)) {
              scan_skip_line(&count_scan);
              rows_len += 1;
            }
          }

          result.len = rows_len;
          result.dat = arena_push_count(arena, RS_Meteo_Wind_Row, rows_len);

          // NOTE(cmat): dir_from_deg and speed_m_s can each be missing independently, so they
          // - each get their own valid-mask and are interpolated separately below (direction
          // - circularly, speed linearly).
          B32 *valid_dir   = arena_push_count(scratch.arena, B32, rows_len * 24);
          B32 *valid_speed = arena_push_count(scratch.arena, B32, rows_len * 24);

          // NOTE(cmat): Second pass parses each row: station;year;month;day;(DIR_hh;speed_hh) x 24.
          Scan scan = { };
          scan_init(&scan, scratch.arena, data);
          for Iter_Index(it, rows_len) {
            scan_u64(&scan);                        scan_require(&scan, str08_lit(";")); // NOTE(cmat): station code, unused.
            U64 year  = scan_u64(&scan);             scan_require(&scan, str08_lit(";"));
            U64 month = scan_u64(&scan);             scan_require(&scan, str08_lit(";"));
            U64 day   = scan_u64(&scan);             scan_require(&scan, str08_lit(";"));

            RS_Meteo_Wind_Row *row = &result.dat[it];
            row->year  = (U32)year;
            row->month = (U32)month;
            row->day   = (U32)day;

            for Iter_Index(h, 24) {
              B32 dir_ok, speed_ok;
              F64 dir_tens     = rs_meteo_scan_f64_optional(&scan, &dir_ok, file_path, str08_lit("dir_from_deg")); scan_require(&scan, str08_lit(";"));
              F64 speed_tenths = rs_meteo_scan_f64_optional(&scan, &speed_ok, file_path, str08_lit("speed_m_s"));

              row->dir_from_deg[h] = dir_tens * 10.0;   // NOTE(cmat): tens of degrees -> degrees (placeholder if missing).
              row->speed_m_s[h]    = speed_tenths / 10.0; // NOTE(cmat): tenths of m/s -> m/s (placeholder if missing).

              valid_dir[it * 24 + h]   = dir_ok;
              valid_speed[it * 24 + h] = speed_ok;

              if (h != 23) { scan_require(&scan, str08_lit(";")); }
            }

            if (scan_error(&scan)) { break; }
            scan_skip_line(&scan);
          }

          for (Scan_Error *it = scan_error(&scan); it; it = it->next) {
            log_fatal("rs_meteo_wind_table_load error: %u:%u: %S", it->line_at, it->char_at, it->message);
          }

          rs_meteo_wind_speed_interpolate_gaps(&result, valid_speed, file_path);
          rs_meteo_wind_dir_interpolate_gaps(&result, valid_dir, file_path);
        }
      }
    }
  }

  lane_broadcast_type(&result, 0);

  log_info("Finished loading wind CSV: %llu days", result.len);
  log_zone_end();
  profiler_end_function();
  return result;
}

// ------------------------------------------------------------
// #-- Calendar helpers

global const U32 rs_meteo_days_in_month_common[12] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
global const U32 rs_meteo_days_in_month_leap[12]   = { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

function B32 rs_meteo_is_leap_year(U32 year) {
  B32 result = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
  return result;
}

function U32 rs_meteo_days_in_year(U32 year) {
  U32 result = rs_meteo_is_leap_year(year) ? 366 : 365;
  return result;
}

function U32 rs_meteo_day_of_year(U32 year, U32 month, U32 day) {
  U32 const *days_in_month = rs_meteo_is_leap_year(year) ? rs_meteo_days_in_month_leap : rs_meteo_days_in_month_common;

  U32 result = day;
  for Iter_Index(it, month - 1) {
    result += days_in_month[it];
  }

  return result;
}

function void rs_meteo_date_from_day_of_year(U32 year, U32 day_of_year, U32 *out_month, U32 *out_day) {
  U32 const *days_in_month = rs_meteo_is_leap_year(year) ? rs_meteo_days_in_month_leap : rs_meteo_days_in_month_common;

  U32 remaining = day_of_year;
  U32 month     = 1;
  for Iter_Index(it, 12) {
    if (remaining <= days_in_month[it]) { break; }
    remaining -= days_in_month[it];
    month     += 1;
  }

  *out_month = month;
  *out_day   = remaining;
}

// ------------------------------------------------------------
// #-- Table Lookups

function F64 rs_meteo_radiation_lookup(RS_Meteo_Radiation_Table *table, U32 year, U32 month, U32 day, U32 hour) {
  F64 result = 0.0;

  if (table->len > 0) {
    U32 doy       = rs_meteo_day_of_year(year, month, day);
    U64 row_index = (doy >= 1) ? (U64)(doy - 1) : 0;
    if (row_index >= table->len) { row_index = table->len - 1; }

    U32 h = (hour <= 23) ? hour : 23;

    result = table->dat[row_index].global_radiation_w_m2[h];
  }

  return result;
}

function F64 rs_meteo_temperature_lookup(RS_Meteo_Temperature_Table *table, U32 year, U32 month, U32 day, U32 hour) {
  F64 result = 0.0;

  if (table->len > 0) {
    U32 doy       = rs_meteo_day_of_year(year, month, day);
    U64 row_index = (doy >= 1) ? (U64)(doy - 1) : 0;
    if (row_index >= table->len) { row_index = table->len - 1; }

    U32 h = (hour <= 23) ? hour : 23;

    result = table->dat[row_index].temperature_c[h];
  }

  return result;
}

function void rs_meteo_wind_lookup(RS_Meteo_Wind_Table *table, U32 year, U32 month, U32 day, U32 hour, F64 *out_dir_from_deg, F64 *out_speed_m_s) {
  F64 dir_from_deg = 0.0;
  F64 speed_m_s    = 0.0;

  if (table->len > 0) {
    U32 doy       = rs_meteo_day_of_year(year, month, day);
    U64 row_index = (doy >= 1) ? (U64)(doy - 1) : 0;
    if (row_index >= table->len) { row_index = table->len - 1; }

    U32 h = (hour <= 23) ? hour : 23;

    dir_from_deg = table->dat[row_index].dir_from_deg[h];
    speed_m_s    = table->dat[row_index].speed_m_s[h];
  }

  *out_dir_from_deg = dir_from_deg;
  *out_speed_m_s    = speed_m_s;
}

function F64 rs_meteo_wind_angle_math_rad(F64 dir_from_deg) {
  // NOTE(cmat): See the header comment - `dir_from_deg` is the compass bearing wind comes
  // - FROM; convert to the standard math angle of the vector it blows TOWARD.
  F64 deg = 270.0 - dir_from_deg;
  F64 rad = deg * (RS_METEO_PI / 180.0);
  return rad;
}

// ------------------------------------------------------------
// #-- Simulated calendar time + solar position

function RS_Sim_Time rs_sim_time_from_elapsed(U32 start_year, U32 start_month, U32 start_day, U32 start_hour, F64 elapsed_seconds) {
  RS_Sim_Time result = {0};

  I64 start_doy   = (I64)rs_meteo_day_of_year(start_year, start_month, start_day); // 1-based.
  F64 total_days  = (F64)(start_doy - 1) + ((F64)start_hour + elapsed_seconds / 3600.0) / 24.0; // fractional days since <start_year>-01-01 00:00.

  I64 year = (I64)start_year;

  // NOTE(cmat): Normalize into [0, days_in_year(year)) - handles a run that crosses a
  // - year boundary, even though the current 2014-11-06..2014-11-30 window never does.
  while (total_days < 0.0) {
    year       -= 1;
    total_days += (F64)rs_meteo_days_in_year((U32)year);
  }
  while (total_days >= (F64)rs_meteo_days_in_year((U32)year)) {
    total_days -= (F64)rs_meteo_days_in_year((U32)year);
    year       += 1;
  }

  U32 day_index_0 = (U32)total_days; // 0-based whole days since Jan 1 of `year`.
  F64 day_frac    = total_days - (F64)day_index_0;
  F64 hour_frac   = day_frac * 24.0;

  U32 month, day;
  rs_meteo_date_from_day_of_year((U32)year, day_index_0 + 1, &month, &day);

  result.year      = (U32)year;
  result.month     = month;
  result.day       = day;
  result.hour      = (U32)hour_frac; // floor.
  result.hour_frac = hour_frac;

  return result;
}

function RS_Solar_Position rs_solar_position(F64 latitude_deg, F64 longitude_deg, RS_Sim_Time time) {
  RS_Solar_Position result = {0};

  U32 doy           = rs_meteo_day_of_year(time.year, time.month, time.day);
  F64 days_in_year  = (F64)rs_meteo_days_in_year(time.year);

  // NOTE(cmat): Spencer (1971) fractional-year angle, centered on local (approximate)
  // - solar time rather than just the UTC hour, matching the standard formulation.
  F64 gamma = (2.0 * RS_METEO_PI / days_in_year) * ((F64)doy - 1.0 + (time.hour_frac - 12.0) / 24.0);

  F64 declination =
      0.006918
    - 0.399912 * cos(gamma)
    + 0.070257 * sin(gamma)
    - 0.006758 * cos(2.0 * gamma)
    + 0.000907 * sin(2.0 * gamma)
    - 0.002697 * cos(3.0 * gamma)
    + 0.001480 * sin(3.0 * gamma);

  F64 eot_minutes = 229.18 * (
      0.000075
    + 0.001868 * cos(gamma)
    - 0.032077 * sin(gamma)
    - 0.014615 * cos(2.0 * gamma)
    - 0.040849 * sin(2.0 * gamma)
  );

  // NOTE(cmat): Input hours are UTC (per the station data descriptions), so the reference
  // - meridian is 0 deg - only the true-longitude correction (4 min/deg) applies on top of
  // - the equation of time, no separate civil-timezone-offset term.
  F64 time_correction_min = eot_minutes + 4.0 * longitude_deg;
  F64 solar_time_min      = time.hour_frac * 60.0 + time_correction_min;
  F64 hour_angle_deg      = (solar_time_min / 4.0) - 180.0;
  F64 hour_angle          = hour_angle_deg * (RS_METEO_PI / 180.0);

  F64 lat_rad = latitude_deg * (RS_METEO_PI / 180.0);

  F64 cos_zenith = sin(lat_rad) * sin(declination) + cos(lat_rad) * cos(declination) * cos(hour_angle);

  result.declination_rad      = declination;
  result.hour_angle_rad       = hour_angle;
  result.equation_of_time_min = eot_minutes;
  result.cos_zenith           = cos_zenith;

  return result;
}
