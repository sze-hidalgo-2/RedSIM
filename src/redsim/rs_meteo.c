#include <math.h>

#define RS_METEO_PI 3.14159265358979323846

// ------------------------------------------------------------
// #-- CSV Loaders

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
              row->global_radiation_w_m2[h] = scan_f64(&scan); // NOTE(cmat): W/m^2 already - no unit conversion needed.
              if (h != 23) { scan_require(&scan, str08_lit(";")); }
            }

            if (scan_error(&scan)) { break; }
            scan_skip_line(&scan); // NOTE(cmat): consume any trailing '\r' before the next row.
          }

          // NOTE(cmat): Log CSV parse errors.
          for (Scan_Error *it = scan_error(&scan); it; it = it->next) {
            log_fatal("rs_meteo_radiation_table_load error: %u:%u: %S", it->line_at, it->char_at, it->message);
          }
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
              F64 tenths_c = scan_f64(&scan); // NOTE(cmat): scan_f64 handles the leading '-' for sub-zero readings.
              row->temperature_c[h] = tenths_c / 10.0; // NOTE(cmat): tenths of degC -> degC.
              if (h != 23) { scan_require(&scan, str08_lit(";")); }
            }

            if (scan_error(&scan)) { break; }
            scan_skip_line(&scan);
          }

          for (Scan_Error *it = scan_error(&scan); it; it = it->next) {
            log_fatal("rs_meteo_temperature_table_load error: %u:%u: %S", it->line_at, it->char_at, it->message);
          }
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
              F64 dir_tens   = scan_f64(&scan); scan_require(&scan, str08_lit(";"));
              F64 speed_tenths = scan_f64(&scan);

              row->dir_from_deg[h] = dir_tens * 10.0;   // NOTE(cmat): tens of degrees -> degrees.
              row->speed_m_s[h]    = speed_tenths / 10.0; // NOTE(cmat): tenths of m/s -> m/s.

              if (h != 23) { scan_require(&scan, str08_lit(";")); }
            }

            if (scan_error(&scan)) { break; }
            scan_skip_line(&scan);
          }

          for (Scan_Error *it = scan_error(&scan); it; it = it->next) {
            log_fatal("rs_meteo_wind_table_load error: %u:%u: %S", it->line_at, it->char_at, it->message);
          }
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
