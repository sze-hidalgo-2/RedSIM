function RS_NOX_Ratio_Table rs_nox_ratio_table_load(Arena *arena, Str08 file_path) {
  profiler_begin_function();
  log_zone_start("Loading NOx ratio CSV: \"%S\"", file_path);

  RS_NOX_Ratio_Table result = {0};

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
          // NOTE(cmat): Skip the header row ("Year;Month;Day;Day of the week;Hour;Period factor;Anual_factor").
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
          result.dat = arena_push_count(arena, RS_NOX_Ratio_Row, rows_len);

          // NOTE(cmat): Second pass actually parses each row:
          // - Year;Month;Day;Day of the week;Hour;Period factor;Anual_factor
          Scan scan = { };
          scan_init(&scan, scratch.arena, data);
          for Iter_Index(it, rows_len) {
            U64   year    = scan_u64(&scan);        scan_require(&scan, str08_lit(";"));
            U64   month   = scan_u64(&scan);        scan_require(&scan, str08_lit(";"));
            U64   day     = scan_u64(&scan);        scan_require(&scan, str08_lit(";"));
            Str08 weekday = scan_identifier(&scan); scan_require(&scan, str08_lit(";"));
            U64   hour    = scan_u64(&scan);        scan_require(&scan, str08_lit(";"));
            F64   period  = scan_f64(&scan);        scan_require(&scan, str08_lit(";"));
            F64   annual  = scan_f64(&scan);

            result.dat[it] = (RS_NOX_Ratio_Row) {
              .year          = (U32)year,
              .month         = (U32)month,
              .day           = (U32)day,
              .hour          = (U32)hour,
              .weekday       = arena_push_str08(arena, weekday), // NOTE(cmat): copy out of scratch before it's released.
              .period_factor = period,
              .annual_factor = annual,
            };

            if (scan_error(&scan)) { break; }
            scan_skip_line(&scan); // NOTE(cmat): consume any trailing '\r' before the next row.
          }

          // NOTE(cmat): Log CSV parse errors.
          for (Scan_Error *it = scan_error(&scan); it; it = it->next) {
            log_fatal("rs_nox_ratio_table_load error: %u:%u: %S", it->line_at, it->char_at, it->message);
          }
        }
      }
    }

    if (result.len > 0) {
      result.start_year  = result.dat[0].year;
      result.start_month = result.dat[0].month;
      result.start_day   = result.dat[0].day;
      result.start_hour  = result.dat[0].hour;
    }
  }

  // NOTE(cmat): Broadcast result (array + weekday strings live in `arena`, shared across lanes).
  lane_broadcast_type(&result, 0);

  log_info("Finished loading NOx ratio CSV: %llu rows (start: %u-%02u-%02u %02u:00, %S)",
      result.len, result.start_year, result.start_month, result.start_day, result.start_hour,
      result.len > 0 ? result.dat[0].weekday : str08_lit("?"));
  log_zone_end();
  profiler_end_function();
  return result;
}

function F64 rs_nox_ratio_table_day_weight(RS_NOX_Ratio_Table *table, F64 elapsed_seconds) {
  if (table->len == 0) return 0.0;

  // NOTE(cmat): Table is one row per hour, row 0 == the simulation's start hour, so the
  // - elapsed simulated time in hours (floored) is directly the row index.
  F64 elapsed_hours = elapsed_seconds / 3600.0;
  I64 hour_index     = (I64)elapsed_hours;

  if (hour_index < 0)                hour_index = 0;
  if (hour_index >= (I64)table->len) hour_index = (I64)table->len - 1;

  return table->dat[hour_index].period_factor;
}
