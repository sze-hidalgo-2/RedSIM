function CSV_Emission_Line_Array csv_emission_lines_load(Arena *arena, Str08 file_path, F32 emission_z) {
  profiler_begin_function();
  log_zone_start("Loading traffic emission CSV: \"%S\"", file_path);

  CSV_Emission_Line_Array result = {0};

  // NOTE(cmat): Same lane-0-loads-then-broadcasts pattern as ugf_grid_init_from_su2.
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
          // NOTE(cmat): Skip the header row ("x0,y0,x1,y1,value_1,value_2").
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
          result.dat = arena_push_count(arena, CSV_Emission_Line, rows_len);

          // NOTE(cmat): Second pass actually parses each row: x0,y0,x1,y1,value_1,value_2.
          Scan scan = { };
          scan_init(&scan, scratch.arena, data);
          for Iter_Index(it, rows_len) {
            F32 x0 = (F32)scan_f64(&scan); scan_require(&scan, str08_lit(","));
            F32 y0 = (F32)scan_f64(&scan); scan_require(&scan, str08_lit(","));
            F32 x1 = (F32)scan_f64(&scan); scan_require(&scan, str08_lit(","));
            F32 y1 = (F32)scan_f64(&scan); scan_require(&scan, str08_lit(","));
            F32 value_1 = (F32)scan_f64(&scan); scan_require(&scan, str08_lit(","));
            F32 value_2 = (F32)scan_f64(&scan); (void)value_2; // NOTE(cmat): unused.

            result.dat[it] = (CSV_Emission_Line) {
              .a       = v3f(x0, y0, emission_z),
              .b       = v3f(x1, y1, emission_z),
              .value_1 = value_1,
            };

            if (scan_error(&scan)) { break; }
            scan_skip_line(&scan); // NOTE(cmat): consume any trailing '\r' before the next row.
          }

          // NOTE(cmat): Log CSV parse errors.
          for (Scan_Error *it = scan_error(&scan); it; it = it->next) {
            log_fatal("csv_emission_lines_load error: %u:%u: %S", it->line_at, it->char_at, it->message);
          }
        }
      }
    }
  }

  // NOTE(cmat): Broadcast result (array lives in `arena`, shared across lanes).
  lane_broadcast_type(&result, 0);

  log_info("Finished loading traffic emission CSV: %llu lines", result.len);
  log_zone_end();
  profiler_end_function();
  return result;
}
