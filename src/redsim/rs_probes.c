// ------------------------------------------------------------
// #-- Probe file parsing

// NOTE(cmat): Next '\n'-terminated line (trailing '\r' removed). Returns 0 at end of stream.
function B32 rs_probe_next_line(Str08 stream, U64 *at, Str08 *out_line) {
  if (*at >= stream.len) { return 0; }

  U64 start = *at;
  U64 end   = start;
  while (end < stream.len && stream.txt[end] != '\n') { end += 1; }

  U64 line_len = end - start;
  if (line_len > 0 && stream.txt[start + line_len - 1] == '\r') { line_len -= 1; }

  *out_line = str08(line_len, stream.txt + start);
  *at       = end + 1;
  return 1;
}

function B32 rs_probe_field_is_number(Str08 field) {
  if (field.len == 0) { return 0; }
  U08 c = field.txt[0];
  return char_is_digit(c) || c == '-' || c == '+' || c == '.';
}

// NOTE(cmat): Splits on '\t' - NOT scan_f64/scan_skip_whitespace, which treat '\t' and '\n' as skippable
// - whitespace and would happily run across fields / lines.
function B32 rs_probe_parse_line(Str08 line, Str08 *out_name, F64 *out_x, F64 *out_y, F64 *out_z) {
  Str08 fields[7];
  U32   count   = 0;
  U64   f_start = 0;
  for (U64 i = 0; i <= line.len && count < 7; i += 1) {
    if (i == line.len || line.txt[i] == '\t') {
      fields[count] = str08_trim(str08(i - f_start, line.txt + f_start));
      count        += 1;
      f_start       = i + 1;
    }
  }

  if (count < 7)                          { return 0; }
  if (fields[0].len == 0)                 { return 0; }
  if (!rs_probe_field_is_number(fields[4])) { return 0; }
  if (!rs_probe_field_is_number(fields[5])) { return 0; }
  if (!rs_probe_field_is_number(fields[6])) { return 0; }

  *out_name = fields[0];
  *out_x    = f64_from_str08(fields[4]);
  *out_y    = f64_from_str08(fields[5]);
  *out_z    = f64_from_str08(fields[6]);
  return 1;
}

function RS_Probe_Array rs_probes_load(Arena *arena, Str08 file_path) {
  profiler_begin_function();
  log_zone_start("Loading probe locations: \"%S\"", file_path);

  RS_Probe_Array result = {0};

  // NOTE(cmat): Same lane-0-loads-then-broadcasts pattern as the other loaders.
  if (lane_index() == 0) {
    SYS_File file_in = { };
    SYS_File_Scope(&file_in, file_path, SYS_File_Access_Flag_Read) {

      U64 file_bytes        = sys_file_size(&file_in);
      SYS_File_Map file_map = { };
      SYS_File_Map_Scope(&file_map, &file_in, range1_u64(0, file_bytes)) {
        Str08 stream = file_map.map_range;
        log_info("Probe file size: %$llu", file_bytes);

        // NOTE(cmat): Skip UTF-8 BOM if present.
        if (stream.len >= 3 && stream.txt[0] == 0xEF && stream.txt[1] == 0xBB && stream.txt[2] == 0xBF) {
          stream = str08(stream.len - 3, stream.txt + 3);
        }

        // NOTE(cmat): Pass 0 counts parsable rows (so we allocate once), pass 1 fills them.
        U64 skipped = 0;
        for (U32 pass = 0; pass < 2; pass += 1) {
          U64   at = 0;
          U64   n  = 0;
          Str08 line = { };
          while (rs_probe_next_line(stream, &at, &line)) {
            Str08 name = { };
            F64   x = 0, y = 0, z = 0;
            if (rs_probe_parse_line(line, &name, &x, &y, &z)) {
              if (pass == 1) {
                result.dat[n] = (RS_Probe) {
                  .name     = arena_push_str08(arena, name), // NOTE(cmat): copy out of the file map before it's unmapped.
                  .position = v3f((F32)x, (F32)y, (F32)z),
                  .cell     = -1,
                };
              }
              n += 1;
            } else if (pass == 0 && str08_trim(line).len > 0) {
              skipped += 1;
            }
          }

          if (pass == 0) {
            result.len = n;
            result.dat = arena_push_count(arena, RS_Probe, n);
          }
        }

        if (skipped > 0) { log_info("Probe file: %llu non-empty lines could not be parsed and were skipped", skipped); }
      }
    }
  }

  lane_broadcast_type(&result, 0);

  log_info("Finished loading probe locations: %llu probes", result.len);
  log_zone_end();
  profiler_end_function();
  return result;
}

// ------------------------------------------------------------
// #-- Locating probes in the (partitioned) mesh

function void rs_probes_locate(RS_Probe_Array *probes, UG_Mesh *mesh, Arena *arena, FL_Scale *scale, F64 search_step_m) {
  if (lane_index() == 0) {
    F32 inv_length = f32_div_safe(1.f, scale->length);
    F32 step       = (F32)(search_step_m / (F64)scale->length);
    U64 located    = 0;

    for (U64 it = 0; it < probes->len; it += 1) {
      RS_Probe *probe = &probes->dat[it];
      probe->cell     = -1;

      // NOTE(cmat): Same transform as the emission lines (their UTM offset is already baked into the probe file).
      V3F p = v3f_mul(inv_length, v3f_sub(probe->position, scale->offset));

      // NOTE(cmat): Short vertical segment centred on the probe, traced with the same marcher the emission
      // - lines use (it only reports cells owned by this rank). The probe's cell is the hit whose
      // - [t_enter, t_exit] contains the segment midpoint t = 0.5; a rank that only sees the segment's end
      // - (the probe's cell lives on another rank) doesn't claim it.
      V3F a = v3f_sub(p, v3f(0.f, 0.f, step));
      V3F b = v3f_add(p, v3f(0.f, 0.f, step));

      Scratch scratch = { };
      Scratch_Scope(&scratch, arena) {
        UG_Segment_Trace trace = ug_mesh_trace_segment_marched(scratch.arena, mesh, a, b, step);
        for (U64 it_hit = 0; it_hit < trace.len; it_hit += 1) {
          UG_Segment_Hit *hit = &trace.dat[it_hit];
          if (hit->t_enter <= 0.5f + 1e-4f && hit->t_exit >= 0.5f - 1e-4f) {
            probe->cell = (I64)hit->cell;
            break;
          }
        }
      }

      if (probe->cell >= 0) { located += 1; }
    }

    log_info("probes: %llu of %llu located in this rank's cells", located, probes->len);

    // NOTE(cmat): With a single rank "not located" really means outside the fluid (inside a building / above the domain).
    if (ipc_rank_count() == 1) {
      for (U64 it = 0; it < probes->len; it += 1) {
        if (probes->dat[it].cell < 0) {
          log_info("probe %S (%.3f, %.3f, %.3f) is not inside any mesh cell - its column will be empty",
                   probes->dat[it].name, (F64)probes->dat[it].position.x, (F64)probes->dat[it].position.y, (F64)probes->dat[it].position.z);
        }
      }
    }
  }
}

// ------------------------------------------------------------
// #-- Hourly CSV

function void rs_probes_csv_open(RS_Probe_Writer *writer, RS_Probe_Array *probes, const char *stem) {
  writer->file = 0;
  if (lane_index() == 0) {
    char path[256];
    if (ipc_rank_count() > 1) {
      snprintf(path, sizeof(path), "%s_rank%llu.csv", stem, (unsigned long long)ipc_rank_index());
    } else {
      snprintf(path, sizeof(path), "%s.csv", stem);
    }

    writer->file = fopen(path, "wb");
    if (!writer->file) {
      log_info("probes: could not open \"%s\" for writing - probe CSV disabled", path);
    } else {
      fprintf(writer->file, "hour,elapsed_s,utc");
      for (U64 it = 0; it < probes->len; it += 1) {
        fprintf(writer->file, ",%.*s", (int)probes->dat[it].name.len, (char *)probes->dat[it].name.txt);
      }
      fprintf(writer->file, "\n");
      fflush(writer->file);
      log_info("probes: writing \"%s\" (%llu probe columns)", path, probes->len);
    }
  }
}

function void rs_probes_csv_write_row(RS_Probe_Writer *writer, RS_Probe_Array *probes, F32 *phi,
                                      U64 hour_index, F64 elapsed_seconds, RS_Sim_Time time) {
  if (lane_index() == 0 && writer->file) {
    fprintf(writer->file, "%llu,%.0f,%04u-%02u-%02uT%02u:00:00Z",
            (unsigned long long)hour_index, elapsed_seconds,
            (unsigned)time.year, (unsigned)time.month, (unsigned)time.day, (unsigned)time.hour);

    for (U64 it = 0; it < probes->len; it += 1) {
      if (probes->dat[it].cell >= 0) {
        fprintf(writer->file, ",%.6e", (double)phi[probes->dat[it].cell]);
      } else {
        fputc(',', writer->file);
      }
    }
    fputc('\n', writer->file);
    fflush(writer->file); // NOTE(cmat): flush every hour so a crashed / stopped run keeps its rows.
  }
}

function void rs_probes_csv_close(RS_Probe_Writer *writer) {
  if (lane_index() == 0 && writer->file) {
    fclose(writer->file);
    writer->file = 0;
  }
}
