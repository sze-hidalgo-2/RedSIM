// ------------------------------------------------------------
// #-- SU2 parse plan (built by lane 0, broadcast to all lanes)

typedef struct UG_SU2_Marker_Plan {
  Str08 tag;
  U64   line_start;   // first line of this marker's element data
  U64   elem_count;
} UG_SU2_Marker_Plan;

typedef struct UG_SU2_Parse_Plan {
  Str08 file_str;

  U64  *line_offsets; // line_offsets[i] = byte offset of start of line i; len = line_count+1
  U64   line_count;

  U64   point_line_start;
  U64   point_count;

  U64   elem_line_start;
  U64   elem_count;

  U64                 marker_count;
  UG_SU2_Marker_Plan *markers;

  U32  *lane_error_flags; // len = lane_count(), one slot per lane, no contention
} UG_SU2_Parse_Plan;

function Str08 su2_str08_slice(Str08 str, U64 begin, U64 end) {
  Str08 result = { .len = end - begin, .txt = str.txt + begin };
  return result;
}

// NOTE(cmat): Cheap O(n) byte scan for '\n'. This is far cheaper than the
// float/int tokenizing it replaces during the structural walk, and lets us
// jump over whole blocks (and later, hand each lane its own byte range) in O(1).
function void su2_build_line_offsets(Arena *arena, UG_SU2_Parse_Plan *plan) {
  profiler_begin_function();
  Str08 str = plan->file_str;
  U64 newline_count = 0;
  for (U64 i = 0; i < str.len; i += 1) {
    if (str.txt[i] == '\n') { newline_count += 1; }
  }
  U64 line_count = newline_count + ((str.len > 0 && str.txt[str.len - 1] != '\n') ? 1 : 0);
  U64 *offsets   = arena_push_count(arena, U64, line_count + 1);
  U64  idx       = 0;
  offsets[idx]   = 0;
  for (U64 i = 0; i < str.len; i += 1) {
    if (str.txt[i] == '\n') {
      idx += 1;
      offsets[idx] = i + 1;
    }
  }
  offsets[line_count]  = str.len;
  plan->line_offsets   = offsets;
  plan->line_count     = line_count;
  profiler_end_function();
}

// ------------------------------------------------------------
// #-- Phase 1 (lane 0 only): structural walk, allocations, no bulk parsing

function void ugf_su2_parse_structural(Arena *arena, UG_Grid *ugrid, UG_SU2_Parse_Plan *plan) {
  profiler_begin_function();
  su2_build_line_offsets(arena, plan);
  plan->lane_error_flags = arena_push_count(arena, U32, lane_count());

  Scratch scratch = { };
  Scratch_Scope(&scratch, arena) {
    Scan scan          = { };
    U64  current_line  = 0;
    scan_init(&scan, scratch.arena, plan->file_str);

    scan_skip_whitespace(&scan);
    scan_require(&scan, str08_lit("NDIME"));
    scan_require(&scan, str08_lit("="));
    U64 dimension = scan_u64(&scan);
    log_info("Dimension: %llu", dimension);

    if (!scan_error(&scan)) {
      scan_skip_line(&scan);
      current_line += 1;

      for (;;) {
        if (scan_end(&scan) || scan_error(&scan)) { break; }
        Str08 block_type = scan_identifier(&scan);

        if (0);
        else if (str08_match(block_type, str08_lit("NPOIN"))) {
          scan_require(&scan, str08_lit("="));
          U64 point_count = scan_u64(&scan);
          log_info("Parsing NPOIN block: %'llu", point_count);
          if (scan_error(&scan)) { break; }
          scan_skip_line(&scan);
          current_line += 1;

          ugrid->verts.len = point_count;
          ugrid->verts.x   = arena_push_count(arena, F32, point_count);
          ugrid->verts.y   = arena_push_count(arena, F32, point_count);
          ugrid->verts.z   = arena_push_count(arena, F32, point_count);

          plan->point_line_start = current_line;
          plan->point_count      = point_count;

          current_line += point_count;
          scan_init(&scan, scratch.arena,
                    su2_str08_slice(plan->file_str, plan->line_offsets[current_line], plan->file_str.len));
        }
        else if (str08_match(block_type, str08_lit("NELEM"))) {
          scan_require(&scan, str08_lit("="));
          U64 element_count = scan_u64(&scan);
          log_info("Parsing NELEM block: %'llu", element_count);
          if (scan_error(&scan)) { break; }
          scan_skip_line(&scan);
          current_line += 1;

          ugrid->elems.len   = element_count;
          ugrid->elems.verts = arena_push_count(arena, V4_U32, element_count);

          plan->elem_line_start = current_line;
          plan->elem_count      = element_count;

          current_line += element_count;
          scan_init(&scan, scratch.arena,
                    su2_str08_slice(plan->file_str, plan->line_offsets[current_line], plan->file_str.len));
        }
        else if (str08_match(block_type, str08_lit("NMARK"))) {
          scan_require(&scan, str08_lit("="));
          U64 mark_count = scan_u64(&scan);
          log_info("Parsing NMARK block: %'llu", mark_count);
          if (scan_error(&scan)) { break; }
          scan_skip_line(&scan);
          current_line += 1;

          ugrid->markers.len   = mark_count;
          ugrid->markers.tags  = arena_push_count(arena, Str08,                mark_count);
          ugrid->markers.elems = arena_push_count(arena, UG_Grid_Marker_Elems, mark_count);

          plan->marker_count = mark_count;
          plan->markers      = arena_push_count(arena, UG_SU2_Marker_Plan, mark_count);

          B32 marker_error = 0;
          for (U64 it = 0; it < mark_count; it += 1) {
            scan_require(&scan, str08_lit("MARKER_TAG"));
            scan_require(&scan, str08_lit("="));
            Str08 tag_name = scan_identifier(&scan);
            ugrid->markers.tags[it] = arena_push_str08(arena, tag_name);
            if (scan_error(&scan)) { marker_error = 1; break; }
            scan_skip_line(&scan);
            current_line += 1;

            scan_require(&scan, str08_lit("MARKER_ELEMS"));
            scan_require(&scan, str08_lit("="));
            U64 marker_elem_count = scan_u64(&scan);
            if (scan_error(&scan)) { marker_error = 1; break; }
            scan_skip_line(&scan);
            current_line += 1;

            ugrid->markers.elems[it].len   = marker_elem_count;
            ugrid->markers.elems[it].verts = arena_push_count(arena, V3U, marker_elem_count);

            plan->markers[it].tag        = ugrid->markers.tags[it];
            plan->markers[it].line_start = current_line;
            plan->markers[it].elem_count = marker_elem_count;

            current_line += marker_elem_count;
            scan_init(&scan, scratch.arena,
                      su2_str08_slice(plan->file_str, plan->line_offsets[current_line], plan->file_str.len));
          }
          if (marker_error) { break; }
        }
        else {
          Str08 message = str08_format(scratch.arena, "unexpected block \"%S\"", block_type);
          scan_error_push(&scan, message);
        }
      }
    }

    for (Scan_Error *it = scan_error(&scan); it; it = it->next) {
      log_fatal("SU2 error: %u:%u: %S", it->line_at, it->char_at, it->message);
    }
  }
  profiler_end_function();
}

// ------------------------------------------------------------
// #-- Phase 2 (every lane): parallel bulk-data fill, disjoint index ranges

function void ugf_su2_parse_fill_parallel(Arena *arena, UG_Grid *ugrid, UG_SU2_Parse_Plan *plan) {
  profiler_begin_function();
  B32 had_error = 0;

  // -- NPOIN --
  {
    Range1_U64 range = lane_range(plan->point_count);
    if (range.min < range.max) {
      Scratch scratch = { };
      Scratch_Scope(&scratch, arena) {
        U64 byte_begin = plan->line_offsets[plan->point_line_start + range.min];
        U64 byte_end   = plan->line_offsets[plan->point_line_start + range.max];
        Scan local_scan = { };
        scan_init(&local_scan, scratch.arena, su2_str08_slice(plan->file_str, byte_begin, byte_end));
        for (U64 it = range.min; it < range.max; it += 1) {
          ugrid->verts.x[it] = (F32)scan_f64(&local_scan);
          ugrid->verts.y[it] = (F32)scan_f64(&local_scan);
          ugrid->verts.z[it] = (F32)scan_f64(&local_scan);
          scan_skip_line(&local_scan);
          if (scan_error(&local_scan)) { had_error = 1; break; }
        }
      }
    }
  }

  // -- NELEM --
  if (!had_error) {
    Range1_U64 range = lane_range(plan->elem_count);
    if (range.min < range.max) {
      Scratch scratch = { };
      Scratch_Scope(&scratch, arena) {
        U64 byte_begin = plan->line_offsets[plan->elem_line_start + range.min];
        U64 byte_end   = plan->line_offsets[plan->elem_line_start + range.max];
        Scan local_scan = { };
        scan_init(&local_scan, scratch.arena, su2_str08_slice(plan->file_str, byte_begin, byte_end));
        for (U64 it = range.min; it < range.max; it += 1) {
          U64 element_type = scan_u64(&local_scan);
          // NOTE(cmat): Tetrahedral.
          if (element_type == 10) {
            U64 e1 = scan_u64(&local_scan);
            U64 e2 = scan_u64(&local_scan);
            U64 e3 = scan_u64(&local_scan);
            U64 e4 = scan_u64(&local_scan);
            ugrid->elems.verts[it] = v4u((U32)e1, (U32)e2, (U32)e3, (U32)e4);
          } else {
            scan_error_push(&local_scan, str08_lit("unsupported element type in element block!"));
          }
          if (scan_error(&local_scan)) { had_error = 1; break; }
          scan_skip_line(&local_scan);
        }
      }
    }
  }

  // -- NMARK --
  if (!had_error) {
    for (U64 m = 0; m < plan->marker_count; m += 1) {
      UG_SU2_Marker_Plan *marker = &plan->markers[m];
      Range1_U64 range = lane_range(marker->elem_count);
      if (range.min >= range.max) { continue; }
      Scratch scratch = { };
      Scratch_Scope(&scratch, arena) {
        U64 byte_begin = plan->line_offsets[marker->line_start + range.min];
        U64 byte_end   = plan->line_offsets[marker->line_start + range.max];
        Scan local_scan = { };
        scan_init(&local_scan, scratch.arena, su2_str08_slice(plan->file_str, byte_begin, byte_end));
        for (U64 it = range.min; it < range.max; it += 1) {
          U64 type = scan_u64(&local_scan);
          // NOTE(cmat): Triangle.
          if (type == 5) {
            U64 e1 = scan_u64(&local_scan);
            U64 e2 = scan_u64(&local_scan);
            U64 e3 = scan_u64(&local_scan);
            ugrid->markers.elems[m].verts[it] = v3u((U32)e1, (U32)e2, (U32)e3);
          } else {
            scan_error_push(&local_scan, str08_lit("unsupported element type in marker block"));
          }
          if (scan_error(&local_scan)) { had_error = 1; break; }
          scan_skip_line(&local_scan);
        }
      }
      if (had_error) { break; }
    }
  }

  if (had_error) { plan->lane_error_flags[lane_index()] = 1; }
  profiler_end_function();
}

function void ugf_su2_log_fill_errors(UG_SU2_Parse_Plan *plan) {
  for (U64 it = 0; it < lane_count(); it += 1) {
    if (plan->lane_error_flags[it]) {
      log_fatal("SU2 error: lane %llu hit a parse error while filling grid data.", it);
    }
  }
}

// Called by every lane: broadcasts the plan + ugrid pointers, then all lanes
// parse their share of the bulk data, then barrier.
function void ugf_su2_parse_shared(UG_Grid *ugrid, Arena *arena, UG_SU2_Parse_Plan *plan) {
  profiler_begin_function();
  lane_broadcast_type(plan,  0);
  lane_broadcast_type(ugrid, 0);
  ugf_su2_parse_fill_parallel(arena, ugrid, plan);
  lane_barrier();
  profiler_end_function();
}

// ------------------------------------------------------------
function void ugf_grid_init_from_su2(UG_Grid *ugrid, Arena *arena, Str08 file_path) {
  profiler_begin_function();
  log_zone_start("Loading su2 file: \"%S\"", file_path);
  UG_SU2_Parse_Plan plan = { };
  if (lane_index() == 0) {
    Zero_Fill(ugrid);
    SYS_File file_in = { };
    SYS_File_Scope(&file_in, file_path, SYS_File_Access_Flag_Read) {
      U64 file_bytes = sys_file_size(&file_in);
      SYS_File_Map file_map = { };
      SYS_File_Map_Scope(&file_map, &file_in, range1_u64(0, file_bytes)) {
        plan.file_str = file_map.map_range;
        log_info("SU2 file size: %$llu", file_bytes);

        ugf_su2_parse_structural(arena, ugrid, &plan);
        ugrid->scale = 1.f;

        // NOTE(cmat): The file must stay mapped through the parallel fill
        // below, since other lanes read directly from plan.file_str.
        ugf_su2_parse_shared(ugrid, arena, &plan);
        ugf_su2_log_fill_errors(&plan);
      }
    }
  } else {
    ugf_su2_parse_shared(ugrid, arena, &plan);
  }
  log_info("Finished loading su2 file!");
  log_zone_end();
  profiler_end_function();
}
