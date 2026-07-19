function void ugf_su2_parse_block_NPOIN(Scan *scan, UG_Mesh *mesh) {
  profiler_begin_function();
  scan_require(scan, str08_lit("="));

  U64 point_count = scan_u64(scan);
  log_info("Parsing NPOIN block: %'llu", point_count);

  if (!scan_error(scan)) {
    mesh->verts.len = point_count;
    mesh->verts.x   = arena_push_count(&mesh->arena, F32, point_count);
    mesh->verts.y   = arena_push_count(&mesh->arena, F32, point_count);
    mesh->verts.z   = arena_push_count(&mesh->arena, F32, point_count);

    for Iter_Index(it, point_count) {
      mesh->verts.x[it] = (F32)scan_f64(scan);
      mesh->verts.y[it] = (F32)scan_f64(scan);
      mesh->verts.z[it] = (F32)scan_f64(scan);

      scan_skip_line(scan);
      if (scan_error(scan)) { break; }
    }
  }

  profiler_end_function();
}

function void ugf_su2_parse_block_NELEM(Scan *scan, UG_Mesh *mesh) {
  profiler_begin_function();
  scan_require(scan, str08_lit("="));

  U64 element_count = scan_u64(scan);
  log_info("Parsing NELEM block: %'llu", element_count);

  if (!scan_error(scan)) {
    mesh->cells.len   = element_count;
    mesh->cells.verts = arena_push_count(&mesh->arena, V4_U32, mesh->cells.len);

    for Iter_Index(it, element_count) {
      U64 element_type = scan_u64(scan);

      // NOTE(cmat): Tetrahedral.
      if (element_type == 10) { 
        U64 e1 = scan_u64(scan);
        U64 e2 = scan_u64(scan);
        U64 e3 = scan_u64(scan);
        U64 e4 = scan_u64(scan);

        mesh->cells.verts[it] = v4u((U32)e1, (U32)e2, (U32)e3, (U32)e4);

      } else {
        scan_error_push(scan, str08_lit("unsupported element type in element block"));
      }

      if (scan_error(scan)) { break; }
      scan_skip_line(scan);
    }
  }

  profiler_end_function();
}

function void ugf_su2_parse_block_NMARK(Scan *scan, UG_Mesh *mesh) {
  profiler_begin_function();
  scan_require(scan, str08_lit("="));

  U64 mark_count = scan_u64(scan);
  log_info("Parsing NMARK block: %'llu", mark_count);

  if (!scan_error(scan)) {
    mesh->markers.len       = mark_count;
    mesh->markers.tags      = arena_push_count(&mesh->arena, Str08,           mark_count);
    mesh->markers.elems     = arena_push_count(&mesh->arena, UG_Marker_Elems, mark_count);

    for Iter_Index(it, mark_count) {
      scan_require(scan, str08_lit("MARKER_TAG"));
      scan_require(scan, str08_lit("="));
      Str08 tag_name          = scan_identifier(scan);
      mesh->markers.tags[it]  = arena_push_str08(&mesh->arena, tag_name);

      scan_require(scan, str08_lit("MARKER_ELEMS"));
      scan_require(scan, str08_lit("="));
      U64 element_count = scan_u64(scan);

      mesh->markers.elems[it].len   = element_count;
      mesh->markers.elems[it].verts = arena_push_count(&mesh->arena, V3U, element_count);

      for Iter_Index(it_elem, element_count) {
        U64 type = scan_u64(scan);

        // NOTE(cmat): Triangle.
        if (type == 5) {
          U64 e1    = scan_u64(scan);
          U64 e2    = scan_u64(scan);
          U64 e3    = scan_u64(scan);
          V3U verts = v3u((U32)e1, (U32)e2, (U32)e3);

          mesh->markers.elems[it].verts[it_elem] = verts;
        } else {
          scan_error_push(scan, str08_lit("unsupported element type in marker block"));
        }

        if (scan_error(scan)) { break; }
      }

      if (scan_error(scan)) { break; }
    }
  }

  profiler_end_function();
}


function void ugf_load_su2(UG_Mesh *mesh, Str08 file_path) {
  profiler_begin_function();
  log_zone_start("Parsing su2 file: \"%S\"", file_path);

  SYS_File file_in = { };
  SYS_File_Scope(&file_in, file_path, SYS_File_Access_Flag_Read) {

    U64 file_bytes        = sys_file_size(&file_in);
    SYS_File_Map file_map = { };
    SYS_File_Map_Scope(&file_map, &file_in, range1_u64(0, file_bytes)) {
      Str08 file_str = file_map.map_range;
      log_info("SU2 file size: %$llu", file_bytes);

      Scratch scratch = { };
      Scratch_Scope(&scratch, 0) {
        Scan scan = { };
        scan_init(&scan, scratch.arena, file_str);

        // NOTE(cmat): Get dimension.
        scan_skip_whitespace(&scan);
        scan_require(&scan, str08_lit("NDIME"));
        scan_require(&scan, str08_lit("="));

        U64 dimension = scan_u64(&scan);
        log_info("Dimension: %llu", dimension);

        // NOTE(cmat): Parse blocks.
        if (!scan_error(&scan)) {
          for (;;) {
            if (scan_end(&scan) || scan_error(&scan)) {
              break;
            }

            Str08 block_type = scan_identifier(&scan);

            if (0);
            else if (str08_match(block_type, str08_lit("NPOIN"))) { ugf_su2_parse_block_NPOIN(&scan, mesh); }
            else if (str08_match(block_type, str08_lit("NELEM"))) { ugf_su2_parse_block_NELEM(&scan, mesh); }
            else if (str08_match(block_type, str08_lit("NMARK"))) { ugf_su2_parse_block_NMARK(&scan, mesh); }
            else {
              Str08 message = str08_format(scratch.arena, "unexpected block \"%S\"", block_type);
              scan_error_push(&scan, message);
            }
          }
        }

        // NOTE(cmat): Log SU2 parse errors.
        for (Scan_Error *it = scan_error(&scan); it; it = it->next) {
          log_fatal("SU2 error: %u:%u: %S", it->line_at, it->char_at, it->message);
        }
      }
    }
  }

  log_zone_end();
  profiler_end_function();
}

