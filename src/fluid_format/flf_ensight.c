enum { 
  FLF_Ensight_Geo_Header_Size         = 5 * 80,
  FLF_Ensight_Var_Header_Size         = 1 * 80,
  FLF_Ensight_Var_Part_Header_Size    = 1 * 80 + sizeof(I32) + 80,
};

force_inline function void flf_ensight_write_c80(U08 *buffer, U64 *offset, Str08 str) {
  Assert(str.len <= 80, "length overflow");
  memory_copy(buffer + *offset,           str.txt, str.len);
  memory_fill(buffer + *offset + str.len, 0,       80 - str.len); 
  *offset += 80;
}

force_inline function void flf_ensight_write_i32(U08 *buffer, U64 *offset, I32 x) {
  memory_copy(buffer + *offset, &x, sizeof(I32));
  *offset += sizeof(I32);
}

force_inline function void flf_ensight_write_f32(U08 *buffer, U64 *offset, F32 x) {
  memory_copy(buffer + *offset, &x, sizeof(F32));
  *offset += sizeof(F32);
}

force_inline function void flf_ensight_write_f32_array(U08 *buffer, U64 *offset, U64 count, F32 *x) {
  memory_copy(buffer + *offset, x, sizeof(F32) * count);
  *offset += sizeof(F32) * count;
}

force_inline function void flf_ensight_export_geo(Str08 geo_file_path, UG_Mesh *mesh) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);
  log_zone_start("Starting ensight geo export: %S", geo_file_path);

  // NOTE(cmat): Create .GEO file with header.
  if (lane_index() == 0 && ipc_rank_index() == 0) {
    SYS_File geo_file = { };
    SYS_File_Scope(&geo_file, geo_file_path, SYS_File_Access_Flag_Create | SYS_File_Access_Flag_Truncate | SYS_File_Access_Flag_Write) {
      U08  header_data[FLF_Ensight_Geo_Header_Size] = { };
      U64  header_offset                            = 0;

      flf_ensight_write_c80(header_data, &header_offset, str08_lit("C binary"));
      flf_ensight_write_c80(header_data, &header_offset, str08_lit("RedSIM"));
      flf_ensight_write_c80(header_data, &header_offset, str08_lit("Version 1.0"));
      flf_ensight_write_c80(header_data, &header_offset, str08_lit("node id off"));
      flf_ensight_write_c80(header_data, &header_offset, str08_lit("element id off"));

      sys_file_write(&geo_file, range1_u64(0, FLF_Ensight_Geo_Header_Size), header_data);
    }
  }

  ipc_rank_barrier();
  U64 part_id = ipc_rank_index() + 1;

  // NOTE(cmat): Part data size.
  U64 part_buffer_len = 0;
  part_buffer_len     += 80 + sizeof(I32) + 80 + 80;                                  // NOTE(cmat): 1. Part header.
  part_buffer_len     += sizeof(I32) + 3 * (sizeof(F32) * mesh->grid.verts.len);      // NOTE(cmat): 2. Vertex coordinate data.
  part_buffer_len     += 80 + sizeof(I32) + 4 * sizeof(I32) * mesh->grid.elems.len;   // NOTE(cmat): 3. Cell data

  U08 *part_buffer_dat = 0;
  if (lane_index() == 0) {
    part_buffer_dat = arena_push_size(scratch.arena, part_buffer_len);
  }

  lane_broadcast_ptr(&part_buffer_dat, 0);

  if (lane_index() == 0) {
    U64 part_buffer_off = 0;

    // NOTE(cmat): 1. Part header.
    log_info("Writing part header");
    flf_ensight_write_c80(part_buffer_dat, &part_buffer_off, str08_lit("part"));
    flf_ensight_write_i32(part_buffer_dat, &part_buffer_off, (I32)part_id);
    flf_ensight_write_c80(part_buffer_dat, &part_buffer_off, str08_lit("part-data"));
    flf_ensight_write_c80(part_buffer_dat, &part_buffer_off, str08_lit("coordinates"));

    // NOTE(cmat): 2. Vertex data
    log_info("Writing part vertex data: %'llu elements", mesh->grid.verts.len);
    flf_ensight_write_i32       (part_buffer_dat, &part_buffer_off, (I32)mesh->grid.verts.len);
    flf_ensight_write_f32_array (part_buffer_dat, &part_buffer_off, mesh->grid.verts.len, mesh->grid.verts.x);
    flf_ensight_write_f32_array (part_buffer_dat, &part_buffer_off, mesh->grid.verts.len, mesh->grid.verts.y);
    flf_ensight_write_f32_array (part_buffer_dat, &part_buffer_off, mesh->grid.verts.len, mesh->grid.verts.z);

    // NOTE(cmat): 3. Cell data.
    log_info("Writing part cell data: %'llu elements", mesh->grid.elems.len);
    flf_ensight_write_c80(part_buffer_dat, &part_buffer_off, str08_lit("tetra4"));
    flf_ensight_write_i32(part_buffer_dat, &part_buffer_off, (I32)mesh->grid.elems.len);

    for Iter_Index(it, mesh->grid.elems.len) {
      V4U verts = mesh->grid.elems.verts[it];

      // NOTE(cmat): Element indexing starts at 1.
      flf_ensight_write_i32(part_buffer_dat, &part_buffer_off, (I32)verts.x + 1);
      flf_ensight_write_i32(part_buffer_dat, &part_buffer_off, (I32)verts.y + 1);
      flf_ensight_write_i32(part_buffer_dat, &part_buffer_off, (I32)verts.z + 1);
      flf_ensight_write_i32(part_buffer_dat, &part_buffer_off, (I32)verts.w + 1);
    }

    Assert(part_buffer_off == part_buffer_len, "Mismatch between last offset and buffer len");
  }

  // NOTE(cmat): Rank write sizes.
  log_info("Exchanging part sizes");
  U64 *rank_part_sizes = 0;
  if (lane_index() == 0) {
    rank_part_sizes = arena_push_count(scratch.arena, U64, ipc_rank_count());
  }

  lane_broadcast_ptr(&rank_part_sizes, 0);
  ipc_rank_gather_all_u64(part_buffer_len, rank_part_sizes);

  if (lane_index() == 0) {
    // NOTE(cmat): Compute offset for our part.
    U64 part_offset = 0;
    for Iter_Index(it, ipc_rank_index()) {
      log_info("%'llu", rank_part_sizes[it]);
      part_offset += rank_part_sizes[it];
    }

    SYS_File geo_file = { };
    SYS_File_Scope(&geo_file, geo_file_path, SYS_File_Access_Flag_Write) {
      Range1_U64 write_range = range1_u64(FLF_Ensight_Geo_Header_Size + part_offset, FLF_Ensight_Geo_Header_Size + part_offset + part_buffer_len);
      sys_file_write(&geo_file, write_range, part_buffer_dat);
    }
  }

  ipc_rank_barrier();
  lane_barrier();
  log_zone_end();
  scratch_end(&scratch);
  profiler_end_function();
}

function void flf_ensight_export_init(FLF_Ensight_Export *export, Str08 folder_path, UG_Mesh *mesh, Arena *arena) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(arena);
  log_zone_start("Starting ensight export: %S", folder_path);

  // TODO(cmat): Alloc on just a single lane.
  export->case_file_path    = str08_format(arena, "%S/data.case",       folder_path);
  export->data_folder_path  = str08_format(arena, "%S/data",            folder_path);
  export->geo_file_path     = str08_format(arena, "%S/data/ugrid.geo",  folder_path);

  // NOTE(cmat): Create output folder hierarchy.
  if (lane_index() == 0) {
    if (ipc_rank_index() == 0) {
      sys_directory_create(folder_path);
      sys_directory_create(export->data_folder_path); 
    }
  }
  
  ipc_rank_barrier();
  lane_barrier();
  flf_ensight_export_geo(export->geo_file_path, mesh);

  log_info("Exchanging cell counts");
  if (lane_index() == 0) {
    export->part_cell_count = arena_push_count(arena, U64, ipc_rank_count());
  }

  lane_broadcast_ptr(&export->part_cell_count, 0);
  ipc_rank_gather_all_u64(mesh->grid.elems.len, export->part_cell_count);

  // NOTE(cmat): Create .case file.
  if (lane_index() == 0) {
    if (ipc_rank_index() == 0) {
      SYS_File case_file = { };
      SYS_File_Scope(&case_file, export->case_file_path, SYS_File_Access_Flag_Create | SYS_File_Access_Flag_Truncate | SYS_File_Access_Flag_Write) {
        Str08 case_header = str08_format(scratch.arena,
          "FORMAT"                                                             "\n"
          "type: ensight gold"                                                 "\n"

          "GEOMETRY"                                                           "\n"
          "model: data/ugrid.geo"                                              "\n"
          "VARIABLE"                                                           "\n"
          "scalar per element: 1 density      data/cell_density.bin******"     "\n"
          "scalar per element: 1 energy       data/cell_energy.bin******"      "\n"
          "scalar per element: 1 pressure     data/cell_pressure.bin******"    "\n"
          "scalar per element: 1 temperature  data/cell_temperature.bin******" "\n"
          "scalar per element: 1 local_index  data/cell_local_index.bin******" "\n"
          "vector per element: 1 velocity     data/cell_velocity.bin******"    "\n"
  
          "TIME"                                                               "\n"
          "time set: 1"                                                        "\n"
          "filename start number: 1"                                           "\n"
          "filename increment: 1"                                              "\n"
          "time values:"                                                       "\n"
          "number of steps: ");

        sys_file_write(&case_file, range1_u64(0, case_header.len), case_header.txt);
        export->step_count_file_range = range1_u64(case_header.len, case_header.len + 24);

        Str08 steps = str08_format(scratch.arena, "%24u", 0);
        if (steps.len == range1_u64_len(export->step_count_file_range)) {
          sys_file_write(&case_file, export->step_count_file_range, steps.txt);
        } else {
          log_fatal("Invalid step count string: %S. Got length %llu expected 24", steps, steps.len);
        }
      }
    }
  }

  lane_broadcast_type(export, 0);
  
  ipc_rank_barrier();
  lane_barrier();
  log_zone_end();
  scratch_end(&scratch);
  profiler_end_function();
}

function void flf_ensight_export_cell_variable(FLF_Ensight_Export *export, Str08 variable_name, U32 variable_component_count, F32 *variable_buffer_dat) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);
  log_info("Exporting cell scalar: %S", variable_name);

  U64 header_len = FLF_Ensight_Var_Header_Size;

  Str08 file_path = str08_format(scratch.arena, "%S/cell_%S.bin%06d", export->data_folder_path, variable_name, (I32)export->timestep_count);

  if (lane_index() == 0 && ipc_rank_index() == 0) {
    U64 header_off                              = 0;
    U08 header_dat[FLF_Ensight_Var_Header_Size] = { };
    flf_ensight_write_c80(header_dat, &header_off, variable_name);

    SYS_File file = { };
    SYS_File_Scope(&file, file_path, SYS_File_Access_Flag_Create | SYS_File_Access_Flag_Truncate | SYS_File_Access_Flag_Write) {
      sys_file_write(&file, range1_u64(0, header_len), header_dat);
    }
  }
  
  lane_barrier();
  ipc_rank_barrier();

  if (lane_index() == 0) {
    U64 part_id                                           = ipc_rank_index() + 1;
    U64 part_header_len                                   = FLF_Ensight_Var_Part_Header_Size;
    U64 part_header_off                                   = 0;
    U08 part_header_dat[FLF_Ensight_Var_Part_Header_Size] = { };

    flf_ensight_write_c80(part_header_dat, &part_header_off, str08_lit("part"));
    flf_ensight_write_i32(part_header_dat, &part_header_off, (I32)part_id);
    flf_ensight_write_c80(part_header_dat, &part_header_off, str08_lit("tetra4"));
    Assert(Stack_Array_Len(part_header_dat) == part_header_off, "offset/length mismatch");

    U64 part_offset = FLF_Ensight_Var_Header_Size;
    for Iter_Index(it, ipc_rank_index()) {
      part_offset += FLF_Ensight_Var_Part_Header_Size;
      part_offset += variable_component_count * export->part_cell_count[it] * sizeof(F32);
    }

    U64 variable_buffer_len = variable_component_count * export->part_cell_count[ipc_rank_index()] * sizeof(F32);
    SYS_File file   = { };
    SYS_File_Scope(&file, file_path, SYS_File_Access_Flag_Write) {
      sys_file_write(&file, range1_u64(part_offset,                   part_offset + part_header_len),                       part_header_dat);
      sys_file_write(&file, range1_u64(part_offset + part_header_len, part_offset + part_header_len + variable_buffer_len), variable_buffer_dat);
    }
  }

  lane_barrier();
  scratch_end(&scratch);
  profiler_end_function();
}

function void flf_ensight_export_flow(FLF_Ensight_Export *export, F32 time, FL_State *state) {
  profiler_begin_function();
  Arena_Temp scratch = scratch_start(0);
  log_zone_start("Exporting ensight flow state");

  export->timestep_count += 1;

  // NOTE(cmat): Reserve maximal length (3 component), use for both scalar and vector.
  U64 cell_count = export->part_cell_count[ipc_rank_index()];
  F32 *variable_buffer = 0;
  if (lane_index() == 0) {
    variable_buffer = arena_push_count(scratch.arena, F32, 3 * cell_count);
  }

  lane_broadcast_ptr(&variable_buffer, 0);

  // NOTE(cmat): Density
  for Iter_Range(it, lane_range(cell_count)) { variable_buffer[it] = state->rho[it]; }
  flf_ensight_export_cell_variable(export, str08_lit("density"), 1, variable_buffer);

  // NOTE(cmat): Energy
  for Iter_Range(it, lane_range(cell_count)) { variable_buffer[it] = state->energy[it]; }
  flf_ensight_export_cell_variable(export, str08_lit("energy"), 1, variable_buffer);

  // NOTE(cmat): Pressure
  for Iter_Range(it, lane_range(cell_count)) { variable_buffer[it] = fl_state_get_pressure(state, it); }
  flf_ensight_export_cell_variable(export, str08_lit("pressure"), 1, variable_buffer);

  // NOTE(cmat): Temperature
  for Iter_Range(it, lane_range(cell_count)) { variable_buffer[it] = fl_state_get_temperature(state, it); }
  flf_ensight_export_cell_variable(export, str08_lit("temperature"), 1, variable_buffer);

  // NOTE(cmat): Local Index
  for Iter_Range(it, lane_range(cell_count)) { variable_buffer[it] = (F32)(it); }
  flf_ensight_export_cell_variable(export, str08_lit("local_index"), 1, variable_buffer);

  // NOTE(cmat): Velocity
  for Iter_Range(it, lane_range(cell_count)) { variable_buffer[0 * cell_count + it] = f32_div_safe(state->rho_v1[it], state->rho[it]); }
  for Iter_Range(it, lane_range(cell_count)) { variable_buffer[1 * cell_count + it] = f32_div_safe(state->rho_v2[it], state->rho[it]); }
  for Iter_Range(it, lane_range(cell_count)) { variable_buffer[2 * cell_count + it] = f32_div_safe(state->rho_v3[it], state->rho[it]); }

  flf_ensight_export_cell_variable(export, str08_lit("velocity"), 3, variable_buffer);


  // NOTE(cmat): Export new timestep.
  // NOTE(cmat): First, we modify the number of steps.
  if (lane_index() == 0 && ipc_rank_index() == 0) {
    SYS_File case_file = { };
    SYS_File_Scope(&case_file, export->case_file_path, SYS_File_Access_Flag_Write) {
      Str08 steps = str08_format(scratch.arena, "%24u", (U32)export->timestep_count);
      if (steps.len == range1_u64_len(export->step_count_file_range)) {
        sys_file_write(&case_file, export->step_count_file_range, steps.txt);
      } else {
        log_fatal("Invalid step count string: %S. Got length %llu expected 24", steps, steps.len);
      }
    }

    // NOTE(cmat): Next, we append the new timestep.
    SYS_File_Scope(&case_file, export->case_file_path, SYS_File_Access_Flag_Write | SYS_File_Access_Flag_Append) {
      Str08 time_step = str08_format(scratch.arena, "\n%.8g", time);
      sys_file_write(&case_file, range1_u64(0, time_step.len), time_step.txt);
    }
  }
 
  lane_barrier();
  log_zone_end();
  scratch_end(&scratch);
  profiler_end_function();
}

