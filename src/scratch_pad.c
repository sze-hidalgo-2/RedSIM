function void ug_partition_rcb_split_recursive(UG_Grid *grid,
                                               Range3_F32 bounds,
                                               U32 first_partition,
                                               U32 partition_count,
                                               Range1_U64 range,
                                               UG_Partition_RCB_Key *rcb_keys) {
  Log_Zone_Scope("RCB split: partitions [%u -- %u), range [%'llu -- %'llu]",
                 first_partition,
                 first_partition + partition_count,
                 range.min,
                 range.max) {

    U64 range_len = range1_u64_len(range);

    if (partition_count == 1 || range_len <= 1) {
      for Iter_Range(it, lane_range(range_len)) {
        rcb_keys[range.min + it].partition = first_partition;
      }
      return;
    }

    U32 split_axis = 0;
    range3_f32_largest_axis(bounds, &split_axis);

    array_sort_radix_u32(
        range_len,
        4,
        split_axis,
        (U32 *)(rcb_keys + range.min));

    U32 left_parts  = partition_count / 2;
    U32 right_parts = partition_count - left_parts;

    U64 left_len  = (range_len * left_parts) / partition_count;
    U64 right_len = range_len - left_len;

    U64 center_index = range.min + left_len;

    V3U center_key = rcb_keys[center_index].center;
    V3F center = v3f(f32_from_radix_key(center_key.x),
                    f32_from_radix_key(center_key.y),
                    f32_from_radix_key(center_key.z));

    Range3_F32 bounds_left  = bounds;
    Range3_F32 bounds_right = bounds;

    bounds_left.max.dat[split_axis]  = center.dat[split_axis];
    bounds_right.min.dat[split_axis] = center.dat[split_axis];

    ug_partition_rcb_split_recursive(
        grid,
        bounds_left,
        first_partition,
        left_parts,
        range1_u64(range.min, center_index),
        rcb_keys);

    ug_partition_rcb_split_recursive(
        grid,
        bounds_right,
        first_partition + left_parts,
        right_parts,
        range1_u64(center_index, range.max),
        rcb_keys);
  }
}









































function void redsim_group_entry(void *user_data) {
  profiler_begin_function();
  log_zone_start("Thread Group Entry");

  log_zone_end();
  Log_Zone_Scope("Thread Group Entered") {
  }

  local_storage UG_Mesh mesh = { };
  if (lane_index() == 0) {
    ug_mesh_init(&mesh);
    ugf_load_su2(&mesh, str08_lit("cube_4M.su2"));
  }

  lane_barrier();
  ug_mesh_compute_cells       (&mesh);
  ug_mesh_reorder_cells       (&mesh);
  ug_mesh_compute_cells_faces (&mesh);
  ug_mesh_assign_ghosts       (&mesh);

  lane_barrier();

  local_storage FL_Solver_Euler solver = { };
  local_storage FL_Boundary_Map boundary = { };
  // local_storage FL_Boundary_Farfield farfield = { };
  if (lane_index() == 0) {

#if 0
    fl_boundary_map_init(&boundary, 3);

    farfield                                = (FL_Boundary_Farfield)  { .velocity = v3f(50, 0, 0), .density = 1.225f, .pressure = 1.0e5f };
    *fl_boundary_map_by_index(&boundary, 0) = (FL_Boundary)           { .type = FL_Boundary_Type_Farfield, .farfield = farfield };
    *fl_boundary_map_by_index(&boundary, 1) = (FL_Boundary)           { .type = FL_Boundary_Type_Slip };
    *fl_boundary_map_by_index(&boundary, 2) = (FL_Boundary)           { .type = FL_Boundary_Type_Slip };
#else
    fl_boundary_map_init(&boundary, 6);
    *fl_boundary_map_by_index(&boundary, 0) = (FL_Boundary)           { .type = FL_Boundary_Type_Slip };
    *fl_boundary_map_by_index(&boundary, 1) = (FL_Boundary)           { .type = FL_Boundary_Type_Slip };
    *fl_boundary_map_by_index(&boundary, 2) = (FL_Boundary)           { .type = FL_Boundary_Type_Slip };
    *fl_boundary_map_by_index(&boundary, 3) = (FL_Boundary)           { .type = FL_Boundary_Type_Slip };
    *fl_boundary_map_by_index(&boundary, 4) = (FL_Boundary)           { .type = FL_Boundary_Type_Slip };
    *fl_boundary_map_by_index(&boundary, 5) = (FL_Boundary)           { .type = FL_Boundary_Type_Slip };
#endif

    fl_solver_euler_init(&solver, &boundary, &mesh);
  }

  // NOTE(cmat): Assign initial condition.
  lane_barrier();

  fl_setup_sod(&solver.flow, solver.mesh);
  // fl_state_set_inner_from_farfield(&solver.flow, &farfield);

  lane_barrier();
  fl_solver_euler_solve(&solver);

  lane_barrier();

  FLF_Ensight_Export export = flf_ensight_export_start(str08_lit("sod_ensight"), &mesh);
  flf_ensight_export_flow(&export, 0.f, &solver.flow);
  flf_ensight_export_end(&export);

