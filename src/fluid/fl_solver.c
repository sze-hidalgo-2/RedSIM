function void fl_solver_euler_halo_pack_send_data(FL_Solver_Euler *euler, FL_State *state) {
  profiler_begin_function();
  UG_Mesh  *mesh = euler->mesh;

  // NOTE(cmat): Gather all "send" data, interleaved.
  for Iter_Range(it_gather, lane_range(mesh->sends.len)) {
    U32 cell_gather = mesh->sends.cell_send[it_gather];
    U64 send_at     = it_gather * 5;
    for Iter_Index(it_state, 5) {
      euler->halo_send_dat[send_at + it_state] = state->states[it_state][cell_gather];
    }
  }

  // NOTE(cmat): Wait for lanes to have gathered all the send data.
  lane_barrier();
  profiler_end_function();
}

function void fl_solver_euler_halo_unpack_receive_data(FL_Solver_Euler *euler, FL_State *state) {
  profiler_begin_function();

  UG_Mesh *mesh = euler->mesh;

  // NOTE(cmat): Deinterleave halo data from receive buffer.
  for Iter_Index(it_state, 5) {
    for Iter_Range(it_halo, lane_range(mesh->halos.len)) {
      state->states[it_state][mesh->cells.len + it_halo] = euler->halo_receive_dat[5 * it_halo + it_state];
    }
  }

  // NOTE(cmat): Wait for lanes to have gathered all the send data.
  lane_barrier();
  profiler_end_function();
}

function void fl_solver_euler_halo_build_request_list(FL_Solver_Euler *euler) {
  profiler_begin_function();
  UG_Mesh  *mesh = euler->mesh;

  ipc_rank_request_list_init(&euler->halo_request_list);

  for Iter_Index(it_rank, mesh->halos.block_len) {
    if (it_rank != ipc_rank_index()) {
      Range1_U64  halo_range  = mesh->halos.block_range[it_rank];
      U64         halo_len    = range1_u64_len(halo_range);

      if (halo_len) {
        // NOTE(cmat): Receive halo data from neighbours.
        U64 own_tag = ipc_rank_index();
        ipc_rank_record_receive(&euler->halo_request_list, halo_len * 5 * sizeof(F32), euler->halo_receive_dat + 5 * halo_range.min, it_rank, own_tag);
      }
    }
  }

  for Iter_Index(it_rank, mesh->sends.block_len) {
    if (it_rank != ipc_rank_index()) {
      Range1_U64  send_range  = mesh->sends.block_range[it_rank];
      U64         send_len    = range1_u64_len(send_range);

      if (send_len) {
        // NOTE(cmat): Send "send" data to neighbours.
        U64 other_tag = it_rank;
        ipc_rank_record_send(&euler->halo_request_list, send_len * 5 * sizeof(F32), euler->halo_send_dat + 5 * send_range.min, it_rank, other_tag);
      }
    }
  }

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_euler_init(FL_Solver_Euler *euler, FL_Boundary_Map *boundary, UG_Mesh *mesh, Arena *arena) {
  Zero_Fill(euler);

  euler->mesh                 = mesh;
  euler->boundary             = boundary;

  fl_state_init(&euler->flow_1, mesh, 1, arena);
  fl_state_init(&euler->flow_2, mesh, 1, arena);

  fl_state_init(&euler->residual, mesh, 0, arena);

  euler->halo_receive_len     = 5 * mesh->halos.len;
  euler->halo_send_len        = 5 * mesh->sends.len;
  if (lane_index() == 0) {
    euler->cell_time_step     = arena_push_count(arena, F64, mesh->cells.len);
    euler->lane_time_step     = arena_push_count(arena, F64, lane_count());
    euler->halo_send_dat      = arena_push_count(arena, F32, euler->halo_send_len);
    euler->halo_receive_dat   = arena_push_count(arena, F32, euler->halo_receive_len);
  }

  lane_broadcast_ptr(&euler->cell_time_step,        0);
  lane_broadcast_ptr(&euler->lane_time_step,        0);
  lane_broadcast_ptr(&euler->halo_send_dat,         0);
  lane_broadcast_ptr(&euler->halo_receive_dat,      0);

  // NOTE(cmat): Build request list for halo synchronization.
  fl_solver_euler_halo_build_request_list(euler);
}

function void fl_solver_compute_ghost(FL_Solver_Euler *euler, FL_State *state) {
  profiler_begin_function();
  UG_Mesh *mesh = euler->mesh;

  // NOTE(cmat): Assign ghost cell values.
  for Iter_Range(it, lane_range(mesh->ghosts.len)) {
    U64 cell_parent_index = mesh->ghosts.parent_cell  [it];
    U08 face_parent_index = mesh->ghosts.parent_face  [it];
    U32 marker_index      = mesh->ghosts.marker_index [it];
    U64 flow_ghost_index  = mesh->cells.len + mesh->halos.len + it;

    UG_Cell_Faces *faces  = &mesh->cells.faces[cell_parent_index];
    V3F normal            = v3f(faces->normal_x[face_parent_index], faces->normal_y[face_parent_index], faces->normal_z[face_parent_index]);
    V5F inner_state       = fl_state_get(state, cell_parent_index);
    V5F ghost_state       = fl_boundary_map_ghost(euler->boundary, marker_index, inner_state, normal, state->gamma);

    fl_state_set(state, flow_ghost_index, ghost_state);
  }

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_compute_residual_range(FL_Solver_Euler *euler, FL_State *state, FL_State *residual, Range1_U64 range, B32 compute_time_step, F64 *cell_time_step) {
  profiler_begin_function();
  U64 range_len = range1_u64_len(range);
  UG_Mesh *mesh = euler->mesh;

  U64 bucket_index = lane_index();
  for Iter_Range(it_range, lane_range(range_len)) {
    U64 it_cell       = range.min + it_range;
    V5F cell_residual = v5f(0, 0, 0, 0, 0);
    V5F left_state    = v5f(state->rho[it_cell], state->rho_v1[it_cell], state->rho_v2[it_cell], state->rho_v3[it_cell], state->energy[it_cell]);

    F64 spectral_sum    = 0.f;
    UG_Cell_Faces faces = mesh->cells.faces[it_cell];
    for Iter_Index(it_face, 4) {
      U32 adjacent    = faces.adjacent[it_face];
      V3F normal      = v3f(faces.normal_x[it_face], faces.normal_y[it_face], faces.normal_z[it_face]);
      F32 area        = faces.area[it_face];
      V5F right_state = v5f(state->rho[adjacent], state->rho_v1[adjacent], state->rho_v2[adjacent], state->rho_v3[adjacent], state->energy[adjacent]);
      FL_Flux flux    = fl_flux_hllc(left_state, right_state, normal, state->gamma);
      cell_residual   = v5f_sub(cell_residual, v5f_mul(area, flux.state));
      spectral_sum   += (F64)area * (F64)flux.lambda_max;
    }

    F32 volume     = mesh->cells.volume[it_cell];
    F32 volume_rcp = 1.f / volume;

    residual->rho     [it_cell] = cell_residual.x1 * volume_rcp;
    residual->rho_v1  [it_cell] = cell_residual.x2 * volume_rcp;
    residual->rho_v2  [it_cell] = cell_residual.x3 * volume_rcp;
    residual->rho_v3  [it_cell] = cell_residual.x4 * volume_rcp;
    residual->energy  [it_cell] = cell_residual.x5 * volume_rcp;

    if (compute_time_step) {
      cell_time_step[it_cell] = (F64)volume / spectral_sum;
    }
  }

  lane_barrier();
  profiler_end_function();
}

function F32 fl_solver_compute_global_time_step(FL_Solver_Euler *euler, F64 *time_steps) {
  profiler_begin_function();

  // NOTE(cmat): Compute minimum time step for each lane.
  F64 lane_time_step = f64_limit_max;
  for Iter_Range(it, lane_range(euler->mesh->cells.len)) {
    lane_time_step = f64_min(lane_time_step, time_steps[it]);
  }

  euler->lane_time_step[lane_index()] = lane_time_step;

  // NOTE(cmat): Compute minimum time step across lanes.
  F64 global_time_step = f64_limit_max;
  if (lane_index() == 0) {
    for Iter_Index(it, lane_count()) {
      global_time_step = f64_min(global_time_step, euler->lane_time_step[it]);
    }
  }

  // NOTE(cmat): Synchronize minimum time_step across lanes.
  lane_broadcast_u64((U64 *)&global_time_step, 0);


  // NOTE(cmat): Synchronize minimum time_step across IPC ranks.
  global_time_step = ipc_rank_minimum_f64(global_time_step);

  profiler_end_function();
  return global_time_step;
}

function void fl_solver_compute_normalized_residual_magnitude(FL_Solver_Euler *euler) {
}

function void fl_solver_global_euler_step(FL_Solver_Euler *euler, FL_State *state_dst, FL_State *state_src, FL_State *residual, F32 time_scale, F64 time_step) {
  profiler_begin_function();
  UG_Mesh  *mesh = euler->mesh;

  F64 scaled_time_step = time_scale * time_step;

  for Iter_Index(it_state, 5) {
    F32 *state_src_array = state_src->states   [it_state];
    F32 *state_dst_array = state_dst->states   [it_state];
    F32 *residual_array  = residual->states    [it_state];

    for Iter_Range(it, lane_range(mesh->cells.len)) {
      state_dst_array[it] = (F32)(state_src_array[it] + scaled_time_step * residual_array[it]);
    }
  }

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_global_euler_step_2(FL_Solver_Euler *euler, FL_State *state_dst, F32 state_1_coeff, FL_State *state_1, F32 state_2_coeff, FL_State *state_2, FL_State *residual, F32 time_scale, F64 time_step) {
  profiler_begin_function();
  UG_Mesh  *mesh = euler->mesh;

  F64 scaled_time_step = time_scale * time_step;

  for Iter_Index(it_state, 5) {
    F32 *state_1_array    = state_1->states   [it_state];
    F32 *state_2_array    = state_2->states   [it_state];
    F32 *state_dst_array  = state_dst->states [it_state];
    F32 *residual_array   = residual->states  [it_state];

    for Iter_Range(it, lane_range(mesh->cells.len)) {
      state_dst_array[it] = (F32)(state_1_coeff * state_1_array[it] + state_2_coeff * state_2_array[it] + time_scale * residual_array[it]);
    }
  }

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_local_euler_step(FL_Solver_Euler *euler, FL_State *state_dst, FL_State *state_src, FL_State *residual, F32 time_scale, F64 *time_steps) {
  profiler_begin_function();
  UG_Mesh  *mesh = euler->mesh;

  for Iter_Index(it_state, 5) {
    F32 *state_src_array = state_src->states   [it_state];
    F32 *state_dst_array = state_dst->states   [it_state];
    F32 *residual_array  = residual->states    [it_state];

    for Iter_Range(it, lane_range(mesh->cells.len)) {
      state_dst_array[it] = (F32)(state_src_array[it] + time_scale * time_steps[it] * residual_array[it]);
    }
  }

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_local_euler_step_2(FL_Solver_Euler *euler, FL_State *state_dst, F32 state_1_coeff, FL_State *state_1, F32 state_2_coeff, FL_State *state_2, FL_State *residual, F32 time_scale, F64 *time_steps) {
  profiler_begin_function();
  UG_Mesh  *mesh = euler->mesh;

  for Iter_Index(it_state, 5) {
    F32 *state_1_array    = state_1->states   [it_state];
    F32 *state_2_array    = state_2->states   [it_state];
    F32 *state_dst_array  = state_dst->states [it_state];
    F32 *residual_array   = residual->states  [it_state];

    for Iter_Range(it, lane_range(mesh->cells.len)) {
      state_dst_array[it] = (F32)(state_1_coeff * state_1_array[it] + state_2_coeff * state_2_array[it] + time_scale * time_steps[it] * residual_array[it]);
    }
  }

  lane_barrier();
  profiler_end_function();
}

function void fl_solver_euler_compute_residual(FL_Solver_Euler *euler, FL_State *state, FL_State *residual, B32 compute_time_step) {
  profiler_begin_function();
  UG_Mesh  *mesh = euler->mesh;

  // NOTE(cmat): Pack cells for halo exchange.
  fl_solver_euler_halo_pack_send_data(euler, state);

  // NOTE(cmat): Start halo cell exchange between IPC ranks.
  IPC_Request_Scope(&euler->halo_request_list) {
    // NOTE(cmat): While we are waiting for the halo cells to arrive from,
    // - other ranks, we compute the ghost cells to save time.
    fl_solver_compute_ghost(euler, state);

    // NOTE(cmat): Next we compute the residual of all interior cells.
    // - Those are cells not touching any halo cells; they can still be in touch with ghost cells.
    fl_solver_compute_residual_range(euler, state, residual, mesh->groups.cells_interior, compute_time_step, euler->cell_time_step);

  } // NOTE(cmat): Wait for halo cells to be exchanged.

  // NOTE(cmat): Unpack received halo data.
  fl_solver_euler_halo_unpack_receive_data(euler, state);

  // NOTE(cmat): Compute residual for remaining boundary cells
  fl_solver_compute_residual_range(euler, state, residual, mesh->groups.cells_boundary, compute_time_step, euler->cell_time_step);
  profiler_end_function();
}

function F64 fl_solver_euler_solve_global_step_forward_euler(FL_Solver_Euler *euler, F32 CFL) {
  profiler_begin_function();
  UG_Mesh  *mesh = euler->mesh;

  // NOTE(cmat): Compute residual.
  fl_solver_euler_compute_residual(euler, &euler->flow_1, &euler->residual, 1);

  // NOTE(cmat): global time-stepping
  F64 time_step = fl_solver_compute_global_time_step(euler, euler->cell_time_step);
  fl_solver_global_euler_step(euler, &euler->flow_1, &euler->flow_1, &euler->residual, CFL, time_step);
  
  profiler_end_function();
  return time_step;
}

function void fl_solver_euler_solve_local_step_forward_euler(FL_Solver_Euler *euler, F32 CFL) {
  profiler_begin_function();
  UG_Mesh  *mesh = euler->mesh;

  // NOTE(cmat): Compute residual.
  fl_solver_euler_compute_residual(euler, &euler->flow_1, &euler->residual, 1);

  // NOTE(cmat): local time-stepping
  fl_solver_local_euler_step(euler, &euler->flow_1, &euler->flow_1, &euler->residual, CFL, euler->cell_time_step);
  profiler_end_function();
}


// NOTE(cmat): SSP-RK(4, 3)
// Q1 = U
// Q2 = Q1 + dt/2 * R(Q1)
// Q2 = Q2 + dt/2 * R(Q2)
// Q2 = 2/3 * Q1 + 1/3 * [ Q2 + dt/2 * R(Q2) ]
// U  = Q2 + dt/2 * R(Q2)

function F64 fl_solver_euler_solve_global_step_SSP_RK_4_3(FL_Solver_Euler *euler, F32 CFL) {
  profiler_begin_function();
  UG_Mesh  *mesh = euler->mesh;

  // NOTE(cmat): Compute R(Q1)
  fl_solver_euler_compute_residual(euler, &euler->flow_1, &euler->residual, 1);

  // NOTE(cmat): Compute time-step, based on R(Q1)
  F64 time_step = fl_solver_compute_global_time_step(euler, euler->cell_time_step);

  // NOTE(cmat): Compute Q2 = Q1 + dt/2 * R(Q1)
  fl_solver_global_euler_step(euler, &euler->flow_2, &euler->flow_1, &euler->residual, CFL, (.5f * time_step));

  // NOTE(cmat): Compute R(Q2)
  fl_solver_euler_compute_residual(euler, &euler->flow_2, &euler->residual, 0);

  // NOTE(cmat): Compute Q2 = Q2 + dt / 2 * R(Q2)
  fl_solver_global_euler_step(euler, &euler->flow_2, &euler->flow_2, &euler->residual, CFL, (.5f * time_step));

  // NOTE(cmat): Compute R(Q2)
  fl_solver_euler_compute_residual(euler, &euler->flow_2, &euler->residual, 0);

  // NOTE(cmat): Compute  Q2 = 2/3 * Q1 + 1/3 * [ Q2 + dt/2 * R(Q2) ]
  //                      Q2 = 2/3 * Q1 + 1/3 * Q2 + dt/2 * R(Q2)
  fl_solver_global_euler_step_2(euler, &euler->flow_2, 2.f/3.f, &euler->flow_1, 1.f/3.f, &euler->flow_2, &euler->residual, CFL, ((1.f / 6.f) * time_step));

  // NOTE(cmat): Compute R(Q2)
  fl_solver_euler_compute_residual(euler, &euler->flow_2, &euler->residual, 0);

  // U = Q2 + dt/2 * R(Q2)
  fl_solver_global_euler_step(euler, &euler->flow_1, &euler->flow_2, &euler->residual, CFL, (.5f * time_step));
  
  profiler_end_function();
  return time_step;
}

function void fl_solver_euler_solve_local_step_SSP_RK_4_3(FL_Solver_Euler *euler, F32 CFL) {
  profiler_begin_function();
  UG_Mesh  *mesh = euler->mesh;

  // NOTE(cmat): Compute R(Q1). We use the timesteps from here for every other step.
  fl_solver_euler_compute_residual(euler, &euler->flow_1, &euler->residual, 1);

  // NOTE(cmat): Compute Q2 = Q1 + dt/2 * R(Q1)
  fl_solver_local_euler_step(euler, &euler->flow_2, &euler->flow_1, &euler->residual, .5f * CFL, euler->cell_time_step);

  // NOTE(cmat): Compute R(Q2)
  fl_solver_euler_compute_residual(euler, &euler->flow_2, &euler->residual, 0);

  // NOTE(cmat): Compute Q2 = Q2 + dt / 2 * R(Q2)
  fl_solver_local_euler_step(euler, &euler->flow_2, &euler->flow_2, &euler->residual, .5f * CFL, euler->cell_time_step);

  // NOTE(cmat): Compute R(Q2)
  fl_solver_euler_compute_residual(euler, &euler->flow_2, &euler->residual, 0);

  // NOTE(cmat): Compute  Q2 = 2/3 * Q1 + 1/3 * [ Q2 + dt/2 * R(Q2) ]
  //                      Q2 = 2/3 * Q1 + 1/3 * Q2 + dt/2 * R(Q2)
  fl_solver_local_euler_step_2(euler, &euler->flow_2, 2.f/3.f, &euler->flow_1, 1.f/3.f, &euler->flow_2, &euler->residual, (1.f / 6.f) * CFL, euler->cell_time_step);

  // NOTE(cmat): Compute R(Q2)
  fl_solver_euler_compute_residual(euler, &euler->flow_2, &euler->residual, 0);

  // U = Q2 + dt/2 * R(Q2)
  fl_solver_local_euler_step(euler, &euler->flow_1, &euler->flow_2, &euler->residual, .5f * CFL, euler->cell_time_step);
  profiler_end_function();
}

function void fl_solver_euler_solve(FL_Solver_Euler *euler, F32 time_target) {
  profiler_begin_function();
  log_zone_start("Solving euler flow");

  // NOTE(cmat): 10 warmup iterations.
  for Iter_Index(it, 10) {
    // fl_solver_euler_solve_step_SSP_RK_4_3(euler, 0.f);
  }

  // NOTE(cmat): Synchronize all ranks, for more accurate benchmarking.
  ipc_rank_barrier();

  F32 CFL = 1.85f;
  // F32 CFL = 0.5f;
  U64 clock_start = sys_performance_clock_now();

  // NOTE(cmat): Iterate.
  F64 time        = 0;
  U64 iteration   = 0;
  for Iter_Index(it, 1000) {
    // fl_solver_euler_solve_local_step_forward_euler(euler, CFL);
    fl_solver_euler_solve_local_step_SSP_RK_4_3(euler, CFL);
    F32 time_step = 0;
    time         += 0; // time_step;
    iteration    += 1;
    log_info("Time: %.2g | Tau: %.2g | Iteration: %'llu", time, time_step, iteration);
  }

  lane_barrier();

  U64 clock_end       = sys_performance_clock_now();
  U64 clock_dt        = clock_end - clock_start;
  F64 clock_seconds   = clock_dt * sys_performance_clock_to_nanoseconds() * 1e-9;

  log_info("Simulation time: %.4f seconds", time);
  log_info("Wall-Clock time: %.4f seconds", clock_seconds);

  log_zone_end();
  profiler_end_function();
}
