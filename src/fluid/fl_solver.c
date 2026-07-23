function void fl_solver_euler_init(FL_Solver_Euler *euler, FL_Boundary_Map *boundary, UG_Mesh *mesh, Arena *arena) {
  Zero_Fill(euler);

  euler->mesh                 = mesh;
  euler->boundary             = boundary;

  fl_state_init(&euler->flow,     mesh, 1, arena);
  fl_state_init(&euler->residual, mesh, 0, arena);

  euler->time_step_bucket_len = lane_count();
  euler->halo_send_len        = 5 * mesh->sends.len;
  if (lane_index() == 0) {
    euler->time_step_bucket_dat = arena_push_count(arena, F64, euler->time_step_bucket_len);
    euler->halo_send_dat        = arena_push_count(arena, F32, euler->halo_send_len);
  }

  lane_broadcast_ptr(&euler->time_step_bucket_dat,  0);
  lane_broadcast_ptr(&euler->halo_send_dat,         0);
}

function void fl_solver_compute_ghost(FL_Solver_Euler *euler) {
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
    V5F inner_state       = fl_state_get(&euler->flow, cell_parent_index);
    V5F ghost_state       = fl_boundary_map_ghost(euler->boundary, marker_index, inner_state, normal, euler->flow.gamma);

    fl_state_set(&euler->flow, flow_ghost_index, ghost_state);
  }

  profiler_end_function();
}

function void fl_solver_compute_residual(FL_Solver_Euler *euler, F32 CFL) {
  profiler_begin_function();

  UG_Mesh  *mesh     = euler->mesh;
  FL_State *flow     = &euler->flow;
  FL_State *residual = &euler->residual;

  U64 bucket_index                          = lane_index();
  euler->time_step_bucket_dat[bucket_index] = f64_limit_max;

  for Iter_Range(it_cell, lane_range(mesh->cells.len)) {
    V5F cell_residual = v5f(0, 0, 0, 0, 0);
    V5F left_state    = v5f(flow->rho[it_cell], flow->rho_v1[it_cell], flow->rho_v2[it_cell], flow->rho_v3[it_cell], flow->energy[it_cell]);

    F64 spectral_sum    = 0.f;
    UG_Cell_Faces faces = mesh->cells.faces[it_cell];
    for Iter_Index(it_face, 4) {
      U32 adjacent    = faces.adjacent[it_face];
      V3F normal      = v3f(faces.normal_x[it_face], faces.normal_y[it_face], faces.normal_z[it_face]);
      F32 area        = faces.area[it_face];
      V5F right_state = v5f(flow->rho[adjacent], flow->rho_v1[adjacent], flow->rho_v2[adjacent], flow->rho_v3[adjacent], flow->energy[adjacent]);
      FL_Flux flux    = fl_flux_hllc(left_state, right_state, normal, flow->gamma);
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

    F64 time_step                               = (F64)CFL * (F64)volume / spectral_sum;
    euler->time_step_bucket_dat[bucket_index]   = f64_min(euler->time_step_bucket_dat[bucket_index], time_step);
  }

  lane_barrier();
  profiler_end_function();
}

function F64 fl_solver_compute_time_step(FL_Solver_Euler *euler) {
  profiler_begin_function();

  // NOTE(cmat): Compute minimum time_step across lanes.
  F64 global_time_step = f64_limit_max;
  if (lane_index() == 0) {
    for Iter_Index(it, lane_count()) {
      global_time_step = f64_min(global_time_step, euler->time_step_bucket_dat[it]);
    }
  }

  // NOTE(cmat): Synchronize minimum time_step across lanes.
  lane_broadcast_u64((U64 *)&global_time_step, 0);

  profiler_end_function();
  return global_time_step;
}

function void fl_solver_euler_step_explicit(FL_Solver_Euler *euler, F32 time_step) {
  profiler_begin_function();

  UG_Mesh  *mesh     = euler->mesh;
  FL_State *flow     = &euler->flow;
  FL_State *residual = &euler->residual;

  for Iter_Index(it_state, 5) {
    F32 * restrict state_array    = flow->states      [it_state];
    F32 * restrict residual_array = residual->states  [it_state];

    for Iter_Range(it, lane_range(mesh->cells.len)) {
      state_array[it] += time_step * residual_array[it];
    }
  }

  profiler_end_function();
}

function void fl_solver_euler_solve_halo_exchange(FL_Solver_Euler *euler, IPC_Sync_List *sync_list) {
  profiler_begin_function();
  UG_Mesh  *mesh = euler->mesh;
  FL_State *flow = &euler->flow;

  // NOTE(cmat): Post IPC receives first, that way we can start immediately when we receive a send.
  for Iter_Index(it_rank, mesh->halos.block_len) {
    if (it_rank != ipc_rank_index()) {
      Range1_U64  halo_range  = mesh->halos.block_range[it_rank];
      U64         halo_len    = range1_u64_len(halo_range);

      if (halo_len) {
        U64 own_tag   = 5 * ipc_rank_index();

        // NOTE(cmat): Receive halo data from neighbours.
        for Iter_Index(it_state, 5) {
          U64 offset = mesh->cells.len + halo_range.min;
          ipc_rank_receive(sync_list, halo_len * sizeof(F32), flow->states[it_state] + offset, it_rank, own_tag + it_state);
        }
      }
    }
  }

  // NOTE(cmat) Gather all "send" data to send to neighbours.
  for Iter_Index(it_state, 5) {
    U32 state_offset = it_state * mesh->sends.len;
    for Iter_Range(it_gather, lane_range(mesh->sends.len)) {
      U32 cell_gather                                 = mesh->sends.cell_send[it_gather];
      euler->halo_send_dat[state_offset + it_gather]  = flow->states[it_state][cell_gather];
    }
  }

  // NOTE(cmat): Wait for lanes to have gathered all the send data.
  lane_barrier();

  // NOTE(cmat): Now we issue the IPC sends, exchaning halo data between ranks.
  for Iter_Index(it_rank, mesh->sends.block_len) {
    if (it_rank != ipc_rank_index()) {
      Range1_U64  send_range  = mesh->sends.block_range[it_rank];
      U64         send_len    = range1_u64_len(send_range);

      if (send_len) {
        U64 other_tag = 5 * it_rank;

        // NOTE(cmat): Send "send" data to neighbours.
        for Iter_Index(it_state, 5) {
          U64 offset = it_state * mesh->sends.len + send_range.min;
          ipc_rank_send(sync_list, send_len * sizeof(F32), euler->halo_send_dat + offset, it_rank, other_tag + it_state);
        }
      }
    }
  }

  lane_barrier();
  profiler_end_function();
}

function F64 fl_solver_euler_solve_step(FL_Solver_Euler *euler, F32 CFL) {
  profiler_begin_function();

  // NOTE(cmat): Exchange halo cells between IPC ranks.
  IPC_Sync_List sync_list = { };
  IPC_Sync_Scope(&sync_list) {
    // NOTE(cmat): Start IPC halo data exchange.
    fl_solver_euler_solve_halo_exchange(euler, &sync_list);

    // NOTE(cmat): While we are waiting for the halo cells to arrive from,
    // - other ranks, we compute the ghost cells to save time.
    fl_solver_compute_ghost(euler);
    lane_barrier();
  }

  // NOTE(cmat): Compute cell residuals.
  fl_solver_compute_residual(euler, CFL);

  // NOTE(cmat): Compute time-step.
  F64 time_step = fl_solver_compute_time_step(euler);

  // NOTE(cmat): Synchronize minimum time_step across IPC ranks.
  time_step = ipc_rank_minimum(time_step);

  // NOTE(cmat): Explicit-Euler integration.
  fl_solver_euler_step_explicit(euler, (F32)time_step);
  
  profiler_end_function();
  return time_step;
}

function void fl_solver_euler_solve(FL_Solver_Euler *euler) {
  profiler_begin_function();
  log_zone_start("Solving euler flow");

  U64 clock_start = sys_performance_clock_now();
  F32 CFL         = 0.85f;

  // NOTE(cmat): Iterate.
  F64 time        = 0;
  F64 time_target = 0.001f;
  for Iter_Index(it, 100) {
    F64 time_step = fl_solver_euler_solve_step(euler, CFL);
    time         += time_step;
    log_info("Time: %.2g | Tau: %.2g", time, time_step);
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
