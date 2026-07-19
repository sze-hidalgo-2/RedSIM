function void fl_solver_euler_init(FL_Solver_Euler *euler, FL_Boundary_Map *boundary, UG_Mesh *mesh) {
  Zero_Fill(euler);
  arena_init(&euler->arena);
  euler->mesh = mesh;

  euler->boundary = boundary;
  Assert(boundary->map_len >= mesh->markers.len, "Boundary Map not large enough for mesh");

  fl_state_init(&euler->flow,     mesh);
  fl_state_init(&euler->residual, mesh);
 
  euler->time_step_bucket_len = lane_count();
  euler->time_step_bucket_dat = arena_push_count(&euler->arena, F64, euler->time_step_bucket_len);
}

function void fl_solver_compute_ghost(FL_Solver_Euler *euler) {
  profiler_begin_function();
  UG_Mesh *mesh = euler->mesh;

  // NOTE(cmat): Assign ghost cell values.
  for Iter_Range(it, lane_range(mesh->ghosts.len)) {
    U64 cell_parent_index = mesh->ghosts.parent_cell  [it];
    U08 face_parent_index = mesh->ghosts.parent_face  [it];
    U32 marker_index      = mesh->ghosts.marker_index [it];
    U64 flow_ghost_index  = mesh->cells.len + it;

    UG_Cell_Faces *faces  = &mesh->cells.faces[cell_parent_index];
    V3F normal            = v3f(faces->normal_x[face_parent_index], faces->normal_y[face_parent_index], faces->normal_z[face_parent_index]);
    V5F inner_state       = fl_state_get(&euler->flow, cell_parent_index);
    V5F ghost_state       = fl_boundary_map_ghost(euler->boundary, marker_index, inner_state, normal, euler->flow.gamma);
    
    fl_state_set(&euler->flow, flow_ghost_index, ghost_state);
  }

  profiler_end_function();
}

function F64 fl_solver_compute_residual(FL_Solver_Euler *euler, F32 CFL) {
  profiler_begin_function();

  UG_Mesh  *mesh     = euler->mesh;
  FL_State *flow     = &euler->flow;
  FL_State *residual = &euler->residual;

  U64 bucket_index                          = lane_index();
  euler->time_step_bucket_dat[bucket_index] = f32_limit_max;

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

  // NOTE(cmat): Compute minimum time_step across buckets.
  lane_barrier();

  F64 global_time_step = f64_limit_max;
  if (lane_index() == 0) {
    for Iter_Index(it, lane_count()) {
      global_time_step = f64_min(global_time_step, euler->time_step_bucket_dat[it]);
    }
  }

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

function F64 fl_solver_euler_solve_step(FL_Solver_Euler *euler, F32 CFL) {
  profiler_begin_function();

  // NOTE(cmat): Assign ghost cells first.
  lane_barrier();
  fl_solver_compute_ghost(euler);

  // NOTE(cmat): Compute cell residuals.
  lane_barrier();
  F64 time_step = fl_solver_compute_residual(euler, CFL);
  
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
  while (time < time_target) {
    F64 time_step = fl_solver_euler_solve_step(euler, CFL);
    time         += time_step;
    log_info("Time: %.2f | Tau: %.8f", time, time_step);
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
